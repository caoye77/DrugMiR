"""
DrugMiR v2 改进模型
====================
修复 v1 的三个问题:
  1. GeneBridge 用 scatter_mean 归一化 + degree normalization
  2. 保留原始特征通道 (concatenate, 不替换)
  3. 门控融合代替注意力 (让模型决定图信息的使用程度)

新增:
  4. 多跳基因桥接 (2-hop: miRNA→Gene→Drug, Drug→Gene→miRNA)
  5. 相似度边使用 KNN 而非固定阈值 (保证每个节点有足够邻居)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
try:
    from torch_scatter import scatter_mean
except ImportError:
    from torch_geometric.utils import scatter
    def scatter_mean(src, index, dim=0, dim_size=None):
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


class FeatureEncoder(nn.Module):
    """特征编码器: 投影 + BatchNorm"""
    def __init__(self, in_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class NormalizedGeneBridge(nn.Module):
    """
    改进的基因桥接层
    关键修复: scatter_mean 代替 scatter_add, 加 degree normalization
    """
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        # 消息变换
        self.mirna_msg = nn.Linear(hidden_dim, hidden_dim)
        self.drug_msg = nn.Linear(hidden_dim, hidden_dim)
        self.gene_msg_to_mirna = nn.Linear(hidden_dim, hidden_dim)
        self.gene_msg_to_drug = nn.Linear(hidden_dim, hidden_dim)

        # 聚合后的归一化
        self.gene_norm = nn.LayerNorm(hidden_dim)
        self.mirna_norm = nn.LayerNorm(hidden_dim)
        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 门控: 控制图信息的比例 (核心改进)
        self.mirna_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.drug_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, mirna_h, drug_h, gene_h, mirna_gene_edge, drug_gene_edge):
        n_mirna, n_drug, n_gene = mirna_h.size(0), drug_h.size(0), gene_h.size(0)

        # === Step 1: 聚合到 Gene (用 scatter_mean 归一化) ===
        gene_agg = gene_h.clone()

        if mirna_gene_edge.numel() > 0:
            src_m, dst_g = mirna_gene_edge[0], mirna_gene_edge[1]
            msg = self.mirna_msg(mirna_h[src_m])
            gene_from_mirna = scatter_mean(msg, dst_g, dim=0, dim_size=n_gene)
            gene_agg = gene_agg + gene_from_mirna

        if drug_gene_edge.numel() > 0:
            src_d, dst_g = drug_gene_edge[0], drug_gene_edge[1]
            msg = self.drug_msg(drug_h[src_d])
            gene_from_drug = scatter_mean(msg, dst_g, dim=0, dim_size=n_gene)
            gene_agg = gene_agg + gene_from_drug

        gene_agg = self.dropout(F.relu(self.gene_norm(gene_agg)))

        # === Step 2: Gene 反传到 miRNA/Drug (scatter_mean) ===
        mirna_from_gene = torch.zeros_like(mirna_h)
        if mirna_gene_edge.numel() > 0:
            src_m, dst_g = mirna_gene_edge[0], mirna_gene_edge[1]
            msg = self.gene_msg_to_mirna(gene_agg[dst_g])
            mirna_from_gene = scatter_mean(msg, src_m, dim=0, dim_size=n_mirna)

        drug_from_gene = torch.zeros_like(drug_h)
        if drug_gene_edge.numel() > 0:
            src_d, dst_g = drug_gene_edge[0], drug_gene_edge[1]
            msg = self.gene_msg_to_drug(gene_agg[dst_g])
            drug_from_gene = scatter_mean(msg, src_d, dim=0, dim_size=n_drug)

        # === Step 3: 门控融合 (不是直接加, 而是学习融合比例) ===
        mirna_gate = self.mirna_gate(torch.cat([mirna_h, mirna_from_gene], dim=-1))
        mirna_out = mirna_h + mirna_gate * mirna_from_gene  # gate 控制图信息的比例
        mirna_out = self.mirna_norm(mirna_out)

        drug_gate = self.drug_gate(torch.cat([drug_h, drug_from_gene], dim=-1))
        drug_out = drug_h + drug_gate * drug_from_gene
        drug_out = self.drug_norm(drug_out)

        return mirna_out, drug_out, gene_agg


class HomoGCNBlock(nn.Module):
    """同质 GCN + 残差 + 门控"""
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            return x
        out = self.gcn(x, edge_index)
        out = self.dropout(F.relu(self.norm(out)))
        g = self.gate(torch.cat([x, out], dim=-1))
        return x + g * out  # 门控残差


class DrugMiRv2(nn.Module):
    """
    DrugMiR v2: 改进版异质图神经网络
    ===================================
    核心改进:
      1. scatter_mean 归一化 (修复高度基因问题)
      2. 门控融合 (保留强特征, 图信息作为增强)
      3. 特征保留通道 (concat 原始特征 + 图增强特征)
      4. 多层基因桥接
    """

    def __init__(self, mirna_feat_dim, drug_feat_dim, n_genes,
                 hidden_dim=128, n_homo_layers=2, n_bridge_layers=2,
                 dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 特征编码
        self.mirna_encoder = FeatureEncoder(mirna_feat_dim, hidden_dim, dropout)
        self.drug_encoder = FeatureEncoder(drug_feat_dim, hidden_dim, dropout)
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)

        # 同质 GCN 层 (带门控)
        self.mirna_gcn_layers = nn.ModuleList([
            HomoGCNBlock(hidden_dim, dropout) for _ in range(n_homo_layers)
        ])
        self.drug_gcn_layers = nn.ModuleList([
            HomoGCNBlock(hidden_dim, dropout) for _ in range(n_homo_layers)
        ])

        # 多层基因桥接 (每层都用门控)
        self.bridge_layers = nn.ModuleList([
            NormalizedGeneBridge(hidden_dim, dropout) for _ in range(n_bridge_layers)
        ])

        # 最终预测: concat 原始特征 + HomoGCN输出 + Bridge输出
        # 3 个通道 × hidden_dim × 2 (mirna + drug)
        pred_in_dim = hidden_dim * 3 * 2
        self.predictor = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mirna_feat, drug_feat,
                mirna_sim_edge, drug_sim_edge,
                mirna_gene_edge, drug_gene_edge,
                pred_mirna_idx, pred_drug_idx):

        # === 特征编码 ===
        mirna_h0 = self.mirna_encoder(mirna_feat)
        drug_h0 = self.drug_encoder(drug_feat)
        gene_h = self.gene_embedding.weight

        # === Channel 1: 原始特征 (保留) ===
        mirna_orig = mirna_h0
        drug_orig = drug_h0

        # === Channel 2: 同质 GCN ===
        mirna_homo = mirna_h0
        for layer in self.mirna_gcn_layers:
            mirna_homo = layer(mirna_homo, mirna_sim_edge)

        drug_homo = drug_h0
        for layer in self.drug_gcn_layers:
            drug_homo = layer(drug_homo, drug_sim_edge)

        # === Channel 3: 多层基因桥接 ===
        mirna_bridge = mirna_h0
        drug_bridge = drug_h0
        gene_bridge = gene_h
        for layer in self.bridge_layers:
            mirna_bridge, drug_bridge, gene_bridge = layer(
                mirna_bridge, drug_bridge, gene_bridge,
                mirna_gene_edge, drug_gene_edge
            )

        # === 三通道拼接预测 ===
        m = torch.cat([
            mirna_orig[pred_mirna_idx],
            mirna_homo[pred_mirna_idx],
            mirna_bridge[pred_mirna_idx],
        ], dim=-1)

        d = torch.cat([
            drug_orig[pred_drug_idx],
            drug_homo[pred_drug_idx],
            drug_bridge[pred_drug_idx],
        ], dim=-1)

        pair = torch.cat([m, d], dim=-1)
        scores = self.predictor(pair).squeeze(-1)

        return scores, None
