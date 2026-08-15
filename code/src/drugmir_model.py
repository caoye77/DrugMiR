"""
DrugMiR 核心模型
================
三层异质图神经网络:
  Layer 1: 同质 GCN — 各自相似度网络内消息传递
  Layer 2: 基因桥接异质消息传递 — miRNA→Gene←Drug
  Layer 3: 注意力融合 — 自动学习各信息源权重

创新点:
  1. 多模态特征融合 (Morgan FP + k-mer + 基因靶标)
  2. 基因桥接跨网络消息传递
  3. 自适应负采样
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class FeatureProjector(nn.Module):
    """将不同维度的原始特征投影到统一的隐藏空间"""

    def __init__(self, mirna_in_dim, drug_in_dim, gene_in_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.mirna_proj = nn.Sequential(
            nn.Linear(mirna_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, mirna_feat, drug_feat, gene_feat):
        return (
            self.mirna_proj(mirna_feat),
            self.drug_proj(drug_feat),
            self.gene_proj(gene_feat),
        )


class HomoGCNLayer(nn.Module):
    """
    Layer 1: 同质 GCN
    在 drug-drug 相似度网络和 miRNA-miRNA 相似度网络内做消息传递
    """

    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.mirna_gcn = GCNConv(hidden_dim, hidden_dim)
        self.drug_gcn = GCNConv(hidden_dim, hidden_dim)
        self.mirna_norm = nn.LayerNorm(hidden_dim)
        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mirna_h, drug_h, mirna_sim_edge, drug_sim_edge):
        # miRNA 相似度网络内消息传递
        if mirna_sim_edge.numel() > 0:
            mirna_out = self.mirna_gcn(mirna_h, mirna_sim_edge)
            mirna_out = self.mirna_norm(mirna_out)
            mirna_out = F.relu(mirna_out)
            mirna_out = self.dropout(mirna_out)
            mirna_h = mirna_h + mirna_out  # residual
        
        # Drug 相似度网络内消息传递
        if drug_sim_edge.numel() > 0:
            drug_out = self.drug_gcn(drug_h, drug_sim_edge)
            drug_out = self.drug_norm(drug_out)
            drug_out = F.relu(drug_out)
            drug_out = self.dropout(drug_out)
            drug_h = drug_h + drug_out  # residual

        return mirna_h, drug_h


class GeneBridgeLayer(nn.Module):
    """
    Layer 2: 基因桥接异质消息传递
    核心创新: miRNA → Gene ← Drug 的跨网络信息传播

    消息传递流程:
      1. Gene 从 miRNA 和 Drug 聚合信息
      2. miRNA 和 Drug 从 Gene 获取跨网络信息
    """

    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        # miRNA → Gene
        self.mirna_to_gene = nn.Linear(hidden_dim, hidden_dim)
        # Drug → Gene
        self.drug_to_gene = nn.Linear(hidden_dim, hidden_dim)
        # Gene → miRNA
        self.gene_to_mirna = nn.Linear(hidden_dim, hidden_dim)
        # Gene → Drug
        self.gene_to_drug = nn.Linear(hidden_dim, hidden_dim)

        self.gene_norm = nn.LayerNorm(hidden_dim)
        self.mirna_norm = nn.LayerNorm(hidden_dim)
        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mirna_h, drug_h, gene_h,
                mirna_gene_edge, drug_gene_edge):
        """
        mirna_gene_edge: (2, E1) - miRNA→Gene 边
        drug_gene_edge:  (2, E2) - Drug→Gene 边
        """
        n_mirna = mirna_h.size(0)
        n_drug = drug_h.size(0)
        n_gene = gene_h.size(0)

        # --- Step 1: 聚合到 Gene ---
        # miRNA → Gene: 对每个 gene, 聚合所有靶向它的 miRNA 的信息
        gene_from_mirna = torch.zeros(n_gene, mirna_h.size(1), device=mirna_h.device)
        if mirna_gene_edge.numel() > 0:
            src_mirna = mirna_gene_edge[0]  # miRNA indices
            dst_gene = mirna_gene_edge[1]   # Gene indices
            msg = self.mirna_to_gene(mirna_h[src_mirna])
            gene_from_mirna.scatter_add_(0, dst_gene.unsqueeze(1).expand_as(msg), msg)

        # Drug → Gene: 对每个 gene, 聚合所有靶向它的 Drug 的信息
        gene_from_drug = torch.zeros(n_gene, drug_h.size(1), device=drug_h.device)
        if drug_gene_edge.numel() > 0:
            src_drug = drug_gene_edge[0]  # Drug indices
            dst_gene = drug_gene_edge[1]  # Gene indices
            msg = self.drug_to_gene(drug_h[src_drug])
            gene_from_drug.scatter_add_(0, dst_gene.unsqueeze(1).expand_as(msg), msg)

        # 更新 gene 表示
        gene_h_new = gene_h + gene_from_mirna + gene_from_drug
        gene_h_new = self.gene_norm(gene_h_new)
        gene_h_new = F.relu(gene_h_new)
        gene_h_new = self.dropout(gene_h_new)

        # --- Step 2: Gene 反向传播到 miRNA 和 Drug ---
        # Gene → miRNA
        mirna_from_gene = torch.zeros(n_mirna, gene_h_new.size(1), device=mirna_h.device)
        if mirna_gene_edge.numel() > 0:
            src_mirna = mirna_gene_edge[0]
            dst_gene = mirna_gene_edge[1]
            msg = self.gene_to_mirna(gene_h_new[dst_gene])
            mirna_from_gene.scatter_add_(0, src_mirna.unsqueeze(1).expand_as(msg), msg)

        # Gene → Drug
        drug_from_gene = torch.zeros(n_drug, gene_h_new.size(1), device=drug_h.device)
        if drug_gene_edge.numel() > 0:
            src_drug = drug_gene_edge[0]
            dst_gene = drug_gene_edge[1]
            msg = self.gene_to_drug(gene_h_new[dst_gene])
            drug_from_gene.scatter_add_(0, src_drug.unsqueeze(1).expand_as(msg), msg)

        # 残差连接
        mirna_h = mirna_h + mirna_from_gene
        mirna_h = self.mirna_norm(mirna_h)
        mirna_h = F.relu(mirna_h)

        drug_h = drug_h + drug_from_gene
        drug_h = self.drug_norm(drug_h)
        drug_h = F.relu(drug_h)

        return mirna_h, drug_h, gene_h_new


class AttentionFusion(nn.Module):
    """
    Layer 3: 注意力融合
    自动学习同质GCN信息和异质桥接信息的权重
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # 为 miRNA 和 Drug 各自学习 2 个视图的权重
        # 视图1: 同质GCN输出, 视图2: 基因桥接输出
        self.mirna_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),  # 2 个视图
        )
        self.drug_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, mirna_homo, mirna_hetero, drug_homo, drug_hetero):
        """
        mirna_homo:   来自同质GCN的miRNA表示
        mirna_hetero: 来自基因桥接的miRNA表示
        drug_homo:    来自同质GCN的Drug表示
        drug_hetero:  来自基因桥接的Drug表示
        """
        # miRNA 注意力权重
        mirna_cat = torch.cat([mirna_homo, mirna_hetero], dim=-1)
        mirna_w = F.softmax(self.mirna_attn(mirna_cat), dim=-1)  # (N, 2)
        mirna_fused = mirna_w[:, 0:1] * mirna_homo + mirna_w[:, 1:2] * mirna_hetero

        # Drug 注意力权重
        drug_cat = torch.cat([drug_homo, drug_hetero], dim=-1)
        drug_w = F.softmax(self.drug_attn(drug_cat), dim=-1)
        drug_fused = drug_w[:, 0:1] * drug_homo + drug_w[:, 1:2] * drug_hetero

        return mirna_fused, drug_fused, mirna_w, drug_w


class Predictor(nn.Module):
    """预测层: 给定 miRNA 和 Drug 的表示, 预测关联分数"""

    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, mirna_h, drug_h, mirna_idx, drug_idx):
        """
        mirna_idx, drug_idx: 要预测的 (miRNA, Drug) 对的索引
        """
        m = mirna_h[mirna_idx]
        d = drug_h[drug_idx]
        pair = torch.cat([m, d], dim=-1)
        return self.mlp(pair).squeeze(-1)


class DrugMiR(nn.Module):
    """
    DrugMiR: Multi-Source Heterogeneous GNN
    ========================================
    完整模型，整合所有组件。
    """

    def __init__(self, mirna_feat_dim, drug_feat_dim, n_genes,
                 hidden_dim=128, n_homo_layers=2, dropout=0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_homo_layers = n_homo_layers

        # 基因特征维度 (用可学习 embedding)
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)

        # 特征投影
        self.projector = FeatureProjector(
            mirna_feat_dim, drug_feat_dim, hidden_dim, hidden_dim, dropout
        )

        # Layer 1: 多层同质 GCN
        self.homo_layers = nn.ModuleList([
            HomoGCNLayer(hidden_dim, dropout) for _ in range(n_homo_layers)
        ])

        # Layer 2: 基因桥接
        self.gene_bridge = GeneBridgeLayer(hidden_dim, dropout)

        # Layer 3: 注意力融合
        self.attention = AttentionFusion(hidden_dim)

        # 预测层
        self.predictor = Predictor(hidden_dim, dropout)

    def forward(self, mirna_feat, drug_feat,
                mirna_sim_edge, drug_sim_edge,
                mirna_gene_edge, drug_gene_edge,
                pred_mirna_idx, pred_drug_idx):
        """
        Args:
            mirna_feat:       (N_mirna, mirna_feat_dim) - k-mer 特征
            drug_feat:        (N_drug, drug_feat_dim)   - Morgan 指纹
            mirna_sim_edge:   (2, E) - miRNA 相似度边
            drug_sim_edge:    (2, E) - Drug 相似度边
            mirna_gene_edge:  (2, E) - miRNA-Gene 边
            drug_gene_edge:   (2, E) - Drug-Gene 边
            pred_mirna_idx:   (B,) - 要预测的 miRNA 索引
            pred_drug_idx:    (B,) - 要预测的 Drug 索引

        Returns:
            scores:   (B,) - 预测分数
            attn_w:   注意力权重 (用于可视化)
        """
        n_genes = self.gene_embedding.num_embeddings
        gene_feat = self.gene_embedding.weight
        device = mirna_feat.device

        # 投影到统一隐藏空间
        mirna_h, drug_h, gene_h = self.projector(mirna_feat, drug_feat, gene_feat)

        # Layer 1: 同质 GCN (多层)
        mirna_homo, drug_homo = mirna_h, drug_h
        for layer in self.homo_layers:
            mirna_homo, drug_homo = layer(mirna_homo, drug_homo,
                                          mirna_sim_edge, drug_sim_edge)

        # Layer 2: 基因桥接异质消息传递
        mirna_hetero, drug_hetero, gene_h = self.gene_bridge(
            mirna_h, drug_h, gene_h, mirna_gene_edge, drug_gene_edge
        )

        # Layer 3: 注意力融合
        mirna_fused, drug_fused, mirna_w, drug_w = self.attention(
            mirna_homo, mirna_hetero, drug_homo, drug_hetero
        )

        # 预测
        scores = self.predictor(mirna_fused, drug_fused,
                                pred_mirna_idx, pred_drug_idx)

        return scores, (mirna_w, drug_w)


class AdaptiveNegativeSampler:
    """
    自适应负采样器
    =====================
    区分高置信度负样本和未知样本:
    - 高分未知样本 → 困难负样本 (更高采样概率)
    - 低分未知样本 → 简单负样本 (正常采样概率)
    """

    def __init__(self, n_mirna, n_drug, pos_set, device='cpu'):
        self.n_mirna = n_mirna
        self.n_drug = n_drug
        self.pos_set = pos_set  # set of (mirna_idx, drug_idx) tuples
        self.device = device
        # 采样概率 (初始均匀)
        self.sample_weights = torch.ones(n_mirna * n_drug, device=device)
        # 标记正样本为 0 概率
        for m, d in pos_set:
            self.sample_weights[m * n_drug + d] = 0.0

    def sample(self, n_samples, model_scores=None):
        """
        采样负样本
        如果提供 model_scores, 使用自适应策略 (概率正比于模型预测分数)
        """
        if model_scores is not None:
            # 自适应: 分数越高越可能被采样 (困难负样本)
            weights = self.sample_weights.clone()
            # 只更新非正样本的权重
            mask = weights > 0
            scores_flat = model_scores.detach().flatten()
            if scores_flat.shape[0] == weights.shape[0]:
                weights[mask] = torch.sigmoid(scores_flat[mask]) + 0.1
        else:
            weights = self.sample_weights

        # 归一化
        weights = weights / weights.sum()

        # 采样
        indices = torch.multinomial(weights, n_samples, replacement=False)
        mirna_idx = indices // self.n_drug
        drug_idx = indices % self.n_drug

        return mirna_idx.to(self.device), drug_idx.to(self.device)

    def simple_sample(self, n_samples):
        """简单随机负采样 (用于初始 epoch)"""
        mirna_idx = []
        drug_idx = []
        while len(mirna_idx) < n_samples:
            m = torch.randint(0, self.n_mirna, (1,)).item()
            d = torch.randint(0, self.n_drug, (1,)).item()
            if (m, d) not in self.pos_set:
                mirna_idx.append(m)
                drug_idx.append(d)
        return (torch.tensor(mirna_idx, device=self.device),
                torch.tensor(drug_idx, device=self.device))
