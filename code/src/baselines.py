"""
DrugMiR Baseline + Ablation 模型
=================================
Baseline:
  1. MLP — 纯特征拼接 + MLP，无图结构
  2. GCN-Basic — 只用关联图的标准 GCN
  3. GAT-Basic — 只用关联图的标准 GAT
  4. BipartiteGCN — 二部图 GCN (类似 GCMDR)
  5. SVD — 矩阵分解

Ablation (DrugMiR 变体):
  A1. w/o HomoGCN — 去掉同质相似度 GCN
  A2. w/o GeneBridge — 去掉基因桥接层
  A3. w/o Attention — 注意力融合改为简单拼接
  A4. w/o DrugSim — 去掉药物相似度边
  A5. w/o miRNASim — 去掉 miRNA 相似度边
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from drugmir_model import (
    FeatureProjector, HomoGCNLayer, GeneBridgeLayer,
    AttentionFusion, Predictor
)


# ============================================================
# Baseline 1: MLP
# ============================================================
class MLPBaseline(nn.Module):
    def __init__(self, mirna_dim, drug_dim, hidden_dim=128, dropout=0.3, **kwargs):
        super().__init__()
        self.mirna_enc = nn.Sequential(
            nn.Linear(mirna_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                mirna_gene_edge, drug_gene_edge, pred_mirna_idx, pred_drug_idx):
        m = self.mirna_enc(mirna_feat)[pred_mirna_idx]
        d = self.drug_enc(drug_feat)[pred_drug_idx]
        return self.predictor(torch.cat([m, d], dim=-1)).squeeze(-1), None


# ============================================================
# Baseline 2: GCN-Basic (二部图上的GCN)
# ============================================================
class GCNBaseline(nn.Module):
    def __init__(self, mirna_dim, drug_dim, hidden_dim=128, n_mirna=1578,
                 n_drug=156, dropout=0.3, **kwargs):
        super().__init__()
        self.n_mirna = n_mirna
        self.n_drug = n_drug
        total = n_mirna + n_drug

        self.mirna_proj = nn.Linear(mirna_dim, hidden_dim)
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                mirna_gene_edge, drug_gene_edge, pred_mirna_idx, pred_drug_idx):
        # 构建 miRNA-Drug 二部图
        m_h = self.mirna_proj(mirna_feat)
        d_h = self.drug_proj(drug_feat)
        x = torch.cat([m_h, d_h], dim=0)

        # 从关联矩阵构建边 (用预测对的正样本)
        src = pred_mirna_idx
        dst = pred_drug_idx + self.n_mirna
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ])

        # 加入相似度边
        if mirna_sim_edge.numel() > 0:
            edge_index = torch.cat([edge_index, mirna_sim_edge], dim=1)
        if drug_sim_edge.numel() > 0:
            dd = drug_sim_edge + self.n_mirna
            edge_index = torch.cat([edge_index, dd], dim=1)

        x = F.relu(self.norm1(self.gcn1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.gcn2(x, edge_index)))

        m = x[pred_mirna_idx]
        d = x[pred_drug_idx + self.n_mirna]
        return self.predictor(torch.cat([m, d], dim=-1)).squeeze(-1), None


# ============================================================
# Baseline 3: GAT-Basic
# ============================================================
class GATBaseline(nn.Module):
    def __init__(self, mirna_dim, drug_dim, hidden_dim=128, n_mirna=1578,
                 n_drug=156, dropout=0.3, **kwargs):
        super().__init__()
        self.n_mirna = n_mirna
        self.mirna_proj = nn.Linear(mirna_dim, hidden_dim)
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)

        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                mirna_gene_edge, drug_gene_edge, pred_mirna_idx, pred_drug_idx):
        m_h = self.mirna_proj(mirna_feat)
        d_h = self.drug_proj(drug_feat)
        x = torch.cat([m_h, d_h], dim=0)

        src = pred_mirna_idx
        dst = pred_drug_idx + self.n_mirna
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

        if mirna_sim_edge.numel() > 0:
            edge_index = torch.cat([edge_index, mirna_sim_edge], dim=1)
        if drug_sim_edge.numel() > 0:
            dd = drug_sim_edge + self.n_mirna
            edge_index = torch.cat([edge_index, dd], dim=1)

        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = self.gat2(x, edge_index)

        m = x[pred_mirna_idx]
        d = x[pred_drug_idx + self.n_mirna]
        return self.predictor(torch.cat([m, d], dim=-1)).squeeze(-1), None


# ============================================================
# Baseline 4: SVD (矩阵分解)
# ============================================================
class SVDBaseline:
    """非神经网络方法，直接矩阵分解"""
    def __init__(self, n_mirna, n_drug, k=64):
        self.k = k
        self.n_mirna = n_mirna
        self.n_drug = n_drug

    def fit_predict(self, assoc_matrix, test_mirna, test_drug):
        import numpy as np
        from scipy.sparse.linalg import svds

        U, S, Vt = svds(assoc_matrix.astype(np.float64), k=self.k)
        pred_matrix = U @ np.diag(S) @ Vt
        scores = pred_matrix[test_mirna, test_drug]
        return scores


# ============================================================
# Ablation 变体
# ============================================================

class DrugMiR_NoHomoGCN(nn.Module):
    """A1: 去掉同质 GCN 层"""
    def __init__(self, mirna_feat_dim, drug_feat_dim, n_genes,
                 hidden_dim=128, dropout=0.3, **kwargs):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.projector = FeatureProjector(mirna_feat_dim, drug_feat_dim, hidden_dim, hidden_dim, dropout)
        self.gene_bridge = GeneBridgeLayer(hidden_dim, dropout)
        self.predictor = Predictor(hidden_dim, dropout)

    def forward(self, mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                mirna_gene_edge, drug_gene_edge, pred_mirna_idx, pred_drug_idx):
        gene_feat = self.gene_embedding.weight
        mirna_h, drug_h, gene_h = self.projector(mirna_feat, drug_feat, gene_feat)
        mirna_h, drug_h, _ = self.gene_bridge(mirna_h, drug_h, gene_h,
                                               mirna_gene_edge, drug_gene_edge)
        scores = self.predictor(mirna_h, drug_h, pred_mirna_idx, pred_drug_idx)
        return scores, None


class DrugMiR_NoGeneBridge(nn.Module):
    """A2: 去掉基因桥接层"""
    def __init__(self, mirna_feat_dim, drug_feat_dim, n_genes,
                 hidden_dim=128, n_homo_layers=2, dropout=0.3, **kwargs):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.projector = FeatureProjector(mirna_feat_dim, drug_feat_dim, hidden_dim, hidden_dim, dropout)
        self.homo_layers = nn.ModuleList([
            HomoGCNLayer(hidden_dim, dropout) for _ in range(n_homo_layers)
        ])
        self.predictor = Predictor(hidden_dim, dropout)

    def forward(self, mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                mirna_gene_edge, drug_gene_edge, pred_mirna_idx, pred_drug_idx):
        gene_feat = self.gene_embedding.weight
        mirna_h, drug_h, _ = self.projector(mirna_feat, drug_feat, gene_feat)
        for layer in self.homo_layers:
            mirna_h, drug_h = layer(mirna_h, drug_h, mirna_sim_edge, drug_sim_edge)
        scores = self.predictor(mirna_h, drug_h, pred_mirna_idx, pred_drug_idx)
        return scores, None


class DrugMiR_NoAttention(nn.Module):
    """A3: 注意力融合改为简单平均"""
    def __init__(self, mirna_feat_dim, drug_feat_dim, n_genes,
                 hidden_dim=128, n_homo_layers=2, dropout=0.3, **kwargs):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.projector = FeatureProjector(mirna_feat_dim, drug_feat_dim, hidden_dim, hidden_dim, dropout)
        self.homo_layers = nn.ModuleList([
            HomoGCNLayer(hidden_dim, dropout) for _ in range(n_homo_layers)
        ])
        self.gene_bridge = GeneBridgeLayer(hidden_dim, dropout)
        self.predictor = Predictor(hidden_dim, dropout)

    def forward(self, mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                mirna_gene_edge, drug_gene_edge, pred_mirna_idx, pred_drug_idx):
        gene_feat = self.gene_embedding.weight
        mirna_h, drug_h, gene_h = self.projector(mirna_feat, drug_feat, gene_feat)

        mirna_homo, drug_homo = mirna_h, drug_h
        for layer in self.homo_layers:
            mirna_homo, drug_homo = layer(mirna_homo, drug_homo, mirna_sim_edge, drug_sim_edge)

        mirna_het, drug_het, _ = self.gene_bridge(mirna_h, drug_h, gene_h,
                                                   mirna_gene_edge, drug_gene_edge)
        # 简单平均代替注意力
        mirna_fused = (mirna_homo + mirna_het) / 2
        drug_fused = (drug_homo + drug_het) / 2

        scores = self.predictor(mirna_fused, drug_fused, pred_mirna_idx, pred_drug_idx)
        return scores, None


# 注册所有模型
BASELINE_MODELS = {
    'MLP': MLPBaseline,
    'GCN': GCNBaseline,
    'GAT': GATBaseline,
}

ABLATION_MODELS = {
    'w/o HomoGCN': DrugMiR_NoHomoGCN,
    'w/o GeneBridge': DrugMiR_NoGeneBridge,
    'w/o Attention': DrugMiR_NoAttention,
}
