"""
DrugMiR model definitions — shared by all GPU experiments.

This file extracts the DrugMiR_Hybrid model from hp_finetune.py with no
behavioral changes, plus utility functions for data loading, training,
and evaluation. Three GPU scripts (multitask, tsne, robustness) all
import from this module to ensure model consistency.
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
try:
    from torch_scatter import scatter_mean
except ImportError:
    # Fallback: PyG >= 2.4 ships its own scatter
    from torch_geometric.utils import scatter as _scatter
    def scatter_mean(src, idx, dim=0, dim_size=None):
        return _scatter(src, idx, dim=dim, dim_size=dim_size, reduce="mean")
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- data loading ----------

def load_data(dd, km=15, kd=10, assoc_file="association_matrix.npy"):
    """Load DrugMiR Dataset 1 / 2 from a processed directory.

    `assoc_file` lets us swap in resistance_matrix.npy or sensitivity_matrix.npy
    for multi-task experiments while reusing all other features.
    """
    assoc = np.load(f"{dd}/{assoc_file}")
    mf = np.load(f"{dd}/mirna_kmer_features.npy")
    df = np.load(f"{dd}/drug_morgan_features.npy")
    ms = np.load(f"{dd}/mirna_similarity.npy")
    ds = np.load(f"{dd}/drug_similarity.npy")
    d = {
        "assoc": assoc,
        "mirna_feat": torch.FloatTensor(mf).to(device),
        "drug_feat":  torch.FloatTensor(df).to(device),
        "n_mirna": assoc.shape[0],
        "n_drug":  assoc.shape[1],
        "mirna_has_feat": torch.FloatTensor((mf.sum(1) > 0).astype(float)).to(device),
        "drug_has_feat":  torch.FloatTensor((df.sum(1) > 0).astype(float)).to(device),
    }
    # KNN edges for HomoGCN
    s1, d1 = [], []
    for i in range(d["n_mirna"]):
        si = ms[i].copy(); si[i] = -1
        for j in np.argsort(si)[-km:]:
            s1.extend([i, j]); d1.extend([j, i])
    d["mirna_sim_edge"] = torch.LongTensor([s1, d1]).to(device)
    s2, d2 = [], []
    for i in range(d["n_drug"]):
        si = ds[i].copy(); si[i] = -1
        for j in np.argsort(si)[-kd:]:
            s2.extend([i, j]); d2.extend([j, i])
    d["drug_sim_edge"] = torch.LongTensor([s2, d2]).to(device)
    # Gene bridge edges
    mg = np.load(f"{dd}/mirna_gene_matrix.npy")
    dg = np.load(f"{dd}/drug_gene_matrix.npy")
    mg_r, mg_c = np.nonzero(mg); dg_r, dg_c = np.nonzero(dg)
    d["mg_src"] = torch.LongTensor(mg_r).to(device)
    d["mg_dst"] = torch.LongTensor(mg_c).to(device)
    d["dg_src"] = torch.LongTensor(dg_r).to(device)
    d["dg_dst"] = torch.LongTensor(dg_c).to(device)
    d["n_gene"] = max(mg_c.max() if len(mg_c) else 0,
                      dg_c.max() if len(dg_c) else 0) + 1
    pr, pc = np.nonzero(assoc)
    d["pos_pairs"] = list(zip(pr.tolist(), pc.tolist()))
    return d


def sample_neg(assoc, n_pos, n):
    """Random negative sampling: n unobserved (i, j) pairs."""
    nm, nd = assoc.shape
    out = []
    while len(out) < n:
        i = np.random.randint(0, nm); j = np.random.randint(0, nd)
        if assoc[i, j] == 0:
            out.append((i, j))
    return out


# ---------- model components ----------

class GG(nn.Module):
    """Channel 2: gated GCN layer over a homogeneous KNN similarity graph.

    Input  : node feature x (shape [N, h]), edge index e
    Output : x + g ⊙ ht  (residual + gating)  — eq. (4)–(6) in the paper.
    """
    def __init__(self, h, dr):
        super().__init__()
        self.gcn = GCNConv(h, h)
        self.gate = nn.Linear(2 * h, h)
        self.norm = nn.BatchNorm1d(h)
        self.drop = nn.Dropout(dr)

    def forward(self, x, e):
        ht = self.drop(self.norm(F.relu(self.gcn(x, e))))
        g = torch.sigmoid(self.gate(torch.cat([x, ht], -1)))
        return x + g * ht


class GB(nn.Module):
    """Channel 3: Gene Bridge — eq. (7)–(12).

    Three steps per layer:
      1. genes aggregate from miRNAs and drugs (scatter_mean)
      2. miRNAs and drugs read back from genes (scatter_mean)
      3. gated residual fusion at miRNA / drug nodes
    """
    def __init__(self, h, dr):
        super().__init__()
        self.mg = nn.Linear(2 * h, h)
        self.dg = nn.Linear(2 * h, h)
        self.norm = nn.BatchNorm1d(h)
        self.drop = nn.Dropout(dr)

    def forward(self, mh, dh, gh, ms, md, ds, dd, ng):
        gm = scatter_mean(mh[ms], md, dim=0, dim_size=ng)
        gd = scatter_mean(dh[ds], dd, dim=0, dim_size=ng)
        ga = self.drop(self.norm(F.relu(gh + gm + gd)))
        mfg = scatter_mean(ga[md], ms, dim=0, dim_size=mh.size(0))
        dfg = scatter_mean(ga[dd], ds, dim=0, dim_size=dh.size(0))
        m_new = mh + torch.sigmoid(self.mg(torch.cat([mh, mfg], -1))) * mfg
        d_new = dh + torch.sigmoid(self.dg(torch.cat([dh, dfg], -1))) * dfg
        return m_new, d_new, ga


class HybridEnc(nn.Module):
    """Channel 1: Adaptive Feature-Embedding Fusion — eq. (1)–(3)."""
    def __init__(self, n, fd, h, dr):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(fd, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h, h),  nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
        )
        self.emb = nn.Embedding(n, h)
        self.gate = nn.Linear(2 * h, h)

    def forward(self, feat, has_feat=None):
        fh = self.feat(feat)
        eh = self.emb.weight
        g = torch.sigmoid(self.gate(torch.cat([fh, eh], -1)))
        if has_feat is not None:
            mask = has_feat.unsqueeze(1)
            return mask * (g * fh + (1 - g) * eh) + (1 - mask) * eh
        return g * fh + (1 - g) * eh


class DrugMiREncoder(nn.Module):
    """Three-channel encoder shared by all DrugMiR variants.

    Returns the 6d joint representation z_ij (eq. 13). Different prediction
    heads can be plugged on top — single-task BCE (the standard model),
    multi-task res/sen heads, or no head at all when extracting embeddings
    for t-SNE.
    """
    def __init__(self, nm, nd, md, dd, ng, h=256, dr=0.5, n_gcn=2, n_br=2):
        super().__init__()
        self.h = h
        self.me = HybridEnc(nm, md, h, dr)
        self.de = HybridEnc(nd, dd, h, dr)
        self.ge = nn.Embedding(ng, h)
        self.mgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        self.dgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        self.br = nn.ModuleList([GB(h, dr) for _ in range(n_br)])

    def encode(self, data):
        """Return the six per-entity representations needed for t-SNE.

        Outputs: m0, mh, mb, d0, dh, db
        m0/d0  : Hybrid Embedding output (Channel 1)
        mh/dh  : Homo GCN output         (Channel 2)
        mb/db  : Gene Bridge output      (Channel 3)
        """
        m0 = self.me(data["mirna_feat"], data.get("mirna_has_feat"))
        d0 = self.de(data["drug_feat"],  data.get("drug_has_feat"))
        mh, dh = m0, d0
        for l in self.mgcn: mh = l(mh, data["mirna_sim_edge"])
        for l in self.dgcn: dh = l(dh, data["drug_sim_edge"])
        mb, db, gh = m0, d0, self.ge.weight
        for l in self.br:
            mb, db, gh = l(mb, db, gh,
                           data["mg_src"], data["mg_dst"],
                           data["dg_src"], data["dg_dst"], data["n_gene"])
        return m0, mh, mb, d0, dh, db

    def joint(self, data, mi, di):
        """6d concatenated representation for a list of (miRNA, drug) pairs."""
        m0, mh, mb, d0, dh, db = self.encode(data)
        zm = torch.cat([m0, mh, mb], -1)[mi]   # [B, 3h]
        zd = torch.cat([d0, dh, db], -1)[di]   # [B, 3h]
        return torch.cat([zm, zd], -1)         # [B, 6h]


class MLPHead(nn.Module):
    """3-layer MLP head: 6h -> 2h -> h -> 1 (eq. 14)."""
    def __init__(self, h, dr=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6 * h, 2 * h), nn.BatchNorm1d(2 * h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(2 * h, h),     nn.BatchNorm1d(h),     nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class DrugMiR_Hybrid(nn.Module):
    """Standard single-task DrugMiR (matches hp_finetune.py exactly)."""
    def __init__(self, nm, nd, md, dd, ng, h=256, dr=0.5, n_gcn=2, n_br=2):
        super().__init__()
        self.enc = DrugMiREncoder(nm, nd, md, dd, ng, h, dr, n_gcn, n_br)
        self.head = MLPHead(h, dr)

    def forward(self, data, mi, di):
        return self.head(self.enc.joint(data, mi, di))


class DrugMiR_MultiTask(nn.Module):
    """Multi-task variant: shared encoder + two parallel heads."""
    def __init__(self, nm, nd, md, dd, ng, h=256, dr=0.5, n_gcn=2, n_br=2):
        super().__init__()
        self.enc = DrugMiREncoder(nm, nd, md, dd, ng, h, dr, n_gcn, n_br)
        self.head_res = MLPHead(h, dr)
        self.head_sen = MLPHead(h, dr)

    def forward(self, data, mi, di):
        z = self.enc.joint(data, mi, di)
        return self.head_res(z), self.head_sen(z)


# ---------- training utilities ----------

def train_epoch(model, data, train_pairs, optimizer, batch_size=2048,
                head=None):
    """One epoch of single-task training with random negative sampling.

    `head` lets multi-task callers pick which logits to use ("res" / "sen").
    """
    model.train()
    neg = sample_neg(data["assoc"], train_pairs, len(train_pairs))
    pairs = train_pairs + neg
    labels = [1.0] * len(train_pairs) + [0.0] * len(neg)
    idx = np.random.permutation(len(pairs))
    total_loss, n_batches = 0.0, 0
    for s in range(0, len(idx), batch_size):
        bi = idx[s:s + batch_size]
        bp = [pairs[i] for i in bi]
        bl = torch.FloatTensor([labels[i] for i in bi]).to(device)
        mi = torch.LongTensor([p[0] for p in bp]).to(device)
        di = torch.LongTensor([p[1] for p in bp]).to(device)
        optimizer.zero_grad()
        out = model(data, mi, di)
        if head is not None:
            out = out[0] if head == "res" else out[1]
        loss = F.binary_cross_entropy_with_logits(out, bl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, data, test_pairs, head=None):
    """Return (AUC, AUPR) on a held-out fold."""
    model.eval()
    neg = sample_neg(data["assoc"], test_pairs, len(test_pairs))
    pairs = test_pairs + neg
    labels = np.array([1.0] * len(test_pairs) + [0.0] * len(neg))
    mi = torch.LongTensor([p[0] for p in pairs]).to(device)
    di = torch.LongTensor([p[1] for p in pairs]).to(device)
    out = model(data, mi, di)
    if head is not None:
        out = out[0] if head == "res" else out[1]
    pr = torch.sigmoid(out).cpu().numpy()
    return roc_auc_score(labels, pr), average_precision_score(labels, pr)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
