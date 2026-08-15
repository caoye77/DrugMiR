"""
Compute Integrated Gradients (IG) on gene embedding for DrugMiR_Hybrid.

For each positive (miRNA, drug) pair, compute the attribution of each gene's
embedding row to the model's predicted association score. Aggregate over all
positive pairs to identify Top-K bridge genes by IG-based importance.

Replaces the heuristic "miRNA_deg × drug_deg" bridge score (paper Section IV-F)
with a model-learned, theoretically-grounded attribution measure.
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)


# ============ Model (same as train_save_for_ig.py) ============

class GG(nn.Module):
    def __init__(s, h, dr):
        super().__init__()
        s.gcn = GCNConv(h, h); s.gate = nn.Linear(2*h, h)
        s.norm = nn.BatchNorm1d(h); s.drop = nn.Dropout(dr)
    def forward(s, x, e):
        ht = s.drop(s.norm(F.relu(s.gcn(x, e))))
        g = torch.sigmoid(s.gate(torch.cat([x, ht], -1)))
        return x + g * ht


class GB(nn.Module):
    def __init__(s, h, dr):
        super().__init__()
        s.mg = nn.Linear(2*h, h); s.dg = nn.Linear(2*h, h)
        s.norm = nn.BatchNorm1d(h); s.drop = nn.Dropout(dr)
    def forward(s, mh, dh, gh, ms, md, ds, dd, ng):
        gm = scatter_mean(mh[ms], md, dim=0, dim_size=ng)
        gd = scatter_mean(dh[ds], dd, dim=0, dim_size=ng)
        ga = s.drop(s.norm(F.relu(gh + gm + gd)))
        mfg = scatter_mean(ga[md], ms, dim=0, dim_size=mh.size(0))
        dfg = scatter_mean(ga[dd], ds, dim=0, dim_size=dh.size(0))
        return (mh + torch.sigmoid(s.mg(torch.cat([mh, mfg], -1))) * mfg,
                dh + torch.sigmoid(s.dg(torch.cat([dh, dfg], -1))) * dfg,
                ga)


class HybridEnc(nn.Module):
    def __init__(s, n, fd, h, dr):
        super().__init__()
        s.feat = nn.Sequential(
            nn.Linear(fd, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr))
        s.emb = nn.Embedding(n, h); s.gate = nn.Linear(2*h, h)
    def forward(s, feat, has_feat=None):
        fh = s.feat(feat); eh = s.emb.weight
        g = torch.sigmoid(s.gate(torch.cat([fh, eh], -1)))
        if has_feat is not None:
            mask = has_feat.unsqueeze(1)
            return mask * (g * fh + (1 - g) * eh) + (1 - mask) * eh
        return g * fh + (1 - g) * eh


class DrugMiR_Hybrid(nn.Module):
    def __init__(s, nm, nd, md, dd, ng, h=256, dr=0.5, n_gcn=2, n_br=2):
        super().__init__()
        s.me = HybridEnc(nm, md, h, dr); s.de = HybridEnc(nd, dd, h, dr)
        s.ge = nn.Embedding(ng, h)
        s.mgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        s.dgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        s.br = nn.ModuleList([GB(h, dr) for _ in range(n_br)])
        s.pred = nn.Sequential(
            nn.Linear(6*h, 2*h), nn.BatchNorm1d(2*h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(2*h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h, 1))
    def forward(s, data, mi, di, gene_emb_override=None):
        """gene_emb_override: if provided, use this instead of s.ge.weight (for IG)."""
        m0 = s.me(data['mirna_feat'], data.get('mirna_has_feat'))
        d0 = s.de(data['drug_feat'], data.get('drug_has_feat'))
        mh = m0; dh = d0
        for l in s.mgcn: mh = l(mh, data['mirna_sim_edge'])
        for l in s.dgcn: dh = l(dh, data['drug_sim_edge'])
        mb = m0; db = d0
        gh = gene_emb_override if gene_emb_override is not None else s.ge.weight
        for l in s.br:
            mb, db, gh = l(mb, db, gh, data['mg_src'], data['mg_dst'],
                           data['dg_src'], data['dg_dst'], data['n_gene'])
        return s.pred(
            torch.cat([torch.cat([m0, mh, mb], -1)[mi],
                       torch.cat([d0, dh, db], -1)[di]], -1)).squeeze(-1)


def load_data(dd, km=15, kd=10):
    assoc = np.load(f"{dd}/association_matrix.npy")
    mf = np.load(f"{dd}/mirna_kmer_features.npy")
    df = np.load(f"{dd}/drug_morgan_features.npy")
    ms = np.load(f"{dd}/mirna_similarity.npy")
    ds = np.load(f"{dd}/drug_similarity.npy")
    d = {'assoc': assoc,
         'mirna_feat': torch.FloatTensor(mf).to(device),
         'drug_feat': torch.FloatTensor(df).to(device),
         'n_mirna': assoc.shape[0], 'n_drug': assoc.shape[1],
         'mirna_has_feat': torch.FloatTensor((mf.sum(1) > 0).astype(float)).to(device),
         'drug_has_feat': torch.FloatTensor((df.sum(1) > 0).astype(float)).to(device)}
    s1, d1 = [], []
    for i in range(d['n_mirna']):
        si = ms[i].copy(); si[i] = -1; tk = np.argsort(si)[-km:]
        for j in tk: s1.extend([i, j]); d1.extend([j, i])
    d['mirna_sim_edge'] = torch.LongTensor([s1, d1]).to(device)
    s2, d2 = [], []
    for i in range(d['n_drug']):
        si = ds[i].copy(); si[i] = -1; tk = np.argsort(si)[-kd:]
        for j in tk: s2.extend([i, j]); d2.extend([j, i])
    d['drug_sim_edge'] = torch.LongTensor([s2, d2]).to(device)
    mg = np.load(f"{dd}/mirna_gene_matrix.npy")
    dg = np.load(f"{dd}/drug_gene_matrix.npy")
    mg_r, mg_c = np.nonzero(mg); dg_r, dg_c = np.nonzero(dg)
    d['mg_src'] = torch.LongTensor(mg_r).to(device)
    d['mg_dst'] = torch.LongTensor(mg_c).to(device)
    d['dg_src'] = torch.LongTensor(dg_r).to(device)
    d['dg_dst'] = torch.LongTensor(dg_c).to(device)
    d['n_gene'] = max(mg_c.max() if len(mg_c) > 0 else 0,
                      dg_c.max() if len(dg_c) > 0 else 0) + 1
    pr, pc = np.nonzero(assoc); d['pos_pairs'] = list(zip(pr.tolist(), pc.tolist()))
    # Store also miRNA-gene and drug-gene degree (for comparison with old bridge score)
    d['mirna_deg_per_gene'] = mg.sum(axis=0)  # how many miRNAs target each gene
    d['drug_deg_per_gene'] = dg.sum(axis=0)
    return d


# ============ IG core ============

def integrated_gradients_on_ge(model, data, pair_batches, n_steps=50, verbose=True):
    """
    Compute IG over gene_emb (model.ge.weight) for positive pairs.
    
    Riemann sum approximation of the path integral:
        IG_g = (E_g - 0) * mean_k [ ∂F_pair / ∂E_g  |  E = (k/N) * E_g ]
    
    Aggregate over many (mi, drug) pairs by summing prediction scores
    (linear sum, so IG distributes linearly).
    
    Returns: ig (ng, h)  — per-row attribution
    """
    model.eval()
    ge_actual = model.ge.weight.detach().clone()  # (ng, h)
    ng, h = ge_actual.shape
    
    ig_accum = torch.zeros_like(ge_actual)  # (ng, h)
    
    n_batches = len(pair_batches)
    for bi, (mi_batch, di_batch) in enumerate(pair_batches):
        for k in range(1, n_steps + 1):
            alpha = float(k) / n_steps
            # Interpolate: from zero baseline to actual ge_actual
            ge_interp = (alpha * ge_actual).clone().detach().requires_grad_(True)
            
            score = model(data, mi_batch, di_batch, gene_emb_override=ge_interp)
            # Use sigmoid(logit) so we attribute the probability rather than raw logit
            score_sum = torch.sigmoid(score).sum()
            
            grad = torch.autograd.grad(score_sum, ge_interp, retain_graph=False)[0]
            ig_accum += grad
        
        if verbose and (bi + 1) % max(1, n_batches // 10) == 0:
            print(f"    Batch {bi+1}/{n_batches} done", flush=True)
    
    # IG = (x - 0) * mean over steps of gradient
    ig = ge_actual * (ig_accum / n_steps)  # (ng, h)
    return ig


# ============ Main ============

if __name__ == '__main__':
    DD1 = os.path.expanduser("~/DrugMiR/data/dataset1")
    CKPT = os.path.expanduser("~/work/DrugMiR/results_final/drugmir_d1_seed42_for_ig.pt")
    OUT_DIR = os.path.expanduser("~/work/DrugMiR/results_final")
    
    print("="*60)
    print("Integrated Gradients on gene embedding (DrugMiR_Hybrid, D1, seed=42)")
    print("="*60, flush=True)
    
    # Load
    t0 = time.time()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    shapes = ckpt['shapes']; cfg = ckpt['config']
    print(f"  Loaded checkpoint with val AUC = {ckpt['best_val_auc']:.4f}")
    print(f"  Shapes: {shapes}", flush=True)
    
    data = load_data(DD1, km=cfg['km'], kd=cfg['kd'])
    
    model = DrugMiR_Hybrid(shapes['nm'], shapes['nd'], shapes['md'], shapes['dd'],
                            shapes['ng'], h=cfg['h'], dr=cfg['dr'],
                            n_gcn=cfg['n_gcn'], n_br=cfg['n_br']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()  # IMPORTANT for IG: deterministic, no dropout
    
    # All positive pairs
    pos = data['pos_pairs']
    print(f"  Total positive pairs: {len(pos)}", flush=True)
    
    # Batch the pairs (memory)
    batch_size = 512
    pair_batches = []
    for s in range(0, len(pos), batch_size):
        sub = pos[s:s + batch_size]
        mi = torch.LongTensor([p[0] for p in sub]).to(device)
        di = torch.LongTensor([p[1] for p in sub]).to(device)
        pair_batches.append((mi, di))
    print(f"  Batched into {len(pair_batches)} batches of size <= {batch_size}", flush=True)
    
    # Sanity check: prediction score on first batch
    with torch.no_grad():
        s_check = torch.sigmoid(model(data, pair_batches[0][0], pair_batches[0][1])).cpu().numpy()
        print(f"  Sanity: first batch pos pair scores: mean={s_check.mean():.4f}, "
              f"min={s_check.min():.4f}, max={s_check.max():.4f}", flush=True)
    
    # Compute IG
    print(f"\n  Computing IG with n_steps=50 ...", flush=True)
    t_ig = time.time()
    ig = integrated_gradients_on_ge(model, data, pair_batches, n_steps=50, verbose=True)
    print(f"  IG done in {time.time()-t_ig:.0f}s", flush=True)
    print(f"  ig shape: {tuple(ig.shape)}", flush=True)
    
    # Per-gene importance: sum of |IG| over feature dim
    gene_importance = ig.abs().sum(dim=-1).cpu().numpy()  # (ng,)
    print(f"  gene_importance range: min={gene_importance.min():.4f}, "
          f"max={gene_importance.max():.4f}, mean={gene_importance.mean():.4f}",
          flush=True)
    
    # Save the full per-gene IG importance for further analysis
    np.save(f"{OUT_DIR}/gene_ig_importance_d1.npy", gene_importance)
    np.save(f"{OUT_DIR}/gene_ig_full_d1.npy", ig.cpu().numpy())
    
    # Top-K bridge genes (filter to genes that are active: appear in mg or dg)
    mirna_deg = data['mirna_deg_per_gene']
    drug_deg = data['drug_deg_per_gene']
    active_mask = (mirna_deg > 0) & (drug_deg > 0)
    active_indices = np.where(active_mask)[0]
    print(f"\n  Active bridge genes (mirna_deg>0 AND drug_deg>0): {len(active_indices)}")
    
    # Among active genes, rank by IG importance
    active_importance = gene_importance[active_indices]
    sort_idx = np.argsort(-active_importance)
    
    # Load gene_mapping for printing
    gene_map = pd.read_csv("data/processed/gene_mapping.csv")
    
    # Old bridge score for comparison
    old_score = mirna_deg * drug_deg
    
    print(f"\n  {'='*78}")
    print(f"  TOP-20 BRIDGE GENES (RANKED BY IG IMPORTANCE)")
    print(f"  {'='*78}")
    print(f"  {'Rank':<6}{'Gene':<14}{'IG_imp':>12}{'old_rank':>10}{'miRNA_deg':>12}{'Drug_deg':>10}{'OldScore':>12}")
    print(f"  {'-'*78}")
    
    # Determine ranking under old score
    old_ranking = np.argsort(-old_score)
    old_rank_of = {g: r for r, g in enumerate(old_ranking, 1)}
    
    top20_records = []
    for rank, ai in enumerate(sort_idx[:20], 1):
        g_idx = int(active_indices[ai])
        g_name = gene_map.iloc[g_idx]['gene_name']
        ig_imp = float(gene_importance[g_idx])
        old_r = old_rank_of.get(g_idx, '-')
        m_deg = int(mirna_deg[g_idx])
        d_deg = int(drug_deg[g_idx])
        old_s = int(old_score[g_idx])
        print(f"  {rank:<6}{g_name:<14}{ig_imp:>12.4f}{old_r:>10}{m_deg:>12}{d_deg:>10}{old_s:>12}")
        top20_records.append({
            'rank_ig': rank,
            'gene_idx': g_idx,
            'gene_name': g_name,
            'ig_importance': ig_imp,
            'old_rank_by_degree_product': old_r,
            'mirna_deg': m_deg,
            'drug_deg': d_deg,
            'old_bridge_score': old_s,
        })
    
    # Save Top-20 records
    pd.DataFrame(top20_records).to_csv(f"{OUT_DIR}/bridge_genes_ig_top20.csv", index=False)
    
    # Also print the OLD top-10 (degree product) for comparison
    print(f"\n  {'='*78}")
    print(f"  OLD TOP-10 BRIDGE GENES (BY DEGREE PRODUCT, for comparison)")
    print(f"  {'='*78}")
    print(f"  {'Rank':<6}{'Gene':<14}{'OldScore':>12}{'IG_imp':>12}{'IG_rank':>10}")
    print(f"  {'-'*78}")
    
    # Build IG ranking dict
    ig_rank_of = {}
    for r, ai in enumerate(sort_idx, 1):
        ig_rank_of[int(active_indices[ai])] = r
    
    for rank, g_idx in enumerate(old_ranking[:10], 1):
        g_name = gene_map.iloc[g_idx]['gene_name']
        old_s = int(old_score[g_idx])
        ig_imp = float(gene_importance[g_idx])
        ig_r = ig_rank_of.get(int(g_idx), '-')
        print(f"  {rank:<6}{g_name:<14}{old_s:>12}{ig_imp:>12.4f}{ig_r:>10}")
    
    print(f"\n  Total IG run time: {time.time()-t0:.0f}s", flush=True)
    print(f"\n  Saved:")
    print(f"    {OUT_DIR}/gene_ig_importance_d1.npy  (shape: {gene_importance.shape})")
    print(f"    {OUT_DIR}/gene_ig_full_d1.npy        (shape: {tuple(ig.shape)})")
    print(f"    {OUT_DIR}/bridge_genes_ig_top20.csv")
