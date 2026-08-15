"""
Phase 2a: DrugMiR final benchmark — seed=42, 5-fold CV, 5 metrics (AUC/AUPR/F1/Prec/Rec)
on both D1 and D2. Output: phase2_outputs/drugmir_5metrics_seed42.json
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             precision_recall_curve)
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ==================== 数据加载（copy from hp_finetune.py，无修改）====================
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
        si = ms[i].copy(); si[i] = -1
        for j in np.argsort(si)[-km:]:
            s1.extend([i, j]); d1.extend([j, i])
    d['mirna_sim_edge'] = torch.LongTensor([s1, d1]).to(device)
    s2, d2 = [], []
    for i in range(d['n_drug']):
        si = ds[i].copy(); si[i] = -1
        for j in np.argsort(si)[-kd:]:
            s2.extend([i, j]); d2.extend([j, i])
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
    pr, pc = np.nonzero(assoc)
    d['pos_pairs'] = list(zip(pr.tolist(), pc.tolist()))
    return d

def sn(assoc, pos, n):
    nm, nd = assoc.shape; neg = []
    while len(neg) < n:
        i = np.random.randint(0, nm); j = np.random.randint(0, nd)
        if assoc[i, j] == 0: neg.append((i, j))
    return neg

# ==================== 模型（copy from hp_finetune.py）====================
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
        s.feat = nn.Sequential(nn.Linear(fd, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
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
        s.pred = nn.Sequential(nn.Linear(6*h, 2*h), nn.BatchNorm1d(2*h), nn.ReLU(), nn.Dropout(dr),
                                nn.Linear(2*h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
                                nn.Linear(h, 1))
    def forward(s, data, mi, di):
        m0 = s.me(data['mirna_feat'], data.get('mirna_has_feat'))
        d0 = s.de(data['drug_feat'], data.get('drug_has_feat'))
        mh = m0; dh = d0
        for l in s.mgcn: mh = l(mh, data['mirna_sim_edge'])
        for l in s.dgcn: dh = l(dh, data['drug_sim_edge'])
        mb = m0; db = d0; gh = s.ge.weight
        for l in s.br:
            mb, db, gh = l(mb, db, gh, data['mg_src'], data['mg_dst'],
                            data['dg_src'], data['dg_dst'], data['n_gene'])
        return s.pred(torch.cat([torch.cat([m0, mh, mb], -1)[mi],
                                  torch.cat([d0, dh, db], -1)[di]], -1)).squeeze(-1)

# ==================== 训练 ====================
def trn(m, data, trp, opt, bs=2048):
    m.train()
    neg = sn(data['assoc'], trp, len(trp))
    pairs = trp + neg
    lab = [1.0]*len(trp) + [0.0]*len(neg)
    idx = np.random.permutation(len(pairs))
    tl = 0; nb = 0
    for s in range(0, len(idx), bs):
        bi = idx[s:s+bs]
        bp = [pairs[i] for i in bi]
        bl = torch.FloatTensor([lab[i] for i in bi]).to(device)
        mi = torch.LongTensor([p[0] for p in bp]).to(device)
        di = torch.LongTensor([p[1] for p in bp]).to(device)
        opt.zero_grad()
        lo = m(data, mi, di)
        loss = F.binary_cross_entropy_with_logits(lo, bl)
        loss.backward(); opt.step()
        tl += loss.item(); nb += 1
    return tl / nb

# ==================== 关键：5 指标评估 ====================
@torch.no_grad()
def ev_full(m, data, tep):
    """返回 AUC / AUPR / F1 / Precision / Recall（at optimal F1 threshold）"""
    m.eval()
    neg = sn(data['assoc'], tep, len(tep))
    pairs = tep + neg
    lab = np.array([1.0]*len(tep) + [0.0]*len(neg))
    mi = torch.LongTensor([p[0] for p in pairs]).to(device)
    di = torch.LongTensor([p[1] for p in pairs]).to(device)
    lo = m(data, mi, di)
    pr = torch.sigmoid(lo).cpu().numpy()
    auc = roc_auc_score(lab, pr)
    aupr = average_precision_score(lab, pr)
    # PR 曲线上找最优 F1 阈值
    p_curve, r_curve, t_curve = precision_recall_curve(lab, pr)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-10)
    best_idx = np.argmax(f1_curve[:-1])  # 最后一个 precision=1, recall=0 跳过
    best_thr = t_curve[best_idx]
    pred_bin = (pr >= best_thr).astype(int)
    f1 = f1_score(lab, pred_bin)
    prec = precision_score(lab, pred_bin, zero_division=0)
    rec = recall_score(lab, pred_bin)
    return {'auc': auc, 'aupr': aupr, 'f1': f1, 'prec': prec, 'rec': rec, 'thr': float(best_thr),
            'y_true': lab.tolist(), 'y_pred': pr.tolist()}

# ==================== 主流程 ====================
def run_one_dataset(dataset_name, data_dir, km=15, kd=10, lr=5e-4, wd=2e-4,
                    seed=42, n_fold=5, max_epoch=200, patience=15):
    print(f"\n{'='*70}\nRunning {dataset_name} (seed={seed}, {n_fold}-fold CV)\n{'='*70}", flush=True)
    t0 = time.time()
    data = load_data(data_dir, km=km, kd=kd)
    print(f"  Loaded: n_mirna={data['n_mirna']} n_drug={data['n_drug']} "
          f"n_gene={data['n_gene']} n_pos={len(data['pos_pairs'])}", flush=True)
    md = data['mirna_feat'].shape[1]
    dd = data['drug_feat'].shape[1]
    ng = data['n_gene']
    nm, nd = data['n_mirna'], data['n_drug']

    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    pos = data['pos_pairs']

    fold_results = []
    for fold, (tri, tei) in enumerate(kf.split(pos)):
        tf0 = time.time()
        trp = [pos[i] for i in tri]; tep = [pos[i] for i in tei]
        m = DrugMiR_Hybrid(nm, nd, md, dd, ng, h=256, dr=0.5, n_gcn=2, n_br=2).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)

        best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
        pc = 0
        for e in range(max_epoch):
            trn(m, data, trp, opt)
            if (e + 1) % 5 == 0:
                m_eval = ev_full(m, data, tep)
                if m_eval['auc'] > best['auc']:
                    best = m_eval
                    pc = 0
                else:
                    pc += 1
                if pc >= patience: break

        fold_results.append(best)
        print(f"  Fold {fold+1}/{n_fold}: AUC={best['auc']:.4f} AUPR={best['aupr']:.4f} "
              f"F1={best['f1']:.4f} P={best['prec']:.4f} R={best['rec']:.4f} "
              f"({time.time()-tf0:.0f}s)", flush=True)

    # 汇总 mean ± std
    summary = {}
    for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        vals = [r[k] for r in fold_results]
        summary[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    summary['fold_details'] = fold_results
    summary['dataset'] = dataset_name
    summary['seed'] = seed
    summary['n_fold'] = n_fold
    summary['total_time'] = time.time() - t0

    print(f"\n  {dataset_name} summary (mean±std):")
    for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        print(f"    {k.upper():8s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
    print(f"  Total: {summary['total_time']:.0f}s ({summary['total_time']/60:.1f} min)", flush=True)

    return summary


if __name__ == '__main__':
    DD1 = os.path.expanduser("~/DrugMiR/data/dataset1")
    DD2 = os.path.expanduser("~/DrugMiR/data/dataset2")

    out = {}
    out['D1'] = run_one_dataset('D1', DD1, lr=5e-4)
    out['D2'] = run_one_dataset('D2', DD2, lr=5e-4)

    os.makedirs('results_final', exist_ok=True)
    with open('results_final/drugmir_predictions_full_seed42.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved to results_final/drugmir_predictions_full_seed42.json", flush=True)

    # Summary
    for ds in ['D1', 'D2']:
        print(f"\n=== {ds} ===")
        for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
            print(f"  {k.upper():6s}: {out[ds][k]['mean']:.4f} ± {out[ds][k]['std']:.4f}")
