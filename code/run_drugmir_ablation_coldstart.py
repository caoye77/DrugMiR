"""
DrugMiR ABLATION cold-start runner: variant {full, no_hybrid} × {S2, S3, S4}.

Combines:
  - data pipeline from run_drugmir_coldstart.py (per-fold KNN, train-only neg
    sampling, gene-bridge unchanged)
  - DrugMiR_Ablation model from run_ablation_multiseed.py (HybridEnc with
    use_hybrid flag)

This script answers the question: "Does Channel 1 (Hybrid Embedding) drive
DrugMiR's cold-start advantage?" by reporting Full and w/o Hybrid side by side
on the same fold splits.

Usage:
  python3 run_drugmir_ablation_coldstart.py --dataset D1 --setting S4 \
      --variant no_hybrid --seed 42
"""
import os, sys, json, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              precision_recall_curve)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_splitter import ColdStartSplitter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================== Model with ablation switches ======================

class GG(nn.Module):
    """Gated GCN (Channel 2 sub-block); always with gating in this script."""
    def __init__(s, h, dr):
        super().__init__()
        s.gcn = GCNConv(h, h)
        s.gate = nn.Linear(2 * h, h)
        s.norm = nn.BatchNorm1d(h)
        s.drop = nn.Dropout(dr)
    def forward(s, x, e):
        ht = s.drop(s.norm(F.relu(s.gcn(x, e))))
        g = torch.sigmoid(s.gate(torch.cat([x, ht], -1)))
        return x + g * ht


class GB(nn.Module):
    """Gene Bridge (Channel 3)."""
    def __init__(s, h, dr):
        super().__init__()
        s.mg = nn.Linear(2 * h, h)
        s.dg = nn.Linear(2 * h, h)
        s.norm = nn.BatchNorm1d(h)
        s.drop = nn.Dropout(dr)
    def forward(s, mh, dh, gh, ms, md, ds, dd, ng):
        gm = scatter_mean(mh[ms], md, dim=0, dim_size=ng)
        gd = scatter_mean(dh[ds], dd, dim=0, dim_size=ng)
        ga = s.drop(s.norm(F.relu(gh + gm + gd)))
        mfg = scatter_mean(ga[md], ms, dim=0, dim_size=mh.size(0))
        dfg = scatter_mean(ga[dd], ds, dim=0, dim_size=dh.size(0))
        new_mh = mh + torch.sigmoid(s.mg(torch.cat([mh, mfg], -1))) * mfg
        new_dh = dh + torch.sigmoid(s.dg(torch.cat([dh, dfg], -1))) * dfg
        return new_mh, new_dh, ga


class HybridEnc(nn.Module):
    """Channel 1. Set use_hybrid=False to disable feature path (pure emb)."""
    def __init__(s, n, fd, h, dr, use_hybrid=True):
        super().__init__()
        s.use_hybrid = use_hybrid
        if use_hybrid:
            s.feat = nn.Sequential(
                nn.Linear(fd, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr))
        s.emb = nn.Embedding(n, h)
        if use_hybrid:
            s.gate = nn.Linear(2 * h, h)
    def forward(s, feat, has_feat=None):
        eh = s.emb.weight
        if not s.use_hybrid:
            return eh
        fh = s.feat(feat)
        g = torch.sigmoid(s.gate(torch.cat([fh, eh], -1)))
        if has_feat is not None:
            mask = has_feat.unsqueeze(1)
            return mask * (g * fh + (1 - g) * eh) + (1 - mask) * eh
        return g * fh + (1 - g) * eh


class DrugMiR_Ablation(nn.Module):
    def __init__(s, nm, nd, md, dd_, ng, variant='full',
                 h=256, dr=0.5, n_gcn=2, n_br=2):
        super().__init__()
        s.variant = variant
        use_hybrid = (variant != 'no_hybrid')
        s.me = HybridEnc(nm, md, h, dr, use_hybrid=use_hybrid)
        s.de = HybridEnc(nd, dd_, h, dr, use_hybrid=use_hybrid)
        s.ge = nn.Embedding(ng, h)
        s.mgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        s.dgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        s.br = nn.ModuleList([GB(h, dr) for _ in range(n_br)])
        s.pred = nn.Sequential(
            nn.Linear(6 * h, 2 * h), nn.BatchNorm1d(2 * h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(2 * h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
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
        if s.variant == 'no_bridge':
            mb = torch.zeros_like(mb); db = torch.zeros_like(db)
        return s.pred(torch.cat([torch.cat([m0, mh, mb], -1)[mi],
                                  torch.cat([d0, dh, db], -1)[di]], -1)).squeeze(-1)


# ============================== Data loading ==============================

def load_static_data(data_dir):
    assoc = np.load(f"{data_dir}/association_matrix.npy")
    mf = np.load(f"{data_dir}/mirna_kmer_features.npy")
    df = np.load(f"{data_dir}/drug_morgan_features.npy")
    ms = np.load(f"{data_dir}/mirna_similarity.npy")
    ds = np.load(f"{data_dir}/drug_similarity.npy")
    mg = np.load(f"{data_dir}/mirna_gene_matrix.npy")
    dg = np.load(f"{data_dir}/drug_gene_matrix.npy")
    mg_r, mg_c = np.nonzero(mg)
    dg_r, dg_c = np.nonzero(dg)
    n_gene = max(mg_c.max() if len(mg_c) > 0 else 0,
                 dg_c.max() if len(dg_c) > 0 else 0) + 1
    return {'assoc': assoc, 'mf': mf, 'df': df, 'ms': ms, 'ds': ds,
            'mg_r': mg_r, 'mg_c': mg_c, 'dg_r': dg_r, 'dg_c': dg_c,
            'n_mirna': assoc.shape[0], 'n_drug': assoc.shape[1], 'n_gene': n_gene}


def build_fold_data(static, fold, km=15, kd=10):
    """Build per-fold data dict with KNN graphs restricted to train entities."""
    ms = static['ms']; ds = static['ds']
    train_mi = np.where(fold.train_mirna_mask)[0]
    train_dr = np.where(fold.train_drug_mask)[0]
    train_mi_set = set(train_mi.tolist())
    train_dr_set = set(train_dr.tolist())
    
    # miRNA KNN (only train→train edges)
    s1, d1 = [], []
    for i in range(static['n_mirna']):
        if i not in train_mi_set: continue
        si = ms[i].copy()
        # Mask out: self + all non-train
        for j in range(len(si)):
            if j == i or j not in train_mi_set:
                si[j] = -1
        topk = np.argsort(si)[-km:]
        for j in topk:
            if si[j] > 0:
                s1.extend([i, j]); d1.extend([j, i])
    
    # drug KNN (only train→train)
    s2, d2 = [], []
    for i in range(static['n_drug']):
        if i not in train_dr_set: continue
        si = ds[i].copy()
        for j in range(len(si)):
            if j == i or j not in train_dr_set:
                si[j] = -1
        topk = np.argsort(si)[-kd:]
        for j in topk:
            if si[j] > 0:
                s2.extend([i, j]); d2.extend([j, i])
    
    mf = torch.FloatTensor(static['mf']).to(device)
    df = torch.FloatTensor(static['df']).to(device)
    d = {
        'assoc': static['assoc'],
        'mirna_feat': mf, 'drug_feat': df,
        'n_mirna': static['n_mirna'], 'n_drug': static['n_drug'],
        'mirna_has_feat': torch.FloatTensor((static['mf'].sum(1) > 0).astype(float)).to(device),
        'drug_has_feat':  torch.FloatTensor((static['df'].sum(1) > 0).astype(float)).to(device),
        'mirna_sim_edge': torch.LongTensor([s1, d1]).to(device) if s1
                          else torch.zeros((2, 0), dtype=torch.long, device=device),
        'drug_sim_edge':  torch.LongTensor([s2, d2]).to(device) if s2
                          else torch.zeros((2, 0), dtype=torch.long, device=device),
        'mg_src': torch.LongTensor(static['mg_r']).to(device),
        'mg_dst': torch.LongTensor(static['mg_c']).to(device),
        'dg_src': torch.LongTensor(static['dg_r']).to(device),
        'dg_dst': torch.LongTensor(static['dg_c']).to(device),
        'n_gene': static['n_gene'],
        'train_mi_mask': fold.train_mirna_mask,
        'train_dr_mask': fold.train_drug_mask,
    }
    return d


# ============================== Train / eval ==============================

def sample_negatives_train(assoc, train_mi_mask, train_dr_mask, n):
    train_mi_idx = np.where(train_mi_mask)[0]
    train_dr_idx = np.where(train_dr_mask)[0]
    neg = []; tries = 0
    while len(neg) < n and tries < n * 200:
        i = int(np.random.choice(train_mi_idx))
        j = int(np.random.choice(train_dr_idx))
        if assoc[i, j] == 0:
            neg.append((i, j))
        tries += 1
    while len(neg) < n:
        i = int(np.random.choice(train_mi_idx))
        j = int(np.random.choice(train_dr_idx))
        neg.append((i, j))
    return neg


def sample_negatives_test(assoc, n):
    nm, nd = assoc.shape
    neg = []
    while len(neg) < n:
        i = np.random.randint(0, nm); j = np.random.randint(0, nd)
        if assoc[i, j] == 0:
            neg.append((i, j))
    return neg


def train_one_epoch(model, data, train_pos, optimizer, bs=2048):
    model.train()
    neg = sample_negatives_train(data['assoc'], data['train_mi_mask'],
                                   data['train_dr_mask'], len(train_pos))
    pairs = list(train_pos) + neg
    labels = [1.0] * len(train_pos) + [0.0] * len(neg)
    idx = np.random.permutation(len(pairs))
    for s in range(0, len(idx), bs):
        bi = idx[s:s+bs]
        bp = [pairs[i] for i in bi]
        bl = torch.FloatTensor([labels[i] for i in bi]).to(device)
        mi = torch.LongTensor([p[0] for p in bp]).to(device)
        di = torch.LongTensor([p[1] for p in bp]).to(device)
        optimizer.zero_grad()
        logits = model(data, mi, di)
        loss = F.binary_cross_entropy_with_logits(logits, bl)
        loss.backward(); optimizer.step()


@torch.no_grad()
def evaluate_5metrics(model, data, test_pos):
    model.eval()
    neg = sample_negatives_test(data['assoc'], len(test_pos))
    pairs = list(test_pos) + neg
    labels = np.array([1.0] * len(test_pos) + [0.0] * len(neg))
    mi = torch.LongTensor([p[0] for p in pairs]).to(device)
    di = torch.LongTensor([p[1] for p in pairs]).to(device)
    logits = model(data, mi, di)
    scores = torch.sigmoid(logits).cpu().numpy()
    auc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    p, r, t = precision_recall_curve(labels, scores)
    f1c = 2 * p * r / (p + r + 1e-10)
    bi = int(np.argmax(f1c[:-1]))
    thr = float(t[bi])
    pb = (scores >= thr).astype(int)
    return {'auc': auc, 'aupr': aupr,
            'f1': float(f1_score(labels, pb)),
            'prec': float(precision_score(labels, pb, zero_division=0)),
            'rec': float(recall_score(labels, pb)),
            'thr': thr}


def run_one_fold(static, fold, variant, seed=42,
                 h=256, dr=0.5, n_gcn=2, n_br=2, lr=1e-3, wd=2e-4,
                 ep=200, pat=15):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    data = build_fold_data(static, fold)
    train_pos = list(map(tuple, fold.train_pairs.tolist()))
    test_pos  = list(map(tuple, fold.test_pairs.tolist()))
    
    md = data['mirna_feat'].shape[1]
    dd_ = data['drug_feat'].shape[1]
    ng = data['n_gene']; nm = data['n_mirna']; nd = data['n_drug']
    model = DrugMiR_Ablation(nm, nd, md, dd_, ng, variant=variant,
                               h=h, dr=dr, n_gcn=n_gcn, n_br=n_br).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
    pc = 0
    for e in range(ep):
        train_one_epoch(model, data, train_pos, opt)
        if (e + 1) % 5 == 0:
            m = evaluate_5metrics(model, data, test_pos)
            if m['auc'] > best['auc']:
                best = m; pc = 0
            else:
                pc += 1
            if pc >= pat:
                break
    return best


def run_dataset_coldstart(dataset_name, data_dir, setting, variant,
                           seed=42, n_fold=5):
    print(f"\n{'='*72}\nDrugMiR-{variant} cold-start: {dataset_name} / {setting} "
          f"/ seed={seed}\n{'='*72}", flush=True)
    t0 = time.time()
    static = load_static_data(data_dir)
    print(f"  n_mirna={static['n_mirna']} n_drug={static['n_drug']} "
          f"n_pos={int(static['assoc'].sum())}", flush=True)
    
    splitter = ColdStartSplitter(static['assoc'], n_folds=n_fold, seed=seed,
                                  min_test_positives=15)
    folds = splitter.split(setting)
    
    fold_results = []
    for f in folds:
        tf0 = time.time()
        best = run_one_fold(static, f, variant=variant, seed=seed)
        fold_results.append(best)
        print(f"  Fold {f.fold_id+1}/{n_fold}: AUC={best['auc']:.4f} "
              f"AUPR={best['aupr']:.4f} F1={best['f1']:.4f} "
              f"P={best['prec']:.4f} R={best['rec']:.4f} "
              f"(train_pos={len(f.train_pairs)} test_pos={len(f.test_pairs)} "
              f"trainM={f.train_mirna_mask.sum()} trainN={f.train_drug_mask.sum()}) "
              f"({time.time()-tf0:.0f}s)", flush=True)
    
    summary = {}
    for k_ in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        vals = [r[k_] for r in fold_results]
        summary[k_] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    summary['fold_details'] = fold_results
    summary['dataset'] = dataset_name; summary['setting'] = setting
    summary['variant'] = variant; summary['seed'] = seed
    summary['total_time'] = time.time() - t0
    print(f"\n  {dataset_name} / {setting} / {variant} summary:")
    for k_ in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        print(f"    {k_.upper():6s}: {summary[k_]['mean']:.4f} ± {summary[k_]['std']:.4f}")
    print(f"  Total: {summary['total_time']:.0f}s\n", flush=True)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['D1', 'D2'], required=True)
    ap.add_argument('--setting', choices=['S2', 'S3', 'S4'], required=True)
    ap.add_argument('--variant', choices=['full', 'no_hybrid', 'no_bridge'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir',
                    default=os.path.expanduser('~/work/DrugMiR/coldstart_outputs'))
    args = ap.parse_args()
    DD = {'D1': os.path.expanduser('~/work/DrugMiR/data/processed'),
          'D2': os.path.expanduser('~/work/DrugMiR/DMGAT_processed')}[args.dataset]
    res = run_dataset_coldstart(args.dataset, DD, args.setting, args.variant,
                                  seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = (f"{args.out_dir}/drugmir_{args.variant}_"
                f"{args.dataset}_{args.setting}_seed{args.seed}.json")
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  Saved to {out_file}", flush=True)
