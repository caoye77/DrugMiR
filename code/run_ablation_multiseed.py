"""
DrugMiR Ablation 5-seed runner.

Faithfully re-implements hp_finetune.py's DrugMiR_Hybrid with optional
ablation flags. Trains the same way as run_multiseed.py but for ablation
variants, with 5 seeds {42, 123, 2024, 7777, 9999} x 5-fold CV per variant.

Variants (matching paper Table IV):
  - full:        complete 3-channel architecture
  - no_gating:   GG -> plain residual (g=1 always)
  - no_bridge:   remove Gene Bridge channel
  - no_homo:     remove HomoGCN channel
  - no_hybrid:   HybridEnc -> pure learnable embedding (no feature MLP)

Usage:
  python3 run_ablation_multiseed.py --variant no_hybrid --dataset D1
  python3 run_ablation_multiseed.py --all_variants --dataset D1
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
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================== Data loading ==============================

def load_data(dd, km=15, kd=10):
    """Identical to hp_finetune.py.load_data()."""
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


# ============================== Model components ==============================

class GG(nn.Module):
    """Gated GCN block. Set use_gate=False for 'no_gating' ablation."""
    def __init__(s, h, dr, use_gate=True):
        super().__init__()
        s.gcn = GCNConv(h, h)
        s.norm = nn.BatchNorm1d(h)
        s.drop = nn.Dropout(dr)
        s.use_gate = use_gate
        if use_gate:
            s.gate = nn.Linear(2 * h, h)
    def forward(s, x, e):
        ht = s.drop(s.norm(F.relu(s.gcn(x, e))))
        if s.use_gate:
            g = torch.sigmoid(s.gate(torch.cat([x, ht], -1)))
            return x + g * ht
        else:
            # Plain residual: x + ht
            return x + ht


class GB(nn.Module):
    """Gene Bridge block (unchanged from hp_finetune.py)."""
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
    """Hybrid Encoder with feature + embedding gating.
    Set use_hybrid=False for 'no_hybrid' ablation (pure embedding only).
    """
    def __init__(s, n, fd, h, dr, use_hybrid=True):
        super().__init__()
        s.use_hybrid = use_hybrid
        if use_hybrid:
            s.feat = nn.Sequential(
                nn.Linear(fd, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
                nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr))
            s.gate = nn.Linear(2 * h, h)
        s.emb = nn.Embedding(n, h)
    def forward(s, feat, has_feat=None):
        eh = s.emb.weight
        if not s.use_hybrid:
            # Pure learnable embedding, no feature channel
            return eh
        fh = s.feat(feat)
        g = torch.sigmoid(s.gate(torch.cat([fh, eh], -1)))
        if has_feat is not None:
            mask = has_feat.unsqueeze(1)
            return mask * (g * fh + (1 - g) * eh) + (1 - mask) * eh
        return g * fh + (1 - g) * eh


class DrugMiR_Ablation(nn.Module):
    """DrugMiR with ablation switches.
    
    variant: 'full' | 'no_gating' | 'no_bridge' | 'no_homo' | 'no_hybrid'
    """
    def __init__(s, nm, nd, md, dd, ng, variant='full',
                 h=256, dr=0.5, n_gcn=2, n_br=2):
        super().__init__()
        s.variant = variant
        use_hybrid = (variant != 'no_hybrid')
        use_gating = (variant != 'no_gating')
        s.use_bridge = (variant != 'no_bridge')
        s.use_homo = (variant != 'no_homo')
        
        s.me = HybridEnc(nm, md, h, dr, use_hybrid=use_hybrid)
        s.de = HybridEnc(nd, dd, h, dr, use_hybrid=use_hybrid)
        s.ge = nn.Embedding(ng, h)
        if s.use_homo:
            s.mgcn = nn.ModuleList([GG(h, dr, use_gate=use_gating) for _ in range(n_gcn)])
            s.dgcn = nn.ModuleList([GG(h, dr, use_gate=use_gating) for _ in range(n_gcn)])
        if s.use_bridge:
            s.br = nn.ModuleList([GB(h, dr) for _ in range(n_br)])
        
        # Predictor input dim depends on which channels are active
        # Channel 1 (m0/d0) always active. Homo (mh/dh) optional. Bridge (mb/db) optional.
        n_active_channels = 1 + (1 if s.use_homo else 0) + (1 if s.use_bridge else 0)
        s.pred = nn.Sequential(
            nn.Linear(2 * h * n_active_channels, 2 * h),
            nn.BatchNorm1d(2 * h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(2 * h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h, 1))
    
    def forward(s, data, mi, di):
        m0 = s.me(data['mirna_feat'], data.get('mirna_has_feat'))
        d0 = s.de(data['drug_feat'], data.get('drug_has_feat'))
        
        m_reprs = [m0]
        d_reprs = [d0]
        
        if s.use_homo:
            mh = m0; dh = d0
            for l in s.mgcn:
                mh = l(mh, data['mirna_sim_edge'])
            for l in s.dgcn:
                dh = l(dh, data['drug_sim_edge'])
            m_reprs.append(mh)
            d_reprs.append(dh)
        
        if s.use_bridge:
            mb = m0; db = d0; gh = s.ge.weight
            for l in s.br:
                mb, db, gh = l(mb, db, gh, data['mg_src'], data['mg_dst'],
                                data['dg_src'], data['dg_dst'], data['n_gene'])
            m_reprs.append(mb)
            d_reprs.append(db)
        
        m_cat = torch.cat(m_reprs, -1)
        d_cat = torch.cat(d_reprs, -1)
        joint = torch.cat([m_cat[mi], d_cat[di]], -1)
        return s.pred(joint).squeeze(-1)


# ============================== Train / eval ==============================

def train_one_epoch(model, data, train_pos, optimizer, bs=2048):
    model.train()
    neg = sn(data['assoc'], train_pos, len(train_pos))
    pairs = train_pos + neg
    labels = [1.0] * len(train_pos) + [0.0] * len(neg)
    idx = np.random.permutation(len(pairs))
    tl = 0; nb = 0
    for s in range(0, len(idx), bs):
        bi = idx[s:s + bs]
        bp = [pairs[i] for i in bi]
        bl = torch.FloatTensor([labels[i] for i in bi]).to(device)
        mi = torch.LongTensor([p[0] for p in bp]).to(device)
        di = torch.LongTensor([p[1] for p in bp]).to(device)
        optimizer.zero_grad()
        logits = model(data, mi, di)
        loss = F.binary_cross_entropy_with_logits(logits, bl)
        loss.backward(); optimizer.step()
        tl += loss.item(); nb += 1
    return tl / nb


@torch.no_grad()
def evaluate(model, data, test_pos):
    model.eval()
    neg = sn(data['assoc'], test_pos, len(test_pos))
    pairs = test_pos + neg
    labels = np.array([1.0] * len(test_pos) + [0.0] * len(neg))
    mi = torch.LongTensor([p[0] for p in pairs]).to(device)
    di = torch.LongTensor([p[1] for p in pairs]).to(device)
    logits = model(data, mi, di)
    scores = torch.sigmoid(logits).cpu().numpy()
    
    auc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    p, r, t = precision_recall_curve(labels, scores)
    f1c = 2 * p * r / (p + r + 1e-10)
    bi = int(np.argmax(f1c[:-1]))
    thr = float(t[bi])
    pb = (scores >= thr).astype(int)
    return {'auc': float(auc), 'aupr': float(aupr),
            'f1': float(f1_score(labels, pb)),
            'prec': float(precision_score(labels, pb, zero_division=0)),
            'rec': float(recall_score(labels, pb)),
            'thr': thr}


def run_variant(data, variant, seeds=[42, 123, 2024, 7777, 9999], n_fold=5,
                h=256, dr=0.5, n_gcn=2, n_br=2, lr=1e-3, wd=2e-4,
                ep=200, pat=15):
    """Run one ablation variant across all (seed, fold) combos."""
    pos = data['pos_pairs']
    md = data['mirna_feat'].shape[1]
    dd_ = data['drug_feat'].shape[1]
    ng = data['n_gene']; nm = data['n_mirna']; nd = data['n_drug']
    
    fold_results = []
    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        for fold_id, (tri, tei) in enumerate(kf.split(pos)):
            t0 = time.time()
            train_pos = [pos[i] for i in tri]
            test_pos  = [pos[i] for i in tei]
            
            model = DrugMiR_Ablation(nm, nd, md, dd_, ng, variant=variant,
                                       h=h, dr=dr, n_gcn=n_gcn, n_br=n_br).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            
            best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
            pc = 0
            for e in range(ep):
                train_one_epoch(model, data, train_pos, optimizer)
                if (e + 1) % 5 == 0:
                    m = evaluate(model, data, test_pos)
                    if m['auc'] > best['auc']:
                        best = m; pc = 0
                    else:
                        pc += 1
                    if pc >= pat:
                        break
            best['seed'] = seed; best['fold'] = fold_id
            fold_results.append(best)
            print(f"    [{variant}] seed={seed} fold={fold_id+1}: "
                  f"AUC={best['auc']:.4f} AUPR={best['aupr']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    
    summary = {}
    for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        vals = [r[k] for r in fold_results]
        summary[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    summary['fold_details'] = fold_results
    summary['variant'] = variant
    summary['n_seeds'] = len(seeds)
    summary['n_folds'] = n_fold
    summary['seeds'] = seeds
    return summary


# ============================== Main ==============================

ALL_VARIANTS = ['full', 'no_gating', 'no_bridge', 'no_homo', 'no_hybrid']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['D1', 'D2'], required=True)
    ap.add_argument('--variant', choices=ALL_VARIANTS + ['all'], default='all')
    ap.add_argument('--out_dir', default=os.path.expanduser('~/work/DrugMiR/ablation_outputs'))
    args = ap.parse_args()
    
    DD = {'D1': os.path.expanduser('~/work/DrugMiR/data/processed'),
          'D2': os.path.expanduser('~/work/DrugMiR/DMGAT_processed')}[args.dataset]
    print(f"Device: {device}", flush=True)
    print(f"Loading data from {DD}...", flush=True)
    data = load_data(DD)
    print(f"  n_mirna={data['n_mirna']} n_drug={data['n_drug']} "
          f"n_pos={len(data['pos_pairs'])} n_gene={data['n_gene']}", flush=True)
    
    variants_to_run = ALL_VARIANTS if args.variant == 'all' else [args.variant]
    
    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}
    for variant in variants_to_run:
        print(f"\n{'='*72}\nABLATION variant={variant} on {args.dataset}\n{'='*72}", flush=True)
        t0 = time.time()
        summary = run_variant(data, variant)
        summary['dataset'] = args.dataset
        summary['total_time'] = time.time() - t0
        out_file = f"{args.out_dir}/ablation_{args.dataset}_{variant}.json"
        with open(out_file, 'w') as f:
            json.dump(summary, f, indent=2)
        all_results[variant] = summary
        print(f"\n  variant={variant} summary (mean ± std over {summary['n_seeds']} seeds × {summary['n_folds']} folds):")
        for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
            print(f"    {k.upper():6s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
        print(f"  Total: {summary['total_time']:.0f}s ({summary['total_time']/60:.1f} min)", flush=True)
        print(f"  Saved to {out_file}", flush=True)
    
    # Final table
    if len(all_results) > 1:
        print(f"\n{'='*72}\nABLATION FINAL SUMMARY on {args.dataset}\n{'='*72}", flush=True)
        full_auc = all_results.get('full', {}).get('auc', {}).get('mean')
        for v in variants_to_run:
            auc = all_results[v]['auc']['mean']
            std = all_results[v]['auc']['std']
            delta = (auc - full_auc) * 100 if full_auc and v != 'full' else 0
            marker = '' if v == 'full' else f"Δ={delta:+.2f}%"
            print(f"  {v:12s}: AUC={auc:.4f} ± {std:.4f}  {marker}", flush=True)
