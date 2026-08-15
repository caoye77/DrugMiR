"""
Re-run DrugMiR 5-fold on D1 and D2 with seed=42, saving (y_true, y_pred)
per fold for Fig.2 ROC/PR curve regeneration.

Reuses hp_finetune.py's data loader and model classes verbatim.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Import everything from hp_finetune
sys.path.insert(0, '.')
from hp_finetune import (
    load_data, sn, trn, ev,
    GG, GB, HybridEnc, DrugMiR_Hybrid, device
)

@torch.no_grad()
def eval_save(m, data, tep, seed):
    """Like ev() but also returns y_true and y_pred arrays."""
    m.eval()
    np.random.seed(seed + 999)  # for stable negative sampling at eval time
    neg = sn(data['assoc'], tep, len(tep))
    pairs = tep + neg
    lab = np.array([1.0]*len(tep) + [0.0]*len(neg))
    mi = torch.LongTensor([p[0] for p in pairs]).to(device)
    di = torch.LongTensor([p[1] for p in pairs]).to(device)
    lo = m(data, mi, di)
    pr = torch.sigmoid(lo).cpu().numpy()
    return roc_auc_score(lab, pr), average_precision_score(lab, pr), lab, pr


def run_dataset(dd, ds_name, seed=42, nf=5, lr=0.001, wd=2e-4, ep=200, pat=15):
    print(f"\n{'='*60}\nLoading {ds_name} from {dd}\n{'='*60}", flush=True)
    data = load_data(dd)
    print(f"  n_mirna={data['n_mirna']}, n_drug={data['n_drug']}, n_gene={data['n_gene']}", flush=True)
    print(f"  n_positive={len(data['pos_pairs'])}", flush=True)

    pos = data['pos_pairs']
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    kf = KFold(n_splits=nf, shuffle=True, random_state=seed)

    fold_data = []  # list of (y_true, y_pred, auc, aupr) per fold
    t_total = time.time()

    for fold, (tri, tei) in enumerate(kf.split(pos)):
        trp = [pos[i] for i in tri]; tep = [pos[i] for i in tei]
        t0 = time.time()

        m = DrugMiR_Hybrid(
            data['n_mirna'], data['n_drug'],
            data['mirna_feat'].shape[1], data['drug_feat'].shape[1],
            data['n_gene'], h=256, dr=0.5, n_gcn=2, n_br=2
        ).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)

        ba, pc, b_lab, b_pr, b_auc, b_aupr = 0, 0, None, None, 0, 0

        for e in range(ep):
            trn(m, data, trp, opt)
            if (e+1) % 5 == 0:
                a, ap, lab, pr = eval_save(m, data, tep, seed=seed*1000 + fold)
                if a > ba:
                    ba = a; b_aupr = ap; b_lab = lab; b_pr = pr; pc = 0
                else:
                    pc += 1
                if pc >= pat: break

        elapsed = time.time() - t0
        print(f"  fold {fold+1}/{nf}: AUC={ba:.4f}, AUPR={b_aupr:.4f}, time={elapsed:.0f}s", flush=True)
        fold_data.append({
            'fold': fold,
            'y_true': b_lab.tolist(),
            'y_pred': b_pr.tolist(),
            'auc': float(ba),
            'aupr': float(b_aupr),
        })

    total_elapsed = time.time() - t_total
    aucs = [f['auc'] for f in fold_data]
    auprs = [f['aupr'] for f in fold_data]
    print(f"\n  {ds_name}: AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}", flush=True)
    print(f"  {ds_name}: AUPR = {np.mean(auprs):.4f} +/- {np.std(auprs):.4f}", flush=True)
    print(f"  Total: {total_elapsed:.0f}s", flush=True)

    return fold_data


if __name__ == '__main__':
    os.makedirs('results_final', exist_ok=True)

    # D1
    D1 = os.path.expanduser('~/DrugMiR/data/dataset1')
    print(f"D1 path: {D1}")
    if not os.path.isdir(D1):
        # Try local
        D1 = 'data/processed'
        print(f"  Trying: {D1}")
    fd1 = run_dataset(D1, 'Dataset 1', seed=42)

    # D2 — find path
    D2_candidates = [
        os.path.expanduser('~/work/DrugMiR/DMGAT_processed'),
        os.path.expanduser('~/DrugMiR/data/dataset2'),
        'DMGAT_processed',
    ]
    D2 = None
    for c in D2_candidates:
        if os.path.isdir(c):
            D2 = c
            break
    if D2 is None:
        print("!! D2 path not found in any candidate, skipping")
    else:
        print(f"\nD2 path: {D2}")
        fd2 = run_dataset(D2, 'Dataset 2', seed=42)

    # Save
    out_d1 = 'results_final/drugmir_d1_predictions_seed42.json'
    with open(out_d1, 'w') as f:
        json.dump(fd1, f)
    print(f"\nSaved: {out_d1}")
    if D2 is not None:
        out_d2 = 'results_final/drugmir_d2_predictions_seed42.json'
        with open(out_d2, 'w') as f:
            json.dump(fd2, f)
        print(f"Saved: {out_d2}")
    print("\nDone.")
