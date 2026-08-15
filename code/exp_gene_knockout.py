#!/usr/bin/env python3
"""
DrugMiR --- Gene-knockout ablation on the Gene Bridge.

WHY THIS EXISTS
---------------
Section III-J currently supports the interpretability claim with an
IG-vs-GradientSHAP agreement of rho = 0.9999. That number is weaker than it
looks: GradientSHAP perturbs IG's baseline around the same origin, so the two
are one estimator family, and Eq. (17) sums |IG| across all positive pairs,
which averages away per-pair disagreement. A reviewer who knows the
gradient-XAI benchmarking literature will say so.

This script replaces a correlational claim with a causal one. If the genes IG
ranks highest are genes the model actually relies on, then removing them should
hurt more than removing the same number of random genes, and more than removing
the genes a purely topological heuristic would have picked.

    zero the embedding rows of the top-k IG genes    -> measure AUC drop
    zero k random gene rows (several seeds)          -> control A
    zero the top-k genes by degree-product           -> control B

Expected ordering:  IG-topk drop  >  degree-topk drop  >  random-k drop

This is inference only. No retraining. Runs on CPU in a few minutes.

USAGE
-----
    python exp_gene_knockout.py --root ~/work/DrugMiR
    python exp_gene_knockout.py --root ~/work/DrugMiR --k 10 50 100 --n_random 5

OUTPUT
------
    results_knockout/gene_knockout.json     all numbers
    results_knockout/gene_knockout.csv      tidy table for the paper

NOTE ON THE CHECKPOINT
----------------------
drugmir_d1_seed42_for_ig.pt was trained at lr=1e-3 with an 80/20 positive split
(train_save_for_ig.py), so its absolute AUC is NOT the headline 0.9618 of
Table II. That does not matter here: every row in the output is measured on the
same model and the same evaluation set, so the comparison is internally valid.
Report it that way -- the intact-model row, not Table II, is the reference.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
import exp_compat  # noqa: F401,E402
from hp_finetune import load_data, DrugMiR_Hybrid, device  # noqa: E402


class DrugMiR_Attrib(DrugMiR_Hybrid):
    """Identical forward pass with an override hook on the gene embedding.

    Copied from exp_shap.py so this script is standalone.
    """

    def forward(s, data, mi, di, gene_emb_override=None):
        m0 = s.me(data['mirna_feat'], data.get('mirna_has_feat'))
        d0 = s.de(data['drug_feat'], data.get('drug_has_feat'))
        mh, dh = m0, d0
        for l in s.mgcn:
            mh = l(mh, data['mirna_sim_edge'])
        for l in s.dgcn:
            dh = l(dh, data['drug_sim_edge'])
        mb, db = m0, d0
        gh = s.ge.weight if gene_emb_override is None else gene_emb_override
        for l in s.br:
            mb, db, gh = l(mb, db, gh, data['mg_src'], data['mg_dst'],
                           data['dg_src'], data['dg_dst'], data['n_gene'])
        return s.pred(torch.cat([torch.cat([m0, mh, mb], -1)[mi],
                                 torch.cat([d0, dh, db], -1)[di]],
                                -1)).squeeze(-1)


# ---------------------------------------------------------------- evaluation

def build_eval_set(data, val_pos, n_neg_per_pos, seed):
    """Positives = held-out fold; negatives sampled from unobserved pairs."""
    rng = np.random.RandomState(seed)
    assoc = data['assoc']
    nm, nd = data['n_mirna'], data['n_drug']
    need = len(val_pos) * n_neg_per_pos
    neg, seen = [], set()
    guard = 0
    while len(neg) < need and guard < need * 200:
        guard += 1
        i = rng.randint(nm)
        j = rng.randint(nd)
        if assoc[i, j] == 0 and (i, j) not in seen:
            seen.add((i, j))
            neg.append((i, j))
    if len(neg) < need:
        print(f"    [warn] only sampled {len(neg)}/{need} negatives")
    pairs = list(val_pos) + neg
    labels = np.array([1.0] * len(val_pos) + [0.0] * len(neg))
    mi = torch.LongTensor([p[0] for p in pairs]).to(device)
    di = torch.LongTensor([p[1] for p in pairs]).to(device)
    return mi, di, labels


@torch.no_grad()
def score(model, data, mi, di, ge_override, batch=4096):
    out = []
    for s in range(0, mi.numel(), batch):
        o = model(data, mi[s:s + batch], di[s:s + batch],
                  gene_emb_override=ge_override)
        out.append(torch.sigmoid(o).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, data, mi, di, labels, ge_override):
    p = score(model, data, mi, di, ge_override)
    return float(roc_auc_score(labels, p)), float(average_precision_score(labels, p))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/work/DrugMiR'))
    ap.add_argument('--data_dir', default=None,
                    help='default <root>/data/processed')
    ap.add_argument('--ckpt', default=None,
                    help='default <root>/results_final/drugmir_d1_seed42_for_ig.pt')
    ap.add_argument('--ig_npy', default=None,
                    help='default <root>/results_final/gene_ig_importance_d1.npy')
    ap.add_argument('--out_dir', default=None,
                    help='default <root>/results_knockout')
    ap.add_argument('--k', type=int, nargs='+', default=[10, 50, 100],
                    help='how many genes to knock out')
    ap.add_argument('--n_random', type=int, default=5,
                    help='random-control repeats per k')
    ap.add_argument('--n_neg', type=int, default=1,
                    help='negatives per positive in the evaluation set')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    root = os.path.expanduser(a.root)
    data_dir = a.data_dir or os.path.join(root, 'data/processed')
    ckpt_p = a.ckpt or os.path.join(root, 'results_final/drugmir_d1_seed42_for_ig.pt')
    ig_p = a.ig_npy or os.path.join(root, 'results_final/gene_ig_importance_d1.npy')
    out_dir = a.out_dir or os.path.join(root, 'results_knockout')
    os.makedirs(out_dir, exist_ok=True)

    for p, what in [(ckpt_p, 'checkpoint'), (ig_p, 'IG importance vector')]:
        if not os.path.exists(p):
            sys.exit(f"ABORT: {what} not found at {p}\n"
                     f"       run scan_nfs.py first to locate it")

    print('=' * 70)
    print('Gene-knockout ablation on the Gene Bridge (D1)')
    print(f'  device = {device}')
    print('=' * 70, flush=True)
    t0 = time.time()

    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    shapes, cfg = ckpt['shapes'], ckpt['config']
    print(f"  checkpoint best_val_auc = {ckpt['best_val_auc']:.4f}")

    # reproduce train_save_for_ig.py's split EXACTLY:
    #   seed -> load_data -> permutation -> 80/20
    seed = cfg.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = load_data(data_dir, km=cfg['km'], kd=cfg['kd'])
    got = (data['n_mirna'], data['n_drug'], data['n_gene'])
    want = (shapes['nm'], shapes['nd'], shapes['ng'])
    if got != want:
        sys.exit(f'ABORT: data has (nm,nd,ng)={got}, checkpoint expects {want}')
    all_pos = data['pos_pairs']
    perm = np.random.permutation(len(all_pos))
    split = int(len(all_pos) * 0.8)
    val_pos = [all_pos[i] for i in perm[split:]]
    print(f'  held-out positives = {len(val_pos)} of {len(all_pos)}')

    model = DrugMiR_Attrib(shapes['nm'], shapes['nd'], shapes['md'],
                           shapes['dd'], shapes['ng'], h=cfg['h'],
                           dr=cfg['dr'], n_gcn=cfg['n_gcn'],
                           n_br=cfg['n_br']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    mi, di, labels = build_eval_set(data, val_pos, a.n_neg, a.seed)
    print(f'  evaluation pairs   = {len(labels)} '
          f'({int(labels.sum())} pos / {int((1 - labels).sum())} neg)\n')

    ge = model.ge.weight.detach().clone()

    # ---- active bridge genes and the two rankings ----
    ig = np.load(ig_p)
    mg = np.load(os.path.join(data_dir, 'mirna_gene_matrix.npy'))
    dg = np.load(os.path.join(data_dir, 'drug_gene_matrix.npy'))
    n = ge.shape[0]
    m_deg = np.pad(mg.sum(0), (0, max(0, n - mg.shape[1])))[:n]
    d_deg = np.pad(dg.sum(0), (0, max(0, n - dg.shape[1])))[:n]
    ig = np.pad(ig, (0, max(0, n - len(ig))))[:n]
    active = np.where((m_deg > 0) & (d_deg > 0))[0]
    print(f'  active bridge genes = {len(active)}')

    ig_order = active[np.argsort(-ig[active])]
    degprod = m_deg * d_deg
    deg_order = active[np.argsort(-degprod[active])]

    def knock(idx):
        g = ge.clone()
        g[torch.as_tensor(np.asarray(idx), dtype=torch.long, device=g.device)] = 0.0
        return g

    # ---- intact reference ----
    base_auc, base_aupr = evaluate(model, data, mi, di, labels, ge)
    print(f'\n  intact model: AUC = {base_auc:.4f}   AUPR = {base_aupr:.4f}')
    print('  (this row, not Table II, is the reference for every drop below)\n')

    results = {'meta': {'ckpt': ckpt_p, 'checkpoint_val_auc': ckpt['best_val_auc'],
                        'config': cfg, 'n_eval_pairs': int(len(labels)),
                        'n_neg_per_pos': a.n_neg, 'seed': a.seed,
                        'n_active_genes': int(len(active))},
               'intact': {'auc': base_auc, 'aupr': base_aupr},
               'knockout': {}}

    rows = []
    for k in a.k:
        print(f'  --- k = {k} ' + '-' * 46)
        entry = {}

        ig_auc, ig_aupr = evaluate(model, data, mi, di, labels,
                                   knock(ig_order[:k]))
        entry['ig_top'] = {'auc': ig_auc, 'aupr': ig_aupr,
                           'd_auc_pp': 100 * (base_auc - ig_auc),
                           'd_aupr_pp': 100 * (base_aupr - ig_aupr)}
        print(f'    IG top-{k}      AUC {ig_auc:.4f}  '
              f'(-{100 * (base_auc - ig_auc):.2f} pp)')

        dg_auc, dg_aupr = evaluate(model, data, mi, di, labels,
                                   knock(deg_order[:k]))
        entry['degree_top'] = {'auc': dg_auc, 'aupr': dg_aupr,
                               'd_auc_pp': 100 * (base_auc - dg_auc),
                               'd_aupr_pp': 100 * (base_aupr - dg_aupr)}
        print(f'    degree top-{k}  AUC {dg_auc:.4f}  '
              f'(-{100 * (base_auc - dg_auc):.2f} pp)')

        r_aucs, r_auprs = [], []
        for r in range(a.n_random):
            rng = np.random.RandomState(1000 + r)
            pick = rng.choice(active, size=min(k, len(active)), replace=False)
            ra, rp = evaluate(model, data, mi, di, labels, knock(pick))
            r_aucs.append(ra)
            r_auprs.append(rp)
        entry['random'] = {'auc_mean': float(np.mean(r_aucs)),
                           'auc_std': float(np.std(r_aucs)),
                           'aupr_mean': float(np.mean(r_auprs)),
                           'aupr_std': float(np.std(r_auprs)),
                           'd_auc_pp': 100 * (base_auc - float(np.mean(r_aucs))),
                           'n_repeats': a.n_random}
        print(f'    random  k={k}   AUC {np.mean(r_aucs):.4f} '
              f'+/- {np.std(r_aucs):.4f}  '
              f'(-{100 * (base_auc - np.mean(r_aucs)):.2f} pp)')

        ok = (entry['ig_top']['d_auc_pp'] > entry['random']['d_auc_pp'])
        print(f'    -> IG beats random control: {"YES" if ok else "NO"}')
        if not ok:
            print('       [!] the interpretability claim does NOT get causal '
                  'support at this k. Report it honestly or drop this k.')

        results['knockout'][str(k)] = entry
        rows.append((k, ig_auc, dg_auc, float(np.mean(r_aucs)),
                     float(np.std(r_aucs)),
                     entry['ig_top']['d_auc_pp'],
                     entry['degree_top']['d_auc_pp'],
                     entry['random']['d_auc_pp']))
        print()

    with open(os.path.join(out_dir, 'gene_knockout.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, 'gene_knockout.csv'), 'w') as f:
        f.write('k,auc_ig_top,auc_degree_top,auc_random_mean,auc_random_std,'
                'drop_ig_pp,drop_degree_pp,drop_random_pp\n')
        for r in rows:
            f.write(','.join(f'{x:.4f}' if isinstance(x, float) else str(x)
                             for x in r) + '\n')

    print('=' * 70)
    print(f'  intact AUC {base_auc:.4f}  |  written to {out_dir}/')
    print(f'  total {time.time() - t0:.0f}s')
    print('=' * 70)
    print('\n  Read the result as: the ordering across the three rows is the')
    print('  finding, not the absolute values. If IG-topk hurts more than both')
    print('  controls, Section III-J has causal support and the rho=0.9999')
    print('  sentence can stay narrow without weakening the contribution.')


if __name__ == '__main__':
    main()
