#!/usr/bin/env python3
"""
DrugMiR --- evaluation under 1:1, 1:5 and 1:10 positive-to-negative ratios.

WHY THIS EXISTS
---------------
Section III-A-2 already concedes the hole:

    "all unobserved pairs are treated as negatives under the open-world
     assumption, so the reported AUPR reflects a balanced 1:1 regime rather
     than the extreme imbalance of realistic large-scale screening"

Naming a limitation without measuring it invites the reviewer to ask for the
measurement. This script supplies it. Only the evaluation set changes -- the
model is untouched -- so no retraining is needed and the whole thing runs on
CPU.

AUC is invariant in expectation to the negative rate; AUPR is not. The point of
the sweep is to show how far AUPR falls and that the ranking survives.

USAGE
-----
    python exp_imbalanced.py --root ~/work/DrugMiR
    python exp_imbalanced.py --root ~/work/DrugMiR --ratios 1 5 10 20 --n_rep 5

OUTPUT
------
    results_imbalanced/imbalanced_eval.json
    results_imbalanced/imbalanced_eval.csv
    results_imbalanced/tab_imbalanced.tex     ready to paste

HONESTY NOTE -- READ BEFORE PUTTING NUMBERS IN THE PAPER
--------------------------------------------------------
The available checkpoint (drugmir_d1_seed42_for_ig.pt) was trained at lr=1e-3
on an 80/20 positive split, not the 5-fold lr=5e-4 configuration behind
Table II. Its 1:1 AUC will therefore NOT equal 0.9618. Do not present these
rows as if they were Table II rows. Present them the way Supplementary Note S1
presents the false-negative sweep: a self-contained sweep whose 1:1 row is its
own reference point. One sentence in the caption settles it.

The alternative -- retraining at 5e-4 under 5-fold so the 1:1 row matches
Table II exactly -- is a GPU job. Worth it only if a reviewer asks.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
import exp_compat  # noqa: F401,E402
from hp_finetune import load_data, DrugMiR_Hybrid, device  # noqa: E402


@torch.no_grad()
def score_pairs(model, data, pairs, batch=4096):
    mi = torch.LongTensor([p[0] for p in pairs]).to(device)
    di = torch.LongTensor([p[1] for p in pairs]).to(device)
    out = []
    for s in range(0, len(pairs), batch):
        o = model(data, mi[s:s + batch], di[s:s + batch])
        out.append(torch.sigmoid(o).cpu().numpy())
    return np.concatenate(out)


def sample_negatives(assoc, nm, nd, n, rng, exclude):
    neg, seen = [], set(exclude)
    guard = 0
    while len(neg) < n and guard < n * 300:
        guard += 1
        i, j = rng.randint(nm), rng.randint(nd)
        if assoc[i, j] == 0 and (i, j) not in seen:
            seen.add((i, j))
            neg.append((i, j))
    return neg


def f1_at_best(labels, probs):
    best, bt = 0.0, 0.5
    for t in np.unique(np.quantile(probs, np.linspace(0.01, 0.99, 99))):
        f = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return float(best), bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/work/DrugMiR'))
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--ratios', type=int, nargs='+', default=[1, 5, 10],
                    help='negatives per positive')
    ap.add_argument('--n_rep', type=int, default=5,
                    help='independent negative draws per ratio')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    root = os.path.expanduser(a.root)
    data_dir = a.data_dir or os.path.join(root, 'data/processed')
    ckpt_p = a.ckpt or os.path.join(root, 'results_final/drugmir_d1_seed42_for_ig.pt')
    out_dir = a.out_dir or os.path.join(root, 'results_imbalanced')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_p):
        sys.exit(f'ABORT: checkpoint not found at {ckpt_p}\n'
                 f'       run scan_nfs.py first')

    print('=' * 70)
    print('Imbalanced evaluation sweep (D1)')
    print(f'  device = {device}   ratios = {a.ratios}   repeats = {a.n_rep}')
    print('=' * 70, flush=True)
    t0 = time.time()

    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    shapes, cfg = ckpt['shapes'], ckpt['config']

    seed = cfg.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = load_data(data_dir, km=cfg['km'], kd=cfg['kd'])
    all_pos = data['pos_pairs']
    perm = np.random.permutation(len(all_pos))
    split = int(len(all_pos) * 0.8)
    val_pos = [all_pos[i] for i in perm[split:]]
    print(f'  held-out positives = {len(val_pos)}')

    model = DrugMiR_Hybrid(shapes['nm'], shapes['nd'], shapes['md'],
                           shapes['dd'], shapes['ng'], h=cfg['h'],
                           dr=cfg['dr'], n_gcn=cfg['n_gcn'],
                           n_br=cfg['n_br']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    assoc = data['assoc']
    nm, nd = data['n_mirna'], data['n_drug']
    pos_scores = score_pairs(model, data, val_pos)
    print(f'  scored positives once, reused across ratios\n')

    results, rows = {'meta': {'ckpt': ckpt_p, 'config': cfg,
                              'n_pos': len(val_pos),
                              'checkpoint_val_auc': ckpt['best_val_auc'],
                              'n_rep': a.n_rep}}, []

    for r in a.ratios:
        aucs, auprs, f1s = [], [], []
        for rep in range(a.n_rep):
            rng = np.random.RandomState(a.seed + 100 * rep)
            neg = sample_negatives(assoc, nm, nd, len(val_pos) * r, rng,
                                   exclude=set(map(tuple, val_pos)))
            if len(neg) < len(val_pos) * r:
                print(f'    [warn] ratio 1:{r} rep {rep}: '
                      f'{len(neg)}/{len(val_pos) * r} negatives')
            neg_scores = score_pairs(model, data, neg)
            probs = np.concatenate([pos_scores, neg_scores])
            labels = np.concatenate([np.ones(len(val_pos)), np.zeros(len(neg))])
            aucs.append(roc_auc_score(labels, probs))
            auprs.append(average_precision_score(labels, probs))
            f1s.append(f1_at_best(labels, probs)[0])
        e = {'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
             'aupr_mean': float(np.mean(auprs)), 'aupr_std': float(np.std(auprs)),
             'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
             'n_neg': int(len(val_pos) * r),
             'prevalence': float(1.0 / (1 + r))}
        results[f'1:{r}'] = e
        rows.append((r, e))
        print(f'  1:{r:<3}  AUC {e["auc_mean"]:.4f}+/-{e["auc_std"]:.4f}   '
              f'AUPR {e["aupr_mean"]:.4f}+/-{e["aupr_std"]:.4f}   '
              f'F1 {e["f1_mean"]:.4f}   (baseline AUPR = prevalence '
              f'{e["prevalence"]:.3f})')

    with open(os.path.join(out_dir, 'imbalanced_eval.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, 'imbalanced_eval.csv'), 'w') as f:
        f.write('ratio,auc_mean,auc_std,aupr_mean,aupr_std,f1_mean,f1_std,prevalence\n')
        for r, e in rows:
            f.write(f'1:{r},{e["auc_mean"]:.4f},{e["auc_std"]:.4f},'
                    f'{e["aupr_mean"]:.4f},{e["aupr_std"]:.4f},'
                    f'{e["f1_mean"]:.4f},{e["f1_std"]:.4f},{e["prevalence"]:.4f}\n')

    # ---- paste-ready LaTeX ----
    tex = [r'\begin{table}[htbp]', r'\centering', r'\footnotesize',
           r'\caption{Effect of the evaluation negative rate on Dataset~1. '
           r'Only the evaluation set changes; the model is unchanged. '
           r'AUC is stable across ratios while AUPR falls toward the positive '
           r'prevalence, which is the expected behaviour and the reason the '
           r'1:1 AUPR in Table~\ref{tab:baseline} should be read as a '
           r'ranking statistic rather than a screening estimate. '
           r'Mean $\pm$ SD over ' + str(a.n_rep) + r' independent negative draws.}',
           r'\label{tab:imbalanced}', r'\begin{tabular}{lccc c}', r'\toprule',
           r'Pos:Neg & AUC & AUPR & F1 & Prevalence \\', r'\midrule']
    for r, e in rows:
        tex.append(f'1:{r} & {e["auc_mean"]:.4f}$\\pm${e["auc_std"]:.4f} & '
                   f'{e["aupr_mean"]:.4f}$\\pm${e["aupr_std"]:.4f} & '
                   f'{e["f1_mean"]:.4f} & {e["prevalence"]:.3f} \\\\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(out_dir, 'tab_imbalanced.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')

    print(f'\n  written to {out_dir}/  ({time.time() - t0:.0f}s)')
    print('  tab_imbalanced.tex is paste-ready; check the caption wording once.')
    print('\n  REMINDER: this checkpoint is lr=1e-3 / 80-20, not the Table II')
    print('  configuration. State that in the caption, the way Supplementary')
    print('  Note S1 states it for the false-negative sweep.')


if __name__ == '__main__':
    main()
