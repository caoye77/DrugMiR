#!/usr/bin/env python
"""
DrugMiR - Experiment (1): masking / stability sweeps.

The supervisor's point was that contribution 1 (adaptive feature-embedding
fusion, "handles missing annotation") has no experiment behind it. This script
provides both readings of that request.

--exp feature   [THE ONE THAT ACTUALLY SUPPORTS CONTRIBUTION 1]
    Artificially strip the biological feature vector from 10/20/30/40% of the
    entities that currently have one, and watch what happens. Three curves:

        full       gated fusion + embedding fallback  -> should degrade slowly
        feat_only  feature path only, NO fallback     -> should collapse
        emb_only   embedding only (== no_hybrid)      -> flat, lower ceiling

    The gap between `full` and `feat_only` IS contribution 1. One figure, and
    the claim stops being rhetorical.

--exp label     [THE LITERAL REQUEST: "mask the label 10/20/30/40%"]
    Relabel that fraction of training-fold positives as negatives, simulating
    database under-annotation. This extends the sweep already in
    scripts_gpu/03_robustness_convergence.py (0/5/10/20/30%, 3-fold) to 40%
    AND to the main text's 5-fold protocol, so it can be promoted out of the
    supplementary material.

Masking notes
-------------
* Only entities that CURRENTLY have features are candidates, so `ratio` means
  "additional annotation loss on top of what the database is already missing".
* Masking zeroes the feature row, and has_feat is recomputed the same way
  load_data derives it -- i.e. exactly the state a genuinely unannotated
  entity would be in.
* The masked set is fixed by --mask_seed, so all three model modes at a given
  ratio see the identical set of crippled entities (paired comparison).

Usage
-----
    python exp_mask.py --dataset D1 --exp feature --smoke     # CPU crash check
    python exp_mask.py --dataset D1 --exp feature
    python exp_mask.py --dataset D2 --exp feature
    python exp_mask.py --dataset D1 --exp label

Output
------
    results_mask/mask_results.csv
    results_mask/mask_<ds>_<exp>.json
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_compat  # noqa: F401,E402
from hp_finetune import load_data, device                      # noqa: E402
from exp_fusion import DrugMiR_Fusion, run_cv                  # noqa: E402

DATA_DIRS = {
    'D1': os.path.expanduser('~/work/DrugMiR/data/processed'),
    'D2': os.path.expanduser('~/work/DrugMiR/DMGAT_processed'),
}
OUT_DIR = os.path.expanduser('~/work/DrugMiR/results_mask')

RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]
FEATURE_MODES = ['full', 'feat_only', 'emb_only']
MODE_TO_FT = {'full': 'gated', 'feat_only': 'feat_only', 'emb_only': 'emb_only'}


def apply_feature_mask(data, ratio, mask_seed=0):
    """Return a shallow copy of `data` with `ratio` of the feature-bearing
    entities stripped of their features (per side, miRNA and drug)."""
    if ratio <= 0:
        return data, {'mirna_masked': 0, 'drug_masked': 0}
    rng = np.random.RandomState(mask_seed)
    out = dict(data)
    counts = {}
    for side, fkey, hkey in (('mirna', 'mirna_feat', 'mirna_has_feat'),
                             ('drug', 'drug_feat', 'drug_has_feat')):
        f = data[fkey].clone()
        idx = torch.nonzero(data[hkey] > 0).squeeze(-1).cpu().numpy()
        n_drop = int(round(ratio * len(idx)))
        if n_drop > 0:
            drop = rng.choice(idx, n_drop, replace=False)
            f[torch.LongTensor(np.sort(drop)).to(f.device)] = 0.0
        out[fkey] = f
        out[hkey] = (f.sum(1) > 0).float()      # same derivation as load_data
        counts[f'{side}_masked'] = int(n_drop)
        counts[f'{side}_with_feat_after'] = int(out[hkey].sum().item())
    return out, counts


CSV_COLS = ['dataset', 'exp', 'mode', 'ratio', 'auc_mean', 'auc_std',
            'aupr_mean', 'aupr_std', 'n_folds', 'seed', 'mask_seed',
            'lr', 'epochs', 'time_s', 'note']


def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['D1', 'D2'])
    ap.add_argument('--exp', required=True, choices=['feature', 'label'])
    ap.add_argument('--modes', default=None,
                    help='comma list, feature exp only (default: all three)')
    ap.add_argument('--ratios', default=None,
                    help='comma list, e.g. 0,0.1,0.2,0.3,0.4')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mask_seed', type=int, default=0)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=2e-4)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--km', type=int, default=15)
    ap.add_argument('--kd', type=int, default=10)
    ap.add_argument('--h', type=int, default=256)
    ap.add_argument('--dr', type=float, default=0.5)
    ap.add_argument('--n_gcn', type=int, default=2)
    ap.add_argument('--n_br', type=int, default=2)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out_dir', default=OUT_DIR)
    a = ap.parse_args()

    if a.smoke:
        a.folds, a.epochs = 1, 10
        if a.ratios is None:
            a.ratios = '0,0.4'

    ratios = ([float(x) for x in a.ratios.split(',')]
              if a.ratios else list(RATIOS))
    modes = ([m.strip() for m in a.modes.split(',')]
             if a.modes else (FEATURE_MODES if a.exp == 'feature' else ['full']))

    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, 'mask_results.csv')

    print('=' * 68)
    print(f'Masking sweep - {a.exp} - {a.dataset}')
    print(f'  device={device}  folds={a.folds}  seed={a.seed}  lr={a.lr}')
    print(f'  ratios={ratios}  modes={modes}')
    if a.smoke:
        print('  *** SMOKE MODE - numbers meaningless, crash check only')
    print('=' * 68, flush=True)

    base_data = load_data(DATA_DIRS[a.dataset], km=a.km, kd=a.kd)
    md = base_data['mirna_feat'].shape[1]
    dd = base_data['drug_feat'].shape[1]
    nm, nd, ng = (base_data['n_mirna'], base_data['n_drug'],
                  base_data['n_gene'])
    n_mf = int(base_data['mirna_has_feat'].sum().item())
    n_df = int(base_data['drug_has_feat'].sum().item())
    print(f'  miRNAs={nm} ({n_mf} with features)  '
          f'drugs={nd} ({n_df} with features)  genes={ng}')
    print(f'  baseline missing rate: miRNA {(1 - n_mf / nm) * 100:.1f}%  '
          f'drug {(1 - n_df / nd) * 100:.1f}%\n', flush=True)

    results = []
    for ratio in ratios:
        if a.exp == 'feature':
            data, counts = apply_feature_mask(base_data, ratio, a.mask_seed)
            note = (f'masked {counts.get("mirna_masked", 0)}m/'
                    f'{counts.get("drug_masked", 0)}d')
        else:
            data, note = base_data, f'pos_drop={ratio:.2f}'

        for mode in modes:
            ft = MODE_TO_FT.get(mode, 'gated')
            tag = f'{a.exp} r={ratio:.0%} {mode}'
            print(f'--- {tag}  ({note}) ---', flush=True)
            t0 = time.time()

            def mk(ft=ft):
                return DrugMiR_Fusion(nm, nd, md, dd, ng, h=a.h, dr=a.dr,
                                      n_gcn=a.n_gcn, n_br=a.n_br,
                                      ch_fusion='concat', ft_fusion=ft)

            aucs, auprs, _ = run_cv(
                data, mk, seed=a.seed, n_folds=a.folds, lr=a.lr, wd=a.wd,
                epochs=a.epochs,
                pos_drop=(ratio if a.exp == 'label' else 0.0))
            dt = time.time() - t0
            row = dict(dataset=a.dataset, exp=a.exp, mode=mode,
                       ratio=ratio,
                       auc_mean=round(float(np.mean(aucs)), 4),
                       auc_std=round(float(np.std(aucs)), 4),
                       aupr_mean=round(float(np.mean(auprs)), 4),
                       aupr_std=round(float(np.std(auprs)), 4),
                       n_folds=a.folds, seed=a.seed, mask_seed=a.mask_seed,
                       lr=a.lr, epochs=a.epochs, time_s=round(dt), note=note)
            results.append(row)
            append_csv(csv_path, row)
            print(f'  >>> AUC={row["auc_mean"]:.4f}+/-{row["auc_std"]:.4f}  '
                  f'AUPR={row["aupr_mean"]:.4f}  ({dt:.0f}s)\n', flush=True)

    jpath = os.path.join(a.out_dir, f'mask_{a.dataset}_{a.exp}.json')
    json.dump(results, open(jpath, 'w'), indent=2)

    print('=' * 68)
    print(f'SUMMARY - {a.exp} sweep - {a.dataset}   (AUC)')
    print('=' * 68)
    hdr = '  ratio  ' + ''.join(f'{m:>16}' for m in modes)
    print(hdr)
    for ratio in ratios:
        line = f'  {ratio:>5.0%}  '
        for m in modes:
            hit = [r for r in results
                   if r['ratio'] == ratio and r['mode'] == m]
            line += (f'{hit[0]["auc_mean"]:>10.4f}+/-{hit[0]["auc_std"]:.3f}'
                     if hit else f'{"-":>16}')
        print(line)
    if a.exp == 'feature' and len(modes) > 1 and len(ratios) > 1:
        def get(m, r):
            h = [x for x in results if x['mode'] == m and x['ratio'] == r]
            return h[0]['auc_mean'] if h else float('nan')
        lo, hi = ratios[0], ratios[-1]
        print(f'\n  drop from {lo:.0%} to {hi:.0%}:')
        for m in modes:
            print(f'    {m:<12}{(get(m, lo) - get(m, hi)) * 100:>7.2f} pts')
        print('  -> the gap between `full` and `feat_only` is contribution 1.')
    print(f'\n  csv : {csv_path}')
    print(f'  json: {jpath}')


if __name__ == '__main__':
    main()
