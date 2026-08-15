#!/usr/bin/env python
"""
DrugMiR - Experiment (2): Fusion strategy comparison.

Answers the supervisor's question "what fusion do you use - concat? sum?" by
comparing BOTH things the paper calls "fusion", in one table:

  Block A -- channel fusion (Fig. 1 panel D, Eq. 13)
      how the three channel outputs (h0, h_homo, h_bridge) are combined into
      z_ij before the MLP head.
      variants: concat (current) | sum | mean | max | attn

  Block B -- feature-embedding fusion (HybridEnc, Eq. 1-3, contribution 1)
      how the biological feature vector and the learnable embedding are
      combined inside Channel 1.
      variants: gated (current) | scalar | concat | sum

`concat` (A) and `gated` (B) ARE the published model, so they double as the
sanity check: they should reproduce the published AUC.

Submodule creation order replicates DrugMiR_Hybrid.__init__ exactly, so the
shared encoder consumes RNG identically to the published runs (same trick as
run_drugmir_variant_coldstart.py).

Usage
-----
    # CPU smoke test first (1 fold, 10 epochs) - just checks it does not crash
    python exp_fusion.py --dataset D1 --block A --smoke

    # full runs
    python exp_fusion.py --dataset D1 --block A
    python exp_fusion.py --dataset D2 --block A
    python exp_fusion.py --dataset D1 --block B
    python exp_fusion.py --dataset D2 --block B

    # single variant (e.g. re-run just one)
    python exp_fusion.py --dataset D1 --block A --only concat

Output
------
    results_fusion/fusion_results.csv   (appended, one row per variant)
    results_fusion/fusion_<ds>_<block>.json
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
import exp_compat  # noqa: F401,E402
from hp_finetune import (load_data, trn, ev, GG, GB, HybridEnc,  # noqa: E402
                         DrugMiR_Hybrid, device)

DATA_DIRS = {
    'D1': os.path.expanduser('~/work/DrugMiR/data/processed'),
    'D2': os.path.expanduser('~/work/DrugMiR/DMGAT_processed'),
}
OUT_DIR = os.path.expanduser('~/work/DrugMiR/results_fusion')

BLOCK_A = ['concat', 'sum', 'mean', 'max', 'attn']
BLOCK_B = ['gated', 'scalar', 'concat', 'sum']


# ============================================================
# Channel 1 with a switchable feature-embedding fusion operator
# ============================================================
class HybridEncF(HybridEnc):
    """HybridEnc with a switchable Channel-1 fusion operator.

    Modes
    -----
    gated      g = sigmoid(W[fh;eh]);  out = g*fh + (1-g)*eh   [CURRENT MODEL]
    scalar     a = sigmoid(alpha);     out = a*fh + (1-a)*eh   (one scalar,
               tests whether *dimension-wise* gating actually matters)
    concat     out = W[fh;eh]            (learned linear mix, no gate)
    sum        out = fh + eh             (no learned combination at all)
    feat_only  out = fh                  (no embedding fallback - used by
               exp_mask.py to show what happens WITHOUT contribution 1)
    emb_only   out = eh                  (== the `no_hybrid` ablation)

    All modes except feat_only/emb_only keep the has_feat fallback, so the
    comparison is not confounded with cold-start behaviour.
    """
    MODES = ('gated', 'scalar', 'concat', 'sum', 'feat_only', 'emb_only')

    def __init__(s, n, fd, h, dr, mode='gated'):
        # creates feat, emb, gate in the original order -> same RNG draw
        super().__init__(n, fd, h, dr)
        assert mode in s.MODES, f'unknown mode {mode}'
        s.mode = mode
        # extra params created AFTER the originals so the shared ones keep
        # their initialisation
        if mode == 'scalar':
            s.alpha = nn.Parameter(torch.zeros(1))
        elif mode == 'concat':
            s.proj = nn.Linear(2 * h, h)

    def forward(s, feat, has_feat=None):
        if s.mode == 'emb_only':
            return s.emb.weight
        fh = s.feat(feat)
        eh = s.emb.weight
        if s.mode == 'feat_only':
            return fh                      # deliberately no fallback
        if s.mode == 'gated':
            g = torch.sigmoid(s.gate(torch.cat([fh, eh], -1)))
            out = g * fh + (1 - g) * eh
        elif s.mode == 'scalar':
            a = torch.sigmoid(s.alpha)
            out = a * fh + (1 - a) * eh
        elif s.mode == 'concat':
            out = s.proj(torch.cat([fh, eh], -1))
        elif s.mode == 'sum':
            out = fh + eh
        else:
            raise ValueError(s.mode)
        if has_feat is not None:
            mask = has_feat.unsqueeze(1)
            out = mask * out + (1 - mask) * eh
        return out


# ============================================================
# Full model with switchable fusion on both levels
# ============================================================
class DrugMiR_Fusion(nn.Module):
    """DrugMiR_Hybrid with ch_fusion / ft_fusion switches.

    ch_fusion='concat' + ft_fusion='gated' == the published DrugMiR_Hybrid.
    """

    def __init__(s, nm, nd, md, dd, ng, h=256, dr=0.5, n_gcn=2, n_br=2,
                 ch_fusion='concat', ft_fusion='gated'):
        super().__init__()
        s.ch_fusion = ch_fusion
        s.ft_fusion = ft_fusion
        s.h = h
        # ---- identical creation order to DrugMiR_Hybrid.__init__ ----
        s.me = HybridEncF(nm, md, h, dr, mode=ft_fusion)
        s.de = HybridEncF(nd, dd, h, dr, mode=ft_fusion)
        s.ge = nn.Embedding(ng, h)
        s.mgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        s.dgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        s.br = nn.ModuleList([GB(h, dr) for _ in range(n_br)])
        per_side = 3 * h if ch_fusion == 'concat' else h
        s.pred = nn.Sequential(
            nn.Linear(2 * per_side, 2 * h), nn.BatchNorm1d(2 * h), nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(2 * h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h, 1))
        # ---- extra params LAST so everything above keeps its RNG ----
        if ch_fusion == 'attn':
            s.attn_m = nn.Linear(h, 1)
            s.attn_d = nn.Linear(h, 1)

    def _fuse(s, chans, attn):
        if s.ch_fusion == 'concat':
            return torch.cat(chans, -1)
        st = torch.stack(chans, 1)                    # [N, 3, h]
        if s.ch_fusion == 'sum':
            return st.sum(1)
        if s.ch_fusion == 'mean':
            return st.mean(1)
        if s.ch_fusion == 'max':
            return st.max(1).values
        if s.ch_fusion == 'attn':
            w = torch.softmax(attn(st).squeeze(-1), -1)      # [N, 3]
            return (st * w.unsqueeze(-1)).sum(1)
        raise ValueError(s.ch_fusion)

    def forward(s, data, mi, di):
        m0 = s.me(data['mirna_feat'], data.get('mirna_has_feat'))
        d0 = s.de(data['drug_feat'], data.get('drug_has_feat'))
        mh = m0
        dh = d0
        for l in s.mgcn:
            mh = l(mh, data['mirna_sim_edge'])
        for l in s.dgcn:
            dh = l(dh, data['drug_sim_edge'])
        mb = m0
        db = d0
        gh = s.ge.weight
        for l in s.br:
            mb, db, gh = l(mb, db, gh, data['mg_src'], data['mg_dst'],
                           data['dg_src'], data['dg_dst'], data['n_gene'])
        zm = s._fuse([m0, mh, mb], getattr(s, 'attn_m', None))[mi]
        zd = s._fuse([d0, dh, db], getattr(s, 'attn_d', None))[di]
        return s.pred(torch.cat([zm, zd], -1)).squeeze(-1)


# ============================================================
# CV driver (mirrors hp_finetune.rcv: seed once, then KFold)
# ============================================================
def run_cv(data, make_model, seed=42, n_folds=5, lr=1e-3, wd=2e-4,
           epochs=200, patience=15, eval_every=5, pos_drop=0.0, verbose=True):
    """5-fold CV at a single seed.

    pos_drop : fraction of TRAINING positives to relabel as negatives
               (used by exp_mask.py for the label-masking sweep). They are
               simply removed from the positive list, so the negative sampler
               may draw them like any other unobserved pair.
    """
    pos = data['pos_pairs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    n_split = n_folds if n_folds >= 2 else 5
    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    aucs, auprs = [], []
    n_params = None
    for fold, (tri, tei) in enumerate(kf.split(pos)):
        if n_folds < 2 and fold >= 1:
            break
        trp = [pos[i] for i in tri]
        tep = [pos[i] for i in tei]
        if pos_drop > 0:
            n_keep = len(trp) - int(round(pos_drop * len(trp)))
            keep = np.random.choice(len(trp), n_keep, replace=False)
            trp = [trp[i] for i in sorted(keep)]
        m = make_model().to(device)
        if n_params is None:
            n_params = sum(p.numel() for p in m.parameters())
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        ba, bp, pc = 0.0, 0.0, 0
        for e in range(epochs):
            trn(m, data, trp, opt)
            if (e + 1) % eval_every == 0:
                a, p = ev(m, data, tep)
                if a > ba:
                    ba, bp, pc = a, p, 0
                else:
                    pc += 1
                if pc >= patience:
                    break
        aucs.append(ba)
        auprs.append(bp)
        if verbose:
            print(f'    fold {fold + 1}/{n_folds}: AUC={ba:.4f} AUPR={bp:.4f}',
                  flush=True)
        del m, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return aucs, auprs, n_params


CSV_COLS = ['dataset', 'block', 'variant', 'auc_mean', 'auc_std',
            'aupr_mean', 'aupr_std', 'n_params', 'n_folds', 'seed',
            'lr', 'epochs', 'time_s']


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
    ap.add_argument('--block', required=True, choices=['A', 'B'])
    ap.add_argument('--only', default=None,
                    help='run a single variant instead of the whole block')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='hp_finetune.rcv default is 1e-3; the paper text says '
                         '5e-4 -- check which reproduces the published AUC')
    ap.add_argument('--wd', type=float, default=2e-4)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--km', type=int, default=15)
    ap.add_argument('--kd', type=int, default=10)
    ap.add_argument('--h', type=int, default=256)
    ap.add_argument('--dr', type=float, default=0.5)
    ap.add_argument('--n_gcn', type=int, default=2)
    ap.add_argument('--n_br', type=int, default=2)
    ap.add_argument('--smoke', action='store_true',
                    help='1 fold, 10 epochs - crash check only')
    ap.add_argument('--out_dir', default=OUT_DIR)
    a = ap.parse_args()

    if a.smoke:
        a.folds, a.epochs = 1, 10

    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, 'fusion_results.csv')

    variants = BLOCK_A if a.block == 'A' else BLOCK_B
    if a.only:
        assert a.only in variants, f'{a.only} not in block {a.block}: {variants}'
        variants = [a.only]

    print('=' * 68)
    print(f'Fusion comparison - Block {a.block} - {a.dataset}')
    print(f'  device={device}  folds={a.folds}  seed={a.seed}  lr={a.lr}  '
          f'epochs={a.epochs}')
    print(f'  variants: {variants}')
    if a.smoke:
        print('  *** SMOKE MODE - numbers are meaningless, only checks it runs')
    print('=' * 68, flush=True)

    data = load_data(DATA_DIRS[a.dataset], km=a.km, kd=a.kd)
    md = data['mirna_feat'].shape[1]
    dd = data['drug_feat'].shape[1]
    nm, nd, ng = data['n_mirna'], data['n_drug'], data['n_gene']
    print(f'  miRNAs={nm} drugs={nd} genes={ng} '
          f'positives={len(data["pos_pairs"])}\n', flush=True)

    results = []
    for v in variants:
        ch = v if a.block == 'A' else 'concat'
        ft = v if a.block == 'B' else 'gated'
        tag = f'{a.block}:{v}'
        print(f'--- {tag}  (ch_fusion={ch}, ft_fusion={ft}) ---', flush=True)
        t0 = time.time()

        def mk(ch=ch, ft=ft):
            return DrugMiR_Fusion(nm, nd, md, dd, ng, h=a.h, dr=a.dr,
                                  n_gcn=a.n_gcn, n_br=a.n_br,
                                  ch_fusion=ch, ft_fusion=ft)

        aucs, auprs, npar = run_cv(data, mk, seed=a.seed, n_folds=a.folds,
                                   lr=a.lr, wd=a.wd, epochs=a.epochs)
        dt = time.time() - t0
        row = dict(dataset=a.dataset, block=a.block, variant=v,
                   auc_mean=round(float(np.mean(aucs)), 4),
                   auc_std=round(float(np.std(aucs)), 4),
                   aupr_mean=round(float(np.mean(auprs)), 4),
                   aupr_std=round(float(np.std(auprs)), 4),
                   n_params=npar, n_folds=a.folds, seed=a.seed,
                   lr=a.lr, epochs=a.epochs, time_s=round(dt))
        results.append(row)
        append_csv(csv_path, row)
        print(f'  >>> {tag}: AUC={row["auc_mean"]:.4f}+/-{row["auc_std"]:.4f}  '
              f'AUPR={row["aupr_mean"]:.4f}  params={npar:,}  ({dt:.0f}s)\n',
              flush=True)

    jpath = os.path.join(a.out_dir, f'fusion_{a.dataset}_{a.block}.json')
    json.dump(results, open(jpath, 'w'), indent=2)

    print('=' * 68)
    print(f'BLOCK {a.block} SUMMARY - {a.dataset}')
    print('=' * 68)
    print(f'  {"variant":<10}{"AUC":>18}{"AUPR":>12}{"params":>14}')
    for r in sorted(results, key=lambda x: -x['auc_mean']):
        print(f'  {r["variant"]:<10}'
              f'{r["auc_mean"]:>10.4f}+/-{r["auc_std"]:.4f}'
              f'{r["aupr_mean"]:>12.4f}{r["n_params"]:>14,}')
    base = 'concat' if a.block == 'A' else 'gated'
    if any(r['variant'] == base for r in results):
        bv = [r for r in results if r['variant'] == base][0]
        print(f'\n  Baseline ({base}) = the published model. If this does not '
              f'match\n  the published AUC, the lr is wrong -- try --lr 5e-4.')
        print(f'  baseline AUC = {bv["auc_mean"]:.4f}')
    print(f'\n  csv : {csv_path}')
    print(f'  json: {jpath}')


if __name__ == '__main__':
    main()
