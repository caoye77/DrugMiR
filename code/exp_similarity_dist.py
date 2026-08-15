#!/usr/bin/env python3
"""
DrugMiR --- pairwise similarity distributions for the KNN graphs.

WHY THIS EXISTS
---------------
Section II-B builds the miRNA similarity graph from cosine similarity between
4-mer frequency vectors and the drug graph from Tanimoto on Morgan
fingerprints, then keeps the k_m = 15 and k_d = 10 nearest neighbours. Nothing
in the paper shows what those similarity distributions look like.

That is an open flank. miRNAs are ~22 nt sequences over a four-letter alphabet,
so their 4-mer frequency vectors are a priori similar to one another, and a
reviewer is entitled to ask whether the KNN graph carries real structure or is
close to arbitrary. ColdstartMHDTI pre-empts exactly this question with a
similarity-distribution figure placed in its dataset section, and reports that
over 99% of its pairs fall below 0.5.

This script produces the same evidence for DrugMiR. It needs no model and no
checkpoint -- only the processed feature matrices -- so it runs anywhere in
about a minute.

USAGE
-----
    python exp_similarity_dist.py --root ~/work/DrugMiR
    python exp_similarity_dist.py --root ~/work/DrugMiR --dataset D2

OUTPUT
------
    results_similarity/similarity_stats.json
    results_similarity/fig_similarity.pdf / .png     -> new manuscript figure
    results_similarity/caption.txt                   -> drafted caption

WHAT TO DO WITH THE NUMBERS
---------------------------
The script prints the quantile at which the KNN cut-off sits. Two outcomes:

  - cut-off sits far out in the tail  -> the graph keeps genuinely close
    neighbours. Report it and the flank is closed.
  - cut-off sits near the bulk        -> the neighbours are not especially
    similar. Say so plainly and note that the HomoGCN channel is a supporting
    component whose ablation costs ~0.1 AUC points anyway (Table VI). An honest
    weak result here is much safer than leaving the question unasked.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix',
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 9,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def cosine_matrix(F):
    F = np.asarray(F, dtype=np.float64)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = F / n
    return U @ U.T


def tanimoto_matrix(F):
    """Jaccard/Tanimoto on binary fingerprints."""
    B = (np.asarray(F) > 0).astype(np.float64)
    inter = B @ B.T
    cnt = B.sum(1)
    union = cnt[:, None] + cnt[None, :] - inter
    union[union == 0] = 1.0
    return inter / union


def offdiag(M):
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    return M[iu]


def summarize(vals, k, name):
    vals = np.asarray(vals, dtype=np.float64)
    q = {f'q{int(p * 100)}': float(np.quantile(vals, p))
         for p in (0.5, 0.75, 0.9, 0.95, 0.99)}
    below = {f'frac_below_{t}': float((vals < t).mean()) for t in (0.3, 0.5, 0.7, 0.9)}
    return {'name': name, 'n_pairs': int(vals.size), 'k_neighbours': k,
            'mean': float(vals.mean()), 'median': float(np.median(vals)),
            'max': float(vals.max()), **q, **below}


def knn_cutoffs(M, k):
    """For each row, the similarity of its k-th nearest neighbour."""
    M = M.copy()
    np.fill_diagonal(M, -np.inf)
    part = np.partition(M, -k, axis=1)[:, -k]
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/work/DrugMiR'))
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--km', type=int, default=15)
    ap.add_argument('--kd', type=int, default=10)
    ap.add_argument('--dataset', default='D1', help='label only, for filenames')
    a = ap.parse_args()

    root = os.path.expanduser(a.root)
    data_dir = a.data_dir or os.path.join(root, 'data/processed')
    out_dir = a.out_dir or os.path.join(root, 'results_similarity')
    os.makedirs(out_dir, exist_ok=True)

    # --- load features without importing torch if we can avoid it ---
    fm_p = os.path.join(data_dir, 'mirna_feat.npy')
    fd_p = os.path.join(data_dir, 'drug_feat.npy')
    if os.path.exists(fm_p) and os.path.exists(fd_p):
        Fm, Fd = np.load(fm_p), np.load(fd_p)
        print(f'  loaded {fm_p} and {fd_p}')
    else:
        print('  feature .npy not found, falling back to load_data()')
        import exp_compat  # noqa: F401
        from hp_finetune import load_data
        d = load_data(data_dir, km=a.km, kd=a.kd)
        Fm = np.asarray(d['mirna_feat'].cpu() if hasattr(d['mirna_feat'], 'cpu')
                        else d['mirna_feat'])
        Fd = np.asarray(d['drug_feat'].cpu() if hasattr(d['drug_feat'], 'cpu')
                        else d['drug_feat'])

    print(f'  Fm {Fm.shape}   Fd {Fd.shape}')
    if Fm.shape[0] > 6000:
        print('  [warn] large miRNA count: the dense similarity matrix may be '
              'memory heavy. Subsample if this fails.')

    Sm = cosine_matrix(Fm)
    Sd = tanimoto_matrix(Fd)
    vm, vd = offdiag(Sm), offdiag(Sd)

    cm = knn_cutoffs(Sm, a.km)
    cd = knn_cutoffs(Sd, a.kd)

    stats = {
        'dataset': a.dataset,
        'mirna': summarize(vm, a.km, 'miRNA 4-mer cosine'),
        'drug': summarize(vd, a.kd, 'drug Morgan Tanimoto'),
        'knn_cutoff': {
            'mirna_kth_neighbour_similarity': {
                'mean': float(cm.mean()), 'median': float(np.median(cm)),
                'min': float(cm.min()), 'max': float(cm.max())},
            'drug_kth_neighbour_similarity': {
                'mean': float(cd.mean()), 'median': float(np.median(cd)),
                'min': float(cd.min()), 'max': float(cd.max())},
        },
    }
    # where does the typical KNN cut-off sit in the overall distribution?
    stats['knn_cutoff']['mirna_cutoff_quantile'] = float((vm < np.median(cm)).mean())
    stats['knn_cutoff']['drug_cutoff_quantile'] = float((vd < np.median(cd)).mean())

    print('\n  miRNA 4-mer cosine:')
    print(f'    median {stats["mirna"]["median"]:.4f}   '
          f'q95 {stats["mirna"]["q95"]:.4f}   '
          f'below 0.5: {100 * stats["mirna"]["frac_below_0.5"]:.2f}%')
    print(f'    k={a.km} cut-off sits at the '
          f'{100 * stats["knn_cutoff"]["mirna_cutoff_quantile"]:.2f}th percentile')
    print('  drug Morgan Tanimoto:')
    print(f'    median {stats["drug"]["median"]:.4f}   '
          f'q95 {stats["drug"]["q95"]:.4f}   '
          f'below 0.5: {100 * stats["drug"]["frac_below_0.5"]:.2f}%')
    print(f'    k={a.kd} cut-off sits at the '
          f'{100 * stats["knn_cutoff"]["drug_cutoff_quantile"]:.2f}th percentile')

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for A, v, c, lab, k, kk in [
            (ax[0], vm, np.median(cm), 'miRNA 4-mer cosine similarity', a.km, 'k_m'),
            (ax[1], vd, np.median(cd), 'Drug Morgan Tanimoto similarity', a.kd, 'k_d')]:
        A.hist(v, bins=80, log=True, color='#4878A8', edgecolor='none')
        A.axvline(c, color='#C0392B', linestyle='--', linewidth=1.2,
                  label=f'median ${kk}={k}$ cut-off = {c:.2f}')
        A.set_xlabel(lab)
        A.set_ylabel('Pair count (log scale)')
        A.legend(frameon=False, loc='upper right')
        A.set_xlim(0, 1)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'fig_similarity.{ext}'))

    with open(os.path.join(out_dir, 'similarity_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    cap = (
        "Pairwise similarity distributions underlying the KNN graphs on "
        f"Dataset~{a.dataset[-1]}. Left: cosine similarity between miRNA 4-mer "
        "frequency vectors. Right: Tanimoto similarity between drug Morgan "
        "fingerprints. Dashed lines mark the median similarity of the "
        f"$k_m={a.km}$-th and $k_d={a.kd}$-th nearest neighbour, i.e.\\ the "
        "effective edge-inclusion threshold. "
        f"{100 * stats['mirna']['frac_below_0.5']:.1f}\\% of miRNA pairs and "
        f"{100 * stats['drug']['frac_below_0.5']:.1f}\\% of drug pairs fall "
        "below 0.5, so the similarity graphs connect a sparse, comparatively "
        "similar minority rather than reproducing a near-complete graph."
    )
    with open(os.path.join(out_dir, 'caption.txt'), 'w') as f:
        f.write(cap + '\n')

    print(f'\n  written to {out_dir}/')
    print('  caption.txt is drafted from the actual numbers -- re-read the last')
    print('  clause once you see them; if the cut-off turns out to sit inside')
    print('  the bulk, rewrite that clause rather than keeping the optimistic one.')


if __name__ == '__main__':
    main()
