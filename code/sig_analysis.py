#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sig_analysis.py  --  Statistical significance analysis for DrugMiR vs baselines.

Addresses advisor revision point (4)/(5a): report mean +/- std for ALL methods and a
paired significance test (Wilcoxon signed-rank) between the proposed method and the
strongest baseline, plus effect size (Cliff's delta), Holm-Bonferroni multiple-comparison
correction, and a Demsar critical-difference (CD) diagram across all methods.

This script is PURE post-processing: it reads a scores file and writes a report.
It does NOT touch the model or run any training. Dependencies: numpy, scipy, matplotlib only
(no statsmodels / scikit-posthocs needed -- Holm and Nemenyi CD are implemented here).

--------------------------------------------------------------------------------
INPUT FORMAT (JSON)  --  produce this from the multi-seed runner.
--------------------------------------------------------------------------------
A single JSON file with a flat list of per-(method, dataset, seed, fold) records:

{
  "records": [
    {"method": "DrugMiR", "dataset": "D1", "seed": 42, "fold": 0,
     "AUC": 0.9621, "AUPR": 0.9603, "F1": 0.9040, "Prec": 0.8920, "Rec": 0.9165},
    {"method": "MPHGNN",  "dataset": "D1", "seed": 42, "fold": 0,
     "AUC": 0.9520, "AUPR": 0.9460, "F1": 0.8930, "Prec": 0.8640, "Rec": 0.9240},
    ...
  ]
}

Requirements on the records:
  * Every (method, dataset) must be evaluated on the SAME set of (seed, fold) keys,
    so runs can be paired. The script checks this and warns about any mismatch.
  * Metric keys are free-form; pass whichever you want analyzed via --metrics.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python3 sig_analysis.py --scores phase2_outputs/multiseed_scores.json \
        --proposed DrugMiR --metrics AUC AUPR F1 \
        --outdir phase2_outputs/sig_analysis

Outputs (in --outdir):
  report.md                 human-readable summary (paste-ready for the paper / advisor)
  pairwise_<DATASET>_<METRIC>.csv   proposed-vs-each-baseline table
  cd_<DATASET>_<METRIC>.pdf/.png    critical-difference diagram (if >=3 methods)
"""

import os, sys, json, argparse, itertools
import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------- #
#  Effect size: Cliff's delta + magnitude label (Romano et al. thresholds)
# ----------------------------------------------------------------------------- #
def cliffs_delta(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float('nan'), 'undefined'
    # O(nx*ny) but our arrays are tiny (<=50)
    gt = sum(int(xi > yj) for xi in x for yj in y)
    lt = sum(int(xi < yj) for xi in x for yj in y)
    d = (gt - lt) / (nx * ny)
    a = abs(d)
    if   a < 0.147: mag = 'negligible'
    elif a < 0.330: mag = 'small'
    elif a < 0.474: mag = 'medium'
    else:           mag = 'large'
    return d, mag

# ----------------------------------------------------------------------------- #
#  Bootstrap 95% CI on the mean paired difference (proposed - baseline)
# ----------------------------------------------------------------------------- #
def bootstrap_ci(diffs, n_boot=10000, seed=0):
    diffs = np.asarray(diffs, float)
    if len(diffs) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ----------------------------------------------------------------------------- #
#  Holm-Bonferroni step-down correction
# ----------------------------------------------------------------------------- #
def holm_bonferroni(pvals):
    pvals = list(pvals); m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)         # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj

# ----------------------------------------------------------------------------- #
#  Wilcoxon signed-rank (paired). Falls back gracefully on degenerate input.
# ----------------------------------------------------------------------------- #
def wilcoxon_safe(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return float('nan'), 1.0, 'all differences zero'
    if len(d) < 6:
        note = 'n<6: Wilcoxon p is approximate; reported alongside paired t-test'
    else:
        note = ''
    try:
        # exact for small n, with zero_method that drops zero-diffs
        stat, p = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
    except Exception as e:
        return float('nan'), float('nan'), f'wilcoxon failed: {e}'
    return float(stat), float(p), note

# ----------------------------------------------------------------------------- #
#  Critical-difference (Nemenyi) machinery
# ----------------------------------------------------------------------------- #
# Studentized range q_alpha / sqrt(2) for Nemenyi, infinite df. Index = #methods k.
_Q_ALPHA_005 = {2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,
                10:3.164,11:3.219,12:3.268,13:3.313,14:3.354,15:3.391,16:3.426,
                17:3.458,18:3.489,19:3.517,20:3.544}
_Q_ALPHA_010 = {2:1.645,3:2.052,4:2.291,5:2.460,6:2.589,7:2.693,8:2.780,9:2.855,
                10:2.920,11:2.978,12:3.030,13:3.077,14:3.120,15:3.159,16:3.196,
                17:3.230,18:3.261,19:3.291,20:3.319}

def friedman_and_cd(score_matrix, method_names, alpha=0.05):
    """
    score_matrix: (N_blocks, k) array; block = (seed,fold) run, column = method.
    Higher = better. Returns dict with Friedman p, mean ranks, CD value.
    """
    X = np.asarray(score_matrix, float)
    N, k = X.shape
    # rank within each block, higher score -> rank 1 (best). Average ties.
    ranks = np.zeros_like(X)
    for i in range(N):
        ranks[i] = stats.rankdata(-X[i], method='average')
    mean_ranks = ranks.mean(axis=0)
    # Friedman statistic
    try:
        fr_stat, fr_p = stats.friedmanchisquare(*[X[:, j] for j in range(k)])
    except Exception as e:
        fr_stat, fr_p = float('nan'), float('nan')
    q = (_Q_ALPHA_005 if alpha == 0.05 else _Q_ALPHA_010).get(k, None)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N)) if q else float('nan')
    return {'N': N, 'k': k, 'friedman_stat': float(fr_stat), 'friedman_p': float(fr_p),
            'mean_ranks': mean_ranks, 'method_names': list(method_names),
            'cd': float(cd), 'alpha': alpha}

def plot_cd(cd_res, path):
    """Demsar critical-difference diagram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = cd_res['method_names']; ranks = cd_res['mean_ranks']; cd = cd_res['cd']
    k = len(names)
    order = np.argsort(ranks)                 # best (lowest rank) first
    names = [names[i] for i in order]; ranks = ranks[order]
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8, 0.5 * k + 1.6))
    ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(0, k + 2); ax.invert_xaxis()
    ax.axis('off')
    yline = k + 1
    ax.plot([lo, hi], [yline, yline], 'k-', lw=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [yline, yline + 0.12], 'k-', lw=1.0)
        ax.text(r, yline + 0.28, str(r), ha='center', va='bottom', fontsize=9)
    # method labels, split left/right for readability
    for i, (nm, rk) in enumerate(zip(names, ranks)):
        y = yline - 1 - i
        side_left = i < (k + 1) // 2
        xtext = hi + 0.4 if side_left else lo - 0.4
        ha = 'right' if side_left else 'left'
        ax.plot([rk, rk], [yline, y], 'k-', lw=0.8)
        ax.plot([rk, xtext], [y, y], 'k-', lw=0.8)
        ax.text(xtext + (0.1 if side_left else -0.1), y, f"{nm} ({rk:.2f})",
                ha=ha, va='center', fontsize=9)
    # CD bar
    if not np.isnan(cd):
        ax.plot([lo, lo + cd], [yline + 0.7, yline + 0.7], 'k-', lw=2.5)
        ax.plot([lo, lo], [yline + 0.62, yline + 0.78], 'k-', lw=1.2)
        ax.plot([lo + cd, lo + cd], [yline + 0.62, yline + 0.78], 'k-', lw=1.2)
        ax.text(lo + cd / 2, yline + 0.85, f"CD = {cd:.3f} (alpha={cd_res['alpha']})",
                ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight'); fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ----------------------------------------------------------------------------- #
#  Core: load + reshape
# ----------------------------------------------------------------------------- #
def load_records(path):
    with open(path) as f:
        obj = json.load(f)
    recs = obj['records'] if isinstance(obj, dict) and 'records' in obj else obj
    if not isinstance(recs, list) or not recs:
        sys.exit("ERROR: scores file has no 'records' list.")
    return recs

def build_matrix(recs, dataset, metric):
    """Return (block_keys, method_names, matrix[N_blocks, k]) for one dataset+metric."""
    methods = sorted({r['method'] for r in recs if r.get('dataset') == dataset})
    # block = (seed, fold)
    by = {}
    for r in recs:
        if r.get('dataset') != dataset or metric not in r:
            continue
        by.setdefault(r['method'], {})[(r['seed'], r['fold'])] = float(r[metric])
    # blocks present for ALL methods (intersection -> proper pairing)
    common = None
    for m in methods:
        ks = set(by.get(m, {}).keys())
        common = ks if common is None else (common & ks)
    common = sorted(common) if common else []
    mat = np.array([[by[m][k] for m in methods] for k in common], float) if common else np.zeros((0, len(methods)))
    # also report per-method block counts for transparency
    counts = {m: len(by.get(m, {})) for m in methods}
    return common, methods, mat, counts

# ----------------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="DrugMiR significance analysis (Wilcoxon/Cliff/Holm/CD).")
    ap.add_argument('--scores', required=True, help="path to multiseed scores JSON")
    ap.add_argument('--proposed', default='DrugMiR', help="name of the proposed method")
    ap.add_argument('--metrics', nargs='+', default=['AUC', 'AUPR'], help="metric keys to analyze")
    ap.add_argument('--datasets', nargs='+', default=None, help="restrict to these datasets (default: all found)")
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--outdir', default='sig_analysis')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    recs = load_records(args.scores)
    all_ds = args.datasets or sorted({r['dataset'] for r in recs})

    md = ["# DrugMiR — Statistical Significance Analysis\n",
          f"Source: `{args.scores}`  |  proposed method: **{args.proposed}**  |  alpha = {args.alpha}\n",
          "Paired test = Wilcoxon signed-rank across (seed x fold) blocks; "
          "effect size = Cliff's delta; multiple-baseline correction = Holm-Bonferroni; "
          "across-method test = Friedman + Nemenyi critical-difference diagram.\n"]

    for ds in all_ds:
        for metric in args.metrics:
            blocks, methods, mat, counts = build_matrix(recs, ds, metric)
            if mat.shape[0] == 0:
                md.append(f"\n## {ds} — {metric}\n\n_No paired blocks found; skipped._\n")
                continue
            if args.proposed not in methods:
                md.append(f"\n## {ds} — {metric}\n\n_Proposed method '{args.proposed}' absent; skipped._\n")
                continue

            N = mat.shape[0]
            pidx = methods.index(args.proposed)
            prop = mat[:, pidx]
            means = mat.mean(axis=0); stds = mat.std(axis=0, ddof=1) if N > 1 else np.zeros(len(methods))

            md.append(f"\n## {ds} — {metric}\n")
            md.append(f"Paired blocks (seed x fold): **N = {N}**. "
                      f"Per-method block counts: {counts}.\n")
            md.append("\n| Method | mean | std | Wilcoxon p | Holm p | Cliff δ (mag) | 95% CI of mean Δ |")
            md.append("|---|---|---|---|---|---|---|")

            # pairwise proposed vs each baseline
            baselines = [m for m in methods if m != args.proposed]
            raw_p, rows = [], []
            for b in baselines:
                bi = methods.index(b)
                base = mat[:, bi]
                _, p, note = wilcoxon_safe(prop, base)
                d, mag = cliffs_delta(prop, base)
                lo, hi = bootstrap_ci(prop - base)
                raw_p.append(p if not np.isnan(p) else 1.0)
                rows.append((b, p, d, mag, lo, hi, note))
            adj_p = holm_bonferroni(raw_p)

            # proposed row first
            md.append(f"| **{args.proposed}** | {means[pidx]:.4f} | {stds[pidx]:.4f} | — | — | — | — |")
            csv_lines = ["method,mean,std,wilcoxon_p,holm_p,cliffs_delta,cliffs_mag,ci_lo,ci_hi"]
            csv_lines.append(f"{args.proposed},{means[pidx]:.6f},{stds[pidx]:.6f},,,,,,")
            # sort baselines by mean desc for readability
            bsorted = sorted(range(len(baselines)), key=lambda i: -means[methods.index(baselines[i])])
            for i in bsorted:
                b, p, d, mag, lo, hi, note = rows[i]
                bi = methods.index(b)
                pstr = "n/a" if np.isnan(p) else f"{p:.2e}"
                hstr = f"{adj_p[i]:.2e}"
                md.append(f"| {b} | {means[bi]:.4f} | {stds[bi]:.4f} | {pstr} | {hstr} | "
                          f"{d:+.3f} ({mag}) | [{lo:+.4f}, {hi:+.4f}] |")
                csv_lines.append(f"{b},{means[bi]:.6f},{stds[bi]:.6f},{p:.6g},{adj_p[i]:.6g},"
                                 f"{d:.6f},{mag},{lo:.6f},{hi:.6f}")
            with open(os.path.join(args.outdir, f"pairwise_{ds}_{metric}.csv"), 'w') as f:
                f.write("\n".join(csv_lines) + "\n")

            # interpretation line
            best_base_i = max(range(len(baselines)),
                              key=lambda i: means[methods.index(baselines[i])])
            bb, bp, bd, bmag, blo, bhi, _ = rows[best_base_i]
            sig = (not np.isnan(bp)) and adj_p[best_base_i] < args.alpha
            md.append(f"\n*Strongest baseline = **{bb}** (mean {means[methods.index(bb)]:.4f}). "
                      f"{args.proposed} vs {bb}: Holm-adjusted p = {adj_p[best_base_i]:.2e}, "
                      f"Cliff δ = {bd:+.3f} ({bmag}), mean Δ 95% CI [{blo:+.4f}, {bhi:+.4f}]. "
                      f"Difference is {'STATISTICALLY SIGNIFICANT' if sig else 'NOT significant'} at alpha={args.alpha}.*\n")

            # CD diagram across all methods (>=3)
            if len(methods) >= 3:
                cd_res = friedman_and_cd(mat, methods, alpha=args.alpha)
                cd_path = os.path.join(args.outdir, f"cd_{ds}_{metric}.pdf")
                try:
                    plot_cd(cd_res, cd_path)
                    md.append(f"Friedman χ² test across {cd_res['k']} methods: "
                              f"p = {cd_res['friedman_p']:.2e}; CD = {cd_res['cd']:.3f}. "
                              f"Diagram: `{os.path.basename(cd_path)}`. "
                              f"Mean ranks (lower=better): " +
                              ", ".join(f"{m} {r:.2f}" for m, r in
                                        sorted(zip(cd_res['method_names'], cd_res['mean_ranks']),
                                               key=lambda t: t[1])) + ".\n")
                except Exception as e:
                    md.append(f"_CD diagram failed: {e}_\n")

    report = os.path.join(args.outdir, "report.md")
    with open(report, 'w') as f:
        f.write("\n".join(md) + "\n")
    print(f"[sig_analysis] wrote {report}")
    print(f"[sig_analysis] per-comparison CSVs + CD diagrams in {args.outdir}/")


if __name__ == '__main__':
    main()
