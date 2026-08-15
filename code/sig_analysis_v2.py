#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sig_analysis.py (v2)  --  Statistical significance analysis for DrugMiR vs baselines.

Key fix vs v1:  per-pair block intersection.  v1 forced the (seed,fold) block
set to be the intersection of ALL methods, which crashed N down to the
smallest-coverage method (MPHGNN, seed=42 only, 5 blocks).  v2 computes each
proposed-vs-baseline pair's own intersection, so DrugMiR vs MPHGNN uses 5
blocks while DrugMiR vs (other 6 baselines) uses 20-30 blocks each.

Per-method mean/std are now computed over each method's OWN block set (not
the intersection).  Friedman/CD diagram requires strict pairing, so it
auto-drops methods with fewer than --min-friedman-blocks blocks and reports
the actually-used N.

Outputs: report.md, pairwise_{DATASET}_{METRIC}.csv, cd_{DATASET}_{METRIC}.pdf/.png

Dependencies: numpy, scipy, matplotlib.
"""

import os, sys, json, argparse
import numpy as np
from scipy import stats

# ---------- effect size, CI, multiple-comparison correction (unchanged) -------
def cliffs_delta(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float('nan'), 'undefined'
    gt = sum(int(xi > yj) for xi in x for yj in y)
    lt = sum(int(xi < yj) for xi in x for yj in y)
    d = (gt - lt) / (nx * ny)
    a = abs(d)
    if   a < 0.147: mag = 'negligible'
    elif a < 0.330: mag = 'small'
    elif a < 0.474: mag = 'medium'
    else:           mag = 'large'
    return d, mag

def bootstrap_ci(diffs, n_boot=10000, seed=0):
    diffs = np.asarray(diffs, float)
    if len(diffs) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def holm_bonferroni(pvals):
    pvals = list(pvals); m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj

def wilcoxon_safe(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 1 or len(b) < 1 or len(a) != len(b):
        return float('nan'), float('nan')
    d = a - b
    if np.allclose(d, 0):
        return float('nan'), 1.0
    try:
        stat, p = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return float(stat), float(p)
    except Exception:
        return float('nan'), float('nan')

# ---------- Nemenyi CD ---------------------------------------------------------
_Q_ALPHA_005 = {2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,
                10:3.164,11:3.219,12:3.268,13:3.313,14:3.354,15:3.391,16:3.426,
                17:3.458,18:3.489,19:3.517,20:3.544}
_Q_ALPHA_010 = {2:1.645,3:2.052,4:2.291,5:2.460,6:2.589,7:2.693,8:2.780,9:2.855,
                10:2.920,11:2.978,12:3.030,13:3.077,14:3.120,15:3.159,16:3.196,
                17:3.230,18:3.261,19:3.291,20:3.319}

def friedman_and_cd(score_matrix, method_names, alpha=0.05):
    X = np.asarray(score_matrix, float)
    N, k = X.shape
    ranks = np.zeros_like(X)
    for i in range(N):
        ranks[i] = stats.rankdata(-X[i], method='average')
    mean_ranks = ranks.mean(axis=0)
    try:
        fr_stat, fr_p = stats.friedmanchisquare(*[X[:, j] for j in range(k)])
    except Exception:
        fr_stat, fr_p = float('nan'), float('nan')
    q = (_Q_ALPHA_005 if alpha == 0.05 else _Q_ALPHA_010).get(k, None)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N)) if q else float('nan')
    return {'N': N, 'k': k, 'friedman_stat': float(fr_stat), 'friedman_p': float(fr_p),
            'mean_ranks': mean_ranks, 'method_names': list(method_names),
            'cd': float(cd), 'alpha': alpha}

def plot_cd(cd_res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = cd_res['method_names']; ranks = cd_res['mean_ranks']; cd = cd_res['cd']
    k = len(names)
    order = np.argsort(ranks)
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
    for i, (nm, rk) in enumerate(zip(names, ranks)):
        y = yline - 1 - i
        side_left = i < (k + 1) // 2
        xtext = hi + 0.4 if side_left else lo - 0.4
        ha = 'right' if side_left else 'left'
        ax.plot([rk, rk], [yline, y], 'k-', lw=0.8)
        ax.plot([rk, xtext], [y, y], 'k-', lw=0.8)
        ax.text(xtext + (0.1 if side_left else -0.1), y, f"{nm} ({rk:.2f})",
                ha=ha, va='center', fontsize=9)
    if not np.isnan(cd):
        ax.plot([lo, lo + cd], [yline + 0.7, yline + 0.7], 'k-', lw=2.5)
        ax.plot([lo, lo], [yline + 0.62, yline + 0.78], 'k-', lw=1.2)
        ax.plot([lo + cd, lo + cd], [yline + 0.62, yline + 0.78], 'k-', lw=1.2)
        ax.text(lo + cd / 2, yline + 0.85, f"CD = {cd:.3f} (alpha={cd_res['alpha']})",
                ha='center', va='bottom', fontsize=9)
    import matplotlib.pyplot as plt2
    fig.savefig(path, bbox_inches='tight'); fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt2.close(fig)

# ---------- record helpers ----------------------------------------------------
def load_records(path):
    with open(path) as f:
        obj = json.load(f)
    recs = obj['records'] if isinstance(obj, dict) and 'records' in obj else obj
    if not isinstance(recs, list) or not recs:
        sys.exit("ERROR: scores file has no 'records' list.")
    return recs

def build_blocks(recs, dataset, metric):
    """method -> {(seed,fold): score}."""
    by = {}
    for r in recs:
        if r.get('dataset') != dataset or metric not in r:
            continue
        by.setdefault(r['method'], {})[(r['seed'], r['fold'])] = float(r[metric])
    return by


# ---------- main --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--proposed', default='DrugMiR')
    ap.add_argument('--metrics', nargs='+', default=['AUC', 'AUPR'])
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--outdir', default='sig_analysis')
    ap.add_argument('--min-friedman-blocks', type=int, default=10,
                    help="Friedman/CD drops methods with fewer blocks than this; default 10")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    recs = load_records(args.scores)
    all_ds = args.datasets or sorted({r['dataset'] for r in recs})

    md = ["# DrugMiR — Statistical Significance Analysis (v2)\n",
          f"Source: `{args.scores}`  |  proposed method: **{args.proposed}**  |  alpha = {args.alpha}\n",
          "Each pairwise test uses the intersection of (seed x fold) blocks for "
          "that pair, so methods with fewer seeds do not reduce N for other "
          "pairs. Per-method mean / std use each method's own full block set. "
          "Friedman / Nemenyi CD requires strict pairing across all methods, "
          f"so methods with fewer than --min-friedman-blocks={args.min_friedman_blocks} "
          "blocks are excluded from that test (still shown in the per-pair table).\n"]

    for ds in all_ds:
        for metric in args.metrics:
            by = build_blocks(recs, ds, metric)
            if args.proposed not in by:
                md.append(f"\n## {ds} — {metric}\n\n_Proposed method absent; skipped._\n")
                continue

            methods = sorted(by.keys())
            counts = {m: len(by[m]) for m in methods}
            # method-level mean/std on each method's OWN block set
            means = {m: float(np.mean(list(by[m].values()))) for m in methods}
            stds  = {m: float(np.std(list(by[m].values()), ddof=1)) if counts[m] > 1 else 0.0
                     for m in methods}

            md.append(f"\n## {ds} — {metric}\n")
            md.append(f"Per-method block count (each method's own seed x fold runs): {counts}.\n")
            md.append("\n| Method | mean | std | N_paired | Wilcoxon p | Holm p | Cliff δ (mag) | 95% CI of mean Δ |")
            md.append("|---|---|---|---|---|---|---|---|")

            baselines = [m for m in methods if m != args.proposed]
            prop_blocks = by[args.proposed]

            raw_p, rows = [], []
            for b in baselines:
                base_blocks = by[b]
                common = sorted(set(prop_blocks) & set(base_blocks))
                if not common:
                    rows.append((b, float('nan'), 0, float('nan'), 'undefined',
                                 float('nan'), float('nan')))
                    raw_p.append(1.0); continue
                a = np.array([prop_blocks[k] for k in common], float)
                bvec = np.array([base_blocks[k] for k in common], float)
                _, p = wilcoxon_safe(a, bvec)
                d, mag = cliffs_delta(a, bvec)
                lo, hi = bootstrap_ci(a - bvec)
                rows.append((b, p, len(common), d, mag, lo, hi))
                raw_p.append(p if not np.isnan(p) else 1.0)
            adj_p = holm_bonferroni(raw_p)

            md.append(f"| **{args.proposed}** | {means[args.proposed]:.4f} | {stds[args.proposed]:.4f} | "
                      f"{counts[args.proposed]} | — | — | — | — |")
            csv = ["method,mean,std,n_paired,wilcoxon_p,holm_p,cliffs_delta,cliffs_mag,ci_lo,ci_hi"]
            csv.append(f"{args.proposed},{means[args.proposed]:.6f},{stds[args.proposed]:.6f},"
                       f"{counts[args.proposed]},,,,,,")
            bsorted = sorted(range(len(baselines)), key=lambda i: -means[baselines[i]])
            for i in bsorted:
                b, p, n, d, mag, lo, hi = rows[i]
                pstr = "n/a" if np.isnan(p) else f"{p:.2e}"
                hstr = "n/a" if np.isnan(adj_p[i]) else f"{adj_p[i]:.2e}"
                cistr = f"[{lo:+.4f}, {hi:+.4f}]" if not np.isnan(lo) else "n/a"
                md.append(f"| {b} | {means[b]:.4f} | {stds[b]:.4f} | {n} | {pstr} | {hstr} | "
                          f"{d:+.3f} ({mag}) | {cistr} |")
                csv.append(f"{b},{means[b]:.6f},{stds[b]:.6f},{n},{p:.6g},{adj_p[i]:.6g},"
                           f"{d:.6f},{mag},{lo:.6f},{hi:.6f}")
            with open(os.path.join(args.outdir, f"pairwise_{ds}_{metric}.csv"), 'w') as f:
                f.write("\n".join(csv) + "\n")

            # interpretation against the strongest baseline (by mean)
            best_i = max(range(len(baselines)), key=lambda i: means[baselines[i]])
            bb, bp, bn, bd, bmag, blo, bhi = rows[best_i]
            sig = (not np.isnan(bp)) and adj_p[best_i] < args.alpha
            md.append(f"\n*Strongest baseline = **{bb}** (mean {means[bb]:.4f}, "
                      f"N_paired={bn}). {args.proposed} vs {bb}: Holm-adjusted p = "
                      f"{('n/a' if np.isnan(adj_p[best_i]) else f'{adj_p[best_i]:.2e}')}, "
                      f"Cliff δ = {bd:+.3f} ({bmag}), mean Δ 95% CI [{blo:+.4f}, {bhi:+.4f}]. "
                      f"Difference is {'STATISTICALLY SIGNIFICANT' if sig else 'NOT significant'} "
                      f"at alpha={args.alpha}.*\n")

            # Friedman / CD: methods with >= min_friedman_blocks, intersect blocks
            elig = [m for m in methods if counts[m] >= args.min_friedman_blocks]
            if len(elig) >= 3:
                common = sorted(set.intersection(*[set(by[m].keys()) for m in elig]))
                if common:
                    mat = np.array([[by[m][k] for m in elig] for k in common], float)
                    excluded = [m for m in methods if m not in elig]
                    cd_res = friedman_and_cd(mat, elig, alpha=args.alpha)
                    cd_path = os.path.join(args.outdir, f"cd_{ds}_{metric}.pdf")
                    try:
                        plot_cd(cd_res, cd_path)
                        exc_note = f"  (excluded from Friedman: {excluded}, fewer than {args.min_friedman_blocks} blocks)" if excluded else ""
                        md.append(f"Friedman χ² across {cd_res['k']} methods on N={cd_res['N']} paired blocks: "
                                  f"p = {cd_res['friedman_p']:.2e}; CD = {cd_res['cd']:.3f}. "
                                  f"Diagram: `{os.path.basename(cd_path)}`.{exc_note} "
                                  f"Mean ranks (lower=better): " +
                                  ", ".join(f"{m} {r:.2f}" for m, r in
                                            sorted(zip(cd_res['method_names'], cd_res['mean_ranks']),
                                                   key=lambda t: t[1])) + ".\n")
                    except Exception as e:
                        md.append(f"_CD diagram failed: {e}_\n")

    with open(os.path.join(args.outdir, "report.md"), 'w') as f:
        f.write("\n".join(md) + "\n")
    print(f"[sig_analysis v2] wrote {os.path.join(args.outdir,'report.md')}")
    print(f"[sig_analysis v2] CSVs + CD diagrams in {args.outdir}/")


if __name__ == "__main__":
    main()
