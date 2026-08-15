"""
Regenerate Fig.2 ROC and PR curves from the rerun fold-level predictions.
Uses real (y_true, y_pred) saved in results_final/drugmir_predictions_full_seed42.json
to plot per-fold curves, the fold-mean curve, and a ±1 std shaded band.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc as sk_auc

with open('results_final/drugmir_predictions_full_seed42.json') as f:
    data = json.load(f)

def collect_curves(folds):
    """For each fold get (fpr, tpr), (rec, prec). Interpolate to common grid."""
    # Use a common grid of 200 points
    fpr_grid = np.linspace(0, 1, 200)
    rec_grid = np.linspace(0, 1, 200)

    tpr_interps = []
    prec_interps = []
    aucs = []
    auprs = []

    for f in folds:
        y_true = np.array(f['y_true'])
        y_pred = np.array(f['y_pred'])
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = sk_auc(fpr, tpr)
        aucs.append(roc_auc)
        # Interp to common grid
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interps.append(tpr_interp)
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        # rec descending from precision_recall_curve, reverse:
        prec_rev = prec[::-1]
        rec_rev = rec[::-1]
        pr_auc = sk_auc(rec_rev, prec_rev)
        auprs.append(pr_auc)
        prec_interp = np.interp(rec_grid, rec_rev, prec_rev)
        prec_interps.append(prec_interp)

    tpr_mean = np.mean(tpr_interps, axis=0)
    tpr_std = np.std(tpr_interps, axis=0)
    prec_mean = np.mean(prec_interps, axis=0)
    prec_std = np.std(prec_interps, axis=0)
    return {
        'fpr_grid': fpr_grid, 'tpr_mean': tpr_mean, 'tpr_std': tpr_std,
        'rec_grid': rec_grid, 'prec_mean': prec_mean, 'prec_std': prec_std,
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'aupr_mean': np.mean(auprs), 'aupr_std': np.std(auprs),
    }

d1 = collect_curves(data['D1']['fold_details'])
d2 = collect_curves(data['D2']['fold_details'])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# === Panel (a): ROC ===
ax = axes[0]
for ds, label, color in [
    (d1, f"Dataset 1 (AUC = {d1['auc_mean']:.4f}$\\pm${d1['auc_std']:.4f})", '#3b7dd8'),
    (d2, f"Dataset 2 (AUC = {d2['auc_mean']:.4f}$\\pm${d2['auc_std']:.4f})", '#e15759'),
]:
    ax.plot(ds['fpr_grid'], ds['tpr_mean'], color=color, lw=2, label=label)
    ax.fill_between(ds['fpr_grid'],
                     ds['tpr_mean'] - ds['tpr_std'],
                     ds['tpr_mean'] + ds['tpr_std'],
                     color=color, alpha=0.18)

ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('(a) ROC Curve', fontsize=11, loc='left')
ax.legend(loc='lower right', fontsize=9)
ax.grid(linestyle=':', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# === Panel (b): PR ===
ax = axes[1]
for ds, label, color in [
    (d1, f"Dataset 1 (AUPR = {d1['aupr_mean']:.4f}$\\pm${d1['aupr_std']:.4f})", '#3b7dd8'),
    (d2, f"Dataset 2 (AUPR = {d2['aupr_mean']:.4f}$\\pm${d2['aupr_std']:.4f})", '#e15759'),
]:
    ax.plot(ds['rec_grid'], ds['prec_mean'], color=color, lw=2, label=label)
    ax.fill_between(ds['rec_grid'],
                     ds['prec_mean'] - ds['prec_std'],
                     ds['prec_mean'] + ds['prec_std'],
                     color=color, alpha=0.18)

ax.set_xlim(0, 1); ax.set_ylim(0.4, 1.01)
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('(b) Precision-Recall Curve', fontsize=11, loc='left')
ax.legend(loc='lower left', fontsize=9)
ax.grid(linestyle=':', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('paper/figures/fig_roc_pr.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig_roc_pr_preview.png', format='png', bbox_inches='tight', dpi=120)
plt.close()

import os
print(f"Saved: paper/figures/fig_roc_pr.pdf  ({os.path.getsize('paper/figures/fig_roc_pr.pdf'):,} bytes)")
print(f"Saved: paper/figures/fig_roc_pr_preview.png  ({os.path.getsize('paper/figures/fig_roc_pr_preview.png'):,} bytes)")
print(f"\nLegend numbers in new figure:")
print(f"  Dataset 1: AUC = {d1['auc_mean']:.4f}±{d1['auc_std']:.4f}, AUPR = {d1['aupr_mean']:.4f}±{d1['aupr_std']:.4f}")
print(f"  Dataset 2: AUC = {d2['auc_mean']:.4f}±{d2['auc_std']:.4f}, AUPR = {d2['aupr_mean']:.4f}±{d2['aupr_std']:.4f}")
