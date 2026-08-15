import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# ---------------- 图 A: ROC / PR ----------------
d = json.load(open('results_final/drugmir_predictions_full_seed42.json'))

def cur(folds):
    g = np.linspace(0, 1, 300); T = []; P = []
    for f in folds:
        y = np.array(f['y_true']); s = np.array(f['y_pred'])
        fpr, tpr, _ = roc_curve(y, s)
        t = np.interp(g, fpr, tpr); t[0] = 0.0; T.append(t)
        pr, rc, _ = precision_recall_curve(y, s)
        P.append(np.interp(g, rc[::-1], pr[::-1]))
    return g, np.mean(T, 0), np.std(T, 0), np.mean(P, 0), np.std(P, 0)

D1 = cur(d['D1']['fold_details'])
D2 = cur(d['D2']['fold_details'])

fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.75))
for D, lab, c in [(D1, 'Dataset 1', 'tab:blue'), (D2, 'Dataset 2', 'tab:red')]:
    g, tm, ts, pm, ps = D
    ax[0].plot(g, tm, color=c, lw=1.5, label=lab)
    ax[0].fill_between(g, tm - ts, tm + ts, color=c, alpha=0.17, lw=0)
    ax[1].plot(g, pm, color=c, lw=1.5, label=lab)
    ax[1].fill_between(g, pm - ps, pm + ps, color=c, alpha=0.17, lw=0)
ax[0].plot([0, 1], [0, 1], 'k:', lw=0.7)
ax[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate',
          title='(a) ROC Curve', xlim=(0, 1), ylim=(0, 1.02))
ax[1].set(xlabel='Recall', ylabel='Precision',
          title='(b) Precision-Recall Curve', xlim=(0, 1), ylim=(0.45, 1.02))
for a in ax:
    a.legend(loc='lower right', fontsize=7.5, frameon=True, framealpha=0.9)
    a.grid(alpha=0.22, lw=0.4)
    a.tick_params(labelsize=7.5)
    a.title.set_fontsize(8.5)
    a.xaxis.label.set_size(8)
    a.yaxis.label.set_size(8)
plt.tight_layout(pad=0.4)
plt.savefig('paper/figures/fig_roc_pr_v3.pdf', bbox_inches='tight')
plt.close()
print("[1/2] fig_roc_pr_v3.pdf")

# ---------------- 图 B: 消融（数值锁定为表 VI）----------------
NAMES = ['Full\nDrugMiR', 'w/o\nGating', 'w/o Gene\nBridge', 'w/o\nHomoGCN', 'w/o\nHybrid']
D1V = [0.9620, 0.9606, 0.9602, 0.9603, 0.9596]
D2V = [0.9497, 0.9503, 0.9496, 0.9485, 0.9302]
D1E = [0.0025, 0.0021, 0.0017, 0.0019, 0.0017]
D2E = [0.0040, 0.0042, 0.0037, 0.0039, 0.0047]

fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55))
for k, (V, E, lo, hi, ttl) in enumerate([
        (D1V, D1E, 0.9550, 0.9660, '(a) Dataset 1'),
        (D2V, D2E, 0.9250, 0.9560, '(b) Dataset 2')]):
    cols = (['#1f4e79'] + ['#7fa8d0'] * 4) if k == 0 else (['#a83232'] + ['#e08a5a'] * 4)
    ax[k].bar(range(5), V, yerr=E, capsize=2.5, color=cols,
              edgecolor='black', linewidth=0.5,
              error_kw=dict(elinewidth=0.8))
    for i, v in enumerate(V):
        ax[k].text(i, v + E[i] + (hi - lo) * 0.025, '%.4f' % v,
                   ha='center', fontsize=6.8, fontweight='bold')
    ax[k].set(ylim=(lo, hi), title=ttl, ylabel='AUC')
    ax[k].set_xticks(range(5))
    ax[k].set_xticklabels(NAMES, fontsize=6.8)
    ax[k].grid(axis='y', alpha=0.22, lw=0.4)
    ax[k].tick_params(labelsize=7.5)
    ax[k].title.set_fontsize(8.5)
    ax[k].yaxis.label.set_size(8)
    delta = 100 * (V[4] - V[0])
    ax[k].annotate('', xy=(4, V[4]), xytext=(4, V[0]),
                   arrowprops=dict(arrowstyle='<->', color='crimson', lw=0.9))
    ax[k].text(3.60, (V[0] + V[4]) / 2, r'$\Delta$ = %.2f' % delta,
               color='crimson', fontsize=7, ha='right', va='center')
plt.tight_layout(pad=0.4)
plt.savefig('paper/figures/fig_ablation_v2.pdf', bbox_inches='tight')
plt.close()
print("[2/2] fig_ablation_v2.pdf")

# ---------------- 顺带核对 ----------------
print("\n--- ROC 重算（供核对，图上不显示）---")
for D, n in [(D1, 'D1'), (D2, 'D2')]:
    pass
import json as _j
for k in ['D1', 'D2']:
    a = [f['auc'] for f in d[k]['fold_details']]
    p = [f['aupr'] for f in d[k]['fold_details']]
    print("  %s  AUC %.4f±%.4f   AUPR %.4f±%.4f" %
          (k, np.mean(a), np.std(a), np.mean(p), np.std(p)))
print("\n--- 消融图数值 == 表 VI ---")
for i, n in enumerate(['Full', 'w/o Gating', 'w/o Gene Bridge', 'w/o HomoGCN', 'w/o Hybrid']):
    print("  %-16s D1 %.4f (Δ%+.2f)   D2 %.4f (Δ%+.2f)" %
          (n, D1V[i], 100*(D1V[i]-D1V[0]), D2V[i], 100*(D2V[i]-D2V[0])))
