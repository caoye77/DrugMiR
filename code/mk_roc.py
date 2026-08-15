import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

d = json.load(open('results_final/drugmir_predictions_full_seed42.json'))

def cur(folds):
    g = np.linspace(0, 1, 300); T = []; P = []
    for f in folds:
        y = np.array(f['y_true']); s = np.array(f['y_pred'])
        fpr, tpr, _ = roc_curve(y, s)
        t = np.interp(g, fpr, tpr); t[0] = 0.0; T.append(t)
        pr, rc, _ = precision_recall_curve(y, s)
        P.append(np.interp(g, rc[::-1], pr[::-1]))
    auc  = np.mean([f['auc']  for f in folds])
    aupr = np.mean([f['aupr'] for f in folds])
    return g, np.mean(T,0), np.std(T,0), np.mean(P,0), np.std(P,0), auc, aupr

D1 = cur(d['D1']['fold_details'])
D2 = cur(d['D2']['fold_details'])

fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.75))
for D, lab, c in [(D1,'Dataset 1','tab:blue'), (D2,'Dataset 2','tab:red')]:
    g, tm, ts, pm, ps, auc, aupr = D
    ax[0].plot(g, tm, color=c, lw=1.5, label='%s  (AUC = %.4f)'  % (lab, auc))
    ax[0].fill_between(g, tm-ts, tm+ts, color=c, alpha=0.17, lw=0)
    ax[1].plot(g, pm, color=c, lw=1.5, label='%s  (AUPR = %.4f)' % (lab, aupr))
    ax[1].fill_between(g, pm-ps, pm+ps, color=c, alpha=0.17, lw=0)
ax[0].plot([0,1],[0,1],'k:',lw=0.7)
ax[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate',
          title='(a) ROC Curve', xlim=(0,1), ylim=(0,1.02))
ax[1].set(xlabel='Recall', ylabel='Precision',
          title='(b) Precision-Recall Curve', xlim=(0,1), ylim=(0.45,1.02))
for a in ax:
    a.legend(loc='lower right', fontsize=7, frameon=True, framealpha=0.92)
    a.grid(alpha=0.22, lw=0.4); a.tick_params(labelsize=7.5)
    a.title.set_fontsize(8.5); a.xaxis.label.set_size(8); a.yaxis.label.set_size(8)
plt.tight_layout(pad=0.4)
plt.savefig('paper/figures/fig_roc_pr_v3.pdf', bbox_inches='tight')
print("已覆盖 fig_roc_pr_v3.pdf")
print("图例数值: D1 AUC=%.4f AUPR=%.4f | D2 AUC=%.4f AUPR=%.4f" % (D1[5],D1[6],D2[5],D2[6]))
