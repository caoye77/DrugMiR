import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc as A

d=json.load(open('results_final/drugmir_predictions_full_seed42.json'))
def cur(folds):
    g=np.linspace(0,1,200); T=[];P=[];a=[];p=[]
    for f in folds:
        y=np.array(f['y_true']); s=np.array(f['y_pred'])
        fpr,tpr,_=roc_curve(y,s); a.append(A(fpr,tpr))
        t=np.interp(g,fpr,tpr); t[0]=0; T.append(t)
        pr,rc,_=precision_recall_curve(y,s); pr,rc=pr[::-1],rc[::-1]
        p.append(A(rc,pr)); P.append(np.interp(g,rc,pr))
    return g,np.mean(T,0),np.std(T,0),np.mean(P,0),np.std(P,0),np.mean(a),np.std(a),np.mean(p),np.std(p)
D1=cur(d['D1']['fold_details']); D2=cur(d['D2']['fold_details'])
print("重算 D1: AUC=%.4f±%.4f AUPR=%.4f±%.4f"%(D1[5],D1[6],D1[7],D1[8]))
print("重算 D2: AUC=%.4f±%.4f AUPR=%.4f±%.4f"%(D2[5],D2[6],D2[7],D2[8]))

fig,ax=plt.subplots(1,2,figsize=(7.0,2.9))
for (D,lab,c) in [(D1,'Dataset 1','tab:blue'),(D2,'Dataset 2','tab:red')]:
    g,tm,ts=D[0],D[1],D[2]
    ax[0].plot(g,tm,color=c,lw=1.6,label='%s (AUC = %.4f)'%(lab,D[5]))
    ax[0].fill_between(g,tm-ts,tm+ts,color=c,alpha=.18,lw=0)
    pm,ps=D[3],D[4]
    ax[1].plot(g,pm,color=c,lw=1.6,label='%s (AUPR = %.4f)'%(lab,D[7]))
    ax[1].fill_between(g,pm-ps,pm+ps,color=c,alpha=.18,lw=0)
ax[0].plot([0,1],[0,1],'k:',lw=.7)
ax[0].set(xlabel='False Positive Rate',ylabel='True Positive Rate',title='(a) ROC Curve',xlim=(0,1),ylim=(0,1.02))
ax[1].set(xlabel='Recall',ylabel='Precision',title='(b) Precision-Recall Curve',xlim=(0,1),ylim=(.4,1.02))
for a_ in ax: a_.legend(loc='lower right',fontsize=6.5,frameon=True); a_.grid(alpha=.25,lw=.4); a_.tick_params(labelsize=7)
    
plt.tight_layout()
plt.savefig('paper/figures/fig_roc_pr_v3.pdf',bbox_inches='tight')
plt.savefig('paper/figures/fig_roc_pr_v3.png',dpi=130,bbox_inches='tight')
print("已生成 paper/figures/fig_roc_pr_v3.pdf")
