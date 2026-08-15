import os,json,argparse
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
RES=os.path.expanduser('~/work/DrugMiR')
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'stix','axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9.5,'savefig.dpi':300,'savefig.bbox':'tight'})
CF,CT,CE='#2c7fb8','#e6550d','#31a354'
def load_mask(ds):
    rows=json.load(open(os.path.join(RES,'results_mask',f'mask_{ds}_feature.json')))
    o={}
    for r in rows: o.setdefault(r['mode'],[]).append((r['ratio']*100,r['auc_mean'],r['auc_std']))
    for m in o: o[m].sort()
    return o
def panel(ax,ds,title):
    d=load_mask(ds)
    for mode,c,mk,lab in [('full',CF,'o','Full (fusion + fallback)'),('feat_only',CT,'s','Feature-only (no fallback)'),('emb_only',CE,'^','Embedding-only')]:
        if mode not in d: continue
        xs=[x for x,_,_ in d[mode]]; ys=[y for _,y,_ in d[mode]]; es=[e for _,_,e in d[mode]]
        ax.errorbar(xs,ys,yerr=es,marker=mk,color=c,lw=1.8,capsize=3,label=lab,markersize=5)
    ax.set_xlabel('Features masked (%)'); ax.set_ylabel('AUC'); ax.set_title(title)
    ax.grid(linestyle=':',alpha=0.4); ax.set_axisbelow(True); return d
def fig_masking(out):
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    panel(ax[0],'D1','Dataset 1 (fully annotated)')
    d2=panel(ax[1],'D2','Dataset 2 (24.8% drugs unfeatured)')
    for a in ax: a.set_ylim(0.90,0.965)
    ax[0].legend(loc='lower left',framealpha=0.9)
    # annotate each curve's DROP from 0% to 40% (matches paper text: feature-only -0.9 pt, full -0.2 pt)
    if 'full' in d2 and 'feat_only' in d2:
        xN=d2['feat_only'][-1][0]
        feat_drop=(d2['feat_only'][0][1]-d2['feat_only'][-1][1])*100
        full_drop=(d2['full'][0][1]-d2['full'][-1][1])*100
        ax[1].annotate(f'feature-only: $-${feat_drop:.1f} pt',
                       xy=(xN,d2['feat_only'][-1][1]), xytext=(xN-2,d2['feat_only'][-1][1]-0.008),
                       ha='right',va='top',fontsize=8.5,color=CT,
                       arrowprops=dict(arrowstyle='-',color=CT,lw=0.6))
        ax[1].annotate(f'full: $-${full_drop:.1f} pt',
                       xy=(xN,d2['full'][-1][1]), xytext=(xN-2,d2['full'][-1][1]+0.005),
                       ha='right',va='bottom',fontsize=8.5,color=CF,
                       arrowprops=dict(arrowstyle='-',color=CF,lw=0.6))
    plt.tight_layout()
    for e in ('pdf','png'): plt.savefig(os.path.join(out,f'fig_masking.{e}'))
    plt.close(); print('  fig_masking.pdf OK')
def fig_igshap(out):
    ig_p=os.path.join(RES,'results_final','gene_ig_importance_d1.npy')
    sh_p=os.path.join(RES,'results_final','gene_shap_importance_d1.npy')
    if not (os.path.exists(ig_p) and os.path.exists(sh_p)): print('  !! npy missing, skip'); return
    ig=np.load(ig_p); sh=np.load(sh_p); n=min(len(ig),len(sh)); ig,sh=ig[:n],sh[:n]
    a=(ig>0)&(sh>0); ig,sh=ig[a],sh[a]
    rho=None
    ag=os.path.join(RES,'results_final','ig_vs_shap_agreement.json')
    if os.path.exists(ag): rho=json.load(open(ag)).get('spearman_rho')
    fig,ax=plt.subplots(figsize=(5.2,5))
    ax.scatter(ig,sh,s=10,alpha=0.35,color=CF,edgecolors='none')
    lim=max(ig.max(),sh.max())*1.05; ax.plot([0,lim],[0,lim],'--',color='0.5',lw=1,label='$y=x$')
    ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    ax.set_xlabel('Integrated Gradients importance'); ax.set_ylabel('GradientSHAP importance')
    ax.set_title('Attribution agreement on bridge genes (D1)')
    if rho is not None: ax.text(0.05,0.92,f'Spearman $\\rho = {rho:.4f}$',transform=ax.transAxes,fontsize=11,bbox=dict(boxstyle='round',fc='white',ec='0.7'))
    ax.grid(linestyle=':',alpha=0.4); ax.set_axisbelow(True); ax.legend(loc='lower right')
    plt.tight_layout()
    for e in ('pdf','png'): plt.savefig(os.path.join(out,f'fig_ig_shap.{e}'))
    plt.close(); print('  fig_ig_shap.pdf OK')
out=os.path.join(RES,'results_final'); os.makedirs(out,exist_ok=True)
fig_masking(out); fig_igshap(out); print('done')
