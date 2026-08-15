import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

z = np.load('results_final/gate_raw.npz')
G, N = {}, {}
for t in ['D1','D2']:
    for s in ['mirna','drug']:
        g = z[f'{t}_{s}_g'][z[f'{t}_{s}_has'].astype(bool)]
        G[(t,s)] = g; N[(t,s)] = len(g)

fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.7))

# ---------------- (a) 逐维度门控 ----------------
style = [(('D1','mirna'), 'tab:blue', '-',  'D1 miRNA'),
         (('D1','drug'),  'tab:blue', '--', 'D1 drug'),
         (('D2','mirna'), 'tab:red',  '-',  'D2 miRNA'),
         (('D2','drug'),  'tab:red',  '--', 'D2 drug')]
for k, c, ls, lab in style:
    v = np.sort(G[k].mean(0))
    ax[0].plot(v, color=c, ls=ls, lw=1.5,
               label='%s  %.2f' % (lab, v.max()-v.min()))
ax[0].axhline(.5, color='k', ls=':', lw=.8)
ax[0].text(128, .492, 'equal weight', fontsize=5.9, ha='center', va='top', color='0.3')
ax[0].set(xlabel='hidden dimension (sorted by mean gate)', ylabel=r'mean gate $g$',
          title='(a) Per-dimension gate', xlim=(-4, 259), ylim=(.16, .615))
leg = ax[0].legend(fontsize=6.1, loc='upper left', frameon=True, framealpha=.93,
                   title='range', title_fontsize=6.1, handlelength=1.6,
                   borderpad=.35, labelspacing=.25, bbox_to_anchor=(.015, .985))
leg.get_frame().set_linewidth(.5)

# ---------------- (b) 每实体平均门控 ----------------
order = [('D1','mirna'), ('D1','drug'), ('D2','mirna'), ('D2','drug')]
data  = [G[k].mean(1) for k in order]
labs  = ['D1 miRNA\n$n$=%d' % N[order[0]], 'D1 drug\n$n$=%d' % N[order[1]],
         'D2 miRNA\n$n$=%d' % N[order[2]], 'D2 drug\n$n$=%d' % N[order[3]]]
bp = ax[1].boxplot(data, tick_labels=labs, widths=.5, showfliers=True,
                   patch_artist=True,
                   medianprops=dict(color='black', lw=1.1),
                   flierprops=dict(marker='.', ms=1.8, mfc='0.45', mec='none', alpha=.55),
                   whiskerprops=dict(lw=.8), capprops=dict(lw=.8))
for p, c in zip(bp['boxes'], ['tab:blue','tab:blue','tab:red','tab:red']):
    p.set_facecolor(c); p.set_alpha(.42); p.set_edgecolor('black'); p.set_linewidth(.6)
# 均值用白心菱形，标签放到箱子右侧，避开中位线
for i, v in enumerate(data):
    ax[1].plot(i+1, v.mean(), marker='D', ms=3.6, mfc='white',
               mec='black', mew=.8, zorder=5)
    ax[1].text(i+1.30, v.mean(), '%.3f' % v.mean(), fontsize=6.3,
               ha='left', va='center', fontweight='bold')
ax[1].axhline(.5, color='k', ls=':', lw=.8)
ax[1].text(4.48, .492, 'equal weight', fontsize=5.9, ha='right', va='top', color='0.3')
ax[1].plot([], [], 'D', ms=3.6, mfc='white', mec='black', mew=.8, label='mean')
ax[1].plot([], [], '.', ms=3, color='0.45', label='outlier')
lg = ax[1].legend(fontsize=6.1, loc='lower right', frameon=True, framealpha=.93,
                  handlelength=1.1, borderpad=.35, labelspacing=.25)
lg.get_frame().set_linewidth(.5)
ax[1].set(ylabel=r'per-entity mean gate $\bar{g}_i$',
          title='(b) Feature vs embedding balance', ylim=(.055, .535), xlim=(.45, 4.85))

for a in ax:
    a.grid(alpha=.2, lw=.4); a.tick_params(labelsize=6.8)
    a.title.set_fontsize(8.4); a.xaxis.label.set_size(7.4); a.yaxis.label.set_size(7.4)
plt.tight_layout(pad=.4)
plt.savefig('paper/figures/fig_gate.pdf', bbox_inches='tight')
print("已重生成 paper/figures/fig_gate.pdf\n")

print("=== 图上每个可见元素的数值（供逐项核对）===")
for k in order:
    m = G[k].mean(1); d = G[k].mean(0)
    q1,q2,q3 = np.percentile(m,[25,50,75]); iqr=q3-q1
    lo=m[m>=q1-1.5*iqr].min(); hi=m[m<=q3+1.5*iqr].max()
    print("  %s %-6s n=%4d | (a) 逐维 %.3f-%.3f 跨度 %.3f | (b) 下须 %.3f Q1 %.3f 中位 %.3f 均值 %.3f Q3 %.3f 上须 %.3f 离群 %d"
          % (k[0],k[1],N[k],d.min(),d.max(),d.max()-d.min(),lo,q1,q2,m.mean(),q3,hi,
             ((m<q1-1.5*iqr)|(m>q3+1.5*iqr)).sum()))
