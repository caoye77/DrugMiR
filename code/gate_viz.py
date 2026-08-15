import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, 'scripts_gpu')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from drugmir_model import (load_data, DrugMiR_Hybrid, train_epoch,
                           evaluate, set_seed, device)

D1 = os.path.expanduser('~/work/DrugMiR/data/processed')
D2 = os.path.expanduser('~/work/DrugMiR/DMGAT_processed')

def run(dd, tag):
    set_seed(42)
    d = load_data(dd)
    print("  n_mirna=%d n_drug=%d  data keys=%s" %
          (d['n_mirna'], d['n_drug'], [k for k in d if 'gene' in k or 'n_' in k]), flush=True)
    ng = d.get('n_gene') or d['gene_edge_m'].max().item() + 1
    pos = [tuple(p) for p in np.argwhere(d['assoc'] == 1)]
    tri, tei = next(iter(KFold(5, shuffle=True, random_state=42).split(pos)))
    ptr = [pos[i] for i in tri]; pte = [pos[i] for i in tei]

    m = DrugMiR_Hybrid(d['n_mirna'], d['n_drug'],
                       d['mirna_feat'].shape[1], d['drug_feat'].shape[1], ng).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=2e-4)
    best, bstate, bad, t0 = 0, None, 0, time.time()
    for ep in range(1, 201):
        train_epoch(m, d, ptr, opt)
        if ep % 5 == 0:
            auc, _ = evaluate(m, d, pte)
            if auc > best:
                best, bad = auc, 0
                bstate = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 15: break
    m.load_state_dict(bstate); m.eval()
    print("  %s best AUC=%.4f (%ds)" % (tag, best, time.time()-t0), flush=True)

    with torch.no_grad():
        res = {}
        for side, enc, feat, hf in [('mirna', m.enc.me, d['mirna_feat'], d['mirna_has_feat']),
                                    ('drug',  m.enc.de, d['drug_feat'],  d['drug_has_feat'])]:
            fh = enc.feat(feat); eh = enc.emb.weight
            g = torch.sigmoid(enc.gate(torch.cat([fh, eh], -1)))
            res[side] = dict(g=g.cpu().numpy(), has=hf.cpu().numpy().astype(bool))
    return res, best

print("训练 D1 ...", flush=True); r1, a1 = run(D1, 'D1')
print("训练 D2 ...", flush=True); r2, a2 = run(D2, 'D2')

print("\n===== 门控统计（仅统计有特征的实体）=====")
stat = {'auc': {'D1': a1, 'D2': a2}}
for tag, r in [('D1', r1), ('D2', r2)]:
    for side in ['mirna', 'drug']:
        g, has = r[side]['g'], r[side]['has']
        gw = g[has]
        stat[f'{tag}_{side}'] = dict(n=int(has.sum()), n_miss=int((~has).sum()),
                                     mean=float(gw.mean()), std=float(gw.std()))
        print("  %s %-6s n=%4d (缺%3d)  ḡ=%.4f ± %.4f  [逐维均值 %.4f~%.4f]" %
              (tag, side, has.sum(), (~has).sum(), gw.mean(), gw.std(),
               gw.mean(0).min(), gw.mean(0).max()))

fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35))
for tag, c, r in [('Dataset 1','tab:blue',r1), ('Dataset 2','tab:red',r2)]:
    v = np.concatenate([r[s]['g'][r[s]['has']].mean(1) for s in ['mirna','drug']])
    ax[0].hist(v, bins=34, alpha=.6, color=c, density=True,
               label='%s ($\\bar{g}$=%.3f)' % (tag, v.mean()))
ax[0].set(xlabel=r'per-entity mean gate $\bar{g}_i$', ylabel='density',
          title='(a) Fully vs partly annotated')
for side, c, lab in [('mirna','tab:green','miRNA (3.7% unfeatured)'),
                     ('drug','tab:orange','Drug (24.8% unfeatured)')]:
    v = r2[side]['g'][r2[side]['has']].mean(1)
    ax[1].hist(v, bins=26, alpha=.6, color=c, density=True,
               label='%s\n$\\bar{g}$=%.3f' % (lab, v.mean()))
ax[1].set(xlabel=r'per-entity mean gate $\bar{g}_i$', ylabel='density',
          title='(b) Dataset 2 by entity type')
for tag, c, r in [('Dataset 1','tab:blue',r1), ('Dataset 2','tab:red',r2)]:
    v = np.concatenate([r[s]['g'][r[s]['has']] for s in ['mirna','drug']]).mean(0)
    ax[2].plot(np.sort(v), lw=1.4, color=c, label=tag)
ax[2].axhline(.5, color='k', ls=':', lw=.7)
ax[2].set(xlabel='hidden dimension (sorted)', ylabel=r'mean gate $g$',
          title='(c) Per-dimension gate')
for a in ax:
    a.legend(fontsize=6.2, frameon=True, framealpha=.9); a.grid(alpha=.22, lw=.4)
    a.tick_params(labelsize=7); a.title.set_fontsize(8.2)
    a.xaxis.label.set_size(7.5); a.yaxis.label.set_size(7.5)
plt.tight_layout(pad=.4)
plt.savefig('paper/figures/fig_gate.pdf', bbox_inches='tight')
json.dump(stat, open('results_final/gate_stats.json','w'), indent=1)
print("\n已生成 paper/figures/fig_gate.pdf")
