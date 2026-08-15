import os, sys, json
import numpy as np, torch
sys.path.insert(0, 'scripts_gpu')
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from drugmir_model import load_data, DrugMiR_Hybrid, train_epoch, evaluate, set_seed, device

def get(dd, tag):
    set_seed(42); d = load_data(dd)
    pos = [tuple(p) for p in np.argwhere(d['assoc'] == 1)]
    tri, tei = next(iter(KFold(5, shuffle=True, random_state=42).split(pos)))
    ptr = [pos[i] for i in tri]; pte = [pos[i] for i in tei]
    m = DrugMiR_Hybrid(d['n_mirna'], d['n_drug'], d['mirna_feat'].shape[1],
                       d['drug_feat'].shape[1], d['n_gene']).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=2e-4)
    best, bs, bad = 0, None, 0
    for ep in range(1, 201):
        train_epoch(m, d, ptr, opt)
        if ep % 5 == 0:
            a, _ = evaluate(m, d, pte)
            if a > best: best, bad, bs = a, 0, {k: v.cpu().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 15: break
    m.load_state_dict(bs); m.eval()
    out = {}
    with torch.no_grad():
        for side, enc, feat, hf in [('mirna', m.enc.me, d['mirna_feat'], d['mirna_has_feat']),
                                    ('drug',  m.enc.de, d['drug_feat'],  d['drug_has_feat'])]:
            fh = enc.feat(feat); eh = enc.emb.weight
            g = torch.sigmoid(enc.gate(torch.cat([fh, eh], -1))).cpu().numpy()
            out[side] = dict(g=g, has=hf.cpu().numpy().astype(bool))
    out['assoc'] = d['assoc']
    print("  %s AUC=%.4f" % (tag, best), flush=True)
    return out

r1 = get(os.path.expanduser('~/work/DrugMiR/data/processed'), 'D1')
r2 = get(os.path.expanduser('~/work/DrugMiR/DMGAT_processed'), 'D2')

print("\n===== (c) 候选一：ḡ 与关联度数的关系 =====")
for tag, r in [('D1', r1), ('D2', r2)]:
    A = r['assoc']
    for side, deg in [('mirna', A.sum(1)), ('drug', A.sum(0))]:
        has = r[side]['has']; gm = r[side]['g'][has].mean(1); dg = deg[has]
        rs, ps = spearmanr(dg, gm); rp, pp = pearsonr(np.log1p(dg), gm)
        print("  %s %-6s n=%4d  Spearman ρ=%+.3f (p=%.3g)  Pearson(log deg) r=%+.3f (p=%.3g)"
              % (tag, side, has.sum(), rs, ps, rp, pp))

print("\n===== (c) 候选二：缺特征实体的邻居是否影响门控 =====")
A = r2['assoc']
for side, ax_, other in [('drug', 0, 'mirna'), ('mirna', 1, 'drug')]:
    has = r2[side]['has']
    oth_has = r2[other]['has']
    # 每个实体的伙伴中缺特征的比例
    frac = []
    for i in range(len(has)):
        p = A[i] if side == 'mirna' else A[:, i]
        idx = np.where(p == 1)[0]
        frac.append((~oth_has[idx]).mean() if len(idx) else np.nan)
    frac = np.array(frac)
    gm = r2[side]['g'].mean(1)
    ok = has & ~np.isnan(frac)
    if ok.sum() > 10:
        rs, ps = spearmanr(frac[ok], gm[ok])
        print("  D2 %-6s ρ(伙伴缺特征比例, ḡ)=%+.3f (p=%.3g)  n=%d" % (side, rs, ps, ok.sum()))

print("\n===== (c) 候选三：有特征 vs 缺特征实体的 g 是否本身就不同 =====")
for tag, r in [('D1', r1), ('D2', r2)]:
    for side in ['mirna', 'drug']:
        has = r[side]['has']
        if (~has).sum() < 5: print("  %s %-6s 缺特征仅 %d 个，跳过" % (tag, side, (~has).sum())); continue
        a = r[side]['g'][has].mean(1); b = r[side]['g'][~has].mean(1)
        u, p = mannwhitneyu(a, b)
        print("  %s %-6s 有特征 ḡ=%.4f (n=%d) | 缺特征 ḡ=%.4f (n=%d)  Mann-Whitney p=%.3g"
              % (tag, side, a.mean(), len(a), b.mean(), len(b), p))

print("\n===== 逐维度跨度（面板 a 的依据）=====")
for tag, r in [('D1', r1), ('D2', r2)]:
    for side in ['mirna', 'drug']:
        v = r[side]['g'][r[side]['has']].mean(0)
        print("  %s %-6s min=%.4f max=%.4f 跨度=%.4f  std(维度间)=%.4f"
              % (tag, side, v.min(), v.max(), v.max()-v.min(), v.std()))
np.savez('results_final/gate_raw.npz',
         **{f'{t}_{s}_g': r[s]['g'] for t, r in [('D1', r1), ('D2', r2)] for s in ['mirna','drug']},
         **{f'{t}_{s}_has': r[s]['has'] for t, r in [('D1', r1), ('D2', r2)] for s in ['mirna','drug']},
         D1_assoc=r1['assoc'], D2_assoc=r2['assoc'])
print("\n原始门控值已存 results_final/gate_raw.npz（画图不必重训）")
