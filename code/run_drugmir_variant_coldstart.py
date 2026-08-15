"""
Variant cold-start runner that REUSES the main DrugMiR_Hybrid pipeline
(run_drugmir_coldstart.py) verbatim — same build_fold_data, same training
loop — and only toggles a channel via --variant {full,no_hybrid,no_bridge}.
This guarantees Table V variants are identical-pipeline to Table IV.
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))

# ---- reuse EVERYTHING from the main cold-start script ----
import run_drugmir_coldstart as M
from hp_finetune import HybridEnc, DrugMiR_Hybrid, GG, GB

device = M.device if hasattr(M, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- wrap HybridEnc to support disabling the feature path (no_hybrid) ----
class HybridEncV(HybridEnc):
    def __init__(s, n, fd, h, dr, use_hybrid=True):
        super().__init__(n, fd, h, dr)
        s.use_hybrid = use_hybrid
    def forward(s, feat, has_feat=None):
        if not s.use_hybrid:
            return s.emb.weight
        return super().forward(feat, has_feat)

class DrugMiR_Variant(nn.Module):
    """Replicates DrugMiR_Hybrid.__init__ EXACTLY (same submodule creation order
    -> identical RNG consumption -> 'full' is bit-identical to Table IV), with
    HybridEncV in the me/de slots so use_hybrid can be toggled, and no_bridge
    handled in forward."""
    def __init__(s, nm, nd, md, dd, ng, variant='full', h=256, dr=0.5, n_gcn=2, n_br=2):
        super().__init__()
        s.variant = variant
        uh = (variant != 'no_hybrid')
        s.me = HybridEncV(nm, md, h, dr, use_hybrid=uh)   # slot 1 (== DrugMiR_Hybrid)
        s.de = HybridEncV(nd, dd, h, dr, use_hybrid=uh)   # slot 2
        s.ge = nn.Embedding(ng, h)                          # slot 3
        s.mgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        s.dgcn = nn.ModuleList([GG(h, dr) for _ in range(n_gcn)])
        s.br = nn.ModuleList([GB(h, dr) for _ in range(n_br)])
        s.pred = nn.Sequential(
            nn.Linear(6 * h, 2 * h), nn.BatchNorm1d(2 * h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(2 * h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h, 1))
    def forward(s, data, mi, di):
        m0 = s.me(data['mirna_feat'], data.get('mirna_has_feat'))
        d0 = s.de(data['drug_feat'], data.get('drug_has_feat'))
        mh = m0; dh = d0
        for l in s.mgcn: mh = l(mh, data['mirna_sim_edge'])
        for l in s.dgcn: dh = l(dh, data['drug_sim_edge'])
        mb = m0; db = d0; gh = s.ge.weight
        for l in s.br:
            mb, db, gh = l(mb, db, gh, data['mg_src'], data['mg_dst'],
                            data['dg_src'], data['dg_dst'], data['n_gene'])
        if s.variant == 'no_bridge':
            mb = torch.zeros_like(mb); db = torch.zeros_like(db)
        return s.pred(torch.cat([torch.cat([m0, mh, mb], -1)[mi],
                                  torch.cat([d0, dh, db], -1)[di]], -1)).squeeze(-1)

def run_one_fold(static, fold, variant, seed, h=256, dr=0.5, n_gcn=2, n_br=2,
                 km=15, kd=10, lr=1e-3, wd=2e-4, ep=200, pat=15):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    data = M.build_fold_data(static, fold, km=km, kd=kd)   # <-- SAME builder as Table IV
    train_pos = [(int(m), int(d)) for m, d in fold.train_pairs]
    test_pos  = [(int(m), int(d)) for m, d in fold.test_pairs]
    md, dd = data['mirna_feat'].shape[1], data['drug_feat'].shape[1]
    ng = data['n_gene']; nm, nd = data['n_mirna'], data['n_drug']
    model = DrugMiR_Variant(nm, nd, md, dd, ng, variant=variant,
                            h=h, dr=dr, n_gcn=n_gcn, n_br=n_br).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
    pc = 0
    for e in range(ep):
        M.train_one_epoch(model, data, train_pos, opt)      # <-- SAME train loop
        if (e + 1) % 5 == 0:
            m = M.evaluate_5metrics(model, data, test_pos)  # <-- SAME eval
            if m['auc'] > best['auc']: best = m; pc = 0
            else: pc += 1
            if pc >= pat: break
    return best

def run(dataset_name, data_dir, setting, variant, seed=42, n_fold=5):
    print(f"\n{'='*72}\nVARIANT {variant} cold-start: {dataset_name}/{setting}/seed={seed}\n{'='*72}", flush=True)
    static = M.load_static_data(data_dir)
    splitter = M.ColdStartSplitter(static['assoc'], n_folds=n_fold, seed=seed)
    aucs=[]; auprs=[]; f1s=[]; precs=[]; recs=[]; t0=time.time()
    for f in splitter.split(setting):
        b = run_one_fold(static, f, variant=variant, seed=seed)
        print(f"  Fold {f.fold_id+1}/{n_fold}: AUC={b['auc']:.4f} AUPR={b['aupr']:.4f} F1={b['f1']:.4f}", flush=True)
        aucs.append(b['auc']); auprs.append(b['aupr']); f1s.append(b['f1']); precs.append(b['prec']); recs.append(b['rec'])
    res={'auc':{'mean':float(np.mean(aucs)),'std':float(np.std(aucs))},
         'aupr':{'mean':float(np.mean(auprs)),'std':float(np.std(auprs))},
         'f1':{'mean':float(np.mean(f1s)),'std':float(np.std(f1s))},
         'prec':{'mean':float(np.mean(precs)),'std':float(np.std(precs))},
         'rec':{'mean':float(np.mean(recs)),'std':float(np.std(recs))},
         'dataset':dataset_name,'setting':setting,'variant':variant,'seed':seed,
         'total_time':time.time()-t0}
    print(f"  {variant} {dataset_name}/{setting}: AUC={res['auc']['mean']:.4f}±{res['auc']['std']:.4f}  ({res['total_time']:.0f}s)")
    return res

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True,choices=['D1','D2'])
    ap.add_argument('--setting',default='S4')
    ap.add_argument('--variant',required=True,choices=['full','no_hybrid','no_bridge'])
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--out_dir',default=os.path.expanduser('~/work/DrugMiR/coldstart_outputs'))
    a=ap.parse_args()
    DD={'D1':os.path.expanduser('~/work/DrugMiR/data/processed'),
        'D2':os.path.expanduser('~/work/DrugMiR/DMGAT_processed')}[a.dataset]
    os.makedirs(a.out_dir,exist_ok=True)
    res=run(a.dataset,DD,a.setting,a.variant,seed=a.seed)
    fn=f"{a.out_dir}/drugmirV_{a.variant}_{a.dataset}_{a.setting}_seed{a.seed}.json"
    json.dump(res,open(fn,'w'),indent=2)
    print("  Saved:",fn)
