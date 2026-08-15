import pathlib, sys
CHANGES=[]
def patch(fname, old, new, label):
    p=pathlib.Path(fname)
    if not p.exists(): print(f'  SKIP  {fname} not found'); return
    s=p.read_text()
    if new in s: print(f'  ok    {label} (already applied)'); return
    if old not in s: print(f'  FAIL  {label}: anchor not found'); CHANGES.append(False); return
    p.write_text(s.replace(old,new,1)); print(f'  DONE  {label}'); CHANGES.append(True)

print('patching...')
patch('exp_fusion.py',
"    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)\n    aucs, auprs = [], []\n    n_params = None\n    for fold, (tri, tei) in enumerate(kf.split(pos)):\n",
"    n_split = n_folds if n_folds >= 2 else 5\n    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)\n    aucs, auprs = [], []\n    n_params = None\n    for fold, (tri, tei) in enumerate(kf.split(pos)):\n        if n_folds < 2 and fold >= 1:\n            break\n",
'exp_fusion.run_cv accepts n_folds=1')

patch('exp_shap.py',
"default=os.path.expanduser('~/DrugMiR/data/dataset1'),",
"default=os.path.expanduser('~/work/DrugMiR/data/processed'),",
'exp_shap --data_dir -> NFS copy')

patch('exp_shap.py',
"    data = load_data(a.data_dir, km=cfg['km'], kd=cfg['kd'])\n    model = DrugMiR_Attrib(",
"    data = load_data(a.data_dir, km=cfg['km'], kd=cfg['kd'])\n    got = (data['n_mirna'], data['n_drug'], data['n_gene'])\n    want = (shapes['nm'], shapes['nd'], shapes['ng'])\n    if got != want:\n        sys.exit(f'ABORT: --data_dir {a.data_dir} has (nm,nd,ng)={got} but the checkpoint expects {want}. Wrong dataset.')\n    print(f'  data/checkpoint shapes match: {got}')\n    model = DrugMiR_Attrib(",
'exp_shap aborts on shape mismatch')

print()
if False in CHANGES: print('SOME PATCHES FAILED'); sys.exit(1)
import py_compile
for f in ('exp_fusion.py','exp_shap.py'): py_compile.compile(f, doraise=True)
print('all applied, syntax OK')
