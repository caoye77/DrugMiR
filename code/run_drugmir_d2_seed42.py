"""
Phase 2a-D2: DrugMiR on D2 (DMGAT_processed/) — seed=42, 5-fold, 5 metrics
"""
import os, sys, json, time
# Import everything from D1 script — reuse model + ev_full + run_one_dataset
sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
import importlib.util
spec = importlib.util.spec_from_file_location("d1_run", "run_drugmir_5metrics_seed42.py")
# But this would re-run D1 because of its __main__. So copy run_one_dataset only:

# === Just copy what we need ===
# Re-execute the function definitions (everything before __main__)
script_path = 'run_drugmir_5metrics_seed42.py'
with open(script_path) as f:
    code = f.read()
# Cut off the __main__ block
main_idx = code.find("if __name__ == '__main__':")
exec(code[:main_idx])  # This defines: load_data, sn, GG, GB, HybridEnc, DrugMiR_Hybrid, trn, ev_full, run_one_dataset

# === Now run D2 ===
DD2 = os.path.expanduser("~/DrugMiR/data/dataset2")
out_d2 = run_one_dataset('D2', DD2, lr=5e-4)  # same lr as D1

# Merge with existing D1 results
with open('phase2_outputs/drugmir_5metrics_seed42.json') as f:
    out = json.load(f)
out['D2'] = out_d2
with open('phase2_outputs/drugmir_5metrics_seed42.json', 'w') as f:
    json.dump(out, f, indent=2)

print("\n✓ D1+D2 merged in phase2_outputs/drugmir_5metrics_seed42.json")
print("\n=== D2 summary ===")
for k in ['auc','aupr','f1','prec','rec']:
    print(f"  {k.upper():6s}: {out_d2[k]['mean']:.4f} ± {out_d2[k]['std']:.4f}")
