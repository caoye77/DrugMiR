"""
LRGCPND final benchmark with best config (E64 + longer training).
Re-runs D1 (for clean log) and D2.
"""
import os, sys, json
sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
exec(open('run_lrgcpnd_d1d2_seed42.py').read().split("if __name__ == '__main__':")[0])

BEST_CFG = {'K': 4, 'E_size': 64, 'reg': 0.05, 'lr': 0.005,
            'max_epoch': 200, 'patience': 20}

DD1 = os.path.expanduser("~/DrugMiR/data/dataset1")
DD2 = os.path.expanduser("~/DrugMiR/data/dataset2")

out = {}
out['D1'] = run_lrgcpnd_dataset('D1', DD1, **BEST_CFG)
out['D2'] = run_lrgcpnd_dataset('D2', DD2, **BEST_CFG)
out['hyperparams'] = BEST_CFG
out['notes'] = (
    "Best LRGCPND config from D1 sweep (4 configs tested). "
    "Numbers under DrugMiR's unified protocol differ from "
    "LRGCPND's original paper (0.9444/0.9283) due to: "
    "(1) different cross-validation split; "
    "(2) per-epoch negative resampling vs original fixed neg set; "
    "(3) optimal-F1 threshold vs original threshold."
)

os.makedirs('phase2_outputs', exist_ok=True)
with open('phase2_outputs/lrgcpnd_5metrics_seed42.json', 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*70}\nLRGCPND FINAL (best config: E64 + longer training)\n{'='*70}")
print(f"{'Metric':8s} | {'D1':25s} | {'D2':25s}")
print("-" * 65)
for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
    d1 = f"{out['D1'][k]['mean']:.4f} ± {out['D1'][k]['std']:.4f}"
    d2 = f"{out['D2'][k]['mean']:.4f} ± {out['D2'][k]['std']:.4f}"
    print(f"{k.upper():8s} | {d1:25s} | {d2:25s}")

print(f"\nReported in original LRGCPND paper:")
print(f"  D1 AUC=0.9444 / D2 AUC=0.9283")
print(f"\nGap explained by unified protocol — see Table II footnote.")
