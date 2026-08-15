"""
LRGCPND hyperparam sweep on D1 — find what closes the gap to 0.9444.
Test 3-4 configs, pick best.
"""
import os, sys, json, time
sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
exec(open('run_lrgcpnd_d1d2_seed42.py').read().split("if __name__ == '__main__':")[0])

DD1 = os.path.expanduser("~/DrugMiR/data/dataset1")

# Config grid — vary E_size, lr, epochs
configs = [
    {'tag': 'baseline (repo default)',     'K': 4, 'E_size': 32,  'reg': 0.05, 'lr': 0.005, 'max_epoch': 100, 'patience': 10},
    {'tag': 'E64 default',                  'K': 4, 'E_size': 64,  'reg': 0.05, 'lr': 0.005, 'max_epoch': 100, 'patience': 10},
    {'tag': 'E64 + longer train',           'K': 4, 'E_size': 64,  'reg': 0.05, 'lr': 0.005, 'max_epoch': 200, 'patience': 20},
    {'tag': 'E128 + slow lr',               'K': 4, 'E_size': 128, 'reg': 0.05, 'lr': 0.001, 'max_epoch': 200, 'patience': 20},
]

print("="*70)
print("LRGCPND hyperparam sweep on D1")
print("="*70)

results = []
for cfg in configs:
    print(f"\n>>> Config: {cfg['tag']}")
    print(f"    K={cfg['K']} E_size={cfg['E_size']} reg={cfg['reg']} lr={cfg['lr']} "
          f"max_epoch={cfg['max_epoch']} patience={cfg['patience']}")
    r = run_lrgcpnd_dataset('D1', DD1,
                             K=cfg['K'], E_size=cfg['E_size'], reg=cfg['reg'],
                             lr=cfg['lr'], max_epoch=cfg['max_epoch'],
                             patience=cfg['patience'])
    r['config_tag'] = cfg['tag']
    r['config'] = cfg
    results.append(r)

# Final ranking
print(f"\n{'='*70}\nSWEEP RESULTS (sorted by D1 AUC)\n{'='*70}")
results.sort(key=lambda r: -r['auc']['mean'])
print(f"{'Config':35s} | {'AUC':16s} | {'AUPR':16s} | {'F1':16s}")
print("-" * 100)
for r in results:
    auc = f"{r['auc']['mean']:.4f}±{r['auc']['std']:.4f}"
    aupr = f"{r['aupr']['mean']:.4f}±{r['aupr']['std']:.4f}"
    f1 = f"{r['f1']['mean']:.4f}±{r['f1']['std']:.4f}"
    print(f"{r['config_tag']:35s} | {auc:16s} | {aupr:16s} | {f1:16s}")

print(f"\nTarget: D1 AUC ≈ 0.9444 (paper reported)")
print(f"Best:   D1 AUC = {results[0]['auc']['mean']:.4f} ({results[0]['config_tag']})")

# Save sweep results
with open('phase2_outputs/lrgcpnd_d1_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved to phase2_outputs/lrgcpnd_d1_sweep.json")
