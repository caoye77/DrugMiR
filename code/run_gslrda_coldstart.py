"""
GSLRDA cold-start runner: S2 / S3 / S4

GSLRDA is implemented with TF1 (tf.compat.v1). Its training routine internally
builds the ncRNA-drug bipartite graph from `trainingSet` triplets passed at
GSLRDA instantiation, so it is intrinsically cold-start safe at the graph
level when we only feed train_pos as trainingSet.

We re-use most of the original `train_gslrda_one_fold` machinery; the only
change is the split source. Negative sampling inside GSLRDA's `next_batch_pairwise`
uses `self.data.ncRNA` and `self.data.drug` dicts which are built from the
trainingSet — meaning the negative pool naturally excludes test entities under
cold-start. So no negative-sampling patch is needed here.

Usage:
  python3 run_gslrda_coldstart.py --dataset D1 --setting S2 --seed 42
"""
import os, sys, json, time, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_splitter import ColdStartSplitter, ColdFold

# Import original GSLRDA wrapper (must come AFTER coldstart_splitter to avoid
# any TF import order shadowing)
import importlib.util
spec = importlib.util.spec_from_file_location(
    'gslrda_orig',
    os.path.expanduser('~/work/DrugMiR/run_gslrda_d1d2_seed42.py'))
gslrda_orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gslrda_orig)


def run_dataset_coldstart(dataset_name, data_dir, setting, seed=42, n_fold=5,
                          max_epoch=100, eval_every=2, patience=5):
    print(f"\n{'='*72}\nGSLRDA cold-start: {dataset_name} / {setting} / seed={seed}\n{'='*72}", flush=True)
    t0 = time.time()
    assoc, pos_pairs, n_mirna, n_drug = gslrda_orig.load_assoc(data_dir)
    print(f"  n_mirna={n_mirna} n_drug={n_drug} n_pos={len(pos_pairs)}", flush=True)

    np.random.seed(seed)
    splitter = ColdStartSplitter(assoc, n_folds=n_fold, seed=seed, min_test_positives=15)
    folds = splitter.split(setting)

    fold_results = []
    for f in folds:
        tf0 = time.time()
        train_pos = [(int(m), int(d)) for m, d in f.train_pairs]
        test_pos  = [(int(m), int(d)) for m, d in f.test_pairs]
        conf = gslrda_orig.make_gslrda_config()
        best = gslrda_orig.train_gslrda_one_fold(
            train_pos, test_pos, assoc, n_mirna, n_drug, conf,
            max_epoch=max_epoch, eval_every=eval_every, patience=patience
        )
        fold_results.append(best)
        print(f"  Fold {f.fold_id+1}/{n_fold}: AUC={best['auc']:.4f} AUPR={best['aupr']:.4f} "
              f"F1={best['f1']:.4f} P={best['prec']:.4f} R={best['rec']:.4f} "
              f"(train_pos={len(train_pos)} test_pos={len(test_pos)}) "
              f"({time.time()-tf0:.0f}s)", flush=True)

    summary = {}
    for k_ in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        vals = [r[k_] for r in fold_results]
        summary[k_] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    summary['fold_details'] = fold_results
    summary['dataset'] = dataset_name
    summary['setting'] = setting
    summary['seed'] = seed
    summary['total_time'] = time.time() - t0
    print(f"\n  {dataset_name} / {setting} summary:")
    for k_ in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        print(f"    {k_.upper():6s}: {summary[k_]['mean']:.4f} ± {summary[k_]['std']:.4f}")
    print(f"  Total: {summary['total_time']:.0f}s\n", flush=True)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['D1', 'D2'], required=True)
    ap.add_argument('--setting', choices=['S2', 'S3', 'S4'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default=os.path.expanduser('~/work/DrugMiR/coldstart_outputs'))
    args = ap.parse_args()
    DD = {'D1': os.path.expanduser('~/DrugMiR/data/dataset1'),
          'D2': os.path.expanduser('~/DrugMiR/data/dataset2')}[args.dataset]
    res = run_dataset_coldstart(args.dataset, DD, args.setting, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = f"{args.out_dir}/gslrda_{args.dataset}_{args.setting}_seed{args.seed}.json"
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  Saved to {out_file}", flush=True)
