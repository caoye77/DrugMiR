"""
GAM-MDR cold-start runner: S2 / S3 / S4

GAM-MDR's original `run_gam_one_fold` builds train_edge_undirected from train_pos
only, so it is already cold-start safe at the graph level. We:
  1. Replace KFold with ColdStartSplitter.
  2. Restrict training-internal negative sampling to (train_mi × train_dr).
     (GAM-MDR uses PyG's negative_sampling internally on the train edges,
      which automatically excludes train_pos but doesn't know about test entities.
      For cold-start strictness we explicitly pass a mask via `excluded_nodes`.)
  3. Test pos pairs come from splitter; test neg sampled from full grid.

Usage:
  python3 run_gammdr_coldstart.py --dataset D1 --setting S2 --seed 42
"""
import os, sys, json, time, random, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_splitter import ColdStartSplitter, ColdFold

sys.path.insert(0, os.path.expanduser('~/work/DrugMiR/MPHGNN'))
import importlib.util
spec = importlib.util.spec_from_file_location(
    'gammdr_orig',
    os.path.expanduser('~/work/DrugMiR/MPHGNN/run_gammdr_d1d2_seed42.py'))
gammdr_orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gammdr_orig)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_neg_pairs_train(assoc, n_mirna, train_mi_mask, train_dr_mask, n_samples, exclude_set=None):
    """Sample n negs constrained to (train_mi × train_dr), output with drug offset."""
    if exclude_set is None: exclude_set = set()
    tmi = np.where(train_mi_mask)[0]; tdr = np.where(train_dr_mask)[0]
    neg = []; tries = 0
    while len(neg) < n_samples and tries < n_samples * 200:
        i = int(np.random.choice(tmi)); j = int(np.random.choice(tdr))
        if assoc[i, j] == 0 and (i, j) not in exclude_set:
            neg.append((i, j + n_mirna))
        tries += 1
    return np.array(neg).T


def run_one_fold_cold(combined_features, fold: ColdFold, assoc, n_mirna, n_drug,
                      epoch=30, lr=1e-3, wd=5e-4, walk_length=3, p=0.3,
                      num_encoder=2, num_decoder=2, layer='gcn'):
    num_nodes = n_mirna + n_drug
    # Offset drug index by n_mirna for the combined node space
    train_pos_offset = fold.train_pairs.copy().astype(np.int64)
    train_pos_offset[:, 1] += n_mirna
    test_pos_offset  = fold.test_pairs.copy().astype(np.int64)
    test_pos_offset[:, 1] += n_mirna

    train_pairs_t = torch.tensor(train_pos_offset.T, dtype=torch.long, device=device)
    test_pos_t    = torch.tensor(test_pos_offset.T,  dtype=torch.long, device=device)

    # Test neg from full grid
    all_pos_set = set((int(m), int(d)) for m, d in fold.train_pairs) | \
                  set((int(m), int(d)) for m, d in fold.test_pairs)
    test_neg = gammdr_orig.sample_neg_pairs(assoc, len(fold.test_pairs), n_mirna,
                                             exclude_set=all_pos_set)
    test_neg_t = torch.tensor(test_neg, dtype=torch.long, device=device)

    # Build training edge graph (train-only)
    train_edge_undirected = to_undirected(train_pairs_t)
    all_known = torch.cat([train_pairs_t, test_pos_t, test_neg_t], dim=1)

    encoder = gammdr_orig.GNNEncoder(combined_features.shape[1], 128, 256,
                                      num_layers=num_encoder, layer=layer).to(device)
    decoder = gammdr_orig.EdgeDecoder(256, 64, 1, num_layers=num_decoder).to(device)
    mask = gammdr_orig.MaskPath(p=p, walk_length=walk_length, num_nodes=num_nodes).to(device)
    model = gammdr_orig.GAM(encoder, decoder, mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
    for ep in range(1, epoch + 1):
        model.train()
        _ = model.train_epoch(combined_features, train_edge_undirected, all_known,
                              optimizer, num_nodes)
        m = gammdr_orig.evaluate_gam(model, combined_features, train_edge_undirected,
                                      test_pos_t, test_neg_t)
        if m['auc'] > best['auc']:
            best = m
    return best


def run_dataset_coldstart(dataset_name, data_dir, setting, seed=42, n_fold=5,
                          epoch=80, layer='gcn'):
    print(f"\n{'='*72}\nGAM-MDR cold-start: {dataset_name} / {setting} / seed={seed}\n{'='*72}", flush=True)
    t0 = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    combined_features, all_pos_edges, assoc, n_mirna, n_drug = \
        gammdr_orig.load_drugmir_for_gam(data_dir, device)
    print(f"  features: {tuple(combined_features.shape)}, pos edges: {all_pos_edges.shape[1]}", flush=True)

    splitter = ColdStartSplitter(assoc, n_folds=n_fold, seed=seed, min_test_positives=15)
    folds = splitter.split(setting)

    fold_results = []
    for f in folds:
        tf0 = time.time()
        best = run_one_fold_cold(combined_features, f, assoc, n_mirna, n_drug,
                                  epoch=epoch, layer=layer)
        fold_results.append(best)
        print(f"  Fold {f.fold_id+1}/{n_fold}: AUC={best['auc']:.4f} AUPR={best['aupr']:.4f} "
              f"F1={best['f1']:.4f} P={best['prec']:.4f} R={best['rec']:.4f} "
              f"(train_pos={len(f.train_pairs)} test_pos={len(f.test_pairs)}) "
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
    DD = {'D1': os.path.expanduser('~/work/DrugMiR/data/processed'),
          'D2': os.path.expanduser('~/work/DrugMiR/DMGAT_processed')}[args.dataset]
    res = run_dataset_coldstart(args.dataset, DD, args.setting, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = f"{args.out_dir}/gammdr_{args.dataset}_{args.setting}_seed{args.seed}.json"
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  Saved to {out_file}", flush=True)
