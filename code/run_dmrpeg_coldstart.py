"""
DMR-PEG cold-start runner: S2 / S3 / S4

DMR-PEG's original `run_dmrpeg_one_fold` already builds the association graph
from train_pos only (line 1038 of run_dmrpeg_d1d2_seed42.py:
    train_assoc = np.zeros_like(assoc)
    train_assoc[train_pos[:, 0], train_pos[:, 1]] = 1
), so it is intrinsically cold-start safe at the GRAPH level. We just need to:
  1. Replace KFold.split(pos_pairs) with ColdStartSplitter.split('S2'/'S3'/'S4').
  2. Restrict the training negative sampling pool to (train_mi × train_dr).
  3. Test negatives are sampled from the full grid (standard evaluation).

Usage:
  python3 run_dmrpeg_coldstart.py --dataset D1 --setting S2 --seed 42
"""
import os, sys, json, time, random, argparse, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_splitter import ColdStartSplitter, ColdFold

# Re-use original DMR-PEG components
sys.path.insert(0, os.path.expanduser('~/work/DrugMiR/MPHGNN'))
import importlib.util
spec = importlib.util.spec_from_file_location(
    'dmrpeg_orig',
    os.path.expanduser('~/work/DrugMiR/MPHGNN/run_dmrpeg_d1d2_seed42.py'))
dmrpeg_orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dmrpeg_orig)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_neg_train(assoc, train_mi_mask, train_dr_mask, n, seen=None):
    """Sample n negatives constrained to train_mi × train_dr."""
    if seen is None: seen = set()
    tmi = np.where(train_mi_mask)[0]; tdr = np.where(train_dr_mask)[0]
    neg = []; tries = 0
    while len(neg) < n and tries < n * 200:
        i = int(np.random.choice(tmi)); j = int(np.random.choice(tdr))
        if assoc[i, j] == 0 and (i, j) not in seen:
            neg.append((i, j))
        tries += 1
    return np.array(neg).reshape(-1, 2)


def run_one_fold_cold(data, fold: ColdFold, num_epochs=20, lr=1e-4, batch_size=256, hidden=128):
    """Adapted from dmrpeg_orig.run_dmrpeg_one_fold with cold-start neg sampling."""
    assoc = data['assoc']
    n_mirna, n_drug = data['n_mirna'], data['n_drug']
    mi_feat = data['mi_feat']; drug_feat = data['drug_feat']; mol_batch = data['mol_batch']

    train_pos = fold.train_pairs.astype(np.int64)
    test_pos  = fold.test_pairs.astype(np.int64)

    # Build assoc graph from train_pos only
    train_assoc = np.zeros_like(assoc)
    train_assoc[train_pos[:, 0], train_pos[:, 1]] = 1
    asso_x, asso_ei = dmrpeg_orig.build_assoc_graph_inputs(train_assoc, device)

    # Sample negatives (1:1) — train neg from train grid only
    pos_set = set(map(tuple, train_pos.tolist())) | set(map(tuple, test_pos.tolist()))
    train_neg = sample_neg_train(assoc, fold.train_mirna_mask, fold.train_drug_mask,
                                  len(train_pos), seen=pos_set)
    train_pairs = np.concatenate([train_pos, train_neg], axis=0)
    train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])

    # Test neg from full grid (standard eval)
    test_neg = dmrpeg_orig.sample_neg(assoc, len(test_pos), seen=pos_set)
    test_pairs = np.concatenate([test_pos, test_neg], axis=0)
    test_labels = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

    train_pairs_t = torch.tensor(train_pairs, dtype=torch.long, device=device)
    train_labels_t = torch.tensor(train_labels, dtype=torch.float32, device=device)
    test_pairs_t = torch.tensor(test_pairs, dtype=torch.long, device=device)

    model = dmrpeg_orig.DMRPEGModel(n_mirna, n_drug, mi_feat.shape[1], drug_feat.shape[1],
                                     hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
    n_train = len(train_pairs)
    for ep in range(num_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_pairs = train_pairs_t[idx]
            batch_labels = train_labels_t[idx]
            optimizer.zero_grad()
            logits = model(mol_batch, asso_x, asso_ei, mi_feat, drug_feat, batch_pairs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            all_scores = []
            for i in range(0, len(test_pairs), 512):
                batch = test_pairs_t[i:i + 512]
                logits = model(mol_batch, asso_x, asso_ei, mi_feat, drug_feat, batch)
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
            y_score = np.concatenate(all_scores)
            m = dmrpeg_orig.get_metrics_5(test_labels, y_score)
            if m['auc'] > best['auc']:
                best = m
    return best


def run_dataset_coldstart(dataset_name, data_dir, setting, seed=42, n_fold=5,
                           num_epochs=20):
    print(f"\n{'='*72}\nDMR-PEG cold-start: {dataset_name} / {setting} / seed={seed}\n{'='*72}", flush=True)
    t0 = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    data = dmrpeg_orig.load_drugmir_for_dmrpeg(data_dir, dataset_name, device)
    print(f"  n_mirna={data['n_mirna']} n_drug={data['n_drug']} pos={int(data['assoc'].sum())}", flush=True)

    splitter = ColdStartSplitter(data['assoc'], n_folds=n_fold, seed=seed, min_test_positives=15)
    folds = splitter.split(setting)

    fold_results = []
    for f in folds:
        tf0 = time.time()
        best = run_one_fold_cold(data, f, num_epochs=num_epochs)
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
    out_file = f"{args.out_dir}/dmrpeg_{args.dataset}_{args.setting}_seed{args.seed}.json"
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  Saved to {out_file}", flush=True)
