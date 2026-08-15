"""
DMGAT cold-start runner: S2 / S3 / S4

DMGAT (simplified) has feature inputs (k-mer + ChemBERTa) plus similarity graphs,
so it can score test entities under cold-start by reading their features and
running them through the encoder + GAT pipeline. The only fold-dependent thing
is the `adj_train` matrix (miRNA-drug bipartite, used in both the loss mask and
in `adj_full` block matrix passed to GAT).

Cold-start adaptation:
  1. Cold-start split via ColdStartSplitter (S2/S3/S4).
  2. adj_train built from fold.train_pairs only — test pair edges absent.
  3. Test negatives sampled from full grid (standard eval).
  4. Train neg sampling restricted to (train_mi × train_dr) grid.
  5. Similarity graphs mi_sim / drug_sim are loaded as-is (they're pre-computed
     biological similarities — external data, not training labels — same logic
     as the gene-bridge edges in DrugMiR).

Usage:
  python3 run_dmgat_coldstart.py --dataset D1 --setting S2 --seed 42
"""
import os, sys, json, time, random, argparse, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_splitter import ColdStartSplitter, ColdFold

# Import original DMGAT components
sys.path.insert(0, os.path.expanduser('~/work/DrugMiR/MPHGNN'))
import importlib.util
spec = importlib.util.spec_from_file_location(
    'dmgat_orig',
    os.path.expanduser('~/work/DrugMiR/MPHGNN/run_dmgat_d1d2_seed42.py'))
dmgat_orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dmgat_orig)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_neg_train(assoc, train_mi_mask, train_dr_mask, n, seen=None):
    if seen is None: seen = set()
    tmi = np.where(train_mi_mask)[0]; tdr = np.where(train_dr_mask)[0]
    neg = []; tries = 0
    while len(neg) < n and tries < n * 200:
        i = int(np.random.choice(tmi)); j = int(np.random.choice(tdr))
        if assoc[i, j] == 0 and (i, j) not in seen:
            neg.append((i, j))
        tries += 1
    return neg


def run_one_fold_cold(data, fold: ColdFold, device, n_gcn=2, n_gat=2,
                     hidden=256, num_epochs=200, lr=5e-3, dropout=0.1):
    assoc = data['assoc']
    n_mirna, n_drug = data['n_mirna'], data['n_drug']
    mi_feat, drug_feat = data['mi_feat'], data['drug_feat']
    mi_sim, drug_sim = data['mi_sim'], data['drug_sim']

    train_pos = fold.train_pairs.astype(np.int64)
    test_pos  = fold.test_pairs.astype(np.int64)

    # adj_train from train_pairs only
    adj_train_np = np.zeros((n_mirna, n_drug), dtype=np.float32)
    for (m, d) in train_pos:
        adj_train_np[m, d] = 1.0

    # Train neg restricted to train grid
    pos_set = set(map(tuple, train_pos.tolist())) | set(map(tuple, test_pos.tolist()))
    train_neg = sample_neg_train(assoc, fold.train_mirna_mask, fold.train_drug_mask,
                                  len(train_pos), seen=pos_set)
    train_mask_np = adj_train_np.copy()
    for (m, d) in train_neg:
        train_mask_np[m, d] = 1.0

    adj_train = torch.tensor(adj_train_np, device=device)
    train_mask = torch.tensor(train_mask_np, device=device)

    # adj_full = [[I_rna, A], [A.T, I_drug]]
    n_total = n_mirna + n_drug
    adj_full_np = np.zeros((n_total, n_total), dtype=np.float32)
    adj_full_np[:n_mirna, :n_mirna] = np.eye(n_mirna)
    adj_full_np[n_mirna:, n_mirna:] = np.eye(n_drug)
    adj_full_np[:n_mirna, n_mirna:] = adj_train_np
    adj_full_np[n_mirna:, :n_mirna] = adj_train_np.T
    adj_full = torch.tensor(adj_full_np, device=device)

    # Build model (re-use original classes)
    linear = dmgat_orig.DMGATLinear(mi_feat.shape[1], drug_feat.shape[1], hidden).to(device)
    r_gcn_list = [dmgat_orig.GCN(hidden, hidden, mi_sim).to(device) for _ in range(n_gcn)]
    d_gcn_list = [dmgat_orig.GCN(hidden, hidden, drug_sim).to(device) for _ in range(n_gcn)]
    gat_list = [dmgat_orig.GAT(hidden, hidden, hidden, n_total, dropout,
                                alpha=0.1, nheads=2, device=device).to(device)
                for _ in range(n_gat)]
    predictor = dmgat_orig.Predictor(hidden, hidden).to(device)
    model = dmgat_orig.DMGATModel(linear, r_gcn_list, d_gcn_list, gat_list, predictor).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = dmgat_orig.MaskedBCELoss()

    # Test neg from full grid (standard eval)
    test_neg_list = dmgat_orig.sample_neg(assoc, len(test_pos), seen=pos_set)
    test_neg = np.array(test_neg_list)

    best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}

    for ep in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        new_p, new_d = model(mi_feat, drug_feat, mi_sim, drug_sim, adj_full)
        loss = loss_fn(new_p, new_d, adj_train, train_mask)
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            new_p, new_d = model(mi_feat, drug_feat, mi_sim, drug_sim, adj_full)
            pred = torch.sigmoid(new_p @ new_d.T).cpu().numpy()
            pos_scores = pred[test_pos[:, 0], test_pos[:, 1]]
            neg_scores = pred[test_neg[:, 0], test_neg[:, 1]]
            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
            y_score = np.concatenate([pos_scores, neg_scores])
            m = dmgat_orig.get_metrics_5(y_true, y_score)
            if m['auc'] > best['auc']:
                best = m
    return best


def run_dataset_coldstart(dataset_name, data_dir, setting, seed=42, n_fold=5,
                          num_epochs=200, n_gcn=2, n_gat=2, hidden=256):
    print(f"\n{'='*72}\nDMGAT cold-start: {dataset_name} / {setting} / seed={seed}\n{'='*72}", flush=True)
    t0 = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    data = dmgat_orig.load_drugmir_for_dmgat(data_dir, device)
    print(f"  n_mirna={data['n_mirna']} n_drug={data['n_drug']} pos={int(data['assoc'].sum())}", flush=True)

    splitter = ColdStartSplitter(data['assoc'], n_folds=n_fold, seed=seed, min_test_positives=15)
    folds = splitter.split(setting)

    fold_results = []
    for f in folds:
        tf0 = time.time()
        best = run_one_fold_cold(data, f, device, n_gcn=n_gcn, n_gat=n_gat,
                                  hidden=hidden, num_epochs=num_epochs)
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
    out_file = f"{args.out_dir}/dmgat_{args.dataset}_{args.setting}_seed{args.seed}.json"
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  Saved to {out_file}", flush=True)
