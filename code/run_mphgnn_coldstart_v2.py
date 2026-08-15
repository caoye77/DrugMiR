"""
MPHGNN cold-start runner v2 — uses cached positive meta-path instances and
only generates negative instances per-fold (the only fold-dependent part).

CRITICAL DIFFERENCE vs v1 (run_mphgnn_coldstart.py):
  v1 regenerated ALL meta-paths every fold (~5h per fold).
  v2 reuses cached pos instances + generates only neg (~1-2 min per fold).

The pos meta-path cache layout (verified):
  MPHGNN/_mp_cache/MiDrug_data_{D1,D2}_k20_seed42.npz
    'data':  (17440, 441, 4)  — 8720 pos + 8720 neg
    'label': (17440,)         — 1s first, 0s last
  Each instance[0][0]  = miRNA id
  Each instance[0][-1] = drug id

We reverse-map each pos cache row to its (miRNA, drug) pair, then for each
cold-start fold we:
  1. Subset positives by fold.train_pairs and fold.test_pairs
  2. Generate per-fold negatives (1:1 ratio) in (train_mi × train_dr) grid
  3. Build train-only dgl heterograph for GNN message passing
  4. Train and evaluate exactly like v1

Usage:
  python3 run_mphgnn_coldstart_v2.py --dataset D1 --setting S2 --seed 42
"""
import os, sys, json, time, random, argparse, warnings
import numpy as np
import torch
import dgl
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import to_undirected
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              precision_recall_curve)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_splitter import ColdStartSplitter, ColdFold

MPHGNN_DIR = os.path.expanduser('~/work/DrugMiR/MPHGNN')
sys.path.insert(0, MPHGNN_DIR)
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================== Meta-path generation (only for negs) ==============================

def meta_path_instance(seed, miRNA_id, drug_id, links, k):
    """Identical to original MPHGNN — used only to generate cold-start negatives."""
    mpi = []
    mpi.extend([[miRNA_id, miRNA, drug, drug_id]
                for miRNA in links['miRNA-gene'][links['miRNA-gene'][:, 0] == miRNA_id][:, 1]
                for drug in links['gene-drug'][links['gene-drug'][:, 1] == drug_id][:, 0]])
    mpi.extend([[miRNA_id, gene1, gene2, drug_id]
                for gene1 in links['miRNA-gene'][links['miRNA-gene'][:, 0] == miRNA_id][:, 1]
                for gene2 in links['gene-gene'][links['gene-gene'][:, 0] == gene1][:, 1]])
    mpi.extend([[miRNA_id, gene1, gene2, drug_id]
                for gene2 in links['gene-drug'][links['gene-drug'][:, 1] == drug_id][:, 0]
                for gene1 in links['gene-gene'][links['gene-gene'][:, 0] == gene2][:, 1]])
    if not mpi:
        all_genes = np.unique(links['gene-drug'][:, 0])
        if len(all_genes) > 0:
            random_gene = int(np.random.choice(all_genes))
            mpi.append([miRNA_id, random_gene, random_gene, drug_id])
        else:
            all_genes = np.unique(links['miRNA-gene'][:, 1])
            random_gene = int(np.random.choice(all_genes))
            mpi.append([miRNA_id, random_gene, random_gene, drug_id])
    target_len = k * (k + 2) + 1
    if len(mpi) < target_len:
        for _ in range(target_len - len(mpi)):
            random.seed(seed)
            mpi.append(random.choice(mpi))
    elif len(mpi) > target_len:
        mpi = mpi[:target_len]
    return mpi


def load_links_and_features(pth_path):
    d = torch.load(pth_path, weights_only=False)
    n_mirna = d['miRNA'].num_nodes
    n_drug = d['drug'].num_nodes
    n_gene = d['gene'].num_nodes
    midrug_ei = d['miRNA', 'MiDrug', 'drug'].edge_index.cpu().numpy().T
    migene_ei = d['miRNA', 'Migene', 'gene'].edge_index.cpu().numpy().T
    genedrug_ei = d['gene', 'Genedrug', 'drug'].edge_index.cpu().numpy().T
    gegen_ei = d['gene', 'GeGe', 'gene'].edge_index.cpu().numpy().T
    gegen_t = (to_undirected(torch.tensor(gegen_ei.T)).T.numpy()
               if gegen_ei.shape[0] > 0 else gegen_ei)
    links = {
        'miRNA-drug': midrug_ei, 'miRNA-gene': migene_ei,
        'gene-drug': genedrug_ei, 'gene-gene': gegen_t,
    }
    features = {
        'miRNA': d['miRNA'].x.float(),
        'drug':  d['drug'].x.float(),
        'gene':  d['gene'].x.float(),
    }
    assoc = np.zeros((n_mirna, n_drug), dtype=np.float32)
    assoc[midrug_ei[:, 0], midrug_ei[:, 1]] = 1
    return links, features, assoc, n_mirna, n_drug, n_gene


def load_cached_pos_instances(cache_path):
    """Load cached meta-path instances; return (pos_instances, pair_to_idx).
    pair_to_idx[(mirna, drug)] = row index in pos_instances.
    """
    c = np.load(cache_path)
    data = c['data']         # (17440, 441, 4)
    label = c['label']       # (17440,)
    pos_mask = (label == 1)
    pos_data = data[pos_mask]  # (n_pos, 441, 4)
    # Build (mirna, drug) → row_idx mapping
    pair_to_idx = {}
    for i, inst in enumerate(pos_data):
        mi = int(inst[0][0])
        dr = int(inst[0][-1])
        pair_to_idx[(mi, dr)] = i
    return pos_data, pair_to_idx


def build_graph_from_train(links, features, n_mirna, n_drug, n_gene, train_pairs):
    train_pairs_np = train_pairs.astype(np.int64)
    drug_mirna_ei = train_pairs_np[:, [1, 0]]
    graph_data = {
        ('gene', 'gene-drug', 'drug'): (torch.tensor(links['gene-drug'][:, 0]),
                                         torch.tensor(links['gene-drug'][:, 1])),
        ('miRNA', 'miRNA-drug', 'drug'): (torch.tensor(train_pairs_np[:, 0]),
                                           torch.tensor(train_pairs_np[:, 1])),
        ('drug', 'drug-miRNA', 'miRNA'): (torch.tensor(drug_mirna_ei[:, 0]),
                                           torch.tensor(drug_mirna_ei[:, 1])),
        ('miRNA', 'miRNA-gene', 'gene'): (torch.tensor(links['miRNA-gene'][:, 0]),
                                           torch.tensor(links['miRNA-gene'][:, 1])),
    }
    g = dgl.heterograph(graph_data, num_nodes_dict={
        'miRNA': n_mirna, 'drug': n_drug, 'gene': n_gene
    })
    g.nodes['miRNA'].data['h'] = features['miRNA']
    g.nodes['drug'].data['h']  = features['drug']
    g.nodes['gene'].data['h']  = features['gene']
    return g


# ============================== Eval ==============================

def get_metrics_5(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    p, r, t = precision_recall_curve(y_true, y_score)
    f1c = 2 * p * r / (p + r + 1e-10)
    bi = int(np.argmax(f1c[:-1]))
    thr = float(t[bi])
    pb = (y_score >= thr).astype(int)
    return {'auc': float(auc), 'aupr': float(aupr), 'f1': float(f1_score(y_true, pb)),
            'prec': float(precision_score(y_true, pb, zero_division=0)),
            'rec': float(recall_score(y_true, pb)), 'thr': thr}


# ============================== Neg sampling ==============================

def sample_neg_in_train_grid(assoc, train_mi_mask, train_dr_mask, n, exclude=None):
    if exclude is None: exclude = set()
    tmi = np.where(train_mi_mask)[0]; tdr = np.where(train_dr_mask)[0]
    neg = []; tries = 0
    while len(neg) < n and tries < n * 200:
        i = int(np.random.choice(tmi)); j = int(np.random.choice(tdr))
        if assoc[i, j] == 0 and (i, j) not in exclude:
            neg.append((i, j))
        tries += 1
    return np.array(neg).reshape(-1, 2)


def sample_neg_unrestricted(assoc, n, exclude=None):
    if exclude is None: exclude = set()
    nm, nd = assoc.shape
    neg = []
    while len(neg) < n:
        i = np.random.randint(0, nm); j = np.random.randint(0, nd)
        if assoc[i, j] == 0 and (i, j) not in exclude:
            neg.append((i, j))
    return np.array(neg).reshape(-1, 2)


# ============================== One fold ==============================

def run_one_fold(fold: ColdFold, pos_instances, pair_to_idx, links, features,
                 assoc, n_mirna, n_drug, n_gene,
                 k=20, hidden_feats=128, num_layer=3, agg_type='BiTrans', topk=3,
                 dropout=0., bn=False, lr=5e-4, batch_size=256, epoch=15, patience=5):
    # 1) build train-only graph
    g_train = build_graph_from_train(links, features, n_mirna, n_drug, n_gene,
                                      fold.train_pairs).to(device)
    feat = {'miRNA': g_train.nodes['miRNA'].data['h'].to(device),
            'drug':  g_train.nodes['drug'].data['h'].to(device),
            'gene':  g_train.nodes['gene'].data['h'].to(device)}

    # 2) sample neg pairs (1:1 with pos)
    pos_train_set = set((int(m), int(d)) for m, d in fold.train_pairs)
    pos_test_set  = set((int(m), int(d)) for m, d in fold.test_pairs)
    all_pos_set   = pos_train_set | pos_test_set
    train_neg = sample_neg_in_train_grid(assoc, fold.train_mirna_mask, fold.train_drug_mask,
                                          len(fold.train_pairs), exclude=all_pos_set)
    test_neg  = sample_neg_unrestricted(assoc, len(fold.test_pairs), exclude=all_pos_set)

    # 3) Build pos meta-path instances from cache (fast: just index)
    train_pos_x = np.stack([pos_instances[pair_to_idx[(int(m), int(d))]]
                             for m, d in fold.train_pairs])
    test_pos_x  = np.stack([pos_instances[pair_to_idx[(int(m), int(d))]]
                             for m, d in fold.test_pairs])

    # 4) Generate neg meta-path instances (the slow but necessary part)
    print(f"    generating {len(train_neg)} train negs + {len(test_neg)} test negs MP...",
          flush=True)
    t_neg = time.time()
    train_neg_x_list = []
    for mi, dr in train_neg:
        train_neg_x_list.append(meta_path_instance(42, int(mi), int(dr), links, k))
    train_neg_x = np.array(train_neg_x_list)
    test_neg_x_list = []
    for mi, dr in test_neg:
        test_neg_x_list.append(meta_path_instance(42, int(mi), int(dr), links, k))
    test_neg_x = np.array(test_neg_x_list)
    print(f"    neg MP generated in {time.time()-t_neg:.1f}s", flush=True)

    # 5) Assemble
    train_x = np.concatenate([train_pos_x, train_neg_x], axis=0)
    train_y = np.concatenate([np.ones(len(train_pos_x)), np.zeros(len(train_neg_x))]).astype(np.float32)
    test_x = np.concatenate([test_pos_x, test_neg_x], axis=0)
    test_y = np.concatenate([np.ones(len(test_pos_x)), np.zeros(len(test_neg_x))]).astype(np.float32)

    train_data = torch.tensor(train_x)
    train_label = torch.tensor(train_y).float()
    test_data = torch.tensor(test_x)
    test_label = torch.tensor(test_y).float()

    train_loader = DataLoader(TensorDataset(train_data, train_label),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_data, test_label),
                             batch_size=batch_size, shuffle=False)

    model = Model(g_train.etypes,
                  {'miRNA': feat['miRNA'].shape[1],
                   'drug':  feat['drug'].shape[1],
                   'gene':  feat['gene'].shape[1]},
                  hidden_feats=hidden_feats, num_emb_layers=num_layer,
                  agg_type=agg_type, dropout=dropout, bn=bn, k=topk).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    pos_w = (train_label == 0).sum().item() / max(1, (train_label == 1).sum().item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))

    best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
    pc = 0
    for ep in range(1, epoch + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(g_train, feat, x).squeeze(dim=1)
            loss = criterion(pred, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        all_pred = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                p = torch.sigmoid(model(g_train, feat, x).squeeze(dim=1))
                all_pred.append(p.cpu().numpy())
        y_score = np.concatenate(all_pred)
        m = get_metrics_5(test_label.cpu().numpy(), y_score)
        if m['auc'] > best['auc']:
            best = m; pc = 0
        else:
            pc += 1
        if pc >= patience:
            break
    return best


# ============================== Dataset driver ==============================

def run_dataset_coldstart(dataset_name, pth_path, cache_path, setting,
                          seed=42, n_fold=5, k=20, epoch=15, batch_size=256, lr=5e-4):
    print(f"\n{'='*72}\nMPHGNN-v2 cold-start: {dataset_name} / {setting} / seed={seed}\n{'='*72}", flush=True)
    t0 = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    links, features, assoc, n_mirna, n_drug, n_gene = load_links_and_features(pth_path)
    pos_instances, pair_to_idx = load_cached_pos_instances(cache_path)
    print(f"  n_mirna={n_mirna} n_drug={n_drug} n_gene={n_gene} n_pos={int(assoc.sum())}", flush=True)
    print(f"  cache loaded: {len(pos_instances)} positive MP instances", flush=True)

    splitter = ColdStartSplitter(assoc, n_folds=n_fold, seed=seed, min_test_positives=15)
    folds = splitter.split(setting)

    fold_results = []
    for f in folds:
        tf0 = time.time()
        # verify cache covers this fold's pairs
        missing = [(int(m), int(d)) for m, d in f.train_pairs if (int(m), int(d)) not in pair_to_idx]
        missing += [(int(m), int(d)) for m, d in f.test_pairs if (int(m), int(d)) not in pair_to_idx]
        if missing:
            raise RuntimeError(f"Cache missing {len(missing)} pairs (first: {missing[0]}). "
                               f"Cache may be from a different positive set.")
        best = run_one_fold(f, pos_instances, pair_to_idx, links, features,
                            assoc, n_mirna, n_drug, n_gene,
                            k=k, epoch=epoch, batch_size=batch_size, lr=lr)
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
    print(f"\n  {dataset_name} / {setting} summary (mean±std):")
    for k_ in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        print(f"    {k_.upper():6s}: {summary[k_]['mean']:.4f} ± {summary[k_]['std']:.4f}")
    print(f"  Total: {summary['total_time']:.0f}s ({summary['total_time']/60:.1f} min)\n", flush=True)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['D1', 'D2'], required=True)
    ap.add_argument('--setting', choices=['S2', 'S3', 'S4'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default=os.path.expanduser('~/work/DrugMiR/coldstart_outputs'))
    args = ap.parse_args()
    pth = {'D1': os.path.join(MPHGNN_DIR, 'MiDrug_data_D1.pth'),
           'D2': os.path.join(MPHGNN_DIR, 'MiDrug_data_D2.pth')}[args.dataset]
    cache = {'D1': os.path.join(MPHGNN_DIR, '_mp_cache/MiDrug_data_D1_k20_seed42.npz'),
             'D2': os.path.join(MPHGNN_DIR, '_mp_cache/MiDrug_data_D2_k20_seed42.npz')}[args.dataset]
    res = run_dataset_coldstart(args.dataset, pth, cache, args.setting, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = f"{args.out_dir}/mphgnn_{args.dataset}_{args.setting}_seed{args.seed}.json"
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  Saved to {out_file}", flush=True)
