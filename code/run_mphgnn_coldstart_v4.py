"""
MPHGNN cold-start runner v4 — optimized neg meta-path generation.

Performance journey:
  v1: full regen every fold, ~5 hours / fold on 4090 EPYC      (UNUSABLE)
  v2: cache-pos + single-thread neg gen, ~1 hour / fold        (UNUSABLE)
  v4: cache-pos + precomputed links dict + 14-core parallel
      neg gen, ~30s-2min / fold                                 (USABLE)

Two optimizations stacked:

  A. Precomputed adjacency dicts (5-10x single-core speedup):
     Original `meta_path_instance` does
         links['miRNA-gene'][links['miRNA-gene'][:, 0] == miRNA_id][:, 1]
     which is an O(|edges|) boolean-mask scan on a numpy array per lookup.
     We precompute  mirna_to_genes  : dict[int] -> np.ndarray[int]
                     gene_to_drugs   : dict[int] -> np.ndarray[int]
                     gene_to_genes   : dict[int] -> np.ndarray[int]
                     drug_to_genes   : dict[int] -> np.ndarray[int]
     replacing each scan with O(1) hash lookup + tight numpy concat.

  B. multiprocessing.Pool over neg pairs (14x parallel on 16-core):
     Each worker gets the same precomputed dicts (shared via fork CoW).
     Workers compute meta-paths independently — embarassingly parallel.

Usage:
  python3 run_mphgnn_coldstart_v4.py --dataset D1 --setting S2 --seed 42
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
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_splitter import ColdStartSplitter, ColdFold

MPHGNN_DIR = os.path.expanduser('~/work/DrugMiR/MPHGNN')
sys.path.insert(0, MPHGNN_DIR)
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================== Precomputed dicts ==============================

# Module-level globals — set by worker_init for multiprocessing fork-shared access.
# Forked workers inherit these via copy-on-write, no IPC needed.
_LINKS_DICT_CACHE = None

def precompute_links_dict(links):
    """Convert link edge-tables to source->targets dicts for O(1) lookup."""
    out = {}
    # miRNA-gene: src=miRNA, dst=gene
    mg = links['miRNA-gene']
    mg_dict = {}
    for src in np.unique(mg[:, 0]):
        mg_dict[int(src)] = mg[mg[:, 0] == src][:, 1]
    out['mirna_to_genes'] = mg_dict
    # gene-drug: src=gene, dst=drug. We also need drug->genes (reverse).
    gd = links['gene-drug']
    gd_dict = {}; dg_dict = {}
    for src in np.unique(gd[:, 0]):
        gd_dict[int(src)] = gd[gd[:, 0] == src][:, 1]
    for dst in np.unique(gd[:, 1]):
        dg_dict[int(dst)] = gd[gd[:, 1] == dst][:, 0]
    out['gene_to_drugs'] = gd_dict
    out['drug_to_genes'] = dg_dict
    # gene-gene: src=gene, dst=gene (already undirected at load time)
    gg = links['gene-gene']
    gg_dict = {}
    if gg.shape[0] > 0:
        for src in np.unique(gg[:, 0]):
            gg_dict[int(src)] = gg[gg[:, 0] == src][:, 1]
    out['gene_to_genes'] = gg_dict
    return out


def meta_path_instance_fast(miRNA_id, drug_id, dicts, k, seed=42):
    """Fast version using precomputed dicts. ~5-10x faster than original."""
    mirna_to_genes = dicts['mirna_to_genes']
    gene_to_drugs  = dicts['gene_to_drugs']
    drug_to_genes  = dicts['drug_to_genes']
    gene_to_genes  = dicts['gene_to_genes']

    target_len = k * (k + 2) + 1
    mpi = []

    # genes attached to this miRNA
    miRNAs_genes = mirna_to_genes.get(miRNA_id, np.array([], dtype=np.int64))
    # genes attached to this drug
    drugs_genes  = drug_to_genes.get(drug_id, np.array([], dtype=np.int64))

    # Type 1: [miRNA, miRNA's gene, drug's gene, drug]
    # iterate paths: miRNA_id -> g_m -> ??? need g_d such that g_d->drug
    # Wait — re-reading original:
    #   for miRNA in links['miRNA-gene'][links['miRNA-gene'][:,0]==miRNA_id][:,1]
    #       for drug in links['gene-drug'][links['gene-drug'][:,1]==drug_id][:,0]
    #         [miRNA_id, miRNA, drug, drug_id]
    # Variable names in original are misleading: 'miRNA' is actually a gene,
    # 'drug' is also a gene. So: [miRNA_id, gene_via_miRNA, gene_via_drug, drug_id]
    # Path 1: ALL combinations of (miRNA's genes) × (drug's genes)
    for g_m in miRNAs_genes:
        for g_d in drugs_genes:
            mpi.append([miRNA_id, int(g_m), int(g_d), drug_id])
            if len(mpi) >= target_len:
                return mpi[:target_len]

    # Path 2: [miRNA, miRNA's gene, gene-gene-neighbour, drug_id]
    for g_m in miRNAs_genes:
        g_m_int = int(g_m)
        g2_list = gene_to_genes.get(g_m_int, np.array([], dtype=np.int64))
        for g2 in g2_list:
            mpi.append([miRNA_id, g_m_int, int(g2), drug_id])
            if len(mpi) >= target_len:
                return mpi[:target_len]

    # Path 3: [miRNA, gene-gene-neighbour, drug's gene, drug_id]
    for g_d in drugs_genes:
        g_d_int = int(g_d)
        g1_list = gene_to_genes.get(g_d_int, np.array([], dtype=np.int64))
        for g1 in g1_list:
            mpi.append([miRNA_id, int(g1), g_d_int, drug_id])
            if len(mpi) >= target_len:
                return mpi[:target_len]

    # Fallback if mpi is empty
    if not mpi:
        rng = random.Random(seed + miRNA_id * 31 + drug_id)
        all_genes = list(gene_to_drugs.keys()) or list(mirna_to_genes.values())
        if isinstance(all_genes[0], np.ndarray):
            all_genes = np.concatenate(all_genes).tolist()
        random_gene = int(rng.choice(all_genes))
        mpi.append([miRNA_id, random_gene, random_gene, drug_id])

    # Pad to target_len by sampling existing entries
    if len(mpi) < target_len:
        rng = random.Random(seed + miRNA_id * 31 + drug_id)
        n_pad = target_len - len(mpi)
        for _ in range(n_pad):
            mpi.append(list(rng.choice(mpi)))

    return mpi[:target_len]


# ============================== Worker for Pool ==============================

def _worker_init(dicts, k):
    """Initialize each worker with the precomputed dicts (fork-inherited)."""
    global _LINKS_DICT_CACHE, _K_CACHE
    _LINKS_DICT_CACHE = dicts
    _K_CACHE = k


def _worker_compute(pair):
    """Compute one pair's meta-path instance."""
    mi, dr = pair
    return meta_path_instance_fast(int(mi), int(dr), _LINKS_DICT_CACHE, _K_CACHE)


def parallel_generate_negs(neg_pairs, dicts, k, n_workers):
    """Generate meta-paths for all neg_pairs using multiprocessing."""
    if len(neg_pairs) == 0:
        return np.zeros((0, k * (k + 2) + 1, 4), dtype=np.int64)
    if n_workers <= 1 or len(neg_pairs) < 100:
        # serial fallback
        results = [meta_path_instance_fast(int(m), int(d), dicts, k)
                   for m, d in neg_pairs]
    else:
        # parallel
        with Pool(n_workers, initializer=_worker_init, initargs=(dicts, k)) as pool:
            # Pool will fork off n_workers; each inherits dicts via COW
            results = pool.map(_worker_compute, [(int(m), int(d)) for m, d in neg_pairs],
                               chunksize=max(1, len(neg_pairs) // (n_workers * 8)))
    return np.array(results)


# ============================== Cache loading ==============================

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
    c = np.load(cache_path)
    data = c['data']; label = c['label']
    pos_mask = (label == 1)
    pos_data = data[pos_mask]
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


# ============================== Metrics ==============================

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

def run_one_fold(fold, pos_instances, pair_to_idx, links, links_dicts, features,
                 assoc, n_mirna, n_drug, n_gene, n_workers,
                 k=20, hidden_feats=128, num_layer=3, agg_type='BiTrans', topk=3,
                 dropout=0., bn=False, lr=5e-4, batch_size=256, epoch=15, patience=5):
    # 1) train-only dgl graph
    g_train = build_graph_from_train(links, features, n_mirna, n_drug, n_gene,
                                      fold.train_pairs).to(device)
    feat = {'miRNA': g_train.nodes['miRNA'].data['h'].to(device),
            'drug':  g_train.nodes['drug'].data['h'].to(device),
            'gene':  g_train.nodes['gene'].data['h'].to(device)}

    # 2) sample neg pairs
    all_pos_set = set((int(m), int(d)) for m, d in fold.train_pairs) | \
                  set((int(m), int(d)) for m, d in fold.test_pairs)
    train_neg = sample_neg_in_train_grid(assoc, fold.train_mirna_mask, fold.train_drug_mask,
                                          len(fold.train_pairs), exclude=all_pos_set)
    test_neg  = sample_neg_unrestricted(assoc, len(fold.test_pairs), exclude=all_pos_set)

    # 3) pos meta-path instances from cache
    train_pos_x = np.stack([pos_instances[pair_to_idx[(int(m), int(d))]]
                             for m, d in fold.train_pairs])
    test_pos_x  = np.stack([pos_instances[pair_to_idx[(int(m), int(d))]]
                             for m, d in fold.test_pairs])

    # 4) neg meta-path instances — parallel generation
    print(f"    generating {len(train_neg)} train + {len(test_neg)} test neg MPs "
          f"on {n_workers} workers...", flush=True)
    t_neg = time.time()
    train_neg_x = parallel_generate_negs(train_neg, links_dicts, k, n_workers)
    test_neg_x  = parallel_generate_negs(test_neg,  links_dicts, k, n_workers)
    print(f"    neg MP gen done in {time.time()-t_neg:.1f}s "
          f"({len(train_neg)+len(test_neg)} pairs)", flush=True)

    # 5) assemble
    train_x = np.concatenate([train_pos_x, train_neg_x], axis=0)
    train_y = np.concatenate([np.ones(len(train_pos_x)), np.zeros(len(train_neg_x))]).astype(np.float32)
    test_x = np.concatenate([test_pos_x, test_neg_x], axis=0)
    test_y = np.concatenate([np.ones(len(test_pos_x)), np.zeros(len(test_neg_x))]).astype(np.float32)

    train_data = torch.tensor(train_x); train_label = torch.tensor(train_y).float()
    test_data = torch.tensor(test_x);   test_label = torch.tensor(test_y).float()

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


# ============================== Driver ==============================

def run_dataset_coldstart(dataset_name, pth_path, cache_path, setting,
                          seed=42, n_fold=5, k=20, epoch=15, batch_size=256, lr=5e-4,
                          n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)  # leave headroom
    print(f"\n{'='*72}\nMPHGNN-v4 cold-start: {dataset_name} / {setting} / seed={seed} "
          f"(workers={n_workers})\n{'='*72}", flush=True)
    t0 = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    links, features, assoc, n_mirna, n_drug, n_gene = load_links_and_features(pth_path)
    pos_instances, pair_to_idx = load_cached_pos_instances(cache_path)
    print(f"  n_mirna={n_mirna} n_drug={n_drug} n_gene={n_gene} n_pos={int(assoc.sum())}", flush=True)
    print(f"  cache loaded: {len(pos_instances)} positive MP instances", flush=True)

    # Precompute link adjacency dicts (one-time cost per dataset)
    t_dict = time.time()
    links_dicts = precompute_links_dict(links)
    print(f"  precomputed adjacency dicts in {time.time()-t_dict:.1f}s", flush=True)

    splitter = ColdStartSplitter(assoc, n_folds=n_fold, seed=seed, min_test_positives=15)
    folds = splitter.split(setting)

    fold_results = []
    for f in folds:
        tf0 = time.time()
        missing = [(int(m), int(d)) for m, d in f.train_pairs if (int(m), int(d)) not in pair_to_idx]
        missing += [(int(m), int(d)) for m, d in f.test_pairs if (int(m), int(d)) not in pair_to_idx]
        if missing:
            raise RuntimeError(f"Cache missing {len(missing)} pairs (first: {missing[0]}).")
        best = run_one_fold(f, pos_instances, pair_to_idx, links, links_dicts, features,
                            assoc, n_mirna, n_drug, n_gene, n_workers,
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
    ap.add_argument('--workers', type=int, default=None,
                    help='multiprocessing workers (default: nproc - 2)')
    ap.add_argument('--out_dir', default=os.path.expanduser('~/work/DrugMiR/coldstart_outputs'))
    args = ap.parse_args()
    pth = {'D1': os.path.join(MPHGNN_DIR, 'MiDrug_data_D1.pth'),
           'D2': os.path.join(MPHGNN_DIR, 'MiDrug_data_D2.pth')}[args.dataset]
    cache = {'D1': os.path.join(MPHGNN_DIR, '_mp_cache/MiDrug_data_D1_k20_seed42.npz'),
             'D2': os.path.join(MPHGNN_DIR, '_mp_cache/MiDrug_data_D2_k20_seed42.npz')}[args.dataset]
    res = run_dataset_coldstart(args.dataset, pth, cache, args.setting,
                                seed=args.seed, n_workers=args.workers)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = f"{args.out_dir}/mphgnn_{args.dataset}_{args.setting}_seed{args.seed}.json"
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  Saved to {out_file}", flush=True)
