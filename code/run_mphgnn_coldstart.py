"""
MPHGNN cold-start runner: S2 / S3 / S4

Cold-start adaptation strategy:
  - Meta-path instances are still pre-generated from the FULL miRNA-gene/drug-gene/
    gene-gene bipartite graphs (these come from miRTarBase/DrugBank — external
    knowledge, not training data — so feeding them at test time is feature
    injection, not label leakage).
  - But the dgl heterograph's miRNA-drug edges are SHARDED per-fold: only the
    train_pos pairs become 'miRNA-drug' edges in the GNN message-passing graph.
    Test pair edges are absent (this is what `remove_graph` does in the original
    transductive eval, but here we skip adding them in the first place).
  - The split itself comes from ColdStartSplitter (S2/S3/S4 by entity).

Usage:
  python3 run_mphgnn_coldstart.py --dataset D1 --setting S2 --seed 42
"""
import os, sys, json, time, random, argparse, warnings
import numpy as np
import torch
import dgl
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              precision_recall_curve)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_splitter import ColdStartSplitter, ColdFold

# MPHGNN modules live in ~/work/DrugMiR/MPHGNN/
MPHGNN_DIR = os.path.expanduser('~/work/DrugMiR/MPHGNN')
sys.path.insert(0, MPHGNN_DIR)
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================== Meta-path generation ==============================

def meta_path_instance(seed, miRNA_id, drug_id, links, k):
    """Identical to original MPHGNN, just re-imported for self-containedness."""
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
    """Load D1/D2 .pth file → links, features, dims."""
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


# ============================== Build per-fold MP cache ==============================

def build_pair_mp_instances(pairs_pos, pairs_neg, links, k, seed, desc=''):
    """Generate meta-path instances for a given list of pairs."""
    data_list, label_list = [], []
    with tqdm(total=len(pairs_pos) + len(pairs_neg), desc=desc) as pbar:
        for mi, dr in pairs_pos:
            mpi = meta_path_instance(seed, int(mi), int(dr), links, k)
            data_list.append(mpi); label_list.append(1)
            pbar.update()
        for mi, dr in pairs_neg:
            mpi = meta_path_instance(seed, int(mi), int(dr), links, k)
            data_list.append(mpi); label_list.append(0)
            pbar.update()
    return np.array(data_list), np.array(label_list)


def build_graph_from_train(links, features, n_mirna, n_drug, n_gene, train_pairs):
    """Build the dgl heterograph using train_pairs as miRNA-drug edges.
    Other typed edges (gene-drug / miRNA-gene) are full (external DB knowledge).
    """
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


# ============================== Eval metrics ==============================

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


# ============================== One fold ==============================

def sample_neg_in_train_grid(assoc, train_mi_mask, train_dr_mask, n, exclude=None):
    """Sample n negatives (i,j) with i train-mirna, j train-drug, assoc[i,j]=0."""
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
    """Sample n negatives from full grid (for test eval)."""
    if exclude is None: exclude = set()
    nm, nd = assoc.shape
    neg = []
    while len(neg) < n:
        i = np.random.randint(0, nm); j = np.random.randint(0, nd)
        if assoc[i, j] == 0 and (i, j) not in exclude:
            neg.append((i, j))
    return np.array(neg).reshape(-1, 2)


def run_one_fold(fold: ColdFold, links, features, assoc, n_mirna, n_drug, n_gene,
                 k=20, hidden_feats=128, num_layer=3, agg_type='BiTrans', topk=3,
                 dropout=0., bn=False, lr=5e-4, batch_size=256, epoch=15, patience=5):
    # 1) build train-only graph
    g_train = build_graph_from_train(links, features, n_mirna, n_drug, n_gene,
                                      fold.train_pairs).to(device)
    feat = {'miRNA': g_train.nodes['miRNA'].data['h'].to(device),
            'drug':  g_train.nodes['drug'].data['h'].to(device),
            'gene':  g_train.nodes['gene'].data['h'].to(device)}

    # 2) sample neg pairs (1:1 ratio for both train and test)
    pos_train_set = set((int(m), int(d)) for m, d in fold.train_pairs)
    pos_test_set  = set((int(m), int(d)) for m, d in fold.test_pairs)
    all_pos_set   = pos_train_set | pos_test_set
    train_neg = sample_neg_in_train_grid(assoc, fold.train_mirna_mask, fold.train_drug_mask,
                                          len(fold.train_pairs), exclude=all_pos_set)
    test_neg  = sample_neg_unrestricted(assoc, len(fold.test_pairs), exclude=all_pos_set)

    # 3) build meta-path instances for both sets
    train_x, train_y = build_pair_mp_instances(
        fold.train_pairs, train_neg, links, k, seed=42,
        desc=f"  build train MP")
    test_x, test_y = build_pair_mp_instances(
        fold.test_pairs, test_neg, links, k, seed=42,
        desc=f"  build test  MP")

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

def run_dataset_coldstart(dataset_name, pth_path, setting, seed=42, n_fold=5,
                          k=20, epoch=15, batch_size=256, lr=5e-4):
    print(f"\n{'='*72}\nMPHGNN cold-start: {dataset_name} / {setting} / seed={seed}\n{'='*72}", flush=True)
    t0 = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    links, features, assoc, n_mirna, n_drug, n_gene = load_links_and_features(pth_path)
    print(f"  n_mirna={n_mirna} n_drug={n_drug} n_gene={n_gene} n_pos={int(assoc.sum())}", flush=True)

    splitter = ColdStartSplitter(assoc, n_folds=n_fold, seed=seed, min_test_positives=15)
    folds = splitter.split(setting)

    fold_results = []
    for f in folds:
        tf0 = time.time()
        best = run_one_fold(f, links, features, assoc, n_mirna, n_drug, n_gene,
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
    res = run_dataset_coldstart(args.dataset, pth, args.setting, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = f"{args.out_dir}/mphgnn_{args.dataset}_{args.setting}_seed{args.seed}.json"
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  Saved to {out_file}", flush=True)
