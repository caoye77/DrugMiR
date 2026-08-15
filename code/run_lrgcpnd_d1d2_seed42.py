"""
Phase 2b: LRGCPND benchmark on D1 + D2
- Reuses LRGCPND model.py and utils.py untouched
- Uses DrugMiR's 5-fold split + per-epoch negative resampling protocol  
- Uses DrugMiR's ev_full() (5 metrics at optimal F1 threshold)
- Output: phase2_outputs/lrgcpnd_5metrics_seed42.json
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             precision_recall_curve)
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# Reuse LRGCPND code as-is
sys.path.insert(0, os.path.expanduser('~/work/DrugMiR/LRGCPND/code'))
from model import LRGCPND
from utils import construct_sparse_mx, normalize, sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp

# ==================== Data loading (matches DrugMiR协议) ====================
def load_assoc(dd):
    """Load only what LRGCPND needs: association matrix → positive pairs."""
    assoc = np.load(f"{dd}/association_matrix.npy")
    pr, pc = np.nonzero(assoc)
    pos_pairs = list(zip(pr.tolist(), pc.tolist()))
    return assoc, pos_pairs, assoc.shape[0], assoc.shape[1]

def sample_neg(assoc, n_samples):
    """Random negative pair sampling — same as DrugMiR's sn()."""
    nm, nd = assoc.shape
    neg = []
    while len(neg) < n_samples:
        i = np.random.randint(0, nm)
        j = np.random.randint(0, nd)
        if assoc[i, j] == 0:
            neg.append((i, j))
    return neg

# ==================== Wrap LRGCPND for our protocol ====================

def build_adj_from_train_pairs(train_pos, n_num, d_num):
    """Build LRGCPND's adjacency matrix from train positive pairs only.
    Critical: this excludes test positives → no info leakage across folds."""
    train_pos_arr = np.array(train_pos)  # shape (N_pos, 2)
    adj = construct_sparse_mx(train_pos_arr, n_num, d_num)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def make_bpr_triplets(train_pos, assoc, n_neg_per_pos=1):
    """Generate BPR triplets (n, d_i_pos, d_j_neg) for one epoch.
    Each positive pair gets matched with n_neg_per_pos negative drugs."""
    triplets = []
    n_mirna, n_drug = assoc.shape
    for (mi, di_pos) in train_pos:
        for _ in range(n_neg_per_pos):
            while True:
                dj = np.random.randint(0, n_drug)
                if assoc[mi, dj] == 0:
                    break
            triplets.append([mi, di_pos, dj])
    return np.array(triplets, dtype=np.int64)

def train_lrgcpnd_one_epoch(model, optimizer, triplets, batch_size=2048):
    """Standard BPR training pass."""
    model.train()
    idx = np.random.permutation(len(triplets))
    total_loss = 0.0; n_batch = 0
    for s in range(0, len(idx), batch_size):
        bi = idx[s:s+batch_size]
        batch = torch.LongTensor(triplets[bi]).cuda()
        n = batch[:, 0]; d_i = batch[:, 1]; d_j = batch[:, 2]
        optimizer.zero_grad()
        _, _, loss = model(n, d_i, d_j)
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n_batch += 1
    return total_loss / max(n_batch, 1)

@torch.no_grad()
def eval_lrgcpnd_5metrics(model, test_pos, assoc):
    """5 metrics on test positives + sampled negatives (DrugMiR ev_full() style).
    LRGCPND forward returns pre_i for positives and pre_j for negatives.
    We feed pos pairs as (n, d_i=pos_drug, d_j=dummy_neg_drug) and read pre_i;
    then feed neg pairs as (n, d_i=neg_drug, d_j=dummy) and read pre_i.
    """
    model.eval()
    neg_pos = sample_neg(assoc, len(test_pos))  # equal-size neg set
    
    # ---- score positives ----
    n_mirna, n_drug = assoc.shape
    pos_arr = np.array(test_pos)  # (N, 2)
    neg_arr = np.array(neg_pos)
    
    # Need a dummy d_j for every sample; pick random unobserved drug
    def add_dummy(arr):
        # arr: (N, 2). Return (N, 3) with random dummy d_j.
        dummies = np.random.randint(0, n_drug, size=(arr.shape[0], 1))
        return np.hstack([arr, dummies]).astype(np.int64)
    
    pos_triplets = add_dummy(pos_arr)
    neg_triplets = add_dummy(neg_arr)
    
    def score_batch(triplets, bs=4096):
        scores = []
        for s in range(0, len(triplets), bs):
            b = torch.LongTensor(triplets[s:s+bs]).cuda()
            pre_i, _, _ = model(b[:, 0], b[:, 1], b[:, 2])
            scores.append(pre_i.cpu().numpy())
        return np.concatenate(scores)
    
    pos_scores = score_batch(pos_triplets)
    neg_scores = score_batch(neg_triplets)
    
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    
    auc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    # Optimal F1 threshold from PR curve
    p_curve, r_curve, t_curve = precision_recall_curve(labels, scores)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-10)
    best_idx = np.argmax(f1_curve[:-1])
    best_thr = t_curve[best_idx]
    pred_bin = (scores >= best_thr).astype(int)
    f1 = f1_score(labels, pred_bin)
    prec = precision_score(labels, pred_bin, zero_division=0)
    rec = recall_score(labels, pred_bin)
    return {'auc': auc, 'aupr': aupr, 'f1': f1, 'prec': prec, 'rec': rec,
            'thr': float(best_thr)}

# ==================== Main run ====================

def run_lrgcpnd_dataset(dataset_name, data_dir, K=4, E_size=32, reg=0.05,
                       lr=0.005, max_epoch=100, patience=10, seed=42,
                       n_fold=5):
    """Train LRGCPND on dataset following DrugMiR 5-fold protocol."""
    print(f"\n{'='*70}\nLRGCPND on {dataset_name} (seed={seed}, {n_fold}-fold CV)\n{'='*70}", flush=True)
    t0 = time.time()
    
    assoc, pos_pairs, n_num, d_num = load_assoc(data_dir)
    print(f"  Loaded: n_mirna={n_num} n_drug={d_num} n_pos={len(pos_pairs)}", flush=True)
    print(f"  LRGCPND params: K={K} E_size={E_size} reg={reg} lr={lr}", flush=True)
    
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    
    fold_results = []
    for fold, (tri, tei) in enumerate(kf.split(pos_pairs)):
        tf0 = time.time()
        train_pos = [pos_pairs[i] for i in tri]
        test_pos = [pos_pairs[i] for i in tei]
        
        # Build adj from train pairs only (no test leakage)
        adj = build_adj_from_train_pairs(train_pos, n_num, d_num)
        
        # Init model
        model = LRGCPND(n_num, d_num, adj, K, E_size, reg).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
        pc = 0
        for e in range(max_epoch):
            # Per-epoch negative resampling (matches DrugMiR protocol)
            triplets = make_bpr_triplets(train_pos, assoc, n_neg_per_pos=1)
            loss = train_lrgcpnd_one_epoch(model, optimizer, triplets)
            if (e + 1) % 5 == 0:
                m = eval_lrgcpnd_5metrics(model, test_pos, assoc)
                if m['auc'] > best['auc']:
                    best = m; pc = 0
                else:
                    pc += 1
                if pc >= patience:
                    break
        
        fold_results.append(best)
        print(f"  Fold {fold+1}/{n_fold}: AUC={best['auc']:.4f} AUPR={best['aupr']:.4f} "
              f"F1={best['f1']:.4f} P={best['prec']:.4f} R={best['rec']:.4f} "
              f"({time.time()-tf0:.0f}s)", flush=True)
    
    summary = {}
    for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        vals = [r[k] for r in fold_results]
        summary[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    summary['fold_details'] = fold_results
    summary['dataset'] = dataset_name
    summary['seed'] = seed
    summary['hyperparams'] = {'K': K, 'E_size': E_size, 'reg': reg, 'lr': lr,
                              'max_epoch': max_epoch, 'patience': patience}
    summary['total_time'] = time.time() - t0
    
    print(f"\n  {dataset_name} summary (mean±std):")
    for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        print(f"    {k.upper():6s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
    print(f"  Total: {summary['total_time']:.0f}s ({summary['total_time']/60:.1f} min)", flush=True)
    
    return summary


if __name__ == '__main__':
    DD1 = os.path.expanduser("~/DrugMiR/data/dataset1")
    DD2 = os.path.expanduser("~/DrugMiR/data/dataset2")
    
    out = {}
    out['D1'] = run_lrgcpnd_dataset('D1', DD1)
    out['D2'] = run_lrgcpnd_dataset('D2', DD2)
    
    os.makedirs('phase2_outputs', exist_ok=True)
    with open('phase2_outputs/lrgcpnd_5metrics_seed42.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved to phase2_outputs/lrgcpnd_5metrics_seed42.json", flush=True)
    
    # Final summary table
    print(f"\n{'='*70}\nLRGCPND FINAL SUMMARY (DrugMiR protocol)\n{'='*70}")
    print(f"{'Metric':8s} | {'D1':25s} | {'D2':25s}")
    print("-" * 65)
    for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        d1 = f"{out['D1'][k]['mean']:.4f} ± {out['D1'][k]['std']:.4f}"
        d2 = f"{out['D2'][k]['mean']:.4f} ± {out['D2'][k]['std']:.4f}"
        print(f"{k.upper():8s} | {d1:25s} | {d2:25s}")
    
    # Compare to paper's reported number for LRGCPND
    print(f"\nPaper Table II for LRGCPND:")
    print(f"  D1: AUC=0.9444  AUPR=0.9441")
    print(f"  D2: AUC=0.9283  AUPR=0.9282")
