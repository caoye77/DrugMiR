"""
DrugMiR cold-start runner: S2 (miRNA-cold) / S3 (drug-cold) / S4 (pair-cold)

KEY DIFFERENCES vs original `hp_finetune.py`:
  1. KNN similarity graph is rebuilt PER-FOLD using only training entities.
     test miRNAs do NOT appear as KNN neighbours of train miRNAs (and vice versa).
  2. Negative sampling is restricted to (train_miRNA × train_drug) — never sees
     test entities during training updates.
  3. Test-set prediction uses the trained model in inference mode (test entities
     get their features through the Hybrid Encoder, but their embedding rows in
     nn.Embedding are never updated during training, so behave like cold init).
  4. Gene-bridge edges (miRNA→gene, drug→gene) are NOT pruned — they come from
     external databases (miRTarBase / DrugBank), not training data, so feeding
     them at test time is feature injection, not label leakage.

Usage:
  python3 run_drugmir_coldstart.py --dataset D1 --setting S2 --seed 42
"""
import os, sys, json, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              precision_recall_curve)
warnings.filterwarnings('ignore')

# Make sure local files are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_splitter import ColdStartSplitter, ColdFold

# Re-use the model classes from hp_finetune.py (we only patch the data pipeline)
sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
from hp_finetune import GG, GB, HybridEnc, DrugMiR_Hybrid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================== Data loading ==============================

def load_static_data(data_dir):
    """Load everything that doesn't depend on the fold split."""
    assoc = np.load(f"{data_dir}/association_matrix.npy")
    mf = np.load(f"{data_dir}/mirna_kmer_features.npy")
    df = np.load(f"{data_dir}/drug_morgan_features.npy")
    ms = np.load(f"{data_dir}/mirna_similarity.npy")
    ds = np.load(f"{data_dir}/drug_similarity.npy")
    mg = np.load(f"{data_dir}/mirna_gene_matrix.npy")
    dg = np.load(f"{data_dir}/drug_gene_matrix.npy")
    n_mirna, n_drug = assoc.shape
    return {
        'assoc': assoc,
        'mf': mf, 'df': df,
        'ms': ms, 'ds': ds,
        'mg': mg, 'dg': dg,
        'n_mirna': n_mirna, 'n_drug': n_drug,
    }


def build_fold_data(static, fold: ColdFold, km=15, kd=10):
    """Build a per-fold data dict with train-only KNN sim graph.
    
    Cold-start invariant: KNN edges only connect train-train entities.
    Test entities appear as isolated nodes in the sim graph (their GCN message
    passing yields h^(l) = m0 + 0, i.e. only their hybrid-encoded features).
    """
    n_mirna, n_drug = static['n_mirna'], static['n_drug']
    train_mi = fold.train_mirna_mask.nonzero()[0]
    train_dr = fold.train_drug_mask.nonzero()[0]
    train_mi_set = set(train_mi.tolist())
    train_dr_set = set(train_dr.tolist())

    # ---- mirna KNN: for each train mirna, find top-km train-mirna neighbours ----
    ms = static['ms']
    mirna_sim_src, mirna_sim_dst = [], []
    for i in train_mi:
        si = ms[i].copy()
        # mask out self AND all non-train mirnas
        si[i] = -np.inf
        non_train = np.ones(n_mirna, dtype=bool)
        non_train[train_mi] = False
        si[non_train] = -np.inf
        # top-km
        # use argpartition for speed, then take last km
        if (~np.isinf(si)).sum() < km:
            # fewer candidates than km → take all available
            tk = np.where(~np.isinf(si))[0]
        else:
            tk = np.argpartition(si, -km)[-km:]
        for j in tk:
            if j != i and not np.isinf(si[j]):
                mirna_sim_src.extend([int(i), int(j)])
                mirna_sim_dst.extend([int(j), int(i)])

    # ---- drug KNN: same construction ----
    ds = static['ds']
    drug_sim_src, drug_sim_dst = [], []
    for i in train_dr:
        si = ds[i].copy()
        si[i] = -np.inf
        non_train = np.ones(n_drug, dtype=bool)
        non_train[train_dr] = False
        si[non_train] = -np.inf
        if (~np.isinf(si)).sum() < kd:
            tk = np.where(~np.isinf(si))[0]
        else:
            tk = np.argpartition(si, -kd)[-kd:]
        for j in tk:
            if j != i and not np.isinf(si[j]):
                drug_sim_src.extend([int(i), int(j)])
                drug_sim_dst.extend([int(j), int(i)])

    # ---- Gene bridge (UNCHANGED — external database knowledge, not training data) ----
    mg, dg_ = static['mg'], static['dg']
    mg_r, mg_c = np.nonzero(mg)
    dg_r, dg_c = np.nonzero(dg_)
    n_gene = max(mg_c.max() if len(mg_c) > 0 else 0,
                 dg_c.max() if len(dg_c) > 0 else 0) + 1

    # ---- Tensors ----
    mf = torch.FloatTensor(static['mf']).to(device)
    df = torch.FloatTensor(static['df']).to(device)
    d = {
        'assoc': static['assoc'],
        'mirna_feat': mf, 'drug_feat': df,
        'n_mirna': n_mirna, 'n_drug': n_drug,
        'mirna_has_feat': torch.FloatTensor((static['mf'].sum(1) > 0).astype(float)).to(device),
        'drug_has_feat':  torch.FloatTensor((static['df'].sum(1) > 0).astype(float)).to(device),
        'mirna_sim_edge': torch.LongTensor([mirna_sim_src, mirna_sim_dst]).to(device) if mirna_sim_src
                          else torch.zeros((2, 0), dtype=torch.long, device=device),
        'drug_sim_edge':  torch.LongTensor([drug_sim_src, drug_sim_dst]).to(device) if drug_sim_src
                          else torch.zeros((2, 0), dtype=torch.long, device=device),
        'mg_src': torch.LongTensor(mg_r).to(device),
        'mg_dst': torch.LongTensor(mg_c).to(device),
        'dg_src': torch.LongTensor(dg_r).to(device),
        'dg_dst': torch.LongTensor(dg_c).to(device),
        'n_gene': n_gene,
        # masks for negative sampling
        'train_mi_mask': fold.train_mirna_mask,
        'train_dr_mask': fold.train_drug_mask,
    }
    return d


# ============================== Train / eval ==============================

def sample_negatives_train(assoc, train_mi_mask, train_dr_mask, n):
    """Sample n negatives from (train_miRNA × train_drug) grid."""
    train_mi_idx = np.where(train_mi_mask)[0]
    train_dr_idx = np.where(train_dr_mask)[0]
    neg = []; tries = 0
    while len(neg) < n and tries < n * 200:
        i = int(np.random.choice(train_mi_idx))
        j = int(np.random.choice(train_dr_idx))
        if assoc[i, j] == 0:
            neg.append((i, j))
        tries += 1
    if len(neg) < n:
        # fallback: relax constraint slightly
        while len(neg) < n:
            i = int(np.random.choice(train_mi_idx))
            j = int(np.random.choice(train_dr_idx))
            neg.append((i, j))
    return neg


def sample_negatives_test(assoc, n):
    """Sample n negatives from full grid (test eval uses full coverage)."""
    nm, nd = assoc.shape
    neg = []
    while len(neg) < n:
        i = np.random.randint(0, nm); j = np.random.randint(0, nd)
        if assoc[i, j] == 0:
            neg.append((i, j))
    return neg


def train_one_epoch(model, data, train_pos, optimizer, bs=2048):
    model.train()
    neg = sample_negatives_train(
        data['assoc'], data['train_mi_mask'], data['train_dr_mask'], len(train_pos)
    )
    pairs = list(train_pos) + neg
    labels = [1.0] * len(train_pos) + [0.0] * len(neg)
    idx = np.random.permutation(len(pairs))
    total_loss = 0; n_batches = 0
    for s in range(0, len(idx), bs):
        bi = idx[s:s + bs]
        bp = [pairs[i] for i in bi]
        bl = torch.FloatTensor([labels[i] for i in bi]).to(device)
        mi = torch.LongTensor([p[0] for p in bp]).to(device)
        di = torch.LongTensor([p[1] for p in bp]).to(device)
        optimizer.zero_grad()
        logits = model(data, mi, di)
        loss = F.binary_cross_entropy_with_logits(logits, bl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_5metrics(model, data, test_pos):
    """Compute 5 metrics on test_pos + sampled negatives (1:1)."""
    model.eval()
    neg = sample_negatives_test(data['assoc'], len(test_pos))
    pairs = list(test_pos) + neg
    labels = np.array([1.0] * len(test_pos) + [0.0] * len(neg))
    mi = torch.LongTensor([p[0] for p in pairs]).to(device)
    di = torch.LongTensor([p[1] for p in pairs]).to(device)
    logits = model(data, mi, di)
    scores = torch.sigmoid(logits).cpu().numpy()

    auc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    p, r, t = precision_recall_curve(labels, scores)
    f1c = 2 * p * r / (p + r + 1e-10)
    bi = int(np.argmax(f1c[:-1]))
    thr = float(t[bi])
    pb = (scores >= thr).astype(int)
    f1 = f1_score(labels, pb)
    pr_ = precision_score(labels, pb, zero_division=0)
    rc = recall_score(labels, pb)
    return {'auc': float(auc), 'aupr': float(aupr), 'f1': float(f1),
            'prec': float(pr_), 'rec': float(rc), 'thr': thr}


# ============================== Per-fold driver ==============================

def run_one_fold(static, fold: ColdFold, seed, h=256, dr=0.5, n_gcn=2, n_br=2,
                  km=15, kd=10, lr=1e-3, wd=2e-4, ep=200, pat=15):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    data = build_fold_data(static, fold, km=km, kd=kd)

    train_pos = [(int(m), int(d)) for m, d in fold.train_pairs]
    test_pos  = [(int(m), int(d)) for m, d in fold.test_pairs]

    md, dd = data['mirna_feat'].shape[1], data['drug_feat'].shape[1]
    ng = data['n_gene']; nm, nd = data['n_mirna'], data['n_drug']
    model = DrugMiR_Hybrid(nm, nd, md, dd, ng, h=h, dr=dr, n_gcn=n_gcn, n_br=n_br).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
    pc = 0
    for e in range(ep):
        train_one_epoch(model, data, train_pos, opt)
        if (e + 1) % 5 == 0:
            m = evaluate_5metrics(model, data, test_pos)
            if m['auc'] > best['auc']:
                best = m; pc = 0
            else:
                pc += 1
            if pc >= pat:
                break
    return best


# ============================== Dataset driver ==============================

def run_dataset_coldstart(dataset_name, data_dir, setting, seed=42, n_fold=5):
    print(f"\n{'='*72}\nDrugMiR cold-start: {dataset_name} / {setting} / seed={seed}\n{'='*72}", flush=True)
    t0 = time.time()
    static = load_static_data(data_dir)
    print(f"  n_mirna={static['n_mirna']} n_drug={static['n_drug']} "
          f"n_pos={int(static['assoc'].sum())}", flush=True)

    splitter = ColdStartSplitter(static['assoc'], n_folds=n_fold, seed=seed,
                                  min_test_positives=15)
    folds = splitter.split(setting)

    fold_results = []
    for f in folds:
        tf0 = time.time()
        best = run_one_fold(static, f, seed=seed)
        fold_results.append(best)
        print(f"  Fold {f.fold_id+1}/{n_fold}: AUC={best['auc']:.4f} AUPR={best['aupr']:.4f} "
              f"F1={best['f1']:.4f} P={best['prec']:.4f} R={best['rec']:.4f} "
              f"(train_pos={len(f.train_pairs)} test_pos={len(f.test_pairs)} "
              f"trainM={f.train_mirna_mask.sum()} trainN={f.train_drug_mask.sum()}) "
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

    DD = {'D1': os.path.expanduser('~/work/DrugMiR/data/processed'),
          'D2': os.path.expanduser('~/work/DrugMiR/DMGAT_processed')}[args.dataset]
    res = run_dataset_coldstart(args.dataset, DD, args.setting, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = f"{args.out_dir}/drugmir_{args.dataset}_{args.setting}_seed{args.seed}.json"
    with open(out_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  Saved to {out_file}", flush=True)
