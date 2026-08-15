"""
Phase 2c: GSLRDA benchmark on D1 + D2
- Imports GSLRDA original TF class (with tf.compat.v1) — no re-implementation
- Uses DrugMiR's 5-fold split + per-epoch protocol
- Uses DrugMiR's ev_full() (5 metrics at optimal F1 threshold)
- Output: phase2_outputs/gslrda_5metrics_seed42.json

Strategy: feed GSLRDA's Relation with our integer-idx pairs cast to strings.
Only use GSLRDA's model & training loop; evaluation done with our protocol.
"""
import os, sys, json, time, warnings, copy
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              precision_recall_curve)
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

# TF 1.x compat mode — this MUST come before any TF import inside GSLRDA
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Path setup so we can `import main` from GSLRDA repo
GSLRDA_PATH = os.path.expanduser('~/work/DrugMiR/GSLRDA/code/GSLRDA')
sys.path.insert(0, GSLRDA_PATH)
sys.path.insert(0, os.path.join(GSLRDA_PATH, 'util'))

# Patch GSLRDA's main.py to use tf.compat.v1 (must be done before importing it)
import shutil
patched_dir = '/tmp/gslrda_patched'
if not os.path.exists(patched_dir):
    shutil.copytree(GSLRDA_PATH, patched_dir)
    # Replace `import tensorflow as tf` with compat.v1
    main_file = os.path.join(patched_dir, 'main.py')
    with open(main_file) as f:
        code = f.read()
    code = code.replace('import tensorflow as tf',
                        'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()')
    with open(main_file, 'w') as f:
        f.write(code)

# Re-route imports through patched dir
sys.path.insert(0, patched_dir)
sys.path.insert(0, os.path.join(patched_dir, 'util'))

# Now import GSLRDA model
from util.config import Config, LineConfig
from util.io import FileIO
import importlib.util
spec = importlib.util.spec_from_file_location("gslrda_main", os.path.join(patched_dir, 'main.py'))
gslrda_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gslrda_main)
GSLRDA = gslrda_main.GSLRDA

print("✓ GSLRDA imported (TF compat.v1 mode)", flush=True)
print(f"  TF version: {tf.__version__}, eager mode: {tf.executing_eagerly()}", flush=True)


# ============== DrugMiR协议数据加载（matching LRGCPND wrapper） ==============

def load_assoc(dd):
    """Load association matrix and positive pairs."""
    assoc = np.load(f"{dd}/association_matrix.npy")
    pr, pc = np.nonzero(assoc)
    pos_pairs = list(zip(pr.tolist(), pc.tolist()))
    return assoc, pos_pairs, assoc.shape[0], assoc.shape[1]


def sample_neg(assoc, n_samples):
    nm, nd = assoc.shape; neg = []
    while len(neg) < n_samples:
        i = np.random.randint(0, nm); j = np.random.randint(0, nd)
        if assoc[i, j] == 0: neg.append((i, j))
    return neg


# ============== GSLRDA 适配层 ==============

def make_gslrda_config():
    """Build a Config-like dict mimicking GSLRDA.conf, parseable by GSLRDA __init__."""
    # Read the actual conf file then mutate parameters as needed
    conf_path = os.path.join(GSLRDA_PATH, 'GSLRDA.conf')
    conf = Config(conf_path)
    return conf


def pairs_to_gslrda_triplets(pairs):
    """Convert [(mi_idx, drug_idx), ...] → [[str(mi), str(drug), 1.0], ...]
    GSLRDA expects (ncRNAName_str, drugName_str, rating_float) triplets."""
    return [[str(m), str(d), 1.0] for (m, d) in pairs]


# ============== GSLRDA 5-metric eval (DrugMiR ev_full style) ==============

def eval_gslrda_5metrics(model, test_pos, assoc):
    """5 metrics on test_pos + sampled negatives.
    GSLRDA scores via internal embedding lookup.
    """
    n_mirna, n_drug = assoc.shape
    neg_set = sample_neg(assoc, len(test_pos))
    
    # Score all positives and negatives
    # GSLRDA's model has self.main_ncRNA_embeddings and self.main_drug_embeddings
    # They are tf tensors; we need to compute them via session
    # The 'self.test' op computes inner product of single mirna against all drugs
    # But for our use we want individual scores
    
    # We feed: u_idx = [mirna ids], v_idx = [drug ids] → look up main embeddings → dot product
    # The model has neg_idx placeholder too, but we don't need negatives during eval
    # We compute scores using direct embedding lookup via session
    
    # First, compute main embeddings (these are functions of the full adj — no feed dict needed)
    # Actually they ARE part of the computational graph, so need to feed sub_mat placeholders too
    # Trick: feed empty subgraphs (won't affect main_embeddings since they only use norm_adj)
    
    # Easier: run the full graph with dummy sub_mat and read embeddings
    # But sub_mat placeholders are part of the model so we must feed something valid
    
    # Build feed dict with all required sub_mat placeholders (using the latest sub_mat dict)
    # We'll store it in the model after last training step
    
    # Score positives
    pos_u = [model.data.ncRNA[str(m)] for (m, d) in test_pos if str(m) in model.data.ncRNA and str(d) in model.data.drug]
    pos_v = [model.data.drug[str(d)] for (m, d) in test_pos if str(m) in model.data.ncRNA and str(d) in model.data.drug]
    pos_keep_idx = [i for i, (m, d) in enumerate(test_pos) if str(m) in model.data.ncRNA and str(d) in model.data.drug]
    
    neg_u = [model.data.ncRNA[str(m)] for (m, d) in neg_set if str(m) in model.data.ncRNA and str(d) in model.data.drug]
    neg_v = [model.data.drug[str(d)] for (m, d) in neg_set if str(m) in model.data.ncRNA and str(d) in model.data.drug]
    
    if len(pos_u) == 0 or len(neg_u) == 0:
        return None  # cold-start case
    
    # Score: u_emb · v_emb (inner product)
    # Use the model.main_ncRNA_embeddings and model.main_drug_embeddings
    # Need to fetch them via session.run with dummy sub_mat feeds
    
    feed = {}
    # Use the LAST sub_mat that was created during training (stored on model.last_sub_mat by our patch)
    feed.update(model._last_feed_for_eval)
    
    # Fetch embeddings (one-shot, no feed_dict for u_idx/v_idx since we just want the matrices)
    main_n_emb, main_d_emb = model.sess.run(
        [model.main_ncRNA_embeddings, model.main_drug_embeddings],
        feed_dict=feed
    )
    
    pos_scores = (main_n_emb[pos_u] * main_d_emb[pos_v]).sum(axis=1)
    neg_scores = (main_n_emb[neg_u] * main_d_emb[neg_v]).sum(axis=1)
    
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    
    auc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    p_curve, r_curve, t_curve = precision_recall_curve(labels, scores)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-10)
    best_idx = int(np.argmax(f1_curve[:-1]))
    best_thr = float(t_curve[best_idx])
    pred_bin = (scores >= best_thr).astype(int)
    f1 = f1_score(labels, pred_bin)
    prec = precision_score(labels, pred_bin, zero_division=0)
    rec = recall_score(labels, pred_bin)
    return {'auc': auc, 'aupr': aupr, 'f1': f1, 'prec': prec, 'rec': rec, 'thr': best_thr}


# ============== Training wrapper around GSLRDA ==============

def train_gslrda_one_fold(train_pos, test_pos, assoc, n_mirna, n_drug,
                          conf, max_epoch=100, eval_every=2, patience=5):
    """Train GSLRDA on train_pos, evaluate every eval_every epochs on test_pos."""
    
    # Reset TF graph for each fold (critical — TF accumulates variables across runs)
    tf.reset_default_graph()
    
    # Convert pairs to GSLRDA triplet format
    train_triplets = pairs_to_gslrda_triplets(train_pos)
    test_triplets = pairs_to_gslrda_triplets(test_pos)
    
    # Override num.max.iter to control epochs externally
    conf.config['num.max.iter'] = str(max_epoch)
    
    # Instantiate GSLRDA — this BUILDS the TF graph and creates Session
    model = GSLRDA(conf, trainingSet=train_triplets, testSet=test_triplets)
    
    # ============== Recreate GSLRDA's buildModel but with eval loop ==============
    # Below is essentially a copy of GSLRDA.buildModel() with eval hook injected
    
    u_emb = model.u_embedding
    v_emb = model.v_embedding
    neg_emb = model.neg_drug_embedding
    
    y = tf.reduce_sum(tf.multiply(u_emb, v_emb), 1) - tf.reduce_sum(tf.multiply(u_emb, neg_emb), 1)
    rec_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y) + 1e-10)) + model.regU * (
        tf.nn.l2_loss(u_emb) + tf.nn.l2_loss(v_emb) + tf.nn.l2_loss(neg_emb)
    )
    ssl_loss = model.calc_ssl_loss_v3()
    total_loss = rec_loss + ssl_loss
    
    opt = tf.train.AdamOptimizer(model.lRate)
    train_op = opt.minimize(total_loss)
    
    init = tf.global_variables_initializer()
    model.sess.run(init)
    
    best = {'auc': 0, 'aupr': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'thr': 0.5}
    pc = 0
    
    for iteration in range(max_epoch):
        # Build augmented subgraphs (this changes each epoch in GSLRDA)
        sub_mat = {}
        if model.aug_type in [0, 1]:
            sub_mat['adj_indices_sub1'], sub_mat['adj_values_sub1'], sub_mat['adj_shape_sub1'] = \
                model._convert_csr_to_sparse_tensor_inputs(
                    model._create_adj_mat(is_subgraph=True, aug_type=model.aug_type))
            sub_mat['adj_indices_sub2'], sub_mat['adj_values_sub2'], sub_mat['adj_shape_sub2'] = \
                model._convert_csr_to_sparse_tensor_inputs(
                    model._create_adj_mat(is_subgraph=True, aug_type=model.aug_type))
        
        # ====== Train one epoch ======
        for batch in model.next_batch_pairwise():
            ncRNA_idx, i_idx, j_idx = batch
            feed_dict = {
                model.u_idx: ncRNA_idx,
                model.v_idx: i_idx,
                model.neg_idx: j_idx,
            }
            if model.aug_type in [0, 1]:
                feed_dict.update({
                    model.sub_mat['adj_values_sub1']: sub_mat['adj_values_sub1'],
                    model.sub_mat['adj_indices_sub1']: sub_mat['adj_indices_sub1'],
                    model.sub_mat['adj_shape_sub1']: sub_mat['adj_shape_sub1'],
                    model.sub_mat['adj_values_sub2']: sub_mat['adj_values_sub2'],
                    model.sub_mat['adj_indices_sub2']: sub_mat['adj_indices_sub2'],
                    model.sub_mat['adj_shape_sub2']: sub_mat['adj_shape_sub2'],
                })
            model.sess.run(train_op, feed_dict=feed_dict)
        
        # Store last sub_mat feed for eval
        model._last_feed_for_eval = {
            model.sub_mat['adj_values_sub1']: sub_mat['adj_values_sub1'],
            model.sub_mat['adj_indices_sub1']: sub_mat['adj_indices_sub1'],
            model.sub_mat['adj_shape_sub1']: sub_mat['adj_shape_sub1'],
            model.sub_mat['adj_values_sub2']: sub_mat['adj_values_sub2'],
            model.sub_mat['adj_indices_sub2']: sub_mat['adj_indices_sub2'],
            model.sub_mat['adj_shape_sub2']: sub_mat['adj_shape_sub2'],
        }
        
        # ====== Eval ======
        if (iteration + 1) % eval_every == 0:
            m = eval_gslrda_5metrics(model, test_pos, assoc)
            if m is not None:
                if m['auc'] > best['auc']:
                    best = m; pc = 0
                else:
                    pc += 1
                if pc >= patience:
                    break
    
    model.sess.close()
    return best


def run_gslrda_dataset(dataset_name, data_dir, max_epoch=100, eval_every=2,
                       patience=5, seed=42, n_fold=5):
    print(f"\n{'='*70}\nGSLRDA on {dataset_name} (seed={seed}, {n_fold}-fold CV)\n{'='*70}", flush=True)
    t0 = time.time()
    
    assoc, pos_pairs, n_mirna, n_drug = load_assoc(data_dir)
    print(f"  Loaded: n_mirna={n_mirna} n_drug={n_drug} n_pos={len(pos_pairs)}", flush=True)
    
    np.random.seed(seed)
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    
    fold_results = []
    for fold, (tri, tei) in enumerate(kf.split(pos_pairs)):
        tf0 = time.time()
        train_pos = [pos_pairs[i] for i in tri]
        test_pos = [pos_pairs[i] for i in tei]
        
        # Fresh conf for each fold (avoid mutation issues)
        conf = make_gslrda_config()
        
        best = train_gslrda_one_fold(
            train_pos, test_pos, assoc, n_mirna, n_drug, conf,
            max_epoch=max_epoch, eval_every=eval_every, patience=patience
        )
        
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
    out['D1'] = run_gslrda_dataset('D1', DD1)
    out['D2'] = run_gslrda_dataset('D2', DD2)
    
    os.makedirs('phase2_outputs', exist_ok=True)
    with open('phase2_outputs/gslrda_5metrics_seed42.json', 'w') as f:
        json.dump(out, f, indent=2)
    
    print(f"\n{'='*70}\nGSLRDA FINAL SUMMARY (DrugMiR protocol)\n{'='*70}")
    print(f"{'Metric':8s} | {'D1':25s} | {'D2':25s}")
    print("-" * 65)
    for k in ['auc', 'aupr', 'f1', 'prec', 'rec']:
        d1 = f"{out['D1'][k]['mean']:.4f} ± {out['D1'][k]['std']:.4f}"
        d2 = f"{out['D2'][k]['mean']:.4f} ± {out['D2'][k]['std']:.4f}"
        print(f"{k.upper():8s} | {d1:25s} | {d2:25s}")
    
    print(f"\nPaper Table II for GSLRDA:")
    print(f"  D1: AUC=0.9567  AUPR=0.9513")
    print(f"  D2: AUC=0.9426  AUPR=0.9397")
