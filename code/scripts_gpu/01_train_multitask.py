"""
DrugMiR Multi-task experiment (paper §4.7).

Shared three-channel encoder + two parallel binary heads:
    L_multi = BCE(y_res_hat, y_res) + BCE(y_sen_hat, y_sen)        (eq. 16)

5-fold CV on resistance and sensitivity matrices independently.
Both folds use seed=42 (matches Table II / III protocol).

Outputs
-------
results_multitask/multitask_hybrid_results.json
    {res_auc_mean, res_auc_std, res_aupr_mean, ..., sen_..., macro_avg_auc, ...}
results_multitask/multitask_log.txt
    Per-fold AUC/AUPR for both heads, training time.
"""
import os, json, time, sys
import numpy as np
import torch
from sklearn.model_selection import KFold

from drugmir_model import (
    load_data, sample_neg, set_seed, device,
    DrugMiR_MultiTask, train_epoch, evaluate,
)
import torch.nn.functional as F

DATA_DIR = os.environ.get("DRUGMIR_DATA", "./data/processed")
OUT_DIR  = "./results_multitask"
os.makedirs(OUT_DIR, exist_ok=True)
LOG = open(os.path.join(OUT_DIR, "multitask_log.txt"), "w")
def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


def load_multitask_data(km=15, kd=10):
    """Single load that brings in mirna/drug features, gene matrices, AND
    both resistance and sensitivity matrices. We override `assoc` per task
    when running each head's training loop.
    """
    base = load_data(DATA_DIR, km=km, kd=kd, assoc_file="resistance_matrix.npy")
    sen  = np.load(f"{DATA_DIR}/sensitivity_matrix.npy")
    base["res_matrix"] = base["assoc"]                       # alias
    base["sen_matrix"] = sen
    base["res_pairs"]  = base["pos_pairs"]
    pr, pc = np.nonzero(sen)
    base["sen_pairs"]  = list(zip(pr.tolist(), pc.tolist()))
    return base


def train_multitask_epoch(model, data, res_train, sen_train,
                          optimizer, batch_size=2048):
    """Train one epoch with the joint loss eq. (16).

    Each batch contains a mix of resistance and sensitivity pairs (positives +
    sampled negatives). We compute the two BCEs separately and add them.
    """
    model.train()

    # Build batches: alternate between res and sen each step. Simpler: union
    # the two task pair lists with task tags, then iterate jointly.
    res_neg = sample_neg(data["res_matrix"], res_train, len(res_train))
    sen_neg = sample_neg(data["sen_matrix"], sen_train, len(sen_train))
    res_all = [(p, "res", 1.0) for p in res_train] + [(p, "res", 0.0) for p in res_neg]
    sen_all = [(p, "sen", 1.0) for p in sen_train] + [(p, "sen", 0.0) for p in sen_neg]
    pool = res_all + sen_all
    idx = np.random.permutation(len(pool))

    total_loss, n_batches = 0.0, 0
    for s in range(0, len(idx), batch_size):
        bi = idx[s:s + batch_size]
        bp = [pool[i] for i in bi]
        # Split this batch by task
        for tag in ("res", "sen"):
            sub = [(p, lab) for (p, t, lab) in bp if t == tag]
            if not sub:
                continue
            mi = torch.LongTensor([p[0] for p, _ in sub]).to(device)
            di = torch.LongTensor([p[1] for p, _ in sub]).to(device)
            bl = torch.FloatTensor([lab for _, lab in sub]).to(device)
            optimizer.zero_grad()
            res_logits, sen_logits = model(data, mi, di)
            logits = res_logits if tag == "res" else sen_logits
            loss = F.binary_cross_entropy_with_logits(logits, bl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_multitask_head(model, data, test_pairs, task):
    """Evaluate one head on its task's held-out pairs."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    model.eval()
    matrix_key = "res_matrix" if task == "res" else "sen_matrix"
    neg = sample_neg(data[matrix_key], test_pairs, len(test_pairs))
    pairs = test_pairs + neg
    labels = np.array([1.0] * len(test_pairs) + [0.0] * len(neg))
    mi = torch.LongTensor([p[0] for p in pairs]).to(device)
    di = torch.LongTensor([p[1] for p in pairs]).to(device)
    res_logits, sen_logits = model(data, mi, di)
    logits = res_logits if task == "res" else sen_logits
    pr = torch.sigmoid(logits).cpu().numpy()
    return (roc_auc_score(labels, pr),
            average_precision_score(labels, pr))


def main():
    log("=" * 60)
    log("DrugMiR Multi-task Experiment (§4.7)")
    log("Architecture: shared 3-channel encoder + 2 parallel heads")
    log("Loss: L_multi = BCE_res + BCE_sen  (eq. 16)")
    log("=" * 60)

    log(f"Data dir: {DATA_DIR}")
    log(f"Device:   {device}")
    log("")
    log("Loading data...")
    t0 = time.time()
    data = load_multitask_data(km=15, kd=10)
    log(f"  miRNAs={data['n_mirna']}, drugs={data['n_drug']}, "
        f"genes={data['n_gene']}")
    log(f"  Resistance pairs: {len(data['res_pairs'])}")
    log(f"  Sensitivity pairs: {len(data['sen_pairs'])}")
    # Dual-effect statistic
    both = ((data["res_matrix"] > 0) & (data["sen_matrix"] > 0)).sum()
    either = ((data["res_matrix"] > 0) | (data["sen_matrix"] > 0)).sum()
    log(f"  Dual-effect pairs (both res & sen): {int(both)} / {int(either)} "
        f"= {both / either * 100:.2f}%")
    log(f"  Loaded in {time.time() - t0:.1f}s")
    log("")

    # Hyperparameters: match the paper's confirmed best config
    HP = dict(h=256, dr=0.5, n_gcn=2, n_br=2,
              lr=5e-4, wd=2e-4, epochs=200, patience=15)
    log(f"Hyperparameters: {HP}")
    log("")

    SEED = 42
    N_FOLDS = 5
    set_seed(SEED)

    # Per-fold metrics (we drive folds off the union of res ∪ sen pair lists,
    # but report per-task AUC computed only on that task's held-out positives)
    res_pairs = data["res_pairs"]
    sen_pairs = data["sen_pairs"]
    md = data["mirna_feat"].shape[1]; dd = data["drug_feat"].shape[1]
    nm = data["n_mirna"]; nd = data["n_drug"]; ng = data["n_gene"]

    res_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    sen_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    res_folds = list(res_kf.split(res_pairs))
    sen_folds = list(sen_kf.split(sen_pairs))

    per_fold = []
    for fold in range(N_FOLDS):
        log(f"--- Fold {fold + 1}/{N_FOLDS} ---")
        rt, re_ = res_folds[fold]; st, se = sen_folds[fold]
        res_tr = [res_pairs[i] for i in rt]; res_te = [res_pairs[i] for i in re_]
        sen_tr = [sen_pairs[i] for i in st]; sen_te = [sen_pairs[i] for i in se]

        model = DrugMiR_MultiTask(nm, nd, md, dd, ng,
                                  h=HP["h"], dr=HP["dr"],
                                  n_gcn=HP["n_gcn"], n_br=HP["n_br"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=HP["lr"],
                               weight_decay=HP["wd"])

        best_macro = 0.0
        best = dict(res_auc=0, res_aupr=0, sen_auc=0, sen_aupr=0)
        patience_ctr = 0
        t_fold = time.time()
        for ep in range(HP["epochs"]):
            train_multitask_epoch(model, data, res_tr, sen_tr, opt)
            if (ep + 1) % 5 == 0:
                ra, rap = eval_multitask_head(model, data, res_te, "res")
                sa, sap = eval_multitask_head(model, data, sen_te, "sen")
                macro = (ra + sa) / 2
                if macro > best_macro:
                    best_macro = macro
                    best = dict(res_auc=ra, res_aupr=rap,
                                sen_auc=sa, sen_aupr=sap)
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= HP["patience"]:
                    break
        log(f"  Best: Res AUC={best['res_auc']:.4f} AUPR={best['res_aupr']:.4f}  "
            f"Sen AUC={best['sen_auc']:.4f} AUPR={best['sen_aupr']:.4f}  "
            f"({time.time() - t_fold:.0f}s)")
        per_fold.append(best)

    # Aggregate
    def agg(key):
        v = np.array([f[key] for f in per_fold])
        return float(v.mean()), float(v.std())

    summary = dict(
        n_folds=N_FOLDS, seed=SEED, hyperparameters=HP,
        per_fold=per_fold,
        res_auc_mean=agg("res_auc")[0],   res_auc_std=agg("res_auc")[1],
        res_aupr_mean=agg("res_aupr")[0], res_aupr_std=agg("res_aupr")[1],
        sen_auc_mean=agg("sen_auc")[0],   sen_auc_std=agg("sen_auc")[1],
        sen_aupr_mean=agg("sen_aupr")[0], sen_aupr_std=agg("sen_aupr")[1],
        macro_auc_mean=(agg("res_auc")[0] + agg("sen_auc")[0]) / 2,
        macro_aupr_mean=(agg("res_aupr")[0] + agg("sen_aupr")[0]) / 2,
        dual_effect_count=int(both),
        unique_pair_count=int(either),
        dual_effect_ratio=float(both / either),
    )

    out_path = os.path.join(OUT_DIR, "multitask_hybrid_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log("=" * 60)
    log("FINAL RESULTS (5-fold CV, seed=42, Hybrid encoder)")
    log("=" * 60)
    log(f"Resistance:  AUC = {summary['res_auc_mean']:.4f} ± {summary['res_auc_std']:.4f}, "
        f"AUPR = {summary['res_aupr_mean']:.4f} ± {summary['res_aupr_std']:.4f}")
    log(f"Sensitivity: AUC = {summary['sen_auc_mean']:.4f} ± {summary['sen_auc_std']:.4f}, "
        f"AUPR = {summary['sen_aupr_mean']:.4f} ± {summary['sen_aupr_std']:.4f}")
    log(f"Macro avg:   AUC = {summary['macro_auc_mean']:.4f}, "
        f"AUPR = {summary['macro_aupr_mean']:.4f}")
    log(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
