"""
DrugMiR Robustness and Convergence Analysis (paper §4.9).

Two experiments:

1. Convergence — train on Dataset 1 for 200 epochs (single fold, seed=42)
   and record train/val AUC at every evaluation step. Verifies smooth
   convergence and absence of overfitting.

2. Robustness to false negatives — Standard negative sampling treats every
   unobserved (i, j) pair as a true negative, but some unobserved pairs are
   actually true positives that just haven't been measured yet (false
   negatives). We simulate increasing false-negative ratios by deliberately
   relabeling a random subset of held-out positive pairs as negatives during
   training, and report how AUC degrades.

Outputs
-------
results_robustness/convergence.json
    epoch -> {train_auc, val_auc, val_aupr}
results_robustness/robustness.json
    fn_ratio -> {auc_mean, auc_std, aupr_mean, aupr_std}
results_robustness/fig_convergence.pdf / .png
results_robustness/fig_robustness.pdf / .png
"""
import os, json, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

from drugmir_model import (
    load_data, sample_neg, set_seed, device, DrugMiR_Hybrid,
)

DATA_DIR = os.environ.get("DRUGMIR_DATA", "./data/processed")
OUT_DIR  = "./results_robustness"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix",
                     "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


# ============================================================
# Experiment 1: convergence
# ============================================================

def run_convergence(data, hp, n_epochs=200, eval_every=2):
    """Train one model, log AUC every `eval_every` epochs.

    We split positives into 80/20 train/val (no overlap with negatives),
    sample fresh negatives every epoch, and track AUC on the held-out 20%.
    """
    set_seed(42)
    pos = data["pos_pairs"]
    n_train = int(0.8 * len(pos))
    perm = np.random.permutation(len(pos))
    train_pos = [pos[i] for i in perm[:n_train]]
    val_pos   = [pos[i] for i in perm[n_train:]]

    md = data["mirna_feat"].shape[1]; dd = data["drug_feat"].shape[1]
    model = DrugMiR_Hybrid(data["n_mirna"], data["n_drug"], md, dd,
                           data["n_gene"],
                           h=hp["h"], dr=hp["dr"],
                           n_gcn=hp["n_gcn"], n_br=hp["n_br"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                           weight_decay=hp["wd"])

    history = []
    for ep in range(n_epochs):
        # train epoch
        model.train()
        neg = sample_neg(data["assoc"], train_pos, len(train_pos))
        pairs = train_pos + neg
        labs = [1.0] * len(train_pos) + [0.0] * len(neg)
        idx = np.random.permutation(len(pairs))
        train_logits, train_labels = [], []
        for s in range(0, len(idx), 2048):
            bi = idx[s:s + 2048]
            bp = [pairs[i] for i in bi]
            bl = torch.FloatTensor([labs[i] for i in bi]).to(device)
            mi = torch.LongTensor([p[0] for p in bp]).to(device)
            di = torch.LongTensor([p[1] for p in bp]).to(device)
            opt.zero_grad()
            out = model(data, mi, di)
            loss = F.binary_cross_entropy_with_logits(out, bl)
            loss.backward()
            opt.step()
            train_logits.append(torch.sigmoid(out).detach().cpu().numpy())
            train_labels.append(bl.cpu().numpy())
        if (ep + 1) % eval_every == 0 or ep == 0:
            tr_auc = roc_auc_score(np.concatenate(train_labels),
                                    np.concatenate(train_logits))
            # val
            model.eval()
            v_neg = sample_neg(data["assoc"], val_pos, len(val_pos))
            v_pairs = val_pos + v_neg
            v_labs = np.array([1.0] * len(val_pos) + [0.0] * len(v_neg))
            mi = torch.LongTensor([p[0] for p in v_pairs]).to(device)
            di = torch.LongTensor([p[1] for p in v_pairs]).to(device)
            with torch.no_grad():
                pr = torch.sigmoid(model(data, mi, di)).cpu().numpy()
            v_auc = roc_auc_score(v_labs, pr)
            v_aupr = average_precision_score(v_labs, pr)
            history.append(dict(epoch=ep + 1, train_auc=float(tr_auc),
                                 val_auc=float(v_auc), val_aupr=float(v_aupr)))
            print(f"  epoch {ep + 1:3d}: train_auc={tr_auc:.4f}, "
                  f"val_auc={v_auc:.4f}, val_aupr={v_aupr:.4f}", flush=True)
    return history


def plot_convergence(history, out_path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [h["train_auc"] for h in history],
            "-", color="#3498db", lw=1.6, label="Train AUC")
    ax.plot(epochs, [h["val_auc"]   for h in history],
            "-", color="#e74c3c", lw=1.6, label="Validation AUC")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
    ax.set_title("Training dynamics on Dataset 1")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right"); ax.grid(linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"))
    plt.close()


# ============================================================
# Experiment 2: false-negative robustness
# ============================================================

def run_robustness(data, hp, fn_ratios=(0.0, 0.05, 0.1, 0.2, 0.3),
                   n_folds=3, n_epochs=80):
    """For each false-negative ratio, run reduced k-fold CV.

    `fn_ratio = 0.1` means 10% of training-fold positives are randomly
    relabeled as negatives, simulating a scenario where the database
    has 10% under-annotation.
    """
    pos = data["pos_pairs"]
    md = data["mirna_feat"].shape[1]; dd = data["drug_feat"].shape[1]
    nm = data["n_mirna"]; nd = data["n_drug"]; ng = data["n_gene"]
    results = {}

    for fn in fn_ratios:
        print(f"\n--- False-negative ratio = {fn:.0%} ---", flush=True)
        fold_aucs, fold_auprs = [], []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold, (tri, tei) in enumerate(kf.split(pos)):
            set_seed(42 + fold)
            train_pos = [pos[i] for i in tri]
            test_pos  = [pos[i] for i in tei]
            # Inject false negatives: relabel a random `fn` fraction of
            # training positives as negatives. We do this by simply
            # removing them from the positive list — they will then be
            # sampled into the negative pool just like other unobserved
            # pairs.
            n_drop = int(fn * len(train_pos))
            if n_drop > 0:
                drop_idx = np.random.choice(len(train_pos), n_drop, replace=False)
                kept = [p for k, p in enumerate(train_pos) if k not in set(drop_idx)]
                effective_train = kept
            else:
                effective_train = train_pos

            model = DrugMiR_Hybrid(nm, nd, md, dd, ng,
                                   h=hp["h"], dr=hp["dr"],
                                   n_gcn=hp["n_gcn"], n_br=hp["n_br"]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                                   weight_decay=hp["wd"])
            # Train (shorter than full training to keep total time manageable)
            best_auc, best_aupr = 0, 0
            patience = 0
            for ep in range(n_epochs):
                model.train()
                neg = sample_neg(data["assoc"], effective_train,
                                  len(effective_train))
                pairs = effective_train + neg
                labs = [1.0] * len(effective_train) + [0.0] * len(neg)
                idx = np.random.permutation(len(pairs))
                for s in range(0, len(idx), 2048):
                    bi = idx[s:s + 2048]
                    bp = [pairs[i] for i in bi]
                    bl = torch.FloatTensor([labs[i] for i in bi]).to(device)
                    mi = torch.LongTensor([p[0] for p in bp]).to(device)
                    di = torch.LongTensor([p[1] for p in bp]).to(device)
                    opt.zero_grad()
                    out = model(data, mi, di)
                    loss = F.binary_cross_entropy_with_logits(out, bl)
                    loss.backward()
                    opt.step()
                if (ep + 1) % 5 == 0:
                    model.eval()
                    t_neg = sample_neg(data["assoc"], test_pos, len(test_pos))
                    t_pairs = test_pos + t_neg
                    t_labs = np.array([1.0] * len(test_pos) + [0.0] * len(t_neg))
                    mi = torch.LongTensor([p[0] for p in t_pairs]).to(device)
                    di = torch.LongTensor([p[1] for p in t_pairs]).to(device)
                    with torch.no_grad():
                        pr = torch.sigmoid(model(data, mi, di)).cpu().numpy()
                    auc = roc_auc_score(t_labs, pr)
                    aupr = average_precision_score(t_labs, pr)
                    if auc > best_auc:
                        best_auc, best_aupr = auc, aupr
                        patience = 0
                    else:
                        patience += 1
                    if patience >= 5:
                        break
            fold_aucs.append(best_auc); fold_auprs.append(best_aupr)
            print(f"  fold {fold + 1}: AUC={best_auc:.4f}, AUPR={best_aupr:.4f}",
                  flush=True)
        results[f"{fn:.2f}"] = dict(
            fn_ratio=float(fn),
            auc_mean=float(np.mean(fold_aucs)),
            auc_std=float(np.std(fold_aucs)),
            aupr_mean=float(np.mean(fold_auprs)),
            aupr_std=float(np.std(fold_auprs)),
        )
    return results


def plot_robustness(results, out_path):
    items = sorted(results.values(), key=lambda r: r["fn_ratio"])
    xs    = [r["fn_ratio"] * 100 for r in items]
    aucs  = [r["auc_mean"]      for r in items]
    auc_s = [r["auc_std"]       for r in items]
    auprs = [r["aupr_mean"]     for r in items]
    auprs_s = [r["aupr_std"]    for r in items]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, aucs, yerr=auc_s, marker="o", color="#3498db", lw=1.8,
                capsize=4, label="AUC")
    ax.errorbar(xs, auprs, yerr=auprs_s, marker="s", color="#e74c3c", lw=1.8,
                capsize=4, label="AUPR")
    ax.set_xlabel("False-negative ratio in training (%)")
    ax.set_ylabel("Test performance")
    ax.set_title("Robustness to false negatives in training")
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc="lower left"); ax.grid(linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"))
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("DrugMiR Robustness + Convergence Analysis (§4.9)")
    print("=" * 60)
    print(f"Data dir: {DATA_DIR}")
    print(f"Device:   {device}\n")

    HP = dict(h=256, dr=0.5, n_gcn=2, n_br=2, lr=5e-4, wd=2e-4)
    print(f"Hyperparameters: {HP}\n")

    print("Loading data...")
    data = load_data(DATA_DIR, km=15, kd=10)
    print(f"  miRNAs={data['n_mirna']}, drugs={data['n_drug']}, "
          f"positives={len(data['pos_pairs'])}\n")

    # ----- Experiment 1: convergence -----
    print(">>> Experiment 1/2: Convergence (200 epochs)")
    t0 = time.time()
    history = run_convergence(data, HP, n_epochs=200, eval_every=2)
    json.dump(history,
              open(os.path.join(OUT_DIR, "convergence.json"), "w"), indent=2)
    plot_convergence(history, os.path.join(OUT_DIR, "fig_convergence.pdf"))
    print(f"Convergence done in {time.time() - t0:.0f}s\n")

    # ----- Experiment 2: robustness -----
    print(">>> Experiment 2/2: False-negative robustness")
    t0 = time.time()
    results = run_robustness(data, HP,
                             fn_ratios=(0.0, 0.05, 0.10, 0.20, 0.30),
                             n_folds=3, n_epochs=80)
    json.dump(results,
              open(os.path.join(OUT_DIR, "robustness.json"), "w"), indent=2)
    plot_robustness(results, os.path.join(OUT_DIR, "fig_robustness.pdf"))
    print(f"Robustness done in {time.time() - t0:.0f}s\n")

    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Files written to {OUT_DIR}:")
    for f in sorted(os.listdir(OUT_DIR)):
        full = os.path.join(OUT_DIR, f)
        print(f"  {f}  ({os.path.getsize(full) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
