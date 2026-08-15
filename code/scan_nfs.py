#!/usr/bin/env python3
"""
DrugMiR --- NFS inventory scan.

Run this FIRST on the CPU box, before any experiment. It reports what already
exists so nothing gets recomputed, and it tells you exactly which of the
pending experiments still need to run.

    python scan_nfs.py --root ~/work/DrugMiR

Pure stdlib + numpy. Reads nothing large; it stats files and peeks at JSON.
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------- catalogue

# (relative path, what it is, which pending task it would satisfy)
CATALOGUE = [
    # --- supplementary build (from the previous round) ---
    ("results/case_study.json",
     "Case-study Top-20 predictions per drug", "Supp Table S1 skeleton"),
    ("results/case_study_evidence.csv",
     "MANUAL 54-pair evidence audit (direct/pathway/none)", "Supp Table S1 evidence column"),
    ("results/case_study_scores.npy",
     "Averaged case-study score matrix", "Supp Table S1 (fallback)"),
    ("results_final/bridge_genes_shap_top20.csv",
     "GradientSHAP Top-20 bridge genes", "Supp Table S2"),
    ("results_final/bridge_genes_ig_top20.csv",
     "IG Top-20 bridge genes", "Supp Table S2 (IG rank column)"),
    ("results_final/gene_ig_importance_d1.npy",
     "Full per-gene IG importance vector", "Supp Table S2 + gene knockout"),
    ("results_final/gene_shap_importance_d1.npy",
     "Full per-gene GradientSHAP vector", "Supp Table S2"),
    ("results_final/ig_vs_shap_agreement.json",
     "IG-vs-SHAP agreement stats", "Supp Table S2 caption"),
    ("results_robustness/convergence.json",
     "200-epoch convergence history", "Supp Note S1"),
    ("results_robustness/robustness.json",
     "False-negative sweep results", "Supp Note S1"),
    ("results_robustness/fig_convergence.pdf",
     "Convergence figure", "Supp Fig S1"),
    ("results_robustness/fig_robustness.pdf",
     "False-negative figure", "Supp Fig S2"),

    # --- inputs the new experiments need ---
    ("results_final/drugmir_d1_seed42_for_ig.pt",
     "TRAINED D1 CHECKPOINT (state_dict+config+shapes)",
     "gene knockout + imbalanced eval  << CRITICAL"),
    ("data/processed/mirna_gene_matrix.npy",
     "miRNA-gene adjacency", "gene knockout (degree control)"),
    ("data/processed/drug_gene_matrix.npy",
     "drug-gene adjacency", "gene knockout (degree control)"),
    ("data/processed/gene_mapping.csv",
     "gene index -> name", "gene knockout reporting"),
    ("results_final/drugmir_predictions_full_seed42.json",
     "DrugMiR per-sample (y_true,y_pred)", "Fig 2 (own curves only)"),

    # --- outputs of the NEW experiments: if present, already done ---
    ("results_knockout/gene_knockout.json",
     "Gene-knockout ablation results", "NEW EXP 1 -- already done?"),
    ("results_imbalanced/imbalanced_eval.json",
     "1:1 / 1:5 / 1:10 evaluation", "NEW EXP 2 -- already done?"),
    ("results_similarity/similarity_stats.json",
     "Similarity-distribution stats", "NEW EXP 3 -- already done?"),
]

# directories worth listing wholesale, in case something was run ad hoc
SCAN_DIRS = ["results", "results_final", "results_robustness", "results_fusion",
             "results_mask", "results_efficiency", "results_knockout",
             "results_imbalanced", "results_similarity", "data/processed"]

# filename fragments that would indicate one of the new experiments was already
# run under a different name
HINTS = [("knockout", "gene knockout"), ("ablat_gene", "gene knockout"),
         ("imbalanc", "imbalanced eval"), ("ratio", "imbalanced eval"),
         ("neg5", "imbalanced eval"), ("neg10", "imbalanced eval"),
         ("similar", "similarity distribution"), ("tanimoto", "similarity distribution"),
         ("kmer_sim", "similarity distribution"), ("baseline_pred", "Fig 2 baseline curves"),
         ("_preds", "saved predictions")]


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/work/DrugMiR"))
    ap.add_argument("--deep", action="store_true",
                    help="also walk the whole root looking for stray result files")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)

    if not os.path.isdir(root):
        sys.exit(f"ABORT: root not found: {root}\n"
                 f"Pass the right path with --root (e.g. --root ~/DrugMiR)")

    print("=" * 78)
    print(f"DrugMiR NFS scan   root = {root}")
    print("=" * 78)

    # ---------------- catalogue check ----------------
    print("\n### 1. Expected files\n")
    have, missing = [], []
    for rel, what, need in CATALOGUE:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            st = os.stat(p)
            ts = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  [ok]      {rel}")
            print(f"            {what}  ({human(st.st_size)}, {ts})")
            have.append(rel)
        else:
            print(f"  [MISSING] {rel}")
            print(f"            {what}   -> needed for: {need}")
            missing.append((rel, need))

    # ---------------- directory listings ----------------
    print("\n### 2. Directory contents\n")
    for d in SCAN_DIRS:
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            print(f"  (no {d}/)")
            continue
        entries = sorted(os.listdir(p))
        print(f"  {d}/  ({len(entries)} entries)")
        for e in entries:
            fp = os.path.join(p, e)
            if os.path.isfile(fp):
                print(f"      {e:<52} {human(os.path.getsize(fp))}")
            else:
                print(f"      {e}/")

    # ---------------- hint sweep ----------------
    print("\n### 3. Files whose names hint at an already-run experiment\n")
    hits = []
    for dp, _, fns in os.walk(root):
        if any(x in dp for x in (".git", "__pycache__", "node_modules")):
            continue
        for fn in fns:
            low = fn.lower()
            for frag, label in HINTS:
                if frag in low:
                    hits.append((os.path.relpath(os.path.join(dp, fn), root), label))
                    break
    if hits:
        for rel, label in sorted(set(hits)):
            print(f"  {rel:<58} -> {label}")
    else:
        print("  (none)")

    # ---------------- peek at key JSONs ----------------
    print("\n### 4. Contents of key result files\n")
    for rel in ["results_final/ig_vs_shap_agreement.json",
                "results_robustness/robustness.json",
                "results_knockout/gene_knockout.json",
                "results_imbalanced/imbalanced_eval.json"]:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                j = json.load(f)
            print(f"  {rel}:")
            txt = json.dumps(j, indent=4)
            for line in txt.splitlines()[:24]:
                print("    " + line)
            if len(txt.splitlines()) > 24:
                print("    ...")
            print()
        except Exception as e:
            print(f"  {rel}: could not parse ({e})\n")

    # ---------------- checkpoint sanity ----------------
    ck = os.path.join(root, "results_final/drugmir_d1_seed42_for_ig.pt")
    if os.path.exists(ck):
        print("\n### 5. Checkpoint metadata\n")
        try:
            import torch
            c = torch.load(ck, map_location="cpu", weights_only=False)
            print(f"  config      = {c.get('config')}")
            print(f"  shapes      = {c.get('shapes')}")
            print(f"  best_val_auc= {c.get('best_val_auc')}")
            print("\n  -> gene knockout and imbalanced evaluation can run on CPU "
                  "from this checkpoint, inference only, no retraining.")
        except ImportError:
            print("  (torch not installed here; skipped)")
        except Exception as e:
            print(f"  could not read: {e}")

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  present: {len(have)} / {len(CATALOGUE)}")
    if missing:
        print("\n  still missing:")
        for rel, need in missing:
            print(f"    - {rel}\n        needed for: {need}")
    crit = os.path.join(root, "results_final/drugmir_d1_seed42_for_ig.pt")
    print()
    if os.path.exists(crit):
        print("  CHECKPOINT PRESENT -> run exp_gene_knockout.py and "
              "exp_imbalanced.py on CPU.")
    else:
        print("  CHECKPOINT ABSENT  -> gene knockout and imbalanced evaluation "
              "need train_save_for_ig.py rerun first (GPU).")
    print("  exp_similarity_dist.py needs only data/processed/ -> always runnable.")
    print()


if __name__ == "__main__":
    main()
