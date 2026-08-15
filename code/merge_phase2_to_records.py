#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_phase2_to_records.py  --  Convert existing phase2_outputs/*.json
into the flat `records` format expected by sig_analysis.py.

Each input JSON is expected to look like (the structure produced by the
Phase-2 runners already on disk):

    {
      "D1": {"auc": {...}, "aupr": {...}, ...,
             "fold_details": [{"auc":..,"aupr":..,"f1":..,"prec":..,"rec":..,"thr":..}, ...],
             "dataset": "D1", "seed": 42, ...},
      "D2": { ...same structure... }
    }

For every (file, dataset, fold) we emit one record with capitalized metric
keys (AUC/AUPR/F1/Prec/Rec) so it plugs into sig_analysis.py without flags.

Method name is inferred from the filename: `{method}_5metrics_seed{n}.json`.
You can override the mapping with --rename (e.g. drugmir->DrugMiR).

Multi-seed: just point --indir at a directory containing both
`drugmir_5metrics_seed42.json` and `drugmir_5metrics_seed123.json` etc.;
each file's own "seed" field is used (with filename as fallback).
"""

import os, re, json, glob, argparse, sys

# -----------------------------------------------------------------------------
# Defaults: rename file-stem -> nice method name as used in the paper
# -----------------------------------------------------------------------------
DEFAULT_RENAME = {
    "drugmir":  "DrugMiR",
    "mphgnn":   "MPHGNN",
    "gammdr":   "GAM-MDR",
    "gam_mdr":  "GAM-MDR",
    "gslrda":   "GSLRDA",
    "dmrpeg":   "DMR-PEG",
    "dmr_peg":  "DMR-PEG",
    "dmgat":    "DMGAT",
    "lrgcpnd":  "LRGCPND",
}

# Map phase2 lowercase metric -> capitalised metric used in sig_analysis
METRIC_MAP = {"auc":"AUC","aupr":"AUPR","f1":"F1","prec":"Prec","rec":"Rec"}

# -----------------------------------------------------------------------------
def parse_filename(fn):
    """Return (method_key, seed_from_name) from '{method}_5metrics_seed{n}[_FINAL].json'."""
    base = os.path.basename(fn).replace(".json","")
    # strip _FINAL or other suffixes after seed
    m = re.match(r"^([a-zA-Z0-9_\-]+?)_5metrics_seed(\d+)(?:_.*)?$", base)
    if m:
        return m.group(1).lower(), int(m.group(2))
    # fallback: take leading token
    return base.lower().split("_")[0], None

def to_records(path, rename):
    method_key, seed_from_name = parse_filename(path)
    nice = rename.get(method_key, method_key)
    try:
        data = json.load(open(path))
    except Exception as e:
        print(f"  [skip] {path}: load error {e}", file=sys.stderr)
        return []
    if not isinstance(data, dict):
        print(f"  [skip] {path}: top-level is not a dict", file=sys.stderr)
        return []
    recs = []
    for ds, payload in data.items():
        if not isinstance(payload, dict):
            continue
        fd = payload.get("fold_details", [])
        if not isinstance(fd, list) or not fd:
            print(f"  [skip] {path}::{ds}: no fold_details", file=sys.stderr)
            continue
        seed = payload.get("seed", seed_from_name)
        if seed is None:
            print(f"  [warn] {path}::{ds}: cannot determine seed", file=sys.stderr)
            seed = -1
        for i, row in enumerate(fd):
            if not isinstance(row, dict):
                continue
            r = {"method": nice, "dataset": ds, "seed": int(seed), "fold": i}
            for src_k, dst_k in METRIC_MAP.items():
                if src_k in row and row[src_k] is not None:
                    try: r[dst_k] = float(row[src_k])
                    except: pass
            recs.append(r)
    return recs

# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="phase2_outputs",
                    help="dir containing *_5metrics_seed*.json")
    ap.add_argument("--glob", default="*_5metrics_seed*.json",
                    help="glob pattern (default picks up multi-seed too)")
    ap.add_argument("--out", default="phase2_outputs/multiseed_records.json")
    ap.add_argument("--rename", nargs="*", default=[],
                    help="extra renames key=Name (e.g. drugmir=DrugMiR)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="filename substrings to exclude (case-insensitive)")
    args = ap.parse_args()

    rename = dict(DEFAULT_RENAME)
    for kv in args.rename:
        if "=" in kv:
            k,v = kv.split("=",1); rename[k.lower()] = v

    files = sorted(glob.glob(os.path.join(args.indir, args.glob)))
    # prefer *_FINAL.json over the non-final twin if both exist for the same method+seed
    by_key = {}
    for f in files:
        if any(x.lower() in os.path.basename(f).lower() for x in args.exclude):
            continue
        mk, sd = parse_filename(f)
        key = (mk, sd)
        if key not in by_key or "_FINAL" in os.path.basename(f).upper():
            by_key[key] = f
    files = sorted(by_key.values())

    if not files:
        sys.exit(f"ERROR: no JSON matched {args.indir}/{args.glob}")

    print(f"[merge] discovered {len(files)} file(s):")
    for f in files: print(f"   - {os.path.basename(f)}")

    all_records = []
    for f in files:
        recs = to_records(f, rename)
        all_records.extend(recs)
        print(f"   {os.path.basename(f):42s} -> +{len(recs)} records")

    # quick coverage table
    cov = {}
    for r in all_records:
        cov.setdefault((r["method"], r["dataset"]), set()).add((r["seed"], r["fold"]))
    print("\n[merge] coverage (method, dataset) -> #(seed,fold) blocks:")
    methods = sorted({k[0] for k in cov}); datasets = sorted({k[1] for k in cov})
    print("  " + " "*22 + "  " + "  ".join(f"{d:>6}" for d in datasets))
    for m in methods:
        row = [f"{len(cov.get((m,d), set())):>6d}" for d in datasets]
        print(f"  {m:>22s}  " + "  ".join(row))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"records": all_records}, f, indent=1)
    print(f"\n[merge] wrote {args.out}  ({len(all_records)} records)")

if __name__ == "__main__":
    main()
