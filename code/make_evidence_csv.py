#!/usr/bin/env python3
"""
DrugMiR --- build case_study_evidence.csv from the recovered literature audit.

WHY THIS EXISTS
---------------
case_study_litreview/ contains two generations of the same audit:

  progress_final.json   (04:18)  54 verdicts, PRE re-verification  -> 14/18/22
  audit_summary.md      (04:27)  re-checked the 14 "Direct" rows   ->  8/24/22

The re-verification demoted six entries from Direct to Pathway and corrected
several mis-attributed first authors. It is the authoritative record, but it was
never written back into the JSON, and the manuscript quotes the PRE-audit split.

This script reads the JSON, applies the audit as an explicit override table, and
writes the CSV that make_supp.py consumes. It refuses to write if the resulting
counts do not come out to 8/24/22, so a silent mismatch cannot slip through.

USAGE
-----
    python make_evidence_csv.py --root ~/work/DrugMiR

OUTPUT
------
    results_final/case_study_evidence.csv
"""

import argparse
import csv
import json
import os
import sys

# --- the six Direct -> Pathway demotions from audit_summary.md ---------------
DOWNGRADES = {
    ("Gemcitabine", 3):  "PMC6678460 could not be verified for miR-103a-3p with gemcitabine specifically",
    ("Gemcitabine", 7):  "Bhutia et al. PLoS One 2013 studies let-7a, not let-7g specifically",
    ("Gemcitabine", 9):  "Li et al. Cancer Res 2009 does not distinguish the 3p from the 5p strand",
    ("Sorafenib",   6):  "Shimizu et al. J Hepatol 2010 studies let-7c and let-7g, not let-7e",
    ("Docetaxel",   6):  "srep41309 concerns miR-26a/miR-30b under trastuzumab; no primary for miR-221 with docetaxel",
    ("Docetaxel",  10):  "Di et al. 2019 concerns lapatinib, not docetaxel; nearest is Zang et al. 2020 on paclitaxel (same taxane class)",
}

# --- verified primary references for the eight surviving Direct rows ---------
VERIFIED_REF = {
    ("Doxorubicin", 8):  "Zhao L et al., J Exp Clin Cancer Res 2016;35:25 (PMC4738800)",
    ("Doxorubicin", 10): "Xu SL et al., Clin Breast Cancer (via review PMC11949885)",
    ("Cisplatin",   7):  "Li W et al., Cancer Cell Int 2016;16:30 (PMC4828824)",
    ("Cisplatin",   8):  "Pan X et al., Gene 2024;927:148738 (PMID 38955306)",
    ("Tamoxifen",   5):  "Yang S et al., Reprod Biol Endocrinol 2022 (PMC9524098)",
    ("Sorafenib",   3):  "Li L et al., Cell Death Discov 2022;8:297 (PMC9237098)",
    ("Sorafenib",   4):  "Qiu Y et al., Cell Death Discov 2019;5:120 (PMC6642098)",
    ("Docetaxel",   7):  "Souza MF et al., Biomolecules 2022;12(2):187 (PMC8961520)",
}

EXPECTED = {"direct": 8, "pathway": 24, "none": 22}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/work/DrugMiR"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.path.expanduser(a.root)

    src = os.path.join(root, "case_study_litreview/progress_final.json")
    out = a.out or os.path.join(root, "results_final/case_study_evidence.csv")
    if not os.path.exists(src):
        sys.exit(f"ABORT: {src} not found")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open(src) as f:
        j = json.load(f)
    rows_in = j["checked"]
    print(f"  read {len(rows_in)} audited pairs from progress_final.json")
    print(f"  pre-audit stats in file: {j.get('stats_so_far')}")

    counts = {"direct": 0, "pathway": 0, "none": 0}
    applied = 0
    rows_out = []

    for r in rows_in:
        drug, rank = r["drug"], int(r["rank"])
        verdict = str(r["verdict"]).strip().lower()
        note = r.get("evidence", "").strip()
        ref = VERIFIED_REF.get((drug, rank), "")

        key = (drug, rank)
        if key in DOWNGRADES:
            if verdict != "direct":
                print(f"  [warn] {drug} rank {rank}: expected Direct before "
                      f"downgrade, found '{verdict}'")
            verdict = "pathway"
            note = f"[demoted from Direct on re-verification] {DOWNGRADES[key]}"
            applied += 1

        counts[verdict] += 1
        rows_out.append({
            "drug": drug,
            "rank": rank,
            "mirna": r["mirna"],
            "score": r.get("score", ""),
            "evidence": verdict,
            "reference": ref,
            "note": note,
        })

    print(f"  applied {applied}/6 downgrades")
    print(f"  post-audit counts: {counts}")

    if counts != EXPECTED:
        sys.exit(f"ABORT: counts {counts} != expected {EXPECTED}. "
                 f"Do not write a CSV that disagrees with audit_summary.md; "
                 f"reconcile first.")

    total = sum(counts.values())
    supported = counts["direct"] + counts["pathway"]
    print(f"  supported = {supported}/{total} = {100 * supported / total:.1f}%"
          f"   (manuscript says 32/54 = 59.3%)")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["drug", "rank", "mirna", "score",
                                          "evidence", "reference", "note"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n  written: {out}")
    print("  now run:  python make_supp.py --root ~/work/DrugMiR")


if __name__ == "__main__":
    main()
