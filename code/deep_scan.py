#!/usr/bin/env python3
"""
DrugMiR --- deep recursive inventory.

scan_nfs.py checks a fixed checklist. This does the opposite: it walks the whole
tree and reports what is actually there, so nothing that was produced in an
earlier session stays invisible.

Output is tiered so the result stays readable:

  1  header + totals
  2  top-level map (one line per entry)
  3  full listing of CORE directories
  4  EVERY .py anywhere, with its first docstring line
  5  every result artifact outside third-party trees
  6  content peek: JSON keys, CSV header + row count, Markdown headings, log tail
  7  every .tex / .pdf / .bib anywhere (manuscript versions)
  8  largest files
  9  recently modified
 10  duplicate basenames (the same filename living in several places)

Third-party baseline trees (MPHGNN/, DMGAT/, ...) are summarised rather than
enumerated, except for their .py and small result files.

USAGE
-----
    python deep_scan.py --root ~/work/DrugMiR
    python deep_scan.py --root ~/work/DrugMiR --full     # no caps, much longer

WRITES
------
    deep_scan.txt    human-readable, this is the one to send back
    deep_scan.json   same inventory, machine-readable, if the txt is too big

Pure stdlib. Runs anywhere in seconds.
"""

import argparse
import csv
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta

# directories summarised rather than walked in full
THIRD_PARTY = {"AMMGC", "DGNNMDA", "DGNNMDA_processed", "DMGAT", "DMGAT_processed",
               "DMR-PEG", "GSLRDA", "LRGCPND", "MPHGNN", "ncRNADrug_processed",
               "MGCNA", "B-NDRA", "GCFMCL"}
SKIP = {"__pycache__", ".git", ".ipynb_checkpoints", "node_modules", ".cache",
        ".mp_cache", "_mp_cache", ".claude", ".vscode", ".idea"}

# directories worth listing file by file
CORE_HINTS = ("results", "data", "figures", "figure", "case_study", "scripts",
              "_session", "supp", "tex", "paper", "manuscript", "logs", "log",
              "outputs", "output", "ablation", "notebooks")

RESULT_EXT = {".json", ".csv", ".tsv", ".npy", ".npz", ".pt", ".pth", ".pkl",
              ".log", ".md", ".txt", ".pdf", ".png", ".jpg", ".svg", ".eps",
              ".tex", ".bib", ".xlsx", ".yaml", ".yml"}
PEEK_EXT = {".json", ".csv", ".md", ".txt", ".log", ".yaml", ".yml"}

out_buf = io.StringIO()


def w(line=""):
    out_buf.write(str(line) + "\n")


def human(n):
    n = float(n)
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024 or u == "T":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def mt(p):
    try:
        return datetime.fromtimestamp(os.path.getmtime(p))
    except OSError:
        return datetime.fromtimestamp(0)


def ts(p):
    return mt(p).strftime("%Y-%m-%d %H:%M")


def is_core(rel):
    low = rel.lower()
    return any(h in low for h in CORE_HINTS)


def walk(root):
    """Yield (dirpath, dirnames, filenames), pruning SKIP, following no symlinks."""
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = sorted(d for d in dns if d not in SKIP and not d.startswith("."))
        yield dp, dns, sorted(fns)


def first_doc_line(path, limit=140):
    """First meaningful line of a .py module docstring, else first code line."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = [next(f, "") for _ in range(30)]
    except OSError:
        return ""
    started = False
    for ln in head:
        s = ln.strip()
        if not started:
            if s.startswith(('"""', "'''")):
                started = True
                s = s.lstrip("\"'").strip()
                if s:
                    return s[:limit]
                continue
            if s.startswith("#") and len(s) > 3:
                return s.lstrip("# ").strip()[:limit]
        else:
            if s and not s.startswith(('"""', "'''")):
                return s[:limit]
            if s.startswith(('"""', "'''")):
                return ""
    return ""


def peek(path, ext, maxlen=700):
    """Short, informative summary of a small text file."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "(unreadable)"
    if size > 3_000_000:
        return f"(skipped, {human(size)})"
    try:
        if ext == ".json":
            with open(path, encoding="utf-8", errors="replace") as f:
                j = json.load(f)
            if isinstance(j, dict):
                keys = list(j.keys())
                s = f"dict, {len(keys)} keys: {keys[:18]}"
                for k in keys[:6]:
                    v = j[k]
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        s += f"\n        {k} = {str(v)[:90]}"
                    elif isinstance(v, list):
                        s += f"\n        {k} = list[{len(v)}]"
                    elif isinstance(v, dict):
                        s += f"\n        {k} = dict{list(v.keys())[:8]}"
                return s
            if isinstance(j, list):
                s = f"list[{len(j)}]"
                if j and isinstance(j[0], dict):
                    s += f", item keys {list(j[0].keys())[:12]}"
                return s
            return f"{type(j).__name__}"
        if ext in (".csv", ".tsv"):
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                r = csv.reader(f, delimiter="\t" if ext == ".tsv" else ",")
                rows = []
                for i, row in enumerate(r):
                    rows.append(row)
                    if i > 3:
                        break
                n = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
            s = f"{n} data rows | header: {rows[0][:14] if rows else '?'}"
            if len(rows) > 1:
                s += f"\n        row1: {rows[1][:14]}"
            return s
        if ext == ".md":
            heads = []
            with open(path, encoding="utf-8", errors="replace") as f:
                for ln in f:
                    if ln.startswith("#"):
                        heads.append(ln.strip()[:110])
                    if len(heads) >= 10:
                        break
            return " / ".join(heads) if heads else "(no headings)"
        # txt, log, yaml -> head and tail
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        if not lines:
            return "(empty)"
        head = " | ".join(l.strip() for l in lines[:3] if l.strip())
        tail = " | ".join(l.strip() for l in lines[-3:] if l.strip())
        s = f"head: {head[:300]}"
        if len(lines) > 6:
            s += f"\n        tail: {tail[:300]}"
        return s
    except Exception as e:
        return f"(peek failed: {type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/work/DrugMiR"))
    ap.add_argument("--out", default=None, help="default <root>/deep_scan.txt")
    ap.add_argument("--full", action="store_true", help="remove all caps")
    ap.add_argument("--recent-days", type=int, default=90)
    a = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(a.root))
    if not os.path.isdir(root):
        sys.exit(f"ABORT: not a directory: {root}")
    cap_dir = 10**9 if a.full else 60
    cap_peek = 10**9 if a.full else 260

    t0 = time.time()
    inv = {"root": root, "generated": datetime.now().isoformat(), "dirs": {}}

    # ---------- gather ----------
    all_files = []          # (relpath, size, mtime)
    per_dir = {}            # reldir -> [(name, size, mtime)]
    for dp, dns, fns in walk(root):
        rel = os.path.relpath(dp, root)
        rel = "." if rel == "." else rel
        entries = []
        for fn in fns:
            fp = os.path.join(dp, fn)
            try:
                if os.path.islink(fp):
                    continue
                sz = os.path.getsize(fp)
            except OSError:
                continue
            entries.append((fn, sz, mt(fp)))
            all_files.append((os.path.join(rel, fn) if rel != "." else fn, sz, mt(fp)))
        per_dir[rel] = entries

    total_sz = sum(s for _, s, _ in all_files)

    # ---------- 1 header ----------
    w("=" * 100)
    w(f"DrugMiR DEEP SCAN     root = {root}")
    w(f"generated {datetime.now():%Y-%m-%d %H:%M}   "
      f"{len(all_files)} files   {len(per_dir)} directories   {human(total_sz)} total")
    w("=" * 100)

    # ---------- 2 top-level map ----------
    w("\n\n### 1. TOP-LEVEL MAP\n")
    top = sorted(os.listdir(root))
    for name in top:
        p = os.path.join(root, name)
        if name in SKIP:
            continue
        if os.path.isdir(p):
            nf = ns = 0
            newest = datetime.fromtimestamp(0)
            for dp2, dns2, fns2 in walk(p):
                dns2[:] = [d for d in dns2 if d not in SKIP]
                for fn in fns2:
                    fp = os.path.join(dp2, fn)
                    try:
                        ns += os.path.getsize(fp)
                    except OSError:
                        continue
                    nf += 1
                    newest = max(newest, mt(fp))
            tag = "  [third-party, summarised]" if name in THIRD_PARTY else ""
            w(f"  {name + '/':<34} {nf:>6} files  {human(ns):>8}   "
              f"newest {newest:%Y-%m-%d}{tag}")
        else:
            w(f"  {name:<34} {'':>6}        {human(os.path.getsize(p)):>8}   "
              f"{ts(p)}")

    # ---------- 3 core directory listings ----------
    w("\n\n### 2. CORE DIRECTORIES, FILE BY FILE\n")
    for rel in sorted(per_dir):
        head = rel.split(os.sep)[0]
        if head in THIRD_PARTY:
            continue
        if rel != "." and not is_core(rel):
            continue
        entries = per_dir[rel]
        if not entries:
            w(f"  {rel}/   (empty)")
            continue
        w(f"  {rel}/   ({len(entries)} files)")
        for fn, sz, m in entries[:cap_dir]:
            w(f"      {fn:<52} {human(sz):>8}  {m:%Y-%m-%d %H:%M}")
        if len(entries) > cap_dir:
            w(f"      ... {len(entries) - cap_dir} more (use --full)")
        w("")

    # ---------- 4 every .py ----------
    w("\n### 3. EVERY PYTHON FILE IN THE TREE\n")
    pys = sorted([f for f in all_files if f[0].endswith(".py")],
                 key=lambda x: x[0])
    w(f"  {len(pys)} .py files\n")
    for rel, sz, m in pys:
        third = rel.split(os.sep)[0] in THIRD_PARTY
        mark = " [3rd-party]" if third else ""
        w(f"  {rel:<62} {human(sz):>7}  {m:%Y-%m-%d}{mark}")
        d = first_doc_line(os.path.join(root, rel))
        if d and not third:
            w(f"        -> {d}")

    # ---------- 5 result artifacts ----------
    w("\n\n### 4. RESULT ARTIFACTS (non third-party)\n")
    arts = sorted([f for f in all_files
                   if os.path.splitext(f[0])[1].lower() in RESULT_EXT
                   and f[0].split(os.sep)[0] not in THIRD_PARTY
                   and not f[0].endswith(".py")],
                  key=lambda x: x[0])
    w(f"  {len(arts)} artifacts\n")
    for rel, sz, m in arts:
        w(f"  {rel:<62} {human(sz):>7}  {m:%Y-%m-%d %H:%M}")

    # ---------- 6 content peek ----------
    w("\n\n### 5. CONTENT PEEK (small text artifacts)\n")
    shown = 0
    for rel, sz, m in arts:
        ext = os.path.splitext(rel)[1].lower()
        if ext not in PEEK_EXT or sz > 3_000_000:
            continue
        if shown >= cap_peek:
            w(f"\n  ... peek cap reached, use --full for the rest")
            break
        w(f"  {rel}  ({human(sz)}, {m:%Y-%m-%d %H:%M})")
        w(f"        {peek(os.path.join(root, rel), ext)}")
        w("")
        shown += 1

    # ---------- 7 manuscript files ----------
    w("\n### 6. MANUSCRIPT FILES (.tex / .bib / .pdf) ANYWHERE\n")
    docs = sorted([f for f in all_files
                   if os.path.splitext(f[0])[1].lower() in (".tex", ".bib", ".pdf")],
                  key=lambda x: -x[2].timestamp())
    for rel, sz, m in docs:
        w(f"  {m:%Y-%m-%d %H:%M}  {human(sz):>8}  {rel}")

    # ---------- 8 largest ----------
    w("\n\n### 7. 40 LARGEST FILES\n")
    for rel, sz, m in sorted(all_files, key=lambda x: -x[1])[:40]:
        w(f"  {human(sz):>8}  {m:%Y-%m-%d}  {rel}")

    # ---------- 9 recent ----------
    cutoff = datetime.now() - timedelta(days=a.recent_days)
    w(f"\n\n### 8. MODIFIED IN THE LAST {a.recent_days} DAYS\n")
    rec = sorted([f for f in all_files if f[2] > cutoff],
                 key=lambda x: -x[2].timestamp())
    w(f"  {len(rec)} files\n")
    for rel, sz, m in rec[:200]:
        w(f"  {m:%Y-%m-%d %H:%M}  {human(sz):>8}  {rel}")
    if len(rec) > 200:
        w(f"  ... {len(rec) - 200} more")

    # ---------- 10 duplicate basenames ----------
    w("\n\n### 9. DUPLICATE FILENAMES (same name, several locations)\n")
    by_name = {}
    for rel, sz, m in all_files:
        by_name.setdefault(os.path.basename(rel), []).append((rel, sz, m))
    dups = {k: v for k, v in by_name.items()
            if len(v) > 1 and os.path.splitext(k)[1].lower() in
            (RESULT_EXT | {".py"})}
    w(f"  {len(dups)} names appear more than once\n")
    for k in sorted(dups):
        v = sorted(dups[k], key=lambda x: -x[2].timestamp())
        if len(v) > 6 and not a.full:
            v = v[:6]
        w(f"  {k}")
        for rel, sz, m in v:
            w(f"      {m:%Y-%m-%d %H:%M}  {human(sz):>7}  {rel}")

    w("\n" + "=" * 100)
    w(f"done in {time.time() - t0:.1f}s")
    w("=" * 100)

    # ---------- write ----------
    out_txt = a.out or os.path.join(root, "deep_scan.txt")
    text = out_buf.getvalue()
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    inv["files"] = [{"path": r, "bytes": s, "mtime": m.isoformat()}
                    for r, s, m in all_files]
    out_json = os.path.splitext(out_txt)[0] + ".json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=1)

    print(text[:4000])
    print("\n" + "-" * 70)
    print(f"FULL REPORT: {out_txt}   ({human(len(text.encode()))}, "
          f"{text.count(chr(10))} lines)")
    print(f"INVENTORY  : {out_json}   ({human(os.path.getsize(out_json))})")
    print("-" * 70)
    print("Send deep_scan.txt back. If it is large, scp it down rather than")
    print("pasting:  scp -P <port> featurize@<host>:work/DrugMiR/deep_scan.txt .")


if __name__ == "__main__":
    main()
