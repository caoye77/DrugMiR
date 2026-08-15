#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_multiseed.py (v3) -- v2 + subprocess fallback for mphgnn.

v2 import-as-module works for 6/7 methods but deadlocks for mphgnn (likely
dgl + module-level side effects). v3 uses subprocess for mphgnn (writes a
tiny stub script and runs it as a separate process), reading the JSON it
writes to disk. Other 6 methods unchanged.
"""

import os, sys, json, time, argparse, traceback, subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "MPHGNN"))

DD1_LEGACY = os.path.expanduser("~/DrugMiR/data/dataset1")
DD2_LEGACY = os.path.expanduser("~/DrugMiR/data/dataset2")
DD1_NEW    = os.path.expanduser("~/work/DrugMiR/data/processed")
DD2_NEW    = os.path.expanduser("~/work/DrugMiR/DMGAT_processed")

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _call_drugmir(mod, name, data_dir, seed, device):
    return mod.run_one_dataset(name, data_dir, km=15, kd=10,
                               lr=5e-4, wd=2e-4, seed=seed)

def _call_lrgcpnd(mod, name, data_dir, seed, device):
    return mod.run_lrgcpnd_dataset(name, data_dir, seed=seed)

def _call_gslrda(mod, name, data_dir, seed, device):
    return mod.run_gslrda_dataset(name, data_dir, seed=seed)

def _call_dmgat(mod, name, data_dir, seed, device):
    return mod.run_dmgat_dataset(name, data_dir, device, seed=seed, n_fold=5)

def _call_dmrpeg(mod, name, data_dir, seed, device):
    return mod.run_dmrpeg_dataset(name, data_dir, device, seed=seed,
                                  n_fold=5, num_epochs=20)

def _call_gammdr(mod, name, data_dir, seed, device):
    return mod.run_gam_dataset(name, data_dir, device, seed=seed,
                               n_fold=5, epoch=80, layer="gcn")


METHODS = {
    "drugmir": {"script": "run_drugmir_5metrics_seed42.py",  "call": _call_drugmir,
                "d1": DD1_LEGACY, "d2": DD2_LEGACY, "subprocess": False},
    "lrgcpnd": {"script": "run_lrgcpnd_d1d2_seed42.py",      "call": _call_lrgcpnd,
                "d1": DD1_LEGACY, "d2": DD2_LEGACY, "subprocess": False},
    "gslrda":  {"script": "run_gslrda_d1d2_seed42.py",       "call": _call_gslrda,
                "d1": DD1_LEGACY, "d2": DD2_LEGACY, "subprocess": False},
    "dmgat":   {"script": "MPHGNN/run_dmgat_d1d2_seed42.py", "call": _call_dmgat,
                "d1": DD1_NEW,    "d2": DD2_NEW,    "subprocess": False},
    "dmrpeg":  {"script": "MPHGNN/run_dmrpeg_d1d2_seed42.py","call": _call_dmrpeg,
                "d1": DD1_NEW,    "d2": DD2_NEW,    "subprocess": False},
    "gammdr":  {"script": "MPHGNN/run_gammdr_d1d2_seed42.py","call": _call_gammdr,
                "d1": DD1_NEW,    "d2": DD2_NEW,    "subprocess": False},
    "mphgnn":  {"script": "MPHGNN/run_mphgnn_d1d2_seed42.py","call": None,
                "d1": "MPHGNN/MiDrug_data_D1.pth",
                "d2": "MPHGNN/MiDrug_data_D2.pth",
                "subprocess": True},
}


def import_script_as_module(script_path):
    full = os.path.join(ROOT, script_path)
    src = open(full).read()
    for marker in ("if __name__ == '__main__':", 'if __name__ == "__main__":'):
        cut = src.find(marker)
        if cut > 0:
            src = src[:cut]; break
    mod_name = "_runner_" + os.path.basename(script_path).replace(".py", "").replace("-", "_")
    mod = type(sys)(mod_name)
    mod.__file__ = full; mod.__name__ = mod_name
    exec(compile(src, full, "exec"), mod.__dict__)
    return mod


def run_mphgnn_subprocess(seed, log):
    """Run mphgnn via subprocess to dodge the dgl-related import deadlock."""
    # Write a small stub script that imports the runner and invokes it.
    stub = os.path.join(ROOT, f"_mphgnn_stub_seed{seed}.py")
    stub_src = f'''
import os, sys, json
sys.path.insert(0, os.path.join({ROOT!r}, "MPHGNN"))
os.chdir({ROOT!r})
# Use the runner's run_mphgnn_dataset function with target seed.
spec_path = os.path.join({ROOT!r}, "MPHGNN", "run_mphgnn_d1d2_seed42.py")
src = open(spec_path).read()
cut = src.find("if __name__ == '__main__':")
if cut > 0:
    src = src[:cut]
ns = {{}}
ns["__name__"] = "_runner_mphgnn"
ns["__file__"] = spec_path
exec(compile(src, spec_path, "exec"), ns)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = {{}}
out["D1"] = ns["run_mphgnn_dataset"]("D1", "MPHGNN/MiDrug_data_D1.pth", device, seed={seed})
out["D2"] = ns["run_mphgnn_dataset"]("D2", "MPHGNN/MiDrug_data_D2.pth", device, seed={seed})
os.makedirs("phase2_outputs", exist_ok=True)
with open("phase2_outputs/mphgnn_5metrics_seed{seed}.json", "w") as f:
    json.dump(out, f, indent=2)
print("STUB_DONE")
'''
    open(stub, "w").write(stub_src)
    log(f"   launching subprocess for mphgnn seed={seed}")
    t0 = time.time()
    try:
        p = subprocess.run(["python3", "-u", stub], cwd=ROOT,
                           capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        log(f"   TIMEOUT after 15min")
        return None
    log(f"   subprocess returned in {time.time()-t0:.0f}s rc={p.returncode}")
    # Always show last lines for visibility
    tail = "\n".join(p.stdout.splitlines()[-30:])
    log(f"   stdout-tail:\n{tail}")
    if p.returncode != 0:
        log(f"   stderr-tail: {p.stderr[-500:]}")
        return None
    target = f"phase2_outputs/mphgnn_5metrics_seed{seed}.json"
    if not os.path.exists(target):
        log(f"   subprocess didn't produce {target}")
        return None
    return json.load(open(target))


def run_method_one_seed(method, seed, log):
    cfg = METHODS[method]
    if cfg.get("subprocess"):
        out = run_mphgnn_subprocess(seed, log)
        if out is None:
            raise RuntimeError("mphgnn subprocess failed (see log)")
        return out
    mod = import_script_as_module(cfg["script"])
    out = {}
    for ds_name, ds_path in (("D1", cfg["d1"]), ("D2", cfg["d2"])):
        t0 = time.time()
        out[ds_name] = cfg["call"](mod, ds_name, ds_path, seed, DEVICE)
        log(f"   {ds_name} done in {time.time()-t0:.0f}s "
            f"(AUC={out[ds_name]['auc']['mean']:.4f})")
    return out


def save_phase2(method, seed, out, log):
    fn = f"phase2_outputs/{method}_5metrics_seed{seed}.json"
    os.makedirs("phase2_outputs", exist_ok=True)
    # mphgnn subprocess already wrote to disk -- avoid double-write
    if not os.path.exists(fn):
        with open(fn, "w") as f:
            json.dump(out, f, indent=2)
    log(f"   saved -> {fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--skip", nargs="+", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--logfile", default="multiseed_run.log")
    args = ap.parse_args()

    methods = [m for m in args.methods if m not in args.skip and m in METHODS]
    if not methods:
        sys.exit("ERROR: no valid methods after filtering.")

    logf = open(args.logfile, "a")
    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"=== START seeds={args.seeds}  methods={methods}  device={DEVICE} ===")
    summary = []; t_all = time.time()

    for seed in args.seeds:
        for method in methods:
            target = f"phase2_outputs/{method}_5metrics_seed{seed}.json"
            if (not args.overwrite) and os.path.exists(target):
                log(f"skip {method} seed={seed} (exists)")
                summary.append({"method": method, "seed": seed, "status": "skip"})
                continue
            log(f"---- {method} seed={seed} ----")
            t0 = time.time()
            try:
                out = run_method_one_seed(method, seed, log)
                save_phase2(method, seed, out, log)
                summary.append({"method": method, "seed": seed, "status": "ok",
                                "d1_auc": out["D1"]["auc"]["mean"],
                                "d2_auc": out["D2"]["auc"]["mean"],
                                "elapsed_s": time.time()-t0})
            except Exception as e:
                log(f"   FAILED: {e}")
                log(traceback.format_exc())
                summary.append({"method": method, "seed": seed, "status": "fail",
                                "err": str(e)})

    log(f"=== ALL DONE in {(time.time()-t_all)/60:.1f} min ===")
    with open("phase2_outputs/_multiseed_summary.json", "w") as f:
        json.dump({"seeds": args.seeds, "methods": methods, "summary": summary}, f, indent=2)
    log("summary -> phase2_outputs/_multiseed_summary.json")


if __name__ == "__main__":
    main()
