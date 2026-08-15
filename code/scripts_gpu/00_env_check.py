"""
DrugMiR GPU Environment Check
Run: python 00_env_check.py
Confirms PyTorch + PyG + dependencies + data files are ready.
Exits non-zero if anything missing.
"""
import sys, os, importlib

print("="*60)
print("DrugMiR GPU Environment Check")
print("="*60)

errors = []
warnings = []

# ---- 1. Python version ----
print(f"\n[1] Python: {sys.version.split()[0]}")
if sys.version_info < (3, 9):
    errors.append("Python >= 3.9 required")

# ---- 2. Core packages ----
required = {
    "torch":       "2.0",
    "torch_geometric": "2.4",
    "torch_scatter":   None,
    "numpy":       "1.21",
    "pandas":      "1.5",
    "sklearn":     "1.0",
    "scipy":       "1.7",
    "matplotlib":  "3.5",
}

print("\n[2] Required packages:")
for pkg, min_ver in required.items():
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "?")
        print(f"   ✓ {pkg:20s} {ver}")
    except ImportError as e:
        print(f"   ✗ {pkg:20s} MISSING — {e}")
        errors.append(f"Missing package: {pkg}")

# ---- 3. CUDA / GPU ----
print("\n[3] CUDA / GPU:")
try:
    import torch
    print(f"   torch.cuda.is_available()  = {torch.cuda.is_available()}")
    print(f"   torch.cuda.device_count()  = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB, "
                  f"compute capability {p.major}.{p.minor}")
        # Sanity check: tensor on GPU
        x = torch.randn(100, 100, device="cuda")
        y = (x @ x.t()).sum().item()
        print(f"   ✓ Forward sanity: tensor sum = {y:.2f}")
    else:
        errors.append("No CUDA device")
except Exception as e:
    errors.append(f"CUDA check failed: {e}")

# ---- 4. PyG functionality (scatter ops are critical for Gene Bridge) ----
print("\n[4] PyG functionality:")
try:
    from torch_geometric.utils import scatter
    import torch
    src = torch.randn(10, 4)
    idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    out = scatter(src, idx, dim=0, reduce="mean")
    assert out.shape == (5, 4), f"unexpected shape {out.shape}"
    print(f"   ✓ scatter_mean OK, output shape={tuple(out.shape)}")
except Exception as e:
    print(f"   ✗ scatter failed: {e}")
    errors.append(f"PyG scatter not working: {e}")

# ---- 5. Data files ----
print("\n[5] Data files:")
DATA_DIR = os.environ.get("DRUGMIR_DATA", "./data/processed")
required_files = [
    "association_matrix.npy",
    "resistance_matrix.npy",
    "sensitivity_matrix.npy",
    "mirna_kmer_features.npy",
    "drug_morgan_features.npy",
    "mirna_gene_matrix.npy",
    "drug_gene_matrix.npy",
    "mirna_similarity.npy",
    "drug_similarity.npy",
]
for f in required_files:
    full = os.path.join(DATA_DIR, f)
    if os.path.exists(full):
        sz = os.path.getsize(full) / 1024
        print(f"   ✓ {f:35s} ({sz:.1f} KB)")
    else:
        print(f"   ✗ {f:35s} MISSING at {full}")
        errors.append(f"Missing data file: {full}")

# ---- 6. Output directory ----
out_dir = "./results_multitask"
os.makedirs(out_dir, exist_ok=True)
test = os.path.join(out_dir, ".write_test")
try:
    open(test, "w").write("ok"); os.remove(test)
    print(f"\n[6] Output dir writable: {out_dir} ✓")
except Exception as e:
    print(f"\n[6] ✗ Cannot write to {out_dir}: {e}")
    errors.append(f"Output dir not writable: {e}")

# ---- Summary ----
print("\n" + "="*60)
if errors:
    print("FAIL — please address before continuing:")
    for e in errors: print(f"  ✗ {e}")
    print("\nQuick install hint (if needed):")
    print("  pip install torch_geometric torch_scatter -f \\")
    print("    https://data.pyg.org/whl/torch-2.2.0+cu121.html")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — ready to run experiments.")
    sys.exit(0)
