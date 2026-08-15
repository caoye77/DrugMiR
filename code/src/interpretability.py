import numpy as np
import pandas as pd

print("=" * 60)
print("#10: Interpretability Analysis")
print("=" * 60)

assoc = np.load("data/processed/association_matrix.npy")
mg = np.load("data/processed/mirna_gene_matrix.npy")
dg = np.load("data/processed/drug_gene_matrix.npy")
drug_sim = np.load("data/processed/drug_similarity.npy")
mirna_sim = np.load("data/processed/mirna_similarity.npy")
mirna_map = pd.read_csv("data/processed/mirna_mapping.csv")
drug_map = pd.read_csv("data/processed/drug_mapping.csv")
gene_map = pd.read_csv("data/processed/gene_mapping.csv")

n_mirna, n_drug = assoc.shape
n_gene = mg.shape[1]

# 1. Top Bridge Genes
print("\n--- 1. Top 20 Bridge Genes (connected to both miRNAs and drugs) ---")
mirna_degree = mg.sum(axis=0)
drug_degree = dg.sum(axis=0)
bridge_score = mirna_degree * drug_degree
top_idx = np.argsort(-bridge_score)[:20]
print(f"  {'Rank':<6s}{'Gene':<14s}{'miRNA_deg':>10s}{'Drug_deg':>10s}{'Bridge':>12s}")
print("  " + "-" * 52)
for i, idx in enumerate(top_idx):
    g = gene_map.iloc[idx]["gene_name"]
    print(f"  {i+1:<6d}{g:<14s}{int(mirna_degree[idx]):>10d}{int(drug_degree[idx]):>10d}{int(bridge_score[idx]):>12d}")

# 2. Most Connected Drugs
print("\n--- 2. Most Connected Drugs ---")
drug_cnt = assoc.sum(axis=0)
for i, idx in enumerate(np.argsort(-drug_cnt)[:10]):
    d = drug_map.iloc[idx]["drug_name"]
    print(f"  {i+1:2d}. {d:20s} {int(drug_cnt[idx]):4d} associations")

# 3. Most Connected miRNAs
print("\n--- 3. Most Connected miRNAs ---")
mirna_cnt = assoc.sum(axis=1)
for i, idx in enumerate(np.argsort(-mirna_cnt)[:10]):
    m = mirna_map.iloc[idx]["mirna_name"]
    print(f"  {i+1:2d}. {m:25s} {int(mirna_cnt[idx]):4d} associations")

# 4. Bridge Path Examples
print("\n--- 4. Gene Bridge Path Analysis ---")
bridge_genes = np.where((mirna_degree > 0) & (drug_degree > 0))[0]
print(f"  Total bridge genes: {len(bridge_genes)}/{n_gene}")
print(f"  Avg miRNA-degree of bridge genes: {mirna_degree[bridge_genes].mean():.1f}")
print(f"  Avg Drug-degree of bridge genes: {drug_degree[bridge_genes].mean():.1f}")

pos_i, pos_j = np.where(assoc > 0)
np.random.seed(42)
print("\n  Example bridge paths:")
for s in np.random.choice(len(pos_i), 8, replace=False):
    mi, di = pos_i[s], pos_j[s]
    m_genes = set(np.where(mg[mi] > 0)[0])
    d_genes = set(np.where(dg[di] > 0)[0])
    shared = m_genes & d_genes
    mn = mirna_map.iloc[mi]["mirna_name"]
    dn = drug_map.iloc[di]["drug_name"]
    print(f"  {mn:25s} <-> {dn:15s}: {len(shared):4d} shared genes (m={len(m_genes)}, d={len(d_genes)})")

# 5. Similarity Network Stats
print("\n--- 5. Similarity Network Statistics ---")
n_d_edges = (np.sum(drug_sim > 0.3) - n_drug) // 2
n_m_edges = (np.sum(mirna_sim > 0.5) - n_mirna) // 2
print(f"  Drug-Drug sim edges (>0.3): {n_d_edges}, avg degree: {n_d_edges*2/n_drug:.1f}")
print(f"  miRNA-miRNA sim edges (>0.5): {n_m_edges}, avg degree: {n_m_edges*2/n_mirna:.1f}")
ds = drug_sim[np.triu_indices(n_drug, k=1)]
ms = mirna_sim[np.triu_indices(n_mirna, k=1)]
print(f"  Drug sim distribution: mean={ds.mean():.4f}, median={np.median(ds):.4f}, max={ds.max():.4f}")
print(f"  miRNA sim distribution: mean={ms.mean():.4f}, median={np.median(ms):.4f}, max={ms.max():.4f}")

# 6. Association density per drug
print("\n--- 6. Association Density Analysis ---")
drug_density = drug_cnt / n_mirna * 100
print(f"  Per-drug association density: mean={drug_density.mean():.2f}%, max={drug_density.max():.2f}%, min={drug_density.min():.2f}%")
print(f"  Drugs with >100 associations: {(drug_cnt > 100).sum()}/{n_drug}")
print(f"  Drugs with <10 associations: {(drug_cnt < 10).sum()}/{n_drug}")

print("\nDone!")
