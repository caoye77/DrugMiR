"""
Build gene interaction data for DMGAT Dataset 2
Match DMGAT ncRNAs/drugs to MGCNA entities, extract gene edges
"""
import pandas as pd
import numpy as np
import os

# Paths
DMGAT_DIR = "DMGAT"
MGCNA_DIR = "data/processed"
OUT_DIR = "DMGAT_processed"

# Load DMGAT entities
adj = pd.read_csv(f"{DMGAT_DIR}/adj_with_sens.csv", index_col=0)
rna = pd.read_csv(f"{DMGAT_DIR}/rna_seq.csv")
drug = pd.read_csv(f"{DMGAT_DIR}/drug_smiles.csv")

dmgat_rna_names = list(adj.index)   # 622 ncRNAs in order
dmgat_drug_names = list(adj.columns) # 121 drugs in order

print(f"DMGAT: {len(dmgat_rna_names)} ncRNAs x {len(dmgat_drug_names)} drugs")

# Load MGCNA mappings
mgcna_mirna = pd.read_csv(f"{MGCNA_DIR}/mirna_mapping.csv")
mgcna_drug = pd.read_csv(f"{MGCNA_DIR}/drug_mapping.csv")
mgcna_mg = np.load(f"{MGCNA_DIR}/mirna_gene_matrix.npy")  # 1578 x 14455
mgcna_dg = np.load(f"{MGCNA_DIR}/drug_gene_matrix.npy")   # 156 x 14455

print(f"MGCNA: {len(mgcna_mirna)} miRNAs, {len(mgcna_drug)} drugs, {mgcna_mg.shape[1]} genes")
print(f"MGCNA mirna-gene: {mgcna_mg.shape}, drug-gene: {mgcna_dg.shape}")

# Build MGCNA lookup dicts
# miRNA: try multiple matching strategies
mgcna_mirna_dict = {}
for idx, row in mgcna_mirna.iterrows():
    name = row['mirna_name'].lower().strip()
    mgcna_mirna_dict[name] = row['idx']
    # Also index by short name (remove hsa- prefix and -5p/-3p suffix)
    short = name.replace('hsa-', '')
    mgcna_mirna_dict[short] = row['idx']
    # Remove -5p/-3p
    if short.endswith('-5p') or short.endswith('-3p'):
        base = short[:-3]
        if base not in mgcna_mirna_dict:
            mgcna_mirna_dict[base] = row['idx']

# Drug: exact match on lowercase
mgcna_drug_dict = {}
for idx, row in mgcna_drug.iterrows():
    mgcna_drug_dict[row['drug_name'].lower().strip()] = row['idx']

# Match DMGAT ncRNAs to MGCNA miRNAs
rna_type_dict = dict(zip(rna['name'], rna['type']))
mirna_gene_rows = []
matched_mirna = 0
unmatched_mirna = 0

for i, name in enumerate(dmgat_rna_names):
    rtype = rna_type_dict.get(name, 'unknown')
    n = name.lower().strip()
    
    mgcna_idx = None
    
    if rtype == 'miRNA':
        # Try direct match
        if n in mgcna_mirna_dict:
            mgcna_idx = mgcna_mirna_dict[n]
        # Try with hsa- prefix
        elif ('hsa-' + n) in mgcna_mirna_dict:
            mgcna_idx = mgcna_mirna_dict['hsa-' + n]
        # Try hsa-mir -> hsa-miR
        elif n.startswith('mir-'):
            candidate = n  # already lowercase
            if candidate in mgcna_mirna_dict:
                mgcna_idx = mgcna_mirna_dict[candidate]
            elif ('hsa-' + candidate) in mgcna_mirna_dict:
                mgcna_idx = mgcna_mirna_dict['hsa-' + candidate]
        # Try with -5p suffix (default mature form)
        if mgcna_idx is None:
            for suffix in ['-5p', '-3p', '']:
                candidate = n + suffix
                if candidate in mgcna_mirna_dict:
                    mgcna_idx = mgcna_mirna_dict[candidate]
                    break
                candidate = 'hsa-' + n + suffix
                if candidate in mgcna_mirna_dict:
                    mgcna_idx = mgcna_mirna_dict[candidate]
                    break
    
    if mgcna_idx is not None:
        mirna_gene_rows.append(mgcna_mg[mgcna_idx])
        matched_mirna += 1
    else:
        mirna_gene_rows.append(np.zeros(mgcna_mg.shape[1]))
        unmatched_mirna += 1

mirna_gene_matrix = np.array(mirna_gene_rows)
print(f"\nmiRNA matching: {matched_mirna} matched, {unmatched_mirna} unmatched")
print(f"mirna_gene_matrix: {mirna_gene_matrix.shape}, edges: {int(mirna_gene_matrix.sum())}")

# Match DMGAT drugs to MGCNA drugs
drug_gene_rows = []
matched_drug = 0
unmatched_drug = 0

for i, name in enumerate(dmgat_drug_names):
    n = name.lower().strip()
    
    mgcna_idx = mgcna_drug_dict.get(n, None)
    
    if mgcna_idx is not None:
        drug_gene_rows.append(mgcna_dg[mgcna_idx])
        matched_drug += 1
    else:
        drug_gene_rows.append(np.zeros(mgcna_dg.shape[1]))
        unmatched_drug += 1

drug_gene_matrix = np.array(drug_gene_rows)
print(f"Drug matching: {matched_drug} matched, {unmatched_drug} unmatched")
print(f"drug_gene_matrix: {drug_gene_matrix.shape}, edges: {int(drug_gene_matrix.sum())}")

# Save
np.save(f"{OUT_DIR}/mirna_gene_matrix.npy", mirna_gene_matrix)
np.save(f"{OUT_DIR}/drug_gene_matrix.npy", drug_gene_matrix)

# Also build edge lists
mg_r, mg_c = np.nonzero(mirna_gene_matrix)
dg_r, dg_c = np.nonzero(drug_gene_matrix)
np.savez(f"{OUT_DIR}/edge_lists.npz",
         mirna_gene_src=mg_r, mirna_gene_dst=mg_c,
         drug_gene_src=dg_r, drug_gene_dst=dg_c)

print(f"\nSaved to {OUT_DIR}/")
print(f"mirna_gene edges: {len(mg_r)}")
print(f"drug_gene edges: {len(dg_r)}")
print(f"n_genes used: {max(mg_c.max() if len(mg_c)>0 else 0, dg_c.max() if len(dg_c)>0 else 0) + 1}")
print("DONE")
