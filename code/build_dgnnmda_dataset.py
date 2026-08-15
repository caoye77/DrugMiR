"""
Preprocess DGNNMDA dataset (432 miRNA x 141 drug, 2049 sensitivity associations)
Build: association matrix, features, similarity, gene interaction data
"""
import pandas as pd
import numpy as np
import os, re

OUT_DIR = "DGNNMDA_processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== 1. Load DGNNMDA data ==========
# miRNA list
mi_lines = open('DGNNMDA/dataset/miRNA.txt').readlines()[1:]  # skip header
mirna_list = []
for l in mi_lines:
    parts = l.strip().split('\t')
    if len(parts) >= 2:
        mirna_list.append(parts[1])
print(f"miRNAs: {len(mirna_list)}")

# Drug list - extract unique drugs from associations
assoc_lines = open('DGNNMDA/dataset/association.txt').readlines()
drug_set = []
assoc_pairs = []
for l in assoc_lines:
    parts = l.strip().split('\t')
    if len(parts) >= 2:
        drug = parts[0].strip()
        mirna = parts[1].strip()
        if drug not in drug_set:
            drug_set.append(drug)
        assoc_pairs.append((drug, mirna))
drug_list = drug_set
print(f"Drugs: {len(drug_list)}")
print(f"Associations: {len(assoc_pairs)}")

# Build association matrix
mirna_idx = {n: i for i, n in enumerate(mirna_list)}
drug_idx = {n: i for i, n in enumerate(drug_list)}
assoc = np.zeros((len(mirna_list), len(drug_list)))
matched_assoc = 0
for drug, mirna in assoc_pairs:
    if mirna in mirna_idx and drug in drug_idx:
        assoc[mirna_idx[mirna], drug_idx[drug]] = 1
        matched_assoc += 1
print(f"Association matrix: {assoc.shape}, positives: {int(assoc.sum())}")
np.save(f"{OUT_DIR}/association_matrix.npy", assoc)

# ========== 2. miRNA features (k-mer from MGCNA sequences) ==========
# Load MGCNA miRNA data for sequence lookup
mgcna_mirna = pd.read_csv('data/processed/mirna_mapping.csv')
mgcna_seq = dict(zip(mgcna_mirna['mirna_name'].str.lower(), mgcna_mirna['sequence']))

# Match DGNNMDA miRNA names to MGCNA
from itertools import product
k = 4
kmers = [''.join(p) for p in product('ACGU', repeat=k)]
kmer_idx = {km: i for i, km in enumerate(kmers)}

feats = []
seq_matched = 0
for name in mirna_list:
    n = name.lower().strip().replace('*', '')
    seq = None
    
    # Try multiple matching strategies
    candidates = [
        n,
        'hsa-' + n,
        n + '-5p',
        n + '-3p', 
        'hsa-' + n + '-5p',
        'hsa-' + n + '-3p',
    ]
    # Also try replacing miR with mir and vice versa
    for c in list(candidates):
        candidates.append(c.replace('mir-', 'miR-'))
        candidates.append(c.replace('miR-', 'mir-'))
    
    for c in candidates:
        if c.lower() in mgcna_seq:
            seq = mgcna_seq[c.lower()]
            break
    
    # Fuzzy match: search for substring
    if seq is None:
        for mgcna_name, mgcna_s in mgcna_seq.items():
            if n.replace('mir-','').replace('miR-','') in mgcna_name.replace('hsa-','').replace('mir-','').replace('miR-',''):
                seq = mgcna_s
                break
    
    vec = np.zeros(len(kmers))
    if seq and isinstance(seq, str) and len(seq) >= k:
        seq = seq.upper().replace('T', 'U')
        for i2 in range(len(seq) - k + 1):
            km = seq[i2:i2+k]
            if km in kmer_idx:
                vec[kmer_idx[km]] += 1
        s = vec.sum()
        if s > 0:
            vec /= s
        seq_matched += 1
    feats.append(vec)

mirna_feat = np.array(feats, dtype=np.float32)
print(f"miRNA features: {mirna_feat.shape}, with sequence: {seq_matched}/{len(mirna_list)}")
np.save(f"{OUT_DIR}/mirna_kmer_features.npy", mirna_feat)

# ========== 3. Drug features (Morgan FP) ==========
# Load MGCNA drug data for SMILES lookup
mgcna_drug = pd.read_csv('data/processed/drug_mapping.csv')
mgcna_smiles = dict(zip(mgcna_drug['drug_name'].str.lower(), mgcna_drug['smiles']))

# Also try DMGAT drug SMILES as backup
try:
    dmgat_drug = pd.read_csv('DMGAT/drug_smiles.csv')
    dmgat_smiles = dict(zip(dmgat_drug['name'].str.lower(), dmgat_drug['smiles']))
except:
    dmgat_smiles = {}

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

drug_feats = []
drug_matched = 0
for name in drug_list:
    n = name.lower().strip()
    smi = mgcna_smiles.get(n, dmgat_smiles.get(n, None))
    
    mol = None
    if smi and smi != 'NotFound':
        mol = Chem.MolFromSmiles(smi)
    
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros(1024)
        DataStructs.ConvertToNumpyArray(fp, arr)
        drug_matched += 1
    else:
        arr = np.zeros(1024)
    drug_feats.append(arr)

drug_feat = np.array(drug_feats, dtype=np.float32)
print(f"Drug features: {drug_feat.shape}, with SMILES: {drug_matched}/{len(drug_list)}")
np.save(f"{OUT_DIR}/drug_morgan_features.npy", drug_feat)

# ========== 4. Similarity matrices ==========
from sklearn.metrics.pairwise import cosine_similarity

mirna_sim = cosine_similarity(mirna_feat)
np.save(f"{OUT_DIR}/mirna_similarity.npy", mirna_sim)

n_drug = drug_feat.shape[0]
drug_sim = np.zeros((n_drug, n_drug))
for i in range(n_drug):
    for j in range(n_drug):
        inter = np.minimum(drug_feat[i], drug_feat[j]).sum()
        union = np.maximum(drug_feat[i], drug_feat[j]).sum()
        drug_sim[i, j] = inter / union if union > 0 else 0
np.save(f"{OUT_DIR}/drug_similarity.npy", drug_sim)
print(f"Similarities: mirna {mirna_sim.shape}, drug {drug_sim.shape}")

# ========== 5. Gene interaction data ==========
mgcna_mg = np.load('data/processed/mirna_gene_matrix.npy')  # 1578 x 14455
mgcna_dg = np.load('data/processed/drug_gene_matrix.npy')   # 156 x 14455

# Build lookup
mgcna_mirna_idx = {}
for _, row in mgcna_mirna.iterrows():
    name = row['mirna_name'].lower().strip()
    mgcna_mirna_idx[name] = row['idx']
    short = name.replace('hsa-', '')
    mgcna_mirna_idx[short] = row['idx']
    if short.endswith('-5p') or short.endswith('-3p'):
        base = short[:-3]
        if base not in mgcna_mirna_idx:
            mgcna_mirna_idx[base] = row['idx']

mgcna_drug_idx = {}
for _, row in mgcna_drug.iterrows():
    mgcna_drug_idx[row['drug_name'].lower().strip()] = row['idx']

# Match miRNA gene data
mirna_gene_rows = []
mg_matched = 0
for name in mirna_list:
    n = name.lower().strip().replace('*', '')
    idx = None
    candidates = [n, 'hsa-'+n, n+'-5p', n+'-3p', 'hsa-'+n+'-5p', 'hsa-'+n+'-3p']
    for c in list(candidates):
        candidates.append(c.replace('mir-','miR-'))
        candidates.append(c.replace('miR-','mir-'))
    for c in candidates:
        if c.lower() in mgcna_mirna_idx:
            idx = mgcna_mirna_idx[c.lower()]
            break
    if idx is None:
        for mgcna_name, mgcna_i in mgcna_mirna_idx.items():
            if n.replace('mir-','').replace('miR-','') in mgcna_name.replace('hsa-','').replace('mir-','').replace('miR-',''):
                idx = mgcna_i
                break
    if idx is not None:
        mirna_gene_rows.append(mgcna_mg[idx])
        mg_matched += 1
    else:
        mirna_gene_rows.append(np.zeros(mgcna_mg.shape[1]))

mirna_gene_matrix = np.array(mirna_gene_rows)
print(f"miRNA-gene matched: {mg_matched}/{len(mirna_list)}")

# Match drug gene data
drug_gene_rows = []
dg_matched = 0
for name in drug_list:
    n = name.lower().strip()
    idx = mgcna_drug_idx.get(n, None)
    if idx is not None:
        drug_gene_rows.append(mgcna_dg[idx])
        dg_matched += 1
    else:
        drug_gene_rows.append(np.zeros(mgcna_dg.shape[1]))

drug_gene_matrix = np.array(drug_gene_rows)
print(f"Drug-gene matched: {dg_matched}/{len(drug_list)}")

np.save(f"{OUT_DIR}/mirna_gene_matrix.npy", mirna_gene_matrix)
np.save(f"{OUT_DIR}/drug_gene_matrix.npy", drug_gene_matrix)

mg_r, mg_c = np.nonzero(mirna_gene_matrix)
dg_r, dg_c = np.nonzero(drug_gene_matrix)
np.savez(f"{OUT_DIR}/edge_lists.npz",
         mirna_gene_src=mg_r, mirna_gene_dst=mg_c,
         drug_gene_src=dg_r, drug_gene_dst=dg_c)

print(f"\nFinal stats:")
print(f"  Association: {assoc.shape}, {int(assoc.sum())} positives")
print(f"  miRNA feat: {mirna_feat.shape}, {seq_matched} with seq")
print(f"  Drug feat: {drug_feat.shape}, {drug_matched} with SMILES")
print(f"  miRNA-gene: {mg_matched}/{len(mirna_list)} matched, {int(mirna_gene_matrix.sum())} edges")
print(f"  Drug-gene: {dg_matched}/{len(drug_list)} matched, {int(drug_gene_matrix.sum())} edges")
print("DONE")
