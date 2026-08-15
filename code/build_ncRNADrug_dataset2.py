"""
Build Dataset 2: Resistance-only from ncRNADrug
Task: Predict miRNA-drug resistance associations
Different from Dataset 1 which predicts general association (resistance + sensitivity)
"""
import csv, os, sys
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product

# ========== Configuration ==========
INPUT_FILE = "ncRNADrug/DR_Curated.txt"  # Put the downloaded file here
MGCNA_DIR = "data/processed"
OUT_DIR = "ncRNADrug_processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Min associations per entity
MIN_MIRNA_ASSOC = 2
MIN_DRUG_ASSOC = 2

# ========== 1. Load and filter ncRNADrug ==========
rows = []
with open(INPUT_FILE, 'r', encoding='latin-1') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        rows.append(row)

print(f"Total entries: {len(rows)}")

# Filter: Human, miRNA, resistance only
human_mirna_res = [r for r in rows 
                   if r.get('Species','').strip() == 'Homo sapiens' 
                   and r.get('ncRNA_Type','').strip() == 'miRNA'
                   and r.get('Effect','').strip() == 'resistant']

print(f"Human miRNA resistance entries: {len(human_mirna_res)}")

# Get unique resistance pairs
res_pairs = set()
for r in human_mirna_res:
    mi = r['ncRNA_Name'].strip()
    dr = r['Drug_Name'].strip()
    res_pairs.add((mi, dr))

print(f"Unique resistance pairs: {len(res_pairs)}")

# Filter by minimum associations
mi_count = Counter(p[0] for p in res_pairs)
dr_count = Counter(p[1] for p in res_pairs)
filtered = set(p for p in res_pairs if mi_count[p[0]] >= MIN_MIRNA_ASSOC and dr_count[p[1]] >= MIN_DRUG_ASSOC)

# Get final entity lists
mirna_set = sorted(set(p[0] for p in filtered))
drug_set = sorted(set(p[1] for p in filtered))

# Re-filter (iterative until stable)
for _ in range(5):
    mi_count2 = Counter(p[0] for p in filtered if p[0] in mirna_set and p[1] in drug_set)
    dr_count2 = Counter(p[1] for p in filtered if p[0] in mirna_set and p[1] in drug_set)
    mirna_set = sorted(m for m in mirna_set if mi_count2.get(m,0) >= MIN_MIRNA_ASSOC)
    drug_set = sorted(d for d in drug_set if dr_count2.get(d,0) >= MIN_DRUG_ASSOC)
    filtered = set(p for p in filtered if p[0] in mirna_set and p[1] in drug_set)

print(f"\nDataset 2 (resistance-only):")
print(f"  miRNAs: {len(mirna_set)}")
print(f"  Drugs: {len(drug_set)}")
print(f"  Resistance pairs: {len(filtered)}")
print(f"  Density: {len(filtered)/(len(mirna_set)*len(drug_set))*100:.2f}%")

# ========== 2. Build association matrix ==========
mi_idx = {m:i for i,m in enumerate(mirna_set)}
dr_idx = {d:i for i,d in enumerate(drug_set)}

assoc = np.zeros((len(mirna_set), len(drug_set)))
for mi, dr in filtered:
    assoc[mi_idx[mi], dr_idx[dr]] = 1

print(f"Association matrix: {assoc.shape}, positives: {int(assoc.sum())}")
np.save(f"{OUT_DIR}/association_matrix.npy", assoc)

# Save mappings
with open(f"{OUT_DIR}/mirna_list.txt", 'w') as f:
    for m in mirna_set: f.write(m + '\n')
with open(f"{OUT_DIR}/drug_list.txt", 'w') as f:
    for d in drug_set: f.write(d + '\n')

# ========== 3. miRNA features (k-mer from MGCNA sequences) ==========
mgcna_mirna = pd.read_csv(f'{MGCNA_DIR}/mirna_mapping.csv')
mgcna_seq = dict(zip(mgcna_mirna['mirna_name'].str.lower(), mgcna_mirna['sequence']))

k = 4
kmers = [''.join(p) for p in product('ACGU', repeat=k)]
kmer_idx_map = {km: i for i, km in enumerate(kmers)}

feats = []
seq_matched = 0
for name in mirna_set:
    n = name.lower().strip()
    seq = mgcna_seq.get(n, None)
    
    vec = np.zeros(len(kmers))
    if seq and isinstance(seq, str) and len(seq) >= k:
        seq = seq.upper().replace('T', 'U')
        for i2 in range(len(seq) - k + 1):
            km = seq[i2:i2+k]
            if km in kmer_idx_map:
                vec[kmer_idx_map[km]] += 1
        s = vec.sum()
        if s > 0:
            vec /= s
        seq_matched += 1
    feats.append(vec)

mirna_feat = np.array(feats, dtype=np.float32)
print(f"\nmiRNA features: {mirna_feat.shape}, with sequence: {seq_matched}/{len(mirna_set)}")
np.save(f"{OUT_DIR}/mirna_kmer_features.npy", mirna_feat)

# ========== 4. Drug features (Morgan FP) ==========
# Get SMILES from MGCNA drug mapping
mgcna_drug = pd.read_csv(f'{MGCNA_DIR}/drug_mapping.csv')
mgcna_smiles = dict(zip(mgcna_drug['drug_name'].str.lower(), mgcna_drug['smiles']))

# Also from DMGAT as backup
try:
    dmgat_drug = pd.read_csv('DMGAT/drug_smiles.csv')
    dmgat_smiles = dict(zip(dmgat_drug['name'].str.lower(), dmgat_drug['smiles']))
except:
    dmgat_smiles = {}

# Build DrugBank ID -> name mapping from ncRNADrug
drugbank_map = {}
for r in human_mirna_res:
    dn = r['Drug_Name'].strip()
    dbid = r.get('DrugBank_ID','').strip()
    if dbid and dbid != 'NA':
        drugbank_map[dn.lower()] = dbid

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

drug_feats = []
drug_matched = 0
for name in drug_set:
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
print(f"Drug features: {drug_feat.shape}, with SMILES: {drug_matched}/{len(drug_set)}")
np.save(f"{OUT_DIR}/drug_morgan_features.npy", drug_feat)

# ========== 5. Similarity matrices ==========
from sklearn.metrics.pairwise import cosine_similarity

mirna_sim = cosine_similarity(mirna_feat)
np.save(f"{OUT_DIR}/mirna_similarity.npy", mirna_sim)

n_drug = drug_feat.shape[0]
drug_sim = np.zeros((n_drug, n_drug))
for i in range(n_drug):
    for j in range(n_drug):
        inter = np.minimum(drug_feat[i], drug_feat[j]).sum()
        union = np.maximum(drug_feat[i], drug_feat[j]).sum()
        drug_sim[i,j] = inter/union if union > 0 else 0
np.save(f"{OUT_DIR}/drug_similarity.npy", drug_sim)

# ========== 6. Gene interaction data ==========
mgcna_mg = np.load(f'{MGCNA_DIR}/mirna_gene_matrix.npy')
mgcna_dg = np.load(f'{MGCNA_DIR}/drug_gene_matrix.npy')

mgcna_mirna_idx = dict(zip(mgcna_mirna['mirna_name'].str.lower(), mgcna_mirna['idx']))
mgcna_drug_idx = dict(zip(mgcna_drug['drug_name'].str.lower(), mgcna_drug['idx']))

# Match miRNA gene data
mirna_gene_rows = []
mg_matched = 0
for name in mirna_set:
    idx = mgcna_mirna_idx.get(name.lower().strip(), None)
    if idx is not None:
        mirna_gene_rows.append(mgcna_mg[idx])
        mg_matched += 1
    else:
        mirna_gene_rows.append(np.zeros(mgcna_mg.shape[1]))

mirna_gene_matrix = np.array(mirna_gene_rows)

# Match drug gene data
drug_gene_rows = []
dg_matched = 0
for name in drug_set:
    idx = mgcna_drug_idx.get(name.lower().strip(), None)
    if idx is not None:
        drug_gene_rows.append(mgcna_dg[idx])
        dg_matched += 1
    else:
        drug_gene_rows.append(np.zeros(mgcna_dg.shape[1]))

drug_gene_matrix = np.array(drug_gene_rows)

np.save(f"{OUT_DIR}/mirna_gene_matrix.npy", mirna_gene_matrix)
np.save(f"{OUT_DIR}/drug_gene_matrix.npy", drug_gene_matrix)

mg_r, mg_c = np.nonzero(mirna_gene_matrix)
dg_r, dg_c = np.nonzero(drug_gene_matrix)
np.savez(f"{OUT_DIR}/edge_lists.npz",
         mirna_gene_src=mg_r, mirna_gene_dst=mg_c,
         drug_gene_src=dg_r, drug_gene_dst=dg_c)

print(f"\nGene matching:")
print(f"  miRNA-gene: {mg_matched}/{len(mirna_set)} matched, {int(mirna_gene_matrix.sum())} edges")
print(f"  Drug-gene: {dg_matched}/{len(drug_set)} matched, {int(drug_gene_matrix.sum())} edges")

print(f"\n{'='*50}")
print(f"DATASET 2 SUMMARY (Resistance-only)")
print(f"{'='*50}")
print(f"  miRNAs: {len(mirna_set)}")
print(f"  Drugs: {len(drug_set)}")
print(f"  Resistance pairs: {int(assoc.sum())}")
print(f"  miRNA with seq: {seq_matched}/{len(mirna_set)} ({100*seq_matched/len(mirna_set):.0f}%)")
print(f"  Drug with SMILES: {drug_matched}/{len(drug_set)} ({100*drug_matched/len(drug_set):.0f}%)")
print(f"  miRNA-gene matched: {mg_matched}/{len(mirna_set)} ({100*mg_matched/len(mirna_set):.0f}%)")
print(f"  Drug-gene matched: {dg_matched}/{len(drug_set)} ({100*dg_matched/len(drug_set):.0f}%)")
print(f"  Total files saved to: {OUT_DIR}/")
print("DONE")
