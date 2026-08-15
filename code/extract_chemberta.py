import numpy as np, pandas as pd, torch, os
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Load
mgcna_dr = pd.read_csv('data/processed/drug_mapping.csv')
df = np.load('DMGAT_processed/drug_morgan_features.npy')
n_drug = df.shape[0]
print(f'Drugs: {n_drug}, with SMILES: {(df.sum(1)>0).sum()}')

# Match each DMGAT drug to MGCNA SMILES by Morgan FP
print('Matching drugs to SMILES...')
smiles_list = []
mgcna_fps = []
for _, row in mgcna_dr.iterrows():
    smi = row['smiles']
    if pd.isna(smi) or smi == 'NotFound':
        mgcna_fps.append((None, None))
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mgcna_fps.append((None, None))
        continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros(1024)
    DataStructs.ConvertToNumpyArray(fp, arr)
    mgcna_fps.append((smi, arr))

for i in range(n_drug):
    fp_row = df[i]
    if fp_row.sum() == 0:
        smiles_list.append(None)
        continue
    found = False
    for smi, arr in mgcna_fps:
        if arr is not None and np.array_equal(arr, fp_row):
            smiles_list.append(smi)
            found = True
            break
    if not found:
        smiles_list.append(None)

print(f'SMILES matched: {sum(1 for s in smiles_list if s)}/{n_drug}')

# ChemBERTa
print('Loading ChemBERTa...')
tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
model = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
model.eval()
h = model.config.hidden_size
print(f'Hidden size: {h}')

print('Extracting embeddings...')
embeddings = []
with torch.no_grad():
    for i, smi in enumerate(smiles_list):
        if smi:
            tok = tokenizer(smi, return_tensors='pt', truncation=True, max_length=512)
            out = model(**tok)
            emb = out.last_hidden_state[:, 0, :].squeeze(0).numpy()
        else:
            emb = np.zeros(h)
        embeddings.append(emb)
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{n_drug}')

drug_chemberta = np.array(embeddings, dtype=np.float32)
np.save('DMGAT_processed/drug_chemberta_features.npy', drug_chemberta)
print(f'Saved: {drug_chemberta.shape}')
print('DONE')
