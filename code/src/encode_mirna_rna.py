"""
用预训练 RNA 模型编码 miRNA 序列
=================================
miRNA 序列很短 (18-25nt)，CPU 就能跑

运行: cd D:\DrugMiR && python src/encode_mirna_rna.py
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print("=" * 60)
print("RNA Pretrained Model - miRNA Encoding")
print("=" * 60)

# Load miRNA data
mirna_map = pd.read_csv("data/processed/mirna_mapping.csv")
print(f"miRNAs: {len(mirna_map)}")
print(f"Sample sequence: {mirna_map.iloc[0]['sequence']}")

# Try multimolecule RnaFm or RnaBert
print("\nLoading pretrained RNA model...")

try:
    from multimolecule import RnaTokenizer, RnaBertModel
    model_name = "multimolecule/rnabert"
    tokenizer = RnaTokenizer.from_pretrained(model_name)
    model = RnaBertModel.from_pretrained(model_name)
    print(f"Model: {model_name}")
except Exception as e1:
    print(f"RnaBert failed: {e1}")
    try:
        from multimolecule import RnaTokenizer, RnaFmModel
        model_name = "multimolecule/rnafm"
        tokenizer = RnaTokenizer.from_pretrained(model_name)
        model = RnaFmModel.from_pretrained(model_name)
        print(f"Model: {model_name}")
    except Exception as e2:
        print(f"RnaFm also failed: {e2}")
        print("Falling back to basic transformer...")
        from transformers import AutoTokenizer, AutoModel
        model_name = "multimolecule/rnabert"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"Model: {model_name} (via AutoModel)")

model.eval()
hidden_size = model.config.hidden_size
print(f"Hidden dim: {hidden_size}")

# Encode all miRNAs
features = []
failed = []

for idx, row in mirna_map.iterrows():
    seq = str(row['sequence']).strip().upper().replace('T', 'U')
    name = row['mirna_name']

    try:
        # Add spaces between nucleotides for tokenizer
        spaced_seq = ' '.join(list(seq))
        inputs = tokenizer(spaced_seq, return_tensors="pt", padding=True,
                          truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over sequence length
            if hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            else:
                embedding = outputs[0].mean(dim=1).squeeze().numpy()
        features.append(embedding)
    except Exception as e:
        if len(failed) < 5:
            print(f"  WARNING: {name} failed: {e}")
        features.append(np.zeros(hidden_size))
        failed.append(name)

    if (idx + 1) % 200 == 0:
        print(f"  Encoded {idx + 1}/{len(mirna_map)} miRNAs...")

features = np.array(features, dtype=np.float32)
print(f"\nEncoding complete: {features.shape}")
print(f"Failed: {len(failed)}/{len(mirna_map)}")

# Compare with k-mer
kmer = np.load("data/processed/mirna_kmer_features.npy")
print(f"\nFeature comparison:")
print(f"  k-mer:     {kmer.shape} (sparse, {(kmer > 0).mean()*100:.1f}% non-zero)")
print(f"  RNA-Pretrained: {features.shape} (dense, mean={features.mean():.4f}, std={features.std():.4f})")

# Save
np.save("data/processed/mirna_rna_pretrained_features.npy", features)
print(f"\nSaved to data/processed/mirna_rna_pretrained_features.npy")
