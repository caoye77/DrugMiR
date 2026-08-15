"""
Step 5: 解析 miRBase 成熟 miRNA 序列
================================
输入: data/raw/miRBase/mature.fa
输出: data/processed/mirna_sequences.csv

输出格式:
  mirna_name | mirna_id | sequence | seq_length
  hsa-mir-21-5p | MIMAT0000076 | UAGCUUAUCAGACUGAUGUUGA | 22
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, get_project_root, normalize_mirna_name,
    print_stats, save_processed
)


def parse_fasta(fasta_path):
    """
    解析 FASTA 文件，提取序列
    miRBase mature.fa 格式:
    >hsa-let-7a-5p MIMAT0000062 Homo sapiens let-7a-5p
    UGAGGUAGUAGGUUGUAUAGUU
    """
    sequences = []
    
    current_name = None
    current_id = None
    current_species = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存上一个序列
                if current_name is not None:
                    sequences.append({
                        'mirna_name': current_name,
                        'mirna_id': current_id,
                        'species': current_species,
                        'sequence': ''.join(current_seq)
                    })
                
                # 解析 header
                parts = line[1:].split()
                current_name = parts[0] if len(parts) > 0 else None
                current_id = parts[1] if len(parts) > 1 else None
                # species: 从第三个字段开始拼接（如 "Homo sapiens"）
                current_species = ' '.join(parts[2:]) if len(parts) > 2 else None
                current_seq = []
            else:
                current_seq.append(line)
    
    # 最后一个序列
    if current_name is not None:
        sequences.append({
            'mirna_name': current_name,
            'mirna_id': current_id,
            'species': current_species,
            'sequence': ''.join(current_seq)
        })
    
    return pd.DataFrame(sequences)


def process_miRBase():
    config = load_config()
    root = get_project_root()
    raw_dir = root / config["paths"]["raw_data"] / "miRBase"
    
    fasta_file = raw_dir / "mature.fa"
    if not fasta_file.exists():
        # 检查其他可能的文件名
        alternatives = ["mature.fa.gz", "mature.fasta", "mature.fa.zip"]
        for alt in alternatives:
            alt_path = raw_dir / alt
            if alt_path.exists():
                print(f"📂 Found compressed file: {alt_path}")
                print(f"   Please decompress first: gunzip {alt_path}")
                return None
        
        print(f"❌ File not found: {fasta_file}")
        print(f"   请从 https://mirbase.org/download/ 下载 mature.fa")
        return None
    
    print(f"📂 Parsing: {fasta_file}")
    df = parse_fasta(fasta_file)
    print(f"   Total sequences: {len(df)}")
    
    # ---- 筛选人类 miRNA ----
    species_prefix = config.get("mirbase", {}).get("species_prefix", "hsa")
    human_mask = df['mirna_name'].str.startswith(species_prefix)
    df_human = df[human_mask].copy()
    print(f"🔍 Human miRNAs ({species_prefix}-*): {len(df_human)}")
    
    # ---- 标准化名称 ----
    df_human['mirna_name_normalized'] = df_human['mirna_name'].apply(normalize_mirna_name)
    
    # ---- 序列长度 ----
    df_human['seq_length'] = df_human['sequence'].str.len()
    print(f"\n📊 Sequence length statistics:")
    print(f"   Mean:   {df_human['seq_length'].mean():.1f}")
    print(f"   Median: {df_human['seq_length'].median():.1f}")
    print(f"   Min:    {df_human['seq_length'].min()}")
    print(f"   Max:    {df_human['seq_length'].max()}")
    
    # ---- RNA → 统一用大写 ----
    df_human['sequence'] = df_human['sequence'].str.upper()
    
    # ---- 去重（同名 miRNA 取第一个） ----
    n_before = len(df_human)
    df_human = df_human.drop_duplicates(subset=['mirna_name_normalized'], keep='first')
    print(f"🔍 Deduplicated: {n_before} → {len(df_human)}")
    
    # ---- 选择输出列 ----
    df_out = df_human[['mirna_name', 'mirna_name_normalized', 'mirna_id', 
                         'sequence', 'seq_length']].copy()
    
    print_stats(df_out, "miRBase Processed")
    save_processed(df_out, "mirna_sequences.csv", config)
    
    return df_out


if __name__ == "__main__":
    df = process_miRBase()
