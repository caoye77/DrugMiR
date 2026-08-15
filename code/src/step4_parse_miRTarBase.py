"""
Step 4: 解析 miRTarBase 数据
================================
输入: data/raw/miRTarBase/hsa_MTI.xlsx (或 .csv)
输出: data/processed/mirna_gene_targets.csv

输出格式:
  mirna_name | gene_symbol | evidence_type | support_type
  hsa-mir-21 | PTEN        | Reporter...   | Functional MTI
"""
import sys
import pandas as pd
from pathlib import Path
import glob

sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, get_project_root, normalize_mirna_name,
    normalize_gene_symbol, print_stats, save_processed
)


def process_miRTarBase():
    config = load_config()
    root = get_project_root()
    raw_dir = root / config["paths"]["raw_data"] / "miRTarBase"
    
    # 查找数据文件
    data_files = []
    for ext in ['*.xlsx', '*.xls', '*.csv', '*.tsv', '*.txt']:
        data_files.extend(glob.glob(str(raw_dir / ext)))
    
    if not data_files:
        print(f"❌ No data files found in {raw_dir}")
        print(f"   请从 https://mirtarbase.cuhk.edu.cn/ 下载数据")
        print(f"   下载 Download → miRTarBase → Homo sapiens")
        return None
    
    print(f"📁 Found: {data_files}")
    
    # 读取第一个文件
    fpath = data_files[0]
    ext = Path(fpath).suffix.lower()
    
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(fpath)
    else:
        df = pd.read_csv(fpath, sep=None, engine='python')
    
    print(f"📂 Raw file: {fpath}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # ---- 列名标准化 ----
    # miRTarBase 常见列名
    col_map = {}
    cols_lower = {c: c.lower().strip() for c in df.columns}
    
    for orig, low in cols_lower.items():
        if 'mirna' in low and 'mirna' not in [col_map.get(k) for k in col_map]:
            col_map[orig] = 'mirna_name'
        elif 'target gene' in low or ('gene' in low and 'symbol' not in low and 'id' not in low):
            if 'gene_symbol' not in col_map.values():
                col_map[orig] = 'target_gene_name'
        elif 'gene' in low and ('symbol' in low or 'name' in low):
            col_map[orig] = 'gene_symbol'
        elif 'experiment' in low:
            col_map[orig] = 'experiments'
        elif 'support' in low:
            col_map[orig] = 'support_type'
        elif 'species' in low and 'target' in low:
            col_map[orig] = 'target_species'
        elif 'species' in low:
            col_map[orig] = 'mirna_species'
    
    df = df.rename(columns=col_map)
    print(f"🔄 Column mapping: {col_map}")
    
    # ---- 筛选人类 ----
    species_col = None
    if 'mirna_species' in df.columns:
        species_col = 'mirna_species'
    elif 'target_species' in df.columns:
        species_col = 'target_species'
    
    if species_col:
        human_mask = df[species_col].str.contains('sapiens|human', case=False, na=False)
        n_before = len(df)
        df = df[human_mask].copy()
        print(f"🔍 Filtered to human: {n_before} → {len(df)}")
    else:
        # 按 miRNA 名称前缀过滤
        if 'mirna_name' in df.columns:
            df['mirna_name_norm'] = df['mirna_name'].apply(normalize_mirna_name)
            df = df[df['mirna_name_norm'].str.startswith('hsa', na=False)].copy()
    
    # ---- 证据强度过滤 ----
    evidence_filter = config.get("mirtarbase", {}).get("evidence_filter", "Strong")
    
    if 'support_type' in df.columns:
        print(f"\n📊 Support type distribution:")
        print(df['support_type'].value_counts())
        
        if evidence_filter == "Strong":
            # 只保留 "Functional MTI" (强证据) 
            strong_mask = df['support_type'].str.contains(
                'functional|strong', case=False, na=False
            )
            n_before = len(df)
            df = df[strong_mask].copy()
            print(f"\n🔍 Filtered to strong evidence: {n_before} → {len(df)}")
    
    # ---- 标准化名称 ----
    if 'mirna_name' in df.columns:
        df['mirna_name'] = df['mirna_name'].apply(normalize_mirna_name)
    
    # 获取基因 symbol
    gene_col = 'gene_symbol' if 'gene_symbol' in df.columns else 'target_gene_name'
    if gene_col in df.columns:
        df['gene_symbol'] = df[gene_col].apply(normalize_gene_symbol)
    
    # ---- 去重 ----
    key_cols = [c for c in ['mirna_name', 'gene_symbol'] if c in df.columns]
    n_before = len(df)
    df = df.drop_duplicates(subset=key_cols)
    print(f"🔍 Deduplicated (miRNA-gene pairs): {n_before} → {len(df)}")
    
    # ---- 选择输出列 ----
    out_cols = ['mirna_name', 'gene_symbol']
    for c in ['experiments', 'support_type']:
        if c in df.columns:
            out_cols.append(c)
    
    df_out = df[out_cols].copy()
    df_out = df_out.dropna(subset=['mirna_name', 'gene_symbol'])
    
    # ---- 统计 ----
    print_stats(df_out, "miRTarBase Processed")
    print(f"  Unique miRNAs: {df_out['mirna_name'].nunique()}")
    print(f"  Unique Genes:  {df_out['gene_symbol'].nunique()}")
    print(f"  Total pairs:   {len(df_out)}")
    
    # ---- 保存 ----
    save_processed(df_out, "mirna_gene_targets.csv", config)
    
    return df_out


if __name__ == "__main__":
    df = process_miRTarBase()
