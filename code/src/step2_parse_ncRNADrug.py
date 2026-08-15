"""
Step 2: 解析 ncRNADrug 数据库
================================
输入: data/raw/ncRNADrug/ 下的关联数据文件
输出: data/processed/mirna_drug_associations.csv

输出格式:
  mirna_name | drug_name | association_type | cancer_type | pmid
  hsa-mir-21 | cisplatin | resistance       | lung cancer | 12345678
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, get_project_root, normalize_mirna_name,
    normalize_drug_name, print_stats, save_processed
)


def parse_ncRNADrug_csv(file_path):
    """
    解析 ncRNADrug 导出的 CSV/TSV/Excel 文件
    ncRNADrug 的导出格式可能因版本而异，这里做兼容处理
    """
    ext = Path(file_path).suffix.lower()
    
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif ext == '.tsv':
        df = pd.read_csv(file_path, sep='\t')
    else:
        # 尝试自动检测分隔符
        df = pd.read_csv(file_path, sep=None, engine='python')
    
    print(f"📂 Raw file loaded: {file_path}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")
    
    return df


def standardize_columns(df):
    """
    标准化列名 —— ncRNADrug 的列名可能不固定
    尝试自动识别关键列
    """
    col_map = {}
    cols_lower = {c: c.lower().strip() for c in df.columns}
    
    # 识别 miRNA 列
    for orig, low in cols_lower.items():
        if any(kw in low for kw in ['mirna', 'mir', 'ncrna_name', 'ncrna']):
            col_map[orig] = 'mirna_name'
            break
    
    # 识别 Drug 列
    for orig, low in cols_lower.items():
        if any(kw in low for kw in ['drug', 'compound', 'chemical']):
            if orig not in col_map:
                col_map[orig] = 'drug_name'
                break
    
    # 识别关联类型列
    for orig, low in cols_lower.items():
        if any(kw in low for kw in ['association', 'relation', 'type', 'effect', 'resistance', 'sensitivity']):
            if orig not in col_map:
                col_map[orig] = 'association_type'
                break
    
    # 识别癌症类型列
    for orig, low in cols_lower.items():
        if any(kw in low for kw in ['cancer', 'disease', 'tumor', 'cell_line']):
            if orig not in col_map:
                col_map[orig] = 'cancer_type'
                break
    
    # 识别 PMID 列
    for orig, low in cols_lower.items():
        if any(kw in low for kw in ['pmid', 'pubmed', 'reference']):
            if orig not in col_map:
                col_map[orig] = 'pmid'
                break
    
    print(f"\n🔄 Column mapping: {col_map}")
    
    if 'mirna_name' not in col_map.values() or 'drug_name' not in col_map.values():
        print("⚠️  WARNING: Could not auto-detect miRNA or Drug column!")
        print(f"   Available columns: {list(df.columns)}")
        print("   Please manually specify column mapping in this script.")
        # 打印前几行帮助调试
        print(df.head())
    
    df = df.rename(columns=col_map)
    return df


def standardize_association_type(assoc):
    """标准化关联类型为 resistance / sensitivity"""
    if pd.isna(assoc):
        return None
    assoc = str(assoc).strip().lower()
    
    if any(kw in assoc for kw in ['resist', 'insensitiv']):
        return 'resistance'
    elif any(kw in assoc for kw in ['sensitiv', 'sensit', 'responsive']):
        return 'sensitivity'
    else:
        return assoc  # 保留原始值，后续手动检查


def process_ncRNADrug():
    """主处理流程"""
    config = load_config()
    root = get_project_root()
    raw_dir = root / config["paths"]["raw_data"] / "ncRNADrug"
    
    # 查找所有可能的数据文件
    data_files = []
    for ext in ['*.csv', '*.tsv', '*.xlsx', '*.xls', '*.txt']:
        data_files.extend(glob.glob(str(raw_dir / ext)))
    
    if not data_files:
        print(f"❌ No data files found in {raw_dir}")
        print(f"   请按 step1_download_guide.md 下载 ncRNADrug 数据")
        print(f"\n   === 备选方案 ===")
        print(f"   如果无法直接下载，可以：")
        print(f"   1. 从 ncRNADrug 网站逐页复制数据到 Excel")
        print(f"   2. 使用论文 Supplementary Material")
        print(f"   3. 检查 GitHub 上是否有人分享过该数据")
        print(f"   4. 使用 SM2miR 数据库作为替代 (http://www.jianglab.cn/SM2miR/)")
        return None
    
    print(f"📁 Found {len(data_files)} file(s): {data_files}")
    
    # 合并所有文件
    all_dfs = []
    for f in data_files:
        df = parse_ncRNADrug_csv(f)
        df = standardize_columns(df)
        all_dfs.append(df)
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # ---- 清洗 ----
    
    # 1. 标准化名称
    if 'mirna_name' in df.columns:
        df['mirna_name'] = df['mirna_name'].apply(normalize_mirna_name)
    
    if 'drug_name' in df.columns:
        df['drug_name'] = df['drug_name'].apply(normalize_drug_name)
    
    # 2. 标准化关联类型
    if 'association_type' in df.columns:
        df['association_type'] = df['association_type'].apply(standardize_association_type)
        print(f"\n📊 Association type distribution:")
        print(df['association_type'].value_counts())
    
    # 3. 只保留 miRNA（过滤掉其他 ncRNA 类型如 lncRNA, circRNA）
    if 'mirna_name' in df.columns:
        mirna_mask = df['mirna_name'].str.contains('mir|let', case=False, na=False)
        n_before = len(df)
        df = df[mirna_mask].copy()
        print(f"\n🔍 Filtered to miRNAs: {n_before} → {len(df)}")
    
    # 4. 只保留人类
    if 'mirna_name' in df.columns:
        human_mask = df['mirna_name'].str.startswith('hsa', na=False)
        n_before = len(df)
        df = df[human_mask].copy()
        print(f"🔍 Filtered to human: {n_before} → {len(df)}")
    
    # 5. 去重
    subset_cols = [c for c in ['mirna_name', 'drug_name', 'association_type'] if c in df.columns]
    if subset_cols:
        n_before = len(df)
        df = df.drop_duplicates(subset=subset_cols).copy()
        print(f"🔍 Deduplicated: {n_before} → {len(df)}")
    
    # 6. 删除缺失关键字段的行
    key_cols = [c for c in ['mirna_name', 'drug_name'] if c in df.columns]
    if key_cols:
        df = df.dropna(subset=key_cols).copy()
    
    # ---- 统计 ----
    print_stats(df, "ncRNADrug Processed")
    
    if 'mirna_name' in df.columns:
        print(f"  Unique miRNAs: {df['mirna_name'].nunique()}")
    if 'drug_name' in df.columns:
        print(f"  Unique Drugs:  {df['drug_name'].nunique()}")
    if 'association_type' in df.columns:
        print(f"\n  Association breakdown:")
        for atype, count in df['association_type'].value_counts().items():
            print(f"    {atype}: {count}")
    
    # ---- 保存 ----
    save_processed(df, "mirna_drug_associations.csv", config)
    
    return df


if __name__ == "__main__":
    df = process_ncRNADrug()
