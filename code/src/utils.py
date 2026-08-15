"""
DrugMiR 工具函数
"""
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    """获取项目根目录（包含 configs/ 的那一层）"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "configs").exists():
            return current
        current = current.parent
    # fallback: 脚本上一层
    return Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        root = get_project_root()
        config_path = root / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def normalize_mirna_name(name):
    """
    标准化 miRNA 名称
    - 统一小写 (hsa-miR-21 → hsa-mir-21)
    - 处理常见变体 (miR vs mir, let vs Let)
    
    注意: miRBase 中 mature miRNA 用 miR（大写R），
    precursor 用 mir（小写r）。
    为了匹配，我们统一转小写。
    """
    if pd.isna(name):
        return None
    name = str(name).strip()
    # 统一转小写
    name = name.lower()
    # 移除多余空格
    name = name.replace(" ", "")
    return name


def normalize_drug_name(name):
    """标准化药物名称（小写 + strip）"""
    if pd.isna(name):
        return None
    return str(name).strip().lower()


def normalize_gene_symbol(symbol):
    """标准化基因名称（大写 + strip）"""
    if pd.isna(symbol):
        return None
    return str(symbol).strip().upper()


def print_stats(df, name="DataFrame"):
    """打印 DataFrame 基本统计"""
    print(f"\n{'='*60}")
    print(f"📊 {name} Statistics")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    for col in df.columns:
        n_unique = df[col].nunique()
        n_null = df[col].isna().sum()
        print(f"  [{col}] unique={n_unique}, null={n_null}")
    print(f"{'='*60}\n")


def save_processed(df, filename, config=None):
    """保存处理后的数据到 processed 目录"""
    if config is None:
        config = load_config()
    root = get_project_root()
    out_dir = ensure_dir(root / config["paths"]["processed_data"])
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path} ({len(df)} rows)")
    return out_path


def load_processed(filename, config=None):
    """加载处理后的数据"""
    if config is None:
        config = load_config()
    root = get_project_root()
    path = root / config["paths"]["processed_data"] / filename
    return pd.read_csv(path)
