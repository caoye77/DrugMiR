"""
Step 3: 解析 DrugBank 数据
================================
输入: data/raw/DrugBank/ (XML 或 CSV)
输出:
  - data/processed/drug_smiles.csv      (drugbank_id, drug_name, smiles)
  - data/processed/drug_targets.csv     (drugbank_id, drug_name, gene_symbol, uniprot_id)

支持两种模式:
  A) 完整 XML 模式 (drugbank_all_full_database.xml) — 信息最全
  B) 轻量 CSV 模式 (drug_links.csv + drug_target_identifiers.csv) — 更快
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, get_project_root, normalize_drug_name,
    normalize_gene_symbol, print_stats, save_processed
)


# ============================================================
# Mode A: Parse full DrugBank XML
# ============================================================

def parse_drugbank_xml(xml_path):
    """
    解析 DrugBank 完整 XML
    提取: DrugBank ID, Drug Name, SMILES, Target Genes
    
    注意: XML 约 1.5GB，解析需要 3-5 分钟
    """
    from lxml import etree
    
    print(f"📂 Parsing DrugBank XML: {xml_path}")
    print(f"   This may take 3-5 minutes...")
    
    ns = '{http://www.drugbank.ca}'
    
    drugs_data = []
    targets_data = []
    
    # 使用 iterparse 避免内存爆炸
    context = etree.iterparse(xml_path, events=('end',), tag=f'{ns}drug')
    
    count = 0
    for event, drug_elem in context:
        # 只处理顶层 drug 元素
        if drug_elem.getparent() is not None and drug_elem.getparent().tag == f'{ns}drugs':
            pass  # 是顶层 drug
        elif drug_elem.getparent() is not None and drug_elem.getparent().tag != f'{ns}drugs':
            continue  # 跳过嵌套的 drug 元素（如 drug-interactions 中的）
        
        # DrugBank ID
        db_id = None
        for identifier in drug_elem.findall(f'{ns}drugbank-id'):
            if identifier.get('primary') == 'true':
                db_id = identifier.text
                break
        if db_id is None:
            id_elem = drug_elem.find(f'{ns}drugbank-id')
            if id_elem is not None:
                db_id = id_elem.text
        
        # Drug name
        name_elem = drug_elem.find(f'{ns}name')
        drug_name = name_elem.text if name_elem is not None else None
        
        # SMILES (from calculated-properties or experimental-properties)
        smiles = None
        for prop_group in [f'{ns}calculated-properties', f'{ns}experimental-properties']:
            props = drug_elem.find(prop_group)
            if props is not None:
                for prop in props.findall(f'{ns}property'):
                    kind = prop.find(f'{ns}kind')
                    if kind is not None and kind.text == 'SMILES':
                        val = prop.find(f'{ns}value')
                        if val is not None:
                            smiles = val.text
                            break
            if smiles:
                break
        
        # 存储药物信息
        if db_id and drug_name:
            drugs_data.append({
                'drugbank_id': db_id,
                'drug_name': drug_name,
                'smiles': smiles
            })
        
        # Targets
        targets_elem = drug_elem.find(f'{ns}targets')
        if targets_elem is not None:
            for target in targets_elem.findall(f'{ns}target'):
                # Gene name
                gene_name = None
                name_e = target.find(f'{ns}name')
                
                # Polypeptide 中有更详细的信息
                polypeptide = target.find(f'{ns}polypeptide')
                if polypeptide is not None:
                    gene_elem = polypeptide.find(f'{ns}gene-name')
                    if gene_elem is not None:
                        gene_name = gene_elem.text
                    
                    # UniProt ID
                    uniprot_id = polypeptide.get('id')  # attribute
                    
                    # Organism
                    org_elem = polypeptide.find(f'{ns}organism')
                    organism = org_elem.text if org_elem is not None else None
                    
                    # 只要人类靶标
                    if organism and 'human' in organism.lower():
                        targets_data.append({
                            'drugbank_id': db_id,
                            'drug_name': drug_name,
                            'gene_symbol': gene_name,
                            'uniprot_id': uniprot_id
                        })
        
        # 清理已处理的元素，释放内存
        drug_elem.clear()
        while drug_elem.getprevious() is not None:
            del drug_elem.getparent()[0]
        
        count += 1
        if count % 2000 == 0:
            print(f"   Processed {count} drugs...")
    
    drugs_df = pd.DataFrame(drugs_data)
    targets_df = pd.DataFrame(targets_data)
    
    print(f"\n✅ XML parsing complete: {count} drugs total")
    
    return drugs_df, targets_df


# ============================================================
# Mode B: Parse DrugBank CSV files (lighter)
# ============================================================

def parse_drugbank_csv(raw_dir):
    """
    解析 DrugBank 的轻量 CSV 文件
    需要: drug_links.csv (含 SMILES) 和 drug_target_identifiers.csv
    """
    print("📂 Parsing DrugBank CSV files...")
    
    drugs_df = None
    targets_df = None
    
    # Drug links (有 DrugBank ID 和名称的映射)
    links_file = raw_dir / "drug_links.csv"
    if links_file.exists():
        links = pd.read_csv(links_file)
        print(f"   drug_links.csv: {links.shape}")
    
    # 尝试各种可能的文件名
    for fname in ["structures.csv", "drug_structures.csv", "all_structures.csv"]:
        fpath = raw_dir / fname
        if fpath.exists():
            structs = pd.read_csv(fpath)
            # 提取 SMILES
            smiles_col = [c for c in structs.columns if 'smiles' in c.lower()]
            if smiles_col:
                drugs_df = structs.rename(columns={smiles_col[0]: 'smiles'})
                break
    
    # Target identifiers
    for fname in ["drug_target_identifiers.csv", "all_target_ids_all.csv",
                   "drug_targets.csv", "target_identifiers.csv"]:
        fpath = raw_dir / fname
        if fpath.exists():
            targets_df = pd.read_csv(fpath)
            print(f"   {fname}: {targets_df.shape}")
            break
    
    return drugs_df, targets_df


# ============================================================
# Main
# ============================================================

def process_drugbank():
    """主处理流程"""
    config = load_config()
    root = get_project_root()
    raw_dir = root / config["paths"]["raw_data"] / "DrugBank"
    
    drugs_df = None
    targets_df = None
    
    # 检测可用的数据文件
    xml_file = raw_dir / "drugbank_all_full_database.xml"
    xml_file_alt = raw_dir / "full database.xml"
    
    if xml_file.exists():
        drugs_df, targets_df = parse_drugbank_xml(xml_file)
    elif xml_file_alt.exists():
        drugs_df, targets_df = parse_drugbank_xml(xml_file_alt)
    else:
        # 尝试 CSV 模式
        drugs_df, targets_df = parse_drugbank_csv(raw_dir)
    
    if drugs_df is None and targets_df is None:
        print(f"❌ No DrugBank data found in {raw_dir}")
        print(f"   请按 step1_download_guide.md 下载 DrugBank 数据")
        print(f"\n   快速方案: 下载 CSV 版本（不需要等 Academic License 审批）")
        print(f"   https://go.drugbank.com/releases/latest#open-data")
        return None, None
    
    # ---- 处理药物 SMILES ----
    if drugs_df is not None:
        # 标准化药物名
        if 'drug_name' in drugs_df.columns:
            drugs_df['drug_name_normalized'] = drugs_df['drug_name'].apply(normalize_drug_name)
        
        # 过滤没有 SMILES 的
        has_smiles = drugs_df['smiles'].notna() & (drugs_df['smiles'].str.len() >= 5)
        print(f"\n💊 Drugs with valid SMILES: {has_smiles.sum()} / {len(drugs_df)}")
        drugs_with_smiles = drugs_df[has_smiles].copy()
        
        print_stats(drugs_with_smiles, "DrugBank - Drugs with SMILES")
        save_processed(drugs_with_smiles, "drug_smiles.csv", config)
    
    # ---- 处理靶基因 ----
    if targets_df is not None:
        # 标准化基因名
        if 'gene_symbol' in targets_df.columns:
            targets_df['gene_symbol'] = targets_df['gene_symbol'].apply(normalize_gene_symbol)
        
        # 去重
        key_cols = [c for c in ['drugbank_id', 'gene_symbol'] if c in targets_df.columns]
        if key_cols:
            targets_df = targets_df.drop_duplicates(subset=key_cols)
        
        # 过滤无效基因名
        targets_df = targets_df[targets_df['gene_symbol'].notna()].copy()
        
        print_stats(targets_df, "DrugBank - Drug Targets")
        save_processed(targets_df, "drug_targets.csv", config)
    
    return drugs_df, targets_df


if __name__ == "__main__":
    drugs_df, targets_df = process_drugbank()
