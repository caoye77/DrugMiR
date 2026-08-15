"""
Step 6: 特征工程 —— 构建 Drug 和 miRNA 的特征向量 + 相似度矩阵
=================================================================

核心思路（来自 BioMediGraph 的教训）：
  所有相似度都来自【独立的外部数据源】，不从关联矩阵衍生。
  - Drug 相似度 ← SMILES 分子指纹（来自 DrugBank）
  - miRNA 相似度 ← 序列 k-mer 频率（来自 miRBase）
  - 跨网络桥接 ← 靶基因网络（来自 miRTarBase + DrugBank targets）

输出:
  - data/processed/drug_features.npy          (N_drugs × 1024 Morgan fingerprint)
  - data/processed/mirna_features.npy         (N_mirnas × 256 k-mer frequency)
  - data/processed/drug_similarity.npy        (N_drugs × N_drugs Tanimoto similarity)
  - data/processed/mirna_similarity.npy       (N_mirnas × N_mirnas cosine similarity)
  - data/processed/node_id_mapping.csv        (统一的 node → ID 映射)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product as iter_product
from collections import Counter
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_project_root, load_processed, print_stats


# ============================================================
# Drug Features: Morgan Fingerprint
# ============================================================

def compute_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    """
    计算 Morgan 指纹（Extended Connectivity Fingerprint, ECFP）
    
    Args:
        smiles_list: SMILES 字符串列表
        radius: Morgan 指纹半径（2 对应 ECFP4）
        n_bits: 指纹位数
    
    Returns:
        fingerprints: numpy array (N × n_bits)
        valid_mask: 哪些 SMILES 成功解析
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        print("❌ RDKit not installed!")
        print("   安装方法:")
        print("   conda install -c conda-forge rdkit")
        print("   或: pip install rdkit-pypi")
        return None, None
    
    fingerprints = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    valid_mask = np.zeros(len(smiles_list), dtype=bool)
    
    failed = []
    for i, smi in enumerate(tqdm(smiles_list, desc="Computing Morgan FPs")):
        if pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints[i] = arr
            valid_mask[i] = True
        else:
            failed.append((i, smi[:50]))
    
    n_valid = valid_mask.sum()
    print(f"✅ Morgan FP computed: {n_valid}/{len(smiles_list)} valid")
    if failed:
        print(f"⚠️  {len(failed)} SMILES failed to parse. Examples: {failed[:5]}")
    
    return fingerprints, valid_mask


def compute_tanimoto_similarity(fps):
    """
    计算 Tanimoto 相似度矩阵
    对于 binary fingerprints: Tanimoto(A,B) = |A∩B| / |A∪B|
    """
    n = fps.shape[0]
    sim = np.zeros((n, n), dtype=np.float32)
    
    for i in tqdm(range(n), desc="Computing Tanimoto similarity"):
        for j in range(i, n):
            intersection = np.sum(fps[i] * fps[j])
            union = np.sum(np.clip(fps[i] + fps[j], 0, 1))
            if union > 0:
                s = intersection / union
            else:
                s = 0.0
            sim[i, j] = s
            sim[j, i] = s
    
    return sim


# ============================================================
# miRNA Features: k-mer Frequency
# ============================================================

def compute_kmer_features(sequences, k=4):
    """
    计算 k-mer 频率特征
    
    对于 RNA (ACGU), k=4 → 4^4 = 256 维特征
    
    Args:
        sequences: RNA 序列列表
        k: k-mer 长度
    
    Returns:
        features: numpy array (N × 4^k)
    """
    bases = ['A', 'C', 'G', 'U']
    # 生成所有可能的 k-mer
    all_kmers = [''.join(p) for p in iter_product(bases, repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
    n_features = len(all_kmers)
    
    features = np.zeros((len(sequences), n_features), dtype=np.float32)
    
    for i, seq in enumerate(tqdm(sequences, desc=f"Computing {k}-mer features")):
        if pd.isna(seq):
            continue
        seq = str(seq).upper().replace('T', 'U')  # DNA → RNA
        
        # 统计 k-mer 频率
        kmer_counts = Counter()
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            if kmer in kmer_to_idx:
                kmer_counts[kmer] += 1
        
        # 归一化为频率
        total = sum(kmer_counts.values())
        if total > 0:
            for kmer, count in kmer_counts.items():
                features[i, kmer_to_idx[kmer]] = count / total
    
    print(f"✅ {k}-mer features computed: {features.shape}")
    print(f"   Non-zero features per sequence: {(features > 0).sum(axis=1).mean():.1f} / {n_features}")
    
    return features


def compute_cosine_similarity(features):
    """计算余弦相似度矩阵"""
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(features)
    return sim.astype(np.float32)


# ============================================================
# Entity Alignment: 对齐 ncRNADrug 中的实体到其他数据库
# ============================================================

def align_drugs(associations_df, drugs_df):
    """
    将 ncRNADrug 中的药物名称与 DrugBank 对齐
    返回匹配成功的药物列表 + 映射关系
    """
    # ncRNADrug 中的药物名 → DrugBank 中的药物名
    ncr_drugs = set(associations_df['drug_name'].unique())
    db_drugs = set(drugs_df['drug_name_normalized'].unique()) if 'drug_name_normalized' in drugs_df.columns else set()
    
    if not db_drugs and 'drug_name' in drugs_df.columns:
        db_drugs = set(drugs_df['drug_name'].str.lower().str.strip().unique())
    
    # 直接匹配
    matched = ncr_drugs & db_drugs
    unmatched = ncr_drugs - db_drugs
    
    print(f"\n💊 Drug alignment:")
    print(f"   ncRNADrug drugs: {len(ncr_drugs)}")
    print(f"   DrugBank drugs:  {len(db_drugs)}")
    print(f"   Matched:         {len(matched)} ({len(matched)/len(ncr_drugs)*100:.1f}%)")
    print(f"   Unmatched:       {len(unmatched)}")
    
    if unmatched and len(unmatched) <= 20:
        print(f"   Unmatched examples: {list(unmatched)[:10]}")
    
    return matched, unmatched


def align_mirnas(associations_df, sequences_df):
    """
    将 ncRNADrug 中的 miRNA 名称与 miRBase 对齐
    """
    ncr_mirnas = set(associations_df['mirna_name'].unique())
    mb_mirnas = set(sequences_df['mirna_name_normalized'].unique()) if 'mirna_name_normalized' in sequences_df.columns else set()
    
    # 直接匹配
    matched = ncr_mirnas & mb_mirnas
    
    # 模糊匹配：ncRNADrug 中的 hsa-mir-21 可能对应 hsa-mir-21-5p 或 hsa-mir-21-3p
    fuzzy_matches = {}
    for ncr_name in ncr_mirnas - matched:
        candidates = [m for m in mb_mirnas if m.startswith(ncr_name)]
        if candidates:
            # 优先选 -5p（主导链），否则选第一个
            five_p = [c for c in candidates if c.endswith('-5p')]
            fuzzy_matches[ncr_name] = five_p[0] if five_p else candidates[0]
    
    total_matched = len(matched) + len(fuzzy_matches)
    unmatched = ncr_mirnas - matched - set(fuzzy_matches.keys())
    
    print(f"\n🧬 miRNA alignment:")
    print(f"   ncRNADrug miRNAs: {len(ncr_mirnas)}")
    print(f"   miRBase miRNAs:   {len(mb_mirnas)}")
    print(f"   Exact matched:    {len(matched)}")
    print(f"   Fuzzy matched:    {len(fuzzy_matches)}")
    print(f"   Total matched:    {total_matched} ({total_matched/len(ncr_mirnas)*100:.1f}%)")
    print(f"   Unmatched:        {len(unmatched)}")
    
    return matched, fuzzy_matches, unmatched


# ============================================================
# Main
# ============================================================

def build_features():
    config = load_config()
    root = get_project_root()
    out_dir = root / config["paths"]["processed_data"]
    
    # ---- 加载预处理数据 ----
    print("📂 Loading processed data...")
    
    try:
        associations = load_processed("mirna_drug_associations.csv", config)
        print(f"   Associations: {associations.shape}")
    except FileNotFoundError:
        print("❌ mirna_drug_associations.csv not found. Run step2 first.")
        return
    
    try:
        drug_smiles = load_processed("drug_smiles.csv", config)
        print(f"   Drug SMILES: {drug_smiles.shape}")
    except FileNotFoundError:
        print("⚠️  drug_smiles.csv not found. Drug features will be skipped.")
        drug_smiles = None
    
    try:
        mirna_seqs = load_processed("mirna_sequences.csv", config)
        print(f"   miRNA sequences: {mirna_seqs.shape}")
    except FileNotFoundError:
        print("⚠️  mirna_sequences.csv not found. miRNA features will be skipped.")
        mirna_seqs = None
    
    # ---- 实体对齐 ----
    if drug_smiles is not None:
        drug_matched, drug_unmatched = align_drugs(associations, drug_smiles)
    
    if mirna_seqs is not None:
        mirna_matched, mirna_fuzzy, mirna_unmatched = align_mirnas(associations, mirna_seqs)
    
    # ---- 构建统一的实体列表 ----
    # 只保留在关联数据中出现的 + 有特征的实体
    final_mirnas = sorted(associations['mirna_name'].unique())
    final_drugs = sorted(associations['drug_name'].unique())
    
    print(f"\n📋 Final entity counts:")
    print(f"   miRNAs in associations: {len(final_mirnas)}")
    print(f"   Drugs in associations:  {len(final_drugs)}")
    
    # ---- 创建 ID 映射 ----
    mirna_to_id = {name: i for i, name in enumerate(final_mirnas)}
    drug_to_id = {name: i for i, name in enumerate(final_drugs)}
    
    # 保存映射
    mapping_data = []
    for name, idx in mirna_to_id.items():
        mapping_data.append({'node_name': name, 'node_id': idx, 'node_type': 'miRNA'})
    for name, idx in drug_to_id.items():
        mapping_data.append({'node_name': name, 'node_id': idx, 'node_type': 'drug'})
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(out_dir / "node_id_mapping.csv", index=False)
    print(f"✅ Node mapping saved: {len(mapping_df)} nodes")
    
    # ---- 计算 Drug 特征 + 相似度 ----
    fp_config = config.get("features", {}).get("morgan_fp", {})
    radius = fp_config.get("radius", 2)
    n_bits = fp_config.get("n_bits", 1024)
    
    if drug_smiles is not None:
        print(f"\n{'='*60}")
        print("🔬 Computing Drug Features (Morgan Fingerprint)")
        print(f"{'='*60}")
        
        # 为每个 drug 获取 SMILES
        drug_smiles_map = {}
        name_col = 'drug_name_normalized' if 'drug_name_normalized' in drug_smiles.columns else 'drug_name'
        for _, row in drug_smiles.iterrows():
            name = str(row[name_col]).lower().strip()
            if pd.notna(row['smiles']):
                drug_smiles_map[name] = row['smiles']
        
        smiles_list = [drug_smiles_map.get(d, None) for d in final_drugs]
        n_with_smiles = sum(1 for s in smiles_list if s is not None)
        print(f"   Drugs with SMILES: {n_with_smiles}/{len(final_drugs)}")
        
        drug_fps, valid_mask = compute_morgan_fingerprints(smiles_list, radius, n_bits)
        
        if drug_fps is not None:
            np.save(out_dir / "drug_features.npy", drug_fps)
            print(f"✅ Drug features saved: {drug_fps.shape}")
            
            # 相似度矩阵
            drug_sim = compute_tanimoto_similarity(drug_fps)
            np.save(out_dir / "drug_similarity.npy", drug_sim)
            
            threshold = config.get("features", {}).get("drug_sim_threshold", 0.3)
            n_edges = np.sum(drug_sim > threshold) - len(final_drugs)  # 减去对角线
            print(f"✅ Drug similarity saved: {drug_sim.shape}")
            print(f"   Edges above threshold {threshold}: {n_edges//2}")
    
    # ---- 计算 miRNA 特征 + 相似度 ----
    kmer_config = config.get("features", {}).get("kmer", {})
    k = kmer_config.get("k", 4)
    
    if mirna_seqs is not None:
        print(f"\n{'='*60}")
        print(f"🧬 Computing miRNA Features ({k}-mer frequency)")
        print(f"{'='*60}")
        
        # 为每个 miRNA 获取序列
        seq_map = {}
        name_col = 'mirna_name_normalized' if 'mirna_name_normalized' in mirna_seqs.columns else 'mirna_name'
        for _, row in mirna_seqs.iterrows():
            name = str(row[name_col]).lower().strip()
            seq_map[name] = row['sequence']
        
        # 包括模糊匹配
        if mirna_seqs is not None:
            for ncr_name, mb_name in mirna_fuzzy.items() if 'mirna_fuzzy' in dir() else []:
                if mb_name in seq_map:
                    seq_map[ncr_name] = seq_map[mb_name]
        
        seq_list = [seq_map.get(m, None) for m in final_mirnas]
        n_with_seq = sum(1 for s in seq_list if s is not None)
        print(f"   miRNAs with sequence: {n_with_seq}/{len(final_mirnas)}")
        
        mirna_features = compute_kmer_features(seq_list, k=k)
        np.save(out_dir / "mirna_features.npy", mirna_features)
        print(f"✅ miRNA features saved: {mirna_features.shape}")
        
        # 相似度矩阵
        mirna_sim = compute_cosine_similarity(mirna_features)
        np.save(out_dir / "mirna_similarity.npy", mirna_sim)
        
        threshold = config.get("features", {}).get("mirna_sim_threshold", 0.5)
        n_edges = np.sum(mirna_sim > threshold) - len(final_mirnas)
        print(f"✅ miRNA similarity saved: {mirna_sim.shape}")
        print(f"   Edges above threshold {threshold}: {n_edges//2}")
    
    print(f"\n{'='*60}")
    print("✅ Feature engineering complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    build_features()
