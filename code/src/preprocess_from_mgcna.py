"""
DrugMiR 统一预处理脚本
========================
从 MGCNA 的 Excel 数据出发，一键构建所有特征和图结构。

输入: data/raw/MGCNA/data/data/*.xlsx
输出: data/processed/ 下的所有文件

运行: cd D:\\DrugMiR && python src/preprocess_from_mgcna.py
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product as iter_product
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 配置
# ============================================================
MGCNA_DATA = Path("data/raw/MGCNA/data/data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Morgan fingerprint 参数
MORGAN_RADIUS = 2
MORGAN_NBITS = 1024

# k-mer 参数
KMER_K = 4

# 相似度阈值（建边用）
DRUG_SIM_THRESHOLD = 0.3
MIRNA_SIM_THRESHOLD = 0.5


def load_excel(name):
    """加载 MGCNA Excel 文件"""
    path = MGCNA_DATA / name
    df = pd.read_excel(path, engine='openpyxl')
    print(f"  ✅ {name}: {df.shape}")
    return df


# ============================================================
# Step 1: 加载原始数据
# ============================================================
def step1_load_data():
    print("\n" + "=" * 60)
    print("📂 Step 1: 加载 MGCNA 原始数据")
    print("=" * 60)

    data = {}
    data['mirna_drug'] = load_excel("miRNA-drug-matrix.xlsx")
    data['drug_smiles'] = load_excel("drug-smiles.xlsx")
    data['mirna_seq'] = load_excel("miRNA-sequences.xlsx")
    data['mirna_gene'] = load_excel("miRNA-gene-matrix.xlsx")
    data['drug_gene'] = load_excel("drug-gene-matrix.xlsx")
    data['gene_name'] = load_excel("gene-name.xlsx")

    # 加载正负边列表
    pos_path = MGCNA_DATA / "pos.edgelist"
    neg_path = MGCNA_DATA / "neg.edgelist"
    if pos_path.exists():
        data['pos_edges'] = pd.read_csv(pos_path, sep='\t', header=None, names=['mirna_idx', 'drug_idx'])
        print(f"  ✅ pos.edgelist: {len(data['pos_edges'])} edges")
    if neg_path.exists():
        data['neg_edges'] = pd.read_csv(neg_path, sep='\t', header=None, names=['mirna_idx', 'drug_idx'])
        print(f"  ✅ neg.edgelist: {len(data['neg_edges'])} edges")

    return data


# ============================================================
# Step 2: 解析实体名称和关联矩阵
# ============================================================
def step2_parse_entities(data):
    print("\n" + "=" * 60)
    print("📋 Step 2: 解析实体名称 + 关联矩阵")
    print("=" * 60)

    # ---- miRNA 名称和序列 ----
    mirna_df = data['mirna_seq'].copy()
    mirna_df.columns = [str(c).strip() for c in mirna_df.columns]
    print(f"  miRNA columns: {list(mirna_df.columns)}")
    # 实际列: ['miRNA'(MIMAT ID), 'Sequence', 'miRNA_name']
    # 智能识别列
    cols_lower = {c: c.lower() for c in mirna_df.columns}
    seq_col = [c for c, low in cols_lower.items() if 'seq' in low]
    name_col = [c for c, low in cols_lower.items() if 'name' in low]
    id_col = [c for c, low in cols_lower.items() if c not in (seq_col + name_col)]
    
    if seq_col and name_col:
        mirna_names = mirna_df[name_col[0]].tolist()
        mirna_sequences = mirna_df[seq_col[0]].tolist()
        mirna_ids = mirna_df[id_col[0]].tolist() if id_col else mirna_names
    else:
        # fallback: 假设 col0=ID, col1=seq, col2=name
        mirna_ids = mirna_df.iloc[:, 0].tolist()
        mirna_sequences = mirna_df.iloc[:, 1].tolist()
        mirna_names = mirna_df.iloc[:, 2].tolist() if mirna_df.shape[1] >= 3 else mirna_ids
    
    print(f"  miRNAs: {len(mirna_names)} (例: {mirna_names[:3]})")
    print(f"  序列示例: {str(mirna_sequences[0])[:50]}...")

    # ---- Drug 名称和 SMILES ----
    drug_df = data['drug_smiles'].copy()
    drug_df.columns = [str(c).strip() for c in drug_df.columns]
    print(f"  Drug columns: {list(drug_df.columns)}")
    # 实际列: ['DrugBank_ID', 'smiles', 'Drug_Name']
    # 智能识别列
    cols_lower = {c: c.lower() for c in drug_df.columns}
    smiles_col = [c for c, low in cols_lower.items() if 'smiles' in low or 'smi' in low]
    dname_col = [c for c, low in cols_lower.items() if 'name' in low]
    did_col = [c for c, low in cols_lower.items() if 'id' in low or 'bank' in low]
    
    if smiles_col and dname_col:
        drug_names = drug_df[dname_col[0]].tolist()
        smiles_list = drug_df[smiles_col[0]].tolist()
    else:
        # fallback: col0=ID, col1=SMILES, col2=Name
        smiles_list = drug_df.iloc[:, 1].tolist()
        drug_names = drug_df.iloc[:, 2].tolist() if drug_df.shape[1] >= 3 else drug_df.iloc[:, 0].tolist()
    
    print(f"  Drugs: {len(drug_names)} (例: {drug_names[:3]})")
    print(f"  SMILES示例: {str(smiles_list[0])[:60]}...")

    # ---- 关联矩阵 ----
    assoc_df = data['mirna_drug'].copy()
    # 第一列可能是 miRNA 名称，也可能直接是矩阵
    # 检查第一列是否是名称
    first_col = assoc_df.iloc[:, 0]
    if first_col.dtype == object:
        # 第一列是名称，去掉
        assoc_matrix = assoc_df.iloc[:, 1:].values.astype(np.float32)
    else:
        assoc_matrix = assoc_df.values.astype(np.float32)
    print(f"  关联矩阵: {assoc_matrix.shape}")
    print(f"  正关联数: {int(assoc_matrix.sum())}")

    # ---- Gene 名称 ----
    gene_df = data['gene_name'].copy()
    gene_df.columns = [str(c).strip() for c in gene_df.columns]
    print(f"  Gene columns: {list(gene_df.columns)}")
    gene_names = gene_df.iloc[:, 0].tolist()
    print(f"  Genes: {len(gene_names)} (例: {gene_names[:3]})")

    # ---- miRNA-Gene 矩阵 ----
    mg_df = data['mirna_gene'].copy()
    first_col = mg_df.iloc[:, 0]
    if first_col.dtype == object:
        mirna_gene_matrix = mg_df.iloc[:, 1:].values.astype(np.float32)
    else:
        mirna_gene_matrix = mg_df.values.astype(np.float32)
    print(f"  miRNA-Gene 矩阵: {mirna_gene_matrix.shape}, 非零: {int(mirna_gene_matrix.sum())}")

    # ---- Drug-Gene 矩阵 ----
    dg_df = data['drug_gene'].copy()
    first_col = dg_df.iloc[:, 0]
    if first_col.dtype == object:
        drug_gene_matrix = dg_df.iloc[:, 1:].values.astype(np.float32)
    else:
        drug_gene_matrix = dg_df.values.astype(np.float32)
    print(f"  Drug-Gene 矩阵: {drug_gene_matrix.shape}, 非零: {int(drug_gene_matrix.sum())}")

    result = {
        'mirna_names': mirna_names,
        'mirna_ids': mirna_ids,
        'mirna_sequences': mirna_sequences,
        'drug_names': drug_names,
        'smiles_list': smiles_list,
        'gene_names': gene_names,
        'assoc_matrix': assoc_matrix,
        'mirna_gene_matrix': mirna_gene_matrix,
        'drug_gene_matrix': drug_gene_matrix,
        'drug_df': drug_df,
        'mirna_df': mirna_df,
    }
    if 'pos_edges' in data:
        result['pos_edges'] = data['pos_edges']
    if 'neg_edges' in data:
        result['neg_edges'] = data['neg_edges']

    return result


# ============================================================
# Step 3: 药物特征 — Morgan Fingerprint + Tanimoto 相似度
# ============================================================
def step3_drug_features(parsed):
    print("\n" + "=" * 60)
    print("💊 Step 3: 药物 Morgan 指纹 + Tanimoto 相似度")
    print("=" * 60)

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        from rdkit import RDLogger
        RDLogger.logger().setLevel(RDLogger.ERROR)
    except ImportError:
        print("  ❌ RDKit 未安装! 使用 MGCNA 原始药物特征作为替代")
        return None, None

    smiles_list = parsed['smiles_list']
    n_drugs = len(smiles_list)
    fps = np.zeros((n_drugs, MORGAN_NBITS), dtype=np.float32)
    valid = 0

    for i, smi in enumerate(tqdm(smiles_list, desc="  Morgan FP")):
        if pd.isna(smi) or not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
            arr = np.zeros(MORGAN_NBITS, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps[i] = arr
            valid += 1

    print(f"  有效 SMILES: {valid}/{n_drugs}")

    # Tanimoto 相似度
    print("  计算 Tanimoto 相似度矩阵...")
    drug_sim = np.zeros((n_drugs, n_drugs), dtype=np.float32)
    for i in range(n_drugs):
        for j in range(i, n_drugs):
            inter = np.sum(fps[i] * fps[j])
            union = np.sum(np.clip(fps[i] + fps[j], 0, 1))
            s = inter / union if union > 0 else 0.0
            drug_sim[i, j] = s
            drug_sim[j, i] = s

    n_edges = np.sum(drug_sim > DRUG_SIM_THRESHOLD) - n_drugs
    print(f"  Drug 相似度边 (>{DRUG_SIM_THRESHOLD}): {n_edges // 2}")

    # 保存
    np.save(OUTPUT_DIR / "drug_morgan_features.npy", fps)
    np.save(OUTPUT_DIR / "drug_similarity.npy", drug_sim)
    print(f"  ✅ 已保存 drug_morgan_features.npy ({fps.shape})")
    print(f"  ✅ 已保存 drug_similarity.npy ({drug_sim.shape})")

    return fps, drug_sim


# ============================================================
# Step 4: miRNA 特征 — k-mer 频率 + Cosine 相似度
# ============================================================
def step4_mirna_features(parsed):
    print("\n" + "=" * 60)
    print(f"🧬 Step 4: miRNA {KMER_K}-mer 频率 + Cosine 相似度")
    print("=" * 60)

    sequences = parsed['mirna_sequences']
    bases = ['A', 'C', 'G', 'U']
    all_kmers = [''.join(p) for p in iter_product(bases, repeat=KMER_K)]
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
    n_features = len(all_kmers)  # 4^4 = 256

    n_mirnas = len(sequences)
    features = np.zeros((n_mirnas, n_features), dtype=np.float32)
    valid = 0

    for i, seq in enumerate(tqdm(sequences, desc="  k-mer 频率")):
        if pd.isna(seq) or not isinstance(seq, str):
            continue
        seq = seq.upper().replace('T', 'U').strip()
        kmer_counts = Counter()
        for j in range(len(seq) - KMER_K + 1):
            kmer = seq[j:j + KMER_K]
            if kmer in kmer_to_idx:
                kmer_counts[kmer] += 1
        total = sum(kmer_counts.values())
        if total > 0:
            for kmer, count in kmer_counts.items():
                features[i, kmer_to_idx[kmer]] = count / total
            valid += 1

    print(f"  有效序列: {valid}/{n_mirnas}")
    print(f"  平均非零特征: {(features > 0).sum(axis=1).mean():.1f}/{n_features}")

    # Cosine 相似度
    print("  计算 Cosine 相似度矩阵...")
    mirna_sim = cosine_similarity(features).astype(np.float32)

    n_edges = np.sum(mirna_sim > MIRNA_SIM_THRESHOLD) - n_mirnas
    print(f"  miRNA 相似度边 (>{MIRNA_SIM_THRESHOLD}): {n_edges // 2}")

    # 保存
    np.save(OUTPUT_DIR / "mirna_kmer_features.npy", features)
    np.save(OUTPUT_DIR / "mirna_similarity.npy", mirna_sim)
    print(f"  ✅ 已保存 mirna_kmer_features.npy ({features.shape})")
    print(f"  ✅ 已保存 mirna_similarity.npy ({mirna_sim.shape})")

    return features, mirna_sim


# ============================================================
# Step 5: 构建基因桥接网络统计
# ============================================================
def step5_gene_bridge(parsed):
    print("\n" + "=" * 60)
    print("🧬 Step 5: 基因桥接网络分析")
    print("=" * 60)

    mg = parsed['mirna_gene_matrix']  # (n_mirna, n_gene)
    dg = parsed['drug_gene_matrix']   # (n_drug, n_gene)

    # miRNA 靶向的基因集合
    mirna_targets = set(np.where(mg.sum(axis=0) > 0)[0])
    drug_targets = set(np.where(dg.sum(axis=0) > 0)[0])
    bridge_genes = mirna_targets & drug_targets

    print(f"  miRNA 靶向基因数: {len(mirna_targets)}")
    print(f"  Drug 靶向基因数: {len(drug_targets)}")
    print(f"  🌉 桥接基因数 (同时被 miRNA 和 Drug 靶向): {len(bridge_genes)}")
    print(f"  miRNA-Gene 非零关联: {int(mg.sum())}")
    print(f"  Drug-Gene 非零关联: {int(dg.sum())}")

    # 保存基因桥接索引
    bridge_idx = sorted(bridge_genes)
    np.save(OUTPUT_DIR / "bridge_gene_indices.npy", np.array(bridge_idx))
    print(f"  ✅ 已保存 bridge_gene_indices.npy ({len(bridge_idx)} genes)")

    return bridge_idx


# ============================================================
# Step 6: 构建边列表 + 保存所有矩阵
# ============================================================
def step6_build_edges(parsed, drug_sim, mirna_sim):
    print("\n" + "=" * 60)
    print("🔗 Step 6: 构建所有边列表")
    print("=" * 60)

    assoc = parsed['assoc_matrix']
    mg = parsed['mirna_gene_matrix']
    dg = parsed['drug_gene_matrix']

    n_mirna = assoc.shape[0]
    n_drug = assoc.shape[1]
    n_gene = mg.shape[1]

    # 1. miRNA-Drug 关联边（正样本）
    pos_i, pos_j = np.where(assoc > 0)
    print(f"  miRNA-Drug 正关联边: {len(pos_i)}")

    # 2. Drug-Drug 相似度边
    if drug_sim is not None:
        dd_i, dd_j = np.where(np.triu(drug_sim, k=1) > DRUG_SIM_THRESHOLD)
        print(f"  Drug-Drug 相似度边: {len(dd_i)}")
    else:
        dd_i, dd_j = np.array([]), np.array([])

    # 3. miRNA-miRNA 相似度边
    if mirna_sim is not None:
        mm_i, mm_j = np.where(np.triu(mirna_sim, k=1) > MIRNA_SIM_THRESHOLD)
        print(f"  miRNA-miRNA 相似度边: {len(mm_i)}")
    else:
        mm_i, mm_j = np.array([]), np.array([])

    # 4. miRNA-Gene 边
    mg_i, mg_j = np.where(mg > 0)
    print(f"  miRNA-Gene 边: {len(mg_i)}")

    # 5. Drug-Gene 边
    dg_i, dg_j = np.where(dg > 0)
    print(f"  Drug-Gene 边: {len(dg_i)}")

    # 保存所有边列表
    edges = {
        'mirna_drug_pos': np.stack([pos_i, pos_j], axis=1),
        'drug_drug_sim': np.stack([dd_i, dd_j], axis=1) if len(dd_i) > 0 else np.zeros((0, 2), dtype=int),
        'mirna_mirna_sim': np.stack([mm_i, mm_j], axis=1) if len(mm_i) > 0 else np.zeros((0, 2), dtype=int),
        'mirna_gene': np.stack([mg_i, mg_j], axis=1),
        'drug_gene': np.stack([dg_i, dg_j], axis=1),
    }
    np.savez(OUTPUT_DIR / "edge_lists.npz", **edges)
    print(f"  ✅ 已保存 edge_lists.npz")

    # 保存关联矩阵
    np.save(OUTPUT_DIR / "association_matrix.npy", assoc)
    np.save(OUTPUT_DIR / "mirna_gene_matrix.npy", mg)
    np.save(OUTPUT_DIR / "drug_gene_matrix.npy", dg)
    print(f"  ✅ 已保存 association_matrix.npy")

    # 保存名称映射
    mirna_map = pd.DataFrame({
        'idx': range(n_mirna),
        'mirna_name': parsed['mirna_names'],
        'mirna_id': parsed['mirna_ids'],
        'sequence': parsed['mirna_sequences']
    })
    mirna_map.to_csv(OUTPUT_DIR / "mirna_mapping.csv", index=False)

    drug_map = pd.DataFrame({
        'idx': range(n_drug),
        'drug_name': parsed['drug_names'],
        'smiles': parsed['smiles_list']
    })
    drug_map.to_csv(OUTPUT_DIR / "drug_mapping.csv", index=False)

    gene_map = pd.DataFrame({
        'idx': range(n_gene),
        'gene_name': parsed['gene_names']
    })
    gene_map.to_csv(OUTPUT_DIR / "gene_mapping.csv", index=False)
    print(f"  ✅ 已保存 mirna_mapping.csv, drug_mapping.csv, gene_mapping.csv")

    return edges


# ============================================================
# Step 7: 生成5折CV数据划分
# ============================================================
def step7_cv_splits(parsed, n_folds=5, seed=42):
    print("\n" + "=" * 60)
    print(f"📊 Step 7: 生成 {n_folds} 折交叉验证划分")
    print("=" * 60)

    assoc = parsed['assoc_matrix']
    np.random.seed(seed)

    # 正样本
    pos_i, pos_j = np.where(assoc > 0)
    n_pos = len(pos_i)
    print(f"  正样本: {n_pos}")

    # 负采样：等量随机负样本
    neg_i_list, neg_j_list = [], []
    neg_set = set()
    pos_set = set(zip(pos_i.tolist(), pos_j.tolist()))

    while len(neg_set) < n_pos:
        ri = np.random.randint(0, assoc.shape[0])
        rj = np.random.randint(0, assoc.shape[1])
        if (ri, rj) not in pos_set and (ri, rj) not in neg_set:
            neg_set.add((ri, rj))

    neg_pairs = np.array(list(neg_set))
    neg_i, neg_j = neg_pairs[:, 0], neg_pairs[:, 1]
    print(f"  负样本: {len(neg_i)}")

    # 合并
    all_i = np.concatenate([pos_i, neg_i])
    all_j = np.concatenate([pos_j, neg_j])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])

    # 打乱
    perm = np.random.permutation(len(all_labels))
    all_i = all_i[perm]
    all_j = all_j[perm]
    all_labels = all_labels[perm]

    # K 折划分
    fold_size = len(all_labels) // n_folds
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else len(all_labels)
        test_mask = np.zeros(len(all_labels), dtype=bool)
        test_mask[start:end] = True
        train_mask = ~test_mask

        folds.append({
            'train_i': all_i[train_mask],
            'train_j': all_j[train_mask],
            'train_labels': all_labels[train_mask],
            'test_i': all_i[test_mask],
            'test_j': all_j[test_mask],
            'test_labels': all_labels[test_mask],
        })
        print(f"  Fold {k + 1}: train={train_mask.sum()}, test={test_mask.sum()}")

    # 保存
    np.savez(OUTPUT_DIR / "cv_splits.npz",
             all_mirna_idx=all_i,
             all_drug_idx=all_j,
             all_labels=all_labels,
             n_folds=n_folds,
             fold_size=fold_size)

    print(f"  ✅ 已保存 cv_splits.npz")

    return folds


# ============================================================
# Step 8: 输出统计报告
# ============================================================
def step8_report(parsed, drug_fps, drug_sim, mirna_feats, mirna_sim, bridge_genes):
    print("\n" + "=" * 60)
    print("📊 Step 8: 最终统计报告")
    print("=" * 60)

    assoc = parsed['assoc_matrix']
    n_mirna, n_drug = assoc.shape
    n_gene = parsed['mirna_gene_matrix'].shape[1]

    report = []
    report.append("=" * 60)
    report.append("DrugMiR Preprocessed Data Statistics")
    report.append("=" * 60)
    report.append("")
    report.append("【节点】")
    report.append(f"  miRNAs:  {n_mirna}")
    report.append(f"  Drugs:   {n_drug}")
    report.append(f"  Genes:   {n_gene}")
    report.append(f"  Total:   {n_mirna + n_drug + n_gene}")
    report.append("")
    report.append("【关联】")
    report.append(f"  miRNA-Drug 正关联: {int(assoc.sum())}")
    report.append(f"  关联密度: {assoc.sum() / (n_mirna * n_drug) * 100:.2f}%")
    report.append("")
    report.append("【特征】")
    if drug_fps is not None:
        report.append(f"  Drug Morgan FP: {drug_fps.shape}")
    if mirna_feats is not None:
        report.append(f"  miRNA k-mer:    {mirna_feats.shape}")
    report.append("")
    report.append("【相似度边】")
    if drug_sim is not None:
        n = np.sum(drug_sim > DRUG_SIM_THRESHOLD) - n_drug
        report.append(f"  Drug-Drug (>{DRUG_SIM_THRESHOLD}):   {n // 2}")
    if mirna_sim is not None:
        n = np.sum(mirna_sim > MIRNA_SIM_THRESHOLD) - n_mirna
        report.append(f"  miRNA-miRNA (>{MIRNA_SIM_THRESHOLD}): {n // 2}")
    report.append("")
    report.append("【基因桥接网络】")
    report.append(f"  miRNA-Gene 边: {int(parsed['mirna_gene_matrix'].sum())}")
    report.append(f"  Drug-Gene 边:  {int(parsed['drug_gene_matrix'].sum())}")
    report.append(f"  桥接基因:      {len(bridge_genes)}")
    report.append("")
    report.append("【数据源】")
    report.append(f"  关联数据: ncRNADrug (via MGCNA)")
    report.append(f"  Drug SMILES: DrugBank (via MGCNA)")
    report.append(f"  miRNA 序列: miRBase (via MGCNA)")
    report.append(f"  miRNA-Gene: miRTarBase (via MGCNA)")
    report.append(f"  Drug-Gene: DrugBank (via MGCNA)")

    text = "\n".join(report)
    print(text)

    with open(OUTPUT_DIR / "data_statistics.txt", 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"\n✅ 已保存 data_statistics.txt")

    # 列出所有输出文件
    print(f"\n📁 输出文件列表 ({OUTPUT_DIR}):")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name:40s} {size_str}")


# ============================================================
# Main
# ============================================================
def main():
    print("🚀 DrugMiR 数据预处理开始")
    print(f"   数据来源: {MGCNA_DATA}")
    print(f"   输出目录: {OUTPUT_DIR}")

    # Step 1: 加载
    data = step1_load_data()

    # Step 2: 解析
    parsed = step2_parse_entities(data)

    # Step 3: 药物特征
    drug_fps, drug_sim = step3_drug_features(parsed)

    # Step 4: miRNA 特征
    mirna_feats, mirna_sim = step4_mirna_features(parsed)

    # Step 5: 基因桥接
    bridge_genes = step5_gene_bridge(parsed)

    # Step 6: 边列表
    step6_build_edges(parsed, drug_sim, mirna_sim)

    # Step 7: CV 划分
    step7_cv_splits(parsed)

    # Step 8: 报告
    step8_report(parsed, drug_fps, drug_sim, mirna_feats, mirna_sim, bridge_genes)

    print("\n" + "=" * 60)
    print("🎉 预处理全部完成！可以开始建模了。")
    print("=" * 60)


if __name__ == "__main__":
    main()
