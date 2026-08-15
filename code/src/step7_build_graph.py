"""
Step 7: 构建异质图 —— 最终的 PyG HeteroData 对象
===================================================

图结构:
  节点类型: miRNA, drug, gene
  边类型:
    1. (miRNA, associated_with, drug)    ← ncRNADrug 关联（训练标签）
    2. (miRNA, similar_to, miRNA)        ← 序列相似度 > threshold
    3. (drug, similar_to, drug)          ← 分子指纹相似度 > threshold
    4. (miRNA, targets, gene)            ← miRTarBase 靶基因
    5. (drug, targets, gene)             ← DrugBank 靶基因

关键设计:
  - 关联边 (1) 用于训练/测试划分
  - 相似度边 (2,3) 提供同类信息传播
  - 靶基因边 (4,5) 实现 miRNA→Gene←Drug 的桥接

输出:
  - data/processed/hetero_graph.pt        (PyG HeteroData)
  - data/processed/association_edges.csv  (带标签的关联边，用于划分)
  - data/processed/graph_statistics.txt   (图统计信息)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_project_root, load_processed, normalize_gene_symbol


def build_graph():
    config = load_config()
    root = get_project_root()
    out_dir = root / config["paths"]["processed_data"]
    
    # ============================================================
    # 1. 加载数据
    # ============================================================
    print("📂 Loading processed data...")
    
    associations = load_processed("mirna_drug_associations.csv", config)
    node_mapping = load_processed("node_id_mapping.csv", config)
    
    # 分离 miRNA 和 drug 的映射
    mirna_map = node_mapping[node_mapping['node_type'] == 'miRNA']
    drug_map = node_mapping[node_mapping['node_type'] == 'drug']
    
    mirna_to_id = dict(zip(mirna_map['node_name'], mirna_map['node_id']))
    drug_to_id = dict(zip(drug_map['node_name'], drug_map['node_id']))
    
    n_mirnas = len(mirna_to_id)
    n_drugs = len(drug_to_id)
    
    print(f"   miRNAs: {n_mirnas}")
    print(f"   Drugs:  {n_drugs}")
    
    # 尝试加载靶基因数据
    try:
        mirna_targets = load_processed("mirna_gene_targets.csv", config)
        print(f"   miRNA-Gene pairs: {len(mirna_targets)}")
    except FileNotFoundError:
        mirna_targets = None
        print("   ⚠️  No miRNA-gene targets (miRTarBase)")
    
    try:
        drug_targets = load_processed("drug_targets.csv", config)
        print(f"   Drug-Gene pairs: {len(drug_targets)}")
    except FileNotFoundError:
        drug_targets = None
        print("   ⚠️  No drug-gene targets (DrugBank)")
    
    # 加载特征矩阵
    drug_features = None
    mirna_features = None
    drug_sim = None
    mirna_sim = None
    
    for fname, var_name in [("drug_features.npy", "drug_features"),
                             ("mirna_features.npy", "mirna_features"),
                             ("drug_similarity.npy", "drug_sim"),
                             ("mirna_similarity.npy", "mirna_sim")]:
        fpath = out_dir / fname
        if fpath.exists():
            data = np.load(fpath)
            locals()[var_name] = data  # 不太好用，下面直接重新加载
            print(f"   {fname}: {data.shape}")
    
    # 重新明确加载
    if (out_dir / "drug_features.npy").exists():
        drug_features = np.load(out_dir / "drug_features.npy")
    if (out_dir / "mirna_features.npy").exists():
        mirna_features = np.load(out_dir / "mirna_features.npy")
    if (out_dir / "drug_similarity.npy").exists():
        drug_sim = np.load(out_dir / "drug_similarity.npy")
    if (out_dir / "mirna_similarity.npy").exists():
        mirna_sim = np.load(out_dir / "mirna_similarity.npy")
    
    # ============================================================
    # 2. 构建关联边（训练标签）
    # ============================================================
    print(f"\n{'='*60}")
    print("🔗 Building association edges (labels)")
    print(f"{'='*60}")
    
    assoc_edges = []
    label_map = {'resistance': 0, 'sensitivity': 1}
    
    for _, row in associations.iterrows():
        m = row['mirna_name']
        d = row['drug_name']
        
        if m in mirna_to_id and d in drug_to_id:
            assoc_type = row.get('association_type', None)
            label = label_map.get(assoc_type, -1)  # -1 for unknown
            assoc_edges.append({
                'mirna_id': mirna_to_id[m],
                'drug_id': drug_to_id[d],
                'mirna_name': m,
                'drug_name': d,
                'association_type': assoc_type,
                'label': label
            })
    
    assoc_df = pd.DataFrame(assoc_edges)
    print(f"   Total association edges: {len(assoc_df)}")
    if 'association_type' in assoc_df.columns:
        print(f"   Resistance: {(assoc_df['label'] == 0).sum()}")
        print(f"   Sensitivity: {(assoc_df['label'] == 1).sum()}")
    
    assoc_df.to_csv(out_dir / "association_edges.csv", index=False)
    
    # ============================================================
    # 3. 构建相似度边
    # ============================================================
    print(f"\n{'='*60}")
    print("🔗 Building similarity edges")
    print(f"{'='*60}")
    
    drug_sim_threshold = config.get("features", {}).get("drug_sim_threshold", 0.3)
    mirna_sim_threshold = config.get("features", {}).get("mirna_sim_threshold", 0.5)
    
    drug_sim_edges = []
    if drug_sim is not None:
        for i in range(n_drugs):
            for j in range(i+1, n_drugs):
                if drug_sim[i, j] > drug_sim_threshold:
                    drug_sim_edges.append((i, j))
                    drug_sim_edges.append((j, i))  # 无向图
        print(f"   Drug-Drug similarity edges: {len(drug_sim_edges)//2} (undirected)")
    
    mirna_sim_edges = []
    if mirna_sim is not None:
        for i in range(n_mirnas):
            for j in range(i+1, n_mirnas):
                if mirna_sim[i, j] > mirna_sim_threshold:
                    mirna_sim_edges.append((i, j))
                    mirna_sim_edges.append((j, i))
        print(f"   miRNA-miRNA similarity edges: {len(mirna_sim_edges)//2} (undirected)")
    
    # ============================================================
    # 4. 构建靶基因桥接边
    # ============================================================
    print(f"\n{'='*60}")
    print("🔗 Building gene bridge edges")
    print(f"{'='*60}")
    
    # 收集所有涉及的基因
    all_genes = set()
    
    mirna_gene_edges_raw = []
    if mirna_targets is not None:
        for _, row in mirna_targets.iterrows():
            m = row.get('mirna_name', '')
            g = row.get('gene_symbol', '')
            if m in mirna_to_id and pd.notna(g):
                all_genes.add(g)
                mirna_gene_edges_raw.append((m, g))
    
    drug_gene_edges_raw = []
    if drug_targets is not None:
        for _, row in drug_targets.iterrows():
            d = row.get('drug_name', '')
            if 'drug_name_normalized' in drug_targets.columns:
                d = row.get('drug_name_normalized', d)
            d = str(d).lower().strip()
            g = row.get('gene_symbol', '')
            if d in drug_to_id and pd.notna(g):
                all_genes.add(g)
                drug_gene_edges_raw.append((d, g))
    
    # 创建基因 ID 映射
    gene_list = sorted(all_genes)
    gene_to_id = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)
    
    print(f"   Total genes (bridge nodes): {n_genes}")
    
    # 构建边
    mirna_gene_edges = []
    for m, g in mirna_gene_edges_raw:
        if g in gene_to_id:
            mirna_gene_edges.append((mirna_to_id[m], gene_to_id[g]))
    
    drug_gene_edges = []
    for d, g in drug_gene_edges_raw:
        if g in gene_to_id:
            drug_gene_edges.append((drug_to_id[d], gene_to_id[g]))
    
    print(f"   miRNA → Gene edges: {len(mirna_gene_edges)}")
    print(f"   Drug → Gene edges:  {len(drug_gene_edges)}")
    
    # 检查桥接效果：有多少基因同时被 miRNA 和 drug 靶向？
    mirna_target_genes = set(g for _, g in mirna_gene_edges_raw if g in gene_to_id)
    drug_target_genes = set(g for _, g in drug_gene_edges_raw if g in gene_to_id)
    bridge_genes = mirna_target_genes & drug_target_genes
    print(f"\n   🌉 Bridge genes (targeted by BOTH miRNA and drug): {len(bridge_genes)}")
    print(f"   This is the key for cross-network message passing!")
    
    # ============================================================
    # 5. 保存为 PyG HeteroData（或 numpy 格式）
    # ============================================================
    print(f"\n{'='*60}")
    print("💾 Saving graph data")
    print(f"{'='*60}")
    
    try:
        import torch
        from torch_geometric.data import HeteroData
        
        data = HeteroData()
        
        # 节点特征
        if mirna_features is not None:
            data['miRNA'].x = torch.FloatTensor(mirna_features)
        else:
            data['miRNA'].x = torch.eye(n_mirnas)  # one-hot fallback
        data['miRNA'].num_nodes = n_mirnas
        
        if drug_features is not None:
            data['drug'].x = torch.FloatTensor(drug_features)
        else:
            data['drug'].x = torch.eye(n_drugs)
        data['drug'].num_nodes = n_drugs
        
        # 基因节点：用 one-hot 或零特征
        data['gene'].x = torch.eye(n_genes) if n_genes <= 2000 else torch.zeros(n_genes, 128)
        data['gene'].num_nodes = n_genes
        
        # 边
        if assoc_df is not None and len(assoc_df) > 0:
            src = torch.LongTensor(assoc_df['mirna_id'].values)
            dst = torch.LongTensor(assoc_df['drug_id'].values)
            data['miRNA', 'associated_with', 'drug'].edge_index = torch.stack([src, dst])
            data['drug', 'associated_with', 'miRNA'].edge_index = torch.stack([dst, src])
        
        if drug_sim_edges:
            edges = torch.LongTensor(drug_sim_edges).t()
            data['drug', 'similar_to', 'drug'].edge_index = edges
        
        if mirna_sim_edges:
            edges = torch.LongTensor(mirna_sim_edges).t()
            data['miRNA', 'similar_to', 'miRNA'].edge_index = edges
        
        if mirna_gene_edges:
            edges = torch.LongTensor(mirna_gene_edges).t()
            data['miRNA', 'targets', 'gene'].edge_index = edges
            data['gene', 'targeted_by', 'miRNA'].edge_index = edges.flip(0)
        
        if drug_gene_edges:
            edges = torch.LongTensor(drug_gene_edges).t()
            data['drug', 'targets', 'gene'].edge_index = edges
            data['gene', 'targeted_by', 'drug'].edge_index = edges.flip(0)
        
        torch.save(data, out_dir / "hetero_graph.pt")
        print(f"✅ PyG HeteroData saved: {out_dir / 'hetero_graph.pt'}")
        print(f"\n   Graph summary:")
        print(data)
        
    except ImportError:
        print("⚠️  PyTorch/PyG not installed. Saving as numpy format instead.")
        print("   Install on GPU server: pip install torch torch-geometric")
        
        # 保存为 numpy 格式（GPU 服务器上再转换）
        graph_data = {
            'n_mirnas': n_mirnas,
            'n_drugs': n_drugs,
            'n_genes': n_genes,
            'association_edges': np.array([(r['mirna_id'], r['drug_id']) 
                                            for _, r in assoc_df.iterrows()]),
            'association_labels': np.array(assoc_df['label'].values),
            'drug_sim_edges': np.array(drug_sim_edges) if drug_sim_edges else np.array([]).reshape(0, 2),
            'mirna_sim_edges': np.array(mirna_sim_edges) if mirna_sim_edges else np.array([]).reshape(0, 2),
            'mirna_gene_edges': np.array(mirna_gene_edges) if mirna_gene_edges else np.array([]).reshape(0, 2),
            'drug_gene_edges': np.array(drug_gene_edges) if drug_gene_edges else np.array([]).reshape(0, 2),
            'gene_list': gene_list,
        }
        np.savez(out_dir / "graph_data.npz", **{k: v for k, v in graph_data.items() 
                                                   if isinstance(v, np.ndarray)})
        
        # 基因映射
        gene_df = pd.DataFrame({'gene_symbol': gene_list, 'gene_id': range(n_genes)})
        gene_df.to_csv(out_dir / "gene_id_mapping.csv", index=False)
    
    # ============================================================
    # 6. 统计报告
    # ============================================================
    stats = []
    stats.append("=" * 60)
    stats.append("DrugMiR Graph Statistics")
    stats.append("=" * 60)
    stats.append(f"")
    stats.append(f"Nodes:")
    stats.append(f"  miRNAs: {n_mirnas}")
    stats.append(f"  Drugs:  {n_drugs}")
    stats.append(f"  Genes:  {n_genes}")
    stats.append(f"  Total:  {n_mirnas + n_drugs + n_genes}")
    stats.append(f"")
    stats.append(f"Edges:")
    stats.append(f"  miRNA-Drug associations: {len(assoc_df)}")
    stats.append(f"  Drug-Drug similarity:    {len(drug_sim_edges)//2}")
    stats.append(f"  miRNA-miRNA similarity:  {len(mirna_sim_edges)//2}")
    stats.append(f"  miRNA-Gene targets:      {len(mirna_gene_edges)}")
    stats.append(f"  Drug-Gene targets:       {len(drug_gene_edges)}")
    total_edges = len(assoc_df) + len(drug_sim_edges)//2 + len(mirna_sim_edges)//2 + len(mirna_gene_edges) + len(drug_gene_edges)
    stats.append(f"  Total:                   {total_edges}")
    stats.append(f"")
    stats.append(f"Bridge genes: {len(bridge_genes)}")
    stats.append(f"")
    stats.append(f"Features:")
    if drug_features is not None:
        stats.append(f"  Drug features: {drug_features.shape}")
    if mirna_features is not None:
        stats.append(f"  miRNA features: {mirna_features.shape}")
    
    report = '\n'.join(stats)
    print(f"\n{report}")
    
    with open(out_dir / "graph_statistics.txt", 'w') as f:
        f.write(report)
    
    print(f"\n✅ All graph data saved to {out_dir}/")


if __name__ == "__main__":
    build_graph()
