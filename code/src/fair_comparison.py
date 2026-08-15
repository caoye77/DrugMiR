"""
公平对比: MGCNA vs DrugMiR v2 (KNN增强版)
==========================================
1. 用 MGCNA 的数据生成其所需的 view 矩阵
2. 用统一的 5折CV 跑 MGCNA
3. 用 KNN 边增强的 DrugMiR v2 跑同一 CV
4. 对比结果

运行: cd ~/work/DrugMiR && python src/fair_comparison.py --gpu 0
"""
import sys, os, time, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from torch_geometric.nn import GCNConv
from scipy import stats

sys.path.append(str(Path(__file__).parent))

# ============================================================
# Part 0: 生成 MGCNA 需要的 view 矩阵
# ============================================================
def generate_mgcna_views():
    """从原始 Excel 生成 MGCNA 的多视图相似度矩阵"""
    print("生成 MGCNA view 矩阵...")
    base = Path("data/raw/MGCNA/data/data")

    # miRNA 序列相似度 (3-mer cosine, threshold 0.8)
    mirna_seq = pd.read_excel(base / "miRNA-sequences.xlsx")
    seqs = mirna_seq.iloc[:, 1].tolist()  # Sequence column

    def kmer_feat(seq, k_list=[1, 2, 3]):
        bases = "ATCG"
        seq = str(seq).upper().replace("U", "T")
        feat = []
        for k in k_list:
            from itertools import product as iprod
            kmers = [''.join(p) for p in iprod(bases, repeat=k)]
            kmer_dict = {km: 0 for km in kmers}
            for i in range(len(seq) - k + 1):
                sub = seq[i:i+k]
                if sub in kmer_dict:
                    kmer_dict[sub] += 1
            total = max(sum(kmer_dict.values()), 1)
            feat.extend([v / total for v in kmer_dict.values()])
        return feat

    mirna_feats = np.array([kmer_feat(s) for s in seqs])
    mm_seq_sim = sk_cosine(mirna_feats)
    mm_seq_sim_binary = (mm_seq_sim > 0.8).astype(float)
    np.fill_diagonal(mm_seq_sim_binary, 1)

    # Drug SMILES 相似度 (MACCS Tanimoto, threshold 0.5)
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys
    from rdkit import DataStructs
    RDLogger.logger().setLevel(RDLogger.ERROR)

    drug_df = pd.read_excel(base / "drug-smiles.xlsx")
    smiles = drug_df.iloc[:, 1].tolist()  # smiles column
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            fps.append(MACCSkeys.GenMACCSKeys(mol))
        else:
            fps.append(None)

    n_drug = len(fps)
    dd_sim = np.zeros((n_drug, n_drug))
    for i in range(n_drug):
        for j in range(i, n_drug):
            if fps[i] and fps[j]:
                s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            else:
                s = 0
            dd_sim[i, j] = s
            dd_sim[j, i] = s
    dd_sim_binary = (dd_sim > 0.5).astype(float)
    np.fill_diagonal(dd_sim_binary, 1)

    # Gaussian kernel similarities
    def gauss_kernel_sim(A, threshold):
        n = A.shape[0]
        ip_sum = sum(np.linalg.norm(A[i]) ** 2 for i in range(n))
        lam = 1 / ((1 / n) * ip_sum) if ip_sum > 0 else 1
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim[i, j] = np.exp(-lam * np.linalg.norm(A[i] - A[j]) ** 2)
        binary = (sim > threshold).astype(float)
        np.fill_diagonal(binary, 1)
        return binary

    # Drug-gene Gaussian sim
    dg = pd.read_excel(base / "drug-gene-matrix.xlsx", header=0, index_col=0)
    dd_gene_sim = gauss_kernel_sim(dg.values, 0.5)

    # miRNA-drug Gaussian sim (for miRNA view and drug view)
    md = pd.read_excel(base / "miRNA-drug-matrix.xlsx", header=0, index_col=0)
    mm_drug_sim = gauss_kernel_sim(md.values, 0.6)
    dd_mirna_sim = gauss_kernel_sim(md.values.T, 0.5)

    views = {
        'mm_s': mm_seq_sim_binary,
        'mm_r': mm_drug_sim,
        'dd_f': dd_sim_binary,
        'dd_g': dd_gene_sim,
        'dd_m': dd_mirna_sim,
    }

    for name, mat in views.items():
        n_edges = (mat > 0).sum() - mat.shape[0]
        print(f"  {name}: {mat.shape}, edges={n_edges}")

    return views


# ============================================================
# Part 1: MGCNA 模型 (复现)
# ============================================================
class MGCNAAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class MGCNAModel(nn.Module):
    def __init__(self, n_mirna=1578, n_drug=156, feat_dim=512, h1=256, h2=128, dec1=256):
        super().__init__()
        self.n_mirna = n_mirna
        self.n_drug = n_drug
        self.feat_dim = feat_dim

        # miRNA encoders (2 views: seq, drug-based)
        self.gcn_m_s1 = GCNConv(feat_dim, h1)
        self.gcn_m_s2 = GCNConv(h1, h2)
        self.gcn_m_r1 = GCNConv(feat_dim, h1)
        self.gcn_m_r2 = GCNConv(h1, h2)

        # Drug encoders (3 views: fingerprint, gene, mirna-based)
        self.gcn_d_f1 = GCNConv(feat_dim, h1)
        self.gcn_d_f2 = GCNConv(h1, h2)
        self.gcn_d_g1 = GCNConv(feat_dim, h1)
        self.gcn_d_g2 = GCNConv(h1, h2)
        self.gcn_d_m1 = GCNConv(feat_dim, h1)
        self.gcn_d_m2 = GCNConv(h1, h2)

        self.attn_m = MGCNAAttention(h2)
        self.attn_d = MGCNAAttention(h2)

        self.decoder1 = nn.Linear(h2 * 4, dec1)
        self.decoder2 = nn.Linear(dec1, 1)

    def forward(self, views, mirna_idx, drug_idx, device):
        torch.manual_seed(1)
        x_m = torch.randn(self.n_mirna, self.feat_dim, device=device)
        x_d = torch.randn(self.n_drug, self.feat_dim, device=device)

        # miRNA views
        m_s = F.relu(self.gcn_m_s2(F.relu(self.gcn_m_s1(x_m, views['mm_s'])), views['mm_s']))
        m_r = F.relu(self.gcn_m_r2(F.relu(self.gcn_m_r1(x_m, views['mm_r'])), views['mm_r']))

        # Drug views
        d_f = F.relu(self.gcn_d_f2(F.relu(self.gcn_d_f1(x_d, views['dd_f'])), views['dd_f']))
        d_g = F.relu(self.gcn_d_g2(F.relu(self.gcn_d_g1(x_d, views['dd_g'])), views['dd_g']))
        d_m = F.relu(self.gcn_d_m2(F.relu(self.gcn_d_m1(x_d, views['dd_m'])), views['dd_m']))

        # Attention fusion
        x_m_fused, _ = self.attn_m(torch.stack([m_s, m_r], dim=1))
        y_d_fused, _ = self.attn_d(torch.stack([d_f, d_g, d_m], dim=1))

        e1 = x_m_fused[mirna_idx]
        e2 = y_d_fused[drug_idx]

        feat = torch.cat([e1 + e2, e1 * e2, e1, e2], dim=1)
        out = self.decoder2(F.relu(self.decoder1(feat)))
        return out.squeeze(-1)


# ============================================================
# Part 2: KNN 边增强
# ============================================================
def build_knn_edges(sim_matrix, k=10):
    """为每个节点保留 top-k 最相似的邻居"""
    n = sim_matrix.shape[0]
    edges_src, edges_dst = [], []
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # 排除自环
        topk = np.argsort(-sims)[:k]
        for j in topk:
            if sim_matrix[i, j] > 0:  # 只保留正相似度
                edges_src.append(i)
                edges_dst.append(j)
    edge_index = np.array([edges_src, edges_dst])
    return edge_index


# ============================================================
# Part 3: 统一评估框架
# ============================================================
def evaluate_model(model_fn, data_dict, seeds=[42, 123, 2024], epochs=150, lr=0.001, wd=2e-4, batch_size=2048):
    """
    model_fn: callable that returns (model, forward_fn)
    统一的多种子 5折CV 评估
    """
    device = data_dict['device']
    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        assoc = data_dict['assoc_matrix']
        n_mirna, n_drug = assoc.shape
        pos_i, pos_j = np.where(assoc > 0)
        n_pos = len(pos_i)

        # 负采样
        pos_set = set(zip(pos_i.tolist(), pos_j.tolist()))
        neg_pairs = []
        while len(neg_pairs) < n_pos:
            m = np.random.randint(n_mirna)
            d = np.random.randint(n_drug)
            if (m, d) not in pos_set and (m, d) not in set(map(tuple, neg_pairs)):
                neg_pairs.append((m, d))
        neg_pairs = np.array(neg_pairs)

        all_m = np.concatenate([pos_i, neg_pairs[:, 0]])
        all_d = np.concatenate([pos_j, neg_pairs[:, 1]])
        all_y = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])

        perm = np.random.permutation(len(all_y))
        all_m, all_d, all_y = all_m[perm], all_d[perm], all_y[perm]

        fold_size = len(all_y) // 5

        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size if fold < 4 else len(all_y)
            test_mask = np.zeros(len(all_y), dtype=bool)
            test_mask[start:end] = True
            train_mask = ~test_mask

            tr_m = torch.LongTensor(all_m[train_mask]).to(device)
            tr_d = torch.LongTensor(all_d[train_mask]).to(device)
            tr_y = torch.FloatTensor(all_y[train_mask]).to(device)
            te_m = torch.LongTensor(all_m[test_mask]).to(device)
            te_d = torch.LongTensor(all_d[test_mask]).to(device)
            te_y = torch.FloatTensor(all_y[test_mask]).to(device)

            model, forward_fn = model_fn()
            model = model.to(device)
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            crit = nn.BCEWithLogitsLoss()

            best_auc = 0
            best_state = None
            pat = 0

            for ep in range(epochs):
                model.train()
                pm = torch.randperm(len(tr_y), device=device)
                for i in range(0, len(tr_y), batch_size):
                    end2 = min(i + batch_size, len(tr_y))
                    idx = pm[i:end2]
                    opt.zero_grad()
                    scores = forward_fn(model, tr_m[idx], tr_d[idx])
                    loss = crit(scores, tr_y[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sch.step()

                if (ep + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        s = forward_fn(model, te_m, te_d)
                        auc = roc_auc_score(te_y.cpu(), torch.sigmoid(s).cpu())
                    if auc > best_auc:
                        best_auc = auc
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        pat = 0
                    else:
                        pat += 1
                        if pat >= 15:
                            break

            if best_state:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            model.eval()
            with torch.no_grad():
                s = forward_fn(model, te_m, te_d)
                probs = torch.sigmoid(s).cpu().numpy()
                y = te_y.cpu().numpy()

            preds = (probs > 0.5).astype(int)
            all_results.append({
                'AUC': roc_auc_score(y, probs),
                'AUPR': average_precision_score(y, probs),
                'F1': f1_score(y, preds),
                'ACC': accuracy_score(y, preds),
            })

    summary = {}
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        summary[key] = (np.mean(vals), np.std(vals))
    return summary, [r['AUC'] for r in all_results]


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load base data
    assoc = np.load("data/processed/association_matrix.npy")
    mirna_feat = torch.FloatTensor(np.load("data/processed/mirna_kmer_features.npy")).to(device)
    drug_feat = torch.FloatTensor(np.load("data/processed/drug_morgan_features.npy")).to(device)
    mirna_sim = np.load("data/processed/mirna_similarity.npy")
    drug_sim = np.load("data/processed/drug_similarity.npy")

    # Load edge data
    edges = np.load("data/processed/edge_lists.npz")
    mg = edges['mirna_gene']
    dg = edges['drug_gene']
    mirna_gene_edge = torch.LongTensor(mg.T).to(device)
    drug_gene_edge = torch.LongTensor(dg.T).to(device)

    n_mirna, n_drug = assoc.shape
    n_gene = 14455

    data_dict = {
        'assoc_matrix': assoc,
        'device': device,
    }

    # ============================================================
    # 1. Generate MGCNA views
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 1: 生成 MGCNA 视图矩阵")
    print("=" * 60)
    mgcna_views_np = generate_mgcna_views()

    # Convert to edge indices on GPU
    mgcna_views = {}
    for name, mat in mgcna_views_np.items():
        ei = mat.nonzero()
        mgcna_views[name] = torch.tensor(np.vstack((ei[0], ei[1])), dtype=torch.long, device=device)

    # ============================================================
    # 2. KNN 边增强 for DrugMiR
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: KNN 边增强")
    print("=" * 60)

    mirna_knn = build_knn_edges(mirna_sim, k=15)
    drug_knn = build_knn_edges(drug_sim, k=10)

    mirna_knn_edge = torch.LongTensor(mirna_knn).to(device)
    drug_knn_edge = torch.LongTensor(drug_knn).to(device)

    # 合并 KNN + 原始阈值边
    mm_orig = edges['mirna_mirna_sim']
    dd_orig = edges['drug_drug_sim']
    if len(mm_orig) > 0:
        mm_all = np.hstack([mirna_knn, mm_orig.T, mm_orig[:, [1, 0]].T])
    else:
        mm_all = mirna_knn
    if len(dd_orig) > 0:
        dd_all = np.hstack([drug_knn, dd_orig.T, dd_orig[:, [1, 0]].T])
    else:
        dd_all = drug_knn

    mirna_sim_edge_knn = torch.LongTensor(mm_all).to(device)
    drug_sim_edge_knn = torch.LongTensor(dd_all).to(device)

    print(f"  miRNA edges (KNN+thresh): {mirna_sim_edge_knn.shape[1]}")
    print(f"  Drug edges (KNN+thresh):  {drug_sim_edge_knn.shape[1]}")

    # ============================================================
    # 3. 定义模型构造函数
    # ============================================================

    # MGCNA
    def make_mgcna():
        model = MGCNAModel(n_mirna, n_drug, 512, 256, 128, 256)

        def fwd(m, mi, di):
            return m(mgcna_views, mi, di, device)

        return model, fwd

    # DrugMiR v2 + KNN
    from drugmir_v2 import DrugMiRv2

    def make_drugmir_knn():
        model = DrugMiRv2(mirna_feat.shape[1], drug_feat.shape[1], n_gene,
                          hidden_dim=128, n_homo_layers=2, n_bridge_layers=2, dropout=0.35)

        def fwd(m, mi, di):
            s, _ = m(mirna_feat, drug_feat, mirna_sim_edge_knn, drug_sim_edge_knn,
                     mirna_gene_edge, drug_gene_edge, mi, di)
            return s

        return model, fwd

    # DrugMiR v2 original
    from train import DrugMiRData
    data_full = DrugMiRData('data/processed', device)

    def make_drugmir_orig():
        model = DrugMiRv2(mirna_feat.shape[1], drug_feat.shape[1], n_gene,
                          hidden_dim=128, n_homo_layers=2, n_bridge_layers=2, dropout=0.35)

        def fwd(m, mi, di):
            s, _ = m(mirna_feat, drug_feat, data_full.mirna_sim_edge, data_full.drug_sim_edge,
                     mirna_gene_edge, drug_gene_edge, mi, di)
            return s

        return model, fwd

    # MLP baseline
    from baselines import MLPBaseline

    def make_mlp():
        model = MLPBaseline(mirna_feat.shape[1], drug_feat.shape[1], 128, 0.35)

        def fwd(m, mi, di):
            s, _ = m(mirna_feat, drug_feat, None, None, None, None, mi, di)
            return s

        return model, fwd

    # ============================================================
    # 4. 运行实验
    # ============================================================
    results = {}
    auc_lists = {}

    for name, make_fn, ep in [
        ("MGCNA (reproduced)", make_mgcna, 60),
        ("DrugMiR+KNN", make_drugmir_knn, 150),
        ("DrugMiR", make_drugmir_orig, 150),
        ("MLP", make_mlp, 150),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")
        t0 = time.time()
        summary, auc_list = evaluate_model(make_fn, data_dict, epochs=ep)
        t1 = time.time()
        results[name] = summary
        auc_lists[name] = auc_list
        print(f"  {name}: AUC={summary['AUC'][0]:.4f}±{summary['AUC'][1]:.4f} "
              f"AUPR={summary['AUPR'][0]:.4f}±{summary['AUPR'][1]:.4f} ({t1 - t0:.0f}s)")

    # ============================================================
    # 5. 统计显著性检验
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Statistical Significance Tests")
    print(f"{'=' * 60}")

    best_name = "DrugMiR+KNN"
    for name in results:
        if name != best_name:
            t, p = stats.ttest_rel(auc_lists[best_name], auc_lists[name])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f"  {best_name} vs {name}: t={t:.3f}, p={p:.6f} {sig}")

    # ============================================================
    # 6. 最终汇总
    # ============================================================
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS (3 seeds × 5 folds = 15 runs)")
    print(f"{'=' * 60}")
    print(f"{'Method':<25s} {'AUC':>16s} {'AUPR':>16s} {'F1':>16s}")
    print("-" * 75)
    for name in ["DrugMiR+KNN", "DrugMiR", "MGCNA (reproduced)", "MLP"]:
        s = results[name]
        print(f"{name:<25s} {s['AUC'][0]:.4f}±{s['AUC'][1]:.4f}"
              f"  {s['AUPR'][0]:.4f}±{s['AUPR'][1]:.4f}"
              f"  {s['F1'][0]:.4f}±{s['F1'][1]:.4f}")

    # Save
    save_r = {k: {kk: [float(vv[0]), float(vv[1])] for kk, vv in v.items()} for k, v in results.items()}
    with open("results/fair_comparison.json", "w") as f:
        json.dump(save_r, f, indent=2)
    print(f"\nResults saved to results/fair_comparison.json")


if __name__ == "__main__":
    main()
