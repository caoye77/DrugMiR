"""
DrugMiR Multi-Task: 同时预测 Resistance 和 Sensitivity
========================================================
Task 1: P(resistance) for each miRNA-drug pair
Task 2: P(sensitivity) for each miRNA-drug pair

这是全领域第一个区分 resistance/sensitivity 的预测模型。

运行: cd ~/work/DrugMiR && python src/train_multitask.py --gpu 0
"""
import sys, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

sys.path.append(str(Path(__file__).parent))
from drugmir_v2 import FeatureEncoder, NormalizedGeneBridge, HomoGCNBlock
try:
    from fair_comparison import build_knn_edges
except:
    def build_knn_edges(sim_matrix, k=10):
        n = sim_matrix.shape[0]
        edges_src, edges_dst = [], []
        for i in range(n):
            sims = sim_matrix[i].copy()
            sims[i] = -1
            topk = np.argsort(-sims)[:k]
            for j in topk:
                if sim_matrix[i, j] > 0:
                    edges_src.append(i)
                    edges_dst.append(j)
        return np.array([edges_src, edges_dst])


class DrugMiRMultiTask(nn.Module):
    """
    Multi-task version: two prediction heads
    - Head 1: P(resistance)
    - Head 2: P(sensitivity)
    """
    def __init__(self, mirna_feat_dim, drug_feat_dim, n_genes,
                 hidden_dim=128, n_homo_layers=2, n_bridge_layers=2, dropout=0.35):
        super().__init__()
        self.mirna_encoder = FeatureEncoder(mirna_feat_dim, hidden_dim, dropout)
        self.drug_encoder = FeatureEncoder(drug_feat_dim, hidden_dim, dropout)
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)

        self.mirna_gcn_layers = nn.ModuleList([
            HomoGCNBlock(hidden_dim, dropout) for _ in range(n_homo_layers)
        ])
        self.drug_gcn_layers = nn.ModuleList([
            HomoGCNBlock(hidden_dim, dropout) for _ in range(n_homo_layers)
        ])
        self.bridge_layers = nn.ModuleList([
            NormalizedGeneBridge(hidden_dim, dropout) for _ in range(n_bridge_layers)
        ])

        pred_in_dim = hidden_dim * 3 * 2  # 3 channels x (mirna + drug)

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        self.head_resistance = nn.Linear(hidden_dim, 1)
        self.head_sensitivity = nn.Linear(hidden_dim, 1)

    def encode(self, mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
               mirna_gene_edge, drug_gene_edge):
        mirna_h0 = self.mirna_encoder(mirna_feat)
        drug_h0 = self.drug_encoder(drug_feat)
        gene_h = self.gene_embedding.weight

        mirna_orig, drug_orig = mirna_h0, drug_h0

        mirna_homo = mirna_h0
        for layer in self.mirna_gcn_layers:
            mirna_homo = layer(mirna_homo, mirna_sim_edge)
        drug_homo = drug_h0
        for layer in self.drug_gcn_layers:
            drug_homo = layer(drug_homo, drug_sim_edge)

        mirna_bridge, drug_bridge, gene_bridge = mirna_h0, drug_h0, gene_h
        for layer in self.bridge_layers:
            mirna_bridge, drug_bridge, gene_bridge = layer(
                mirna_bridge, drug_bridge, gene_bridge,
                mirna_gene_edge, drug_gene_edge)

        return (mirna_orig, mirna_homo, mirna_bridge,
                drug_orig, drug_homo, drug_bridge)

    def predict(self, encoded, pred_mirna_idx, pred_drug_idx):
        mirna_orig, mirna_homo, mirna_bridge, drug_orig, drug_homo, drug_bridge = encoded

        m = torch.cat([
            mirna_orig[pred_mirna_idx],
            mirna_homo[pred_mirna_idx],
            mirna_bridge[pred_mirna_idx],
        ], dim=-1)
        d = torch.cat([
            drug_orig[pred_drug_idx],
            drug_homo[pred_drug_idx],
            drug_bridge[pred_drug_idx],
        ], dim=-1)

        pair = torch.cat([m, d], dim=-1)
        shared_feat = self.shared(pair)

        res_score = self.head_resistance(shared_feat).squeeze(-1)
        sen_score = self.head_sensitivity(shared_feat).squeeze(-1)

        return res_score, sen_score

    def forward(self, mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                mirna_gene_edge, drug_gene_edge, pred_mirna_idx, pred_drug_idx):
        encoded = self.encode(mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                              mirna_gene_edge, drug_gene_edge)
        return self.predict(encoded, pred_mirna_idx, pred_drug_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--batch_size', type=int, default=2048)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    mirna_feat = torch.FloatTensor(np.load("data/processed/mirna_kmer_features.npy")).to(device)
    drug_feat = torch.FloatTensor(np.load("data/processed/drug_morgan_features.npy")).to(device)
    res_matrix = np.load("data/processed/resistance_matrix.npy")
    sen_matrix = np.load("data/processed/sensitivity_matrix.npy")
    mirna_sim = np.load("data/processed/mirna_similarity.npy")
    drug_sim = np.load("data/processed/drug_similarity.npy")
    edges = np.load("data/processed/edge_lists.npz")

    mirna_gene_edge = torch.LongTensor(edges['mirna_gene'].T).to(device)
    drug_gene_edge = torch.LongTensor(edges['drug_gene'].T).to(device)

    # KNN edges
    mirna_knn = build_knn_edges(mirna_sim, k=15)
    drug_knn = build_knn_edges(drug_sim, k=10)
    mm_orig = edges['mirna_mirna_sim']
    dd_orig = edges['drug_drug_sim']
    mm_all = np.hstack([mirna_knn, mm_orig.T, mm_orig[:, [1, 0]].T]) if len(mm_orig) > 0 else mirna_knn
    dd_all = np.hstack([drug_knn, dd_orig.T, dd_orig[:, [1, 0]].T]) if len(dd_orig) > 0 else drug_knn
    mirna_sim_edge = torch.LongTensor(mm_all).to(device)
    drug_sim_edge = torch.LongTensor(dd_all).to(device)

    n_mirna, n_drug = res_matrix.shape
    n_gene = 14455

    print(f"miRNAs: {n_mirna}, Drugs: {n_drug}")
    print(f"Resistance pairs: {res_matrix.sum()}")
    print(f"Sensitivity pairs: {sen_matrix.sum()}")

    # Build samples: all positions where res OR sen > 0 are positive for that task
    # Negative: random pairs where both res and sen are 0
    pos_res_i, pos_res_j = np.where(res_matrix > 0)
    pos_sen_i, pos_sen_j = np.where(sen_matrix > 0)

    # Union of all positive pairs
    all_pos = set()
    for m, d in zip(pos_res_i, pos_res_j):
        all_pos.add((m, d))
    for m, d in zip(pos_sen_i, pos_sen_j):
        all_pos.add((m, d))

    all_pos_list = list(all_pos)
    n_pos = len(all_pos_list)
    print(f"Total unique positive pairs: {n_pos}")

    # Sample negatives
    np.random.seed(42)
    neg_pairs = []
    while len(neg_pairs) < n_pos:
        m = np.random.randint(n_mirna)
        d = np.random.randint(n_drug)
        if (m, d) not in all_pos:
            neg_pairs.append((m, d))
            all_pos.add((m, d))  # prevent duplicates
    neg_pairs = np.array(neg_pairs)

    # Build full dataset
    all_m = np.array([p[0] for p in all_pos_list] + neg_pairs[:, 0].tolist())
    all_d = np.array([p[1] for p in all_pos_list] + neg_pairs[:, 1].tolist())
    all_res_y = np.array([res_matrix[m, d] for m, d in zip(all_m[:n_pos], all_d[:n_pos])] + [0] * n_pos)
    all_sen_y = np.array([sen_matrix[m, d] for m, d in zip(all_m[:n_pos], all_d[:n_pos])] + [0] * n_pos)

    print(f"Total samples: {len(all_m)} (pos: {n_pos}, neg: {n_pos})")
    print(f"Resistance labels: {all_res_y.sum()} pos, {(all_res_y == 0).sum()} neg")
    print(f"Sensitivity labels: {all_sen_y.sum()} pos, {(all_sen_y == 0).sum()} neg")

    # 5-fold CV
    seeds = [42, 123, 2024]
    all_results = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        perm = np.random.permutation(len(all_m))
        all_m_s = all_m[perm]
        all_d_s = all_d[perm]
        all_res_s = all_res_y[perm]
        all_sen_s = all_sen_y[perm]

        fold_size = len(all_m) // 5

        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size if fold < 4 else len(all_m)
            test_mask = np.zeros(len(all_m), dtype=bool)
            test_mask[start:end] = True
            train_mask = ~test_mask

            tr_m = torch.LongTensor(all_m_s[train_mask]).to(device)
            tr_d = torch.LongTensor(all_d_s[train_mask]).to(device)
            tr_res = torch.FloatTensor(all_res_s[train_mask]).to(device)
            tr_sen = torch.FloatTensor(all_sen_s[train_mask]).to(device)
            te_m = torch.LongTensor(all_m_s[test_mask]).to(device)
            te_d = torch.LongTensor(all_d_s[test_mask]).to(device)
            te_res = torch.FloatTensor(all_res_s[test_mask]).to(device)
            te_sen = torch.FloatTensor(all_sen_s[test_mask]).to(device)

            model = DrugMiRMultiTask(
                mirna_feat.shape[1], drug_feat.shape[1], n_gene,
                args.hidden, 2, 2, args.dropout
            ).to(device)

            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            crit = nn.BCEWithLogitsLoss()

            best_auc = 0
            best_state = None
            pat = 0

            for ep in range(args.epochs):
                model.train()
                pm = torch.randperm(len(tr_m), device=device)
                for i in range(0, len(tr_m), args.batch_size):
                    end2 = min(i + args.batch_size, len(tr_m))
                    idx = pm[i:end2]
                    opt.zero_grad()
                    r_score, s_score = model(
                        mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                        mirna_gene_edge, drug_gene_edge, tr_m[idx], tr_d[idx]
                    )
                    loss = crit(r_score, tr_res[idx]) + crit(s_score, tr_sen[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sch.step()

                if (ep + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        r_s, s_s = model(
                            mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                            mirna_gene_edge, drug_gene_edge, te_m, te_d
                        )
                        r_auc = roc_auc_score(te_res.cpu(), torch.sigmoid(r_s).cpu())
                        s_auc = roc_auc_score(te_sen.cpu(), torch.sigmoid(s_s).cpu())
                        avg_auc = (r_auc + s_auc) / 2

                    if avg_auc > best_auc:
                        best_auc = avg_auc
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
                r_s, s_s = model(
                    mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                    mirna_gene_edge, drug_gene_edge, te_m, te_d
                )
                r_prob = torch.sigmoid(r_s).cpu().numpy()
                s_prob = torch.sigmoid(s_s).cpu().numpy()
                r_y = te_res.cpu().numpy()
                s_y = te_sen.cpu().numpy()

            metrics = {
                'Res_AUC': roc_auc_score(r_y, r_prob),
                'Res_AUPR': average_precision_score(r_y, r_prob),
                'Sen_AUC': roc_auc_score(s_y, s_prob),
                'Sen_AUPR': average_precision_score(s_y, s_prob),
                'Avg_AUC': (roc_auc_score(r_y, r_prob) + roc_auc_score(s_y, s_prob)) / 2,
            }
            all_results.append(metrics)

            if seed == 42 and fold == 0:
                print(f"\n  Fold 1 preview: Res_AUC={metrics['Res_AUC']:.4f}, Sen_AUC={metrics['Sen_AUC']:.4f}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Multi-Task Results (3 seeds x 5 folds = {len(all_results)} runs)")
    print(f"{'=' * 60}")
    for key in ['Res_AUC', 'Res_AUPR', 'Sen_AUC', 'Sen_AUPR', 'Avg_AUC']:
        vals = [r[key] for r in all_results]
        print(f"  {key:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Save
    save_r = {}
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        save_r[key] = [float(np.mean(vals)), float(np.std(vals))]
    with open('results/multitask_results.json', 'w') as f:
        json.dump(save_r, f, indent=2)
    print(f"\nSaved to results/multitask_results.json")


if __name__ == "__main__":
    main()
