"""
DrugMiR 训练脚本
================
5折交叉验证, 支持 GPU 训练

运行: python src/train.py [--gpu 0] [--epochs 200] [--hidden 128]
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
from drugmir_model import DrugMiR, AdaptiveNegativeSampler


# ============================================================
# 数据加载
# ============================================================
class DrugMiRData:
    """加载预处理好的数据"""

    def __init__(self, data_dir="data/processed", device='cpu'):
        self.device = device
        data_dir = Path(data_dir)

        print("📂 加载数据...")

        # 特征
        self.mirna_feat = torch.FloatTensor(
            np.load(data_dir / "mirna_kmer_features.npy")
        ).to(device)
        self.drug_feat = torch.FloatTensor(
            np.load(data_dir / "drug_morgan_features.npy")
        ).to(device)

        # 关联矩阵
        self.assoc_matrix = np.load(data_dir / "association_matrix.npy")

        # 边列表
        edges = np.load(data_dir / "edge_lists.npz")

        # miRNA 相似度边 (无向: 补充反向边)
        mm = edges['mirna_mirna_sim']
        if len(mm) > 0:
            mm_rev = mm[:, [1, 0]]
            mm_all = np.vstack([mm, mm_rev])
            self.mirna_sim_edge = torch.LongTensor(mm_all.T).to(device)
        else:
            self.mirna_sim_edge = torch.zeros(2, 0, dtype=torch.long, device=device)

        # Drug 相似度边 (无向)
        dd = edges['drug_drug_sim']
        if len(dd) > 0:
            dd_rev = dd[:, [1, 0]]
            dd_all = np.vstack([dd, dd_rev])
            self.drug_sim_edge = torch.LongTensor(dd_all.T).to(device)
        else:
            self.drug_sim_edge = torch.zeros(2, 0, dtype=torch.long, device=device)

        # miRNA-Gene 边
        mg = edges['mirna_gene']
        self.mirna_gene_edge = torch.LongTensor(mg.T).to(device)

        # Drug-Gene 边
        dg = edges['drug_gene']
        self.drug_gene_edge = torch.LongTensor(dg.T).to(device)

        # CV 划分
        cv = np.load(data_dir / "cv_splits.npz")
        self.all_mirna_idx = cv['all_mirna_idx']
        self.all_drug_idx = cv['all_drug_idx']
        self.all_labels = cv['all_labels']
        self.n_folds = int(cv['n_folds'])
        self.fold_size = int(cv['fold_size'])

        # 统计
        self.n_mirna = self.mirna_feat.shape[0]
        self.n_drug = self.drug_feat.shape[0]
        self.n_gene = int(np.load(data_dir / "mirna_gene_matrix.npy").shape[1])

        print(f"  miRNAs: {self.n_mirna}, Drugs: {self.n_drug}, Genes: {self.n_gene}")
        print(f"  miRNA feat: {self.mirna_feat.shape}")
        print(f"  Drug feat:  {self.drug_feat.shape}")
        print(f"  miRNA-miRNA edges: {self.mirna_sim_edge.shape[1]}")
        print(f"  Drug-Drug edges:   {self.drug_sim_edge.shape[1]}")
        print(f"  miRNA-Gene edges:  {self.mirna_gene_edge.shape[1]}")
        print(f"  Drug-Gene edges:   {self.drug_gene_edge.shape[1]}")
        print(f"  Total samples: {len(self.all_labels)}, Folds: {self.n_folds}")

    def get_fold(self, fold_idx):
        """获取第 k 折的训练/测试集"""
        n = len(self.all_labels)
        start = fold_idx * self.fold_size
        end = start + self.fold_size if fold_idx < self.n_folds - 1 else n

        test_mask = np.zeros(n, dtype=bool)
        test_mask[start:end] = True
        train_mask = ~test_mask

        train_mirna = torch.LongTensor(self.all_mirna_idx[train_mask]).to(self.device)
        train_drug = torch.LongTensor(self.all_drug_idx[train_mask]).to(self.device)
        train_labels = torch.FloatTensor(self.all_labels[train_mask]).to(self.device)

        test_mirna = torch.LongTensor(self.all_mirna_idx[test_mask]).to(self.device)
        test_drug = torch.LongTensor(self.all_drug_idx[test_mask]).to(self.device)
        test_labels = torch.FloatTensor(self.all_labels[test_mask]).to(self.device)

        return (train_mirna, train_drug, train_labels,
                test_mirna, test_drug, test_labels)


# ============================================================
# 训练一个 Fold
# ============================================================
def train_one_fold(model, data, fold_idx, args):
    """训练一个 fold 并返回测试指标"""

    (train_mirna, train_drug, train_labels,
     test_mirna, test_drug, test_labels) = data.get_fold(fold_idx)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    n_train = len(train_labels)
    batch_size = args.batch_size

    for epoch in range(args.epochs):
        model.train()

        # 打乱训练数据
        perm = torch.randperm(n_train, device=data.device)
        train_mirna_s = train_mirna[perm]
        train_drug_s = train_drug[perm]
        train_labels_s = train_labels[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            end = min(i + batch_size, n_train)
            batch_m = train_mirna_s[i:end]
            batch_d = train_drug_s[i:end]
            batch_y = train_labels_s[i:end]

            optimizer.zero_grad()

            scores, attn_w = model(
                data.mirna_feat, data.drug_feat,
                data.mirna_sim_edge, data.drug_sim_edge,
                data.mirna_gene_edge, data.drug_gene_edge,
                batch_m, batch_d
            )

            loss = criterion(scores, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # 每 10 个 epoch 评估一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_scores, _ = model(
                    data.mirna_feat, data.drug_feat,
                    data.mirna_sim_edge, data.drug_sim_edge,
                    data.mirna_gene_edge, data.drug_gene_edge,
                    test_mirna, test_drug
                )
                test_probs = torch.sigmoid(test_scores).cpu().numpy()
                test_y = test_labels.cpu().numpy()

                auc = roc_auc_score(test_y, test_probs)
                aupr = average_precision_score(test_y, test_probs)
                preds = (test_probs > 0.5).astype(int)
                acc = accuracy_score(test_y, preds)
                f1 = f1_score(test_y, preds)

            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"AUC: {auc:.4f} | AUPR: {aupr:.4f} | "
                  f"ACC: {acc:.4f} | F1: {f1:.4f}")

            # Early stopping
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    # 加载最佳模型进行最终评估
    if best_state is not None:
        model.load_state_dict({k: v.to(data.device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        test_scores, attn_weights = model(
            data.mirna_feat, data.drug_feat,
            data.mirna_sim_edge, data.drug_sim_edge,
            data.mirna_gene_edge, data.drug_gene_edge,
            test_mirna, test_drug
        )
        test_probs = torch.sigmoid(test_scores).cpu().numpy()
        test_y = test_labels.cpu().numpy()

    # 计算所有指标
    auc = roc_auc_score(test_y, test_probs)
    aupr = average_precision_score(test_y, test_probs)
    preds = (test_probs > 0.5).astype(int)
    acc = accuracy_score(test_y, preds)
    f1 = f1_score(test_y, preds)
    prec = precision_score(test_y, preds)
    rec = recall_score(test_y, preds)

    metrics = {
        'AUC': auc, 'AUPR': aupr, 'ACC': acc,
        'F1': f1, 'Precision': prec, 'Recall': rec,
    }

    return metrics, test_probs, test_y, attn_weights


# ============================================================
# Main: 5折CV
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='DrugMiR Training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id (-1 for CPU)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2, help='homo GCN layers')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--patience', type=int, default=20, help='early stopping')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    # 设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("🖥️  Using CPU")

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 加载数据
    data = DrugMiRData(args.data_dir, device)

    # 结果目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 5折交叉验证
    all_metrics = []
    all_probs = []
    all_labels = []

    print(f"\n{'='*60}")
    print(f"🚀 开始 {data.n_folds} 折交叉验证")
    print(f"{'='*60}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Hidden: {args.hidden}")
    print(f"  HomoGCN Layers: {args.n_layers}, Dropout: {args.dropout}")
    print(f"  Batch Size: {args.batch_size}, Patience: {args.patience}")

    start_time = time.time()

    for fold in range(data.n_folds):
        print(f"\n{'─'*40}")
        print(f"📊 Fold {fold+1}/{data.n_folds}")
        print(f"{'─'*40}")

        # 新建模型
        model = DrugMiR(
            mirna_feat_dim=data.mirna_feat.shape[1],
            drug_feat_dim=data.drug_feat.shape[1],
            n_genes=data.n_gene,
            hidden_dim=args.hidden,
            n_homo_layers=args.n_layers,
            dropout=args.dropout,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold == 0:
            print(f"  模型参数量: {n_params:,}")

        metrics, probs, labels, attn_w = train_one_fold(model, data, fold, args)
        all_metrics.append(metrics)
        all_probs.append(probs)
        all_labels.append(labels)

        print(f"\n  ✅ Fold {fold+1} Results:")
        for k, v in metrics.items():
            print(f"     {k}: {v:.4f}")

        # 保存每个 fold 的模型
        torch.save(model.state_dict(), save_dir / f"drugmir_fold{fold+1}.pt")

    elapsed = time.time() - start_time

    # 汇总结果
    print(f"\n{'='*60}")
    print(f"📊 5折交叉验证最终结果")
    print(f"{'='*60}")

    results_summary = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        results_summary[key] = (mean, std)
        print(f"  {key:12s}: {mean:.4f} ± {std:.4f}")

    print(f"\n  训练总时间: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  结果保存: {save_dir}/")

    # 保存结果
    np.savez(
        save_dir / "cv_results.npz",
        metrics=all_metrics,
        probs=np.concatenate(all_probs),
        labels=np.concatenate(all_labels),
    )

    # 保存文本报告
    with open(save_dir / "results.txt", 'w') as f:
        f.write("DrugMiR 5-Fold Cross Validation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Hyperparameters:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nModel parameters: {n_params:,}\n\n")
        f.write("Results:\n")
        for key, (mean, std) in results_summary.items():
            f.write(f"  {key}: {mean:.4f} ± {std:.4f}\n")
        f.write(f"\nTraining time: {elapsed:.1f}s\n")

    print(f"\n🎉 训练完成!")


if __name__ == "__main__":
    main()
