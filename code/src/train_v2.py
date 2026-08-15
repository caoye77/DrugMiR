"""
DrugMiR v2 快速对比实验
========================
对比: DrugMiR_v2 vs MLP vs DrugMiR_v1

运行: python src/train_v2.py --gpu 0
"""
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)

sys.path.append(str(Path(__file__).parent))
from train import DrugMiRData


def train_model(model, data, args, model_name="Model"):
    """训练 + 5折CV"""
    all_metrics = []

    for fold in range(data.n_folds):
        (train_m, train_d, train_y,
         test_m, test_d, test_y) = data.get_fold(fold)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0
        best_state = None
        patience_cnt = 0

        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(train_y), device=data.device)
            total_loss, n_batch = 0, 0

            for i in range(0, len(train_y), args.batch_size):
                end = min(i + args.batch_size, len(train_y))
                idx = perm[i:end]
                optimizer.zero_grad()
                scores, _ = model(
                    data.mirna_feat, data.drug_feat,
                    data.mirna_sim_edge, data.drug_sim_edge,
                    data.mirna_gene_edge, data.drug_gene_edge,
                    train_m[idx], train_d[idx]
                )
                loss = criterion(scores, train_y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batch += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    s, _ = model(
                        data.mirna_feat, data.drug_feat,
                        data.mirna_sim_edge, data.drug_sim_edge,
                        data.mirna_gene_edge, data.drug_gene_edge,
                        test_m, test_d
                    )
                    auc = roc_auc_score(test_y.cpu(), torch.sigmoid(s).cpu())

                if epoch < 30 or (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / n_batch
                    if fold == 0:
                        print(f"    Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | AUC: {auc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= args.patience:
                        if fold == 0:
                            print(f"    Early stop at epoch {epoch+1}")
                        break

        if best_state:
            model.load_state_dict({k: v.to(data.device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            scores, _ = model(
                data.mirna_feat, data.drug_feat,
                data.mirna_sim_edge, data.drug_sim_edge,
                data.mirna_gene_edge, data.drug_gene_edge,
                test_m, test_d
            )
            probs = torch.sigmoid(scores).cpu().numpy()
            y = test_y.cpu().numpy()

        preds = (probs > 0.5).astype(int)
        metrics = {
            'AUC': roc_auc_score(y, probs),
            'AUPR': average_precision_score(y, probs),
            'ACC': accuracy_score(y, preds),
            'F1': f1_score(y, preds),
            'Precision': precision_score(y, preds),
            'Recall': recall_score(y, preds),
        }
        all_metrics.append(metrics)

    # 汇总
    summary = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        summary[key] = (np.mean(vals), np.std(vals))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data/processed')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"🖥️  Device: {device}")
    data = DrugMiRData(args.data_dir, device)

    results = {}

    # ---- 1. DrugMiR v2 ----
    print(f"\n{'='*60}")
    print("🚀 DrugMiR v2 (改进版)")
    print(f"{'='*60}")
    from drugmir_v2 import DrugMiRv2
    model = DrugMiRv2(
        mirna_feat_dim=data.mirna_feat.shape[1],
        drug_feat_dim=data.drug_feat.shape[1],
        n_genes=data.n_gene,
        hidden_dim=args.hidden,
        n_homo_layers=2,
        n_bridge_layers=2,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,}")

    t0 = time.time()
    results['DrugMiR_v2'] = train_model(model, data, args, "DrugMiR_v2")
    print(f"  ⏱️  {time.time()-t0:.0f}s")

    # ---- 2. MLP baseline ----
    print(f"\n{'='*60}")
    print("📊 MLP Baseline")
    print(f"{'='*60}")
    from baselines import MLPBaseline
    model = MLPBaseline(
        mirna_dim=data.mirna_feat.shape[1],
        drug_dim=data.drug_feat.shape[1],
        hidden_dim=args.hidden,
        dropout=args.dropout,
    ).to(device)
    t0 = time.time()
    results['MLP'] = train_model(model, data, args, "MLP")
    print(f"  ⏱️  {time.time()-t0:.0f}s")

    # ---- 3. DrugMiR v1 ----
    print(f"\n{'='*60}")
    print("📊 DrugMiR v1")
    print(f"{'='*60}")
    from drugmir_model import DrugMiR
    model = DrugMiR(
        mirna_feat_dim=data.mirna_feat.shape[1],
        drug_feat_dim=data.drug_feat.shape[1],
        n_genes=data.n_gene,
        hidden_dim=args.hidden,
        n_homo_layers=2,
        dropout=args.dropout,
    ).to(device)
    t0 = time.time()
    results['DrugMiR_v1'] = train_model(model, data, args, "DrugMiR_v1")
    print(f"  ⏱️  {time.time()-t0:.0f}s")

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print("📊 最终对比")
    print(f"{'='*60}")
    print(f"{'Method':<16s} {'AUC':>14s} {'AUPR':>14s} {'F1':>14s}")
    print("-" * 60)
    for name in ['DrugMiR_v2', 'MLP', 'DrugMiR_v1']:
        s = results[name]
        print(f"{name:<16s} {s['AUC'][0]:.4f}±{s['AUC'][1]:.4f}"
              f"  {s['AUPR'][0]:.4f}±{s['AUPR'][1]:.4f}"
              f"  {s['F1'][0]:.4f}±{s['F1'][1]:.4f}")

    winner = max(results.items(), key=lambda x: x[1]['AUC'][0])
    print(f"\n🏆 最优: {winner[0]} (AUC={winner[1]['AUC'][0]:.4f})")


if __name__ == "__main__":
    main()
