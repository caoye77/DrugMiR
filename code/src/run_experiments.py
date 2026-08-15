"""
DrugMiR 完整实验脚本
=====================
一键运行: Baseline对比 + Ablation Study + 可视化

运行: python src/run_experiments.py --gpu 0
"""
import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)

sys.path.append(str(Path(__file__).parent))
from drugmir_model import DrugMiR
from baselines import BASELINE_MODELS, ABLATION_MODELS
from train import DrugMiRData


# ============================================================
# 通用训练函数
# ============================================================
def train_and_evaluate(model, data, args, model_name="Model"):
    """训练模型并返回5折CV指标"""
    all_metrics = []
    all_probs = []
    all_labels = []

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

            total_loss = 0
            n_batch = 0
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

            # 每 20 epoch 检查 early stopping
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    s, _ = model(
                        data.mirna_feat, data.drug_feat,
                        data.mirna_sim_edge, data.drug_sim_edge,
                        data.mirna_gene_edge, data.drug_gene_edge,
                        test_m, test_d
                    )
                    auc = roc_auc_score(test_y.cpu(), torch.sigmoid(s).cpu())
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= args.patience // 2:
                        break

        # 最终评估
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

        auc = roc_auc_score(y, probs)
        aupr = average_precision_score(y, probs)
        preds = (probs > 0.5).astype(int)

        metrics = {
            'AUC': auc, 'AUPR': aupr,
            'ACC': accuracy_score(y, preds),
            'F1': f1_score(y, preds),
            'Precision': precision_score(y, preds),
            'Recall': recall_score(y, preds),
        }
        all_metrics.append(metrics)
        all_probs.append(probs)
        all_labels.append(y)

        # 重置模型
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    # 汇总
    summary = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        summary[key] = (np.mean(vals), np.std(vals))

    return summary, np.concatenate(all_probs), np.concatenate(all_labels)


# ============================================================
# 可视化
# ============================================================
def generate_plots(results, save_dir):
    """生成论文用图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12, 'figure.dpi': 300,
        'figure.figsize': (8, 6),
        'axes.grid': True, 'grid.alpha': 0.3,
    })

    save_dir = Path(save_dir)

    # ---- 1. Baseline 对比柱状图 ----
    methods = list(results.keys())
    auc_means = [results[m]['AUC'][0] for m in methods]
    auc_stds = [results[m]['AUC'][1] for m in methods]
    aupr_means = [results[m]['AUPR'][0] for m in methods]
    aupr_stds = [results[m]['AUPR'][1] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, auc_means, width, yerr=auc_stds,
                   label='AUC', color='#2196F3', capsize=3, alpha=0.85)
    bars2 = ax.bar(x + width/2, aupr_means, width, yerr=aupr_stds,
                   label='AUPR', color='#FF9800', capsize=3, alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0.7, 1.0)

    # 在柱子上标数值
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.4f}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.4f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_barplot.png', bbox_inches='tight')
    plt.savefig(save_dir / 'comparison_barplot.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✅ comparison_barplot.png")

    # ---- 2. ROC 曲线 (如果有 probs) ----
    if 'DrugMiR_probs' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for name in ['DrugMiR', 'MLP', 'GCN', 'GAT']:
            key_p = f'{name}_probs'
            key_l = f'{name}_labels'
            if key_p in results:
                fpr, tpr, _ = roc_curve(results[key_l], results[key_p])
                auc_val = results[name]['AUC'][0]
                ax1.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})', linewidth=2)

                prec, rec, _ = precision_recall_curve(results[key_l], results[key_p])
                aupr_val = results[name]['AUPR'][0]
                ax2.plot(rec, prec, label=f'{name} (AUPR={aupr_val:.4f})', linewidth=2)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc='lower right')

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc='lower left')

        plt.tight_layout()
        plt.savefig(save_dir / 'roc_pr_curves.png', bbox_inches='tight')
        plt.savefig(save_dir / 'roc_pr_curves.pdf', bbox_inches='tight')
        plt.close()
        print(f"  ✅ roc_pr_curves.png")

    # ---- 3. Ablation 柱状图 ----
    ablation_methods = [m for m in methods if m.startswith('w/o') or m == 'DrugMiR']
    if len(ablation_methods) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(ablation_methods))
        colors = ['#4CAF50' if m == 'DrugMiR' else '#EF5350' for m in ablation_methods]

        vals = [results[m]['AUC'][0] for m in ablation_methods]
        errs = [results[m]['AUC'][1] for m in ablation_methods]

        bars = ax.bar(x, vals, yerr=errs, color=colors, capsize=4, alpha=0.85)
        ax.set_ylabel('AUC')
        ax.set_title('Ablation Study')
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_methods, rotation=25, ha='right')
        ax.set_ylim(min(vals) - 0.03, max(vals) + 0.02)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_dir / 'ablation_study.png', bbox_inches='tight')
        plt.savefig(save_dir / 'ablation_study.pdf', bbox_inches='tight')
        plt.close()
        print(f"  ✅ ablation_study.png")

    # ---- 4. 结果表格 (LaTeX) ----
    with open(save_dir / 'results_table.tex', 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance comparison of DrugMiR with baseline methods.}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & AUC & AUPR & ACC & F1 & Precision & Recall \\\\\n")
        f.write("\\midrule\n")
        for m in methods:
            vals = results[m]
            row = f"{m}"
            for key in ['AUC', 'AUPR', 'ACC', 'F1', 'Precision', 'Recall']:
                mean, std = vals[key]
                row += f" & {mean:.4f}$\\pm${std:.4f}"
            row += " \\\\\n"
            # DrugMiR 行加粗
            if m == 'DrugMiR':
                f.write("\\midrule\n")
                row = row.replace(m, f"\\textbf{{{m}}}")
            f.write(row)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  ✅ results_table.tex")


# ============================================================
# Main
# ============================================================
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
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = DrugMiRData(args.data_dir, device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    results = {}

    # ============================================================
    # Part 1: DrugMiR (主模型)
    # ============================================================
    print(f"\n{'='*60}")
    print("🚀 Part 1: DrugMiR (主模型)")
    print(f"{'='*60}")

    model = DrugMiR(
        mirna_feat_dim=data.mirna_feat.shape[1],
        drug_feat_dim=data.drug_feat.shape[1],
        n_genes=data.n_gene,
        hidden_dim=args.hidden, n_homo_layers=2, dropout=args.dropout,
    ).to(device)

    t0 = time.time()
    summary, probs, labels = train_and_evaluate(model, data, args, "DrugMiR")
    t1 = time.time()
    results['DrugMiR'] = summary
    results['DrugMiR_probs'] = probs
    results['DrugMiR_labels'] = labels
    print(f"  DrugMiR: AUC={summary['AUC'][0]:.4f}±{summary['AUC'][1]:.4f} "
          f"AUPR={summary['AUPR'][0]:.4f} ({t1-t0:.0f}s)")

    # ============================================================
    # Part 2: Baseline 对比
    # ============================================================
    print(f"\n{'='*60}")
    print("📊 Part 2: Baseline 对比")
    print(f"{'='*60}")

    for name, ModelClass in BASELINE_MODELS.items():
        print(f"\n  训练 {name}...")
        model = ModelClass(
            mirna_dim=data.mirna_feat.shape[1],
            drug_dim=data.drug_feat.shape[1],
            hidden_dim=args.hidden,
            n_mirna=data.n_mirna,
            n_drug=data.n_drug,
            dropout=args.dropout,
        ).to(device)

        t0 = time.time()
        summary, probs, labels = train_and_evaluate(model, data, args, name)
        t1 = time.time()
        results[name] = summary
        results[f'{name}_probs'] = probs
        results[f'{name}_labels'] = labels
        print(f"  {name}: AUC={summary['AUC'][0]:.4f}±{summary['AUC'][1]:.4f} "
              f"AUPR={summary['AUPR'][0]:.4f} ({t1-t0:.0f}s)")

    # ============================================================
    # Part 3: Ablation Study
    # ============================================================
    print(f"\n{'='*60}")
    print("🔬 Part 3: Ablation Study")
    print(f"{'='*60}")

    for name, ModelClass in ABLATION_MODELS.items():
        print(f"\n  训练 {name}...")
        model = ModelClass(
            mirna_feat_dim=data.mirna_feat.shape[1],
            drug_feat_dim=data.drug_feat.shape[1],
            n_genes=data.n_gene,
            hidden_dim=args.hidden,
            n_homo_layers=2,
            dropout=args.dropout,
        ).to(device)

        t0 = time.time()
        summary, _, _ = train_and_evaluate(model, data, args, name)
        t1 = time.time()
        results[name] = summary
        print(f"  {name}: AUC={summary['AUC'][0]:.4f}±{summary['AUC'][1]:.4f} "
              f"AUPR={summary['AUPR'][0]:.4f} ({t1-t0:.0f}s)")

    # ============================================================
    # Part 4: 汇总 + 可视化
    # ============================================================
    print(f"\n{'='*60}")
    print("📊 最终结果汇总")
    print(f"{'='*60}")

    print(f"\n{'Method':<20s} {'AUC':>14s} {'AUPR':>14s} {'F1':>14s}")
    print("-" * 65)
    method_order = ['DrugMiR'] + list(BASELINE_MODELS.keys()) + list(ABLATION_MODELS.keys())
    for name in method_order:
        if name in results and isinstance(results[name], dict):
            s = results[name]
            print(f"{name:<20s} {s['AUC'][0]:.4f}±{s['AUC'][1]:.4f}"
                  f"  {s['AUPR'][0]:.4f}±{s['AUPR'][1]:.4f}"
                  f"  {s['F1'][0]:.4f}±{s['F1'][1]:.4f}")

    # 生成可视化
    print(f"\n📈 生成图表...")
    plot_results = {k: v for k, v in results.items() if isinstance(v, dict) or isinstance(v, np.ndarray)}
    generate_plots(plot_results, save_dir)

    # 保存原始结果
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_results[k] = {kk: [float(vv[0]), float(vv[1])] for kk, vv in v.items()}
    with open(save_dir / 'all_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  ✅ all_results.json")

    print(f"\n🎉 全部实验完成! 结果保存在 {save_dir}/")


if __name__ == "__main__":
    main()
