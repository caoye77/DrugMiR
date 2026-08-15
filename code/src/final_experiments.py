"""
DrugMiR 最终优化实验
=====================
1. Dropout=0.5 调优
2. Morgan+ChemBERTa 拼接
3. 保存 ROC/PR 曲线数据
4. 多任务 ablation (单任务 vs 多任务)

运行: cd ~/work/DrugMiR && python src/final_experiments.py --gpu 0
"""
import sys, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, accuracy_score, roc_curve, precision_recall_curve)

sys.path.append(str(Path(__file__).parent))
from drugmir_v2 import DrugMiRv2
from baselines import MLPBaseline, GCNBaseline, GATBaseline

def build_knn_edges(sim_matrix, k=10):
    n = sim_matrix.shape[0]
    src, dst = [], []
    for i in range(n):
        sims = sim_matrix[i].copy(); sims[i] = -1
        topk = np.argsort(-sims)[:k]
        for j in topk:
            if sim_matrix[i, j] > 0:
                src.append(i); dst.append(j)
    return np.array([src, dst])

def run_cv_full(make_model, fwd_fn, name, assoc, device, seeds=[42,123,2024],
                epochs=150, save_curves=False):
    """Full CV with optional ROC curve saving"""
    n_mirna, n_drug = assoc.shape
    all_aucs, all_auprs, all_f1s = [], [], []
    all_probs_concat, all_labels_concat = [], []

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed(seed)
        pos_i, pos_j = np.where(assoc > 0); n_pos = len(pos_i)
        pos_set = set(zip(pos_i.tolist(), pos_j.tolist()))
        neg = []
        while len(neg) < n_pos:
            m = np.random.randint(n_mirna); d = np.random.randint(n_drug)
            if (m, d) not in pos_set: neg.append((m, d)); pos_set.add((m, d))
        neg = np.array(neg)
        am = np.concatenate([pos_i, neg[:, 0]]); ad = np.concatenate([pos_j, neg[:, 1]])
        ay = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])
        perm = np.random.permutation(len(ay)); am, ad, ay = am[perm], ad[perm], ay[perm]
        fs = len(ay) // 5

        for fold in range(5):
            s = fold * fs; e = s + fs if fold < 4 else len(ay)
            te = np.zeros(len(ay), dtype=bool); te[s:e] = True; tr = ~te
            trm = torch.LongTensor(am[tr]).to(device)
            trd = torch.LongTensor(ad[tr]).to(device)
            try2 = torch.FloatTensor(ay[tr]).to(device)
            tem = torch.LongTensor(am[te]).to(device)
            ted = torch.LongTensor(ad[te]).to(device)
            tey = torch.FloatTensor(ay[te]).to(device)

            model = make_model().to(device)
            opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            crit = nn.BCEWithLogitsLoss()
            best = 0; bst = None; pat = 0

            for ep in range(epochs):
                model.train()
                pm = torch.randperm(len(try2), device=device)
                for i in range(0, len(try2), 2048):
                    end2 = min(i + 2048, len(try2)); idx = pm[i:end2]
                    opt.zero_grad()
                    s2 = fwd_fn(model, trm[idx], trd[idx])
                    crit(s2, try2[idx]).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                sch.step()
                if (ep + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        s2 = fwd_fn(model, tem, ted)
                        auc = roc_auc_score(tey.cpu(), torch.sigmoid(s2).cpu())
                    if auc > best:
                        best = auc; bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pat = 0
                    else:
                        pat += 1
                        if pat >= 15: break

            if bst: model.load_state_dict({k: v.to(device) for k, v in bst.items()})
            model.eval()
            with torch.no_grad():
                s2 = fwd_fn(model, tem, ted)
                probs = torch.sigmoid(s2).cpu().numpy(); y = tey.cpu().numpy()

            all_aucs.append(roc_auc_score(y, probs))
            all_auprs.append(average_precision_score(y, probs))
            preds = (probs > 0.5).astype(int)
            all_f1s.append(f1_score(y, preds))

            if save_curves and seed == 42:
                all_probs_concat.append(probs)
                all_labels_concat.append(y)

    m_auc = np.mean(all_aucs); s_auc = np.std(all_aucs)
    m_aupr = np.mean(all_auprs); s_aupr = np.std(all_auprs)
    m_f1 = np.mean(all_f1s); s_f1 = np.std(all_f1s)
    print(f"  {name}: AUC={m_auc:.4f}+/-{s_auc:.4f}  AUPR={m_aupr:.4f}+/-{s_aupr:.4f}  F1={m_f1:.4f}")

    result = {
        'AUC': (m_auc, s_auc), 'AUPR': (m_aupr, s_aupr), 'F1': (m_f1, s_f1),
        'auc_list': all_aucs
    }
    if save_curves and all_probs_concat:
        result['probs'] = np.concatenate(all_probs_concat)
        result['labels'] = np.concatenate(all_labels_concat)
    return result


def run_multitask_cv(mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                     mirna_gene_edge, drug_gene_edge, task_matrix, device,
                     name, dropout=0.5, seeds=[42,123,2024], epochs=150):
    """Single-task CV for resistance-only or sensitivity-only"""
    n_mirna, n_drug = task_matrix.shape
    all_aucs = []

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed(seed)
        pos_i, pos_j = np.where(task_matrix > 0); n_pos = len(pos_i)
        if n_pos == 0: print(f"  {name}: no positive samples"); return None

        pos_set = set(zip(pos_i.tolist(), pos_j.tolist()))
        neg = []
        while len(neg) < n_pos:
            m = np.random.randint(n_mirna); d = np.random.randint(n_drug)
            if (m, d) not in pos_set: neg.append((m, d)); pos_set.add((m, d))
        neg = np.array(neg)
        am = np.concatenate([pos_i, neg[:, 0]]); ad = np.concatenate([pos_j, neg[:, 1]])
        ay = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])
        perm = np.random.permutation(len(ay)); am, ad, ay = am[perm], ad[perm], ay[perm]
        fs = len(ay) // 5

        for fold in range(5):
            s = fold * fs; e = s + fs if fold < 4 else len(ay)
            te = np.zeros(len(ay), dtype=bool); te[s:e] = True; tr = ~te
            trm = torch.LongTensor(am[tr]).to(device); trd = torch.LongTensor(ad[tr]).to(device)
            try2 = torch.FloatTensor(ay[tr]).to(device)
            tem = torch.LongTensor(am[te]).to(device); ted = torch.LongTensor(ad[te]).to(device)
            tey = torch.FloatTensor(ay[te]).to(device)

            model = DrugMiRv2(mirna_feat.shape[1], drug_feat.shape[1], 14455,
                              128, 2, 2, dropout).to(device)
            opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            crit = nn.BCEWithLogitsLoss()
            best = 0; bst = None; pat = 0

            for ep in range(epochs):
                model.train(); pm = torch.randperm(len(try2), device=device)
                for i in range(0, len(try2), 2048):
                    end2 = min(i + 2048, len(try2)); idx = pm[i:end2]
                    opt.zero_grad()
                    s2, _ = model(mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                                  mirna_gene_edge, drug_gene_edge, trm[idx], trd[idx])
                    crit(s2, try2[idx]).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                sch.step()
                if (ep + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        s2, _ = model(mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                                      mirna_gene_edge, drug_gene_edge, tem, ted)
                        auc = roc_auc_score(tey.cpu(), torch.sigmoid(s2).cpu())
                    if auc > best: best = auc; bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pat = 0
                    else:
                        pat += 1
                        if pat >= 15: break
            if bst: model.load_state_dict({k: v.to(device) for k, v in bst.items()})
            model.eval()
            with torch.no_grad():
                s2, _ = model(mirna_feat, drug_feat, mirna_sim_edge, drug_sim_edge,
                              mirna_gene_edge, drug_gene_edge, tem, ted)
            all_aucs.append(roc_auc_score(tey.cpu(), torch.sigmoid(s2).cpu()))

    print(f"  {name}: AUC={np.mean(all_aucs):.4f}+/-{np.std(all_aucs):.4f}")
    return {'AUC': (np.mean(all_aucs), np.std(all_aucs)), 'auc_list': all_aucs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load all data
    mirna_feat = torch.FloatTensor(np.load("data/processed/mirna_kmer_features.npy")).to(device)
    drug_morgan = torch.FloatTensor(np.load("data/processed/drug_morgan_features.npy")).to(device)
    drug_chemberta = torch.FloatTensor(np.load("data/processed/drug_chemberta_features.npy")).to(device)
    drug_concat = torch.cat([drug_morgan, drug_chemberta], dim=1).to(device)
    assoc = np.load("data/processed/association_matrix.npy")
    mirna_sim = np.load("data/processed/mirna_similarity.npy")
    drug_sim = np.load("data/processed/drug_similarity.npy")
    edges = np.load("data/processed/edge_lists.npz")
    mirna_gene_edge = torch.LongTensor(edges['mirna_gene'].T).to(device)
    drug_gene_edge = torch.LongTensor(edges['drug_gene'].T).to(device)

    # KNN edges
    mirna_knn = build_knn_edges(mirna_sim, k=15)
    drug_knn = build_knn_edges(drug_sim, k=10)
    mm_orig = edges['mirna_mirna_sim']; dd_orig = edges['drug_drug_sim']
    mm_all = np.hstack([mirna_knn, mm_orig.T, mm_orig[:, [1, 0]].T]) if len(mm_orig) > 0 else mirna_knn
    dd_all = np.hstack([drug_knn, dd_orig.T, dd_orig[:, [1, 0]].T]) if len(dd_orig) > 0 else drug_knn
    mse = torch.LongTensor(mm_all).to(device)
    dse = torch.LongTensor(dd_all).to(device)
    mge = mirna_gene_edge; dge = drug_gene_edge

    print(f"Morgan: {drug_morgan.shape}, ChemBERTa: {drug_chemberta.shape}, Concat: {drug_concat.shape}")

    results = {}

    # ============================================================
    # #1: Dropout optimization + #3: ROC curves
    # ============================================================
    print("\n" + "=" * 60)
    print("Exp 1: Dropout=0.5 + ROC/PR curves")
    print("=" * 60)

    # DrugMiR best config (dropout=0.5)
    results['DrugMiR'] = run_cv_full(
        lambda: DrugMiRv2(256, 1024, 14455, 128, 2, 2, 0.5),
        lambda m, mi, di: m(mirna_feat, drug_morgan, mse, dse, mge, dge, mi, di)[0],
        "DrugMiR (d=0.5)", assoc, device, save_curves=True)

    # MLP baseline for ROC comparison
    results['MLP'] = run_cv_full(
        lambda: MLPBaseline(256, 1024, 128, 0.5),
        lambda m, mi, di: m(mirna_feat, drug_morgan, None, None, None, None, mi, di)[0],
        "MLP (d=0.5)", assoc, device, save_curves=True)

    # GCN for ROC
    results['GCN'] = run_cv_full(
        lambda: GCNBaseline(256, 1024, 128, 1578, 156, 0.5),
        lambda m, mi, di: m(mirna_feat, drug_morgan, mse, dse, mge, dge, mi, di)[0],
        "GCN (d=0.5)", assoc, device, save_curves=True)

    # GAT for ROC
    results['GAT'] = run_cv_full(
        lambda: GATBaseline(256, 1024, 128, 1578, 156, 0.5),
        lambda m, mi, di: m(mirna_feat, drug_morgan, mse, dse, mge, dge, mi, di)[0],
        "GAT (d=0.5)", assoc, device, save_curves=True)

    # ============================================================
    # #2: Morgan + ChemBERTa concat
    # ============================================================
    print("\n" + "=" * 60)
    print("Exp 2: Feature concatenation")
    print("=" * 60)

    results['DrugMiR+Concat'] = run_cv_full(
        lambda: DrugMiRv2(256, 1792, 14455, 128, 2, 2, 0.5),
        lambda m, mi, di: m(mirna_feat, drug_concat, mse, dse, mge, dge, mi, di)[0],
        "DrugMiR+Concat(Morgan+ChemBERTa)", assoc, device)

    # ============================================================
    # #4: Multi-task ablation
    # ============================================================
    print("\n" + "=" * 60)
    print("Exp 3: Multi-task ablation")
    print("=" * 60)

    res_matrix = np.load("data/processed/resistance_matrix.npy")
    sen_matrix = np.load("data/processed/sensitivity_matrix.npy")

    print(f"Resistance pairs: {res_matrix.sum()}, Sensitivity pairs: {sen_matrix.sum()}")

    results['Resistance-only'] = run_multitask_cv(
        mirna_feat, drug_morgan, mse, dse, mge, dge, res_matrix, device,
        "Resistance-only", dropout=0.5)

    results['Sensitivity-only'] = run_multitask_cv(
        mirna_feat, drug_morgan, mse, dse, mge, dge, sen_matrix, device,
        "Sensitivity-only", dropout=0.5)

    # ============================================================
    # Statistical tests
    # ============================================================
    print("\n" + "=" * 60)
    print("Statistical Tests")
    print("=" * 60)

    for name in ['MLP', 'GCN', 'GAT']:
        t, p = stats.ttest_rel(results['DrugMiR']['auc_list'], results[name]['auc_list'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f"  DrugMiR vs {name}: t={t:.3f}, p={p:.6f} {sig}")

    if 'DrugMiR+Concat' in results:
        t, p = stats.ttest_rel(results['DrugMiR+Concat']['auc_list'], results['DrugMiR']['auc_list'])
        print(f"  Concat vs Morgan-only: t={t:.3f}, p={p:.6f}")

    # ============================================================
    # Save ROC/PR curve data
    # ============================================================
    print("\n" + "=" * 60)
    print("Saving ROC/PR curve data")
    print("=" * 60)

    curve_data = {}
    for name in ['DrugMiR', 'MLP', 'GCN', 'GAT']:
        if 'probs' in results[name]:
            probs = results[name]['probs']
            labels = results[name]['labels']
            fpr, tpr, _ = roc_curve(labels, probs)
            prec, rec, _ = precision_recall_curve(labels, probs)
            curve_data[name] = {
                'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                'precision': prec.tolist(), 'recall': rec.tolist(),
                'auc': results[name]['AUC'][0],
                'aupr': results[name]['AUPR'][0],
            }

    with open('results/roc_pr_data.json', 'w') as f:
        json.dump(curve_data, f)
    print("  Saved results/roc_pr_data.json")

    # ============================================================
    # Generate plots
    # ============================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 13, 'figure.dpi': 300, 'axes.grid': True, 'grid.alpha': 0.3})

    # ROC curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'DrugMiR': '#E53935', 'MLP': '#1E88E5', 'GCN': '#43A047', 'GAT': '#FB8C00'}

    for name in ['DrugMiR', 'MLP', 'GCN', 'GAT']:
        if name in curve_data:
            cd = curve_data[name]
            ax1.plot(cd['fpr'], cd['tpr'], label=f"{name} (AUC={cd['auc']:.4f})",
                     color=colors[name], linewidth=2.5)
            ax2.plot(cd['recall'], cd['precision'], label=f"{name} (AUPR={cd['aupr']:.4f})",
                     color=colors[name], linewidth=2.5)

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve'); ax1.legend(loc='lower right', fontsize=11)

    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve'); ax2.legend(loc='lower left', fontsize=11)

    plt.tight_layout()
    plt.savefig('results/roc_pr_curves.png', bbox_inches='tight')
    plt.savefig('results/roc_pr_curves.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved results/roc_pr_curves.png/pdf")

    # Comparison barplot (all methods)
    methods = ['DrugMiR', 'MLP', 'GCN', 'GAT']
    auc_m = [results[m]['AUC'][0] for m in methods]
    auc_s = [results[m]['AUC'][1] for m in methods]
    aupr_m = [results[m]['AUPR'][0] for m in methods]
    aupr_s = [results[m]['AUPR'][1] for m in methods]

    x = np.arange(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    b1 = ax.bar(x - w/2, auc_m, w, yerr=auc_s, label='AUC', color='#2196F3', capsize=4, alpha=0.85)
    b2 = ax.bar(x + w/2, aupr_m, w, yerr=aupr_s, label='AUPR', color='#FF9800', capsize=4, alpha=0.85)
    ax.set_ylabel('Score'); ax.set_title('Performance Comparison (5-Fold CV)')
    ax.set_xticks(x); ax.set_xticklabels(methods); ax.legend(); ax.set_ylim(0.88, 0.97)
    for bar in b1:
        h = bar.get_height(); ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f'{h:.4f}', ha='center', fontsize=9)
    for bar in b2:
        h = bar.get_height(); ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f'{h:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('results/comparison_barplot_final.png', bbox_inches='tight')
    plt.savefig('results/comparison_barplot_final.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved results/comparison_barplot_final.png/pdf")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<35s} {'AUC':>16s} {'AUPR':>16s} {'F1':>12s}")
    print("-" * 80)
    for name in ['DrugMiR', 'DrugMiR+Concat', 'MLP', 'GCN', 'GAT',
                  'Resistance-only', 'Sensitivity-only']:
        if name in results and results[name] is not None:
            r = results[name]
            auc_str = f"{r['AUC'][0]:.4f}+/-{r['AUC'][1]:.4f}"
            aupr_str = f"{r['AUPR'][0]:.4f}+/-{r['AUPR'][1]:.4f}" if 'AUPR' in r else "N/A"
            f1_str = f"{r['F1'][0]:.4f}" if 'F1' in r else "N/A"
            print(f"  {name:<33s} {auc_str:>16s} {aupr_str:>16s} {f1_str:>12s}")

    # Save all results
    save_r = {}
    for k, v in results.items():
        if v is not None:
            save_r[k] = {kk: [float(vv[0]), float(vv[1])] for kk, vv in v.items()
                         if isinstance(vv, tuple) and len(vv) == 2}
    with open('results/final_results.json', 'w') as f:
        json.dump(save_r, f, indent=2)
    print("\nSaved results/final_results.json")
    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
