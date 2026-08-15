"""
Generate new Fig 6: Top-10 Bridge Genes Ranked by Integrated Gradients (IG).

Replaces the previous degree-product visualization with IG-based attribution
from the trained DrugMiR_Hybrid model.

Three panels:
(a) Top-10 by IG (with bar showing IG importance)
(b) Drug-targeting and miRNA-targeting degree of these top-10 genes
(c) Scatter: degree-product ranking vs IG ranking, showing the divergence
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.expanduser("~/work/DrugMiR/results_final")
DD1 = os.path.expanduser("~/DrugMiR/data/dataset1")
FIG_OUT = os.path.expanduser("~/work/DrugMiR/paper/figures/fig_bridge_genes.pdf")

# Load IG importance and gene mapping
ig_imp = np.load(f"{OUT_DIR}/gene_ig_importance_d1.npy")
gene_map = pd.read_csv("data/processed/gene_mapping.csv")
mg = np.load(f"{DD1}/mirna_gene_matrix.npy")
dg = np.load(f"{DD1}/drug_gene_matrix.npy")
mirna_deg = mg.sum(axis=0)
drug_deg = dg.sum(axis=0)
old_score = mirna_deg * drug_deg

# Only consider active bridge genes
active = (mirna_deg > 0) & (drug_deg > 0)
active_idx = np.where(active)[0]

# Top-10 by IG
ig_active = ig_imp[active_idx]
top10_ig_local = np.argsort(-ig_active)[:10]
top10_ig_global = active_idx[top10_ig_local]
top10_ig_names = [gene_map.iloc[i]['gene_name'] for i in top10_ig_global]
top10_ig_vals = ig_imp[top10_ig_global]
top10_ig_mdeg = mirna_deg[top10_ig_global]
top10_ig_ddeg = drug_deg[top10_ig_global]

# Annotated functions (curated)
GENE_FUNCS = {
    'HSF1': 'heat-shock TF; chemoresistance',
    'CASP3': 'apoptosis executioner',
    'RARG': 'retinoic acid receptor',
    'BCL2': 'anti-apoptotic',
    'EDN1': 'endothelin signaling',
    'EFEMP1': 'EGF-containing fibulin',
    'MAPK1': 'MAPK/ERK',
    'CYP3A4': 'drug metabolism',
    'BAX': 'pro-apoptotic',
    'AKT1': 'PI3K/AKT survival',
    'ALK': 'receptor tyrosine kinase',
    'MAPK3': 'MAPK/ERK',
    'MCL1': 'anti-apoptotic',
    'BCL2L1': 'anti-apoptotic',
    'PTGS2': 'COX-2 inflammation',
    'CASP9': 'apoptosis initiator',
    'RELA': 'NF-kB subunit',
    'PARP1': 'DNA damage repair',
    'CASP8': 'apoptosis initiator',
    'CCND1': 'cell cycle (cyclin D1)',
    'CDKN1A': 'cell cycle (p21)',
    'TP53': 'tumor suppressor',
    'SOD2': 'oxidative stress',
    'MYC': 'oncogene',
    'IGF1R': 'IGF signaling',
    'XIAP': 'apoptosis inhibitor',
}

# Old top-10 by degree
old_rank_order = np.argsort(-old_score)
old_top10_global = old_rank_order[:10]
old_top10_names = [gene_map.iloc[i]['gene_name'] for i in old_top10_global]

# Set up figure (3 panels)
fig = plt.figure(figsize=(16, 5))

# ========= Panel (a): IG Top-10 horizontal bar =========
ax1 = fig.add_subplot(1, 3, 1)
y_pos = np.arange(10)[::-1]  # reverse so rank 1 on top
ax1.barh(y_pos, top10_ig_vals, color='#3b7dd8', edgecolor='black', linewidth=0.6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{n}" for n in top10_ig_names], fontsize=10)
ax1.set_xlabel('Integrated Gradients (IG) Importance', fontsize=11)
ax1.set_title('(a) Top-10 Bridge Genes by IG', fontsize=11, loc='left')
for i, (yi, v, n) in enumerate(zip(y_pos, top10_ig_vals, top10_ig_names)):
    func = GENE_FUNCS.get(n, '')
    ax1.text(v + 0.05, yi, func, va='center', fontsize=8, color='#555555')
ax1.set_xlim(0, max(top10_ig_vals) * 1.6)
ax1.grid(axis='x', linestyle=':', alpha=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ========= Panel (b): miRNA-deg vs drug-deg of top-10 =========
ax2 = fig.add_subplot(1, 3, 2)
y_pos = np.arange(10)[::-1]
width = 0.4
ax2.barh(y_pos + width/2, top10_ig_mdeg, width, label='miRNA degree',
          color='#3b7dd8', edgecolor='black', linewidth=0.5)
ax2.barh(y_pos - width/2, top10_ig_ddeg, width, label='Drug degree',
          color='#e15759', edgecolor='black', linewidth=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top10_ig_names, fontsize=10)
ax2.set_xlabel('Connectivity Degree', fontsize=11)
ax2.set_title('(b) Targeting Degrees of IG Top-10', fontsize=11, loc='left')
ax2.legend(loc='lower right', fontsize=9, frameon=True)
ax2.grid(axis='x', linestyle=':', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ========= Panel (c): IG-rank vs degree-rank scatter =========
ax3 = fig.add_subplot(1, 3, 3)
# For all active genes, rank by IG and by old-score
ig_sort = np.argsort(-ig_imp[active_idx])
ig_rank = np.empty(len(active_idx)); ig_rank[ig_sort] = np.arange(1, len(active_idx) + 1)
old_sort = np.argsort(-old_score[active_idx])
old_rank = np.empty(len(active_idx)); old_rank[old_sort] = np.arange(1, len(active_idx) + 1)

# Scatter
ax3.scatter(old_rank, ig_rank, s=4, alpha=0.25, color='gray', edgecolors='none')
# Highlight IG top-10 (red triangles)
for g_local in top10_ig_local:
    g_global = active_idx[g_local]
    g_name = gene_map.iloc[g_global]['gene_name']
    or_v = old_rank[g_local]
    ir_v = ig_rank[g_local]
    ax3.scatter([or_v], [ir_v], s=70, color='#3b7dd8', edgecolor='black',
                 linewidth=1, zorder=3, marker='o')
    # label only those where ranks differ significantly
    if abs(or_v - ir_v) > 50 or ir_v <= 5:
        ax3.annotate(g_name, (or_v, ir_v), xytext=(6, 4), textcoords='offset points',
                     fontsize=8, color='#3b7dd8', fontweight='bold')
# Highlight old top-10 (blue squares) that ARE NOT in IG top-10
ig_top10_set = set(top10_ig_global.tolist())
for g_global in old_top10_global:
    if g_global in ig_top10_set:
        continue
    # find local idx
    g_local = np.where(active_idx == g_global)[0][0]
    or_v = old_rank[g_local]; ir_v = ig_rank[g_local]
    g_name = gene_map.iloc[g_global]['gene_name']
    ax3.scatter([or_v], [ir_v], s=70, color='#e15759', edgecolor='black',
                 linewidth=1, zorder=3, marker='s')
    ax3.annotate(g_name, (or_v, ir_v), xytext=(6, 4), textcoords='offset points',
                 fontsize=8, color='#e15759', fontweight='bold')
# Identity line
mx = len(active_idx)
ax3.plot([0, mx], [0, mx], color='k', linestyle=':', alpha=0.4, linewidth=0.8)
ax3.set_xscale('log'); ax3.set_yscale('log')
ax3.set_xlim(0.9, mx); ax3.set_ylim(0.9, mx)
ax3.set_xlabel('Degree-Product Rank (low → high importance)', fontsize=11)
ax3.set_ylabel('IG Rank (low → high importance)', fontsize=11)
ax3.set_title('(c) IG vs Degree-Product Ranking', fontsize=11, loc='left')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

# Custom legend for panel (c)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b7dd8', markeredgecolor='black',
           markersize=8, label='IG Top-10'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#e15759', markeredgecolor='black',
           markersize=8, label='Degree Top-10 (not in IG Top-10)'),
]
ax3.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)

plt.tight_layout()
plt.savefig(FIG_OUT, format='pdf', bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {FIG_OUT}")

# Also save PNG preview
fig_png = FIG_OUT.replace('.pdf', '.png')
fig = plt.figure(figsize=(16, 5))
# (rerender quickly - same as above) easiest: re-load and re-render or just convert
