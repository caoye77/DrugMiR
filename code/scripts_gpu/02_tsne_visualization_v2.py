"""
DrugMiR t-SNE Visualization v2 — biology-aware coloring.

Drug coloring: 7-class pharmacological classification (manual mapping
based on known mechanism of action).

miRNA coloring: top-N largest miRNA families by family name prefix
(e.g. let-7, miR-15, miR-17, miR-21, …). Members of the same family share
seed sequence and typically regulate overlapping target sets, so a model
that has learned biological structure should cluster them together.

This script reuses the already-computed embeddings stored in
results_tsne/embeddings.npz — no retraining needed.
"""
import os, json, re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

DATA_DIR = os.environ.get("DRUGMIR_DATA", "./data/processed")
OUT_DIR  = "./results_tsne"
os.makedirs(OUT_DIR, exist_ok=True)


# ----- Drug pharmacological classification -----
# 156-drug manual mapping. Categories are picked to produce visually
# distinguishable groups while remaining biologically meaningful.
DRUG_CLASS = {
    # Platinum-based DNA crosslinkers
    "Cisplatin": "Platinum",
    "Carboplatin": "Platinum",
    "Oxaliplatin": "Platinum",

    # Anthracyclines / topoisomerase II inhibitors (DNA damage)
    "Doxorubicin": "Anthracycline/TopoII",
    "Epirubicin":  "Anthracycline/TopoII",
    "Anthracycline": "Anthracycline/TopoII",
    "Mitoxantrone":  "Anthracycline/TopoII",
    "Pirarubicin":   "Anthracycline/TopoII",
    "Etoposide":     "Anthracycline/TopoII",
    "Teniposide":    "Anthracycline/TopoII",

    # Antimetabolites (nucleoside / folate antagonists)
    "Gemcitabine":   "Antimetabolite",
    "Capecitabine":  "Antimetabolite",
    "Fluorouracil":  "Antimetabolite",
    "Cytarabine":    "Antimetabolite",
    "Methotrexate":  "Antimetabolite",
    "Pemetrexed":    "Antimetabolite",
    "Azacitidine":   "Antimetabolite",
    "Decitabine":    "Antimetabolite",
    "Fludarabine":   "Antimetabolite",
    "Hydroxyurea":   "Antimetabolite",
    "Raltitrexed":   "Antimetabolite",
    "Ribavirin":     "Antimetabolite",

    # Microtubule-targeting (taxanes + vinca alkaloids)
    "Docetaxel":   "Microtubule",
    "Taxol":       "Microtubule",
    "Cabazitaxel": "Microtubule",
    "Vincristine": "Microtubule",
    "Vinblastine": "Microtubule",
    "Vinorelbine": "Microtubule",
    "Eribulin":    "Microtubule",

    # Topoisomerase I inhibitors (camptothecin family)
    "Camptothecin":            "TopoI inhibitor",
    "10-Hydroxycamptothecin":  "TopoI inhibitor",
    "Irinotecan":              "TopoI inhibitor",
    "Topotecan":               "TopoI inhibitor",

    # Alkylating agents
    "Cyclophosphamide": "Alkylator",
    "Mafosfamide":      "Alkylator",
    "Temozolomide":     "Alkylator",
    "Dacarbazine":      "Alkylator",
    "Carmustine":       "Alkylator",
    "Melphalan":        "Alkylator",
    "Bleomycin":        "Alkylator",
    "Mitomycin C":      "Alkylator",

    # Hormonal / endocrine therapy
    "Tamoxifen":     "Hormonal",
    "Fulvestrant":   "Hormonal",
    "Anastrozole":   "Hormonal",
    "Letrozole":     "Hormonal",
    "Exemestane":    "Hormonal",
    "Estradiol":     "Hormonal",
    "Progesterone":  "Hormonal",
    "Stanolone":     "Hormonal",
    "Enzalutamide":  "Hormonal",
    "Prednisolone":  "Hormonal",
    "Prednisone":    "Hormonal",
    "Methylprednisolone": "Hormonal",
    "Dexamethasone": "Hormonal",
    "Cortisone":     "Hormonal",
    "Calcitriol":    "Hormonal",
    "Tacalcitol":    "Hormonal",
    "Tretinoin":     "Hormonal",

    # Kinase inhibitors (large class — most targeted therapies)
    "Imatinib":     "Kinase inhibitor",
    "Dasatinib":    "Kinase inhibitor",
    "Nilotinib":    "Kinase inhibitor",
    "Ponatinib":    "Kinase inhibitor",
    "Gefitinib":    "Kinase inhibitor",
    "Erlotinib":    "Kinase inhibitor",
    "Osimertinib":  "Kinase inhibitor",
    "Lapatinib":    "Kinase inhibitor",
    "Afatinib":     "Kinase inhibitor",
    "Icotinib":     "Kinase inhibitor",
    "Neratinib":    "Kinase inhibitor",
    "Sorafenib":    "Kinase inhibitor",
    "Sunitinib":    "Kinase inhibitor",
    "Regorafenib":  "Kinase inhibitor",
    "Pazopanib":    "Kinase inhibitor",
    "Axitinib":     "Kinase inhibitor",
    "Lenvatinib":   "Kinase inhibitor",
    "Cabozantinib": "Kinase inhibitor",
    "Vandetanib":   "Kinase inhibitor",
    "Anlotinib":    "Kinase inhibitor",
    "Nintedanib":   "Kinase inhibitor",
    "Dovitinib":    "Kinase inhibitor",
    "Vemurafenib":  "Kinase inhibitor",
    "Plx-4720":     "Kinase inhibitor",
    "Dabrafenib":   "Kinase inhibitor",
    "Trametinib":   "Kinase inhibitor",
    "Selumetinib":  "Kinase inhibitor",
    "Crizotinib":   "Kinase inhibitor",
    "Ceritinib":    "Kinase inhibitor",
    "Lorlatinib":   "Kinase inhibitor",
    "Tivantinib":   "Kinase inhibitor",
    "Ruxolitinib":  "Kinase inhibitor",
    "Ibrutinib":    "Kinase inhibitor",
    "Saracatinib":  "Kinase inhibitor",
    "Gilteritinib": "Kinase inhibitor",
    "Quizartinib":  "Kinase inhibitor",
    "Palbociclib":  "Kinase inhibitor",
    "Ribociclib":   "Kinase inhibitor",
    "Alpelisib":    "Kinase inhibitor",
    "Apitolisib":   "Kinase inhibitor",
    "Sirolimus":    "Kinase inhibitor",
    "Temsirolimus": "Kinase inhibitor",
    "Everolimus":   "Kinase inhibitor",
    "Osi-027":      "Kinase inhibitor",
    "Onvansertib":  "Kinase inhibitor",
    "Staurosporine":"Kinase inhibitor",

    # Other targeted (proteasome, BCL2, PARP, HSP90, HDAC, BET, etc.)
    "Bortezomib":      "Other targeted",
    "Carfilzomib":     "Other targeted",
    "Ixazomib":        "Other targeted",
    "Olaparib":        "Other targeted",
    "Veliparib":       "Other targeted",
    "Navitoclax":      "Other targeted",
    "Abt-737":         "Other targeted",
    "Vorinostat":      "Other targeted",
    "Panobinostat":    "Other targeted",
    "Trichostatin A":  "Other targeted",
    "Quisinostat":     "Other targeted",
    "Tanespimycin":    "Other targeted",
    "Jq1":             "Other targeted",
    "Nutlin-3":        "Other targeted",
    "Tipifarnib":      "Other targeted",
    "Lenalidomide":    "Other targeted",
    "Arsenic trioxide":"Other targeted",
    "Mitotane":        "Other targeted",
}

# Anything not in the dict above falls into "Other / non-cancer"
def get_drug_class(name):
    return DRUG_CLASS.get(name, "Other / non-cancer")


# ----- miRNA family parsing -----
# Examples: hsa-let-7a-5p -> "let-7"
#           hsa-miR-21-3p -> "miR-21"
#           hsa-miR-15a-5p -> "miR-15"   (group a/b/c variants together)
#           hsa-miR-200a-3p -> "miR-200"
def get_mirna_family(name):
    n = name.replace("hsa-", "")
    # let-7 family
    m = re.match(r"(let-\d+)", n)
    if m: return m.group(1)
    # miR-NNN family (group letter suffixes a/b/c/...)
    m = re.match(r"(miR-\d+)", n)
    if m: return m.group(1)
    return "other"


def main():
    print("=" * 60)
    print("DrugMiR t-SNE v2 — biology-aware coloring")
    print("=" * 60)

    # Load already-computed embeddings (no retraining)
    embs = np.load(os.path.join(OUT_DIR, "embeddings.npz"))
    print(f"Loaded embeddings: {list(embs.keys())}")

    # Load mappings
    drug_df  = pd.read_csv(os.path.join(DATA_DIR, "drug_mapping.csv"))
    mirna_df = pd.read_csv(os.path.join(DATA_DIR, "mirna_mapping.csv"))
    drug_classes  = drug_df["drug_name"].apply(get_drug_class).values
    mirna_fams    = mirna_df["mirna_name"].apply(get_mirna_family).values

    print(f"\nDrug class distribution:")
    for cls, cnt in Counter(drug_classes).most_common():
        print(f"  {cls:25s}: {cnt}")

    # Pick top-N miRNA families (must have >=15 members to be visually
    # meaningful); everything else lumped into "other"
    fam_counts = Counter(mirna_fams)
    top_fams = [f for f, c in fam_counts.most_common() if f != "other" and c >= 15]
    print(f"\nTop miRNA families (>=15 members):")
    for f in top_fams:
        print(f"  {f:12s}: {fam_counts[f]}")
    # Limit to first 8 for legend readability
    top_fams = top_fams[:8]
    mirna_grouped = np.array([f if f in top_fams else "other"
                               for f in mirna_fams])

    # t-SNE projections (re-compute for consistency)
    print("\nRunning t-SNE on each channel...")
    tsne_2d = {}
    for name in ["m0", "mh", "mb", "d0", "dh", "db"]:
        arr = embs[name]
        n = arr.shape[0]
        perp = 30 if n > 200 else max(5, int(np.sqrt(n)))
        proj = TSNE(n_components=2, perplexity=perp, init="pca",
                    random_state=42, max_iter=1000).fit_transform(arr)
        tsne_2d[name] = proj
        print(f"  {name}: ({arr.shape[0]} x {arr.shape[1]}) -> 2D, perp={perp}")
    np.savez(os.path.join(OUT_DIR, "tsne_2d.npz"), **tsne_2d)

    # ----- Plot: 2x3 grid, biology-aware coloring -----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 12, "axes.labelsize": 10.5,
        "legend.fontsize": 8.5, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Color palettes
    drug_classes_sorted = sorted(set(drug_classes),
        key=lambda x: -Counter(drug_classes)[x])
    drug_color_map = dict(zip(drug_classes_sorted,
        plt.cm.tab10(np.linspace(0, 1, len(drug_classes_sorted)))))

    mirna_fams_sorted = top_fams + ["other"]
    mirna_color_map = dict(zip(mirna_fams_sorted,
        plt.cm.tab10(np.linspace(0, 1, len(mirna_fams_sorted)))))
    # Make "other" pale grey to push background
    mirna_color_map["other"] = (0.75, 0.75, 0.75, 0.35)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        ("m0", "miRNA: Hybrid Embedding (Channel 1)", "mirna", axes[0, 0]),
        ("mh", "miRNA: Homo GCN (Channel 2)",         "mirna", axes[0, 1]),
        ("mb", "miRNA: Gene Bridge (Channel 3)",      "mirna", axes[0, 2]),
        ("d0", "Drug: Hybrid Embedding (Channel 1)",  "drug",  axes[1, 0]),
        ("dh", "Drug: Homo GCN (Channel 2)",          "drug",  axes[1, 1]),
        ("db", "Drug: Gene Bridge (Channel 3)",       "drug",  axes[1, 2]),
    ]

    for key, title, kind, ax in panels:
        proj = tsne_2d[key]
        if kind == "mirna":
            # plot "other" first so colored families sit on top
            for fam in ["other"] + top_fams:
                mask = mirna_grouped == fam
                if not mask.any(): continue
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=[mirna_color_map[fam]], s=18 if fam != "other" else 8,
                           alpha=0.85 if fam != "other" else 0.35,
                           edgecolors="white" if fam != "other" else "none",
                           linewidths=0.3, label=fam if fam != "other" else None)
        else:  # drug
            for cls in drug_classes_sorted:
                mask = drug_classes == cls
                if not mask.any(): continue
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=[drug_color_map[cls]], s=55, alpha=0.85,
                           edgecolors="white", linewidths=0.5, label=cls)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    # Legends — one for miRNA (top row), one for drug (bottom row)
    handles_m = [plt.Line2D([], [], marker="o", linestyle="",
                            markerfacecolor=mirna_color_map[f],
                            markeredgecolor="white", markeredgewidth=0.4,
                            markersize=8, label=f)
                 for f in top_fams]
    axes[0, 2].legend(handles=handles_m, loc="center left",
                      bbox_to_anchor=(1.02, 0.5), title="miRNA family",
                      fontsize=8.5)

    handles_d = [plt.Line2D([], [], marker="o", linestyle="",
                            markerfacecolor=drug_color_map[c],
                            markeredgecolor="white", markeredgewidth=0.4,
                            markersize=8, label=c)
                 for c in drug_classes_sorted]
    axes[1, 2].legend(handles=handles_d, loc="center left",
                      bbox_to_anchor=(1.02, 0.5), title="Drug class",
                      fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_tsne.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "fig_tsne.png"), dpi=200)
    print(f"\nSaved fig_tsne.pdf and .png")

    # Save labels for downstream verification
    np.savez(os.path.join(OUT_DIR, "labels.npz"),
             drug_classes=drug_classes, mirna_families=mirna_grouped)

    print("\nDone.")


if __name__ == "__main__":
    main()
