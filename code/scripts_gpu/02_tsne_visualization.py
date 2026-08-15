"""
DrugMiR t-SNE Embedding Visualization (paper §4.8).

Trains the standard single-task DrugMiR_Hybrid on Dataset 1 with seed=42,
then dumps the six per-entity representations (m0/mh/mb/d0/dh/db) and
projects each to 2D with t-SNE for plotting.

Outputs
-------
results_tsne/embeddings.npz
    six (N, 256) arrays + miRNA / drug indices
results_tsne/tsne_2d.npz
    six (N, 2) t-SNE projections
results_tsne/fig_tsne.pdf / .png
    Six-panel figure showing miRNA and drug embeddings per channel.

Notes
-----
This script trains on the FULL Dataset 1 (no held-out fold) because the
purpose is to inspect representation geometry on the entire vocabulary,
not to evaluate prediction performance.
"""
import os, json, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from drugmir_model import (
    load_data, sample_neg, set_seed, device, DrugMiR_Hybrid,
)

DATA_DIR = os.environ.get("DRUGMIR_DATA", "./data/processed")
OUT_DIR  = "./results_tsne"
os.makedirs(OUT_DIR, exist_ok=True)


def train_full(model, data, optimizer, n_epochs=200, batch_size=2048):
    """Train on every positive pair (no fold split) for embedding extraction."""
    pos = data["pos_pairs"]
    for ep in range(n_epochs):
        model.train()
        neg = sample_neg(data["assoc"], pos, len(pos))
        pairs = pos + neg
        labels = [1.0] * len(pos) + [0.0] * len(neg)
        idx = np.random.permutation(len(pairs))
        losses = []
        for s in range(0, len(idx), batch_size):
            bi = idx[s:s + batch_size]
            bp = [pairs[i] for i in bi]
            bl = torch.FloatTensor([labels[i] for i in bi]).to(device)
            mi = torch.LongTensor([p[0] for p in bp]).to(device)
            di = torch.LongTensor([p[1] for p in bp]).to(device)
            optimizer.zero_grad()
            out = model(data, mi, di)
            loss = F.binary_cross_entropy_with_logits(out, bl)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep + 1}/{n_epochs}  loss={np.mean(losses):.4f}",
                  flush=True)


def main():
    print("=" * 60)
    print("DrugMiR t-SNE Embedding Visualization (§4.8)")
    print("=" * 60)
    print(f"Data dir: {DATA_DIR}")
    print(f"Device:   {device}")

    set_seed(42)
    data = load_data(DATA_DIR, km=15, kd=10)
    md = data["mirna_feat"].shape[1]; dd = data["drug_feat"].shape[1]
    nm, nd, ng = data["n_mirna"], data["n_drug"], data["n_gene"]
    print(f"  miRNAs={nm}, drugs={nd}, genes={ng}, "
          f"positives={len(data['pos_pairs'])}\n")

    # Train
    model = DrugMiR_Hybrid(nm, nd, md, dd, ng,
                           h=256, dr=0.5, n_gcn=2, n_br=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=2e-4)
    print("Training on full Dataset 1 (200 epochs)...")
    t0 = time.time()
    train_full(model, data, opt, n_epochs=200)
    print(f"Trained in {time.time() - t0:.0f}s")

    # Extract embeddings
    print("\nExtracting embeddings...")
    model.eval()
    with torch.no_grad():
        m0, mh, mb, d0, dh, db = model.enc.encode(data)
    embs = dict(
        m0=m0.cpu().numpy(), mh=mh.cpu().numpy(), mb=mb.cpu().numpy(),
        d0=d0.cpu().numpy(), dh=dh.cpu().numpy(), db=db.cpu().numpy(),
    )
    np.savez(os.path.join(OUT_DIR, "embeddings.npz"), **embs)
    print("  Saved embeddings.npz")
    for name, arr in embs.items():
        print(f"    {name}: shape {arr.shape}")

    # Color labels: degree (number of associations) for each entity, used to
    # color the points so that highly-connected entities pop visually
    m_deg = data["assoc"].sum(axis=1)
    d_deg = data["assoc"].sum(axis=0)

    # t-SNE
    print("\nRunning t-SNE on each channel...")
    tsne_2d = {}
    for name, arr in embs.items():
        is_drug = name.startswith("d")
        n = arr.shape[0]
        # perplexity guidelines: 5..50, scale by sqrt(N)
        perp = 30 if n > 200 else max(5, int(np.sqrt(n)))
        t0 = time.time()
        proj = TSNE(n_components=2, perplexity=perp, init="pca",
                    random_state=42, max_iter=1000).fit_transform(arr)
        tsne_2d[name] = proj
        print(f"  {name}: ({arr.shape[0]} x {arr.shape[1]}) -> 2D, "
              f"perp={perp}, {time.time() - t0:.0f}s")
    np.savez(os.path.join(OUT_DIR, "tsne_2d.npz"), **tsne_2d)

    # Plot 6-panel figure
    print("\nDrawing figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
    panels = [
        ("m0", "miRNA: Hybrid Embedding (Channel 1)", m_deg, axes[0, 0]),
        ("mh", "miRNA: Homo GCN (Channel 2)",          m_deg, axes[0, 1]),
        ("mb", "miRNA: Gene Bridge (Channel 3)",       m_deg, axes[0, 2]),
        ("d0", "Drug: Hybrid Embedding (Channel 1)",   d_deg, axes[1, 0]),
        ("dh", "Drug: Homo GCN (Channel 2)",           d_deg, axes[1, 1]),
        ("db", "Drug: Gene Bridge (Channel 3)",        d_deg, axes[1, 2]),
    ]
    for key, title, deg, ax in panels:
        proj = tsne_2d[key]
        sc = ax.scatter(proj[:, 0], proj[:, 1],
                        c=np.log1p(deg), cmap="viridis",
                        s=10, alpha=0.7, edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                     label="log(1 + degree)")
    plt.tight_layout()
    pdf = os.path.join(OUT_DIR, "fig_tsne.pdf")
    png = os.path.join(OUT_DIR, "fig_tsne.png")
    plt.savefig(pdf); plt.savefig(png, dpi=200)
    print(f"  Saved {pdf}\n  Saved {png}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
