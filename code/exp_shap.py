#!/usr/bin/env python
"""
DrugMiR - Experiment (4): GradientSHAP attribution, cross-checked against IG.

The supervisor and his labmate both suggested adding SHAP alongside the
existing Integrated Gradients analysis ("keep IG, just add one").

Why this is hand-rolled instead of `pip install captum`
------------------------------------------------------
Installing captum drags in torch 2.13 + the whole CUDA 13 stack (3.5 GB) and
shadows the environment's torch, which breaks torch_geometric's GCNConv.
More importantly: GradientSHAP is IG with the fixed step alpha = k/N replaced
by a random alpha plus a Gaussian-smoothed baseline. Reusing the existing IG
loop means both methods share the same baseline, the same batching and the
same |.|-sum aggregation -- so the agreement between them is a real result,
not an artefact of two libraries doing two different things.

    IG           alpha = k / n_steps,   x = alpha * E
    GradientSHAP alpha ~ U(0,1),        x = alpha * (E + sigma * noise)

What it reports
---------------
    * per-gene GradientSHAP importance (same shape as gene_ig_importance_d1.npy)
    * Top-20 bridge genes by SHAP
    * Spearman correlation between IG and SHAP rankings over active genes
    * Top-k overlap (k = 10, 20, 50) and the consensus gene list

Usage
-----
    python exp_shap.py --smoke              # 2 samples, 1 batch - crash check
    python exp_shap.py                      # full run
    python exp_shap.py --n_samples 100 --sigma 0.15

Output
------
    results_final/gene_shap_importance_d1.npy
    results_final/bridge_genes_shap_top20.csv
    results_final/ig_vs_shap_agreement.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
import exp_compat  # noqa: F401,E402
from hp_finetune import load_data, DrugMiR_Hybrid, device       # noqa: E402


class DrugMiR_Attrib(DrugMiR_Hybrid):
    """DrugMiR_Hybrid with a gene-embedding override hook.

    Identical forward pass; the only change is that the gene embedding can be
    replaced by an external tensor so gradients flow back to it. Same trick
    compute_ig_bridge.py uses.
    """

    def forward(s, data, mi, di, gene_emb_override=None):
        m0 = s.me(data['mirna_feat'], data.get('mirna_has_feat'))
        d0 = s.de(data['drug_feat'], data.get('drug_has_feat'))
        mh = m0
        dh = d0
        for l in s.mgcn:
            mh = l(mh, data['mirna_sim_edge'])
        for l in s.dgcn:
            dh = l(dh, data['drug_sim_edge'])
        mb = m0
        db = d0
        gh = s.ge.weight if gene_emb_override is None else gene_emb_override
        for l in s.br:
            mb, db, gh = l(mb, db, gh, data['mg_src'], data['mg_dst'],
                           data['dg_src'], data['dg_dst'], data['n_gene'])
        return s.pred(torch.cat([torch.cat([m0, mh, mb], -1)[mi],
                                 torch.cat([d0, dh, db], -1)[di]],
                                -1)).squeeze(-1)


def gradient_shap_on_ge(model, data, pair_batches, n_samples=50, sigma=0.1,
                        seed=42, verbose=True):
    """GradientSHAP over the gene embedding.

    Expected gradient with a random scaling alpha ~ U(0,1) and a Gaussian
    perturbation of the baseline, then multiplied by the input as in IG.
    """
    model.eval()
    ge = model.ge.weight.detach().clone()
    acc = torch.zeros_like(ge)
    scale = float(ge.std().item()) * sigma
    g = torch.Generator(device='cpu').manual_seed(seed)
    n_b = len(pair_batches)
    for bi, (mi, di) in enumerate(pair_batches):
        for _ in range(n_samples):
            alpha = float(torch.rand(1, generator=g).item())
            noise = torch.randn(ge.shape, generator=g).to(ge.device) * scale
            x = (alpha * (ge + noise)).clone().detach().requires_grad_(True)
            score = torch.sigmoid(model(data, mi, di,
                                        gene_emb_override=x)).sum()
            acc += torch.autograd.grad(score, x, retain_graph=False)[0]
        if verbose and (bi + 1) % max(1, n_b // 10) == 0:
            print(f'    batch {bi + 1}/{n_b} done', flush=True)
    return ge * (acc / n_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',
                    default=os.path.expanduser('~/work/DrugMiR/data/processed'),
                    help='must match what compute_ig_bridge.py used')
    ap.add_argument('--ckpt', default=os.path.expanduser(
        '~/work/DrugMiR/results_final/drugmir_d1_seed42_for_ig.pt'))
    ap.add_argument('--out_dir', default=os.path.expanduser(
        '~/work/DrugMiR/results_final'))
    ap.add_argument('--ig_npy', default=None,
                    help='default: <out_dir>/gene_ig_importance_d1.npy')
    ap.add_argument('--gene_map',
                    default='data/processed/gene_mapping.csv')
    ap.add_argument('--n_samples', type=int, default=50)
    ap.add_argument('--sigma', type=float, default=0.1)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    ig_npy = a.ig_npy or os.path.join(a.out_dir, 'gene_ig_importance_d1.npy')

    print('=' * 68)
    print('GradientSHAP on gene embedding (D1, seed=42)')
    print(f'  device={device}  n_samples={a.n_samples}  sigma={a.sigma}')
    print('=' * 68, flush=True)

    t0 = time.time()
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    shapes, cfg = ckpt['shapes'], ckpt['config']
    print(f'  checkpoint val AUC = {ckpt["best_val_auc"]:.4f}')
    print(f'  shapes = {shapes}', flush=True)

    data = load_data(a.data_dir, km=cfg['km'], kd=cfg['kd'])
    got = (data['n_mirna'], data['n_drug'], data['n_gene'])
    want = (shapes['nm'], shapes['nd'], shapes['ng'])
    if got != want:
        sys.exit(f'ABORT: --data_dir {a.data_dir} has (nm,nd,ng)={got} but the checkpoint expects {want}. Wrong dataset.')
    print(f'  data/checkpoint shapes match: {got}')
    model = DrugMiR_Attrib(shapes['nm'], shapes['nd'], shapes['md'],
                           shapes['dd'], shapes['ng'], h=cfg['h'],
                           dr=cfg['dr'], n_gcn=cfg['n_gcn'],
                           n_br=cfg['n_br']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    pos = data['pos_pairs']
    batches = []
    for s in range(0, len(pos), a.batch_size):
        sub = pos[s:s + a.batch_size]
        batches.append((
            torch.LongTensor([p[0] for p in sub]).to(device),
            torch.LongTensor([p[1] for p in sub]).to(device)))
    if a.smoke:
        batches = batches[:1]
        a.n_samples = 2
        print('  *** SMOKE MODE - 1 batch, 2 samples, values meaningless')
    print(f'  {len(pos)} positive pairs in {len(batches)} batches\n',
          flush=True)

    t1 = time.time()
    shap = gradient_shap_on_ge(model, data, batches, n_samples=a.n_samples,
                               sigma=a.sigma, seed=a.seed)
    print(f'  GradientSHAP done in {time.time() - t1:.0f}s', flush=True)

    shap_imp = shap.abs().sum(-1).cpu().numpy()
    np.save(os.path.join(a.out_dir, 'gene_shap_importance_d1.npy'), shap_imp)

    # ---- active bridge genes: reachable from both sides ----
    mg = np.load(f'{a.data_dir}/mirna_gene_matrix.npy')
    dg = np.load(f'{a.data_dir}/drug_gene_matrix.npy')
    m_deg, d_deg = mg.sum(0), dg.sum(0)
    n = len(shap_imp)
    m_deg = np.pad(m_deg, (0, max(0, n - len(m_deg))))[:n]
    d_deg = np.pad(d_deg, (0, max(0, n - len(d_deg))))[:n]
    active = np.where((m_deg > 0) & (d_deg > 0))[0]
    print(f'  active bridge genes: {len(active)}')

    gene_map = pd.read_csv(a.gene_map)
    order = active[np.argsort(-shap_imp[active])]

    rows = [dict(rank_shap=i + 1, gene_idx=int(gi),
                 gene_name=gene_map.iloc[gi]['gene_name'],
                 shap_importance=float(shap_imp[gi]),
                 mirna_deg=int(m_deg[gi]), drug_deg=int(d_deg[gi]))
            for i, gi in enumerate(order[:20])]
    pd.DataFrame(rows).to_csv(
        os.path.join(a.out_dir, 'bridge_genes_shap_top20.csv'), index=False)

    print('\n  TOP-20 BRIDGE GENES BY GradientSHAP')
    print(f'  {"rank":<6}{"gene":<14}{"SHAP":>12}{"IG rank":>10}')
    print('  ' + '-' * 44)

    agree = {}
    if os.path.exists(ig_npy):
        ig_imp = np.load(ig_npy)[:n]
        ig_order = active[np.argsort(-ig_imp[active])]
        ig_rank = {int(g): r + 1 for r, g in enumerate(ig_order)}
        for r in rows:
            print(f'  {r["rank_shap"]:<6}{r["gene_name"]:<14}'
                  f'{r["shap_importance"]:>12.4f}'
                  f'{ig_rank.get(r["gene_idx"], "-"):>10}')
        rho, pval = spearmanr(ig_imp[active], shap_imp[active])
        agree = {'spearman_rho': float(rho), 'spearman_p': float(pval),
                 'n_active_genes': int(len(active))}
        for k in (10, 20, 50):
            si = set(order[:k].tolist())
            gi = set(ig_order[:k].tolist())
            inter = sorted(si & gi)
            agree[f'top{k}_overlap'] = len(inter)
            agree[f'top{k}_consensus_genes'] = [
                str(gene_map.iloc[x]['gene_name']) for x in inter]
        print(f'\n  IG vs SHAP Spearman rho = {rho:.4f}  (p = {pval:.2e})')
        for k in (10, 20, 50):
            print(f'  Top-{k} overlap: {agree[f"top{k}_overlap"]}/{k}')
        print(f'  Top-20 consensus: '
              f'{", ".join(agree["top20_consensus_genes"][:12])}'
              f'{" ..." if agree["top20_overlap"] > 12 else ""}')
    else:
        for r in rows:
            print(f'  {r["rank_shap"]:<6}{r["gene_name"]:<14}'
                  f'{r["shap_importance"]:>12.4f}{"n/a":>10}')
        print(f'\n  !! IG file not found at {ig_npy} - skipped the comparison')

    meta = dict(n_samples=a.n_samples, sigma=a.sigma, seed=a.seed,
                batch_size=a.batch_size, n_pairs=len(pos),
                smoke=bool(a.smoke), runtime_s=round(time.time() - t0),
                **agree)
    json.dump(meta, open(os.path.join(a.out_dir,
                                      'ig_vs_shap_agreement.json'), 'w'),
              indent=2)

    print(f'\n  saved to {a.out_dir}:')
    print('    gene_shap_importance_d1.npy')
    print('    bridge_genes_shap_top20.csv')
    print('    ig_vs_shap_agreement.json')
    print(f'  total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
