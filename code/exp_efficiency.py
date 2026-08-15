#!/usr/bin/env python
"""
DrugMiR - Experiment (5): efficiency / "lightweight" evidence.

The paper calls the Gene Bridge "lightweight" in four places (abstract, Need 2,
contribution 2, conclusion) with nothing behind it but the architectural
argument that scatter-mean avoids metapath enumeration. This produces the
numbers.

Three parts, and the first two need no GPU at all:

  1. PARAMETER BUDGET  (CPU, instant)
     Total parameters and the per-channel split, so "the Gene Bridge is
     lightweight" becomes a number instead of an adjective.

  2. ENUMERATION COST  (CPU, seconds)
     The real complexity claim, measured directly off the gene matrices and
     without ever running MPHGNN:
         scatter-mean  visits |E_mg| + |E_dg| gene edges
         metapath      must instantiate every miRNA-gene-drug path, i.e.
                       sum over genes of  deg_miRNA(g) * deg_drug(g)
     The ratio between the two is the whole argument, quantified.

  3. WALL-CLOCK + MEMORY  (needs GPU to be meaningful)
     Per-epoch training time, full-inference time, peak VRAM.

IMPORTANT for part 3: time every method inside ONE instance session. Timings
taken on different machines are not comparable, and a reviewer who spots
mixed hardware will discount the whole table.

Usage
-----
    python exp_efficiency.py --dataset D1 --parts 12      # CPU: params + paths
    python exp_efficiency.py --dataset D1                 # all three (GPU)
    python exp_efficiency.py --dataset D1 --epochs 5

Output
------
    results_efficiency/efficiency_<ds>.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/work/DrugMiR'))
import exp_compat  # noqa: F401,E402
from hp_finetune import (load_data, trn, DrugMiR_Hybrid, device)  # noqa: E402

DATA_DIRS = {
    'D1': os.path.expanduser('~/work/DrugMiR/data/processed'),
    'D2': os.path.expanduser('~/work/DrugMiR/DMGAT_processed'),
}
OUT_DIR = os.path.expanduser('~/work/DrugMiR/results_efficiency')


def count(module):
    return sum(p.numel() for p in module.parameters())


def param_budget(model):
    ch1 = count(model.me) + count(model.de)
    ch2 = count(model.mgcn) + count(model.dgcn)
    ch3 = count(model.br) + count(model.ge)
    ch3_no_emb = count(model.br)
    head = count(model.pred)
    total = count(model)
    return {
        'total': total,
        'channel1_hybrid_encoder': ch1,
        'channel2_homo_gcn': ch2,
        'channel3_gene_bridge_incl_emb': ch3,
        'channel3_gene_bridge_layers_only': ch3_no_emb,
        'gene_embedding_table': count(model.ge),
        'prediction_head': head,
        'gene_bridge_pct_of_total': round(100.0 * ch3_no_emb / total, 2),
        'accounted': ch1 + ch2 + ch3 + head,
    }


def enumeration_cost(data_dir):
    """scatter-mean edge count vs the number of metapath instances a
    metapath-based model (e.g. MPHGNN) would have to materialise."""
    mg = np.load(f'{data_dir}/mirna_gene_matrix.npy')
    dg = np.load(f'{data_dir}/drug_gene_matrix.npy')
    n_g = max(mg.shape[1], dg.shape[1])
    m_deg = np.pad(mg.sum(0), (0, n_g - mg.shape[1]))
    d_deg = np.pad(dg.sum(0), (0, n_g - dg.shape[1]))

    e_mg = int((mg > 0).sum())
    e_dg = int((dg > 0).sum())
    scatter_ops = e_mg + e_dg
    # miRNA -> gene <- drug instances, per gene: deg_m * deg_d
    per_gene = m_deg * d_deg
    metapath_len3 = int(per_gene.sum())
    # length-4 miRNA-gene-drug-gene style paths grow with a second hop
    metapath_len4 = int((per_gene * np.maximum(d_deg, 1)).sum())

    return {
        'n_genes': int(n_g),
        'edges_mirna_gene': e_mg,
        'edges_drug_gene': e_dg,
        'scatter_mean_ops': scatter_ops,
        'metapath_instances_MGD': metapath_len3,
        'metapath_instances_MGDG': metapath_len4,
        'ratio_MGD_over_scatter': round(metapath_len3 / max(scatter_ops, 1), 1),
        'ratio_MGDG_over_scatter': round(metapath_len4 / max(scatter_ops, 1), 1),
        'bridge_genes_active': int(((m_deg > 0) & (d_deg > 0)).sum()),
        'max_per_gene_instances': int(per_gene.max()),
    }


def timing(data, model_fn, epochs=5, lr=1e-3, wd=2e-4):
    pos = data['pos_pairs']
    n_tr = int(0.8 * len(pos))
    trp = pos[:n_tr]
    model = model_fn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    trn(model, data, trp, opt)                       # warm-up, not timed
    if on_cuda:
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(epochs):
        trn(model, data, trp, opt)
    if on_cuda:
        torch.cuda.synchronize()
    per_epoch = (time.time() - t0) / epochs

    model.eval()
    nm, nd = data['n_mirna'], data['n_drug']
    mi = torch.LongTensor(np.repeat(np.arange(nm), nd)).to(device)
    di = torch.LongTensor(np.tile(np.arange(nd), nm)).to(device)
    if on_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        for s in range(0, len(mi), 65536):
            model(data, mi[s:s + 65536], di[s:s + 65536])
    if on_cuda:
        torch.cuda.synchronize()
    infer = time.time() - t1

    out = {
        'sec_per_epoch': round(per_epoch, 3),
        'epochs_timed': epochs,
        'full_inference_sec': round(infer, 3),
        'full_inference_pairs': int(nm * nd),
        'device': str(device),
    }
    if on_cuda:
        out['peak_vram_mb'] = round(
            torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        out['gpu_name'] = torch.cuda.get_device_name(0)
    else:
        out['note'] = 'CPU run - timings are NOT comparable to GPU numbers'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='D1', choices=['D1', 'D2'])
    ap.add_argument('--parts', default='123',
                    help='which parts to run, e.g. 12 for CPU-only')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--km', type=int, default=15)
    ap.add_argument('--kd', type=int, default=10)
    ap.add_argument('--h', type=int, default=256)
    ap.add_argument('--dr', type=float, default=0.5)
    ap.add_argument('--n_gcn', type=int, default=2)
    ap.add_argument('--n_br', type=int, default=2)
    ap.add_argument('--out_dir', default=OUT_DIR)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    dd_path = DATA_DIRS[a.dataset]
    report = {'dataset': a.dataset, 'config': vars(a)}

    print('=' * 68)
    print(f'Efficiency report - {a.dataset}   (device={device})')
    print('=' * 68, flush=True)

    data = load_data(dd_path, km=a.km, kd=a.kd)
    md = data['mirna_feat'].shape[1]
    dd = data['drug_feat'].shape[1]
    nm, nd, ng = data['n_mirna'], data['n_drug'], data['n_gene']

    def mk():
        return DrugMiR_Hybrid(nm, nd, md, dd, ng, h=a.h, dr=a.dr,
                              n_gcn=a.n_gcn, n_br=a.n_br)

    if '1' in a.parts:
        pb = param_budget(mk())
        report['parameters'] = pb
        print('\n[1] PARAMETER BUDGET')
        print(f'  {"total":<38}{pb["total"]:>12,}')
        for k in ('channel1_hybrid_encoder', 'channel2_homo_gcn',
                  'channel3_gene_bridge_layers_only', 'gene_embedding_table',
                  'prediction_head'):
            print(f'  {k:<38}{pb[k]:>12,}')
        print(f'\n  Gene Bridge layers are {pb["gene_bridge_pct_of_total"]}% '
              f'of all parameters.')

    if '2' in a.parts:
        ec = enumeration_cost(dd_path)
        report['enumeration'] = ec
        print('\n[2] ENUMERATION COST  (the actual "lightweight" claim)')
        print(f'  genes                          {ec["n_genes"]:>12,}')
        print(f'  active bridge genes            '
              f'{ec["bridge_genes_active"]:>12,}')
        print(f'  miRNA-gene edges               '
              f'{ec["edges_mirna_gene"]:>12,}')
        print(f'  drug-gene edges                {ec["edges_drug_gene"]:>12,}')
        print(f'  scatter-mean ops per layer     '
              f'{ec["scatter_mean_ops"]:>12,}')
        print(f'  metapath M-G-D instances       '
              f'{ec["metapath_instances_MGD"]:>12,}')
        print(f'  metapath M-G-D-G instances     '
              f'{ec["metapath_instances_MGDG"]:>12,}')
        print(f'\n  -> metapath enumeration costs '
              f'{ec["ratio_MGD_over_scatter"]}x (M-G-D) / '
              f'{ec["ratio_MGDG_over_scatter"]}x (M-G-D-G) more than '
              f'scatter-mean.')

    if '3' in a.parts:
        print(f'\n[3] WALL-CLOCK + MEMORY  (timing {a.epochs} epochs)',
              flush=True)
        tm = timing(data, mk, epochs=a.epochs)
        report['timing'] = tm
        for k, v in tm.items():
            print(f'  {k:<38}{v}')
        if not torch.cuda.is_available():
            print('\n  !! CPU run: report these only as relative numbers, '
                  'and redo on GPU\n     with every baseline in the SAME '
                  'session before putting them in the paper.')

    out = os.path.join(a.out_dir, f'efficiency_{a.dataset}.json')
    json.dump(report, open(out, 'w'), indent=2)
    print(f'\n  saved: {out}')


if __name__ == '__main__':
    main()
