# DrugMiR

Source code and data for **"DrugMiR: A Multi-Source Heterogeneous Graph Neural Network with Adaptive Feature-Embedding Fusion for Predicting miRNA-Mediated Drug Resistance and Sensitivity"**.

DrugMiR predicts miRNA-mediated drug resistance and sensitivity from a heterogeneous graph of miRNAs, drugs and their shared target genes. It pairs a three-channel encoder with a direction-aware prediction head, and is designed for the setting in which sequence or structural annotations are missing for part of the entities.

## Repository layout

```
code/                    model, experiments and figure scripts
  src/                   core model definition
  scripts_gpu/           experiment harnesses
  load_matrix.py         loader for the released gene-interaction matrices
data/
  dataset1/              D1, curated from ncRNADrug
  dataset2/              D2, curated from NoncoRNA and ncDR
results_final/           main comparison, case study, attribution, efficiency
ablation_outputs/        channel ablation, five seeds
coldstart_drugmir/       inductive (cold-start) results for DrugMiR
coldstart_baselines/     inductive (cold-start) results for the four baselines
```

## Datasets

|                        | Dataset 1 | Dataset 2 |
| ---------------------- | --------- | --------- |
| miRNAs                 | 1,578     | 622       |
| Drugs                  | 156       | 121       |
| Positive associations  | 8,720     | 2,686     |
| miRNA feature coverage | 100%      | 96.3%     |
| Drug feature coverage  | 100%      | 75.2%     |
| miRNA–gene edges       | 242,924   | 88,228    |
| Drug–gene edges        | 96,456    | 48,315    |

Each dataset directory contains:

| File                                        | Contents                                |
| ------------------------------------------- | --------------------------------------- |
| `association_matrix.npy`                    | binary miRNA × drug association matrix  |
| `resistance_matrix.npy`                     | resistance-only associations            |
| `sensitivity_matrix.npy`                    | sensitivity-only associations           |
| `mirna_kmer_features.npy`                   | 256-dimensional 4-mer frequency vectors |
| `drug_morgan_features.npy`                  | 1024-bit Morgan fingerprints            |
| `mirna_similarity.npy`, `drug_similarity.npy` | pairwise similarity matrices          |
| `mirna_gene_matrix.npz`, `drug_gene_matrix.npz` | gene-interaction matrices, scipy CSR |

The two gene-interaction matrices are around 1% non-zero and are released in sparse CSR format rather than as dense arrays. Load them with the provided helper:

```python
from code.load_matrix import load_matrix

A_mg = load_matrix("data/dataset1/mirna_gene_matrix")   # (1578, 14455), dense
A_dg = load_matrix("data/dataset1/drug_gene_matrix")    # (156, 14455), dense
```

Data were derived from the following public resources: ncRNADrug, NoncoRNA, ncDR, miRBase, DrugBank, miRTarBase and the Comparative Toxicogenomics Database. Please cite the original sources when reusing them.

## Requirements

Python 3.9 or higher, PyTorch 2.0 or higher, plus `numpy`, `scipy`, `scikit-learn`, `pandas` and `matplotlib`.

A CUDA-capable GPU is recommended but not required. The model holds 7.1M parameters and trains at roughly 0.23 s per epoch on a single GPU with a 925 MB peak memory footprint.

## Reproducing the results

| Experiment              | Script                           | Output directory       |
| ----------------------- | -------------------------------- | ---------------------- |
| Main comparison         | `run_multiseed.py`               | `results_final/`       |
| Significance analysis   | `sig_analysis.py`                | `results_final/`       |
| Cold-start evaluation   | `run_drugmir_coldstart.py`       | `coldstart_drugmir/`   |
| Cold-start baselines    | `run_mphgnn_coldstart_v4.py` and companions | `coldstart_baselines/` |
| Channel ablation        | `run_ablation_multiseed.py`      | `ablation_outputs/`    |
| Bridge-gene attribution | `compute_ig_bridge.py`           | `results_final/`       |
| Computational cost      | `exp_efficiency.py`              | `results_final/`       |
| ROC and PR curves       | `rerun_drugmir_save_preds_v2.py` | `results_final/`       |
| Fusion-gate analysis    | `gate_viz.py`                    | `results_final/`       |

The released result files correspond to the tables and figures reported in the article.

## License

MIT. See `LICENSE`.

## Citation

The citation will be added once the article is published.
