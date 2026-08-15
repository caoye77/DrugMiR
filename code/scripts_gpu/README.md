# DrugMiR GPU Experiments — Run Order

Paper sections this directory fills in:
- §4.7 Multi-task Prediction of Resistance and Sensitivity
- §4.8 Embedding Visualization (t-SNE)
- §4.9 Robustness and Convergence Analysis

## File layout

```
00_env_check.py             # run first — confirms env is ready
drugmir_model.py            # shared model (DrugMiR_Hybrid, DrugMiREncoder, …)
01_train_multitask.py       # §4.7
02_tsne_visualization.py    # §4.8
03_robustness_convergence.py# §4.9
```

## Setup

Assumes the data directory `data/processed/` is reachable. Override with:

```bash
export DRUGMIR_DATA=/path/to/data/processed
```

## Run order

```bash
# 1. Confirm environment
python 00_env_check.py
# If anything fails, fix before proceeding. Most common fix:
#   pip install torch_geometric torch_scatter \
#     -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 2. Multi-task (§4.7) — ~30–40 min on RTX 4090
python 01_train_multitask.py

# 3. t-SNE visualization (§4.8) — ~15 min
python 02_tsne_visualization.py

# 4. Robustness + convergence (§4.9) — ~60–90 min
python 03_robustness_convergence.py
```

## Outputs

```
results_multitask/
    multitask_log.txt
    multitask_hybrid_results.json     # plug into Table VI
results_tsne/
    embeddings.npz                    # 6 (N, 256) arrays
    tsne_2d.npz
    fig_tsne.pdf / .png               # plug into Fig. 7
results_robustness/
    convergence.json
    fig_convergence.pdf / .png        # plug into Fig. 8a
    robustness.json
    fig_robustness.pdf / .png         # plug into Fig. 8b
```

## Tips

- All scripts pin `seed=42` to match the paper's protocol.
- If you only have time for one, do `01_train_multitask.py` first — it's
  the most paper-critical (currently red placeholder text in §4.7).
- After running, copy the `*.json` and figures back to your local laptop:
  ```bash
  scp -P <port> -r featurize@workspace.featurize.cn:~/DrugMiR/results_multitask .
  scp -P <port> -r featurize@workspace.featurize.cn:~/DrugMiR/results_tsne .
  scp -P <port> -r featurize@workspace.featurize.cn:~/DrugMiR/results_robustness .
  ```
- If a script crashes mid-run, just re-run it. None of them rely on
  resume-from-checkpoint logic; the experiments are short enough that a full
  re-run is faster than recovering partial state.
