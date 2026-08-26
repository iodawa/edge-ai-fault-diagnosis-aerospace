# Notebooks

Exploratory / interactive companions to `src/edge_fault_dx/`, numbered to
match the six methodology phases:

| Notebook | Phase |
|---|---|
| `01_eda.ipynb` | 1–2: data collection, EDA, preprocessing |
| `02_baseline_training.ipynb` | 3: workstation baseline MA1DCNN |
| `03_optuna_tuning.ipynb` | 3: Optuna hyperparameter search |
| `04_compression_pruning_quant.ipynb` | 4–5: pruning, INT8 quantization, evaluation |
| `05_jetson_benchmarking.ipynb` | 6: on-device latency/power benchmarking |

Notebooks are for exploration only — reusable logic belongs in
`src/edge_fault_dx/` so it's tested and importable from `scripts/`.
