# System / Pipeline Architecture

This document connects the repository's code layout to the six-phase
research methodology in
[`docs/praxis/06-methodology-graphical-model.md`](praxis/06-methodology-graphical-model.md).

```
 N-CMAPSS (.h5)                     Jetson Nano
      │                                  ▲
      ▼                                  │
 src/edge_fault_dx/data/            deployment/jetson/
  loaders.py, preprocessing.py       tegrastats_logger.py
      │                                  ▲
      ▼                                  │
 src/edge_fault_dx/models/          src/edge_fault_dx/deployment/
  ma1dcnn.py  ──train.py──►          export_onnx.py → tensorrt_build.py
  (workstation baseline,                              → inference_server.py
   Optuna-tuned)                          ▲
      │                                   │
      ▼                                   │
 src/edge_fault_dx/compression/           │
  pruning.py, quantization.py  ───────────┘
  (→ eo_ma1dcnn.py)
      │
      ▼
 src/edge_fault_dx/evaluation/
  metrics.py (accuracy, macro-precision, recall, F1, ROC-AUC)
  noise_robustness.py (SNR sweep)
```

| Pipeline stage | Code | Notebook | Config |
|---|---|---|---|
| 1. Collect data / set up hardware | `src/edge_fault_dx/data/loaders.py` | `notebooks/01_eda.ipynb` | — |
| 2. Preprocess / augment | `src/edge_fault_dx/data/preprocessing.py` | `notebooks/01_eda.ipynb` | `configs/baseline.yaml` |
| 3. Train baseline MA1DCNN | `src/edge_fault_dx/models/ma1dcnn.py`, `training/train.py`, `training/optuna_search.py` | `notebooks/02_baseline_training.ipynb`, `03_optuna_tuning.ipynb` | `configs/baseline.yaml`, `configs/optuna.yaml` |
| 4. Compress (prune + quantize) | `src/edge_fault_dx/compression/pruning.py`, `quantization.py` | `notebooks/04_compression_pruning_quant.ipynb` | `configs/compression.yaml` |
| 5. Evaluate vs. hypotheses | `src/edge_fault_dx/evaluation/metrics.py`, `noise_robustness.py` | `notebooks/04_compression_pruning_quant.ipynb` | — |
| 6. Deploy / benchmark on Jetson | `src/edge_fault_dx/deployment/*`, `deployment/jetson/tegrastats_logger.py` | `notebooks/05_jetson_benchmarking.ipynb` | `configs/jetson_deploy.yaml` |

Results from stage 5–6 land in `results/tables/` and `results/figures/`
and are summarized in the performance report deliverable
(`results/reports/`).
