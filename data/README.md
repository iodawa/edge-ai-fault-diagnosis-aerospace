# Data

Raw and intermediate data are **not committed** to this repository (see
`.gitignore`). This folder defines the expected local layout.

## `data/raw/`
Place the NASA N-CMAPSS HDF5 files here, unmodified as downloaded from the
NASA PCoE Data Repository ("Turbofan Engine Degradation Simulation-2",
2021). Expected layout:

```
data/raw/
└── N-CMAPSS_DS03-012.h5   # (and any other DS0x subsets you use)
```

## `data/interim/`
Cleaned/normalized/augmented intermediates produced by
`src/edge_fault_dx/data/preprocessing.py` — safe to delete and
regenerate.

## `data/processed/`
Final time-ordered train/validation/test splits consumed directly by
`src/edge_fault_dx/training/train.py`.

## Jetson-Bench logs
On-device latency/power logs generated during Phase 6 (benchmarking) are
written to `results/tables/jetson_bench/` rather than `data/`, since
they're a research output, not an input dataset.
