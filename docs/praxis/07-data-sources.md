# Data Sources
| ID | Name | Source | Description | Format | Time span | # records | Access |
|---|---|---|---|---|---|---|---|
| 1 | N-CMAPSS | NASA PCoE Data Repository — Turbofan Engine Degradation Sim-2 (2021) | Turbofan run-to-failure sensor data; 9 sub-datasets combined (DS01, DS02, DS03, DS04, DS05, DS06, DS07, DS08a, DS08c), matching the scope used in Akindoju (2025). A 10th file (DS08d) was found corrupted at NASA's own source and excluded — see [`docs/model-card.md`](../model-card.md). | HDF5 (.h5) | Full run-to-failure flight cycles (synthetic) | 69,900,301 combined (dev + test, 9 sub-datasets) | Y |
| 2 | Jetson-Bench | Self-generated on-device log (NVIDIA Jetson Nano, tegrastats/jtop + Python harness) | Per-inference latency and power-draw measurements, baseline vs. compressed model. | CSV | Praxis benchmarking window (2026) | TBD (1 record / N-CMAPSS test-set inference) | Y |
## Dataset 1 detail — N-CMAPSS
**Purpose:** Provides labeled multivariate sensor time-series data to
train and validate the MA1DCNN fault-classification model.
**Data treatment:**
- Normalize each sensor channel to zero mean and unit variance before
  training.
- Apply Gaussian and impulse noise augmentation to reduce the
  synthetic-to-reality gap.
- Split each engine unit's trajectory into time-ordered train,
  validation, and test windows.
- Derive the classification label as five independent binary
  per-component flags (fan_fail, lpc_fail, hpc_fail, hpt_fail,
  lpt_fail) per row, assigned via a documented fault-code-to-component
  mapping for each source file combined with an RUL ≤ 30 cycle
  "imminent failure" threshold, following Akindoju (2025), Table 4.1.
**Sample record layout:** (illustrative — see
[`docs/data-dictionary.md`](../data-dictionary.md) for the full, verified
field list)
| Unit | Cycle | RUL (cycles) | fan_fail | lpc_fail | hpc_fail | hpt_fail | lpt_fail |
|---|---|---|---|---|---|---|---|
| 7 | 1 | 79 | 0 | 0 | 0 | 0 | 0 |
| 7 | 68 | 12 | 0 | 0 | 0 | 1 | 1 |
Second row illustrates a DS03-sourced unit within 30 cycles of failure —
DS03 maps to HPT + LPT per the fault-code table, so both flags activate
once the RUL threshold is crossed.
See [`docs/data-dictionary.md`](../data-dictionary.md) for the full field
reference (W flight-condition descriptors, X_s sensor measurements, A
auxiliary fields including `hs`).
## Hypothesis-to-data alignment
| Hypothesis / Dataset | Statement | Required data elements | How the dataset provides it | Potential limitations |
|---|---|---|---|---|
| H1 / Dataset 1 (N-CMAPSS) | Compressed MA1DCNN on Jetson Nano will maintain ≥93.25% accuracy, within 5% of the 98.16% baseline. | Labeled multivariate sensor time-series with fault-class annotations. | N-CMAPSS supplies labeled run-to-failure trajectories across all 9 usable sub-datasets (60 dev-split units), with each row labeled by 5 independent per-component fault flags derived from Akindoju's documented fault-code mapping and an RUL ≤ 30 threshold, for supervised multi-label training. | Synthetic simulation lacks real flight sensor noise, creating a reality gap vs. in-flight conditions. Labels are derived (via the RUL threshold and file-to-component mapping) rather than natively provided, so labeling accuracy is bounded by how well that threshold and mapping reflect the true degradation state. |
| H2 / Dataset 2 (Jetson-Bench) | Edge-optimized MA1DCNN will reduce Jetson Nano inference latency by ≥25% vs. FP32 baseline. | Per-inference wall-clock timestamps for baseline and compressed models across the test set. | Jetson-Bench logs GPU-synchronized inference timing per sample, enabling p50/p95/p99 comparison. | Thermal throttling and background processes may introduce measurement variance vs. controlled benchmarking. |
| H3 / Dataset 2 (Jetson-Bench) | Compressing MA1DCNN on Jetson Nano will cut per-inference energy 20% and power draw 3%. | Per-inference power/energy readings across compression levels, paired with accuracy metrics. | Jetson-Bench logs onboard INA3221 power draw per inference, spot-checked against an external meter. | Onboard sensor bias and ambient temperature may affect power-draw measurement accuracy. |
