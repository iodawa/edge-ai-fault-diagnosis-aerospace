 # Data Sources

| ID | Name | Source | Description | Format | Time span | # records | Access |
|---|---|---|---|---|---|---|---|
| 1 | N-CMAPSS | NASA PCoE Data Repository — Turbofan Engine Degradation Sim-2 (2021) | Turbofan run-to-failure sensor data; 3 sub-datasets combined (DS01, DS03, DS08a), each a distinct fault mode. | HDF5 (.h5) | Full run-to-failure flight cycles (synthetic) | ~26.1M combined (3 of 8 sub-datasets used) | Y |
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
- Derive the classification label from each row's `hs` (health state)
  flag: healthy rows are pooled into one shared class regardless of
  source file; degraded rows are labeled by which of the three files
  they came from, yielding 4 classes total.

**Sample record layout:** (illustrative — see
[`docs/data-dictionary.md`](../data-dictionary.md) for the full, verified
field list)

| Unit | Cycle | Alt (ft) | Mach | TRA (%) | T2 | T24 | T30 | hs | RUL (cycle) |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 1 | 3001 | 0.349 | 75.15 | 520.37 | 620.83 | 1486.23 | 1 (healthy) | 79 |

See [`docs/data-dictionary.md`](../data-dictionary.md) for the full field
reference (W flight-condition descriptors, X_s sensor measurements, A
auxiliary fields including `hs`).

## Hypothesis-to-data alignment

| Hypothesis / Dataset | Statement | Required data elements | How the dataset provides it | Potential limitations |
|---|---|---|---|---|
| H1 / Dataset 1 (N-CMAPSS) | Compressed MA1DCNN on Jetson Nano will maintain ≥93.25% accuracy, within 5% of the 98.16% baseline. | Labeled multivariate sensor time-series with fault-class annotations. | N-CMAPSS supplies labeled run-to-failure trajectories across 3 combined sub-datasets (24 units, 3 distinct fault modes plus a pooled healthy class) for supervised training. | Synthetic simulation lacks real flight sensor noise, creating a reality gap vs. in-flight conditions. Combining separate simulation batches risks the model learning per-file artifacts instead of true degradation signatures — mitigated by pooling healthy samples across files. |
| H2 / Dataset 2 (Jetson-Bench) | Edge-optimized MA1DCNN will reduce Jetson Nano inference latency by ≥25% vs. FP32 baseline. | Per-inference wall-clock timestamps for baseline and compressed models across the test set. | Jetson-Bench logs GPU-synchronized inference timing per sample, enabling p50/p95/p99 comparison. | Thermal throttling and background processes may introduce measurement variance vs. controlled benchmarking. |
| H3 / Dataset 2 (Jetson-Bench) | Compressing MA1DCNN on Jetson Nano will cut per-inference energy 20% and power draw 3%. | Per-inference power/energy readings across compression levels, paired with accuracy metrics. | Jetson-Bench logs onboard INA3221 power draw per inference, spot-checked against an external meter. | Onboard sensor bias and ambient temperature may affect power-draw measurement accuracy. |
