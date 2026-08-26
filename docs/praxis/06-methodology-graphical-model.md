# Methodology — Graphical Model of Research

Six-phase research pipeline. Each phase below maps to a `src/edge_fault_dx/`
module and a `notebooks/` exploration file (see `docs/architecture.md`).

## 1. Collect data and establish baseline environment
- **What:** Obtain the NASA N-CMAPSS dataset and set up the Jetson Nano
  benchmarking environment.
- **Why:** Acquire training data and establish the hardware testbed before
  model development.
- **How:** Download N-CMAPSS HDF5 files from NASA PCoE. Flash JetPack on
  the Jetson Nano. Use pandas/h5py for EDA.

## 2. Pre-process and augment data
- **What:** Clean, normalize, and augment sensor data; split into train,
  validation, and test sets.
- **Why:** Prepare data for training and close the synthetic-to-real gap.
- **How:** Normalize sensor channels. Inject Gaussian and impulse noise.
  Perform a time-ordered train/validation/test split.

## 3. Design and train baseline MA1DCNN
- **What:** Build and train a full-precision multi-head attention 1D-CNN
  on a workstation GPU.
- **Why:** Establish the 98.16% accuracy gold-standard ceiling before any
  compression is applied.
- **How:** Implement the MA1DCNN architecture per Wang et al. (2020),
  train with Optuna-tuned hyperparameters.

## 4. Optimize and compress the model
- **What:** Apply mixed-precision training, pruning, and INT8 quantization
  to shrink the model for edge deployment.
- **Why:** Close the algorithm-to-hardware gap and enable real-time
  inference on the Jetson Nano.
- **How:** Prune filters (Liao et al., 2024); quantize INT8 via TensorRT
  (Ron et al., 2022; Isenkul, 2025).

## 5. Evaluate and validate against hypotheses
- **What:** Compare edge-model accuracy, latency, and energy results
  against the proposal's target thresholds.
- **Why:** Confirm the model is safe, accurate, and fast enough for
  real-time pilot alerting.
- **How:** Test against the 93.25% floor, 25% latency target, and
  20%/3% energy/power targets; run a noise-robustness check.

## 6. Deploy and benchmark on Jetson Nano
- **What:** Compile the compressed model to a TensorRT engine and
  benchmark it on-device.
- **Why:** Measure real inference latency and power draw under
  edge-hardware constraints.
- **How:** Export to ONNX, compile with TensorRT, log latency and
  INA3221 power via `tegrastats`/`jtop`.
