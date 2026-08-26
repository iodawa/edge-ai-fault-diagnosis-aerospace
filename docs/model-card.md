# Model Card — E-O MA1DCNN

Following the structure of Mitchell et al. (2019), *Model Cards for Model
Reporting*.

## Model details
- **Architecture:** Multi-Head Attention 1D-CNN (MA1DCNN), edge-optimized
  variant (E-O MA1DCNN) via structured channel pruning + INT8 quantization.
- **Baseline:** Full-precision MA1DCNN trained on a workstation GPU,
  hyperparameters tuned with Optuna.
- **Compression:** Mixed-precision training → structured pruning
  (25/50/75% channel ratios) → post-training INT8 quantization →
  TensorRT compilation.
- **Format:** Exported to ONNX, compiled to a TensorRT engine for
  deployment.

## Intended use
- **Primary use:** Real-time, on-device fault classification for aircraft
  turbofan engine components during flight operations, running on
  resource-constrained edge hardware (NVIDIA Jetson Nano class).
- **Primary users:** Aerospace maintenance teams / onboard health
  monitoring systems.
- **Out of scope:** Direct flight-control or autonomous shutdown
  decisions — outputs are decision-support for maintenance workflows, not
  a substitute for certified flight-safety systems. Sub-threshold
  confidence outputs are routed to human review, not acted on
  automatically.

## Training data
NASA N-CMAPSS (DS03 subset), ~9.8M records, 100 engine units, 7 failure
modes. See [`docs/praxis/07-data-sources.md`](praxis/07-data-sources.md)
and [`docs/data-dictionary.md`](data-dictionary.md).

## Metrics
| Metric | Baseline (FP32, workstation) | Edge target |
|---|---|---|
| Accuracy | 98.16% | ≥ 93.25% |
| Macro-precision | TBD | within 5% of baseline |
| Recall / F1 | TBD | within 5% of baseline |
| ROC-AUC | TBD | report alongside accuracy |
| Inference latency | FP32 baseline | ≥ 25% reduction |
| Energy / inference | uncompressed baseline | ≥ 20% reduction |
| Power draw | uncompressed baseline | ≥ 3% reduction |

## Limitations
- N-CMAPSS is a synthetic simulation; it lacks real in-flight sensor
  noise, creating a synthetic-to-reality gap.
- Onboard INA3221 power sensing carries bias and ambient-temperature
  sensitivity; results are spot-checked against an external USB power
  meter.
- Thermal throttling / background processes on the Jetson Nano can add
  latency-measurement variance.
- Results are benchmark-validated on synthetic data pending real-sensor
  confirmation.

## Ethical / safety considerations
Sub-threshold-confidence predictions must degrade to human review rather
than silently passing through, so that speed gains never risk an
unverified false negative reaching the cockpit.
