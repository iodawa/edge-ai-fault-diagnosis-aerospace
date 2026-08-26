# Scope of Work (SOW)

This praxis focuses on an Edge AI model for real-time aircraft engine
fault classification. Fault classification identifies which component is
failing during a flight cycle, while predictive maintenance uses that
classification over time for scheduled maintenance decisions.

**Architecture:** Multi-Head Attention One-Dimensional Convolution Neural
Network (MA1DCNN).

**Novelty:** Addresses the algorithm-to-hardware gap left by prior work —
compressing and deploying an attention-based fault-diagnosis model on
resource-constrained edge hardware.

**Objective:** Design an optimized MA1DCNN enabling rapid, on-device fault
classification to support maintenance decisions.

**Methodology:** Apply Optuna tuning and mixed-precision training, then
structured pruning and INT8 quantization, to achieve accuracy within 5% of
the 98.16% workstation baseline (≥ 93.25%), while preserving
macro-precision, recall, and F1 score.

**Validation:** Benchmark the model on an NVIDIA Jetson Nano.

**Deliverables:**
- A scalable Edge-Optimized MA1DCNN (E-O MA1DCNN) fault classification model
- A performance report
- A comparison of power-draw and energy-per-inference metrics against a
  non-edge baseline
