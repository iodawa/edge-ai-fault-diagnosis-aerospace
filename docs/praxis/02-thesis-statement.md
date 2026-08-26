# Thesis Statement

> An edge-optimized MA1DCNN predictive model is required to accelerate
> real-time aircraft engine fault diagnosis, preventing catastrophic
> failures and reducing unscheduled maintenance costs.

| Field | Content |
|---|---|
| Research product | Edge-Optimized MA1DCNN Predictive Model |
| Format | Python script |
| Deliverable usage | Aerospace maintenance teams will use the model directly on edge devices during flight operations. |
| Tie back to problem statement | By enabling real-time edge processing, the model overcomes delays due to computational complexity and facilitates on-time fault prediction and identification. |
| New contributions | Integrates Optuna hyperparameter tuning and mixed-precision training to enable high-accuracy MA1DCNN model architectures on resource-constrained hardware. |
| Scope | This edge-based, lightweight diagnostic model can be adapted to non-aerospace automated systems using resource-constrained edge AI chips such as NVIDIA Jetson. |
| Main methodology | Machine learning: Multi-Head Attention 1D-CNN, structured channel pruning, post-training INT8 quantization via TensorRT. |
| Inputs | Flight-condition **W** (altitude, Mach, TRA, T2) and sensors **Xs** (Nf, Nc: rpm; T24, T30, T50: °R; Ps30, P40: psia) from NASA N-CMAPSS (DS03). |
| Outputs | Fault classification (binary labels: `fan_fail`, `lpc_fail`, `hpc_fail`, `hpt_fail`, `lpt_fail`); power draw (W); energy/inference (J); Remaining Useful Life / RUL (cycles). |
