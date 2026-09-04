# Annotated Bibliography
| Ref # | Strategic purpose | Source |
|---|---|---|
| 1 | Establishes the problem or context | Verhagen, W.J.C., Santos, B.F., Freeman, F., van Kessel, P., Zarouchas, D., Loutas, T., Yeun, R.C.K., & Heiets, I. (2023). Condition-based maintenance in aviation: Challenges and opportunities. *Aerospace, 10*(9), Article 762. |
| 2 | Justifies the scope or urgency of the issue | Stanton, I., Munir, K., Ikram, A., & El-Bakry, M. (2023). Predictive maintenance analytics and implementation for aircraft: Challenges and opportunities. *Systems Engineering, 26*(2), 216–237. |
| 3 | Summarizes similar solutions from literature | Liao, J.-X., Wei, S.-L., Xie, C.-L., Zeng, T., Sun, J., Zhang, X., & Fan, F.-L. (2024). BearingPGA-Net: A lightweight and deployable bearing fault diagnosis network via decoupled knowledge distillation and FPGA acceleration. *IEEE Transactions on Instrumentation and Measurement, 73*, 1–14. |
| 4 | Supports methodology (model, metric, etc.) | Ron, D.A., Freire, P.J., Prilepsky, J.E., Kamilian-Kopae, M., Napoli, A., & Turitsyn, S.K. (2022). Experimental implementation of a neural network optical channel equalizer in restricted hardware using pruning and quantization. *Scientific Reports, 12*, Article 8713. |
| 5 | Identifies a gap or limitation this work addresses | Wang, L., Chaw, J.K., Ang, M.C., Cheng, X., Zaman, H.B., Gunasekaran, S.S., & Mahmoud, M.A. (2025). A systematic review of knowledge distillation in industrial predictive maintenance: Applications, methods and challenges. *ICT Express, 11*, 100123. |
| 6 | Validates thesis targets with quantitative benchmark data | Isenkul, M.E. (2025). Energy-aware deep learning for real-time video analysis through pruning, quantization, and hardware optimization. *Journal of Real-Time Image Processing, 22*(3), Article 125. https://doi.org/10.1007/s11554-025-01703-0 |
| 7 | Establishes the closest prior work and the specific gap this praxis addresses | Akindoju, T. M. (2025). *Aerospace fault diagnosis accuracy in aircraft engine using one-dimensional convolutional neural networks with multi-head attention mechanism* [Praxis, The George Washington University]. |
---
## 1. Verhagen et al. (2023) — Condition-based maintenance in aviation
**Summary:** Comprehensive assessment of condition-based maintenance (CBM)
across the aircraft lifecycle, drawing on the four-year EU ReMAP project
(expert panels, stakeholder workshops, interviews). Shows CBM reduces
downtime and maintenance expense when predicted failures become scheduled
maintenance, with potential EU savings up to €700M/yr. Documents
regulatory momentum, including the FAA's 2022 Integrated Aviation Health
Management authorization.
**Methodology:** Systematic lifecycle assessment, EU ReMAP project.
**Evaluation:** Current, authoritative review that directly cites the
foundational Gerdes et al. (2016) cost-evaluation work. Survey-level —
presents no new hardware experiments or on-device measurements.
**Relevance:** Defines the operational problem this praxis targets: high
costs from unscheduled maintenance, with condition-based monitoring as the
established remedy. Provides CBM adoption context and motivates
edge-enabled fault diagnosis.
## 2. Stanton et al. (2023) — Predictive maintenance analytics for aircraft
**Summary:** Reviews corrective, preventive, and predictive maintenance
strategies and the analytics pipelines used to implement them. Frames
corrective/unscheduled maintenance as the most expensive approach and
motivates a shift toward data-driven predictive maintenance, surveying ML
techniques and their implementation challenges.
**Methodology:** Structured literature review of predictive-maintenance
strategy and analytics implementation.
**Evaluation:** Peer-reviewed in *Systems Engineering* (Wiley). Updates
the maintenance-cost argument for the ML era, more current than older
foundational sources such as Gerdes et al. (2016). Review-level — reports
no deployment or hardware-efficiency metrics of its own.
**Relevance:** Justifies the urgency of ML-based fault detection to reduce
the high cost of unscheduled maintenance and establishes the benefits an
edge-deployed MA1DCNN aims to deliver: less downtime, faster fault
identification.
## 3. Liao et al. (2024) — BearingPGA-Net
**Summary:** Compresses a bearing fault-diagnosis CNN via decoupled
knowledge distillation and layer-wise fixed-point quantization (parameter
size reduced to 2.83k, 50% fewer bits/parameter). Deploys on a low-power
Kintex-7 FPGA using custom Verilog modules. Achieves >200x faster diagnosis
than a CPU with <0.4 percentage-point drop in F1/recall/precision, and
>95% F1 under -6 dB noise on CWRU.
**Methodology:** Decoupled knowledge distillation + layer-wise fixed-point
quantization; full FPGA deployment with custom Verilog acceleration.
**Evaluation:** Deployable, low-power bearing-fault CNN with standardized
F1/recall metrics — a rare hardware-accelerated benchmark comparable to
edge diagnostic architectures. Relies on specialized FPGA engineering
(fixed-point quantization, custom Verilog) that reduces accessibility and
portability compared to edge-AI platforms such as Jetson Nano.
**Relevance:** No existing study provides consolidated watt-level power or
per-inference energy measurements for a compressed fault-diagnosis CNN on
edge devices — the empirical gap this praxis addresses. This praxis's
Jetson Nano power/energy measurements bridge the FPGA-literature gap and
establish the GPU-based baseline needed to evaluate compressed MA1DCNN
under realistic low-power deployment. Its bearing-fault findings inform
methodology but are not a turbofan-engine accuracy benchmark.
## 4. Ron et al. (2022) — Neural network optical channel equalizer, pruning + quantization
**Summary:** Compresses an MLP optical-channel equalizer with pruning and
INT8 quantization, achieving major complexity/memory reductions while
preserving Q-factor performance across nonlinear launch powers in coherent
fiber-optic transmission experiments. Deploys the compressed equalizer on
Raspberry Pi 4 and Jetson Nano, measuring latency and energy: ~56% faster
inference, ~57% energy savings, with only ~3% instantaneous-power
reduction during real-time symbol recovery.
**Methodology:** Pruning + INT8 quantization on an MLP equalizer;
evaluates Q-factor, latency, and energy on Raspberry Pi 4 and Jetson Nano.
**Evaluation:** Significant insight into pruning/INT8 quantization for
optical-channel equalizers, extending Jetson-Nano and Raspberry-Pi
watt-level measurement — though the setup relies on simulated data and
isn't fully transferable. Uses an MLP equalizer rather than a
fault-diagnosis CNN; limited by TensorFlow Lite's sparse-inference
constraints and lacks coherent-receiver hardware measurement.
**Relevance:** Provides hardware-grounded latency and watt-level
measurements for compressed neural models — empirical evidence that
strengthens this praxis's pruning/quantization/energy-efficiency
evaluation on constrained edge hardware. Confirms that energy reduction
arises mainly from decreased inference time rather than power variation,
reinforcing the framework distinguishing instantaneous power from total
energy consumption.
## 5. Wang et al. (2025) — Knowledge distillation in industrial predictive maintenance
**Summary:** Systematic review of 48 predictive-maintenance studies using
knowledge distillation (KD), identifying major application domains, core
KD paradigms, and efficiency-accuracy trade-offs. Shows KD enables
lightweight industrial models while highlighting challenges in data
scarcity, cross-domain generalization, and deployment on constrained
predictive-maintenance hardware.
**Methodology:** Systematic literature review following PRISMA guidelines
and Kitchenham's evidence-based review framework.
**Evaluation:** Methodologically rigorous with comprehensive coverage;
weaknesses include dependence on secondary evidence, limited empirical
validation, and persistent challenges in data scarcity, cross-domain
generalization, and constrained industrial deployment.
**Relevance:** Synthesizes knowledge-distillation methods for predictive
maintenance, informing efficiency-constraint and model-compression
strategy. Its 48-study synthesis highlights practical compression gains,
methodological patterns, and unresolved limitations that shape this
praxis's evaluation of lightweight, deployable architectures.
## 6. Isenkul (2025) — Energy-aware deep learning for real-time video analysis
**Summary:** Benchmarks pruning, INT8 quantization, and TensorRT hardware
optimization together on real-time video models (YOLOv8, MobileNet) using
Jetson-class edge hardware. Pruning cuts energy up to 35% at 92% accuracy;
quantization adds 18% further savings plus 25% faster inference; TensorRT
optimization yields a 30% energy reduction. Paired power and accuracy
measurements isolate each technique's individual contribution.
**Methodology:** Structured pruning, post-training INT8 quantization, and
TensorRT hardware-aware optimization, benchmarked for energy, accuracy,
and inference speed on edge GPU hardware.
**Evaluation:** The strongest same-hardware, same-technique-stack evidence
available — benchmarks pruning, quantization, and TensorRT together rather
than in isolation. Targets video object-detection models rather than
time-series fault diagnosis, so its percentages are directional evidence,
not a guaranteed transfer.
**Relevance:** Justifies this praxis's thesis-level energy-reduction
targets by showing the same three-technique compression stack achieves
large, measurable energy savings on comparable edge hardware. Reinforces
mixed-precision training and Optuna tuning as safeguards that pair
compression with accuracy preservation.
## 7. Akindoju (2025) — Aerospace fault diagnosis accuracy using MA1DCNN
**Summary:** A 2025 GWU praxis applying a Multi-Head Attention 1D-CNN
(MA1DCNN) to multi-label fault classification on the full usable
N-CMAPSS dataset (9 sub-datasets combined). Uses normalization,
segmentation, undersampling of healthy rows for class balance, Cosine
Annealing Warm Restarts, Gaussian noise augmentation, and Optuna
hyperparameter tuning, with SHAP analysis for interpretability. (Her
Abstract describes this step as "selective oversampling" with "focal
loss" — but her detailed Methodology and best-model hyperparameters,
Section 4.2.3 and Table 4.4, show undersampling of negative rows per
label and Focal = FALSE. This entry follows the more detailed,
internally-cited account.) On 5,335,270 test samples, achieves Macro-F1
= 99.46%, Macro-Precision = 99.92%, and accuracy = 98.16% — slightly
below a 1DCNN-BiLSTM-CBAM variant (98.23%) and XGBoost (98.25%)
benchmarked in the same study.
**Methodology:** MA1DCNN trained on all 9 usable N-CMAPSS sub-datasets;
labels constructed as 5 binary per-component fault flags (fan, LPC, HPC,
HPT, LPT) via a documented fault-code-to-component mapping table
combined with an RUL ≤ 30 cycle "imminent failure" threshold; raw HDF5
converted to a chunked Parquet pipeline with online feature scaling.
**Evaluation:** The most directly comparable prior work available —
same architecture family, same dataset, and the most complete labeling
methodology found in the literature for this specific problem. Its
future-work section (real-flight telemetry testing, adaptive
thresholds, adversarial robustness, joint RUL+fault prediction) never
proposes model compression or physical edge-hardware deployment.
**Relevance:** The closest available prior work and the direct source
of this praxis's 98.16% workstation-accuracy baseline. This praxis
directly adopts Akindoju's data scope (9 usable N-CMAPSS sub-datasets),
labeling methodology (the 5-flag multi-label per-component scheme
derived from her fault-code mapping and RUL ≤ 30 threshold), and
reported baseline — see
`deliverables/relabel_decision_akindoju_alignment.docx` for the full
decision record. The gap this praxis fills is squarely the
algorithm-to-hardware step Akindoju's own future work never addresses:
structured pruning, INT8 quantization, and physical
deployment/benchmarking on an NVIDIA Jetson Nano.
