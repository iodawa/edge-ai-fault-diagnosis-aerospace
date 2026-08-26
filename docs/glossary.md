# Glossary & Acronyms

## Glossary of terms

| Term | Definition |
|---|---|
| Edge AI Devices (Model) | AI models deployed directly on resource-constrained hardware to enable safe, rapid, real-time decision-making, such as predicting aircraft engine faults at the sensor source. |
| MA1DCNN | Multi-Head Attention 1-Dimensional Convolutional Neural Network — combines 1D convolution layers (local pattern extraction from time-series sensor data) with multi-head self-attention (efficient capture of long-range dependencies). |
| ROC-AUC | Receiver Operating Characteristic – Area Under the Curve. Evaluates a model's overall effectiveness and discrimination — its ability to distinguish engine fault conditions from healthy conditions. |
| NASA CMAPSS (N-CMAPSS) | New Commercial Modular Aero-Propulsion System Simulation — NASA's benchmark dataset simulating a large turbofan engine's run-to-failure and degradation process under realistic flight conditions. |
| Macro-Precision | Measures a model's ability to identify true fault events while minimizing false alarms, evaluating all fault classes equally so majority classes don't overshadow minority classes. |
| Optuna | Hyperparameter-tuning engine (learning rate, dropout, batch size, weights) that balances diagnostic accuracy against efficient operation on resource-constrained edge hardware. |
| Mixed-Precision Training | Optimization technique that preserves accuracy on resource-constrained edge hardware while reducing memory footprint and increasing data-processing throughput. |
| W (Flight-Condition Descriptors) | N-CMAPSS's variable group for aircraft operating conditions during flight: altitude, flight Mach number, throttle-resolver angle (TRA), fan-inlet temperature (T2). |
| Xs (Sensor Measurements) | N-CMAPSS's variable group for physical sensor readings: fan/core speed (Nf, Nc), station temperatures (T24, T30, P40). |
| EDA (Exploratory Data Analysis) | Initial analysis of a dataset to summarize its main characteristics, find hidden patterns, and spot anomalies using visual tools and summary statistics before formal modeling. |

## Acronyms

| Acronym | Definition |
|---|---|
| AI | Artificial Intelligence |
| CNN | Convolutional Neural Network |
| MA1DCNN | Multi-Head Attention One-Dimensional Convolution Neural Network |
| E-O MA1DCNN | Edge-Optimized Multi-Head Attention One-Dimensional Convolution Neural Network |
| ADCNN | Attention-Based Convolutional Neural Network |
| ROC-AUC | Receiver Operating Characteristic Area Under the Curve |
| NASA | National Aeronautics and Space Administration |
| CMAPSS | Commercial Modular Aero-Propulsion System Simulation |
| ReMAP | Real-time condition-based Maintenance for Adaptive Aircraft maintenance Planning |
