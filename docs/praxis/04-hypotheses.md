# Research Hypotheses
## H1 — Accuracy under compression
**Statement:** Compressed MA1DCNN deployed on NVIDIA Jetson Nano will
maintain accuracy, macro-precision, and noise-robustness each within a 5%
relative drop from the 98.16% workstation baseline (≥ 93.25%), versus an
equally-compressed standard 1D-CNN tested under identical conditions.
- **Independent variables:** Model architecture only (E-O MA1DCNN vs.
  standard 1D-CNN); compression and hardware held constant.
- **Dependent variables:** Accuracy (%), macro-precision (%),
  noise-induced accuracy drop (percentage points) due to added sensor
  noise.
- **Testability:** Both models evaluated on a commonly held-out N-CMAPSS
  set at fixed SNR levels (0/10/20 dB), ≥ 5 trials, results reported as
  mean ± 95% CI, compared via a paired test.
- **Note on multi-label accuracy:** With the project's move to a
  5-flag multi-label fault scheme, "accuracy" above refers specifically
  to subset accuracy (exact match across all 5 component flags per
  sample) — the standard scikit-learn `accuracy_score` definition for
  multi-label data. Per-label accuracy, Hamming loss, and Jaccard index
  are tracked separately (see `docs/model-card.md`) and are diagnostic,
  not substitutes for this definition when evaluating the 93.25% floor.
## H2 — Inference latency
**Statement:** Edge-optimized MA1DCNN (mixed-precision + INT8) will cut
end-to-end inference latency (single-sample, incl. preprocessing) on
Jetson Nano by ≥ 25% vs. the FP32 baseline.
- **Independent variables:** Optimization state — edge-optimized
  (mixed-precision + INT8) vs. baseline FP32.
- **Dependent variable:** End-to-end inference latency (ms), including
  preprocessing and data transfer.
- **Testability:** Measured at batch size 1, 10 warm-up runs excluded,
  ≥ 100 timed trials, mean/SD/95% CI reported; 25% target tested via
  paired comparison.
## H3 — Energy and power draw
**Statement:** Compressing MA1DCNN (pruning + INT8) on Jetson Nano will
cut energy/inference ≥ 20% and power draw ≥ 3% vs. uncompressed (energy
gain driven mainly by latency, not power alone), while holding accuracy,
precision, recall, and F1 within 5% of baseline.
- **Independent variables:** Compression level — FP32 baseline vs.
  pruned+quantized model at defined ratios (25/50/75% channel pruning +
  INT8).
- **Dependent variables:** Energy/inference (J), power draw (W); accuracy,
  macro-precision, recall, F1 (%), each vs. baseline.
- **Testability:** Power sampled at 10 Hz via inline USB meter (idle
  subtracted), ≥ 5 trials/level on same test set, mean ± 95% CI vs.
  baseline via paired test.
