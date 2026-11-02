"""Noise-robustness evaluation (Phase 5, H1) --
src/edge_fault_dx/evaluation/noise_robustness.py.

Extracted from h1_standard_cnn_and_noise_robustness.ipynb Sections 7/10
(real results in h1_notebook_code_walkthrough.docx Sections 10-11 and
praxis_writeup.docx Section 4.5.1). This replaces the one-line TODO stub
previously at this path.

Nothing in this codebase previously added noise at evaluation time.
preprocessing.ipynb already applies Gaussian noise at -6dB SNR, but only as
TRAINING-split augmentation, applied once at the row level before
windowing -- a different mechanism for a different purpose (making the
trained model more robust), not the EVALUATION-time perturbation H1's
testability clause asks for: noise added to the held-out, already-normalized
validation windows, at specific SNR levels, repeated across multiple
independent trials, to measure how much accuracy each model loses as noise
increases.

Real, measured results at 20% compression, both reusing this project's own
already-confirmed checkpoints (praxis_writeup.docx Section 4.5.1):

    -- Confounded comparison (MA1DCNN QAT vs. StandardCNN PTQ-only; the two
       checkpoints differed in compression method, not just architecture) --
    SNR (dB) | MA1DCNN mean    | StandardCNN mean
    0        | 0.4747 +/- .0013 | 0.4851 +/- .0010
    10       | 0.7159 +/- .0004 | 0.6925 +/- .0010
    20       | 0.8262 +/- .0005 | 0.8111 +/- .0006
    paired_comparison(): t = 9.8485, p = 0.0000 (highly significant)

    -- Controlled comparison (both QAT at 20%, compression method held
       constant -- the comparison H1's own testability clause specifies) --
    SNR (dB) | MA1DCNN mean    | StandardCNN mean
    0        | 0.4747 +/- .0013 | 0.4858 +/- .0014
    10       | 0.7159 +/- .0004 | 0.7015 +/- .0006
    20       | 0.8262 +/- .0005 | 0.8207 +/- .0007
    paired_comparison(): t = 0.2834, p = 0.7810 (NOT significant)

The confounded comparison's highly significant result did not survive
holding compression method constant -- it was an artifact of StandardCNN's
checkpoint having had less compression-time fine-tuning (PTQ only, vs.
MA1DCNN's QAT), not evidence that MA1DCNN's attention modules make it more
robust to sensor noise. See praxis_writeup.docx Section 4.5.1 for the full
account, including the additive, non-disruptive re-test (Sections 6.1/8.1 of
the notebook) that produced the controlled numbers without touching the
original confounded run.

NOTE ON PROVENANCE: every function below is copied, with signatures and
imports completed to make this a runnable module, from
h1_notebook_code_walkthrough.docx Sections 10.1-10.3 and h1_completion_plan.
docx Section 6 -- both planned this harness before it was ever run. Check
this against the notebook's literal source before treating it as
byte-identical. One real gap, left open on purpose: resweep_threshold()
(compression/quantization.py) was never wired into the compression loop
that produces the checkpoints this harness evaluates for StandardCNN, so
every StandardCNN verdict feeding into the comparison above used a fixed
0.4215 threshold, not a threshold re-swept per compressed variant the way
three of MA1DCNN's own checkpoints were (praxis_writeup.docx Section 4.5.1's
own note).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats

SNR_LEVELS_DB = [0, 10, 20]   # H1's own testability spec
N_TRIALS = 5                  # H1's own minimum


# ---------------------------------------------------------------------------
# Noise injection -- the actual decibel math
# ---------------------------------------------------------------------------
def add_snr_noise(X, snr_db, rng):
    """Adds zero-mean Gaussian noise to a window, scaled so the noisy signal
    sits at exactly snr_db decibels relative to X's own per-window power --
    the standard SNR_dB = 10*log10(signal_power / noise_power) definition,
    inverted to solve for the noise variance that produces a given target
    SNR.

    X : np.ndarray, shape (28, 50) -- one already-normalized window
    """
    signal_power = np.mean(X ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=X.shape).astype(np.float32)
    return X + noise


class NoisyWindowedFaultDataset(torch.utils.data.Dataset):
    """Wraps an existing WindowedFaultDataset (training/train.py), adding
    independent add_snr_noise() noise to every window at a fixed SNR, with a
    given numpy Generator -- a new seed per trial is what makes repeated
    trials independent, not repeats of the identical noise draw. Noise is
    added to all 28 channels uniformly, matching H1's own dependent-variable
    language ("noise-induced accuracy drop... due to added sensor noise")
    rather than singling out the subset preprocessing.ipynb's training-time
    augmentation already touches.
    """

    def __init__(self, base_dataset, snr_db, seed):
        self.base = base_dataset
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X, y = self.base[idx]
        X_noisy = add_snr_noise(X.numpy(), self.snr_db, self.rng)
        return torch.from_numpy(X_noisy), y


# ---------------------------------------------------------------------------
# Trial loop -- reuses train.py's evaluate() exactly as-is
# ---------------------------------------------------------------------------
def run_noise_robustness_trials(model, val_base_dataset, threshold, loss_fn, device):
    """Returns {snr_db: [trial_1_metrics, trial_2_metrics, ...]}. Clean
    metrics (no noise) are a separate baseline computed once via the
    existing evaluate() against val_base_dataset directly, kept outside this
    dict. evaluate() (training/train.py) takes a model, a loader, a loss
    function, a device, and a threshold -- it has no idea whether the
    loader's windows are clean or noisy, which is exactly what makes it
    reusable here with zero changes.
    """
    from ..training.train import evaluate  # imported here to avoid a hard import-time cycle

    results = {}
    for snr_db in SNR_LEVELS_DB:
        trial_metrics = []
        for trial in range(N_TRIALS):
            seed = hash((snr_db, trial)) % (2 ** 31)  # distinct, reproducible per (snr, trial)
            noisy_dataset = NoisyWindowedFaultDataset(val_base_dataset, snr_db, seed)
            noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=256, shuffle=False)
            metrics = evaluate(model, noisy_loader, loss_fn, device, threshold)
            trial_metrics.append(metrics)
        results[snr_db] = trial_metrics
    return results


# ---------------------------------------------------------------------------
# Turning per-trial metrics into H1's actual reported numbers
# ---------------------------------------------------------------------------
def summarize_trials(trial_metrics, metric_key="f1_macro"):
    """mean +/- 95% CI (t-distribution, small-N-appropriate for N_TRIALS=5)."""
    values = np.array([m[metric_key] for m in trial_metrics])
    mean = values.mean()
    sem = stats.sem(values)
    ci = sem * stats.t.ppf(0.975, df=len(values) - 1)
    return mean, ci


def paired_comparison(ma1dcnn_drops, standard_cnn_drops):
    """Paired t-test over matched (same snr_db, same trial index)
    accuracy-drop values -- the paired-test comparison H1's testability
    clause asks for.

    accuracy_drop[trial] = clean_f1_macro - noisy_f1_macro[trial], per model.
    """
    t_stat, p_value = stats.ttest_rel(ma1dcnn_drops, standard_cnn_drops)
    return t_stat, p_value
