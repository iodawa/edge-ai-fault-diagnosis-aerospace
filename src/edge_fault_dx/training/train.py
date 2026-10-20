"""Training loop for MA1DCNN (Phase 5) -- edge-ai-fault-diagnosis-aerospace.

Ported from train.ipynb, where every function below was validated against its
own synthetic self-checks (see run_self_checks() / --self-check) and then run
for real against N-CMAPSS windowed data. The judgment calls behind each named
config -- why regularization, why AdamW was tried and rejected, why
conv-stack dropout, why BatchNorm1d is a real trade-off not a clean win, and
the 30-trial Optuna search that produced trial 13 -- are documented in
overfitting_results_and_options.docx, ma1dcnn_challenges_results_optuna.docx,
and overfitting_adamw_comparison.xlsx ("Optuna Search Results" / "Final
Verdict" sheets). This file carries no narrative, only the code and the
pointers to where the narrative lives.

Default config: TRIAL_13_MODEL_CONFIG / TRIAL_13_TRAIN_CONFIG -- the Optuna
search's Pareto-front knee-point pick (trial 13 of 30, see
optuna_search.py / optuna_search.ipynb). Per this project's "prove it
standalone first" discipline, trial 13 was not ported here until it had been
confirmed non-overfitting across two genuinely independent real re-runs of
this exact run_training(), in separate Colab sessions, plus the original
Optuna search's own (shorter) evaluation:

    source                 val_f1_macro   best epoch kept
    Optuna search           0.8375        n/a (search's own objective, not this run_training())
    Independent re-run A    0.8450        5
    Independent re-run B    0.8440        6   <- this checkpoint's weights (ma1dcnn_trial13.pt)

The two independent re-runs agree within 0.001 macro-F1 of each other -- a
real non-overfitting confirmation. (Correction: an earlier version of this
docstring cited 0.8401 and described "Re-run 1 (epoch 8): 0.8412" /
"Re-run 2 (epoch 5): 0.8450" as two separate re-runs. Checking the actual
executed notebook cells directly showed epochs 5 and 8 were two epochs from
the SAME single run, not two independent re-runs, and 0.8401 did not match
any real executed number. The corrected picture above is confirmed against
actual executed Colab cell output, not narrative claims -- see
phase4_architecture.docx for the full correction.) A real caveat carried
over from re-run B's real metrics, not fixed here: trial 13's
threshold=0.4215 trades precision for recall (val_precision_macro 0.7689 /
val_recall_macro 0.9383) -- a deliberate decision about that trade-off, not
a silent default, belongs before this config is deployed.

Usage
-----
    python train.py                                  # trains trial 13 (default), saves ma1dcnn_trial13.pt
    python train.py --config batchnorm                # trains a different named config
    python train.py --config trial13 --epochs 10 --windowed-dir /path/to/windowed
    python train.py --list-configs                    # show every named config and exit
    python train.py --self-check                      # run the ported synthetic sanity checks, then exit

Windowed data
-------------
Expects the *_train_windowed.npz / *_val_windowed.npz files windowing.ipynb
produces, one pair per file in NCMAPSS_FILES, under --windowed-dir (or the
EDGE_FAULT_DX_WINDOWED_DIR environment variable). The notebook's real runs
used a Google Drive path inside Colab; that path is not meaningful outside
Colab, so this module takes the directory as a normal CLI argument instead.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, precision_score, recall_score

# MA1DCNN now lives in its own module (src/edge_fault_dx/models/ma1dcnn.py), not inline
# here -- see that file's docstring for why. Path-based, not cwd-based, so this import
# resolves correctly whether train.py is run directly (`python train.py`, this repo's own
# documented usage) or imported from elsewhere (`from train import ...`, as
# threshold_sweep.py and false_alarm_review.py both do) regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # -> src/edge_fault_dx/
from models.ma1dcnn import MA1DCNN, build_ma1dcnn  # noqa: E402 -- re-exported: `from train import build_ma1dcnn` (threshold_sweep.py, false_alarm_review.py) keeps working unchanged.

LOGGER = logging.getLogger("edge_fault_dx.training.train")


# =============================================================================
# 1. Constants shared with ma1dcnn.ipynb / windowing.ipynb
# =============================================================================

# Column order fixes which label column a fault flag lands in, and its length
# (5) fixes the model's output width. Repeated verbatim from loaders.ipynb /
# ma1dcnn.ipynb / windowing.ipynb -- do not reorder independently of those.
FAULT_COLUMNS = ["fan_fail", "lpc_fail", "hpc_fail", "hpt_fail", "lpt_fail"]
N_FAULT_COLUMNS = len(FAULT_COLUMNS)  # 5

# Same 9 files, same fixed order, repeated from loaders.ipynb / windowing.ipynb.
NCMAPSS_FILES = [
    "N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS02-006.h5", "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS04.h5", "N-CMAPSS_DS05.h5", "N-CMAPSS_DS06.h5",
    "N-CMAPSS_DS07.h5", "N-CMAPSS_DS08a-009.h5", "N-CMAPSS_DS08c-008.h5",
]

# windowing.ipynb's own output directory is this module's input. Saved files
# are named "<file>_<split>_windowed.npz", which _windowed_file_path() below
# reproduces exactly. The notebook hardcoded a Colab Drive path here; this
# module takes it from an env var (with a plain relative fallback) and lets
# --windowed-dir override it per run -- see build_dataloaders()/run_training().
DEFAULT_WINDOWED_DIR = os.environ.get("EDGE_FAULT_DX_WINDOWED_DIR", "data/windowed")

# Repeated from ma1dcnn.ipynb. Fixed at 28 because windowing.ipynb's own
# INCLUDE_FLIGHT_CONDITIONS=False default was used to produce the real
# data/windowed/ files -- see train_design_questions.docx for why the
# 28-vs-32 question stays genuinely open elsewhere; it is windowing.ipynb's
# call, not something this module decides.
DEFAULT_N_INPUT_CHANNELS = 28

# Table 4.4 (Akindoju, 2025) -- architecture hyperparameters, repeated verbatim from ma1dcnn.ipynb.
BEFORE_TUNING_CONFIG = {
    "n_input_channels": DEFAULT_N_INPUT_CHANNELS,
    "channel_1": 32, "channel_2": 64, "channel_3": 128,
    "kernel_size": 7, "fc_units": 64, "dropout": 0.5,
    "n_outputs": N_FAULT_COLUMNS,
}
AFTER_TUNING_CONFIG = {
    "n_input_channels": DEFAULT_N_INPUT_CHANNELS,
    "channel_1": 32, "channel_2": 128, "channel_3": 128,
    "kernel_size": 5, "fc_units": 16, "dropout": 0.39919,
    "n_outputs": N_FAULT_COLUMNS,
}


# =============================================================================
# 2. Model -- now defined in models/ma1dcnn.py (see this file's import above);
# MA1DCNN and build_ma1dcnn are re-exported from there, not redefined here.
# =============================================================================


# =============================================================================
# 3. Training-level constants -- repeated verbatim from train.ipynb Section 3
# =============================================================================

TRAIN_BEFORE_TUNING_CONFIG = {
    "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 128,
    "threshold": 0.5, "use_focal": False,
}
TRAIN_AFTER_TUNING_CONFIG = {
    "learning_rate": 0.00019963, "weight_decay": 4.29e-06, "batch_size": 256,
    "threshold": 0.76957, "use_focal": False,
}
# Same architecture/base settings as TRAIN_BEFORE_TUNING_CONFIG, weight_decay raised
# 10x -- added after the real baseline run showed val_loss rising from epoch 1 while
# train_loss kept falling (immediate overfitting), to test stronger L2 alone.
TRAIN_REGULARIZED_CONFIG = {
    "learning_rate": 1e-3, "weight_decay": 1e-3, "batch_size": 128,
    "threshold": 0.5, "use_focal": False,
}

# Same architecture as BEFORE_TUNING_CONFIG with dropout added inside the conv stack
# (0.2 on each of the 3 conv blocks), not just once before the FC layer. Neither
# coupled Adam nor decoupled AdamW weight_decay alone closed the generalization gap --
# see overfitting_results_and_options.docx.
BEFORE_TUNING_CONFIG_CONV_DROPOUT = dict(BEFORE_TUNING_CONFIG, conv_dropout=0.2)

# Same architecture as BEFORE_TUNING_CONFIG_CONV_DROPOUT, with BatchNorm1d added after
# each conv block (LAYERED ON TOP of conv_dropout=0.2, not replacing it).
BEFORE_TUNING_CONFIG_CONV_DROPOUT_BN = dict(BEFORE_TUNING_CONFIG_CONV_DROPOUT, use_batchnorm=True)

# CosineAnnealingWarmRestarts is confirmed by Akindoju's own abstract; T_0/T_mult are
# this project's own reasonable defaults, not stated anywhere in the praxis.
SCHEDULER_T_0 = 10
SCHEDULER_T_MULT = 2

N_EPOCHS = 30                 # upper bound; early stopping usually ends a run sooner
EARLY_STOPPING_PATIENCE = 5   # epochs allowed with no val_f1_macro improvement before stopping
RANDOM_SEED = 0                # for reproducible weight init and training order

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trial 13 from optuna_search.py's completed 30-trial multi-objective search
# (directions=["maximize", "minimize"] over val_f1_macro / max generalization gap, TPE
# sampler, N_SEEDS_PER_TRIAL=2, real N-CMAPSS windowed data). study.best_trials /
# print_pareto_front(study) returned a 5-trial Pareto front (trials 9, 1, 12, 13, 4);
# trial 13 is the recommended pick -- a knee-point calculation (Branke, Deb, Dierolf, &
# Osswald, 2004) over the front's marginal F1-per-gap exchange rate shows a sharp drop
# right after it. Both trial 13 (val_f1_macro=0.8375, max_gap=-0.0188) and trial 12
# (0.8445, -0.0089) Pareto-dominate every hand-run config above outright -- see
# overfitting_adamw_comparison.xlsx's "Optuna Search Results" sheet for the full front
# and the exchange-rate math. channel_1=32 and channel_3=128 were held fixed by the
# search (not part of its space), inherited from BEFORE_TUNING_CONFIG unchanged, same
# as every Optuna trial. use_focal was never part of the search space either -- False,
# matching every config above.
TRIAL_13_MODEL_CONFIG = dict(
    BEFORE_TUNING_CONFIG,
    channel_2=64, kernel_size=5, fc_units=16,
    dropout=0.5945728900118123,
    conv_dropout=0.33869914326562556,
    use_batchnorm=False,
)
TRIAL_13_TRAIN_CONFIG = {
    "learning_rate": 0.0004689002238683733,
    "weight_decay": 0.002164379191774169,
    "batch_size": 256,
    "threshold": 0.4214610774407138,  # see module docstring -- trades precision for recall, by design
    "use_focal": False,
}

# Named-config registry, keyed by the --config CLI flag. "trial13" is the default --
# the only config confirmed non-overfitting across 3 independent evaluations (module
# docstring). The others stay here as a faithful record of the full sweep documented
# in overfitting_adamw_comparison.xlsx, selectable for comparison/reproduction, not as
# equally-recommended alternatives.
CONFIGS = {
    "baseline": {
        "model_config": BEFORE_TUNING_CONFIG,
        "train_config": TRAIN_BEFORE_TUNING_CONFIG,
        "optimizer_cls": torch.optim.Adam,
    },
    "regularized": {
        "model_config": BEFORE_TUNING_CONFIG,
        "train_config": TRAIN_REGULARIZED_CONFIG,
        "optimizer_cls": torch.optim.Adam,
    },
    "adamw": {
        "model_config": BEFORE_TUNING_CONFIG,
        "train_config": TRAIN_REGULARIZED_CONFIG,
        "optimizer_cls": torch.optim.AdamW,  # the regularizer that made the gap WORSE -- kept for the record, not recommended
    },
    "conv_dropout": {
        "model_config": BEFORE_TUNING_CONFIG_CONV_DROPOUT,
        "train_config": TRAIN_REGULARIZED_CONFIG,
        "optimizer_cls": torch.optim.Adam,
    },
    "batchnorm": {
        "model_config": BEFORE_TUNING_CONFIG_CONV_DROPOUT_BN,
        "train_config": TRAIN_REGULARIZED_CONFIG,
        "optimizer_cls": torch.optim.Adam,
    },
    "trial13": {
        "model_config": TRIAL_13_MODEL_CONFIG,
        "train_config": TRIAL_13_TRAIN_CONFIG,
        "optimizer_cls": torch.optim.Adam,
    },
}
DEFAULT_CONFIG_NAME = "trial13"


# =============================================================================
# 4. WindowedFaultDataset -- load and concatenate all 9 files' windows
# =============================================================================

def _windowed_file_path(filename, windowed_dir, split):
    """Reproduce windowing.ipynb's own save-path convention exactly: '<file>_<split>_windowed.npz'."""
    out_name = filename.replace(".h5", f"_{split}_windowed.npz")
    return os.path.join(windowed_dir, out_name)


class WindowedFaultDataset(torch.utils.data.Dataset):
    """All 9 files' windows for ONE split (train or val), concatenated into a single dataset.

    Loaded fully into memory at construction time -- windowing.ipynb's real output is
    roughly 710k train windows x 28 channels x 50 timesteps, a few GB as float32, well
    within a normal training host's RAM. If this ever stops being true, this is the
    class to make lazy/chunked.
    """

    def __init__(self, windowed_dir=DEFAULT_WINDOWED_DIR, split="train", files=None):
        files = files if files is not None else NCMAPSS_FILES
        X_parts, y_parts, unit_id_parts, start_cycle_parts = [], [], [], []

        for filename in files:
            path = _windowed_file_path(filename, windowed_dir, split)
            with np.load(path) as npz:
                X_parts.append(npz["X"])              # (n_windows_i, 28, 50)
                y_parts.append(npz["y"])              # (n_windows_i, 5)
                unit_id_parts.append(npz["unit_id"])  # (n_windows_i,) -- traceability only
                start_cycle_parts.append(npz["start_cycle"])

        # float32 for X (standard for model inputs) and for y too -- BCEWithLogitsLoss
        # needs a FLOAT target, not the int8 labels windowing.ipynb saved; casting once
        # here, at load time, means every batch downstream is already the right dtype.
        self.X = np.concatenate(X_parts, axis=0).astype(np.float32)
        self.y = np.concatenate(y_parts, axis=0).astype(np.float32)
        self.unit_id = np.concatenate(unit_id_parts, axis=0)
        self.start_cycle = np.concatenate(start_cycle_parts, axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # torch.from_numpy shares memory with self.X/self.y rather than copying -- fine
        # here since DataLoader workers only ever read, never write, these tensors.
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# =============================================================================
# 5. build_dataloaders -- the reusable loader-building function
# =============================================================================

def build_dataloaders(windowed_dir=DEFAULT_WINDOWED_DIR, batch_size=TRAIN_BEFORE_TUNING_CONFIG["batch_size"],
                       files=None, num_workers=0):
    """Build train and val DataLoaders over WindowedFaultDataset.

    num_workers=0 by default -- safe everywhere. Raise it if data loading becomes the
    training bottleneck on your hardware.
    """
    train_dataset = WindowedFaultDataset(windowed_dir=windowed_dir, split="train", files=files)
    val_dataset = WindowedFaultDataset(windowed_dir=windowed_dir, split="val", files=files)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)   # shuffled every epoch
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)    # never shuffled
    return train_loader, val_loader


# =============================================================================
# 6. compute_metrics -- the five validation metrics
# =============================================================================

def compute_metrics(y_true, logits, threshold):
    """Threshold sigmoid(logits) at `threshold` and score against y_true.

    Parameters
    ----------
    y_true : torch.Tensor, shape (n, 5)
    logits : torch.Tensor, shape (n, 5) -- RAW logits, same convention as MA1DCNN.forward()
    threshold : float -- e.g. TRAIN_BEFORE_TUNING_CONFIG["threshold"] (0.5) or a tuned value
    """
    probs = torch.sigmoid(logits)                            # (n, 5), each entry independently in [0, 1]
    y_pred = (probs >= threshold).to(torch.int64).cpu().numpy()
    y_true_np = y_true.to(torch.int64).cpu().numpy()

    return {
        "f1_macro": f1_score(y_true_np, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true_np, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_np, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true_np, y_pred),      # fraction of (sample, label) entries that disagree
        "jaccard_macro": jaccard_score(y_true_np, y_pred, average="macro", zero_division=0),
    }


# =============================================================================
# 7. train_one_epoch -- one pass over the training set
# =============================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """One full pass over `loader`, updating `model`'s weights. Returns the average training loss."""
    model.train()  # enables dropout
    total_loss, n_batches = 0.0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()          # PyTorch accumulates gradients by default -- must clear them each step
        logits = model(X)              # (batch, 5) raw logits
        loss = loss_fn(logits, y)      # BCEWithLogitsLoss -- expects raw logits, applies sigmoid internally
        loss.backward()                # computes gradients for every parameter via autograd
        optimizer.step()                # applies one optimizer update using those gradients

        total_loss += loss.item()      # .item() pulls a plain Python float out of the 0-d loss tensor
        n_batches += 1

    return total_loss / n_batches


# =============================================================================
# 8. evaluate -- one pass over the validation set
# =============================================================================

def evaluate(model, loader, loss_fn, device, threshold):
    """One full pass over `loader` with gradients disabled. Returns a dict of metrics plus 'loss'."""
    model.eval()  # disables dropout -- deterministic forward passes for a fair evaluation
    total_loss, n_batches = 0.0, 0
    all_logits, all_y = [], []

    with torch.no_grad():  # no autograd bookkeeping needed -- saves memory and time during evaluation
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu())  # move off GPU immediately -- keeps GPU memory free during accumulation
            all_y.append(y.cpu())

    all_logits = torch.cat(all_logits, dim=0)  # every validation window's logits, in one tensor
    all_y = torch.cat(all_y, dim=0)
    metrics = compute_metrics(all_y, all_logits, threshold)  # computed ONCE, over the whole split
    metrics["loss"] = total_loss / n_batches
    return metrics


# =============================================================================
# 9. run_training -- the full loop, tying everything together
# =============================================================================

def run_training(model_config, train_config, windowed_dir=DEFAULT_WINDOWED_DIR, n_epochs=N_EPOCHS,
                  patience=EARLY_STOPPING_PATIENCE, device=DEVICE, files=None, verbose=True,
                  optimizer_cls=torch.optim.Adam):
    """Build everything, train with early stopping, return the BEST model and the full per-epoch history.

    Parameters
    ----------
    model_config : dict -- e.g. BEFORE_TUNING_CONFIG or TRIAL_13_MODEL_CONFIG, passed to build_ma1dcnn()
    train_config : dict -- e.g. TRAIN_BEFORE_TUNING_CONFIG or TRIAL_13_TRAIN_CONFIG
        (learning_rate, weight_decay, batch_size, threshold)
    optimizer_cls : type -- torch.optim.Adam (default) or torch.optim.AdamW. Adam applies
        weight_decay as coupled L2 (added into the gradient before Adam's adaptive
        per-parameter scaling); AdamW decouples it, applying weight_decay directly to the
        parameter instead (Loshchilov & Hutter, 2019). See overfitting_results_and_options.docx
        Section 4.2 and the "adamw" config above for why this was tried and not adopted.

    Returns
    -------
    model : MA1DCNN -- weights restored to the BEST epoch's checkpoint (highest val_f1_macro)
    history : list[dict] -- one record per epoch actually run: epoch, train_loss, and val_* for every metric
    """
    torch.manual_seed(RANDOM_SEED)  # reproducible weight init and (with shuffle=True) batch order

    train_loader, val_loader = build_dataloaders(
        windowed_dir=windowed_dir, batch_size=train_config["batch_size"], files=files)

    model = build_ma1dcnn(model_config).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optimizer_cls(model.parameters(), lr=train_config["learning_rate"],
                               weight_decay=train_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T_0, T_mult=SCHEDULER_T_MULT)

    history = []
    best_f1 = -1.0           # guaranteed to be beaten by epoch 1's real (>=0) f1_macro
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, threshold=train_config["threshold"])
        scheduler.step()  # advances the cosine schedule -- called once per EPOCH, not per batch

        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        if verbose:
            LOGGER.info(
                "epoch %3d/%d | train_loss %.4f | val_loss %.4f | val_f1_macro %.4f | val_hamming %.4f",
                epoch, n_epochs, train_loss, val_metrics["loss"], val_metrics["f1_macro"], val_metrics["hamming_loss"],
            )

        if val_metrics["f1_macro"] > best_f1:  # strictly greater -- a TIE does not count as improvement
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # clone -- a snapshot, not a live reference
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    LOGGER.info("early stopping: no val_f1_macro improvement for %d epochs", patience)
                break

    model.load_state_dict(best_state)  # return the BEST checkpoint, not whatever epoch the loop happened to end on
    return model, history


# =============================================================================
# 10. Synthetic self-checks -- ported from train.ipynb, run via --self-check
# =============================================================================

def _make_synthetic_windowed_files(windowed_dir, files, n_train, n_val, learnable=True, seed=0):
    rng = np.random.default_rng(seed)
    for filename in files:
        for split, n in [("train", n_train), ("val", n_val)]:
            if learnable:
                # Generative, not discriminative: draw y FIRST, then build channel j (j=0..4)
                # with a strong mean shift (+2.0 if y[:,j]==1 else -2.0) plus unit noise -- a
                # clear, reliably learnable signal for global-average-pooling + FC to pick up.
                y = rng.integers(0, 2, size=(n, 5)).astype(np.int8)
                X = rng.standard_normal((n, 28, 50)).astype(np.float32)
                for j in range(5):
                    shift = np.where(y[:, j] == 1, 2.0, -2.0).astype(np.float32)
                    X[:, j, :] += shift[:, None]
            else:
                # label has NOTHING to do with X -- unlearnable by construction.
                X = rng.standard_normal((n, 28, 50)).astype(np.float32)
                y = rng.integers(0, 2, size=(n, 5)).astype(np.int8)
            unit_id = np.arange(n)
            start_cycle = np.arange(n) * 50
            np.savez_compressed(_windowed_file_path(filename, windowed_dir, split),
                                 X=X, y=y, unit_id=unit_id, start_cycle=start_cycle)


def _check_windowed_fault_dataset_concatenates_all_files():
    with tempfile.TemporaryDirectory() as windowed_dir:
        files = ["FAKE_A.h5", "FAKE_B.h5"]
        sizes = {"FAKE_A.h5": 7, "FAKE_B.h5": 3}
        rng = np.random.default_rng(0)

        for filename, n in sizes.items():
            X = rng.random((n, 28, 50)).astype(np.float32)
            y = rng.integers(0, 2, size=(n, 5)).astype(np.int8)
            unit_id = np.arange(n)
            start_cycle = np.arange(n) * 50
            np.savez_compressed(_windowed_file_path(filename, windowed_dir, "train"),
                                 X=X, y=y, unit_id=unit_id, start_cycle=start_cycle)

        ds = WindowedFaultDataset(windowed_dir=windowed_dir, split="train", files=files)
        assert len(ds) == sum(sizes.values()), len(ds)
        assert ds.X.dtype == np.float32
        assert ds.y.dtype == np.float32

        x0, y0 = ds[0]
        assert x0.shape == (28, 50), x0.shape
        assert y0.shape == (5,), y0.shape
        assert x0.dtype == torch.float32
        assert y0.dtype == torch.float32

        x_last, _ = ds[len(ds) - 1]
        assert np.allclose(x_last.numpy(), ds.X[-1])

    LOGGER.info("WindowedFaultDataset: PASSED (concatenates all files, correct dtypes and shapes)")


def _check_build_dataloaders_shuffles_train_not_val():
    with tempfile.TemporaryDirectory() as windowed_dir:
        files = ["FAKE_A.h5"]
        rng = np.random.default_rng(1)
        n = 20
        for split in ["train", "val"]:
            X = rng.random((n, 28, 50)).astype(np.float32)
            y = rng.integers(0, 2, size=(n, 5)).astype(np.int8)
            np.savez_compressed(_windowed_file_path("FAKE_A.h5", windowed_dir, split),
                                 X=X, y=y, unit_id=np.arange(n), start_cycle=np.arange(n))

        train_loader, val_loader = build_dataloaders(windowed_dir=windowed_dir, batch_size=4, files=files)

        assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler), type(train_loader.sampler)
        assert isinstance(val_loader.sampler, torch.utils.data.SequentialSampler), type(val_loader.sampler)
        assert len(train_loader.dataset) == n
        assert len(val_loader.dataset) == n

    LOGGER.info("build_dataloaders: PASSED (train uses RandomSampler, val uses SequentialSampler)")


def _check_compute_metrics_known_cases():
    y_true = torch.tensor([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ], dtype=torch.float32)

    perfect_logits = torch.where(y_true == 1, torch.tensor(10.0), torch.tensor(-10.0))
    m = compute_metrics(y_true, perfect_logits, threshold=0.5)
    assert m["f1_macro"] == 1.0, m
    assert m["precision_macro"] == 1.0, m
    assert m["recall_macro"] == 1.0, m
    assert m["hamming_loss"] == 0.0, m
    assert m["jaccard_macro"] == 1.0, m

    wrong_logits = torch.where(y_true == 1, torch.tensor(-10.0), torch.tensor(10.0))
    m_wrong = compute_metrics(y_true, wrong_logits, threshold=0.5)
    assert m_wrong["f1_macro"] == 0.0, m_wrong
    assert m_wrong["hamming_loss"] == 1.0, m_wrong
    assert m_wrong["jaccard_macro"] == 0.0, m_wrong

    LOGGER.info("compute_metrics: PASSED (perfect prediction -> 1.0/0.0, fully inverted prediction -> 0.0/1.0)")


def _check_train_one_epoch_reduces_loss_on_a_tiny_batch():
    torch.manual_seed(0)
    config = dict(BEFORE_TUNING_CONFIG, channel_1=8, channel_2=16, channel_3=16, fc_units=8, dropout=0.0)
    model = build_ma1dcnn(config).to(DEVICE)

    X = torch.randn(16, 28, 50)
    y = (torch.rand(16, 5) > 0.5).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    first_loss = train_one_epoch(model, loader, optimizer, loss_fn, DEVICE)
    last_loss = first_loss
    for _ in range(19):
        last_loss = train_one_epoch(model, loader, optimizer, loss_fn, DEVICE)

    assert last_loss < first_loss, (first_loss, last_loss)
    assert last_loss == last_loss  # NaN != NaN -- this also fails loudly if training diverged

    LOGGER.info("train_one_epoch: PASSED (loss fell from %.4f to %.4f over 20 epochs on a fixed tiny batch)",
                first_loss, last_loss)


def _check_evaluate_returns_expected_keys_and_ranges():
    torch.manual_seed(0)
    model = build_ma1dcnn(dict(BEFORE_TUNING_CONFIG, channel_1=8, channel_2=16, channel_3=16, fc_units=8)).to(DEVICE)
    X = torch.randn(13, 28, 50)  # 13 -- deliberately not a multiple of batch_size, to exercise a ragged last batch
    y = (torch.rand(13, 5) > 0.5).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=4)

    metrics = evaluate(model, loader, nn.BCEWithLogitsLoss(), DEVICE, threshold=0.5)

    expected_keys = {"f1_macro", "precision_macro", "recall_macro", "hamming_loss", "jaccard_macro", "loss"}
    assert set(metrics.keys()) == expected_keys, metrics.keys()
    for key in ["f1_macro", "precision_macro", "recall_macro", "hamming_loss", "jaccard_macro"]:
        assert 0.0 <= metrics[key] <= 1.0, (key, metrics[key])
    assert metrics["loss"] > 0, metrics["loss"]

    LOGGER.info("evaluate: PASSED (expected keys present, all metrics in [0,1], handles a ragged final batch)")


def _check_run_training_learns_a_separable_synthetic_task():
    with tempfile.TemporaryDirectory() as windowed_dir:
        files = ["FAKE_A.h5", "FAKE_B.h5"]
        _make_synthetic_windowed_files(windowed_dir, files, n_train=200, n_val=60, learnable=True, seed=0)

        small_model_config = dict(BEFORE_TUNING_CONFIG, channel_1=8, channel_2=16, channel_3=16, fc_units=8, dropout=0.1)
        fast_train_config = dict(TRAIN_BEFORE_TUNING_CONFIG, learning_rate=1e-2, weight_decay=0.0, batch_size=32)

        model, history = run_training(small_model_config, fast_train_config, windowed_dir=windowed_dir,
                                       n_epochs=15, patience=5, files=files, verbose=True)

        assert 1 <= len(history) <= 15
        best_f1_from_history = max(h["val_f1_macro"] for h in history)
        assert best_f1_from_history > 0.7, (
            f"expected run_training to learn the separable synthetic task well, "
            f"got best val_f1_macro={best_f1_from_history}")

        _, val_loader = build_dataloaders(windowed_dir=windowed_dir, batch_size=32, files=files)
        recheck = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), DEVICE, threshold=fast_train_config["threshold"])
        assert abs(recheck["f1_macro"] - best_f1_from_history) < 1e-6, (recheck["f1_macro"], best_f1_from_history)

    LOGGER.info("run_training: PASSED (learned separable synthetic task, best val_f1_macro=%.4f, "
                "returned model confirmed to be the best checkpoint not just the last epoch)", best_f1_from_history)


def _check_run_training_early_stopping_triggers():
    with tempfile.TemporaryDirectory() as windowed_dir:
        files = ["FAKE_A.h5"]
        _make_synthetic_windowed_files(windowed_dir, files, n_train=40, n_val=20, learnable=True, seed=1)

        small_model_config = dict(BEFORE_TUNING_CONFIG, channel_1=8, channel_2=16, channel_3=16, fc_units=8)
        # learning_rate=0.0 -- weights never move, so val_f1_macro is IDENTICAL every epoch
        # after the first. With strict ">" for "improved", epoch 2 onward never counts as an
        # improvement, so patience=2 must stop training at exactly epoch 3.
        frozen_train_config = dict(TRAIN_BEFORE_TUNING_CONFIG, learning_rate=0.0, weight_decay=0.0, batch_size=8)

        model, history = run_training(small_model_config, frozen_train_config, windowed_dir=windowed_dir,
                                       n_epochs=10, patience=2, files=files, verbose=True)

        assert len(history) == 3, f"expected early stopping at exactly epoch 3, got {len(history)} epochs"

    LOGGER.info("run_training: PASSED (early stopping triggers at exactly the configured patience)")


def run_self_checks():
    """Run every synthetic sanity check ported from train.ipynb, in order. Raises on first failure."""
    _check_windowed_fault_dataset_concatenates_all_files()
    _check_build_dataloaders_shuffles_train_not_val()
    _check_compute_metrics_known_cases()
    _check_train_one_epoch_reduces_loss_on_a_tiny_batch()
    _check_evaluate_returns_expected_keys_and_ranges()
    _check_run_training_learns_a_separable_synthetic_task()
    _check_run_training_early_stopping_triggers()
    LOGGER.info("all self-checks PASSED")


# =============================================================================
# 11. CLI entry point
# =============================================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train MA1DCNN against real N-CMAPSS windowed data (default config: trial13).")
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME, choices=sorted(CONFIGS),
                         help=f"named config to train (default: {DEFAULT_CONFIG_NAME})")
    parser.add_argument("--windowed-dir", default=DEFAULT_WINDOWED_DIR,
                         help="directory containing windowing.ipynb's *_train_windowed.npz / "
                              "*_val_windowed.npz files (default: %(default)s, or $EDGE_FAULT_DX_WINDOWED_DIR)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="max epochs (default: %(default)s)")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE,
                         help="early-stopping patience in epochs (default: %(default)s)")
    parser.add_argument("--output", default=None,
                         help="checkpoint path to save to (default: ma1dcnn_<config>.pt)")
    parser.add_argument("--device", default=None, help="override auto-detected device, e.g. 'cpu' or 'cuda:0'")
    parser.add_argument("--list-configs", action="store_true", help="print every named config and exit")
    parser.add_argument("--self-check", action="store_true",
                         help="run the ported synthetic sanity checks against temp data, then exit")
    parser.add_argument("--quiet", action="store_true", help="suppress per-epoch logging")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.list_configs:
        for name, cfg in CONFIGS.items():
            marker = " (default)" if name == DEFAULT_CONFIG_NAME else ""
            LOGGER.info("%s%s: optimizer=%s, lr=%s, weight_decay=%s, batch_size=%s, threshold=%s",
                        name, marker, cfg["optimizer_cls"].__name__,
                        cfg["train_config"]["learning_rate"], cfg["train_config"]["weight_decay"],
                        cfg["train_config"]["batch_size"], cfg["train_config"]["threshold"])
        return

    if args.self_check:
        run_self_checks()
        return

    cfg = CONFIGS[args.config]
    device = torch.device(args.device) if args.device else DEVICE
    LOGGER.info("training config=%s on device=%s", args.config, device)

    model, history = run_training(
        model_config=cfg["model_config"], train_config=cfg["train_config"],
        windowed_dir=args.windowed_dir, n_epochs=args.epochs, patience=args.patience,
        device=device, verbose=not args.quiet, optimizer_cls=cfg["optimizer_cls"],
    )

    output_path = Path(args.output) if args.output else Path(f"ma1dcnn_{args.config}.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    best = max(history, key=lambda h: h["val_f1_macro"])
    LOGGER.info(
        "saved %s model weights to %s (best epoch %d/%d: val_f1_macro=%.4f, val_loss=%.4f, "
        "gap=%.4f)", args.config, output_path, best["epoch"], len(history), best["val_f1_macro"],
        best["val_loss"], best["val_loss"] - history[best["epoch"] - 1]["train_loss"],
    )


if __name__ == "__main__":
    main()
