"""
training/optuna_search.py -- Hyperparameter + architecture search for MA1DCNN (Phase 6)

The "What's next" item 3 from both ma1dcnn.ipynb and train.ipynb: reuses run_training's own
model_config/train_config arguments to search systematically, rather than the hand-picked,
one-config-at-a-time cells in train.ipynb's Section 10 (Baseline, Regularized, AdamW,
Conv-Dropout, BatchNorm).

## Build scope (same discipline as every notebook before this one)

Standalone: every piece below is repeated verbatim from train.ipynb/ma1dcnn.ipynb rather than
imported, so this file runs on its own -- no Drive, no real N-CMAPSS data needed to run or
understand anything except run_real_search() at the bottom. The two synthetic self-checks
copied from train.ipynb prove the repeated code still works standalone in THIS file, not just
trusted because it was already proven once in train.ipynb -- copy-paste is not proof.

## Design decisions specific to this file (documented, not silently made)

1. **Multi-objective, not a single scalarized score.** Optuna optimizes TWO objectives --
   maximize val_f1_macro, minimize the max generalization gap -- rather than combining them
   into one number with an arbitrary weight. This mirrors exactly how every run this session
   has actually been judged by hand (see overfitting_adamw_comparison.xlsx's Final Verdict:
   F1 and gap compared side by side, never blended into a single score). The result is a
   Pareto front (study.best_trials), not one "best" answer -- the same way Conv-Dropout
   (tightest gap) and BatchNorm (highest F1) were both defensible "best" runs depending on
   which axis you weight more.

2. **Multi-seed averaging per trial, not a single run.** This project has directly measured
   real run-to-run variance on CPU for every config tried twice (Baseline: 0.7895/0.7995/
   0.8050; Regularized: 0.8085/0.8212; AdamW: both runs bad but differently so; BatchNorm:
   0.8122/0.8274). A single-run objective would let Optuna's sampler chase that noise and
   crown a lucky seed as the "best" hyperparameter set. N_SEEDS_PER_TRIAL calls to
   run_training() per trial, averaged, directly addresses this -- at N_SEEDS_PER_TRIAL x the
   compute cost of a single run per trial.

   Note: run_training() always calls torch.manual_seed(RANDOM_SEED) with the SAME fixed seed
   internally (see run_training below) -- it is NOT re-seeded differently per call. This
   project has already empirically proven (every "Run 2" in overfitting_adamw_comparison.xlsx)
   that identical seeds still produce different real numbers on CPU, because multi-threaded
   BLAS/MKL ops are not bit-reproducible run to run. So simply calling run_training()
   N_SEEDS_PER_TRIAL times already captures that real variance -- no need to vary RANDOM_SEED
   itself to get independent samples, and no change to run_training's own signature was needed.

3. **No per-epoch pruning in this version.** Optuna's MedianPruner needs intermediate values
   reported mid-training (trial.report() inside the epoch loop), which would mean either
   modifying run_training() itself (already tested and used by every Section 10 cell in
   train.ipynb -- changing its internals risks everything built on top of it) or duplicating
   its training loop here (defeating the explicit goal of REUSING run_training, stated in both
   notebooks' "What's next" sections). Left as a documented limitation, not silently skipped:
   a future version could add an optional `on_epoch_end` callback parameter to run_training
   for this purpose, additive and backward-compatible the same way conv_dropout/use_batchnorm
   were added.

4. **Search space bounds come from two sources, named per-parameter below:** Table 4.4
   (Akindoju, 2025)'s own before/after tuning columns, widened slightly to give Optuna room to
   explore past what that original search found; and this session's own real findings (e.g.
   weight_decay=1e-3 outperformed Table 4.4's tuned 4.29e-06, so the range extends to cover
   both). conv_dropout and use_batchnorm are this session's additions, with no Table 4.4
   precedent -- their ranges come from what was actually tried (conv_dropout=0.2 in every real
   run so far).

5. **SQLite-backed persistence, not in-memory only.** A real 30-trial search can run for
   hours on Colab CPU, and an in-memory optuna.Study is gone the instant the runtime
   disconnects -- every trial completed so far would be lost with nothing to show for the
   compute spent. run_study's optional `storage` argument points Optuna at a `.db` file
   (`optuna.create_study(..., storage=storage, load_if_exists=True)`); each trial is written
   to it as soon as it completes, not just at the end. `n_trials` is then read as the TOTAL
   trial count the study should reach, not "run this many more" -- calling run_study again
   with the same `study_name`/`storage`/`n_trials` after a disconnect resumes toward that same
   target instead of restarting from trial 0 or overshooting it. storage=None (the default)
   keeps the original in-memory-only behavior, which is what every synthetic self-check below
   uses, so test runs never leave stray .db files behind.
"""

import os
import tempfile
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, jaccard_score

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "optuna is not installed. In Colab, run `!pip install optuna -q` in its own cell "
        "before this one, then re-run this cell."
    ) from exc


# ==============================================================================================
# 1. Constants -- repeated verbatim from ma1dcnn.ipynb / train.ipynb (standalone convention)
# ==============================================================================================
FAULT_COLUMNS = ["fan_fail", "lpc_fail", "hpc_fail", "hpt_fail", "lpt_fail"]
N_FAULT_COLUMNS = len(FAULT_COLUMNS)

NCMAPSS_FILES = [
    "N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS02-006.h5", "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS04.h5", "N-CMAPSS_DS05.h5", "N-CMAPSS_DS06.h5",
    "N-CMAPSS_DS07.h5", "N-CMAPSS_DS08a-009.h5", "N-CMAPSS_DS08c-008.h5",
]

DEFAULT_WINDOWED_DIR = "/content/drive/MyDrive/edge-ai-fault-diagnosis-aerospace/data/windowed"
CAM_REDUCTION_RATIO = 4
DEFAULT_N_INPUT_CHANNELS = 28

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
# This session's two real architecture changes (for reference/baseline comparison against the
# search below, not used to constrain it).
BEFORE_TUNING_CONFIG_CONV_DROPOUT = dict(BEFORE_TUNING_CONFIG, conv_dropout=0.2)
BEFORE_TUNING_CONFIG_CONV_DROPOUT_BN = dict(BEFORE_TUNING_CONFIG_CONV_DROPOUT, use_batchnorm=True)

TRAIN_BEFORE_TUNING_CONFIG = {
    "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 128, "threshold": 0.5, "use_focal": False,
}
TRAIN_REGULARIZED_CONFIG = {
    "learning_rate": 1e-3, "weight_decay": 1e-3, "batch_size": 128, "threshold": 0.5, "use_focal": False,
}

SCHEDULER_T_0 = 10
SCHEDULER_T_MULT = 2
N_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This file's own new constants.
N_SEEDS_PER_TRIAL = 2   # see design note 2 above -- raise if compute allows, for a less noisy objective
N_TRIALS = 30            # a starting point, not a tuned value -- raise/lower based on real wall-clock cost
# See design note 5 above -- on Drive so the study survives a disconnected Colab runtime.
DEFAULT_STUDY_DB = "/content/drive/MyDrive/edge-ai-fault-diagnosis-aerospace/data/ma1dcnn_search.db"


# ==============================================================================================
# 2. Model -- repeated verbatim from train.ipynb's Section 2 (which itself synced use_batchnorm
#    from ma1dcnn.ipynb). NOT re-tested here beyond the two run_training self-checks below --
#    already proven in both of those notebooks.
# ==============================================================================================
class Conv1dSamePadding(nn.Module):
    """1D convolution that keeps the sequence length unchanged (stride=1, odd kernel)."""

    def __init__(self, in_channels, out_channels, kernel_size, conv_dropout=0.0, use_batchnorm=False):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"Conv1dSamePadding needs an odd kernel_size, got {kernel_size}")
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.ReLU()
        self.conv_dropout = nn.Dropout(p=conv_dropout)

    def forward(self, x):
        return self.conv_dropout(self.activation(self.bn(self.conv(x))))


class ChannelAttentionModule(nn.Module):
    """CAM: one learned attention weight per CHANNEL, shared across all timesteps."""

    def __init__(self, n_channels, reduction_ratio=CAM_REDUCTION_RATIO):
        super().__init__()
        hidden = max(1, n_channels // reduction_ratio)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(n_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, n_channels), nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, length = x.shape
        squeezed = self.squeeze(x).view(batch, channels)
        channel_weights = self.excite(squeezed).view(batch, channels, 1)
        return x * channel_weights


class ElementwiseAttentionModule(nn.Module):
    """EAM: one learned attention weight per TIMESTEP, shared across all channels."""

    def __init__(self, n_channels):
        super().__init__()
        self.score = nn.Conv1d(in_channels=n_channels, out_channels=1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        attention = self.activation(self.score(x))
        return x * attention


class MA1DCNN(nn.Module):
    """3 conv blocks -> CAM -> EAM -> global average pooling -> FC head -> 5 logits."""

    def __init__(self, n_input_channels, channel_1, channel_2, channel_3,
                 kernel_size, fc_units, dropout, n_outputs,
                 cam_reduction_ratio=CAM_REDUCTION_RATIO, conv_dropout=0.0, use_batchnorm=False):
        super().__init__()
        self.conv1 = Conv1dSamePadding(n_input_channels, channel_1, kernel_size, conv_dropout, use_batchnorm)
        self.conv2 = Conv1dSamePadding(channel_1, channel_2, kernel_size, conv_dropout, use_batchnorm)
        self.conv3 = Conv1dSamePadding(channel_2, channel_3, kernel_size, conv_dropout, use_batchnorm)
        self.cam = ChannelAttentionModule(channel_3, reduction_ratio=cam_reduction_ratio)
        self.eam = ElementwiseAttentionModule(channel_3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channel_3, fc_units)
        self.fc_activation = nn.ReLU()
        self.output_layer = nn.Linear(fc_units, n_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.cam(x)
        x = self.eam(x)
        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc_activation(x)
        return self.output_layer(x)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def build_ma1dcnn(config):
    """Construct an MA1DCNN from a plain hyperparameter dict (e.g. BEFORE_TUNING_CONFIG)."""
    return MA1DCNN(
        n_input_channels=config["n_input_channels"], channel_1=config["channel_1"],
        channel_2=config["channel_2"], channel_3=config["channel_3"],
        kernel_size=config["kernel_size"], fc_units=config["fc_units"],
        dropout=config["dropout"], n_outputs=config["n_outputs"],
        conv_dropout=config.get("conv_dropout", 0.0),
        use_batchnorm=config.get("use_batchnorm", False),
    )


# ==============================================================================================
# 3. Data loading -- repeated verbatim from train.ipynb's Sections 4-5.
# ==============================================================================================
def _windowed_file_path(filename, windowed_dir, split):
    out_name = filename.replace(".h5", f"_{split}_windowed.npz")
    return os.path.join(windowed_dir, out_name)


class WindowedFaultDataset(torch.utils.data.Dataset):
    """All 9 files' windows for ONE split (train or val), concatenated into a single dataset."""

    def __init__(self, windowed_dir=DEFAULT_WINDOWED_DIR, split="train", files=None):
        files = files if files is not None else NCMAPSS_FILES
        X_parts, y_parts, unit_id_parts, start_cycle_parts = [], [], [], []

        for filename in files:
            path = _windowed_file_path(filename, windowed_dir, split)
            with np.load(path) as npz:
                X_parts.append(npz["X"])
                y_parts.append(npz["y"])
                unit_id_parts.append(npz["unit_id"])
                start_cycle_parts.append(npz["start_cycle"])

        self.X = np.concatenate(X_parts, axis=0).astype(np.float32)
        self.y = np.concatenate(y_parts, axis=0).astype(np.float32)
        self.unit_id = np.concatenate(unit_id_parts, axis=0)
        self.start_cycle = np.concatenate(start_cycle_parts, axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def build_dataloaders(windowed_dir=DEFAULT_WINDOWED_DIR, batch_size=TRAIN_BEFORE_TUNING_CONFIG["batch_size"],
                       files=None, num_workers=0):
    train_dataset = WindowedFaultDataset(windowed_dir=windowed_dir, split="train", files=files)
    val_dataset = WindowedFaultDataset(windowed_dir=windowed_dir, split="val", files=files)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ==============================================================================================
# 4. compute_metrics / train_one_epoch / evaluate / run_training -- repeated verbatim from
#    train.ipynb's Sections 6-9.
# ==============================================================================================
def compute_metrics(y_true, logits, threshold):
    probs = torch.sigmoid(logits)
    y_pred = (probs >= threshold).to(torch.int64).cpu().numpy()
    y_true_np = y_true.to(torch.int64).cpu().numpy()

    return {
        "f1_macro": f1_score(y_true_np, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true_np, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_np, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true_np, y_pred),
        "jaccard_macro": jaccard_score(y_true_np, y_pred, average="macro", zero_division=0),
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, loader, loss_fn, device, threshold):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_logits, all_y = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)
    metrics = compute_metrics(all_y, all_logits, threshold)
    metrics["loss"] = total_loss / n_batches
    return metrics


def run_training(model_config, train_config, windowed_dir=DEFAULT_WINDOWED_DIR, n_epochs=N_EPOCHS,
                  patience=EARLY_STOPPING_PATIENCE, device=DEVICE, files=None, verbose=True,
                  optimizer_cls=torch.optim.Adam):
    """Build everything, train with early stopping, return the BEST model and the full per-epoch history."""
    torch.manual_seed(RANDOM_SEED)

    train_loader, val_loader = build_dataloaders(
        windowed_dir=windowed_dir, batch_size=train_config["batch_size"], files=files)

    model = build_ma1dcnn(model_config).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optimizer_cls(model.parameters(), lr=train_config["learning_rate"],
                               weight_decay=train_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T_0, T_mult=SCHEDULER_T_MULT)

    history = []
    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, threshold=train_config["threshold"])
        scheduler.step()

        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        if verbose:
            print(f"epoch {epoch:>3}/{n_epochs} | train_loss {train_loss:.4f} | val_loss {val_metrics['loss']:.4f} "
                  f"| val_f1_macro {val_metrics['f1_macro']:.4f} | val_hamming {val_metrics['hamming_loss']:.4f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"early stopping: no val_f1_macro improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    return model, history


# ==============================================================================================
# 5. Synthetic data for self-checks -- repeated verbatim from train.ipynb.
# ==============================================================================================
def _make_synthetic_windowed_files(windowed_dir, files, n_train, n_val, learnable=True, seed=0):
    rng = np.random.default_rng(seed)
    for filename in files:
        for split, n in [("train", n_train), ("val", n_val)]:
            if learnable:
                y = rng.integers(0, 2, size=(n, 5)).astype(np.int8)
                X = rng.standard_normal((n, 28, 50)).astype(np.float32)
                for j in range(5):
                    shift = np.where(y[:, j] == 1, 2.0, -2.0).astype(np.float32)
                    X[:, j, :] += shift[:, None]
            else:
                X = rng.standard_normal((n, 28, 50)).astype(np.float32)
                y = rng.integers(0, 2, size=(n, 5)).astype(np.int8)
            unit_id = np.arange(n)
            start_cycle = np.arange(n) * 50
            np.savez_compressed(_windowed_file_path(filename, windowed_dir, split),
                                 X=X, y=y, unit_id=unit_id, start_cycle=start_cycle)


def _test_run_training_learns_a_separable_synthetic_task():
    """Re-proves the standalone copy of run_training above still works in THIS file."""
    with tempfile.TemporaryDirectory() as windowed_dir:
        files = ["FAKE_A.h5", "FAKE_B.h5"]
        _make_synthetic_windowed_files(windowed_dir, files, n_train=200, n_val=60, learnable=True, seed=0)

        small_model_config = dict(BEFORE_TUNING_CONFIG, channel_1=8, channel_2=16, channel_3=16, fc_units=8, dropout=0.1)
        fast_train_config = dict(TRAIN_BEFORE_TUNING_CONFIG, learning_rate=1e-2, weight_decay=0.0, batch_size=32)

        model, history = run_training(small_model_config, fast_train_config, windowed_dir=windowed_dir,
                                       n_epochs=15, patience=5, files=files, verbose=False)

        assert 1 <= len(history) <= 15
        best_f1_from_history = max(h["val_f1_macro"] for h in history)
        assert best_f1_from_history > 0.7, f"expected run_training to learn the separable synthetic task well, got best val_f1_macro={best_f1_from_history}"

        _, val_loader = build_dataloaders(windowed_dir=windowed_dir, batch_size=32, files=files)
        recheck = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), DEVICE, threshold=fast_train_config["threshold"])
        assert abs(recheck["f1_macro"] - best_f1_from_history) < 1e-6, (recheck["f1_macro"], best_f1_from_history)

    print(f"run_training (standalone copy in optuna_search.py): PASSED (learned separable synthetic task, "
          f"best val_f1_macro={best_f1_from_history:.4f})")


def _test_run_training_early_stopping_triggers():
    with tempfile.TemporaryDirectory() as windowed_dir:
        files = ["FAKE_A.h5"]
        _make_synthetic_windowed_files(windowed_dir, files, n_train=40, n_val=20, learnable=True, seed=1)

        small_model_config = dict(BEFORE_TUNING_CONFIG, channel_1=8, channel_2=16, channel_3=16, fc_units=8)
        frozen_train_config = dict(TRAIN_BEFORE_TUNING_CONFIG, learning_rate=0.0, weight_decay=0.0, batch_size=8)

        model, history = run_training(small_model_config, frozen_train_config, windowed_dir=windowed_dir,
                                       n_epochs=10, patience=2, files=files, verbose=False)

        assert len(history) == 3, f"expected early stopping at exactly epoch 3, got {len(history)} epochs"

    print("run_training (standalone copy in optuna_search.py): PASSED (early stopping triggers at exactly the configured patience)")


# ==============================================================================================
# 6. The search itself -- NEW in this file.
# ==============================================================================================
def objective(trial, windowed_dir=DEFAULT_WINDOWED_DIR, files=None,
              n_epochs=N_EPOCHS, patience=EARLY_STOPPING_PATIENCE, n_seeds=N_SEEDS_PER_TRIAL):
    """One Optuna trial: sample a (model_config, train_config), train it n_seeds times, return
    (mean best val_f1_macro, mean max generalization gap) -- see design notes 1 and 2 at the top
    of this file for why both are multi-valued rather than one scalarized score, and why
    n_seeds > 1 rather than trusting a single run.

    channel_1=32 and channel_3=128 are held fixed, matching Table 4.4's own before/after
    columns (neither value changed between Optuna's before- and after-tuning trials there) --
    searching them too would blow up the space without evidence they matter.
    """
    model_config = dict(
        BEFORE_TUNING_CONFIG,
        # Table 4.4: Channel 2 was 64 before tuning, 128 after -- both kept as the only two
        # validated options rather than an arbitrary continuous range.
        channel_2=trial.suggest_categorical("channel_2", [64, 128]),
        # Table 4.4: Kernel was 7 before tuning, 5 after -- Conv1dSamePadding requires odd,
        # both already are.
        kernel_size=trial.suggest_categorical("kernel_size", [5, 7]),
        # Table 4.4: fc_units was 64 before tuning, 16 after -- 32 added as an untried midpoint.
        fc_units=trial.suggest_categorical("fc_units", [16, 32, 64]),
        # Table 4.4: Dropout was 0.5 before tuning, 0.39919 after -- widened slightly either side.
        dropout=trial.suggest_float("dropout", 0.2, 0.6),
        # This session's own addition -- conv_dropout=0.2 is the only value actually run for
        # real so far (Conv-Dropout Run 1, BatchNorm Runs 1-2); 0.0 means off.
        conv_dropout=trial.suggest_float("conv_dropout", 0.0, 0.4),
        # This session's own addition -- both values have now been run for real (conv-dropout
        # alone vs. layered with BatchNorm1d), with the honest tradeoff documented in
        # overfitting_adamw_comparison.xlsx: BatchNorm raised peak F1 but widened the gap on
        # both of its runs. Left open here rather than pre-deciding the answer.
        use_batchnorm=trial.suggest_categorical("use_batchnorm", [False, True]),
    )
    train_config = {
        # Table 4.4: Learning Rate was 1e-3 before tuning, 0.00019963 after -- log-uniform
        # across that range plus a little headroom on each side.
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        # Table 4.4: Weight Decay was 1e-4 before tuning, 4.29e-06 after -- but this session's
        # OWN hand experiments (Regularized, Conv-Dropout, BatchNorm) all used 1e-3 and
        # outperformed the untuned baseline, so the upper bound extends to 1e-2 to cover that
        # empirically-useful region Table 4.4's own search never visited.
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        # Table 4.4: Batch Size was 128 before tuning, 256 after -- 64 added as an untried,
        # smaller option.
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        # Table 4.4: Threshold was 0.5 before tuning, 0.76957 after.
        "threshold": trial.suggest_float("threshold", 0.3, 0.8),
        # Table 4.4: Use Focal is FALSE in both the before- and after-tuning best trials
        # (train_design_questions.docx already settled this) -- not reopened here.
        "use_focal": False,
    }

    f1_scores, max_gaps = [], []
    for _ in range(n_seeds):
        _, history = run_training(model_config, train_config, windowed_dir=windowed_dir,
                                   n_epochs=n_epochs, patience=patience, device=DEVICE,
                                   files=files, verbose=False)
        f1_scores.append(max(h["val_f1_macro"] for h in history))
        max_gaps.append(max(h["val_loss"] - h["train_loss"] for h in history))

    # Logged per-seed, not just averaged -- so a trial that looks good on average but was
    # actually wildly inconsistent (like BatchNorm's two real runs) is still visible later,
    # the same honesty this project's hand-run experiments have been held to throughout.
    trial.set_user_attr("f1_scores_per_seed", f1_scores)
    trial.set_user_attr("max_gaps_per_seed", max_gaps)

    return sum(f1_scores) / len(f1_scores), sum(max_gaps) / len(max_gaps)


def run_study(n_trials=N_TRIALS, windowed_dir=DEFAULT_WINDOWED_DIR, files=None,
              n_epochs=N_EPOCHS, patience=EARLY_STOPPING_PATIENCE, n_seeds=N_SEEDS_PER_TRIAL,
              study_name="ma1dcnn_search", storage=None):
    """Run the real (or a test-scale) search and return the completed optuna.Study.

    study.best_trials is the Pareto front (maximize val_f1_macro, minimize max gap) -- there is
    deliberately no single "best_params", per design note 1 at the top of this file.

    storage: an Optuna storage URL, e.g. f"sqlite:///{DEFAULT_STUDY_DB}" -- see design note 5
    at the top of this file. When given, trials persist to that file as they complete and
    n_trials is the TOTAL trial count the study should reach: calling run_study again with the
    same study_name/storage/n_trials (after a disconnect, or just to check progress) resumes
    toward that same target rather than restarting from trial 0 or running n_trials MORE on
    top of what's already there. None (the default) is the original in-memory-only behavior,
    used by every synthetic self-check in this file so test runs leave no .db files behind.
    """
    bound_objective = partial(objective, windowed_dir=windowed_dir, files=files,
                               n_epochs=n_epochs, patience=patience, n_seeds=n_seeds)
    study = optuna.create_study(directions=["maximize", "minimize"], study_name=study_name,
                                 storage=storage, load_if_exists=storage is not None)

    already_done = len(study.trials)
    remaining = n_trials - already_done
    if remaining <= 0:
        print(f"{study_name}: already has {already_done} trials recorded"
              + (f" in {storage}" if storage else "") + f" -- nothing to run (n_trials={n_trials}).")
        return study
    if storage is not None and already_done > 0:
        print(f"{study_name}: resuming from {storage} -- {already_done} trials already recorded, "
              f"running {remaining} more to reach {n_trials}.")

    study.optimize(bound_objective, n_trials=remaining)
    return study


def print_pareto_front(study):
    """Readable summary of study.best_trials -- the Pareto-optimal (F1, gap) tradeoffs found."""
    print(f"{len(study.trials)} trials run, {len(study.best_trials)} on the Pareto front:\n")
    for t in sorted(study.best_trials, key=lambda t: -t.values[0]):
        print(f"  trial {t.number:>3} | val_f1_macro={t.values[0]:.4f} | max_gap={t.values[1]:+.4f} | params={t.params}")


# ==============================================================================================
# 7. Self-check for the search wiring itself -- a tiny, REAL (not mocked) Optuna study against
#    synthetic in-memory data, proving objective()/run_study() work end to end before trusting
#    them against real data and real compute cost.
# ==============================================================================================
def _test_run_study_completes_on_synthetic_data():
    with tempfile.TemporaryDirectory() as windowed_dir:
        files = ["FAKE_A.h5", "FAKE_B.h5"]
        _make_synthetic_windowed_files(windowed_dir, files, n_train=200, n_val=60, learnable=True, seed=2)

        study = run_study(n_trials=3, windowed_dir=windowed_dir, files=files,
                           n_epochs=3, patience=2, n_seeds=1, study_name="synthetic_self_check")

        assert len(study.trials) == 3, f"expected 3 completed trials, got {len(study.trials)}"
        for t in study.trials:
            assert t.state == optuna.trial.TrialState.COMPLETE, (t.number, t.state)
            f1, gap = t.values
            assert 0.0 <= f1 <= 1.0, f"val_f1_macro out of [0,1] range: {f1}"
            assert isinstance(gap, float), f"max gap should be a float, got {type(gap)}"
            assert "f1_scores_per_seed" in t.user_attrs and "max_gaps_per_seed" in t.user_attrs

        assert len(study.best_trials) >= 1, "expected at least one Pareto-optimal trial"

    print(f"run_study: PASSED (3/3 trials completed on synthetic data, "
          f"{len(study.best_trials)} on the Pareto front, both objectives well-formed)")


def _test_run_study_resumes_from_sqlite_storage():
    """Proves design note 5's resume semantics for real: a second run_study() call against the
    same sqlite file and study_name picks up where the first left off, rather than restarting
    from zero or running n_trials MORE trials on top of what's already recorded."""
    with tempfile.TemporaryDirectory() as windowed_dir:
        files = ["FAKE_A.h5", "FAKE_B.h5"]
        _make_synthetic_windowed_files(windowed_dir, files, n_train=200, n_val=60, learnable=True, seed=3)
        db_path = os.path.join(windowed_dir, "resume_self_check.db")
        storage = f"sqlite:///{db_path}"
        common_kwargs = dict(windowed_dir=windowed_dir, files=files, n_epochs=2, patience=2,
                              n_seeds=1, study_name="resume_self_check", storage=storage)

        study1 = run_study(n_trials=2, **common_kwargs)
        assert len(study1.trials) == 2, f"expected 2 trials after the first call, got {len(study1.trials)}"

        study2 = run_study(n_trials=4, **common_kwargs)
        assert len(study2.trials) == 4, (
            f"expected run_study to resume to 4 TOTAL trials (2 already recorded + 2 more), "
            f"got {len(study2.trials)} -- either it restarted from zero or it ran 4 more on top")
        # The first 2 trials' numbers/values must be unchanged, not re-run.
        for t1, t2 in zip(study1.trials, study2.trials[:2]):
            assert t1.number == t2.number and t1.values == t2.values, "resumed study altered an already-completed trial"

        study3 = run_study(n_trials=4, **common_kwargs)
        assert len(study3.trials) == 4, f"expected zero new trials when already at the n_trials target, got {len(study3.trials)}"

    print("run_study: PASSED (sqlite storage persists trials across calls and resumes toward "
          "the n_trials target without re-running or duplicating completed trials)")


if __name__ == "__main__":
    # Same discipline as every notebook in this project: prove it standalone on synthetic data
    # FIRST. These four checks run in well under a minute on CPU -- no Drive, no real data.
    _test_run_training_learns_a_separable_synthetic_task()
    _test_run_training_early_stopping_triggers()
    _test_run_study_completes_on_synthetic_data()
    _test_run_study_resumes_from_sqlite_storage()
    print("\nAll self-checks passed.")

    # The real search is NOT run automatically -- it needs Drive mounted (DEFAULT_WINDOWED_DIR)
    # and costs N_TRIALS x N_SEEDS_PER_TRIAL full run_training() calls (at up to N_EPOCHS each).
    # In Colab, after mounting Drive, call explicitly:
    #
    #   study = run_study(n_trials=30, storage=f"sqlite:///{DEFAULT_STUDY_DB}")
    #   print_pareto_front(study)
    #
    # If the runtime disconnects partway through, just re-run the same cell -- run_study sees
    # the trials already recorded in DEFAULT_STUDY_DB and resumes toward n_trials=30 instead of
    # starting over.
