"""Normalize, augment, and split loaders.py's saved data for training
(Phase 2).

Turns loaders.py's already-labeled, per-file .npz output into data the
MA1DCNN can actually train on:

    1. Per-sensor-channel z-score normalization (W, X_s, X_v only).
    2. A time-ordered train/validation split, per whole engine unit
       (never by individual row -- see split_units_train_val()).
    3. Optional Gaussian + impulse noise augmentation on the training
       split only, citing Liao et al. (2024)'s synthetic-to-real-gap
       precedent for the -6dB default.

Design notes worth keeping in mind while reading this file:
  * This module only ever READS loaders.py's output (DEFAULT_PROCESSED_DIR)
    and only ever WRITES its own, separate output (DEFAULT_PREPROCESSED_DIR)
    -- loaders.py's saved files are never at risk of being overwritten here.
  * Every function follows the same chunked, one-file-at-a-time discipline
    loaders.py established, for the same reason: holding all 9 files'
    rows in memory at once is what crashed the original pipeline (see
    loaders.py's module docstring for the full story).
  * A (unit/cycle/Fc/hs), Y (RUL), and labels are never normalized or
    augmented -- they are identifiers, a regression target, and binary
    flags, not continuous measurements a z-score or injected noise would
    meaningfully apply to.
"""

from __future__ import annotations

import os

import numpy as np

from edge_fault_dx.data.loaders import NCMAPSS_FILES

# ---------------------------------------------------------------------------
# 1. Constants -- every judgment call in this module (what counts as a
#    channel to normalize, how big the validation split is, how harsh the
#    training noise is) is named here, once, rather than buried in a
#    function body.
# ---------------------------------------------------------------------------

# Where loaders.py already saved its labeled, per-file .npz output. This
# module only ever READS from here, never writes here.
DEFAULT_PROCESSED_DIR = "/content/drive/MyDrive/edge-ai-fault-diagnosis-aerospace/data/processed"

# Where THIS module's own output (normalized, split, optionally
# noise-augmented) gets saved -- a separate folder, so loaders.py's output
# is never at risk of being overwritten.
DEFAULT_PREPROCESSED_DIR = "/content/drive/MyDrive/edge-ai-fault-diagnosis-aerospace/data/preprocessed"

# Which groups get z-score normalized. A (unit/cycle/Fc/hs) is
# bookkeeping, Y (RUL) is only ever used to build the label, and labels
# are already binary 0/1 -- normalizing any of those three would not
# make sense.
NORMALIZE_GROUPS: list[str] = ["W", "X_s", "X_v"]

# Column layout of the A group, per docs/data-dictionary.md: column 0 is
# the engine unit ID -- what the train/validation split below groups by.
A_UNIT_COLUMN = 0

# Fraction of each file's DEV units held out for validation. This is a
# judgment call (20% is a common default), not a value derived from
# Akindoju's thesis or the N-CMAPSS papers.
VALIDATION_UNIT_FRACTION = 0.2

# Training-time noise-augmentation level, in dB SNR. -6dB is the
# synthetic-to-real-gap precedent cited above (Liao et al., 2024). It is
# intentionally much harsher than the 0/10/20dB levels
# docs/praxis/04-hypotheses.md (H1) later tests the TRAINED model
# against -- augmentation noise should be at least as harsh as
# evaluation noise, or the model never actually learns to handle it.
AUGMENT_GAUSSIAN_SNR_DB = -6.0

# What fraction of individual sensor readings get replaced with an
# impulse (sensor-glitch) spike during training augmentation, and how
# large that spike is, in units of that channel's own standard
# deviation. Both are judgment calls -- there is no cited source for
# these two specific numbers.
AUGMENT_IMPULSE_FRACTION = 0.01
AUGMENT_IMPULSE_MAGNITUDE = 3.0


# ---------------------------------------------------------------------------
# 2. Reading loaders.py's saved output
# ---------------------------------------------------------------------------

def load_processed_file(
    filename: str,
    out_dir: str = DEFAULT_PROCESSED_DIR,
    split: str = "dev",
) -> dict[str, np.ndarray]:
    """Load ONE of loaders.py's already-saved, labeled .npz files.

    Mirrors loaders.py's own function of the same name -- every other
    function in this module reads through this one, never through a raw
    np.load() call scattered around, so there is exactly one place that
    knows the on-disk file-naming convention.
    """
    out_name = filename.replace(".h5", f"_{split}.npz")
    out_path = os.path.join(out_dir, out_name)
    with np.load(out_path) as npz:
        return {key: npz[key] for key in npz.files}


def iter_processed_files(
    out_dir: str = DEFAULT_PROCESSED_DIR,
    split: str = "dev",
    files: list[str] | None = None,
):
    """Yield one processed file's data at a time -- never all 9 in memory at once."""
    files = files if files is not None else NCMAPSS_FILES
    for filename in files:
        yield filename, load_processed_file(filename, out_dir=out_dir, split=split)


# ---------------------------------------------------------------------------
# 3. Time-ordered train/validation split, per engine unit
# ---------------------------------------------------------------------------

def split_units_train_val(
    unit_ids: np.ndarray,
    val_fraction: float = VALIDATION_UNIT_FRACTION,
):
    """Split one file's rows into train/validation by WHOLE engine unit.

    Why by unit, not by row: consecutive rows within one unit's
    run-to-failure trajectory are highly correlated -- they are the same
    simulated aircraft, minutes apart. Splitting individual rows randomly
    would let the model see one cycle of a unit in training and the very
    next cycle of that SAME unit in validation: an easy, unrealistic
    shortcut that would make validation accuracy look far better than
    real generalization. Keeping every unit's rows entirely in one split
    avoids that leakage.

    Parameters
    ----------
    unit_ids : np.ndarray, shape (n_rows,)
        The A[:, 0] column for one file -- which unit each row belongs to.
    val_fraction : float
        Fraction of this file's DISTINCT units (not rows) held out.

    Returns
    -------
    train_mask, val_mask : np.ndarray[bool], shape (n_rows,)
        Row-aligned boolean masks. Every row is in exactly one of the two.
    train_units, val_units : np.ndarray
        The distinct unit IDs assigned to each split, for logging/checks.
    """
    unique_units = np.unique(unit_ids)  # ascending, deduplicated

    if len(unique_units) < 2:
        # Too few units to hold any out -- everything goes to train
        # rather than raising an error, so this still works on tiny
        # synthetic test fixtures with a single unit.
        train_mask = np.ones(unit_ids.shape[0], dtype=bool)
        val_mask = np.zeros(unit_ids.shape[0], dtype=bool)
        return train_mask, val_mask, unique_units, unique_units[:0]

    n_val = max(1, round(val_fraction * len(unique_units)))
    val_units = unique_units[-n_val:]     # deterministic: always the LAST n_val unit IDs
    train_units = unique_units[:-n_val]   # everything else

    train_mask = np.isin(unit_ids, train_units)
    val_mask = np.isin(unit_ids, val_units)

    return train_mask, val_mask, train_units, val_units


# ---------------------------------------------------------------------------
# 4. Streaming per-channel normalization statistics (training rows only)
# ---------------------------------------------------------------------------

def compute_channel_stats(
    out_dir: str = DEFAULT_PROCESSED_DIR,
    files: list[str] | None = None,
    groups: list[str] = NORMALIZE_GROUPS,
    val_fraction: float = VALIDATION_UNIT_FRACTION,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-channel mean/std across ALL files' TRAINING rows only.

    Uses the single-pass identity Var(X) = E[X^2] - E[X]^2 so a second
    pass over the data is never needed -- three small running totals
    (count, sum, sum-of-squares) are accumulated one file at a time,
    combined into mean/std only once, at the end.

    Why training rows only: if validation rows leaked into these
    statistics, the model would be implicitly "peeking" at data it is
    later evaluated on, inflating validation performance in a way that
    would not hold up on genuinely new data.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        e.g. stats["W"]["mean"], stats["W"]["std"] -- one mean/std
        vector per channel within each group.
    """
    files = files if files is not None else NCMAPSS_FILES

    count = 0
    sums: dict[str, np.ndarray | None] = {g: None for g in groups}
    sums_sq: dict[str, np.ndarray | None] = {g: None for g in groups}

    for filename in files:
        data = load_processed_file(filename, out_dir=out_dir)  # one file's arrays in memory

        unit_ids = data["A"][:, A_UNIT_COLUMN]
        train_mask, _, _, _ = split_units_train_val(unit_ids, val_fraction)

        for g in groups:
            train_rows = data[g][train_mask]  # only this file's TRAIN rows contribute
            if sums[g] is None:
                sums[g] = train_rows.sum(axis=0)
                sums_sq[g] = (train_rows ** 2).sum(axis=0)
            else:
                sums[g] += train_rows.sum(axis=0)
                sums_sq[g] += (train_rows ** 2).sum(axis=0)

        count += int(train_mask.sum())
        del data  # free this file's arrays before loading the next one

    stats = {}
    for g in groups:
        mean = sums[g] / count
        var = sums_sq[g] / count - mean ** 2   # single-pass variance identity
        std = np.sqrt(np.maximum(var, 0))      # clip tiny negative values from floating-point rounding
        std = np.where(std < 1e-8, 1.0, std)   # a ~zero-variance channel would otherwise divide by ~zero
        stats[g] = {"mean": mean, "std": std}

    return stats


# ---------------------------------------------------------------------------
# 5. Applying normalization
# ---------------------------------------------------------------------------

def apply_normalization(
    data: dict[str, np.ndarray],
    stats: dict[str, dict[str, np.ndarray]],
    groups: list[str] = NORMALIZE_GROUPS,
) -> dict[str, np.ndarray]:
    """Return a NEW dict with `groups` z-score normalized; everything else untouched."""
    out = dict(data)  # shallow copy -- untouched keys share the same array object, which is fine since we never mutate them
    for g in groups:
        out[g] = (data[g] - stats[g]["mean"]) / stats[g]["std"]
    return out


# ---------------------------------------------------------------------------
# 6. Noise augmentation for training data (Gaussian + impulse)
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    x: np.ndarray,
    snr_db: float = AUGMENT_GAUSSIAN_SNR_DB,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise to every channel at a target SNR (dB).

    SNR_dB = 10 * log10(signal_power / noise_power), rearranged to solve
    for noise_power:
        noise_power = signal_power / 10^(SNR_dB / 10)

    Signal power is measured PER CHANNEL (per column), so a channel with
    a larger natural scale gets proportionally more noise -- this keeps
    the requested SNR accurate on every channel, not just on average
    across all of them.
    """
    rng = rng or np.random.default_rng()
    signal_power = np.mean(x ** 2, axis=0)              # shape: (n_channels,)
    noise_power = signal_power / (10 ** (snr_db / 10))  # rearranged SNR formula, solved for noise power
    sigma = np.sqrt(noise_power)
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + noise


def add_impulse_noise(
    x: np.ndarray,
    fraction: float = AUGMENT_IMPULSE_FRACTION,
    magnitude: float = AUGMENT_IMPULSE_MAGNITUDE,
    rng: np.random.Generator | None = None,
):
    """Corrupt a random fraction of individual readings with a large spike.

    Returns
    -------
    x_out : np.ndarray, same shape as x
    corrupted_mask : np.ndarray[bool], same shape as x
        True at every position that got a spike -- returned mainly so
        tests (and curious readers) can check the actual corruption rate.
    """
    rng = rng or np.random.default_rng()
    channel_std = x.std(axis=0)  # per-channel scale, so a spike is meaningful relative to that channel
    corrupted_mask = rng.random(x.shape) < fraction
    spike_sign = rng.choice([-1.0, 1.0], size=x.shape)
    spikes = spike_sign * magnitude * channel_std  # broadcasts (n_channels,) across all rows
    x_out = np.where(corrupted_mask, x + spikes, x)
    return x_out, corrupted_mask


# ---------------------------------------------------------------------------
# 7. Putting it together
# ---------------------------------------------------------------------------

def process_and_save_preprocessed(
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    out_dir: str = DEFAULT_PREPROCESSED_DIR,
    files: list[str] | None = None,
    val_fraction: float = VALIDATION_UNIT_FRACTION,
    augment: bool = True,
):
    """Turn loaders.py's saved dev-split output into train/val/test-ready data.

    Two passes over the files, exactly like loaders.py's own chunked
    design -- never holding more than one file's arrays in memory:

    - Pass 1 streams every file once, computes each file's train/
      validation unit split, and accumulates per-channel normalization
      stats from TRAINING rows only (compute_channel_stats()).
    - Pass 2 streams every file again, applies the now-known stats to
      normalize W/X_s/X_v, adds noise augmentation to the TRAINING rows
      only, and saves separate train/validation .npz files per input file.

    If a matching TEST-split file also exists on disk (i.e. loaders.py has
    already been run with split="test"), it is normalized with the same
    training-derived stats and saved as-is -- no further splitting, no
    augmentation, since it represents genuinely held-out data. That
    branch is skipped gracefully (not an error) if no test-split file
    exists yet.

    Returns
    -------
    summaries : list[dict]
        One small summary dict per file -- row counts and output paths,
        never the actual data arrays.
    stats : dict[str, dict[str, np.ndarray]]
        The training-derived normalization statistics used, in case a
        caller wants to log or persist them (e.g. for reuse at inference
        time).
    """
    files = files if files is not None else NCMAPSS_FILES
    os.makedirs(out_dir, exist_ok=True)

    stats = compute_channel_stats(out_dir=processed_dir, files=files, val_fraction=val_fraction)

    rng = np.random.default_rng(0)  # fixed seed -- augmented noise is reproducible run to run
    summaries = []

    for filename in files:
        data = load_processed_file(filename, out_dir=processed_dir)  # one file's arrays in memory

        unit_ids = data["A"][:, A_UNIT_COLUMN]
        train_mask, val_mask, _, _ = split_units_train_val(unit_ids, val_fraction)

        normalized = apply_normalization(data, stats)
        train_data = {k: v[train_mask] for k, v in normalized.items()}
        val_data = {k: v[val_mask] for k, v in normalized.items()}

        if augment:
            train_data["W"] = add_gaussian_noise(train_data["W"], rng=rng)
            train_data["X_s"], _ = add_impulse_noise(train_data["X_s"], rng=rng)

        base_name = filename.replace(".h5", "")
        train_path = os.path.join(out_dir, f"{base_name}_train.npz")
        val_path = os.path.join(out_dir, f"{base_name}_val.npz")
        np.savez_compressed(train_path, **train_data)
        np.savez_compressed(val_path, **val_data)

        summary = {
            "filename": filename,
            "n_train_rows": int(train_mask.sum()),
            "n_val_rows": int(val_mask.sum()),
            "train_path": train_path,
            "val_path": val_path,
        }

        # Test-split file is OPTIONAL, per the note above -- skipped
        # gracefully, not an error, if it does not exist yet.
        test_name = filename.replace(".h5", "_test.npz")
        test_in_path = os.path.join(processed_dir, test_name)
        if os.path.exists(test_in_path):
            with np.load(test_in_path) as npz:
                test_data = {k: npz[k] for k in npz.files}
            test_normalized = apply_normalization(test_data, stats)  # same train-derived stats, no augmentation
            test_out_path = os.path.join(out_dir, f"{base_name}_test.npz")
            np.savez_compressed(test_out_path, **test_normalized)
            summary["n_test_rows"] = test_data["W"].shape[0]
            summary["test_path"] = test_out_path

        summaries.append(summary)
        del data, normalized, train_data, val_data  # free this file's memory before the next iteration

    return summaries, stats


if __name__ == "__main__":
    # Quick manual run when executing this file directly in Colab, after
    # loaders.py has already been run: normalizes, splits, and saves the
    # real train/validation data.
    process_and_save_preprocessed()
