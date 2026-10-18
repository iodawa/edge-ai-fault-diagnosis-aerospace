"""Windowing -- src/edge_fault_dx/data/windowing.py (Phase 2, step 3).

Ported from windowing.ipynb, where every function below was validated
against its own synthetic self-checks (see run_self_checks() / --self-check)
and then run for real against N-CMAPSS preprocessed data (Section 9 of that
notebook: 710,036 train windows / 175,177 val windows across the 9 source
files -- the numbers already cited throughout doc_final.docx and
praxis_writeup.docx). Same "prove it standalone first" discipline as
train.py's own module docstring describes, and the same relationship: this
file carries the code, not the narrative. The four judgment calls below --
window length/stride, the "last" vs "any" label-collapsing rule, whether W
becomes extra input channels, and skip-not-pad for short units -- are
discussed in pipeline_sequence_and_dependencies.docx and in windowing.ipynb's
own markdown cells; nothing about the reasoning changed in this move, only
where the code lives.

Turns preprocessing.ipynb's / preprocessing.py's normalized, split,
row-level output into the fixed-length (channels, window_length) windows
MA1DCNN.forward() actually requires. Sits between preprocessing (upstream)
and train.py (downstream) in the pipeline:

    loaders -> preprocessing -> **windowing** -> train.py (MA1DCNN)

Input
-----
Expects preprocessing.py's own saved "<file>_train.npz" / "<file>_val.npz"
files, one pair per file in NCMAPSS_FILES, under --preprocessed-dir (or the
EDGE_FAULT_DX_PREPROCESSED_DIR environment variable).

Output
------
Writes "<file>_<split>_windowed.npz" files (arrays X, y, unit_id,
start_cycle) to --windowed-dir (or the EDGE_FAULT_DX_WINDOWED_DIR
environment variable) -- the exact filename convention and env var name
train.py's own _windowed_file_path() / DEFAULT_WINDOWED_DIR already expect,
so no change is needed on that side.

Usage
-----
    python windowing.py                                  # windows both splits (train, val), all 9 files
    python windowing.py --split train                     # windows only the train split
    python windowing.py --preprocessed-dir /path/to/preprocessed --windowed-dir /path/to/windowed
    python windowing.py --window-length 100 --stride 50    # override the default (non-)overlapping window shape
    python windowing.py --self-check                      # run the ported synthetic sanity checks, then exit

The notebook's real runs used a Google Drive path inside Colab; that path is
not meaningful outside Colab, so this module takes both directories as
normal CLI arguments / environment variables instead.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile

import numpy as np

LOGGER = logging.getLogger("edge_fault_dx.data.windowing")


# =============================================================================
# 1. Constants -- every judgment call, named once
# =============================================================================

# Column order fixes which label column a fault flag lands in. Repeated
# verbatim from loaders.py / preprocessing.py / ma1dcnn.py -- do not reorder
# independently of those.
FAULT_COLUMNS = ["fan_fail", "lpc_fail", "hpc_fail", "hpt_fail", "lpt_fail"]

# Same 9 files, same fixed order, repeated from loaders.py / preprocessing.py.
NCMAPSS_FILES = [
    "N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS02-006.h5", "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS04.h5", "N-CMAPSS_DS05.h5", "N-CMAPSS_DS06.h5",
    "N-CMAPSS_DS07.h5", "N-CMAPSS_DS08a-009.h5", "N-CMAPSS_DS08c-008.h5",
]

# preprocessing.py's own output directory is this module's input. Saved
# files are named "<file>_<split>.npz". The notebook hardcoded a Colab Drive
# path here; this module takes it from an env var (with a plain relative
# fallback) and lets --preprocessed-dir override it per run.
DEFAULT_PREPROCESSED_DIR = os.environ.get("EDGE_FAULT_DX_PREPROCESSED_DIR", "data/preprocessed")

# This module's own output directory, and train.py's input. Saved files are
# named "<file>_<split>_windowed.npz", which _windowed_file_path() below
# reproduces exactly -- matching train.py's own DEFAULT_WINDOWED_DIR /
# EDGE_FAULT_DX_WINDOWED_DIR already, so no change is needed on that side.
DEFAULT_WINDOWED_DIR = os.environ.get("EDGE_FAULT_DX_WINDOWED_DIR", "data/windowed")

# Judgment call 1 of 4: how many consecutive cycles make one window, and how
# far apart consecutive windows start. Not one of Akindoju's tuned numbers --
# picked to match the L=50 used throughout ma1dcnn.py's own tests.
# stride == window_length means NON-overlapping windows: simplest to reason
# about and to test, and keeps windows statistically independent of each
# other. A smaller stride (overlapping windows, more training data,
# correlated samples) is a legitimate later tuning choice -- the code below
# supports it, this default just doesn't use it yet.
WINDOW_LENGTH = 50
WINDOW_STRIDE = 50

# Judgment call 2 of 4: how a window's many per-row labels become ONE label.
# "last" = the window's own final row's label (the window is read as "here
# is the history LEADING UP TO this point" -- standard framing for
# RUL-style prognostics). "any" = flagged if ANY row inside the window was a
# failure row, even if the window's last row is healthy. Both are
# implemented; run_self_checks()'s demo shows a worked example where they
# disagree.
LABEL_RULE = "last"

# Judgment call 3 of 4: whether W's 4 flight-condition columns become extra
# input channels. False keeps 28 channels, matching
# DEFAULT_N_INPUT_CHANNELS already baked into ma1dcnn.py's
# BEFORE_/AFTER_TUNING_CONFIG. True gives 32 -- implemented so this question
# can be revisited later without rewriting the windowing logic.
INCLUDE_FLIGHT_CONDITIONS = False

# Which columns of A hold the unit id and the cycle number -- per
# loaders.py's own comment: "A_dev # unit, cycle, Fc, hs". Untouched by
# preprocessing.py (A passes through it unchanged), so these column indices
# are still correct for preprocessing.py's saved output too.
UNIT_COL = 0
CYCLE_COL = 1


# =============================================================================
# 2. Collapse a window's per-row labels into one label
# =============================================================================

def _collapse_window_labels(window_labels, label_rule):
    """Turn one window's (window_length, 5) per-row labels into a single (5,) label.

    Parameters
    ----------
    window_labels : np.ndarray, shape (window_length, 5)
        The per-row label matrix for exactly the rows inside one window.
    label_rule : str
        "last" -- use the window's own final row's label.
        "any"  -- flag a column if ANY row inside the window set it.
    """
    if label_rule == "last":
        return window_labels[-1]  # the row the window "ends on" -- what happens right after this window's history
    elif label_rule == "any":
        # sum > 0 across rows -- catches a fault flagged ANYWHERE inside the window, not just at the end
        return (window_labels.sum(axis=0) > 0).astype(np.int8)
    else:
        # fail loudly on a typo'd rule name rather than silently defaulting to one of the two
        raise ValueError(f"label_rule must be 'last' or 'any', got {label_rule!r}")


# =============================================================================
# 3. Slice one already-ordered unit's rows into fixed-length windows
# =============================================================================

def _windows_for_one_unit(X, labels, window_length=WINDOW_LENGTH, stride=WINDOW_STRIDE, label_rule=LABEL_RULE):
    """Slice one unit's (n_rows, C) sensor matrix into (n_windows, C, window_length) windows.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, C)
        One unit's stacked sensor channels, already time-ordered.
    labels : np.ndarray, shape (n_rows, 5)
        That same unit's per-row labels, row-aligned with X.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, C, window_length)
        Channels-first, matching what MA1DCNN.forward() expects directly.
    window_labels : np.ndarray, shape (n_windows, 5)
    starts : np.ndarray, shape (n_windows,)
        The row index (within this unit) each window started at -- kept for
        traceability, not used in training itself.
    """
    n_rows = X.shape[0]

    if n_rows < window_length:
        # Judgment call 4: a unit shorter than one window produces ZERO
        # windows -- never padded with made-up data. The caller
        # (_build_windows_for_file) is responsible for logging that this
        # happened, out loud.
        return (np.empty((0, X.shape[1], window_length), dtype=X.dtype),
                np.empty((0, labels.shape[1]), dtype=labels.dtype),
                np.empty((0,), dtype=np.int64))

    # Every valid starting row for a full-length window: the last possible
    # start is n_rows - window_length, so a window never reads past this
    # unit's own last row.
    starts = np.arange(0, n_rows - window_length + 1, stride)
    n_windows = len(starts)

    windows = np.empty((n_windows, X.shape[1], window_length), dtype=X.dtype)
    window_labels = np.empty((n_windows, labels.shape[1]), dtype=labels.dtype)

    for i, start in enumerate(starts):
        end = start + window_length
        windows[i] = X[start:end, :].T  # (window_length, C) -> (C, window_length): channels-first for the model
        window_labels[i] = _collapse_window_labels(labels[start:end, :], label_rule)

    return windows, window_labels, starts


# =============================================================================
# 4. Stack the model's input channels
# =============================================================================

def _stack_channels(file_data, include_flight_conditions=INCLUDE_FLIGHT_CONDITIONS):
    """Build the (n_rows, C) channel matrix windowing will slice up.

    C is 28 (X_s + X_v only) by default, or 32 if W is included -- see
    Judgment call 3 above. Column order is fixed: [W if included] then X_s
    then X_v, so channel index is reproducible run to run.
    """
    if include_flight_conditions:
        return np.concatenate([file_data["W"], file_data["X_s"], file_data["X_v"]], axis=1)
    return np.concatenate([file_data["X_s"], file_data["X_v"]], axis=1)


# =============================================================================
# 5. Group one file's rows by unit, order by cycle, window each unit separately
# =============================================================================

def _build_windows_for_file(file_data, window_length=WINDOW_LENGTH, stride=WINDOW_STRIDE,
                             label_rule=LABEL_RULE, include_flight_conditions=INCLUDE_FLIGHT_CONDITIONS):
    """Window one file's worth of rows, unit by unit, never crossing a unit boundary.

    Parameters
    ----------
    file_data : dict
        Exactly what load_processed_file() below returns -- keys W, X_s,
        X_v, A, Y, labels, all row-aligned, matching loaders.py's own saved
        format.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, C, window_length)
    window_labels : np.ndarray, shape (n_windows, 5)
    unit_ids : np.ndarray, shape (n_windows,) -- which unit each window came from, for traceability
    start_cycles : np.ndarray, shape (n_windows,) -- the cycle number each window started at
    """
    X_full = _stack_channels(file_data, include_flight_conditions)
    labels_full = file_data["labels"]
    A = file_data["A"]

    unit_col = A[:, UNIT_COL]
    cycle_col = A[:, CYCLE_COL]

    all_windows, all_labels, all_unit_ids, all_start_cycles = [], [], [], []
    skipped_units = []  # never silent -- collected here, logged once below

    for unit in np.unique(unit_col):
        unit_mask = unit_col == unit                                    # every row belonging to this ONE unit
        order = np.argsort(cycle_col[unit_mask], kind="stable")         # time-order WITHIN this unit only

        X_unit = X_full[unit_mask][order]
        labels_unit = labels_full[unit_mask][order]
        cycles_unit = cycle_col[unit_mask][order]

        windows, window_labels, starts = _windows_for_one_unit(
            X_unit, labels_unit, window_length=window_length, stride=stride, label_rule=label_rule)

        if windows.shape[0] == 0:
            skipped_units.append((unit, X_unit.shape[0]))
            continue  # this unit contributes NOTHING -- no partial or padded window

        all_windows.append(windows)
        all_labels.append(window_labels)
        all_unit_ids.append(np.full(windows.shape[0], unit, dtype=unit_col.dtype))
        all_start_cycles.append(cycles_unit[starts])  # convert row-index starts into actual cycle numbers

    for unit, n in skipped_units:
        LOGGER.info("    [skip] unit %s: only %d rows (< window_length=%d), no windows produced", unit, n, window_length)

    if not all_windows:
        # every unit in this file was too short -- return well-shaped empty arrays, not None
        n_channels = X_full.shape[1]
        return (np.empty((0, n_channels, window_length), dtype=X_full.dtype),
                np.empty((0, labels_full.shape[1]), dtype=labels_full.dtype),
                np.empty((0,), dtype=unit_col.dtype),
                np.empty((0,), dtype=cycle_col.dtype))

    return (np.concatenate(all_windows, axis=0),
            np.concatenate(all_labels, axis=0),
            np.concatenate(all_unit_ids, axis=0),
            np.concatenate(all_start_cycles, axis=0))


# =============================================================================
# 6. Read one already-preprocessed file back
# =============================================================================

def load_processed_file(filename, data_dir=DEFAULT_PREPROCESSED_DIR, split="train"):
    """Load ONE already-preprocessed file back from its saved .npz.

    split is "train" or "val" -- preprocessing.py's own split names; there
    is no single combined "dev" split to read, since preprocessing's own
    train/validation split replaced it.
    """
    out_name = filename.replace(".h5", f"_{split}.npz")
    path = os.path.join(data_dir, out_name)
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


# =============================================================================
# 7. Process and save all files' windows -- ONE FILE AT A TIME
# =============================================================================

def process_and_save_all_windowed_files(data_dir=DEFAULT_PREPROCESSED_DIR, out_dir=DEFAULT_WINDOWED_DIR,
                                         split="train", files=None, window_length=WINDOW_LENGTH,
                                         stride=WINDOW_STRIDE, label_rule=LABEL_RULE,
                                         include_flight_conditions=INCLUDE_FLIGHT_CONDITIONS):
    """Load, window, and save all files to disk ONE AT A TIME, for ONE split.

    Call this once with split="train" and again with split="val" --
    preprocessing.py saves those as two separate files per source file, so
    there is no single combined split to window in one pass.

    Returns
    -------
    list[dict]
        One small summary dict per file -- {"filename", "n_rows",
        "n_windows", "output_path", "output_mb"} -- never the arrays
        themselves, so this function's return value stays small no matter
        how many files or windows are produced.
    """
    files = files if files is not None else NCMAPSS_FILES
    os.makedirs(out_dir, exist_ok=True)

    summaries = []
    running_total_windows = 0

    for i, filename in enumerate(files, start=1):
        LOGGER.info("[%d/%d] Loading %s (%s) ...", i, len(files), filename, split)  # logged BEFORE the slow step, so a hang is visible

        file_data = load_processed_file(filename, data_dir=data_dir, split=split)  # only THIS file in memory now
        n_rows = file_data["W"].shape[0]
        LOGGER.info("    read %s rows", f"{n_rows:,}")

        windows, window_labels, unit_ids, start_cycles = _build_windows_for_file(
            file_data, window_length=window_length, stride=stride,
            label_rule=label_rule, include_flight_conditions=include_flight_conditions)
        n_windows = windows.shape[0]
        LOGGER.info("    built %s windows (window_length=%d, stride=%d)", f"{n_windows:,}", window_length, stride)

        out_name = filename.replace(".h5", f"_{split}_windowed.npz")
        out_path = os.path.join(out_dir, out_name)
        np.savez_compressed(out_path, X=windows, y=window_labels, unit_id=unit_ids, start_cycle=start_cycles)
        output_mb = os.path.getsize(out_path) / (1024 * 1024)
        LOGGER.info("    saved to %s (%.1f MB)", out_path, output_mb)

        summaries.append({
            "filename": filename, "n_rows": n_rows, "n_windows": n_windows,
            "output_path": out_path, "output_mb": output_mb,
        })
        running_total_windows += n_windows

        del file_data, windows, window_labels, unit_ids, start_cycles  # free this file's memory before the next

    LOGGER.info("Done -- %d files processed, %s total windows saved to %s", len(files), f"{running_total_windows:,}", out_dir)
    return summaries


# =============================================================================
# 8. Read windowed files back
# =============================================================================

def load_windowed_file(filename, out_dir=DEFAULT_WINDOWED_DIR, split="train"):
    """Load ONE file's already-saved windows back from its .npz."""
    out_name = filename.replace(".h5", f"_{split}_windowed.npz")
    path = os.path.join(out_dir, out_name)
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def iter_windowed_files(out_dir=DEFAULT_WINDOWED_DIR, split="train", files=None):
    """Yield one file's windows at a time -- the same chunked read pattern loaders.py uses."""
    files = files if files is not None else NCMAPSS_FILES
    for filename in files:
        yield filename, load_windowed_file(filename, out_dir=out_dir, split=split)


# =============================================================================
# 9. Synthetic self-checks -- ported from windowing.ipynb, run via --self-check
# =============================================================================

def _check_windows_for_one_unit_shapes_and_boundaries():
    n_rows, n_channels, L, stride = 7, 2, 3, 2
    X = np.arange(n_rows * n_channels, dtype=np.float64).reshape(n_rows, n_channels)
    labels = np.zeros((n_rows, 5), dtype=np.int8)
    labels[5, 2] = 1  # a fault flag on row index 5 (the 6th row) only

    windows, window_labels, starts = _windows_for_one_unit(X, labels, window_length=L, stride=stride, label_rule="last")

    assert starts.tolist() == [0, 2, 4], starts.tolist()
    assert windows.shape == (3, n_channels, L), windows.shape
    assert window_labels.shape == (3, 5), window_labels.shape

    expected_window1 = X[2:5, :].T
    assert np.array_equal(windows[1], expected_window1)

    # Row index 5 is the only row with a fault flag. Window 0 = rows
    # [0,1,2], window 1 = rows [2,3,4], window 2 = rows [4,5,6]. Only
    # window 2's SPAN includes row 5 -- but row 5 is NOT window 2's last row
    # (row 6 is). This is exactly where "last" and "any" disagree.
    assert window_labels[0].tolist() == [0, 0, 0, 0, 0]
    assert window_labels[1].tolist() == [0, 0, 0, 0, 0]
    assert window_labels[2].tolist() == [0, 0, 0, 0, 0], "label_rule='last' looks at row 6, which has no fault"

    windows_any, window_labels_any, _ = _windows_for_one_unit(X, labels, window_length=L, stride=stride, label_rule="any")
    assert window_labels_any[2].tolist() == [0, 0, 1, 0, 0], "label_rule='any' should catch row 5's flag inside window 2"

    # A unit shorter than window_length must produce ZERO windows, not an error and not a padded fake window.
    windows_short, labels_short, starts_short = _windows_for_one_unit(
        X[:2, :], labels[:2, :], window_length=L, stride=stride, label_rule="last")
    assert windows_short.shape == (0, n_channels, L)
    assert starts_short.shape == (0,)

    LOGGER.info("_windows_for_one_unit: PASSED (window count, shapes, content, both label rules, short-unit skip)")


def _check_build_windows_for_file_respects_unit_boundaries():
    rng = np.random.default_rng(0)

    # Row counts chosen so window_length=3, stride=3 divides each unit's
    # rows EXACTLY (9/3=3 windows, 6/3=2 windows) -- keeps the expected
    # window boundaries simple to reason about by hand.
    unit1_rows, unit2_rows = 9, 6
    n_rows = unit1_rows + unit2_rows

    A = np.zeros((n_rows, 4))
    A[:unit1_rows, 0] = 1
    A[:unit1_rows, 1] = np.arange(1, unit1_rows + 1)
    A[unit1_rows:, 0] = 2
    A[unit1_rows:, 1] = np.arange(1, unit2_rows + 1)

    X_s = rng.random((n_rows, 14))
    X_v = rng.random((n_rows, 14))
    W = rng.random((n_rows, 4))

    # Fault flag on the LAST cycle of each unit -- only the window whose
    # own last row is that final row should ever see it.
    labels = np.zeros((n_rows, 5), dtype=np.int8)
    labels[unit1_rows - 1, 0] = 1
    labels[unit1_rows + unit2_rows - 1, 3] = 1

    file_data = {"W": W, "X_s": X_s, "X_v": X_v, "A": A, "Y": np.zeros((n_rows, 1)), "labels": labels}

    windows, window_labels, unit_ids, start_cycles = _build_windows_for_file(
        file_data, window_length=3, stride=3, label_rule="last")

    # 9 rows / stride 3 = 3 windows for unit 1; 6 rows / stride 3 = 2 windows for unit 2 -- 5 total, in unit order.
    assert windows.shape == (5, 28, 3), windows.shape
    assert unit_ids.tolist() == [1, 1, 1, 2, 2], unit_ids.tolist()
    assert start_cycles.tolist() == [1, 4, 7, 1, 4], start_cycles.tolist()

    # Confirm each window's DATA really is the right slice of X_s -- not just the right shape.
    for i, uid in enumerate(unit_ids):
        offset = 0 if uid == 1 else unit1_rows
        start_row = offset + int(start_cycles[i] - 1)
        expected = X_s[start_row:start_row + 3, :]
        assert np.array_equal(windows[i, :14, :].T, expected), (i, uid)

    assert window_labels[2].tolist() == [1, 0, 0, 0, 0], "unit 1's last window should catch its end-of-life fan_fail"
    assert window_labels[4].tolist() == [0, 0, 0, 1, 0], "unit 2's last window should catch its end-of-life hpt_fail"
    for i in [0, 1, 3]:
        assert window_labels[i].tolist() == [0, 0, 0, 0, 0], i

    # include_flight_conditions=True should give 32 channels, same window count.
    windows32, _, _, _ = _build_windows_for_file(file_data, window_length=3, stride=3,
                                                  label_rule="last", include_flight_conditions=True)
    assert windows32.shape == (5, 32, 3), windows32.shape

    # A unit with fewer rows than window_length must be skipped entirely, never silently.
    file_data_tiny_unit = dict(file_data)
    A_tiny = A.copy()
    A_tiny[unit1_rows:, 0] = 3
    file_data_tiny_unit["A"] = A_tiny
    windows_tiny, _, unit_ids_tiny, _ = _build_windows_for_file(
        file_data_tiny_unit, window_length=7, stride=7, label_rule="last")
    assert set(unit_ids_tiny.tolist()) == {1}, unit_ids_tiny.tolist()

    LOGGER.info("_build_windows_for_file: PASSED (unit grouping, cycle ordering, boundary respect, both channel modes, short-unit skip)")


def _check_process_and_save_all_windowed_files_roundtrip():
    """Builds two tiny synthetic "processed file" .npz files -- matching loaders.py's exact
    saved format, with two units of different lengths inside one of them -- runs them
    through process_and_save_all_windowed_files(), reloads with load_windowed_file(), and
    streams both back with iter_windowed_files()."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        rng = np.random.default_rng(42)
        # DS05 has two units (9 and 6 rows); DS04 has one unit (12 rows).
        # window_length=10 below, so DS05's units (9, 6) are BOTH too short
        # and should be skipped; DS04's 12 rows give exactly 1 window.
        specs = {
            "N-CMAPSS_DS05.h5": {"units": {1: 9, 2: 6}},
            "N-CMAPSS_DS04.h5": {"units": {1: 12}},
        }
        expected_windows_per_file = {}

        for fname, spec in specs.items():
            rows = list(spec["units"].items())
            n_rows = sum(n for _, n in rows)

            A = np.zeros((n_rows, 4))
            labels = np.zeros((n_rows, 5), dtype=np.int8)
            offset, n_expected = 0, 0
            for unit, n in rows:
                A[offset:offset + n, 0] = unit
                A[offset:offset + n, 1] = np.arange(1, n + 1)
                labels[offset + n - 1, 0] = 1  # a fault on the last cycle of every unit
                n_expected += (n // 10) if n >= 10 else 0  # window_length=10, stride=10 used below
                offset += n

            file_data = {
                "W": rng.random((n_rows, 4)), "X_s": rng.random((n_rows, 14)), "X_v": rng.random((n_rows, 14)),
                "A": A, "Y": rng.random((n_rows, 1)), "labels": labels,
            }
            np.savez_compressed(os.path.join(data_dir, fname.replace(".h5", "_train.npz")), **file_data)
            expected_windows_per_file[fname] = n_expected

        test_files = list(specs.keys())
        summaries = process_and_save_all_windowed_files(
            data_dir=data_dir, out_dir=out_dir, split="train", files=test_files,
            window_length=10, stride=10, label_rule="last")

        for s in summaries:
            assert s["n_windows"] == expected_windows_per_file[s["filename"]], (s["filename"], s["n_windows"])
            assert os.path.exists(s["output_path"])

        reloaded = load_windowed_file("N-CMAPSS_DS05.h5", out_dir=out_dir, split="train")
        assert reloaded["X"].shape[1:] == (28, 10)  # both DS05 units were too short -- 0 windows, but shape stays well-formed
        assert set(reloaded["unit_id"].tolist()) <= {1.0, 2.0}

        total = sum(data["X"].shape[0] for _, data in iter_windowed_files(out_dir=out_dir, split="train", files=test_files))
        assert total == sum(expected_windows_per_file.values())

    LOGGER.info("process_and_save_all_windowed_files: PASSED (roundtrip save/load/iterate, short-unit skip preserved end to end)")


def _demo_last_vs_any():
    """Not a test -- a worked example (ported from windowing.ipynb Section 11) showing
    where LABEL_RULE="last" and LABEL_RULE="any" disagree. One fault flag is buried in
    the middle of the second window; logged so the difference can be seen, not just read."""
    n_rows, L, stride = 12, 4, 4
    X = np.zeros((n_rows, 1))  # values don't matter for this demo -- only the labels do
    labels = np.zeros((n_rows, 5), dtype=np.int8)
    labels[5, 2] = 1  # a hpc_fail flag buried in the MIDDLE of what becomes window index 1 (rows 4-7)

    _, labels_last, starts = _windows_for_one_unit(X, labels, window_length=L, stride=stride, label_rule="last")
    _, labels_any, _ = _windows_for_one_unit(X, labels, window_length=L, stride=stride, label_rule="any")

    LOGGER.info("%8s %14s %16s %16s", "window", "rows covered", "last-row label", "any-row label")
    for i, start in enumerate(starts):
        rows_str = f"{start}-{start + L - 1}"
        LOGGER.info("%8d %14s %16s %16s", i, rows_str, str(labels_last[i].tolist()), str(labels_any[i].tolist()))


def run_self_checks():
    """Run every synthetic sanity check ported from windowing.ipynb, in order. Raises on first failure."""
    _check_windows_for_one_unit_shapes_and_boundaries()
    _check_build_windows_for_file_respects_unit_boundaries()
    _check_process_and_save_all_windowed_files_roundtrip()
    _demo_last_vs_any()
    LOGGER.info("all self-checks PASSED")


# =============================================================================
# 10. CLI entry point
# =============================================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build fixed-length windows from preprocessing.py's row-level N-CMAPSS output.")
    parser.add_argument("--preprocessed-dir", default=DEFAULT_PREPROCESSED_DIR,
                         help="directory containing preprocessing.py's '<file>_train.npz' / "
                              "'<file>_val.npz' files (default: %(default)s, or $EDGE_FAULT_DX_PREPROCESSED_DIR)")
    parser.add_argument("--windowed-dir", default=DEFAULT_WINDOWED_DIR,
                         help="directory to write '<file>_<split>_windowed.npz' files to -- "
                              "train.py's own windowed-data input (default: %(default)s, or $EDGE_FAULT_DX_WINDOWED_DIR)")
    parser.add_argument("--split", default="both", choices=["train", "val", "both"],
                         help="which split(s) to window (default: %(default)s)")
    parser.add_argument("--window-length", type=int, default=WINDOW_LENGTH,
                         help="rows per window (default: %(default)s)")
    parser.add_argument("--stride", type=int, default=WINDOW_STRIDE,
                         help="rows between consecutive window starts (default: %(default)s)")
    parser.add_argument("--label-rule", default=LABEL_RULE, choices=["last", "any"],
                         help="how a window's per-row labels collapse to one label (default: %(default)s)")
    parser.add_argument("--include-flight-conditions", action="store_true", default=INCLUDE_FLIGHT_CONDITIONS,
                         help="also stack W's 4 flight-condition columns as input channels (28 -> 32 channels)")
    parser.add_argument("--self-check", action="store_true",
                         help="run the ported synthetic sanity checks against temp data, then exit")
    parser.add_argument("--quiet", action="store_true", help="suppress per-file logging")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.self_check:
        run_self_checks()
        return

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        LOGGER.info("=== windowing split=%s ===", split)
        process_and_save_all_windowed_files(
            data_dir=args.preprocessed_dir, out_dir=args.windowed_dir, split=split,
            window_length=args.window_length, stride=args.stride, label_rule=args.label_rule,
            include_flight_conditions=args.include_flight_conditions,
        )


if __name__ == "__main__":
    main()
