"""Load N-CMAPSS HDF5 data and construct multi-label fault-diagnosis
labels (Phase 1).

This module implements the label-construction logic already agreed on
in deliverables/relabel_decision_akindoju_alignment.docx and
deliverables/rul_and_fault_classification_approach.docx:

    Rule 1 ("when"):  a row is a failure row if RUL <= 30 cycles.
    Rule 2 ("what"):  for failure rows, turn on the fault flag(s) for
                       whichever component(s) that row's source file
                       is mapped to (Akindoju, 2025, Table 4.1).

Revision history
-----------------
The first version of this module concatenated all 9 files into one
in-memory dataset (a single load_all_files() call). That crashed a
standard Colab runtime: 69,900,301 rows x 41 numeric columns works out
to roughly 20-25GB held in RAM at once, well above the ~12-13GB
free-tier ceiling -- and the crash was SILENT (Colab kills and
restarts the runtime with no visible error, which just looks like a
cell that "finished" but printed nothing). See
deliverables/pipeline_run_record.docx for the full story.

This version fixes both problems at once:
  1. Files are processed and saved to disk ONE AT A TIME
     (process_and_save_all_files()) -- memory footprint is capped at
     roughly one file's worth of data (a few GB at most), not all 9.
  2. Every step prints as it happens -- no cell should ever look like
     it "did nothing" while it is actually working or has crashed.

Design notes still worth keeping in mind while reading this file:
  * The T group (engine health parameters) is NEVER loaded here. It is
    simulation-only ground truth that would not exist on a real
    aircraft, so it must stay out of the model's input features (see
    docs/data-dictionary.md, Sections 9-10). Y (RUL) IS loaded, but
    only to build the label below -- it is not saved as a model input.
"""

from __future__ import annotations

import os

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# 1. Fixed constants -- these ARE the label-construction logic. Nothing
#    below this block should need to change to load a new N-CMAPSS file
#    that is already covered by Table 4.1.
# ---------------------------------------------------------------------------

# Rule 2 ("what"): file -> which component(s) that file was built to
# fail, straight from Akindoju (2025), Table 4.1. This dict is checked
# against the same 9-file, 60-unit total documented in
# docs/data-dictionary.md, so if that table ever changes, this must too.
FAULT_COMPONENT_MAP: dict[str, list[str]] = {
    "N-CMAPSS_DS01-005.h5": ["hpt"],
    "N-CMAPSS_DS02-006.h5": ["hpt"],
    "N-CMAPSS_DS03-012.h5": ["hpt", "lpt"],
    "N-CMAPSS_DS04.h5": ["fan"],
    "N-CMAPSS_DS05.h5": ["hpc"],
    "N-CMAPSS_DS06.h5": ["hpc", "lpc"],
    "N-CMAPSS_DS07.h5": ["lpt"],
    "N-CMAPSS_DS08a-009.h5": ["fan", "lpc", "hpc", "hpt", "lpt"],
    "N-CMAPSS_DS08c-008.h5": ["fan", "lpc", "hpc", "hpt", "lpt"],
}

# The 9 usable files, in one fixed order -- every summary this module
# produces lists files in this same order.
NCMAPSS_FILES: list[str] = list(FAULT_COMPONENT_MAP.keys())

# Fixed output-column order for the label matrix. Downstream code
# (the model's output layer, evaluation/metrics.py) can rely on column
# 0 always being fan_fail, column 1 always lpc_fail, and so on.
FAULT_COLUMNS: list[str] = ["fan_fail", "lpc_fail", "hpc_fail", "hpt_fail", "lpt_fail"]

# Rule 1 ("when"): the RUL cutoff, in cycles, below which a row counts
# as "failing." This number is a judgment call carried over from
# Akindoju (2025) / the original N-CMAPSS scoring paper, not a
# physical constant -- see rul_and_fault_classification_approach.docx.
RUL_FAILURE_THRESHOLD = 30

# Default location of the raw .h5 files once Google Drive is mounted
# in Colab, matching the path already used in notebooks/01_eda.ipynb.
DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/edge-ai-fault-diagnosis-aerospace/data/raw"

# Where processed (per-file, labeled) output gets saved -- matches the
# repo's existing data/processed/ folder convention.
DEFAULT_PROCESSED_DIR = "/content/drive/MyDrive/edge-ai-fault-diagnosis-aerospace/data/processed"


def mount_google_drive() -> None:
    """Mount Google Drive inside a Colab runtime.

    Call this once at the top of a Colab notebook, before calling any
    function below that reads from or writes to DEFAULT_DRIVE_DIR /
    DEFAULT_PROCESSED_DIR.
    """
    try:
        # This import only succeeds inside a real Colab runtime, so it
        # doubles as the "are we in Colab?" check.
        from google.colab import drive
    except ImportError:
        # Not in Colab (e.g. a local unit test run) -- there is no
        # Drive to mount, so do nothing rather than raise an error.
        print("Not running in Colab -- skipping Drive mount.")
        return

    print("Mounting Google Drive ...")
    drive.mount("/content/drive")  # triggers Colab's one-time browser auth prompt
    print("Drive mounted.")


def _build_labels(rul: np.ndarray, fault_components: list[str]) -> np.ndarray:
    """Construct the 5-column binary fault-flag matrix for one file.

    Parameters
    ----------
    rul : np.ndarray, shape (n_rows,)
        The RUL value for every row in one file (from that file's Y
        group), used only here to decide failure vs. healthy.
    fault_components : list[str]
        This file's mapped component(s), e.g. ["hpt", "lpt"] for DS03
        -- looked up from FAULT_COMPONENT_MAP by the caller.

    Returns
    -------
    np.ndarray, shape (n_rows, 5), dtype int8
        1 = that component is flagged failing on that row, 0 = not.
        Column order matches FAULT_COLUMNS.
    """
    n_rows = rul.shape[0]

    # Start every row as all-zero (healthy on all 5 flags) -- Rule 2
    # only ever turns flags ON below, it never needs to turn any off.
    labels = np.zeros((n_rows, len(FAULT_COLUMNS)), dtype=np.int8)

    # Rule 1 ("when"): True for every row within RUL_FAILURE_THRESHOLD
    # cycles of end-of-life; False (healthy) for everything else.
    is_failure_row = rul <= RUL_FAILURE_THRESHOLD

    # Rule 2 ("what"): for THIS file's mapped component(s) only, set
    # that column to 1 on every failure row. A file mapped to more than
    # one component (e.g. DS03 -> hpt + lpt) sets more than one column,
    # which is exactly how one row ends up with two flags on at once.
    for component in fault_components:
        col_index = FAULT_COLUMNS.index(f"{component}_fail")  # e.g. "hpt" -> "hpt_fail" -> column 3
        labels[is_failure_row, col_index] = 1  # only failure rows get touched; healthy rows stay 0

    return labels


def load_file(
    filename: str,
    data_dir: str = DEFAULT_DRIVE_DIR,
    split: str = "dev",
) -> dict[str, np.ndarray]:
    """Load ONE N-CMAPSS file's model inputs and construct its labels.

    This is the only function that reads raw HDF5. It handles exactly
    one file at a time by design -- see process_and_save_all_files()
    below for how the 9 files are processed together without holding
    all of them in memory at once.

    Parameters
    ----------
    filename : str
        Must be one of the 9 keys in FAULT_COMPONENT_MAP.
    data_dir : str
        Directory the raw .h5 files live in. Defaults to the Google
        Drive path; pass a local folder instead for testing.
    split : str
        "dev" or "test" -- N-CMAPSS stores each group twice, suffixed
        _dev / _test (e.g. "W_dev", "W_test").

    Returns
    -------
    dict with keys "W", "X_s", "X_v", "A", "Y", "labels" -- numpy
    arrays, all with the same row count, row-aligned to each other.
    """
    if filename not in FAULT_COMPONENT_MAP:
        # Fail loudly instead of silently loading a file with no known
        # fault-component mapping -- there would be no valid label for it.
        raise ValueError(f"{filename!r} is not one of the 9 mapped files in Table 4.1.")

    path = f"{data_dir}/{filename}"  # build the full path once, reused for every group below

    with h5py.File(path, "r") as f:
        # [:] copies each group out of the HDF5 file into a normal numpy
        # array -- without it, `f[...]` stays a lazy on-disk reference.
        w = f[f"W_{split}"][:]      # flight-condition descriptors (4 cols) -- model input
        x_s = f[f"X_s_{split}"][:]  # physical sensor measurements (14 cols) -- model input
        x_v = f[f"X_v_{split}"][:]  # virtual/derived sensors (14 cols) -- model input (practical compromise)
        a = f[f"A_{split}"][:]      # unit, cycle, Fc, hs -- identifying columns, not a model input
        y = f[f"Y_{split}"][:]      # RUL -- used ONLY below to build the label, never a model input

    rul = y.reshape(-1)  # Y comes back as an (n_rows, 1) column; flatten to 1-D for the threshold check
    labels = _build_labels(rul, FAULT_COMPONENT_MAP[filename])  # apply Rules 1 + 2 for this file

    return {"W": w, "X_s": x_s, "X_v": x_v, "A": a, "Y": y, "labels": labels}


def process_and_save_all_files(
    data_dir: str = DEFAULT_DRIVE_DIR,
    out_dir: str = DEFAULT_PROCESSED_DIR,
    split: str = "dev",
    files: list[str] | None = None,
) -> list[dict]:
    """Load, label, and save all 9 files to disk ONE AT A TIME.

    This replaces the earlier load_all_files(), which held all 9
    files in memory simultaneously (~20-25GB) and crashed a standard
    Colab runtime with no visible error. This version never holds
    more than one file's data in memory at once, and prints progress
    at every step so a hang or crash is visible immediately rather
    than looking identical to a working cell.

    Parameters
    ----------
    data_dir : str
        Directory the raw .h5 files live in.
    out_dir : str
        Directory to save the processed, labeled .npz files into.
        Created automatically if it doesn't already exist.
    split : str
        "dev" or "test".
    files : list[str] or None
        Which files to process, in order. Defaults to all 9
        (NCMAPSS_FILES). Overridable for a quick test on 1-2 files.

    Returns
    -------
    list[dict]
        One small summary dict per file -- {"filename", "n_rows",
        "output_path", "output_mb"} -- NOT the actual data arrays, so
        this function's own return value never grows large no matter
        how many files are processed.
    """
    files = files if files is not None else NCMAPSS_FILES  # default to the full documented 9-file scope

    os.makedirs(out_dir, exist_ok=True)  # create data/processed/ (or the given out_dir) if it isn't there yet

    summaries = []          # one small dict per file -- never holds the actual arrays
    running_total_rows = 0  # accumulated as a plain int, not by keeping every file's arrays around

    for i, filename in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Loading {filename} ...")  # printed BEFORE the slow step, so a hang is visible here

        loaded = load_file(filename, data_dir=data_dir, split=split)  # only THIS file's arrays are in memory now
        n_rows = loaded["W"].shape[0]
        print(f"    read {n_rows:,} rows")

        out_name = filename.replace(".h5", f"_{split}.npz")  # e.g. "N-CMAPSS_DS05.h5" -> "N-CMAPSS_DS05_dev.npz"
        out_path = os.path.join(out_dir, out_name)

        # savez_compressed writes all 6 arrays into one .npz file on disk.
        # This is the step that lets us free loaded's memory right after --
        # nothing later needs to keep this file's arrays in RAM.
        np.savez_compressed(out_path, **loaded)
        output_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"    saved to {out_path} ({output_mb:.1f} MB)")

        summaries.append({
            "filename": filename,
            "n_rows": n_rows,
            "output_path": out_path,
            "output_mb": output_mb,
        })
        running_total_rows += n_rows

        del loaded  # explicitly drop the reference so this file's ~1-4GB is freed before the next loop iteration

    print(f"Done -- {len(files)} files processed, {running_total_rows:,} total rows saved to {out_dir}")
    return summaries


def load_processed_file(filename: str, out_dir: str = DEFAULT_PROCESSED_DIR, split: str = "dev") -> dict[str, np.ndarray]:
    """Load ONE already-processed file back from its saved .npz.

    Use this to inspect or spot-check a single file's labeled data
    without re-reading the original HDF5.
    """
    out_name = filename.replace(".h5", f"_{split}.npz")
    out_path = os.path.join(out_dir, out_name)
    with np.load(out_path) as npz:
        # dict(npz) copies each array out of the open .npz handle --
        # after this, the file itself can be closed (the `with` block
        # above does that automatically) without losing the data.
        return {key: npz[key] for key in npz.files}


def iter_processed_files(out_dir: str = DEFAULT_PROCESSED_DIR, split: str = "dev", files: list[str] | None = None):
    """Yield one processed file's data at a time, instead of loading all 9 into memory.

    This is the "chunked" access pattern for anything downstream (e.g.
    preprocessing.py, or a training loop) that needs to see every row
    across all 9 files but should never hold more than one file's
    worth of data in memory at once -- the same principle that fixed
    process_and_save_all_files() above, applied to reading it back.

    Example
    -------
    for filename, file_data in iter_processed_files():
        ...  # do something with one file's arrays, then move on --
             # file_data goes out of scope and is freed before the
             # next iteration's data is loaded.
    """
    files = files if files is not None else NCMAPSS_FILES
    for filename in files:
        yield filename, load_processed_file(filename, out_dir=out_dir, split=split)


if __name__ == "__main__":
    # Quick manual run when executing this file directly in Colab:
    # mounts Drive, then processes and saves all 9 files one at a
    # time, printing progress throughout so nothing looks silent.
    mount_google_drive()
    process_and_save_all_files()
