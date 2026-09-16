"""Unit tests for edge_fault_dx.data.loaders and edge_fault_dx.data.preprocessing.

Every test here runs against small, synthetic, in-memory or temp-directory
data -- none of them need the real N-CMAPSS dataset or Google Drive, so
they run in CI (see .github/workflows/ci.yml) on every push.

Run locally with:
    pytest tests/test_data.py -v
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np

from edge_fault_dx.data import loaders
from edge_fault_dx.data import preprocessing


# ---------------------------------------------------------------------------
# loaders.py
# ---------------------------------------------------------------------------

def test_build_labels_healthy_row_is_all_zeros():
    # RUL=145 is well above the 30-cycle failure threshold, so every
    # column should stay 0 regardless of which component this file maps to.
    labels = loaders._build_labels(np.array([145]), loaders.FAULT_COMPONENT_MAP["N-CMAPSS_DS05.h5"])
    assert labels.tolist() == [[0, 0, 0, 0, 0]]


def test_build_labels_failure_row_flags_only_mapped_components():
    # DS03 maps to ["hpt", "lpt"] -- a failure row (RUL=12) must flag
    # exactly those two columns and no others.
    labels = loaders._build_labels(np.array([12]), loaders.FAULT_COMPONENT_MAP["N-CMAPSS_DS03-012.h5"])
    assert labels.tolist() == [[0, 0, 0, 1, 1]]


def test_build_labels_threshold_boundary():
    # RUL=30 is exactly at the failure threshold (<=30 counts as failing);
    # RUL=31 is just above it and must stay healthy.
    labels = loaders._build_labels(np.array([30, 31]), loaders.FAULT_COMPONENT_MAP["N-CMAPSS_DS04.h5"])
    assert labels.tolist() == [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]


def test_process_and_save_all_files_end_to_end():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        specs = {
            "N-CMAPSS_DS05.h5": np.array([[145], [20]]),        # DS05 -> hpc; 1 healthy, 1 hpc-fail
            "N-CMAPSS_DS04.h5": np.array([[5], [100], [30]]),   # DS04 -> fan; 2 fail, 1 healthy
        }
        for fname, y in specs.items():
            n = y.shape[0]
            with h5py.File(os.path.join(data_dir, fname), "w") as f:
                f.create_dataset("W_dev", data=np.random.rand(n, 4))
                f.create_dataset("X_s_dev", data=np.random.rand(n, 14))
                f.create_dataset("X_v_dev", data=np.random.rand(n, 14))
                f.create_dataset("A_dev", data=np.random.rand(n, 4))
                f.create_dataset("Y_dev", data=y)

        test_files = list(specs.keys())
        summaries = loaders.process_and_save_all_files(
            data_dir=data_dir, out_dir=out_dir, split="dev", files=test_files
        )

        # Every file's row count is reported correctly and its output
        # file actually exists on disk.
        assert summaries[0]["n_rows"] == 2 and summaries[1]["n_rows"] == 3
        assert all(os.path.exists(s["output_path"]) for s in summaries)

        # Reloading a saved file gives back the exact labels Rule 1 + 2
        # should have produced for it.
        reloaded = loaders.load_processed_file("N-CMAPSS_DS05.h5", out_dir=out_dir, split="dev")
        assert reloaded["labels"].tolist() == [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]

        # iter_processed_files() streams every file without dropping rows.
        total = sum(
            data["W"].shape[0]
            for _, data in loaders.iter_processed_files(out_dir=out_dir, split="dev", files=test_files)
        )
        assert total == 5


def test_load_file_rejects_unmapped_filename():
    with tempfile.TemporaryDirectory() as data_dir:
        try:
            loaders.load_file("NOT_A_REAL_FILE.h5", data_dir=data_dir)
        except ValueError:
            pass
        else:
            raise AssertionError("expected a ValueError for an unmapped filename")


# ---------------------------------------------------------------------------
# preprocessing.py
# ---------------------------------------------------------------------------

def test_load_processed_file_round_trips():
    with tempfile.TemporaryDirectory() as out_dir:
        fake = {"W": np.array([[1.0, 2.0]]), "labels": np.array([[0, 0, 0, 0, 0]], dtype=np.int8)}
        # load_processed_file expects a "<name>_{split}.npz" file -- save
        # directly under that convention rather than the plain .h5 name.
        np.savez_compressed(os.path.join(out_dir, "FAKE_dev.npz"), **fake)

        reloaded = preprocessing.load_processed_file("FAKE.h5", out_dir=out_dir, split="dev")
        assert np.array_equal(reloaded["W"], fake["W"])
        assert np.array_equal(reloaded["labels"], fake["labels"])


def test_split_units_train_val_holds_out_last_unit():
    # 5 units, 10 rows each, in order -- a stand-in for one file's A[:, 0].
    unit_ids = np.repeat([1, 2, 3, 4, 5], 10)

    train_mask, val_mask, train_units, val_units = preprocessing.split_units_train_val(
        unit_ids, val_fraction=0.2
    )

    # 20% of 5 units = 1 unit held out -- must be unit 5, the last one.
    assert val_units.tolist() == [5]
    assert train_units.tolist() == [1, 2, 3, 4]

    # Every row is in exactly one split -- never both, never neither.
    assert (train_mask | val_mask).all()
    assert not (train_mask & val_mask).any()
    assert train_mask.sum() == 40
    assert val_mask.sum() == 10


def test_split_units_train_val_single_unit_edge_case():
    # A single-unit file should not crash, and should keep everything in
    # train rather than trying to hold out its only unit.
    single_unit_ids = np.repeat([7], 15)
    train_mask, val_mask, train_units, val_units = preprocessing.split_units_train_val(
        single_unit_ids, val_fraction=0.2
    )
    assert train_mask.all() and not val_mask.any()
    assert len(val_units) == 0


def test_compute_channel_stats_excludes_validation_rows():
    with tempfile.TemporaryDirectory() as out_dir:
        rng = np.random.default_rng(7)
        # 2 units, 500 rows each. val_fraction=0.5 holds out unit 2
        # entirely, so only unit 1's rows should count toward stats.
        n_per_unit = 500
        unit_col = np.repeat([1, 2], n_per_unit)
        n = len(unit_col)

        # Known distribution for unit 1's W column: mean=100, std=5.
        # Unit 2 is given a wildly different distribution specifically so
        # a leakage bug would be impossible to miss.
        w = np.zeros((n, 2))
        w[unit_col == 1] = rng.normal(loc=100.0, scale=5.0, size=(n_per_unit, 2))
        w[unit_col == 2] = rng.normal(loc=-999.0, scale=1.0, size=(n_per_unit, 2))

        data = {
            "W": w,
            "X_s": rng.random((n, 3)),
            "X_v": rng.random((n, 3)),
            "A": np.stack([unit_col, np.zeros(n), np.zeros(n), np.ones(n)], axis=1),
            "Y": rng.integers(0, 300, size=(n, 1)),
            "labels": np.zeros((n, 5), dtype=np.int8),
        }
        np.savez_compressed(os.path.join(out_dir, "FAKE_dev.npz"), **data)

        stats = preprocessing.compute_channel_stats(
            out_dir=out_dir, files=["FAKE.h5"], groups=["W"], val_fraction=0.5
        )

        # If unit 2's rows leaked into the stats, this mean would be
        # nowhere near 100 -- this IS the leakage check, not just an
        # arithmetic check.
        assert np.allclose(stats["W"]["mean"], 100.0, atol=1.0)
        assert np.allclose(stats["W"]["std"], 5.0, atol=1.0)


def test_apply_normalization():
    rng = np.random.default_rng(0)
    w = rng.normal(loc=50.0, scale=10.0, size=(1000, 4))  # known mean=50, std=10
    x_s = rng.random((1000, 14))
    data = {
        "W": w, "X_s": x_s, "X_v": rng.random((1000, 14)),
        "A": rng.integers(0, 5, size=(1000, 4)),
        "Y": rng.integers(0, 300, size=(1000, 1)),
        "labels": np.zeros((1000, 5), dtype=np.int8),
    }
    stats = {
        "W": {"mean": w.mean(axis=0), "std": w.std(axis=0)},
        "X_s": {"mean": x_s.mean(axis=0), "std": x_s.std(axis=0)},
        "X_v": {"mean": data["X_v"].mean(axis=0), "std": data["X_v"].std(axis=0)},
    }

    normalized = preprocessing.apply_normalization(data, stats)

    # Normalized channels should now have ~0 mean, ~1 std.
    assert np.allclose(normalized["W"].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(normalized["W"].std(axis=0), 1, atol=1e-6)

    # Untouched groups must be byte-for-byte identical to the input.
    assert np.array_equal(normalized["A"], data["A"])
    assert np.array_equal(normalized["Y"], data["Y"])
    assert np.array_equal(normalized["labels"], data["labels"])


def test_add_gaussian_noise_hits_target_snr():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0, scale=1.0, size=(200_000, 3))  # large N so measured SNR is stable

    target_snr_db = 10.0
    noisy = preprocessing.add_gaussian_noise(x, snr_db=target_snr_db, rng=rng)
    actual_noise = noisy - x
    measured_snr_db = 10 * np.log10(np.mean(x ** 2, axis=0) / np.mean(actual_noise ** 2, axis=0))
    # Random sampling means this will not be exact -- within 0.5dB is a
    # tight, reliable tolerance at 200,000 samples.
    assert np.allclose(measured_snr_db, target_snr_db, atol=0.5)


def test_add_impulse_noise_hits_target_rate_and_leaves_rest_untouched():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100_000, 2))
    noisy, mask = preprocessing.add_impulse_noise(x, fraction=0.05, rng=rng)

    measured_fraction = mask.mean()
    assert abs(measured_fraction - 0.05) < 0.01
    assert np.all(noisy[mask] != x[mask])          # every masked position must actually differ
    assert np.array_equal(noisy[~mask], x[~mask])  # every unmasked position must be untouched


def test_process_and_save_preprocessed_end_to_end():
    with tempfile.TemporaryDirectory() as processed_dir, tempfile.TemporaryDirectory() as out_dir:
        rng = np.random.default_rng(42)

        # Build 2 synthetic "loaders.py output" files -- 3 units each, 20
        # rows/unit, so a 20% validation split holds out exactly 1 unit
        # (20 rows) per file, leaving 2 units (40 rows) for training.
        specs = {"FAKE_FILE_A.h5": 3, "FAKE_FILE_B.h5": 3}
        for fname, n_units in specs.items():
            rows_per_unit = 20
            n = n_units * rows_per_unit
            unit_col = np.repeat(np.arange(1, n_units + 1), rows_per_unit)
            a = np.stack(
                [unit_col, np.tile(np.arange(rows_per_unit), n_units), np.zeros(n), np.ones(n)],
                axis=1,
            )
            data = {
                "W": rng.normal(loc=50.0, scale=10.0, size=(n, 4)),
                "X_s": rng.normal(loc=0.0, scale=5.0, size=(n, 14)),
                "X_v": rng.normal(loc=0.0, scale=5.0, size=(n, 14)),
                "A": a,
                "Y": rng.integers(0, 300, size=(n, 1)),
                "labels": np.zeros((n, 5), dtype=np.int8),
            }
            np.savez_compressed(os.path.join(processed_dir, fname.replace(".h5", "_dev.npz")), **data)

        test_files = list(specs.keys())
        summaries, stats = preprocessing.process_and_save_preprocessed(
            processed_dir=processed_dir, out_dir=out_dir, files=test_files, val_fraction=0.2, augment=True
        )

        # Every file produced a train and a val output on disk.
        assert all(os.path.exists(s["train_path"]) and os.path.exists(s["val_path"]) for s in summaries)

        # No test-split input existed, so no test output should exist either.
        assert all("test_path" not in s for s in summaries)

        # Row accounting: 1 held-out unit x 20 rows = 20 val rows/file;
        # the other 2 units x 20 rows = 40 train rows/file.
        for s in summaries:
            assert s["n_val_rows"] == 20
            assert s["n_train_rows"] == 40

        # Reload one train file: shapes are right, and W's mean is still
        # near 0 even after augmentation noise was layered on top of
        # normalization (a loose bound -- noise is expected to shift it
        # a little, just not by a lot).
        with np.load(summaries[0]["train_path"]) as npz:
            reloaded_train = {k: npz[k] for k in npz.files}
        assert abs(reloaded_train["W"].mean()) < 1.0
        assert reloaded_train["labels"].shape == (40, 5)

        # Reload one val file: right shape, and -- unlike train -- never augmented.
        with np.load(summaries[0]["val_path"]) as npz:
            reloaded_val = {k: npz[k] for k in npz.files}
        assert reloaded_val["W"].shape == (20, 4)
