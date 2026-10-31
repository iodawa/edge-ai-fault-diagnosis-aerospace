"""Sweep the classification threshold for a trained model and report precision/
recall/F1/hamming at each one -- edge-ai-fault-diagnosis-aerospace.

This does NOT retrain anything. It loads the already-trained checkpoint (e.g.
ma1dcnn_trial13_prod.pt), runs ONE forward pass over the validation set to get
every window's raw logits, then re-scores those same cached logits at many
threshold values. It reuses train.py's own build_ma1dcnn, build_dataloaders,
and compute_metrics directly, so every number here is computed exactly the
way train.py itself would compute it at that threshold -- not a re-derivation,
not an approximation.

Usage (run from the same directory as train.py, e.g. /content in Colab)
-------------------------------------------------------------------------
    python threshold_sweep.py \
        --config trial13 \
        --checkpoint /content/drive/MyDrive/edge-ai-fault-diagnosis-aerospace/data/ma1dcnn_trial13_prod.pt \
        --windowed-dir /content/drive/MyDrive/edge-ai-fault-diagnosis-aerospace/data/windowed \
        --out /content/drive/MyDrive/edge-ai-fault-diagnosis-aerospace/data/threshold_sweep_trial13.csv

Prints a table to the console AND writes the same data as a CSV (threshold,
precision_macro, recall_macro, f1_macro, hamming_loss) -- paste either one
back to continue the analysis.
"""
from __future__ import annotations

import argparse
import csv
import sys

import torch

sys.path.insert(0, ".")  # so `import train` finds train.py in the current directory
from train import CONFIGS, DEFAULT_WINDOWED_DIR, build_dataloaders, build_ma1dcnn, compute_metrics  # noqa: E402

# trial13's shipped default -- the one this whole analysis is trying to confirm or replace.
TRIAL13_DEFAULT_THRESHOLD = 0.4214610774407138


def sweep(model, val_loader, device, thresholds):
    """Run ONE pass over val_loader, cache logits, then re-score at every threshold."""
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            logits = model(X)
            all_logits.append(logits.cpu())
            all_y.append(y)
    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)

    rows = []
    for t in thresholds:
        m = compute_metrics(all_y, all_logits, threshold=t)
        rows.append((t, m["precision_macro"], m["recall_macro"], m["f1_macro"], m["hamming_loss"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="trial13", choices=sorted(CONFIGS),
                         help="which named config's model architecture to build (default: %(default)s)")
    parser.add_argument("--checkpoint", required=True, help="path to the saved .pt state_dict")
    parser.add_argument("--windowed-dir", default=DEFAULT_WINDOWED_DIR,
                         help="directory with the *_train_windowed.npz / *_val_windowed.npz files")
    parser.add_argument("--device", default=None, help="override auto-detected device, e.g. 'cpu' or 'cuda:0'")
    parser.add_argument("--out", default="threshold_sweep.csv", help="where to write the CSV (default: %(default)s)")
    parser.add_argument("--step", type=float, default=0.025, help="threshold step size (default: %(default)s)")
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CONFIGS[args.config]

    model = build_ma1dcnn(cfg["model_config"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    _, val_loader = build_dataloaders(windowed_dir=args.windowed_dir, batch_size=256)

    n_steps = int(round((0.95 - 0.05) / args.step)) + 1
    thresholds = [round(0.05 + args.step * i, 4) for i in range(n_steps)]

    rows = sweep(model, val_loader, device, thresholds)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "precision_macro", "recall_macro", "f1_macro", "hamming_loss"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}\n")

    print(f"{'threshold':>9} {'precision':>10} {'recall':>8} {'f1':>8} {'hamming':>8}")
    for t, p, r, f1, h in rows:
        marker = "  <-- trial13 shipped default (0.4215)" if abs(t - TRIAL13_DEFAULT_THRESHOLD) < (args.step / 2) else ""
        print(f"{t:9.3f} {p:10.4f} {r:8.4f} {f1:8.4f} {h:8.4f}{marker}")


if __name__ == "__main__":
    main()
