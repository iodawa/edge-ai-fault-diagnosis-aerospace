"""CLI entry point: train the baseline MA1DCNN.

Usage:
    python scripts/run_baseline.py --config configs/baseline.yaml

TODO: wire this to src/edge_fault_dx/training/train.py.
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Wire this up to training/train.py using {args.config}")


if __name__ == "__main__":
    main()
