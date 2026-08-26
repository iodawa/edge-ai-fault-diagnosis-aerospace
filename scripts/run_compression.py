"""CLI entry point: prune + quantize the trained baseline model.

Usage:
    python scripts/run_compression.py --config configs/compression.yaml

TODO: wire this to src/edge_fault_dx/compression/{pruning,quantization}.py.
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Wire this up to compression/*.py using {args.config}")


if __name__ == "__main__":
    main()
