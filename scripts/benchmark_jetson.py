"""CLI entry point: benchmark the deployed model on Jetson Nano.

Usage:
    python scripts/benchmark_jetson.py --config configs/jetson_deploy.yaml

TODO: wire this to src/edge_fault_dx/deployment/*.py and
deployment/jetson/tegrastats_logger.py.
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Wire this up to deployment/*.py using {args.config}")


if __name__ == "__main__":
    main()
