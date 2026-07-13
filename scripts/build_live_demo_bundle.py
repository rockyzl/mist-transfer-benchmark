#!/usr/bin/env python3
"""Build the ignored, serving-only QM9 live-demo model bundle."""

from __future__ import annotations

import argparse

from mist_transfer_benchmark.live_demo import build_live_demo_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_demo_v1.toml")
    parser.add_argument("--bundle-dir", default="data/private/qm9/live-demo-v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_live_demo_bundle(
        config_path=args.config,
        bundle_dir=args.bundle_dir,
        overwrite=args.overwrite,
    )
    print(f"built {args.bundle_dir}; seconds={result['build_seconds']:.1f}")


if __name__ == "__main__":
    main()
