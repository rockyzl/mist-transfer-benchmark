#!/usr/bin/env python3
"""Run the versioned paper evaluation; default smoke mode is intentionally cheap."""

from __future__ import annotations

import argparse
import hashlib
import tomllib
from pathlib import Path

import numpy as np

from mist_transfer_benchmark.qm9.paper_evaluation import ArrayTargetLoader, run_protocol


def _smoke_data(seed: int = 17) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    smiles = [
        "CC", "CCC", "CCCC", "CCO", "CCCO", "CCN", "CCCN", "COC", "COCC", "CNC",
        "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1", "c1ccncc1", "C1CCCCC1",
        "CC1CCCCC1", "C1CCNCC1", "C1CCOCC1", "CC(=O)O", "CCC(=O)O", "CC(=O)N",
        "CCC(=O)N", "CC#N", "CCC#N", "CCF", "CCCl", "CCBr", "CO", "CN",
    ] * 3
    x = rng.normal(size=(len(smiles), 16))
    weights = rng.normal(size=(16, 12))
    y = x @ weights + rng.normal(scale=0.1, size=(len(smiles), 12))
    return x, y, smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qm9_paper_evaluation_smoke.toml"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/qm9-paper-evaluation-smoke"))
    parser.add_argument("--data", type=Path, help="NPZ containing x, y, and smiles arrays")
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        protocol = tomllib.load(handle)
    scaffold_ids = None
    if args.data is None:
        x, y, smiles = _smoke_data()
        target_identity = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
    else:
        payload = np.load(args.data, allow_pickle=False)
        x = payload["x"]
        y = payload["y"]
        smiles = payload["smiles"].astype(str).tolist()
        target_identity = hashlib.sha256(args.data.read_bytes()).hexdigest()
        scaffold_ids = (
            payload["scaffold_group_ids"] if "scaffold_group_ids" in payload else None
        )
    manifest = run_protocol(
        protocol,
        x,
        ArrayTargetLoader(
            y,
            provenance={"source": str(args.data or "synthetic-smoke-v1")},
            full_target_identity=target_identity,
        ),
        smiles,
        args.output,
        feature_schema={
            "representation": "prepared-engineered-features",
            "columns": int(x.shape[1]),
            "dtype": str(x.dtype),
        },
        precomputed_scaffold_group_ids=scaffold_ids,
    )
    print(f"completed {len(manifest['completed_cells'])} cells; output={args.output}")


if __name__ == "__main__":
    main()
