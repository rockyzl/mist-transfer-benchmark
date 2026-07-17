#!/usr/bin/env python3
"""Run the versioned paper evaluation; default smoke mode is intentionally cheap."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
from scipy import sparse

from mist_transfer_benchmark.qm9.data import load_qm9_identities
from mist_transfer_benchmark.qm9.io import sha256_file
from mist_transfer_benchmark.qm9.paper_evaluation import ArrayTargetLoader, run_protocol
from mist_transfer_benchmark.qm9.phase2_targets import load_targets_for_indices


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
    parser.add_argument("--qm9-csv", type=Path, help="authenticated private QM9 CSV")
    parser.add_argument("--feature-matrix", type=Path, help="SciPy sparse feature NPZ")
    parser.add_argument("--feature-manifest", type=Path, help="feature artifact manifest JSON")
    parser.add_argument("--scaffold-groups", type=Path, help="safe Unicode NPY group cache")
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        protocol = tomllib.load(handle)
    protocol["implementation_identity"] = {
        path: sha256_file(Path(path))
        for path in (
            "scripts/run_qm9_paper_evaluation.py",
            "src/mist_transfer_benchmark/qm9/paper_evaluation.py",
            "src/mist_transfer_benchmark/qm9/engineered_features.py",
        )
    }
    real_arguments = (
        args.qm9_csv,
        args.feature_matrix,
        args.feature_manifest,
        args.scaffold_groups,
    )
    if args.data is not None and any(value is not None for value in real_arguments):
        parser.error("--data cannot be combined with the real QM9 artifact arguments")
    if any(value is not None for value in real_arguments) and not all(
        value is not None for value in real_arguments
    ):
        parser.error(
            "real evaluation requires --qm9-csv, --feature-matrix, "
            "--feature-manifest, and --scaffold-groups together"
        )
    scaffold_ids = None
    if args.qm9_csv is not None:
        identities = load_qm9_identities(args.qm9_csv)
        x = sparse.load_npz(args.feature_matrix).tocsr()
        if x.shape[0] != identities.row_count:
            raise ValueError("feature matrix row count differs from authenticated QM9")
        y = load_targets_for_indices(
            args.qm9_csv,
            np.arange(identities.row_count, dtype=np.int64),
            identities,
        )
        smiles = list(identities.source_smiles)
        scaffold_ids = np.load(args.scaffold_groups, allow_pickle=False)
        feature_manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
        if feature_manifest["source_csv_sha256"] != sha256_file(args.qm9_csv):
            raise ValueError("feature manifest belongs to a different QM9 CSV")
        if feature_manifest["feature_matrix"]["file_sha256"] != sha256_file(
            args.feature_matrix
        ):
            raise ValueError("feature matrix bytes differ from the frozen manifest")
        if feature_manifest["scaffold_groups"]["file_sha256"] != sha256_file(
            args.scaffold_groups
        ):
            raise ValueError("scaffold cache bytes differ from the frozen manifest")
        feature_schema = feature_manifest["feature_schema"]
        target_identity = sha256_file(args.qm9_csv)
        target_provenance = {
            "source": str(args.qm9_csv),
            "source_csv_sha256": target_identity,
            "row_identity_sha256": identities.row_identity_sha256,
        }
    elif args.data is None:
        x, y, smiles = _smoke_data()
        target_identity = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
        target_provenance = {"source": "synthetic-smoke-v1"}
        feature_schema = {
            "representation": "synthetic-smoke-features",
            "columns": int(x.shape[1]),
            "dtype": str(x.dtype),
        }
    else:
        payload = np.load(args.data, allow_pickle=False)
        x = payload["x"]
        y = payload["y"]
        smiles = payload["smiles"].astype(str).tolist()
        target_identity = hashlib.sha256(args.data.read_bytes()).hexdigest()
        target_provenance = {"source": str(args.data)}
        feature_schema = {
            "representation": "prepared-engineered-features",
            "columns": int(x.shape[1]),
            "dtype": str(x.dtype),
        }
        scaffold_ids = (
            payload["scaffold_group_ids"] if "scaffold_group_ids" in payload else None
        )
    manifest = run_protocol(
        protocol,
        x,
        ArrayTargetLoader(
            y,
            provenance=target_provenance,
            full_target_identity=target_identity,
        ),
        smiles,
        args.output,
        feature_schema=feature_schema,
        precomputed_scaffold_group_ids=scaffold_ids,
    )
    print(f"completed {len(manifest['completed_cells'])} cells; output={args.output}")


if __name__ == "__main__":
    main()
