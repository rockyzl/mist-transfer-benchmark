#!/usr/bin/env python3
"""Prepare private, reusable QM9 features for the repeated paper evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from mist_transfer_benchmark.qm9.data import load_qm9_identities
from mist_transfer_benchmark.qm9.engineered_features import (
    build_count_ecfp4_plus_globals,
    engineered_feature_schema,
)
from mist_transfer_benchmark.qm9.io import atomic_write_json, sha256_file
from mist_transfer_benchmark.qm9.paper_evaluation import _array_sha256, scaffold_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qm9-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    feature_path = args.output / "feature_matrix.npz"
    scaffold_path = args.output / "scaffold_group_ids.npy"
    manifest_path = args.output / "manifest.json"
    if any(path.exists() for path in (feature_path, scaffold_path, manifest_path)):
        raise FileExistsError("paper feature output already exists; refusing to overwrite")

    identities = load_qm9_identities(args.qm9_csv)
    matrix = build_count_ecfp4_plus_globals(
        identities.source_smiles,
        progress=lambda message: print(message, flush=True),
    )
    sparse.save_npz(feature_path, matrix, compressed=True)
    scaffold_ids = np.asarray(scaffold_groups(identities.source_smiles), dtype=str)
    np.save(scaffold_path, scaffold_ids, allow_pickle=False)
    manifest = {
        "schema_version": "qm9-paper-feature-artifact-v1",
        "source_csv_sha256": sha256_file(args.qm9_csv),
        "source_row_identity_sha256": identities.row_identity_sha256,
        "source_smiles_sha256": identities.raw_smiles_sha256,
        "rows": identities.row_count,
        "feature_schema": engineered_feature_schema(),
        "feature_matrix": {
            "path": feature_path.name,
            "file_sha256": sha256_file(feature_path),
            "semantic_sha256": _array_sha256(matrix),
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
        },
        "scaffold_groups": {
            "path": scaffold_path.name,
            "file_sha256": sha256_file(scaffold_path),
            "groups_sha256": _array_sha256(scaffold_ids),
            "count": int(len(np.unique(scaffold_ids))),
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
