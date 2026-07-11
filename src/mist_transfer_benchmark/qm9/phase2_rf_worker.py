"""Isolated random-forest worker so a failed forest cannot erase Ridge evidence."""

from __future__ import annotations

import argparse
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tomllib
from scipy import sparse

from .data import load_qm9_identities
from .io import atomic_write_json
from .phase2_metrics import native_metrics
from .phase2_scaler import fit_frozen_scaler
from .phase2_selection import random_forest_candidates
from .phase2_targets import load_targets_for_indices
from .split import reconstruct_candidate_split


def _rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return float(value) / divisor


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_worker(
    *,
    config_path: Path,
    source_path: Path,
    feature_path: Path,
    candidate_id: str,
    mode: str,
    output_path: Path,
) -> None:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    candidates = {item.candidate_id: item for item in random_forest_candidates(config)}
    if candidate_id not in candidates:
        raise ValueError(f"unknown frozen random-forest candidate: {candidate_id}")
    if candidates[candidate_id].parameters["n_jobs"] > config["resource_budget"][
        "cpu_worker_limit"
    ]:
        raise ValueError("random-forest n_jobs exceeds the frozen CPU-worker limit")
    data = load_qm9_identities(source_path)
    split = reconstruct_candidate_split(data.row_count)
    matrix = sparse.load_npz(feature_path)
    y_train = load_targets_for_indices(source_path, split.train, data)
    scaler, _ = fit_frozen_scaler(config, y_train)
    candidate = candidates[candidate_id]
    started = time.monotonic()
    candidate.estimator.fit(matrix[split.train], scaler.transform(y_train))
    if mode == "validation":
        y_validation = load_targets_for_indices(source_path, split.validation, data)
        prediction = scaler.inverse_transform(candidate.estimator.predict(matrix[split.validation]))
        metrics = native_metrics(y_validation, prediction, scaler.scale_)
        atomic_write_json(
            output_path,
            {
                "candidate_id": candidate_id,
                "parameters": candidate.parameters,
                "parameters_sha256": candidate.parameters_sha256,
                "runtime_seconds": time.monotonic() - started,
                "peak_rss_gib": _rss_gib(),
                "mean_normalized_mae_across_12_targets": metrics[
                    "mean_normalized_mae_across_12_targets"
                ],
                "metrics": metrics,
            },
            mode=0o600,
        )
    elif mode == "test-prediction":
        prediction = scaler.inverse_transform(candidate.estimator.predict(matrix[split.test]))
        _atomic_save_array(output_path, prediction)
    else:
        raise ValueError(f"unsupported worker mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", choices=("validation", "test-prediction"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_worker(
        config_path=args.config,
        source_path=args.source,
        feature_path=args.features,
        candidate_id=args.candidate,
        mode=args.mode,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
