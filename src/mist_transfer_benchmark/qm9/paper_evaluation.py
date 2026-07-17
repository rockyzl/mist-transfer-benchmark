"""Versioned, leakage-aware repeated evaluation for paper-grade QM9 comparisons.

This module deliberately separates model selection (validation only) from the
single test evaluation performed after each model family is frozen.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import TARGET_COLUMNS
from .phase2_metrics import native_metrics

SCHEMA_VERSION = "qm9-paper-evaluation-v1"
SPLIT_KINDS = ("random", "scaffold")


class PaperEvaluationError(ValueError):
    """Raised when a paper-evaluation contract or leakage boundary is violated."""


class ArrayTargetLoader:
    """In-memory target loader with an explicit, auditable test-access gate."""

    def __init__(
        self,
        targets: np.ndarray,
        *,
        provenance: dict[str, object],
        full_target_identity: str,
    ):
        values = np.asarray(targets, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(TARGET_COLUMNS):
            raise PaperEvaluationError("target loader requires a finite [rows, 12] array")
        if not np.all(np.isfinite(values)):
            raise PaperEvaluationError("target loader values must be finite")
        self._targets = values
        self.provenance = dict(provenance)
        if not isinstance(full_target_identity, str) or len(full_target_identity) != 64:
            raise PaperEvaluationError("target loader requires an immutable SHA-256 identity")
        self.full_target_identity = full_target_identity
        self.test_authorized = False
        self.access_log: list[dict[str, object]] = []

    @property
    def rows(self) -> int:
        return len(self._targets)

    def load_selection(
        self, train_indices: np.ndarray, validation_indices: np.ndarray, *, cell_id: str
    ) -> tuple[np.ndarray, np.ndarray]:
        self.access_log.append(
            {"event": "selection", "cell_id": cell_id, "test_authorized": self.test_authorized}
        )
        if self.test_authorized:
            raise PaperEvaluationError(
                "selection target reads are forbidden after test authorization"
            )
        return self._targets[train_indices].copy(), self._targets[validation_indices].copy()

    def authorize_test(self, *, freeze_gate_sha256: str) -> None:
        if not freeze_gate_sha256:
            raise PaperEvaluationError("test authorization requires the durable freeze-gate hash")
        self.test_authorized = True
        self.access_log.append({"event": "authorize-test", "gate_sha256": freeze_gate_sha256})

    def load_test(self, test_indices: np.ndarray, *, cell_id: str) -> np.ndarray:
        self.access_log.append(
            {"event": "test", "cell_id": cell_id, "test_authorized": self.test_authorized}
        )
        if not self.test_authorized:
            raise PaperEvaluationError("test targets cannot be read before the global freeze gate")
        return self._targets[test_indices].copy()


@dataclass(frozen=True)
class ExternalPredictionArtifact:
    path: Path
    provenance: dict[str, object]


def _process_peak_rss() -> dict[str, object]:
    """Return the process-wide cumulative RSS high-water mark from getrusage."""

    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    multiplier = 1 if sys.platform == "darwin" else 1024
    return {
        "bytes": int(raw * multiplier),
        "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "semantics": "cumulative-process-high-water-mark-not-model-isolated",
        "platform_unit_interpretation": "bytes-on-macos-kibibytes-elsewhere",
    }


def _gpu_memory_observation(family: str) -> dict[str, object]:
    """Avoid attributing CUDA allocator metrics to CPU/sklearn backends."""

    return {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "available": False,
        "reason": (
            f"{family}-uses-a-CPU-or-non-torch-backend; "
            "torch-CUDA-allocator-metrics-would-be-misleading"
        ),
    }


def _inference_observation(rows: int, seconds: float) -> dict[str, float | int]:
    elapsed = max(float(seconds), np.finfo(np.float64).eps)
    return {
        "rows": int(rows),
        "seconds": float(seconds),
        "rows_per_second": float(rows / elapsed),
        "milliseconds_per_row": float(1000.0 * elapsed / rows),
    }


@dataclass(frozen=True)
class EvaluationSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, rows: int) -> None:
        joined = np.concatenate((self.train, self.validation, self.test))
        if len(joined) != rows or not np.array_equal(
            np.sort(joined), np.arange(rows, dtype=np.int64)
        ):
            raise PaperEvaluationError("split must be a disjoint cover of every row")
        if min(len(self.train), len(self.validation), len(self.test)) == 0:
            raise PaperEvaluationError("train, validation, and test must all be nonempty")


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray | sparse.spmatrix) -> str:
    digest = hashlib.sha256()
    if sparse.issparse(value):
        matrix = sparse.csr_matrix(value)
        header = {"kind": "csr", "shape": list(matrix.shape), "dtype": str(matrix.dtype)}
        digest.update(json.dumps(header, sort_keys=True).encode())
        for array in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(np.ascontiguousarray(array).tobytes())
    else:
        array = np.ascontiguousarray(value)
        header = {"kind": "dense", "shape": list(array.shape), "dtype": str(array.dtype)}
        digest.update(json.dumps(header, sort_keys=True).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez(temporary_name, **arrays)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.save(temporary_name, array, allow_pickle=False)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def scaffold_groups(smiles: Sequence[str]) -> np.ndarray:
    """Return deterministic Bemis-Murcko group IDs without using any labels."""

    groups: list[str] = []
    for row, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise PaperEvaluationError(f"invalid SMILES at row {row}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
        # Acyclic molecules have an empty Murcko scaffold. Their full canonical
        # structure is used so they do not collapse into one giant artificial group.
        groups.append(scaffold or f"acyclic:{Chem.MolToSmiles(molecule, canonical=True)}")
    return np.asarray(groups, dtype=object)


def molecular_identity_groups(smiles: Sequence[str]) -> dict[str, np.ndarray]:
    """Return exact, canonical, and connectivity identities for duplicate-safe splitting."""

    exact: list[str] = []
    canonical: list[str] = []
    connectivity: list[str] = []
    for row, value in enumerate(smiles):
        raw = str(value)
        molecule = Chem.MolFromSmiles(raw)
        if molecule is None:
            raise PaperEvaluationError(f"invalid SMILES at row {row}")
        exact.append(raw)
        canonical.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
        connectivity.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
    return {
        "exact_smiles": np.asarray(exact, dtype=object),
        "canonical_smiles": np.asarray(canonical, dtype=object),
        "connectivity_smiles": np.asarray(connectivity, dtype=object),
    }


def merge_group_relations(*relations: np.ndarray) -> np.ndarray:
    """Return connected components of multiple equality relations."""

    if not relations or any(len(value) != len(relations[0]) for value in relations):
        raise PaperEvaluationError("group relations must have one equally sized array")
    rows = len(relations[0])
    parent = np.arange(rows, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for relation in relations:
        first: dict[object, int] = {}
        for row, value in enumerate(relation):
            if value in first:
                union(row, first[value])
            else:
                first[value] = row
    return np.asarray([find(row) for row in range(rows)], dtype=np.int64)


def group_separation_audit(
    split: EvaluationSplit, groups: dict[str, np.ndarray]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, values in groups.items():
        partitions = [set(values[index]) for index in (split.train, split.validation, split.test)]
        overlaps = {
            "train_validation": len(partitions[0] & partitions[1]),
            "train_test": len(partitions[0] & partitions[2]),
            "validation_test": len(partitions[1] & partitions[2]),
        }
        if any(overlaps.values()):
            raise PaperEvaluationError(f"{name} molecular identities cross split partitions")
        result[name] = {"unique_groups": len(set(values)), "cross_partition_groups": overlaps}
    return result


def similarity_groups(
    smiles: Sequence[str], *, threshold: float = 0.70, fp_size: int = 2048
) -> np.ndarray:
    """Cluster ECFP4 fingerprints with deterministic Butina Tanimoto clustering.

    This exact implementation is O(n^2) in pairwise similarities. It is intended
    to be cached in the data-preparation stage; the runner never recomputes it
    while tuning models.
    """

    if not 0.0 < threshold < 1.0:
        raise PaperEvaluationError("similarity threshold must be between zero and one")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=fp_size, includeChirality=True, useBondTypes=True
    )
    fingerprints = []
    for row, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise PaperEvaluationError(f"invalid SMILES at row {row}")
        fingerprints.append(generator.GetFingerprint(molecule))
    distances: list[float] = []
    for index in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index], fingerprints[:index]
        )
        distances.extend(1.0 - value for value in similarities)
    clusters = Butina.ClusterData(
        distances, len(fingerprints), 1.0 - threshold, isDistData=True, reordering=True
    )
    result = np.empty(len(fingerprints), dtype=np.int64)
    for cluster_id, members in enumerate(clusters):
        result[np.asarray(members, dtype=np.int64)] = cluster_id
    return result


def _group_split(
    group_ids: np.ndarray, *, seed: int, fractions: tuple[float, float, float]
) -> EvaluationSplit:
    groups = np.asarray(group_ids)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < 3:
        raise PaperEvaluationError("group-aware splitting requires at least three groups")
    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(len(unique))
    ordered = tie_order[np.argsort(-counts[tie_order], kind="stable")]
    targets = np.asarray(fractions) * len(groups)
    assigned = np.zeros(3, dtype=np.int64)
    group_partition = np.empty(len(unique), dtype=np.int8)
    for group_index in ordered:
        deficits = targets - assigned
        partition = int(np.argmax(deficits / targets))
        group_partition[group_index] = partition
        assigned[partition] += counts[group_index]
    partitions = [np.flatnonzero(group_partition[inverse] == value) for value in range(3)]
    result = EvaluationSplit(*(np.asarray(value, dtype=np.int64) for value in partitions))
    result.validate(len(groups))
    return result


def make_evaluation_split(
    rows: int,
    *,
    kind: str,
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    scaffold_group_ids: np.ndarray | None = None,
    random_group_ids: np.ndarray | None = None,
) -> EvaluationSplit:
    """Create a frozen row-random or scaffold-grouped primary split."""

    if kind not in SPLIT_KINDS:
        raise PaperEvaluationError(f"unsupported split kind: {kind}")
    if rows < 3 or len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise PaperEvaluationError("split fractions must be three values summing to one")
    if any(value <= 0 for value in fractions):
        raise PaperEvaluationError("split fractions must be positive")
    if kind == "random":
        if random_group_ids is None or len(random_group_ids) != rows:
            raise PaperEvaluationError(
                "grouped-random split requires one connectivity identity per row"
            )
        return _group_split(np.asarray(random_group_ids), seed=seed, fractions=fractions)
    groups = scaffold_group_ids
    if groups is None or len(groups) != rows:
        raise PaperEvaluationError("scaffold split requires one precomputed group ID per row")
    return _group_split(np.asarray(groups), seed=seed, fractions=fractions)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_target_scale: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Paired row bootstrap CIs for aggregate and all twelve normalized MAEs."""

    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    scale = np.asarray(train_target_scale, dtype=np.float64)
    # Delegate complete shape/finite validation to the frozen native metric contract.
    point = native_metrics(truth, prediction, scale)
    if samples < 2 or not 0.0 < confidence < 1.0:
        raise PaperEvaluationError("bootstrap requires >=2 samples and 0<confidence<1")
    rng = np.random.default_rng(seed)
    per_target = np.empty((samples, len(TARGET_COLUMNS)), dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(truth), size=len(truth))
        per_target[sample] = np.mean(np.abs(prediction[indices] - truth[indices]), axis=0) / scale
    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(per_target, alpha, axis=0)
    upper = np.quantile(per_target, 1.0 - alpha, axis=0)
    aggregate_samples = np.mean(per_target, axis=1)
    return {
        "method": "paired-row-percentile-bootstrap",
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
        "aggregate": {
            "point": point["mean_normalized_mae_across_12_targets"],
            "lower": float(np.quantile(aggregate_samples, alpha)),
            "upper": float(np.quantile(aggregate_samples, 1.0 - alpha)),
        },
        "per_target": {
            name: {
                "point": point["per_target"][name][
                    "mae_over_training_target_standard_deviation"
                ],
                "lower": float(lower[index]),
                "upper": float(upper[index]),
            }
            for index, name in enumerate(TARGET_COLUMNS)
        },
    }


def _score(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> float:
    return float(np.mean(np.mean(np.abs(y_pred - y_true), axis=0) / scale))


def _new_model(family: str, params: dict[str, Any], seed: int):
    if family == "engineered_ridge":
        return make_pipeline(StandardScaler(with_mean=False), Ridge(**params))
    if family == "mlp":
        return make_pipeline(
            StandardScaler(with_mean=False),
            MLPRegressor(random_state=seed, **params),
        )
    if family == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise PaperEvaluationError(
                "xgboost model requested but the optional xgboost package is unavailable"
            ) from error
        return XGBRegressor(random_state=seed, n_jobs=1, **params)
    raise PaperEvaluationError(f"unsupported model family: {family}")


def select_model_parameters(
    family: str,
    candidates: list[dict[str, Any]],
    x: np.ndarray | sparse.spmatrix,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    split: EvaluationSplit,
    *,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    """Select a family using train/validation only; never index test targets."""

    if not candidates:
        raise PaperEvaluationError(f"{family} has no candidates")
    train_scale = np.std(y_train, axis=0, ddof=0)
    if np.any(train_scale <= 0):
        raise PaperEvaluationError("training target scale contains zero")
    records = []
    best: tuple[float, dict[str, Any]] | None = None
    selection_started = time.perf_counter()
    for candidate in candidates:
        model = _new_model(family, candidate, seed)
        fit_started = time.perf_counter()
        model.fit(x[split.train], y_train)
        fit_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        prediction = np.asarray(model.predict(x[split.validation]), dtype=np.float64)
        inference_seconds = time.perf_counter() - inference_started
        score = _score(y_validation, prediction, train_scale)
        records.append(
            {
                "parameters": candidate,
                "validation_score": score,
                "training_seconds": fit_seconds,
                "validation_inference": _inference_observation(
                    len(split.validation), inference_seconds
                ),
                "process_peak_rss": _process_peak_rss(),
                "gpu_memory": _gpu_memory_observation(family),
            }
        )
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    selected_model = _new_model(family, best[1], seed)
    selected_model.fit(x[split.train], y_train)
    selected_prediction = np.asarray(
        selected_model.predict(x[split.validation]), dtype=np.float64
    )
    return (
        {
            "family": family,
            "selection_uses": "validation-only-no-test-target-read",
            "candidates": records,
            "selected_parameters": best[1],
            "selected_validation_score": best[0],
            "selection_seconds": time.perf_counter() - selection_started,
        },
        selected_prediction,
    )


def select_ensemble_weights(
    predictions: dict[str, np.ndarray], y_validation: np.ndarray, train_scale: np.ndarray
) -> dict[str, object]:
    """Freeze nonnegative sum-to-one weights using validation targets only."""

    if not predictions:
        raise PaperEvaluationError("ensemble requires at least one prediction")
    names = sorted(predictions)
    stack = np.stack([predictions[name] for name in names])

    def objective(weights: np.ndarray) -> float:
        return _score(
            y_validation,
            np.tensordot(weights, stack, axes=(0, 0)),
            train_scale,
        )

    if len(names) == 1:
        weights = np.ones(1)
        success = True
    else:
        fitted = minimize(
            objective,
            np.full(len(names), 1.0 / len(names)),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(names),
            constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
            options={"maxiter": 300, "ftol": 1e-10},
        )
        weights = np.clip(fitted.x, 0.0, 1.0)
        weights /= weights.sum()
        success = bool(fitted.success)
    return {
        "model_order": names,
        "weights": {name: float(value) for name, value in zip(names, weights, strict=True)},
        "validation_score": objective(weights),
        "optimizer_success": success,
        "selection_uses": "validation-only-no-test-target-read",
    }


def blend_predictions(
    selection: dict[str, object], predictions: dict[str, np.ndarray]
) -> np.ndarray:
    names = list(selection["model_order"])
    weights = selection["weights"]
    return sum(float(weights[name]) * predictions[name] for name in names)


def paired_delta_bootstrap(
    y_true: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, float | int]:
    """Paired CI for candidate minus reference normalized MAE; negative favors candidate."""

    row_delta = np.mean(
        (np.abs(candidate - y_true) - np.abs(reference - y_true)) / scale,
        axis=1,
    )
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample = rng.integers(0, len(row_delta), size=len(row_delta))
        values[index] = np.mean(row_delta[sample])
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(np.mean(row_delta)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
        "samples": samples,
        "seed": seed,
    }
def evaluate_frozen_model(
    selection: dict[str, object],
    x: np.ndarray | sparse.spmatrix,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    split: EvaluationSplit,
    *,
    seed: int,
    bootstrap_samples: int,
    bootstrap_confidence: float,
) -> tuple[dict[str, object], np.ndarray]:
    """Refit frozen parameters and perform the family's single test evaluation."""

    family = str(selection["family"])
    frozen_parameters = dict(selection["selected_parameters"])
    train_scale = np.std(y_train, axis=0, ddof=0)
    refit_indices = np.concatenate((split.train, split.validation))
    model = _new_model(family, frozen_parameters, seed)
    fit_started = time.perf_counter()
    model.fit(x[refit_indices], np.vstack((y_train, y_validation)))
    refit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    test_prediction = np.asarray(model.predict(x[split.test]), dtype=np.float64)
    inference_seconds = time.perf_counter() - inference_started
    metrics = native_metrics(y_test, test_prediction, train_scale)
    ci = bootstrap_confidence_intervals(
        y_test,
        test_prediction,
        train_scale,
        samples=bootstrap_samples,
        seed=seed + 900_000,
        confidence=bootstrap_confidence,
    )
    return (
        {
            **selection,
            "test_evaluations_after_freeze": 1,
            "timing_seconds": {
                "selection_total": selection["selection_seconds"],
                "frozen_refit": refit_seconds,
                "test_inference": inference_seconds,
            },
            "resource_observation": {
                "schema_version": "qm9-paper-resource-observation-v1",
                "process_peak_rss": _process_peak_rss(),
                "gpu_memory": _gpu_memory_observation(family),
                "test_inference": _inference_observation(len(split.test), inference_seconds),
                "model_artifact": {
                    "bytes": None,
                    "reason": "fitted-model-is-not-persisted-by-paper-evaluation-phase-a",
                },
            },
            "test_metrics": metrics,
            "test_bootstrap_ci": ci,
        },
        test_prediction,
    )


def nearest_train_tanimoto(
    smiles: Sequence[str], train_indices: np.ndarray, query_indices: np.ndarray
) -> np.ndarray:
    """Compute exact ECFP4 nearest-train similarity without target access."""

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True, useBondTypes=True
    )
    fingerprints = []
    for row, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise PaperEvaluationError(f"invalid SMILES at row {row}")
        fingerprints.append(generator.GetFingerprint(molecule))
    train_fingerprints = [fingerprints[int(index)] for index in train_indices]
    return np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fingerprints))
            for index in query_indices
        ],
        dtype=np.float64,
    )


def similarity_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_target_scale: np.ndarray,
    similarities: np.ndarray,
    *,
    edges: Sequence[float],
) -> list[dict[str, object]]:
    """Report fixed test strata; strata never participate in model selection."""

    values = np.asarray(similarities, dtype=np.float64)
    if values.shape != (len(y_true),) or not np.all((0.0 <= values) & (values <= 1.0)):
        raise PaperEvaluationError("similarity values must be one finite [0,1] value per test row")
    boundaries = np.asarray(edges, dtype=np.float64)
    if len(boundaries) < 2 or not np.all(np.diff(boundaries) > 0):
        raise PaperEvaluationError("similarity stratum edges must be strictly increasing")
    result = []
    for position, (lower, upper) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        mask = (values >= lower) & (
            values <= upper if position == len(boundaries) - 2 else values < upper
        )
        if not np.any(mask):
            continue
        result.append(
            {
                "lower_inclusive": float(lower),
                "upper_inclusive": bool(position == len(boundaries) - 2),
                "upper": float(upper),
                "metrics": native_metrics(y_true[mask], y_pred[mask], train_target_scale),
            }
        )
    return result


def load_external_predictions(
    artifact: ExternalPredictionArtifact,
    split: EvaluationSplit,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Validate complete validation/test external predictions and provenance."""

    required_provenance = {
        "model_id",
        "source_status",
        "task_finetuned_on_qm9",
        "training_split_identity",
    }
    if set(artifact.provenance) != required_provenance:
        raise PaperEvaluationError("external prediction provenance schema is not exact")
    if artifact.provenance["task_finetuned_on_qm9"] is True:
        raise PaperEvaluationError("known task-finetuned QM9 artifacts are denied for new splits")
    if artifact.provenance["source_status"] in {"released-qm9", "legacy-released-qm9"}:
        raise PaperEvaluationError("released QM9 external artifacts are denied for new splits")
    path = Path(artifact.path)
    if path.suffix != ".npz":
        raise PaperEvaluationError("external prediction artifact must be NPZ")
    expected_keys = {
        "validation_predictions",
        "validation_source_row_index",
        "test_predictions",
        "test_source_row_index",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise PaperEvaluationError("external prediction NPZ schema is not exact")
        validation = np.asarray(payload["validation_predictions"], dtype=np.float64)
        validation_index = np.asarray(payload["validation_source_row_index"], dtype=np.int64)
        test = np.asarray(payload["test_predictions"], dtype=np.float64)
        test_index = np.asarray(payload["test_source_row_index"], dtype=np.int64)
    if validation.shape != (len(split.validation), 12) or not np.array_equal(
        validation_index, split.validation
    ):
        raise PaperEvaluationError("external validation predictions do not match frozen row order")
    if test.shape != (len(split.test), 12) or not np.array_equal(test_index, split.test):
        raise PaperEvaluationError("external test predictions do not match frozen row order")
    identity = {
        "prediction_sha256": _file_sha256(path),
        "prediction_bytes": path.stat().st_size,
        "provenance": artifact.provenance,
        "provenance_sha256": canonical_hash(artifact.provenance),
    }
    return validation, test, identity


def evaluate_external_predictions(
    prediction: np.ndarray,
    y_test: np.ndarray,
    train_target_scale: np.ndarray,
    *,
    identity: dict[str, object],
    bootstrap_samples: int,
    bootstrap_seed: int,
    bootstrap_confidence: float = 0.95,
) -> dict[str, object]:
    """Evaluate indexed frozen external/MIST predictions without training or selection."""

    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(y_test), len(TARGET_COLUMNS)):
        raise PaperEvaluationError("external predictions must be [test rows, 12] in target order")
    return {
        "source": "external-frozen-predictions-no-fine-tuning-performed-by-this-pipeline",
        "artifact_identity": identity,
        "resource_observation": {
            "schema_version": "qm9-paper-resource-observation-v1",
            "process_peak_rss": None,
            "gpu_memory": {
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "available": False,
                "reason": "external-predictions-do-not-carry-runtime-GPU-telemetry",
            },
            "test_inference": None,
            "training_seconds": None,
            "model_artifact": {
                "bytes": int(identity.get("prediction_bytes", 0)) or None,
                "reason": "external-prediction-artifact-size-not-model-weight-size",
            },
            "reason": "runtime-resource-use-cannot-be-reconstructed-from-saved-predictions",
        },
        "test_evaluations_after_freeze": 1,
        "test_metrics": native_metrics(y_test, prediction, train_target_scale),
        "test_bootstrap_ci": bootstrap_confidence_intervals(
            y_test,
            prediction,
            train_target_scale,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence=bootstrap_confidence,
        ),
    }


def run_protocol(
    protocol: dict[str, Any],
    x: np.ndarray | sparse.spmatrix,
    target_loader: ArrayTargetLoader,
    smiles: Sequence[str],
    output_dir: str | Path,
    *,
    feature_schema: dict[str, object],
    external_prediction_resolver: (
        Callable[[str, int], ExternalPredictionArtifact | None] | None
    ) = None,
    precomputed_scaffold_group_ids: np.ndarray | None = None,
    test_similarity_resolver: Callable[[str, int, EvaluationSplit], np.ndarray] | None = None,
) -> dict[str, object]:
    """Run or resume every frozen split/seed cell and write an auditable manifest."""

    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise PaperEvaluationError(f"protocol schema must be {SCHEMA_VERSION}")
    if list(protocol.get("target_order", [])) != list(TARGET_COLUMNS):
        raise PaperEvaluationError("target order differs from the frozen twelve-target contract")
    resource_contract = protocol.get("resource_instrumentation", {})
    if resource_contract != {
        "enabled": True,
        "process_rss_method": "resource.getrusage-RUSAGE_SELF-ru_maxrss",
        "gpu_policy": "null-with-reason-unless-truthful-backend-telemetry",
    }:
        raise PaperEvaluationError("resource instrumentation contract is not frozen")
    if target_loader.rows != len(smiles) or x.shape[0] != len(smiles):
        raise PaperEvaluationError("features, targets, and SMILES row counts differ")
    if not feature_schema:
        raise PaperEvaluationError("a nonempty frozen feature schema is required")
    split_kinds = list(protocol["splits"]["kinds"])
    if any(kind not in {"random", "scaffold"} for kind in split_kinds):
        raise PaperEvaluationError(
            "primary paper splits are random/scaffold; similarity belongs to test stratification"
        )
    external_enabled = bool(protocol["external_predictions"]["enabled"])
    if external_enabled != (external_prediction_resolver is not None):
        raise PaperEvaluationError(
            "external prediction resolver must be present exactly when external mode is enabled"
        )
    if protocol["external_predictions"].get("released_qm9_mist_reuse_forbidden") is not True:
        raise PaperEvaluationError("protocol must forbid released-QM9 MIST reuse on new splits")

    identities = molecular_identity_groups(smiles)
    scaffold_ids = (
        scaffold_groups(smiles)
        if precomputed_scaffold_group_ids is None
        else np.asarray(precomputed_scaffold_group_ids)
    )
    if len(scaffold_ids) != len(smiles):
        raise PaperEvaluationError("precomputed scaffold groups must contain one ID per row")
    scaffold_split_ids = merge_group_relations(
        scaffold_ids, identities["connectivity_smiles"]
    )
    splits: dict[str, EvaluationSplit] = {}
    audits: dict[str, object] = {}
    selection_targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    selection_target_hashes: dict[str, object] = {}
    external: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    for kind in split_kinds:
        for raw_seed in protocol["seeds"]:
            seed = int(raw_seed)
            cell_id = f"{kind}-seed-{seed}"
            split = make_evaluation_split(
                len(smiles),
                kind=kind,
                seed=seed,
                fractions=tuple(protocol["splits"]["fractions"]),
                scaffold_group_ids=scaffold_split_ids,
                random_group_ids=identities["connectivity_smiles"],
            )
            audit_groups = dict(identities)
            if kind == "scaffold":
                audit_groups["bemis_murcko_scaffold"] = scaffold_ids
            audits[cell_id] = group_separation_audit(split, audit_groups)
            splits[cell_id] = split
            y_train, y_validation = target_loader.load_selection(
                split.train, split.validation, cell_id=cell_id
            )
            selection_targets[cell_id] = (y_train, y_validation)
            selection_target_hashes[cell_id] = {
                "train": _array_sha256(y_train),
                "validation": _array_sha256(y_validation),
            }
            if external_enabled:
                assert external_prediction_resolver is not None
                artifact = external_prediction_resolver(kind, seed)
                if artifact is None:
                    raise PaperEvaluationError(f"external artifact is required for {cell_id}")
                external[cell_id] = load_external_predictions(artifact, split)
                expected_manifest = protocol["external_predictions"].get(
                    "expected_artifact_manifest_sha256", {}
                )
                observed_manifest_sha = canonical_hash(external[cell_id][2])
                if expected_manifest.get(cell_id) != observed_manifest_sha:
                    raise PaperEvaluationError(
                        f"external artifact manifest hash differs for {cell_id}"
                    )

    input_identity = {
        "features_sha256": _array_sha256(x),
        "feature_schema": feature_schema,
        "feature_schema_sha256": canonical_hash(feature_schema),
        "smiles_sha256": canonical_hash([str(value) for value in smiles]),
        "identity_group_sha256": {
            name: canonical_hash(values.tolist()) for name, values in identities.items()
        },
        "scaffold_group_sha256": canonical_hash(scaffold_ids.tolist()),
        "scaffold_connectivity_merged_group_sha256": canonical_hash(
            scaffold_split_ids.tolist()
        ),
        "selection_target_sha256": selection_target_hashes,
        "target_provenance": target_loader.provenance,
        "target_provenance_sha256": canonical_hash(target_loader.provenance),
        "full_target_identity_sha256": target_loader.full_target_identity,
        "external_artifacts": {cell_id: value[2] for cell_id, value in external.items()},
    }
    input_identity_sha = canonical_hash(input_identity)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol_sha = canonical_hash(protocol)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_sha256") != protocol_sha:
            raise PaperEvaluationError("resume manifest belongs to a different protocol")
        if manifest.get("input_identity_sha256") != input_identity_sha:
            raise PaperEvaluationError("resume inputs differ from the frozen run identity")
        for relative, expected_sha in manifest.get("artifact_sha256", {}).items():
            path = output / relative
            if not path.is_file() or _file_sha256(path) != expected_sha:
                raise PaperEvaluationError(f"persisted artifact identity differs: {relative}")
        for cell_id, expected_sha in manifest.get("similarity_cache_sha256", {}).items():
            similarity_path = output / "similarities" / f"{cell_id}.npz"
            if not similarity_path.is_file() or _file_sha256(similarity_path) != expected_sha:
                raise PaperEvaluationError(f"similarity cache identity differs for {cell_id}")
        if manifest.get("complete") is True:
            for cell_id, expected_sha in manifest.get("selection_sha256", {}).items():
                path = output / "selections" / f"{cell_id}.json"
                if not path.is_file() or _file_sha256(path) != expected_sha:
                    raise PaperEvaluationError(f"selection checkpoint hash differs for {cell_id}")
            for cell_id, expected_sha in manifest.get("completed_cell_sha256", {}).items():
                path = output / "cells" / f"{cell_id}.json"
                if not path.is_file() or _file_sha256(path) != expected_sha:
                    raise PaperEvaluationError(f"completed checkpoint hash differs for {cell_id}")
            summary_path = output / "summary.json"
            if not summary_path.is_file() or _file_sha256(summary_path) != manifest.get(
                "summary_sha256"
            ):
                raise PaperEvaluationError("cross-cell summary identity differs")
            return manifest
    else:
        manifest = {
            "schema_version": "qm9-paper-evaluation-manifest-v1",
            "protocol_sha256": protocol_sha,
            "input_identity_sha256": input_identity_sha,
            "scientific_status": "global-freeze-gate-precedes-every-test-target-read",
            "completed_cells": [],
            "completed_cell_sha256": {},
            "selection_sha256": {},
            "similarity_cache_sha256": {},
            "prediction_sha256": {},
            "artifact_sha256": {},
            "events": [],
        }
        protocol_snapshot = output / "protocol.snapshot.json"
        identity_path = output / "input_identity.json"
        _atomic_json(protocol_snapshot, protocol)
        _atomic_json(identity_path, input_identity)
        manifest["artifact_sha256"] = {
            "protocol.snapshot.json": _file_sha256(protocol_snapshot),
            "input_identity.json": _file_sha256(identity_path),
        }
        _atomic_json(manifest_path, manifest)

    def record_event(event: str, artifact: Path) -> None:
        manifest["events"].append(
            {
                "sequence": len(manifest["events"]) + 1,
                "event": event,
                "artifact": str(artifact.relative_to(output)),
                "sha256": _file_sha256(artifact),
            }
        )

    completed = set(manifest["completed_cells"])
    selected = set(manifest.get("selected_cells", []))
    # Stage 1: finish validation-only selection for every cell. No expression in
    # this stage indexes y[split.test].
    for kind in split_kinds:
        for seed in protocol["seeds"]:
            cell_id = f"{kind}-seed-{seed}"
            selection_path = output / "selections" / f"{cell_id}.json"
            if cell_id in selected:
                if not selection_path.is_file():
                    raise PaperEvaluationError(f"selection checkpoint missing for {cell_id}")
                if _file_sha256(selection_path) != manifest["selection_sha256"].get(cell_id):
                    raise PaperEvaluationError(f"selection checkpoint hash differs for {cell_id}")
                continue
            split = splits[cell_id]
            y_train, y_validation = selection_targets[cell_id]
            selections: dict[str, object] = {}
            validation_predictions: dict[str, np.ndarray] = {}
            for family, model_config in protocol["models"].items():
                if not model_config.get("enabled", True):
                    continue
                selection, prediction = select_model_parameters(
                    family,
                    model_config["candidates"],
                    x,
                    y_train,
                    y_validation,
                    split,
                    seed=int(seed),
                )
                selections[family] = selection
                validation_predictions[family] = prediction
            train_scale = np.std(y_train, axis=0, ddof=0)
            traditional_ensemble = select_ensemble_weights(
                validation_predictions, y_validation, train_scale
            )
            all_model_ensemble = None
            if external_enabled:
                validation_predictions["mist"] = external[cell_id][0]
                all_model_ensemble = select_ensemble_weights(
                    validation_predictions, y_validation, train_scale
                )
            selection_payload = {
                "schema_version": "qm9-paper-evaluation-selection-v1",
                "cell_id": cell_id,
                "split_kind": kind,
                "seed": int(seed),
                "models": selections,
                "traditional_ensemble": traditional_ensemble,
                "all_model_ensemble": all_model_ensemble,
                "group_separation_audit": audits[cell_id],
                "test_targets_read": False,
            }
            _atomic_json(selection_path, selection_payload)
            record_event("selection-frozen", selection_path)
            manifest.setdefault("selected_cells", []).append(cell_id)
            manifest["selected_cells"].sort()
            manifest["selection_sha256"][cell_id] = _file_sha256(selection_path)
            manifest["artifact_sha256"][str(selection_path.relative_to(output))] = _file_sha256(
                selection_path
            )
            _atomic_json(manifest_path, manifest)
            selected.add(cell_id)
    expected_cells = len(split_kinds) * len(protocol["seeds"])
    if len(selected) != expected_cells:
        raise PaperEvaluationError("global freeze gate cannot close before every cell is selected")
    freeze_gate = {
        "schema_version": "qm9-paper-evaluation-global-freeze-v1",
        "protocol_sha256": protocol_sha,
        "selected_cells": sorted(selected),
        "selection_sha256": manifest["selection_sha256"],
        "test_access_authorized": True,
        "condition": "all split/seed/model selections persisted before any test-target read",
        "input_identity_sha256": input_identity_sha,
    }
    freeze_path = output / "global_freeze_gate.json"
    _atomic_json(freeze_path, freeze_gate)
    record_event("global-freeze-gate-closed", freeze_path)
    gate_sha = _file_sha256(freeze_path)
    target_loader.authorize_test(freeze_gate_sha256=gate_sha)
    manifest["test_access_gate_sha256"] = gate_sha
    manifest["artifact_sha256"]["global_freeze_gate.json"] = gate_sha
    _atomic_json(manifest_path, manifest)

    # Stage 2: the global gate is durable. Refit frozen selections, then evaluate test.
    for kind in split_kinds:
        for seed in protocol["seeds"]:
            cell_id = f"{kind}-seed-{seed}"
            cell_path = output / "cells" / f"{cell_id}.json"
            if cell_id in completed:
                if not cell_path.is_file():
                    raise PaperEvaluationError(f"manifest checkpoint missing for {cell_id}")
                if _file_sha256(cell_path) != manifest["completed_cell_sha256"].get(cell_id):
                    raise PaperEvaluationError(f"completed checkpoint hash differs for {cell_id}")
                continue
            split = splits[cell_id]
            y_train, y_validation = selection_targets[cell_id]
            y_test = target_loader.load_test(split.test, cell_id=cell_id)
            selection_payload = json.loads(
                (output / "selections" / f"{cell_id}.json").read_text(encoding="utf-8")
            )
            models: dict[str, object] = {}
            predictions: dict[str, np.ndarray] = {}
            cell_started = time.perf_counter()
            for family, selection in selection_payload["models"].items():
                result, prediction = evaluate_frozen_model(
                    selection,
                    x,
                    y_train,
                    y_validation,
                    y_test,
                    split,
                    seed=int(seed),
                    bootstrap_samples=int(protocol["bootstrap"]["samples"]),
                    bootstrap_confidence=float(protocol["bootstrap"]["confidence"]),
                )
                models[family] = result
                predictions[family] = prediction
                prediction_dir = output / "predictions"
                prediction_dir.mkdir(exist_ok=True)
                prediction_path = prediction_dir / f"{cell_id}-{family}.npy"
                _atomic_npy(prediction_path, prediction)
                prediction_sha = _file_sha256(prediction_path)
                manifest["prediction_sha256"][f"{cell_id}:{family}"] = prediction_sha
                manifest["artifact_sha256"][str(prediction_path.relative_to(output))] = (
                    prediction_sha
                )
            train_scale = np.std(y_train, axis=0, ddof=0)
            if external_enabled:
                predictions["mist"] = external[cell_id][1]
                models["mist"] = evaluate_external_predictions(
                    predictions["mist"],
                    y_test,
                    train_scale,
                    identity=external[cell_id][2],
                    bootstrap_samples=int(protocol["bootstrap"]["samples"]),
                    bootstrap_seed=int(seed) + 950_000,
                    bootstrap_confidence=float(protocol["bootstrap"]["confidence"]),
                )
            ensemble_inference_started = time.perf_counter()
            traditional_prediction = blend_predictions(
                selection_payload["traditional_ensemble"], predictions
            )
            ensemble_inference_seconds = time.perf_counter() - ensemble_inference_started
            predictions["traditional_ensemble"] = traditional_prediction
            models["traditional_ensemble"] = {
                "selection": selection_payload["traditional_ensemble"],
                "resource_observation": {
                    "schema_version": "qm9-paper-resource-observation-v1",
                    "process_peak_rss": _process_peak_rss(),
                    "gpu_memory": _gpu_memory_observation("traditional-ensemble-blend"),
                    "test_inference": _inference_observation(
                        len(split.test), ensemble_inference_seconds
                    ),
                    "training_seconds": 0.0,
                    "model_artifact": {
                        "bytes": None,
                        "reason": "ensemble-is-weights-in-selection-artifact-not-a-model-file",
                    },
                },
                "test_metrics": native_metrics(y_test, traditional_prediction, train_scale),
                "test_bootstrap_ci": bootstrap_confidence_intervals(
                    y_test,
                    traditional_prediction,
                    train_scale,
                    samples=int(protocol["bootstrap"]["samples"]),
                    seed=int(seed) + 970_000,
                    confidence=float(protocol["bootstrap"]["confidence"]),
                ),
            }
            if external_enabled:
                all_inference_started = time.perf_counter()
                all_prediction = blend_predictions(
                    selection_payload["all_model_ensemble"], predictions
                )
                all_inference_seconds = time.perf_counter() - all_inference_started
                predictions["all_model_ensemble"] = all_prediction
                models["all_model_ensemble"] = {
                    "selection": selection_payload["all_model_ensemble"],
                    "resource_observation": {
                        "schema_version": "qm9-paper-resource-observation-v1",
                        "process_peak_rss": _process_peak_rss(),
                        "gpu_memory": _gpu_memory_observation("all-model-ensemble-blend"),
                        "test_inference": _inference_observation(
                            len(split.test), all_inference_seconds
                        ),
                        "training_seconds": 0.0,
                        "model_artifact": {
                            "bytes": None,
                            "reason": "ensemble-is-weights-in-selection-artifact-not-a-model-file",
                        },
                    },
                    "test_metrics": native_metrics(y_test, all_prediction, train_scale),
                    "test_bootstrap_ci": bootstrap_confidence_intervals(
                        y_test,
                        all_prediction,
                        train_scale,
                        samples=int(protocol["bootstrap"]["samples"]),
                        seed=int(seed) + 980_000,
                        confidence=float(protocol["bootstrap"]["confidence"]),
                    ),
                }
            similarity_path = output / "similarities" / f"{cell_id}.npz"
            if similarity_path.exists():
                expected_sha = manifest["similarity_cache_sha256"].get(cell_id)
                if expected_sha and _file_sha256(similarity_path) != expected_sha:
                    raise PaperEvaluationError(f"similarity cache identity differs for {cell_id}")
                try:
                    with np.load(similarity_path, allow_pickle=False) as payload:
                        if set(payload.files) != {
                            "source_row_index",
                            "nearest_train_tanimoto",
                        }:
                            raise PaperEvaluationError("similarity cache schema is not exact")
                        if not np.array_equal(payload["source_row_index"], split.test):
                            raise PaperEvaluationError("similarity cache row order differs")
                        similarities = np.asarray(payload["nearest_train_tanimoto"])
                except (OSError, ValueError, PaperEvaluationError):
                    if expected_sha:
                        raise
                    similarity_path.unlink()
                    similarities = (
                        test_similarity_resolver(kind, int(seed), split)
                        if test_similarity_resolver is not None
                        else nearest_train_tanimoto(smiles, split.train, split.test)
                    )
                    _atomic_npz(
                        similarity_path,
                        source_row_index=split.test,
                        nearest_train_tanimoto=np.asarray(similarities),
                    )
                if not expected_sha:
                    reconciled_sha = _file_sha256(similarity_path)
                    manifest["similarity_cache_sha256"][cell_id] = reconciled_sha
                    manifest["artifact_sha256"][str(similarity_path.relative_to(output))] = (
                        reconciled_sha
                    )
                    record_event("orphan-similarity-cache-reconciled", similarity_path)
            else:
                similarities = (
                    test_similarity_resolver(kind, int(seed), split)
                    if test_similarity_resolver is not None
                    else nearest_train_tanimoto(smiles, split.train, split.test)
                )
                _atomic_npz(
                    similarity_path,
                    source_row_index=split.test,
                    nearest_train_tanimoto=np.asarray(similarities),
                )
                manifest["similarity_cache_sha256"][cell_id] = _file_sha256(similarity_path)
                manifest["artifact_sha256"][str(similarity_path.relative_to(output))] = (
                    _file_sha256(similarity_path)
                )
                record_event("similarity-cache-frozen", similarity_path)
            for family, prediction in predictions.items():
                models[family]["test_similarity_strata"] = similarity_stratified_metrics(
                    y_test,
                    prediction,
                    train_scale,
                    similarities,
                    edges=protocol["similarity_analysis"]["edges"],
                )
            comparisons = {
                family: paired_delta_bootstrap(
                    y_test,
                    traditional_prediction,
                    prediction,
                    train_scale,
                    samples=int(protocol["bootstrap"]["samples"]),
                    seed=int(seed) + 990_000 + position,
                    confidence=float(protocol["bootstrap"]["confidence"]),
                )
                for position, (family, prediction) in enumerate(predictions.items())
                if family not in {"traditional_ensemble", "all_model_ensemble", "mist"}
            }
            all_model_comparisons = None
            if external_enabled:
                all_model_comparisons = {
                    reference: paired_delta_bootstrap(
                        y_test,
                        predictions["all_model_ensemble"],
                        predictions[reference],
                        train_scale,
                        samples=int(protocol["bootstrap"]["samples"]),
                        seed=int(seed) + 995_000 + position,
                        confidence=float(protocol["bootstrap"]["confidence"]),
                    )
                    for position, reference in enumerate(("traditional_ensemble", "mist"))
                }
            cell = {
                "schema_version": "qm9-paper-evaluation-cell-v1",
                "cell_id": cell_id,
                "split_kind": kind,
                "seed": int(seed),
                "split_counts": {
                    "train": len(split.train),
                    "validation": len(split.validation),
                    "test": len(split.test),
                },
                "models": models,
                "paired_delta_traditional_ensemble_minus_model": comparisons,
                "paired_delta_all_model_ensemble_minus_reference": all_model_comparisons,
                "test_target_access_after_gate_sha256": gate_sha,
                "wall_seconds": time.perf_counter() - cell_started,
            }
            _atomic_json(cell_path, cell)
            record_event("test-cell-completed", cell_path)
            manifest["completed_cells"].append(cell_id)
            manifest["completed_cells"].sort()
            manifest["completed_cell_sha256"][cell_id] = _file_sha256(cell_path)
            manifest["artifact_sha256"][str(cell_path.relative_to(output))] = _file_sha256(
                cell_path
            )
            _atomic_json(manifest_path, manifest)
            completed.add(cell_id)
    manifest["complete"] = len(completed) == (
        len(split_kinds) * len(protocol["seeds"])
    )
    summary: dict[str, object] = {
        "schema_version": "qm9-paper-evaluation-summary-v1",
        "paper_output_factors": [
            "fixed-seeds",
            "grouped-random-and-scaffold-splits",
            "bootstrap-confidence-intervals",
            "twelve-target-statistics",
            "training-and-refit-time",
            "inference-throughput-and-resource-high-water-marks",
        ],
    }
    by_method: dict[str, list[dict[str, object]]] = {}
    paired_by_reference: dict[str, list[dict[str, object]]] = {}
    for cell_id in sorted(completed):
        cell = json.loads((output / "cells" / f"{cell_id}.json").read_text(encoding="utf-8"))
        for method, result in cell["models"].items():
            resource_observation = result.get("resource_observation", {})
            inference = resource_observation.get("test_inference")
            timing = result.get("timing_seconds", {})
            by_method.setdefault(method, []).append(
                {
                    "cell_id": cell_id,
                    "split_kind": cell["split_kind"],
                    "seed": cell["seed"],
                    "score": result["test_metrics"]["mean_normalized_mae_across_12_targets"],
                    "selection_seconds": timing.get("selection_total"),
                    "training_or_refit_seconds": timing.get(
                        "frozen_refit", resource_observation.get("training_seconds")
                    ),
                    "test_inference": inference,
                    "process_peak_rss": resource_observation.get("process_peak_rss"),
                    "gpu_memory": resource_observation.get("gpu_memory"),
                    "model_artifact": resource_observation.get("model_artifact"),
                }
            )
        for reference, interval in cell[
            "paired_delta_traditional_ensemble_minus_model"
        ].items():
            paired_by_reference.setdefault(reference, []).append(
                {
                    "cell_id": cell_id,
                    "split_kind": cell["split_kind"],
                    "seed": cell["seed"],
                    **interval,
                }
            )
    summary["methods"] = {
        method: {
            "cells": records,
            "mean_across_split_seed_cells": float(np.mean([item["score"] for item in records])),
            "by_split": {
                kind: float(
                    np.mean([item["score"] for item in records if item["split_kind"] == kind])
                )
                for kind in split_kinds
            },
            "cost_summary": {
                "selection_seconds_sum": float(
                    sum(item["selection_seconds"] or 0.0 for item in records)
                ),
                "training_or_refit_seconds_sum": float(
                    sum(item["training_or_refit_seconds"] or 0.0 for item in records)
                ),
                "test_inference_seconds_sum": float(
                    sum(
                        item["test_inference"]["seconds"]
                        for item in records
                        if item["test_inference"] is not None
                    )
                ),
                "test_rows_sum": int(
                    sum(
                        item["test_inference"]["rows"]
                        for item in records
                        if item["test_inference"] is not None
                    )
                ),
                "effective_test_rows_per_second": float(
                    sum(
                        item["test_inference"]["rows"]
                        for item in records
                        if item["test_inference"] is not None
                    )
                    / max(
                        sum(
                            item["test_inference"]["seconds"]
                            for item in records
                            if item["test_inference"] is not None
                        ),
                        np.finfo(np.float64).eps,
                    )
                ),
                "peak_process_rss_bytes_max": max(
                    (
                        item["process_peak_rss"]["bytes"]
                        for item in records
                        if item["process_peak_rss"] is not None
                    ),
                    default=None,
                ),
                "rss_semantics": "cumulative-process-high-water-mark-not-additive",
                "gpu_memory_note": (
                    "null unless the evaluated backend exposes truthful CUDA telemetry"
                ),
            },
        }
        for method, records in by_method.items()
    }
    summary["paired_delta_traditional_ensemble_minus_model"] = {
        reference: {
            "cells": records,
            "mean_point_across_split_seed_cells": float(
                np.mean([item["point"] for item in records])
            ),
            "by_split_mean_point": {
                kind: float(
                    np.mean([item["point"] for item in records if item["split_kind"] == kind])
                )
                for kind in split_kinds
            },
        }
        for reference, records in paired_by_reference.items()
    }
    summary_path = output / "summary.json"
    _atomic_json(summary_path, summary)
    manifest["summary_sha256"] = _file_sha256(summary_path)
    manifest["artifact_sha256"]["summary.json"] = _file_sha256(summary_path)
    record_event("cross-cell-summary-completed", summary_path)
    _atomic_json(manifest_path, manifest)
    return manifest
