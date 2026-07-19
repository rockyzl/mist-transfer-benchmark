"""Train-only post-specified model selection for the fixed-QM9 comparison.

This module intentionally has no outer-validation or test-target loader API.
Every ranking decision is made on an inner group-safe tune partition of the
fixed training rows.  Its final product is a hash-pinned selection lock, not a
test score.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .constants import TARGET_COLUMNS
from .selection_v3_data import build_group_folds, fit_feature_recipe

CONFIG_SCHEMA = "qm9-model-selection-v3-config-v1"
LOCK_SCHEMA = "qm9-model-selection-v3-lock-v1"
EXPECTED_SEEDS = [20260713, 20260729, 20260811, 20260823, 20260907]
EXPECTED_RECIPES = [
    "descriptor_scaled_only",
    "all_sparse_scaled",
    "all_raw",
    "log_count_descriptor_scaled",
    "drop_constant_descriptor_scaled",
]
REVIEWED_CONFIG_SHA256 = "f082bec7e63db44b61c4b8f02765601b353bd4c5e6ba8bfa976316223e10912f"


class ModelSelectionError(ValueError):
    """Raised when a train-only selection or budget boundary is violated."""


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def process_peak_rss() -> dict[str, object]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "bytes": raw * (1 if os.uname().sysname == "Darwin" else 1024),
        "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "semantics": "cumulative-process-high-water-mark-not-candidate-isolated",
    }


@dataclass
class BudgetLedger:
    max_fits: int
    max_seconds: float
    started: float = field(default_factory=time.perf_counter)
    fits: list[dict[str, object]] = field(default_factory=list)

    def reserve(
        self,
        family: str,
        candidate_id: str,
        seed: int,
        *,
        wave: str = "smoke",
        strategy: str = "identity",
        fold_id: int = -1,
    ) -> None:
        if len(self.fits) >= self.max_fits:
            raise ModelSelectionError("candidate-fit budget exhausted")
        if time.perf_counter() - self.started >= self.max_seconds:
            raise ModelSelectionError("wall-clock selection budget exhausted")
        trial_key = [wave, strategy, candidate_id, int(fold_id), int(seed)]
        if any(row.get("trial_key") == trial_key for row in self.fits):
            raise ModelSelectionError("duplicate candidate trial key")
        self.fits.append(
            {
                "sequence": len(self.fits) + 1,
                "family": family,
                "candidate_id": candidate_id,
                "seed": seed,
                "training_seed": seed,
                "fold_id": int(fold_id),
                "wave": wave,
                "strategy": strategy,
                "trial_key": trial_key,
                "target_estimator_fits": 12 if family.startswith("xgboost") else 1,
                "status": "reserved",
            }
        )

    def complete(self, seconds: float) -> None:
        self.fits[-1].update(status="completed", seconds=float(seconds))

    def fail(self, seconds: float, error: BaseException) -> None:
        self.fits[-1].update(
            status="failed",
            seconds=float(seconds),
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def payload(self) -> dict[str, object]:
        return {
            "max_candidate_fits": self.max_fits,
            "max_wall_seconds": self.max_seconds,
            "completed_candidate_fits": sum(row["status"] == "completed" for row in self.fits),
            "candidate_pipeline_fits": len(self.fits),
            "target_estimator_fits": sum(int(row["target_estimator_fits"]) for row in self.fits),
            "elapsed_seconds": time.perf_counter() - self.started,
            "fits": self.fits,
        }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ModelSelectionError(f"config schema must be {CONFIG_SCHEMA}")
    if canonical_hash(config) != REVIEWED_CONFIG_SHA256:
        raise ModelSelectionError("config differs from the reviewed numeric contract hash")
    if (
        list(config["training_seeds"]) != EXPECTED_SEEDS
        or int(config["fold_assignment_seed"]) != EXPECTED_SEEDS[0]
        or list(config["target_order"]) != list(TARGET_COLUMNS)
        or list(config["feature_selection"]["recipes"]) != EXPECTED_RECIPES
    ):
        raise ModelSelectionError("reviewed config axes differ")


def group_safe_inner_split(
    groups: Sequence[object], *, tune_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(groups)
    unique = np.unique(values)
    rng = np.random.default_rng(seed)
    tune_groups = set(rng.permutation(unique)[: max(1, round(len(unique) * tune_fraction))])
    tune = np.asarray([i for i, value in enumerate(values) if value in tune_groups], dtype=np.int64)
    train = np.asarray(
        [i for i, value in enumerate(values) if value not in tune_groups], dtype=np.int64
    )
    if not len(train) or not len(tune) or set(values[train]) & set(values[tune]):
        raise ModelSelectionError("inner group split is invalid")
    return train, tune


def normalized_mae(
    truth: np.ndarray, prediction: np.ndarray, train_targets: np.ndarray
) -> tuple[float, np.ndarray]:
    scale = np.std(train_targets, axis=0)
    if np.any(scale <= 0) or not np.all(np.isfinite(prediction)):
        raise ModelSelectionError("invalid target scale or prediction")
    per_target = np.mean(np.abs(prediction - truth), axis=0) / scale
    return float(per_target.mean()), per_target


def rank_candidates(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ModelSelectionError("cannot rank an empty candidate set")
    if any(record.get("selection_partition") != "inner-tune-only" for record in records):
        raise ModelSelectionError("candidate ranking attempted outside inner tune")
    return sorted(
        (dict(record) for record in records), key=lambda row: (row["mean_mnmae"], row["id"])
    )


def assert_paired_ablation(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    allowed = {"id", "scale_all_sparse"}
    differing = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
    if differing - allowed or differing != allowed:
        raise ModelSelectionError("preprocessing ablation changes more than one factor")


def monitor_curve(
    train_loss: Sequence[float],
    validation: Sequence[float],
    *,
    max_epochs: int,
    restored_hash: str,
    best_hash: str,
    thresholds: Mapping[str, Any],
    per_target_nmae: Sequence[Sequence[float]] | None = None,
    gradient_norm: Sequence[float] | None = None,
    parameter_norm: Sequence[float] | None = None,
    gpu_memory_bytes: Sequence[int | None] | None = None,
    process_rss_bytes: Sequence[int] | None = None,
) -> dict[str, object]:
    train = np.asarray(train_loss, dtype=float)
    tune = np.asarray(validation, dtype=float)
    anomalies: list[str] = []
    warnings: list[str] = []
    if (
        not len(train)
        or len(train) != len(tune)
        or not np.all(np.isfinite(train))
        or not np.all(np.isfinite(tune))
    ):
        anomalies.append("nonfinite-or-misaligned-curve")
    if restored_hash != best_hash:
        anomalies.append("restored-checkpoint-hash-differs-from-best")
    target_curve = np.asarray(per_target_nmae if per_target_nmae is not None else [])
    if target_curve.size and target_curve.shape != (len(train), len(TARGET_COLUMNS)):
        anomalies.append("per-target-curve-shape-differs")
    gradients = np.asarray(gradient_norm if gradient_norm is not None else [], dtype=float)
    parameters = np.asarray(parameter_norm if parameter_norm is not None else [], dtype=float)
    if gradients.size and (len(gradients) != len(train) or not np.all(np.isfinite(gradients))):
        anomalies.append("gradient-norm-is-nonfinite-or-misaligned")
    if parameters.size and (len(parameters) != len(train) or not np.all(np.isfinite(parameters))):
        anomalies.append("parameter-norm-is-nonfinite-or-misaligned")
    if gradients.size and np.any(
        gradients > float(thresholds.get("gradient_norm_hard_max", np.inf))
    ):
        anomalies.append("gradient-norm-exceeds-hard-maximum")
    if gradients.size:
        low = gradients < float(thresholds.get("gradient_norm_hard_min", -np.inf))
        run = maximum_low_run = 0
        for value in low:
            run = run + 1 if value else 0
            maximum_low_run = max(maximum_low_run, run)
        if maximum_low_run >= int(thresholds.get("gradient_min_consecutive_batches", 1)):
            anomalies.append("gradient-norm-below-hard-minimum-consecutively")
    if target_curve.size and target_curve.shape == (len(train), len(TARGET_COLUMNS)):
        best_so_far = target_curve[0].copy()
        degradation = float(thresholds.get("target_degradation_warning", np.inf))
        for row in target_curve[1:]:
            if np.any(row - best_so_far > degradation):
                warnings.append("per-target-nmae-degraded")
                break
            best_so_far = np.minimum(best_so_far, row)
    increases = 0
    maximum = 0
    for before, after in zip(tune, tune[1:], strict=False):
        increases = increases + 1 if after > before else 0
        maximum = max(maximum, increases)
    if maximum >= int(thresholds["validation_increase_mark_after"]):
        warnings.append("inner-tune-mnmae-increased-consecutively")
    return {
        "status": "abnormal" if anomalies else "warning" if warnings else "normal",
        "train_standardized_mse": train.tolist(),
        "inner_tune_mnmae": tune.tolist(),
        "inner_tune_per_target_nmae": target_curve.tolist(),
        "gradient_norm": gradients.tolist(),
        "parameter_norm": parameters.tolist(),
        "gpu_memory_bytes": list(gpu_memory_bytes or []),
        "process_rss_bytes": list(process_rss_bytes or []),
        "best_epoch": int(np.argmin(tune)) + 1 if len(tune) else None,
        "restored_checkpoint_sha256": restored_hash,
        "best_checkpoint_sha256": best_hash,
        "stop_reason": "patience-exhausted" if len(train) < max_epochs else "max-epochs-reached",
        "maximum_consecutive_validation_increases": maximum,
        "warnings": warnings,
        "anomalies": anomalies,
    }


def _scale_features(
    x: np.ndarray | sparse.spmatrix, train: np.ndarray, tune: np.ndarray, *, scale_all: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if scale_all:
        scaler = StandardScaler(with_mean=False).fit(x[train])
        return (
            scaler.transform(x[train]),
            scaler.transform(x[tune]),
            {"kind": "all_sparse_scaled", "fit_rows_sha256": canonical_hash(train.tolist())},
        )
    return (
        x[train],
        x[tune],
        {
            "kind": "descriptor_scaled_only-or-precomputed-recipe",
            "fit_rows_sha256": canonical_hash(train.tolist()),
        },
    )


def fit_real_xgboost(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_tune: sparse.csr_matrix,
    y_tune: np.ndarray,
    candidate: Mapping[str, Any],
    *,
    seed: int,
    early_stopping_rounds: int,
) -> tuple[np.ndarray, list[int]]:
    from xgboost import XGBRegressor

    params = {key: value for key, value in candidate.items() if key != "id"}
    predictions, rounds = [], []
    for target in range(len(TARGET_COLUMNS)):
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=seed + target,
            early_stopping_rounds=early_stopping_rounds,
            **params,
        )
        model.fit(
            x_train, y_train[:, target], eval_set=[(x_tune, y_tune[:, target])], verbose=False
        )
        predictions.append(model.predict(x_tune))
        rounds.append(int(getattr(model, "best_iteration", params["n_estimators"] - 1)) + 1)
    return np.column_stack(predictions), rounds


def fit_real_torch_mlp(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_tune: sparse.csr_matrix,
    y_tune: np.ndarray,
    candidate: Mapping[str, Any],
    training: Mapping[str, Any],
    monitoring: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    import torch

    torch.manual_seed(seed)
    mean, scale = y_train.mean(axis=0), y_train.std(axis=0)
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ModelSelectionError("MLP target scale is nonfinite or zero")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    widths = [x_train.shape[1], *candidate["hidden_dims"], len(TARGET_COLUMNS)]
    layers: list[Any] = []
    for left, right in zip(widths[:-2], widths[1:-1], strict=True):
        layers.extend(
            [
                torch.nn.Linear(left, right),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(candidate["dropout"])),
            ]
        )
    layers.append(torch.nn.Linear(widths[-2], widths[-1]))
    model = torch.nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
    )
    batch_size = min(int(training["batch_size"]), len(y_train))
    max_epochs, patience = int(training["max_epochs"]), int(training["patience"])
    train_curve, tune_curve, target_curve, grad_curve, param_curve, rss_curve, gpu_curve = (
        [] for _ in range(7)
    )
    best_score, best_state, best_hash, stale = float("inf"), None, None, 0
    stop_reason = "max-epochs-reached"
    low_gradient_batches = 0
    for epoch in range(max_epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(y_train))
        losses, gradients = [], []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            xb = torch.from_numpy(x_train[batch].toarray().astype(np.float32)).to(device)
            yb = torch.from_numpy(((y_train[batch] - mean) / scale).astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            if not torch.isfinite(loss):
                raise ModelSelectionError("nonfinite MLP loss")
            loss.backward()
            grad = float(
                torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))
            )
            if grad > float(monitoring["gradient_norm_hard_max"]):
                raise ModelSelectionError("MLP gradient exceeds threshold")
            if grad < float(monitoring["gradient_norm_hard_min"]):
                low_gradient_batches += 1
            else:
                low_gradient_batches = 0
            if low_gradient_batches >= int(monitoring["gradient_min_consecutive_batches"]):
                raise ModelSelectionError("MLP gradient stayed below threshold")
            gradients.append(grad)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            tune_tensor = torch.from_numpy(x_tune.toarray().astype(np.float32)).to(device)
            standardized = model(tune_tensor).cpu().numpy()
        prediction = standardized * scale + mean
        score, per_target = normalized_mae(y_tune, prediction, y_train)
        train_curve.append(float(np.mean(losses)))
        tune_curve.append(score)
        target_curve.append(per_target.tolist())
        grad_curve.append(max(gradients))
        param_curve.append(
            float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())))
        )
        rss_curve.append(int(process_peak_rss()["bytes"]))
        gpu_curve.append(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        if epoch == 4 and score > float(monitoring["max_validation_mnmae_after_five_epochs"]):
            raise ModelSelectionError("MLP epoch-five MNMAE threshold exceeded")
        if epoch == 9:
            improvement = (train_curve[0] - min(train_curve)) / max(abs(train_curve[0]), 1e-12)
            if improvement < float(monitoring["minimum_relative_train_improvement_by_epoch_ten"]):
                raise ModelSelectionError("MLP epoch-ten improvement threshold missed")
        if score < best_score - float(training["min_delta"]):
            best_score, stale = score, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_hash = canonical_hash({key: value.tolist() for key, value in best_state.items()})
        else:
            stale += 1
            if stale >= patience:
                stop_reason = "patience-exhausted"
                break
    if best_state is None or best_hash is None:
        raise ModelSelectionError("MLP produced no checkpoint")
    model.load_state_dict(best_state)
    restored = canonical_hash({key: value.tolist() for key, value in model.state_dict().items()})
    curve = monitor_curve(
        train_curve,
        tune_curve,
        max_epochs=max_epochs,
        restored_hash=restored,
        best_hash=best_hash,
        thresholds=monitoring,
        per_target_nmae=target_curve,
        gradient_norm=grad_curve,
        parameter_norm=param_curve,
        gpu_memory_bytes=gpu_curve,
        process_rss_bytes=rss_curve,
    )
    curve["stop_reason"] = stop_reason
    if curve["status"] == "abnormal":
        raise ModelSelectionError("MLP monitoring marked abnormal")
    model.eval()
    with torch.no_grad():
        tune_tensor = torch.from_numpy(x_tune.toarray().astype(np.float32)).to(device)
        final = model(tune_tensor).cpu().numpy() * scale + mean
    return final, curve


def recursive_why(
    identifier: str,
    question: str,
    evidence: Mapping[str, Any],
    decision: str,
    alternatives: Sequence[str],
    *,
    parent_id: str | None = None,
    root_status: str = "supported",
    containment: str = "preserve artifacts and continue only under the frozen contract",
    proposed_change: str = "none",
) -> dict[str, object]:
    payload = {
        "schema_version": "qm9-model-selection-v3-why-node-v2",
        "issue_id": identifier,
        "parent_id": parent_id,
        "gate": "G2-G3",
        "test_or_outer_labels_accessed": False,
        "fact": {
            "observation": question,
            "evidence": dict(evidence),
            "evidence_hashes": [canonical_hash(evidence)],
        },
        "hypotheses": [
            {
                "statement": f"{decision} is supported by the frozen train-only evidence",
                "supporting_evidence": [canonical_hash(evidence)],
                "contradicting_evidence": list(alternatives),
            }
        ],
        "falsification": {
            "next_check": "independent hash and ranking review",
            "predicted_result_if_hypothesis_is_wrong": "ranking or artifact hashes differ",
        },
        "root_cause": {
            "status": root_status,
            "statement": decision if root_status == "supported" else "unknown",
            "confidence": "high" if root_status == "supported" else "low",
        },
        "correction": {
            "containment": containment,
            "proposed_change": proposed_change,
            "changes_scientific_contract": False,
            "requires_new_protocol_version": False,
            "fresh_output_required": root_status != "supported",
        },
        "review": {
            "independent_reviewer": "",
            "decision": "pending",
            "evidence_hashes": [],
        },
        "decision": decision,
        "alternatives": list(alternatives),
        "version_bump_decision": "not-required" if root_status == "supported" else "pending",
    }
    payload["sha256"] = canonical_hash(payload)
    return payload


def _ridge_record(
    x: np.ndarray | sparse.spmatrix,
    y: np.ndarray,
    train: np.ndarray,
    tune: np.ndarray,
    *,
    alpha: float,
    seed: int,
    ledger: BudgetLedger,
) -> dict[str, object]:
    candidate_id = f"alpha-{alpha:g}"
    ledger.reserve("ridge", candidate_id, seed)
    started = time.perf_counter()
    mean, scale = y[train].mean(axis=0), y[train].std(axis=0)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4, max_iter=10_000).fit(
        x[train], (y[train] - mean) / scale
    )
    prediction = model.predict(x[tune]) * scale + mean
    score, per_target = normalized_mae(y[tune], prediction, y[train])
    ledger.complete(time.perf_counter() - started)
    return {
        "id": candidate_id,
        "seed": seed,
        "mean_mnmae": score,
        "per_target_nmae": dict(zip(TARGET_COLUMNS, per_target.tolist(), strict=True)),
        "selection_partition": "inner-tune-only",
    }


def _smoke_xgb_record(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    tune: np.ndarray,
    *,
    candidate: Mapping[str, Any],
    seed: int,
    ledger: BudgetLedger,
) -> dict[str, object]:
    ledger.reserve("xgboost", str(candidate["id"]), seed)
    started = time.perf_counter()
    model = ExtraTreesRegressor(
        n_estimators=12,
        max_depth=min(int(candidate["max_depth"]), 5),
        random_state=seed,
        n_jobs=1,
    ).fit(x[train], y[train])
    prediction = model.predict(x[tune])
    score, per_target = normalized_mae(y[tune], prediction, y[train])
    ledger.complete(time.perf_counter() - started)
    return {
        "id": str(candidate["id"]),
        "seed": seed,
        "mean_mnmae": score,
        "per_target_nmae": dict(zip(TARGET_COLUMNS, per_target.tolist(), strict=True)),
        "selection_partition": "inner-tune-only",
        "backend": "smoke-extra-trees-surrogate",
    }


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["id"]), []).append(record)
    return rank_candidates(
        [
            {
                "id": candidate_id,
                "seed_scores": [float(row["mean_mnmae"]) for row in rows],
                "mean_mnmae": float(np.mean([row["mean_mnmae"] for row in rows])),
                "sample_std": float(np.std([row["mean_mnmae"] for row in rows], ddof=1))
                if len(rows) > 1
                else 0.0,
                "selection_partition": "inner-tune-only",
            }
            for candidate_id, rows in grouped.items()
        ]
    )


def _legacy_run_real_selection_wave(
    config: Mapping[str, Any],
    matrix: sparse.csr_matrix,
    fixed_train_targets: np.ndarray,
    identity_groups: Sequence[str],
    scaffold_groups: Sequence[str],
    output: Path,
    *,
    wave: str,
    input_identity: Mapping[str, Any],
) -> dict[str, object]:
    """Run an authenticated train-only wave; outer target arrays are not accepted."""
    validate_config(config)
    if wave not in {"scaler-ablation", "feature-screen", "full-selection"}:
        raise ModelSelectionError("unknown selection wave")
    if output.exists() and any(output.iterdir()):
        raise ModelSelectionError("selection output must be new and empty")
    output.mkdir(parents=True, exist_ok=True)
    y = np.asarray(fixed_train_targets, dtype=np.float64)
    if (
        matrix.shape[0] != len(y)
        or len(y) != len(identity_groups)
        or len(y) != len(scaffold_groups)
    ):
        raise ModelSelectionError("train-only matrix/targets/groups are misaligned")
    indices = np.arange(len(y), dtype=np.int64)
    identity_folds = build_group_folds(
        indices,
        identity_groups,
        int(config["identity_folds"]),
        int(config["training_seeds"][0]),
        strategy="identity",
    )
    scaffold_folds = build_group_folds(
        indices,
        scaffold_groups,
        int(config["scaffold_stress_folds"]),
        int(config["training_seeds"][0]),
        strategy="scaffold",
    )
    budget = BudgetLedger(
        int(config["budget"]["max_candidate_fits"]),
        float(config["budget"]["max_wall_seconds"]),
    )
    records: dict[str, list[dict[str, object]]] = {
        "features": [],
        "ridge": [],
        "xgboost": [],
        "mlp": [],
        "scaffold_stress": [],
    }
    recipes = (
        ["descriptor_scaled_only", "all_sparse_scaled"]
        if wave == "scaler-ablation"
        else list(config["feature_selection"]["recipes"])
    )
    # Every fold receives the raw full matrix. Recipe statistics are fitted on
    # that fold's train indices and applied to its tune indices only.
    for recipe in recipes:
        fold_scores = []
        for fold in identity_folds:
            train = np.asarray(fold["train_source_indices"], dtype=np.int64)
            tune = np.asarray(fold["tune_source_indices"], dtype=np.int64)
            x_train, x_tune, provenance = fit_feature_recipe(recipe, matrix, train, tune)
            if wave == "scaler-ablation":
                candidate = config["mlp"]["candidates"][1]
                budget.reserve("mlp-scaler-ablation", recipe, int(fold["fold"]))
                started = time.perf_counter()
                prediction, curve = fit_real_torch_mlp(
                    x_train,
                    y[train],
                    x_tune,
                    y[tune],
                    candidate,
                    config["mlp"],
                    config["monitoring"],
                    seed=int(config["training_seeds"][int(fold["fold"])]),
                )
                budget.complete(time.perf_counter() - started)
                score, per_target = normalized_mae(y[tune], prediction, y[train])
                records["mlp"].append(
                    {
                        "id": recipe,
                        "fold": int(fold["fold"]),
                        "mean_mnmae": score,
                        "per_target_nmae": per_target.tolist(),
                        "curve": curve,
                        "recipe_provenance": provenance,
                        "selection_partition": "inner-tune-only",
                    }
                )
            else:
                best = float("inf")
                for alpha in config["ridge"]["alphas"]:
                    budget.reserve(
                        "ridge-feature-screen", f"{recipe}:alpha-{alpha}", int(fold["fold"])
                    )
                    started = time.perf_counter()
                    model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-4, max_iter=10_000)
                    mean, scale = y[train].mean(axis=0), y[train].std(axis=0)
                    model.fit(x_train, (y[train] - mean) / scale)
                    prediction = model.predict(x_tune) * scale + mean
                    score, per_target = normalized_mae(y[tune], prediction, y[train])
                    budget.complete(time.perf_counter() - started)
                    records["ridge"].append(
                        {
                            "id": f"{recipe}:alpha-{alpha}",
                            "recipe": recipe,
                            "alpha": float(alpha),
                            "fold": int(fold["fold"]),
                            "mean_mnmae": score,
                            "per_target_nmae": per_target.tolist(),
                            "recipe_provenance": provenance,
                            "selection_partition": "inner-tune-only",
                            "status": "completed",
                        }
                    )
                    best = min(best, score)
                fold_scores.append(best)
        recipe_values = (
            [row["mean_mnmae"] for row in records["mlp"] if row["id"] == recipe]
            if wave == "scaler-ablation"
            else fold_scores
        )
        records["features"].append(
            {
                "id": recipe,
                "mean_mnmae": float(np.mean(recipe_values or fold_scores)),
                "fold_scores": [float(value) for value in (recipe_values or fold_scores)],
                "selection_partition": "inner-tune-only",
            }
        )
    feature_ranking = rank_candidates(records["features"])
    selected_recipe = str(feature_ranking[0]["id"])
    if wave == "full-selection":
        for fold in identity_folds:
            train = np.asarray(fold["train_source_indices"], dtype=np.int64)
            tune = np.asarray(fold["tune_source_indices"], dtype=np.int64)
            x_train, x_tune, provenance = fit_feature_recipe(selected_recipe, matrix, train, tune)
            seed = int(config["training_seeds"][int(fold["fold"])])
            for candidate in config["xgboost"]["candidates"]:
                budget.reserve("xgboost", str(candidate["id"]), seed)
                started = time.perf_counter()
                prediction, rounds = fit_real_xgboost(
                    x_train,
                    y[train],
                    x_tune,
                    y[tune],
                    candidate,
                    seed=seed,
                    early_stopping_rounds=int(config["xgboost"]["early_stopping_rounds"]),
                )
                budget.complete(time.perf_counter() - started)
                score, per_target = normalized_mae(y[tune], prediction, y[train])
                records["xgboost"].append(
                    {
                        "id": str(candidate["id"]),
                        "fold": int(fold["fold"]),
                        "mean_mnmae": score,
                        "per_target_nmae": per_target.tolist(),
                        "per_target_rounds": rounds,
                        "recipe_provenance": provenance,
                        "selection_partition": "inner-tune-only",
                    }
                )
            for candidate in config["mlp"]["candidates"]:
                budget.reserve("mlp", str(candidate["id"]), seed)
                started = time.perf_counter()
                prediction, curve = fit_real_torch_mlp(
                    x_train,
                    y[train],
                    x_tune,
                    y[tune],
                    candidate,
                    config["mlp"],
                    config["monitoring"],
                    seed=seed,
                )
                budget.complete(time.perf_counter() - started)
                score, per_target = normalized_mae(y[tune], prediction, y[train])
                records["mlp"].append(
                    {
                        "id": str(candidate["id"]),
                        "fold": int(fold["fold"]),
                        "mean_mnmae": score,
                        "per_target_nmae": per_target.tolist(),
                        "curve": curve,
                        "recipe_provenance": provenance,
                        "selection_partition": "inner-tune-only",
                    }
                )
    # Scaffold folds are stress-only and never enter candidate ranking.
    for fold in scaffold_folds:
        train = np.asarray(fold["train_source_indices"], dtype=np.int64)
        tune = np.asarray(fold["tune_source_indices"], dtype=np.int64)
        x_train, x_tune, provenance = fit_feature_recipe(selected_recipe, matrix, train, tune)
        model = Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=10_000)
        mean, scale = y[train].mean(axis=0), y[train].std(axis=0)
        model.fit(x_train, (y[train] - mean) / scale)
        score, _ = normalized_mae(y[tune], model.predict(x_tune) * scale + mean, y[train])
        records["scaffold_stress"].append(
            {
                "fold": int(fold["fold"]),
                "mean_mnmae": score,
                "recipe_provenance": provenance,
                "used_for_ranking": False,
            }
        )
    rankings: dict[str, list[dict[str, object]]] = {"features": feature_ranking}
    if records["ridge"]:
        rankings["ridge"] = aggregate_records(records["ridge"])
    if records["xgboost"]:
        rankings["xgboost"] = aggregate_records(records["xgboost"])
    if records["mlp"]:
        rankings["mlp"] = aggregate_records(records["mlp"])
    why = [
        recursive_why(
            "why-feature",
            "Which fold-local recipe leads identity-group CV?",
            {"ranking": feature_ranking},
            selected_recipe,
            [str(row["id"]) for row in feature_ranking[1:]],
        )
    ]
    for family in ("ridge", "xgboost", "mlp"):
        if family in rankings:
            why.append(
                recursive_why(
                    f"why-{family}",
                    f"Which {family} candidate leads identity-group CV?",
                    {"ranking": rankings[family]},
                    str(rankings[family][0]["id"]),
                    [str(row["id"]) for row in rankings[family][1:]],
                    parent_id="why-feature",
                )
            )
    artifacts: dict[str, object] = {
        "feature_audit.json": {
            "input_identity": dict(input_identity),
            "identity_fold_count": len(identity_folds),
            "scaffold_stress_fold_count": len(scaffold_folds),
        },
        "candidate_records/features.json": records["features"],
        "candidate_records/ridge.json": records["ridge"],
        "candidate_records/xgboost.json": records["xgboost"],
        "candidate_records/mlp.json": records["mlp"],
        "candidate_records/scaffold_stress.json": records["scaffold_stress"],
        "budget_ledger.json": budget.payload(),
        "recursive_why/index.json": why,
    }
    artifact_records = {}
    for relative, payload in artifacts.items():
        path = output / relative
        atomic_json(path, payload)
        artifact_records[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    scientific = wave == "full-selection" and bool(records["xgboost"] and records["mlp"])
    lock = {
        "schema_version": LOCK_SCHEMA,
        "wave": wave,
        "run_mode": "authenticated-train-only",
        "scientific_result": scientific,
        "input_identity": dict(input_identity),
        "input_identity_sha256": canonical_hash(input_identity),
        "selected": {
            "feature_recipe": selected_recipe,
            **{family: ranking[0] for family, ranking in rankings.items() if family != "features"},
        },
        "outer_validation_targets_read": False,
        "test_targets_read": False,
        "artifact_sha256": artifact_records,
    }
    lock["lock_sha256"] = canonical_hash(lock)
    lock_path = output / "selection_lock.json"
    atomic_json(lock_path, lock)
    sidecar = output / "selection_lock.sha256"
    descriptor, temporary = tempfile.mkstemp(prefix=".selection_lock.sha256.", dir=output)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{file_sha256(lock_path)}  selection_lock.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, sidecar)
    manifest = {
        "schema_version": "qm9-model-selection-v3-manifest-v1",
        "wave": wave,
        "run_mode": "authenticated-train-only",
        "scientific_result": scientific,
        "selection_lock_sha256": file_sha256(lock_path),
        "complete": True,
        "outer_validation_targets_read": False,
        "test_targets_read": False,
    }
    atomic_json(output / "selection_manifest.json", manifest)
    return manifest


def _aggregate_fold_seed_records(
    records: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    *,
    expected_folds: int,
    expected_seeds: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["id"]), []).append(record)
    aggregates: list[dict[str, object]] = []
    for candidate_id, rows in grouped.items():
        expected_repeats = expected_folds * expected_seeds
        if len(rows) != expected_repeats:
            raise ModelSelectionError(f"candidate {candidate_id} has incomplete fold×seed repeats")
        pairs = {(int(row["fold"]), int(row["seed"])) for row in rows}
        if len(pairs) != expected_repeats:
            raise ModelSelectionError(f"candidate {candidate_id} repeats a fold×seed pair")
        fold_scores: list[float] = []
        fold_per_target: list[list[float]] = []
        for fold_id in sorted({int(row["fold"]) for row in rows}):
            fold_rows = [row for row in rows if int(row["fold"]) == fold_id]
            if len(fold_rows) != expected_seeds:
                raise ModelSelectionError(f"candidate {candidate_id} has incomplete seed repeats")
            tune = np.asarray(fold_rows[0]["tune_local_indices"], dtype=np.int64)
            source_order = fold_rows[0]["tune_source_indices"]
            if any(
                row["tune_local_indices"] != tune.tolist()
                or row["tune_source_indices"] != source_order
                for row in fold_rows
            ):
                raise ModelSelectionError("seed predictions have different tune row order")
            prediction = np.mean(
                [np.asarray(row["prediction"], dtype=float) for row in fold_rows], axis=0
            )
            scale = np.asarray(fold_rows[0]["fold_target_scale"], dtype=float)
            per_target = np.mean(np.abs(prediction - targets[tune]), axis=0) / scale
            fold_per_target.append(per_target.tolist())
            fold_scores.append(float(per_target.mean()))
        if len(fold_scores) != expected_folds:
            raise ModelSelectionError(f"candidate {candidate_id} has incomplete folds")
        scores = np.asarray(fold_scores, dtype=float)
        seed_scores = {
            str(seed): float(np.mean([row["mean_mnmae"] for row in rows if row["seed"] == seed]))
            for seed in sorted({int(row["seed"]) for row in rows})
        }
        aggregates.append(
            {
                "id": candidate_id,
                "family": rows[0]["family"],
                "recipe": rows[0]["recipe"],
                "params": dict(rows[0]["params"]),
                "complexity_tier": int(rows[0]["complexity_tier"]),
                "runtime_tier": int(rows[0]["runtime_tier"]),
                "mean_mnmae": float(scores.mean()),
                "sample_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "standard_error": float(scores.std(ddof=1) / np.sqrt(len(scores)))
                if len(scores) > 1
                else 0.0,
                "repeat_count": expected_repeats,
                "fold_mnmae": fold_scores,
                "fold_per_target_nmae": fold_per_target,
                "seed_stability": {
                    "per_seed_mean_mnmae": seed_scores,
                    "sample_std": float(np.std(list(seed_scores.values()), ddof=1))
                    if len(seed_scores) > 1
                    else 0.0,
                    "range": float(np.ptp(list(seed_scores.values()))),
                },
                "selection_partition": "inner-tune-only",
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            row["mean_mnmae"],
            row["sample_std"],
            row["complexity_tier"],
            row["runtime_tier"],
            row["id"],
        ),
    )


def _promote_primary(
    ranking: Sequence[Mapping[str, Any]], *, count: int, tie_tolerance: float
) -> list[dict[str, object]]:
    if not ranking or count < 1 or tie_tolerance < 0:
        raise ModelSelectionError("invalid primary promotion contract")
    best = ranking[0]
    boundary = float(best["mean_mnmae"]) + float(best["standard_error"])
    one_se = [dict(row) for row in ranking if float(row["mean_mnmae"]) <= boundary]
    outside = [dict(row) for row in ranking if float(row["mean_mnmae"]) > boundary]
    promoted = (one_se + outside)[:count]
    for row in promoted:
        row["within_one_standard_error"] = float(row["mean_mnmae"]) <= boundary
        row["one_standard_error_boundary"] = boundary
    return promoted


def _rank_with_scaffold(
    promoted: Sequence[Mapping[str, Any]],
    scaffold: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float,
) -> list[dict[str, object]]:
    stress = {str(row["id"]): row for row in scaffold}
    if any(str(row["id"]) not in stress for row in promoted):
        raise ModelSelectionError("promoted candidate lacks scaffold stress evidence")
    eligible = [dict(row) for row in promoted if row["within_one_standard_error"]]
    if not eligible:
        eligible = [dict(promoted[0])]
    for row in eligible:
        row["scaffold_mean_mnmae"] = float(stress[str(row["id"])]["mean_mnmae"])
        row["scaffold_sample_std"] = float(stress[str(row["id"])]["sample_std"])
    best_scaffold = min(float(row["scaffold_mean_mnmae"]) for row in eligible)
    eligible = [
        row
        for row in eligible
        if float(row["scaffold_mean_mnmae"]) <= best_scaffold + tie_tolerance
    ]
    best_dispersion = min(float(row["sample_std"]) for row in eligible)
    eligible = [
        row for row in eligible if float(row["sample_std"]) <= best_dispersion + tie_tolerance
    ]
    return sorted(
        eligible,
        key=lambda row: (
            row["complexity_tier"],
            row["runtime_tier"],
            row["id"],
        ),
    )


def _pipeline_registry(
    config: Mapping[str, Any], wave: str, prior: Mapping[str, Any] | None
) -> list[dict[str, object]]:
    if wave == "scaler-ablation":
        ridge = [
            {
                "id": f"ridge|{recipe}|alpha-{float(alpha):g}",
                "family": "ridge",
                "recipe": recipe,
                "params": {"alpha": float(alpha)},
                "complexity_tier": int(config["ridge"]["complexity_tiers"][position]),
                "runtime_tier": int(config["ridge"]["runtime_tiers"][position]),
            }
            for recipe in config["feature_selection"]["recipes"]
            for position, alpha in enumerate(config["ridge"]["alphas"])
        ]
        xgboost = [
            {
                "id": f"xgboost|{recipe}|{candidate['id']}",
                "family": "xgboost",
                "recipe": recipe,
                "params": {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"complexity_tier", "runtime_tier"}
                }
                | {
                    "n_estimators": min(
                        int(candidate["n_estimators"]),
                        int(config["waves"]["scaler_ablation"]["xgboost_n_estimators_cap"]),
                    )
                },
                "complexity_tier": int(candidate["complexity_tier"]),
                "runtime_tier": int(candidate["runtime_tier"]),
            }
            for recipe in config["feature_selection"]["recipes"]
            for candidate in config["xgboost"]["candidates"]
        ]
        mlp = [
            {
                "id": f"mlp|{recipe}|{candidate['id']}",
                "family": "mlp",
                "recipe": recipe,
                "params": {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"complexity_tier", "runtime_tier"}
                },
                "complexity_tier": int(candidate["complexity_tier"]),
                "runtime_tier": int(candidate["runtime_tier"]),
                "training_overrides": {
                    "max_epochs": int(config["waves"]["scaler_ablation"]["mlp_epoch_cap"])
                },
            }
            for recipe in ("descriptor_scaled_only", "all_sparse_scaled")
            for candidate in config["mlp"]["candidates"]
        ]
        return ridge + xgboost + mlp
    if prior is None:
        raise ModelSelectionError(f"{wave} requires prior promotions")
    registry = [dict(row) for row in prior["promoted"]]
    expected_cap = int(config["waves"][wave.replace("-", "_")]["candidate_cap"])
    if not registry or len(registry) > expected_cap:
        raise ModelSelectionError("prior promotion count exceeds this wave candidate cap")
    for row in registry:
        family = str(row["family"])
        if family in {"xgboost", "mlp"}:
            candidate_id = str(row["id"]).split("|")[-1]
            source = next(
                candidate
                for candidate in config[family]["candidates"]
                if candidate["id"] == candidate_id
            )
            row["params"] = {
                key: value
                for key, value in source.items()
                if key not in {"complexity_tier", "runtime_tier"}
            }
            row.pop("training_overrides", None)
    return registry


def _fit_pipeline(
    candidate: Mapping[str, Any],
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_tune: sparse.csr_matrix,
    y_tune: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    backend_runner: Callable[..., tuple[np.ndarray, Mapping[str, Any]]] | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if backend_runner is not None:
        prediction, metadata = backend_runner(
            candidate, x_train, y_train, x_tune, y_tune, config, seed
        )
        return np.asarray(prediction, dtype=float), dict(metadata)
    family = candidate["family"]
    if family == "ridge":
        mean, scale = y_train.mean(axis=0), y_train.std(axis=0)
        model = Ridge(
            alpha=float(candidate["params"]["alpha"]),
            solver="lsqr",
            tol=1e-4,
            max_iter=10_000,
        ).fit(x_train, (y_train - mean) / scale)
        return model.predict(x_tune) * scale + mean, {"backend": "sklearn-ridge"}
    if family == "xgboost":
        prediction, rounds = fit_real_xgboost(
            x_train,
            y_train,
            x_tune,
            y_tune,
            candidate["params"],
            seed=seed,
            early_stopping_rounds=int(config["xgboost"]["early_stopping_rounds"]),
        )
        return prediction, {"backend": "xgboost", "per_target_rounds": rounds}
    if family == "mlp":
        training = dict(config["mlp"])
        training.update(candidate.get("training_overrides", {}))
        prediction, curve = fit_real_torch_mlp(
            x_train,
            y_train,
            x_tune,
            y_tune,
            candidate["params"],
            training,
            config["monitoring"],
            seed=seed,
        )
        return prediction, {"backend": "torch", "curve": curve}
    raise ModelSelectionError(f"unsupported pipeline family: {family}")


def _assert_prediction_contract(prediction: np.ndarray, tune_rows: int) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    expected = (int(tune_rows), len(TARGET_COLUMNS))
    if values.shape != expected:
        raise ModelSelectionError(
            f"backend prediction shape differs: observed={values.shape} expected={expected}"
        )
    if not np.all(np.isfinite(values)):
        raise ModelSelectionError("backend prediction contains nonfinite values")
    return values


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _solve_ensemble_weights(
    predictions: Sequence[np.ndarray],
    targets: np.ndarray,
    folds: Sequence[Mapping[str, Any]],
    *,
    excluded_fold: int | None,
    config: Mapping[str, Any],
) -> np.ndarray:
    members = len(predictions)
    included = [fold for fold in folds if int(fold["fold"]) != excluded_fold]

    def objective(weights: np.ndarray) -> float:
        values = []
        for fold in included:
            train = np.asarray(fold["train_source_indices"], dtype=np.int64)
            tune = np.asarray(fold["tune_source_indices"], dtype=np.int64)
            blended = sum(weights[index] * predictions[index][tune] for index in range(members))
            scale = targets[train].std(axis=0)
            values.append(float((np.mean(np.abs(blended - targets[tune]), axis=0) / scale).mean()))
        return float(np.mean(values))

    starts = [np.eye(members)[index] for index in range(members)]
    starts.append(np.full(members, 1.0 / members))
    solutions: list[tuple[float, tuple[float, ...], np.ndarray]] = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * members,
            constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
            options={
                "ftol": float(config["ensemble"]["ftol"]),
                "maxiter": int(config["ensemble"]["maxiter"]),
            },
        )
        weights = np.asarray(result.x, dtype=float)
        if (
            not result.success
            or not np.all(np.isfinite(weights))
            or weights.min() < -1e-10
            or abs(weights.sum() - 1.0) > 1e-10
        ):
            continue
        weights[(weights < 0) & (weights >= -1e-10)] = 0
        weights /= weights.sum()
        value = objective(weights)
        solutions.append((value, tuple(np.round(weights, 12)), weights))
    if not solutions:
        raise ModelSelectionError("ensemble SLSQP produced no feasible solution")
    best_value = min(value for value, _, _ in solutions)
    tied = [row for row in solutions if row[0] <= best_value + 1e-12]
    return min(tied, key=lambda row: row[1])[2]


def _crossfit_ensemble(
    predictions: Sequence[np.ndarray],
    targets: np.ndarray,
    folds: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[list[float]]]:
    result = np.full_like(targets, np.nan, dtype=np.float64)
    weights: list[list[float]] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        fitted = _solve_ensemble_weights(
            predictions, targets, folds, excluded_fold=fold_id, config=config
        )
        tune = np.asarray(fold["tune_source_indices"], dtype=np.int64)
        result[tune] = sum(
            fitted[index] * predictions[index][tune] for index in range(len(predictions))
        )
        weights.append(fitted.tolist())
    if not np.all(np.isfinite(result)):
        raise ModelSelectionError("cross-fitted ensemble OOF is incomplete")
    return result, weights


def _oof_semantic_hash(
    values: np.ndarray,
    *,
    candidate_id: str,
    strategy: str,
    seeds: Sequence[int],
    source_indices: np.ndarray,
    folds: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    header = {
        "schema_version": config["oof"]["semantic_hash_schema"],
        "candidate_id": candidate_id,
        "strategy": strategy,
        "ordered_training_seeds": list(seeds),
        "shape": list(array.shape),
        "dtype": "float64",
        "target_order": list(TARGET_COLUMNS),
        "fixed_train_ordered_row_sha256": canonical_hash(source_indices.tolist()),
        "fold_assignment_sha256": canonical_hash(
            [
                {
                    "fold": fold["fold"],
                    "tune_source_indices": source_indices[
                        np.asarray(fold["tune_source_indices"], dtype=np.int64)
                    ].tolist(),
                }
                for fold in folds
            ]
        ),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _persist_failure(
    output: Path,
    config: Mapping[str, Any],
    wave: str,
    error: BaseException,
    ledger: BudgetLedger | None,
    input_identity: Mapping[str, Any],
    prior_sha256: str | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    why = recursive_why(
        "WHY-FAILURE-001",
        f"Train-only {wave} failed closed: {type(error).__name__}: {error}",
        {"config_sha256": canonical_hash(config), "prior_lock_sha256": prior_sha256},
        "unknown",
        [],
        root_status="unknown",
        containment="fail closed; preserve this output; do not retry in place",
        proposed_change="independent review required before a fresh output",
    )
    atomic_json(output / "recursive_why/failure.json", why)
    atomic_json(output / "recursive_why/index.json", [why])
    if ledger is not None:
        atomic_json(output / "budget_ledger.json", ledger.payload())
    artifacts = {}
    for relative in (
        "recursive_why/failure.json",
        "recursive_why/index.json",
        "budget_ledger.json",
    ):
        path = output / relative
        if path.is_file():
            artifacts[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    lock = {
        "schema_version": LOCK_SCHEMA,
        "wave": wave,
        "run_mode": "authenticated-train-only-failed",
        "scientific_result": False,
        "publication_ready": False,
        "config_sha256": canonical_hash(config),
        "prior_selection_lock_sha256": prior_sha256,
        "input_identity": dict(input_identity),
        "input_identity_sha256": canonical_hash(input_identity),
        "selected": {},
        "promoted": [],
        "outer_validation_targets_read": False,
        "test_targets_read": False,
        "artifact_sha256": artifacts,
    }
    lock["lock_sha256"] = canonical_hash(lock)
    lock_path = output / "selection_lock.json"
    atomic_json(lock_path, lock)
    descriptor, temporary = tempfile.mkstemp(prefix=".selection_lock.sha256.", dir=output)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{file_sha256(lock_path)}  selection_lock.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / "selection_lock.sha256")
    atomic_json(
        output / "selection_manifest.json",
        {
            "schema_version": "qm9-model-selection-v3-manifest-v2",
            "wave": wave,
            "run_mode": "authenticated-train-only-failed",
            "scientific_result": False,
            "publication_ready": False,
            "complete": False,
            "selection_lock_sha256": file_sha256(lock_path),
            "outer_validation_targets_read": False,
            "test_targets_read": False,
        },
    )


def _validate_authenticated_input_identity(
    identity: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    required = {
        "source_csv_sha256",
        "source_row_identity_sha256",
        "feature_matrix_file_sha256",
        "feature_matrix_semantic_sha256",
        "feature_manifest_sha256",
        "scaffold_groups_file_sha256",
        "phase1_run_sha256",
        "fixed_split_ordered_sha256",
        "fixed_split_membership_sha256",
        "fixed_train_ordered_indices",
        "reviewed_config_sha256",
    }
    missing = sorted(required - set(identity))
    if missing:
        raise ModelSelectionError(f"authenticated input identity fields are missing: {missing}")
    if identity["reviewed_config_sha256"] != canonical_hash(config):
        raise ModelSelectionError("authenticated input config lineage differs")
    for digest_field in (
        "source_csv_sha256",
        "source_row_identity_sha256",
        "feature_matrix_file_sha256",
        "feature_matrix_semantic_sha256",
        "feature_manifest_sha256",
        "scaffold_groups_file_sha256",
        "phase1_run_sha256",
    ):
        value = identity[digest_field]
        if not isinstance(value, str) or len(value) != 64:
            raise ModelSelectionError(f"authenticated input digest is invalid: {digest_field}")


def _run_real_selection_wave_impl(
    config: Mapping[str, Any],
    matrix: sparse.csr_matrix,
    fixed_train_targets: np.ndarray,
    identity_groups: Sequence[str],
    scaffold_groups: Sequence[str],
    output: Path,
    *,
    wave: str,
    input_identity: Mapping[str, Any],
    fixed_train_source_indices: Sequence[int] | None = None,
    prior_selection_lock: Path | None = None,
    backend_runner: Callable[..., tuple[np.ndarray, Mapping[str, Any]]] | None = None,
) -> dict[str, object]:
    """Execute a hash-chained train-only wave with complete fold×seed evidence."""
    validate_config(config)
    if wave not in {"scaler-ablation", "feature-screen", "full-selection"}:
        raise ModelSelectionError("unknown selection wave")
    _validate_authenticated_input_identity(input_identity, config)
    expected_prior = {
        "scaler-ablation": None,
        "feature-screen": "scaler-ablation",
        "full-selection": "feature-screen",
    }[wave]
    if output.exists() and any(output.iterdir()):
        raise ModelSelectionError("selection output must be new and empty")
    prior: dict[str, Any] | None = None
    prior_sha256: str | None = None
    if expected_prior is None and prior_selection_lock is not None:
        raise ModelSelectionError("scaler-ablation cannot consume a prior selection lock")
    if expected_prior is not None:
        if prior_selection_lock is None:
            error = ModelSelectionError(f"{wave} requires --prior-selection-lock")
            _persist_failure(output, config, wave, error, None, input_identity, None)
            raise error
        try:
            prior = validate_selection_lock(prior_selection_lock)
            prior_sha256 = file_sha256(prior_selection_lock)
            if (
                prior.get("wave") != expected_prior
                or prior.get("config_sha256") != canonical_hash(config)
                or prior.get("scientific_result") is not False
                or prior.get("input_identity_sha256") != canonical_hash(input_identity)
                or prior.get("input_identity") != dict(input_identity)
            ):
                raise ModelSelectionError(
                    "prior selection lock is stale, input-mismatched, or from the wrong wave"
                )
        except BaseException as error:
            _persist_failure(output, config, wave, error, None, input_identity, prior_sha256)
            raise
    output.mkdir(parents=True, exist_ok=True)
    y = np.asarray(fixed_train_targets, dtype=np.float64)
    raw = sparse.csr_matrix(matrix)
    if (
        raw.shape[0] != len(y)
        or y.shape != (raw.shape[0], len(TARGET_COLUMNS))
        or len(identity_groups) != len(y)
        or len(scaffold_groups) != len(y)
    ):
        raise ModelSelectionError("train-only matrix/targets/groups are misaligned")
    indices = np.arange(len(y), dtype=np.int64)
    if fixed_train_source_indices is None:
        raise ModelSelectionError("fixed_train_source_indices are required")
    source_indices_raw = np.asarray(fixed_train_source_indices)
    if (
        source_indices_raw.ndim != 1
        or source_indices_raw.dtype.kind not in "iu"
        or len(source_indices_raw) != len(y)
    ):
        raise ModelSelectionError("fixed-train source indices are missing or misaligned")
    source_indices = source_indices_raw.astype(np.int64, copy=False)
    if len(np.unique(source_indices)) != len(source_indices):
        raise ModelSelectionError("fixed-train source indices contain duplicates")
    authenticated_indices = input_identity.get("fixed_train_ordered_indices")
    if authenticated_indices is None or list(authenticated_indices) != source_indices.tolist():
        raise ModelSelectionError("fixed-train source index order differs from authentication")
    fold_seed = int(config["fold_assignment_seed"])
    limit_key = wave.replace("-", "_")
    wave_config = config["waves"][limit_key]
    active_seeds = list(config["training_seeds"][: int(wave_config["training_seed_count"])])
    identity_folds = build_group_folds(
        indices,
        identity_groups,
        int(wave_config["identity_folds"]),
        fold_seed,
        strategy="identity",
    )
    scaffold_folds = (
        build_group_folds(
            indices,
            scaffold_groups,
            int(wave_config["scaffold_folds"]),
            fold_seed,
            strategy="scaffold",
        )
        if "scaffold_folds" in wave_config
        else []
    )
    wave_fit_ceiling = int(config["budget"]["wave_fit_limits"][limit_key])
    ledger = BudgetLedger(
        int(config["budget"]["max_candidate_fits"]),
        float(config["budget"]["max_wall_seconds"]),
    )
    registry = _pipeline_registry(config, wave, prior)
    if len(registry) > int(wave_config["candidate_cap"]) or (
        wave == "scaler-ablation" and len(registry) != int(wave_config["candidate_cap"])
    ):
        raise ModelSelectionError("candidate registry differs from the frozen wave cap")
    if wave == "full-selection" and len(registry) < 2:
        raise ModelSelectionError(
            "W3 has one finalist; single-finalist freeze is not implemented and fails closed"
        )
    maximum_planned_fits = len(registry) * len(identity_folds) * len(active_seeds)
    if wave != "scaler-ablation":
        maximum_planned_fits += len(registry) * len(scaffold_folds) * len(active_seeds)
    if maximum_planned_fits > wave_fit_ceiling or maximum_planned_fits > int(
        config["budget"]["max_candidate_fits"]
    ):
        raise ModelSelectionError("maximum preregistered trial plan exceeds its fit ceiling")
    records: list[dict[str, object]] = []
    stress_records: list[dict[str, object]] = []
    oof: dict[str, np.ndarray] = (
        {
            str(candidate["id"]): np.full(
                (len(active_seeds), len(y), len(TARGET_COLUMNS)),
                np.nan,
                dtype=np.float64,
            )
            for candidate in registry
        }
        if wave == "full-selection"
        else {}
    )
    scaffold_oof = (
        {
            str(candidate["id"]): np.full(
                (len(active_seeds), len(y), len(TARGET_COLUMNS)), np.nan, dtype=np.float64
            )
            for candidate in registry
        }
        if wave == "full-selection"
        else {}
    )
    try:
        for candidate in registry:
            for fold in identity_folds:
                train = np.asarray(fold["train_source_indices"], dtype=np.int64)
                tune = np.asarray(fold["tune_source_indices"], dtype=np.int64)
                x_train, x_tune, provenance = fit_feature_recipe(
                    str(candidate["recipe"]), raw, train, tune
                )
                for seed_position, seed in enumerate(active_seeds):
                    ledger.reserve(
                        str(candidate["family"]),
                        str(candidate["id"]),
                        int(seed),
                        wave=wave_config["protocol_wave"],
                        strategy="identity",
                        fold_id=int(fold["fold"]),
                    )
                    started = time.perf_counter()
                    try:
                        prediction, metadata = _fit_pipeline(
                            candidate,
                            x_train,
                            y[train],
                            x_tune,
                            y[tune],
                            config,
                            int(seed),
                            backend_runner,
                        )
                        prediction = _assert_prediction_contract(prediction, len(tune))
                        score, per_target = normalized_mae(y[tune], prediction, y[train])
                    except BaseException as error:
                        ledger.fail(time.perf_counter() - started, error)
                        raise
                    ledger.complete(time.perf_counter() - started)
                    if wave == "full-selection":
                        if np.any(np.isfinite(oof[str(candidate["id"])][seed_position, tune])):
                            raise ModelSelectionError("identity OOF row written more than once")
                        oof[str(candidate["id"])][seed_position, tune] = prediction
                    records.append(
                        {
                            "id": candidate["id"],
                            "family": candidate["family"],
                            "recipe": candidate["recipe"],
                            "params": candidate["params"],
                            "complexity_tier": candidate["complexity_tier"],
                            "runtime_tier": candidate["runtime_tier"],
                            "fold": int(fold["fold"]),
                            "seed": int(seed),
                            "mean_mnmae": score,
                            "per_target_nmae": per_target.tolist(),
                            "prediction": np.asarray(prediction, dtype=float).tolist(),
                            "prediction_sha256": canonical_hash(
                                np.asarray(prediction, dtype=float).tolist()
                            ),
                            "tune_local_indices": tune.tolist(),
                            "tune_source_indices": source_indices[tune].tolist(),
                            "tune_source_indices_sha256": canonical_hash(
                                source_indices[tune].tolist()
                            ),
                            "fold_target_scale": y[train].std(axis=0).tolist(),
                            "recipe_provenance": provenance,
                            "metadata": metadata,
                            "selection_partition": "inner-tune-only",
                            "status": "completed",
                        }
                    )
        aggregate = _aggregate_fold_seed_records(
            records,
            y,
            expected_folds=len(identity_folds),
            expected_seeds=len(active_seeds),
        )
        by_family: dict[str, list[dict[str, object]]] = {}
        for row in aggregate:
            by_family.setdefault(str(row["family"]), []).append(row)
        registry_by_id = {str(row["id"]): row for row in registry}
        if wave == "scaler-ablation":
            family_coverage = int(wave_config["family_coverage"])
            promoted_flat = []
            for family in ("ridge", "xgboost", "mlp"):
                promoted_flat.extend(by_family.get(family, [])[:family_coverage])
            for row in aggregate:
                if len(promoted_flat) >= int(wave_config["promotion_cap"]) or any(
                    existing["id"] == row["id"] for existing in promoted_flat
                ):
                    continue
                promoted_flat.append(row)
            stress_candidates: list[dict[str, object]] = []
        else:
            one_se = _promote_primary(
                aggregate,
                count=len(aggregate),
                tie_tolerance=float(config["ranking"]["tie_tolerance"]),
            )
            stress_candidates = (
                one_se
                if wave == "full-selection"
                else [row for row in one_se if row["within_one_standard_error"]]
            )
        planned_fits = len(registry) * len(identity_folds) * len(active_seeds) + len(
            stress_candidates
        ) * len(scaffold_folds) * len(active_seeds)
        if planned_fits > wave_fit_ceiling or planned_fits > int(
            config["budget"]["max_candidate_fits"]
        ):
            raise ModelSelectionError("precomputed wave trial plan exceeds its fit ceiling")
        for summary in stress_candidates:
            candidate = registry_by_id[str(summary["id"])]
            for fold in scaffold_folds:
                train = np.asarray(fold["train_source_indices"], dtype=np.int64)
                tune = np.asarray(fold["tune_source_indices"], dtype=np.int64)
                x_train, x_tune, provenance = fit_feature_recipe(
                    str(candidate["recipe"]), raw, train, tune
                )
                for seed_position, seed in enumerate(active_seeds):
                    ledger.reserve(
                        f"{candidate['family']}-scaffold-stress",
                        str(candidate["id"]),
                        int(seed),
                        wave=wave_config["protocol_wave"],
                        strategy="scaffold",
                        fold_id=int(fold["fold"]),
                    )
                    started = time.perf_counter()
                    try:
                        prediction, metadata = _fit_pipeline(
                            candidate,
                            x_train,
                            y[train],
                            x_tune,
                            y[tune],
                            config,
                            int(seed),
                            backend_runner,
                        )
                        prediction = _assert_prediction_contract(prediction, len(tune))
                        score, per_target = normalized_mae(y[tune], prediction, y[train])
                    except BaseException as error:
                        ledger.fail(time.perf_counter() - started, error)
                        raise
                    ledger.complete(time.perf_counter() - started)
                    if wave == "full-selection":
                        if np.any(
                            np.isfinite(scaffold_oof[str(candidate["id"])][seed_position, tune])
                        ):
                            raise ModelSelectionError("scaffold OOF row written more than once")
                        scaffold_oof[str(candidate["id"])][seed_position, tune] = prediction
                    stress_records.append(
                        {
                            "id": candidate["id"],
                            "family": candidate["family"],
                            "recipe": candidate["recipe"],
                            "params": candidate["params"],
                            "complexity_tier": candidate["complexity_tier"],
                            "runtime_tier": candidate["runtime_tier"],
                            "fold": int(fold["fold"]),
                            "seed": int(seed),
                            "mean_mnmae": score,
                            "per_target_nmae": per_target.tolist(),
                            "prediction": np.asarray(prediction, dtype=float).tolist(),
                            "prediction_sha256": canonical_hash(
                                np.asarray(prediction, dtype=float).tolist()
                            ),
                            "tune_local_indices": tune.tolist(),
                            "tune_source_indices": source_indices[tune].tolist(),
                            "tune_source_indices_sha256": canonical_hash(
                                source_indices[tune].tolist()
                            ),
                            "fold_target_scale": y[train].std(axis=0).tolist(),
                            "recipe_provenance": provenance,
                            "metadata": metadata,
                            "selection_partition": "inner-tune-only-scaffold-stress",
                            "used_for_primary_rank": False,
                            "used_for_one_se_tiebreak": True,
                            "claim_scope": "train-only-scaffold-stress",
                            "status": "completed",
                        }
                    )
        stress_aggregate = (
            _aggregate_fold_seed_records(
                [{**row, "selection_partition": "inner-tune-only"} for row in stress_records],
                y,
                expected_folds=len(scaffold_folds),
                expected_seeds=len(active_seeds),
            )
            if stress_records
            else []
        )
        if wave == "scaler-ablation":
            primary = aggregate[0]
        else:
            robustness_ranking = _rank_with_scaffold(
                stress_candidates,
                stress_aggregate,
                tie_tolerance=float(config["ranking"]["tie_tolerance"]),
            )
            promoted_flat = robustness_ranking[: int(wave_config["promotion_cap"])]
            primary = robustness_ranking[0]
        ensemble: dict[str, object] | None = None
        oof_artifacts: dict[str, dict[str, object]] = {}
        if wave == "full-selection":
            finalists = sorted(aggregate, key=lambda row: str(row["id"]))
            if not 2 <= len(finalists) <= 3:
                raise ModelSelectionError("W3 requires two or three frozen base finalists")
            identity_tensors = [oof[str(row["id"])] for row in finalists]
            scaffold_tensors = [scaffold_oof[str(row["id"])] for row in finalists]
            if any(
                not np.all(np.isfinite(array)) for array in [*identity_tensors, *scaffold_tensors]
            ):
                raise ModelSelectionError("W3 finalist OOF predictions are incomplete")
            identity_mean = [array.mean(axis=0) for array in identity_tensors]
            scaffold_mean = [array.mean(axis=0) for array in scaffold_tensors]
            identity_ensemble, identity_crossfit_weights = _crossfit_ensemble(
                identity_mean, y, identity_folds, config
            )
            scaffold_ensemble, scaffold_crossfit_weights = _crossfit_ensemble(
                scaffold_mean, y, scaffold_folds, config
            )
            deployment_weights = _solve_ensemble_weights(
                identity_mean, y, identity_folds, excluded_fold=None, config=config
            )

            def ensemble_summary(
                prediction: np.ndarray, folds: Sequence[Mapping[str, Any]]
            ) -> dict[str, object]:
                fold_scores = []
                for fold in folds:
                    train = np.asarray(fold["train_source_indices"], dtype=np.int64)
                    tune = np.asarray(fold["tune_source_indices"], dtype=np.int64)
                    scale = y[train].std(axis=0)
                    fold_scores.append(
                        float((np.mean(np.abs(prediction[tune] - y[tune]), axis=0) / scale).mean())
                    )
                scores = np.asarray(fold_scores)
                return {
                    "mean_mnmae": float(scores.mean()),
                    "sample_std": float(scores.std(ddof=1)),
                    "standard_error": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                    "fold_mnmae": fold_scores,
                }

            identity_ensemble_summary = ensemble_summary(identity_ensemble, identity_folds)
            scaffold_ensemble_summary = ensemble_summary(scaffold_ensemble, scaffold_folds)
            ensemble = {
                "id": "ensemble|traditional-oof-nonnegative-sum-one",
                "family": "ensemble",
                "recipe": "multiple-frozen-finalists",
                "params": {"solver": config["ensemble"]["solver"]},
                "complexity_tier": int(config["ensemble"]["complexity_tier"]),
                "runtime_tier": int(config["ensemble"]["runtime_tier"]),
                "members": [str(row["id"]) for row in finalists],
                "identity_crossfit_weights": identity_crossfit_weights,
                "scaffold_crossfit_weights": scaffold_crossfit_weights,
                "deployment_weights": deployment_weights.tolist(),
                "weight_constraints": "nonnegative-sum-to-one",
                "fit_partition": "fixed-train-OOF-only",
                **identity_ensemble_summary,
                "selection_partition": "inner-tune-only",
            }
            ensemble_stress = {
                **ensemble,
                **scaffold_ensemble_summary,
                "selection_partition": "inner-tune-only",
            }
            combined = [*aggregate, ensemble]
            combined_one_se = _promote_primary(
                combined,
                count=len(combined),
                tie_tolerance=float(config["ranking"]["tie_tolerance"]),
            )
            combined_stress = [*stress_aggregate, ensemble_stress]
            final_ranking = _rank_with_scaffold(
                combined_one_se,
                combined_stress,
                tie_tolerance=float(config["ranking"]["tie_tolerance"]),
            )
            primary = final_ranking[0]
            promoted_flat = [primary]
            selected_arrays: dict[str, np.ndarray] = {
                "fixed_train_source_row_indices": source_indices,
                "fixed_train_source_row_indices_sha256_ascii": np.frombuffer(
                    canonical_hash(source_indices.tolist()).encode("ascii"), dtype=np.uint8
                ),
            }
            semantic_hashes: dict[str, str] = {}
            for position, finalist in enumerate(finalists):
                candidate_id = str(finalist["id"])
                identity_name = f"candidate_{position}_identity_oof"
                scaffold_name = f"candidate_{position}_scaffold_oof"
                selected_arrays[identity_name] = identity_tensors[position]
                selected_arrays[scaffold_name] = scaffold_tensors[position]
                selected_arrays[f"candidate_{position}_identity_seed_mean"] = identity_mean[
                    position
                ]
                selected_arrays[f"candidate_{position}_scaffold_seed_mean"] = scaffold_mean[
                    position
                ]
                semantic_hashes[identity_name] = _oof_semantic_hash(
                    identity_tensors[position],
                    candidate_id=candidate_id,
                    strategy="identity",
                    seeds=active_seeds,
                    source_indices=source_indices,
                    folds=identity_folds,
                    config=config,
                )
                semantic_hashes[scaffold_name] = _oof_semantic_hash(
                    scaffold_tensors[position],
                    candidate_id=candidate_id,
                    strategy="scaffold",
                    seeds=active_seeds,
                    source_indices=source_indices,
                    folds=scaffold_folds,
                    config=config,
                )
                semantic_hashes[f"candidate_{position}_identity_seed_mean"] = _oof_semantic_hash(
                    identity_mean[position],
                    candidate_id=candidate_id,
                    strategy="identity-seed-mean",
                    seeds=active_seeds,
                    source_indices=source_indices,
                    folds=identity_folds,
                    config=config,
                )
                semantic_hashes[f"candidate_{position}_scaffold_seed_mean"] = _oof_semantic_hash(
                    scaffold_mean[position],
                    candidate_id=candidate_id,
                    strategy="scaffold-seed-mean",
                    seeds=active_seeds,
                    source_indices=source_indices,
                    folds=scaffold_folds,
                    config=config,
                )
            selected_arrays["ensemble_identity_crossfit_oof"] = identity_ensemble
            selected_arrays["ensemble_scaffold_crossfit_oof"] = scaffold_ensemble
            ensemble["identity_crossfit_prediction_sha256"] = _oof_semantic_hash(
                identity_ensemble,
                candidate_id=str(ensemble["id"]),
                strategy="identity-crossfit-ensemble",
                seeds=active_seeds,
                source_indices=source_indices,
                folds=identity_folds,
                config=config,
            )
            ensemble["scaffold_crossfit_prediction_sha256"] = _oof_semantic_hash(
                scaffold_ensemble,
                candidate_id=str(ensemble["id"]),
                strategy="scaffold-crossfit-ensemble",
                seeds=active_seeds,
                source_indices=source_indices,
                folds=scaffold_folds,
                config=config,
            )
            ensemble["deployment_weights_sha256"] = canonical_hash(deployment_weights.tolist())
            ensemble["weights_feasible"] = bool(
                np.all(deployment_weights >= 0)
                and abs(float(deployment_weights.sum()) - 1.0) <= 1e-10
            )
            path = output / "oof/finalists_and_ensemble.npz"
            _atomic_npz(path, selected_arrays)
            oof_artifacts[str(path.relative_to(output))] = {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "shape_contract": [len(active_seeds), len(y), 12],
                "dtype": "float64",
                "semantic_sha256": semantic_hashes,
            }
        if len(ledger.fits) != planned_fits or any(
            row["status"] != "completed" for row in ledger.fits
        ):
            raise ModelSelectionError(
                f"wave fit ledger differs: observed={len(ledger.fits)} expected={planned_fits}"
            )
        why = [
            recursive_why(
                f"WHY-{wave.upper()}-001",
                "Which pipeline survives the frozen train-only promotion rule?",
                {"primary": aggregate, "scaffold": stress_aggregate},
                str(primary["id"]),
                [str(row["id"]) for row in aggregate if row["id"] != primary["id"]],
            )
        ]
        artifacts: dict[str, object] = {
            "candidate_records/identity.json": records,
            "candidate_records/identity_aggregate.json": aggregate,
            "candidate_records/scaffold_stress.json": stress_records,
            "candidate_records/scaffold_aggregate.json": stress_aggregate,
            "candidate_records/promoted.json": promoted_flat,
            "recursive_why/index.json": why,
            "budget_ledger.json": ledger.payload(),
            "feature_audit.json": {
                "input_identity": dict(input_identity),
                "identity_folds": len(identity_folds),
                "scaffold_folds": len(scaffold_folds),
            },
        }
        if ensemble is not None:
            artifacts["ensemble.json"] = ensemble
        artifact_records: dict[str, dict[str, object]] = dict(oof_artifacts)
        for relative, payload in artifacts.items():
            path = output / relative
            atomic_json(path, payload)
            artifact_records[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        lock = {
            "schema_version": LOCK_SCHEMA,
            "wave": wave,
            "run_mode": "authenticated-train-only",
            "scientific_result": False,
            "publication_ready": False,
            "config_sha256": canonical_hash(config),
            "prior_selection_lock_sha256": prior_sha256,
            "input_identity": dict(input_identity),
            "input_identity_sha256": canonical_hash(input_identity),
            "selected": {
                "primary_pipeline": primary,
                "base_finalists": aggregate if wave == "full-selection" else promoted_flat,
                "ensemble": ensemble,
            },
            "promoted": promoted_flat,
            "outer_validation_targets_read": False,
            "test_targets_read": False,
            "artifact_sha256": artifact_records,
        }
        lock["lock_sha256"] = canonical_hash(lock)
        lock_path = output / "selection_lock.json"
        atomic_json(lock_path, lock)
        descriptor, temporary = tempfile.mkstemp(prefix=".selection_lock.sha256.", dir=output)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{file_sha256(lock_path)}  selection_lock.json\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output / "selection_lock.sha256")
        manifest = {
            "schema_version": "qm9-model-selection-v3-manifest-v2",
            "wave": wave,
            "run_mode": "authenticated-train-only",
            "scientific_result": False,
            "publication_ready": False,
            "complete": True,
            "selection_lock_sha256": file_sha256(lock_path),
            "outer_validation_targets_read": False,
            "test_targets_read": False,
        }
        atomic_json(output / "selection_manifest.json", manifest)
        return manifest
    except BaseException as error:
        _persist_failure(output, config, wave, error, ledger, input_identity, prior_sha256)
        raise


def run_real_selection_wave(
    config: Mapping[str, Any],
    matrix: sparse.csr_matrix,
    fixed_train_targets: np.ndarray,
    identity_groups: Sequence[str],
    scaffold_groups: Sequence[str],
    output: Path,
    *,
    wave: str,
    input_identity: Mapping[str, Any],
    fixed_train_source_indices: Sequence[int] | None = None,
    prior_selection_lock: Path | None = None,
    backend_runner: Callable[..., tuple[np.ndarray, Mapping[str, Any]]] | None = None,
) -> dict[str, object]:
    """Persist a complete failure state for every fresh-output real-wave exception."""
    fresh_output = not output.exists() or (output.is_dir() and not any(output.iterdir()))
    try:
        return _run_real_selection_wave_impl(
            config,
            matrix,
            fixed_train_targets,
            identity_groups,
            scaffold_groups,
            output,
            wave=wave,
            input_identity=input_identity,
            fixed_train_source_indices=fixed_train_source_indices,
            prior_selection_lock=prior_selection_lock,
            backend_runner=backend_runner,
        )
    except BaseException as error:
        if fresh_output and not (output / "selection_lock.json").is_file():
            try:
                _persist_failure(
                    output,
                    config,
                    wave,
                    error,
                    None,
                    input_identity,
                    file_sha256(prior_selection_lock)
                    if prior_selection_lock is not None and prior_selection_lock.is_file()
                    else None,
                )
            except BaseException as persistence_error:
                print(
                    "failed to create QM9 v3 failure evidence: "
                    f"{type(persistence_error).__name__}: {persistence_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise


def persist_real_preflight_failure(
    config: Mapping[str, Any],
    output: Path,
    wave: str,
    error: BaseException,
    input_identity: Mapping[str, Any],
    prior_selection_lock: Path | None = None,
) -> None:
    """Persist runner authentication/config failures before model arrays exist."""
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print(
            f"cannot persist preflight failure into nonempty/non-directory output: {output}",
            file=sys.stderr,
            flush=True,
        )
        return
    try:
        _persist_failure(
            output,
            config,
            wave,
            error,
            None,
            input_identity,
            file_sha256(prior_selection_lock)
            if prior_selection_lock is not None and prior_selection_lock.is_file()
            else None,
        )
    except BaseException as persistence_error:
        print(
            "failed to create QM9 v3 preflight failure evidence: "
            f"{type(persistence_error).__name__}: {persistence_error}",
            file=sys.stderr,
            flush=True,
        )


def run_smoke_selection(
    config: Mapping[str, Any],
    output: Path,
    *,
    supplied_features: np.ndarray | sparse.spmatrix | None = None,
    supplied_fixed_train_targets: np.ndarray | None = None,
    supplied_groups: Sequence[object] | None = None,
    supplied_feature_audit: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Exercise the complete train-only state machine with deterministic synthetic data."""
    validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise ModelSelectionError("selection output must be a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(303)
    if supplied_features is None:
        x = rng.normal(size=(120, 20))
        y = x @ rng.normal(size=(20, 12)) + rng.normal(scale=0.1, size=(120, 12))
        groups = np.repeat(np.arange(60), 2)
        feature_audit = {"shape": list(x.shape), "finite": True, "mode": "synthetic-smoke"}
        mode = "synthetic-smoke-v1"
    else:
        if supplied_fixed_train_targets is None or supplied_groups is None:
            raise ModelSelectionError("supplied features require fixed-train targets and groups")
        x = sparse.csr_matrix(supplied_features)
        y = np.asarray(supplied_fixed_train_targets, dtype=np.float64)
        groups = np.asarray(supplied_groups)
        if x.shape[0] != len(y) or len(y) != len(groups) or y.shape[1] != 12:
            raise ModelSelectionError("supplied train-only arrays are misaligned")
        feature_audit = dict(supplied_feature_audit or {})
        mode = "authenticated-fixed-train-smoke-v1"
    seeds = list(config["training_seeds"][:2])
    budget = config["budget"]
    ledger = BudgetLedger(
        int(budget["smoke_max_candidate_fits"]), float(budget["smoke_max_wall_seconds"])
    )
    ridge_records = []
    xgb_records = []
    split_hashes = {}
    for seed in seeds:
        train, tune = group_safe_inner_split(
            groups, tune_fraction=float(config["inner_tune_fraction"]), seed=int(seed)
        )
        split_hashes[str(seed)] = canonical_hash({"train": train.tolist(), "tune": tune.tolist()})
        for alpha in config["ridge"]["alphas"]:
            ridge_records.append(
                _ridge_record(x, y, train, tune, alpha=float(alpha), seed=int(seed), ledger=ledger)
            )
        for candidate in config["xgboost"]["candidates"][:2]:
            xgb_records.append(
                _smoke_xgb_record(
                    np.asarray(x.toarray() if sparse.issparse(x) else x),
                    y,
                    train,
                    tune,
                    candidate=candidate,
                    seed=int(seed),
                    ledger=ledger,
                )
            )
    # The smoke MLP record uses deterministic curves to validate monitoring and
    # the paired-ablation state machine without pretending to benchmark Torch.
    ablation = list(config["mlp"]["preprocessing_ablation"])
    left_contract = {**config["mlp"]["candidates"][1], **ablation[0]}
    right_contract = {**config["mlp"]["candidates"][1], **ablation[1]}
    assert_paired_ablation(left_contract, right_contract)
    checkpoint_hash = canonical_hash({"seed": seeds[0], "epoch": 5})
    mlp_monitor = monitor_curve(
        [1.0, 0.7, 0.5, 0.4, 0.35],
        [0.4, 0.3, 0.25, 0.24, 0.245],
        max_epochs=12,
        restored_hash=checkpoint_hash,
        best_hash=checkpoint_hash,
        thresholds=config["monitoring"],
        per_target_nmae=np.tile(np.linspace(0.2, 0.1, 5)[:, None], (1, 12)),
        gradient_norm=[2.0, 1.2, 0.9, 0.7, 0.6],
        parameter_norm=[10.0, 10.1, 10.2, 10.3, 10.4],
        gpu_memory_bytes=[None] * 5,
        process_rss_bytes=[int(process_peak_rss()["bytes"])] * 5,
    )
    mlp_records = {
        "preprocessing_ablation": {
            "left": left_contract,
            "right": right_contract,
            "only_variable": "scale_all_sparse",
        },
        "candidate": dict(config["mlp"]["candidates"][1]),
        "monitoring": mlp_monitor,
        "selection_partition": "inner-tune-only",
    }
    ridge_ranking = aggregate_records(ridge_records)
    xgb_ranking = aggregate_records(xgb_records)
    feature_records = [
        {
            "id": recipe,
            "mean_mnmae": 0.20 + position * 0.01,
            "selection_partition": "inner-tune-only",
            "backend": "synthetic-smoke",
        }
        for position, recipe in enumerate(config["feature_selection"]["recipes"])
    ]
    feature_ranking = rank_candidates(feature_records)
    selected = {
        "feature_recipe": feature_ranking[0]["id"],
        "ridge": ridge_ranking[0],
        "xgboost": xgb_ranking[0],
        "mlp": {
            "id": mlp_records["candidate"]["id"],
            "preprocessing": "paired-ablation-pending-real-selection",
        },
    }
    input_identity = {
        "mode": mode,
        "features_sha256": canonical_hash(
            np.asarray(x.toarray() if sparse.issparse(x) else x).tolist()
        ),
        "fixed_train_targets_sha256": canonical_hash(y.tolist()),
        "inner_split_sha256": split_hashes,
    }
    artifacts = {
        "feature_audit.json": feature_audit,
        "candidate_records/ridge.json": ridge_records,
        "candidate_records/xgboost.json": xgb_records,
        "candidate_records/mlp.json": mlp_records,
        "candidate_records/features.json": feature_records,
        "budget_ledger.json": ledger.payload(),
        "recursive_why/index.json": [
            recursive_why(
                "why-feature",
                "Which recipe leads on inner tune?",
                {"ranking": feature_ranking},
                str(feature_ranking[0]["id"]),
                [str(row["id"]) for row in feature_ranking[1:]],
            )
        ],
    }
    artifact_hashes = {}
    for relative, payload in artifacts.items():
        path = output / relative
        atomic_json(path, payload)
        artifact_hashes[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    lock = {
        "schema_version": LOCK_SCHEMA,
        "run_mode": "synthetic-smoke",
        "scientific_result": False,
        "publication_ready": False,
        "scientific_status": "post-specified-train-only-selection",
        "input_identity": input_identity,
        "selected": selected,
        "outer_validation_targets_read": False,
        "test_targets_read": False,
        "artifact_sha256": artifact_hashes,
    }
    lock["lock_sha256"] = canonical_hash(lock)
    lock_path = output / "selection_lock.json"
    atomic_json(lock_path, lock)
    descriptor, temporary = tempfile.mkstemp(prefix=".selection_lock.sha256.", dir=output)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{file_sha256(lock_path)}  selection_lock.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / "selection_lock.sha256")
    manifest = {
        "schema_version": "qm9-model-selection-v3-manifest-v1",
        "config_sha256": canonical_hash(config),
        "code_sha256": file_sha256(Path(__file__)),
        "selection_lock_sha256": file_sha256(lock_path),
        "complete": True,
        "outer_validation_targets_read": False,
        "test_targets_read": False,
        "process_peak_rss": process_peak_rss(),
        "run_mode": "synthetic-smoke",
        "scientific_result": False,
        "publication_ready": False,
    }
    atomic_json(output / "selection_manifest.json", manifest)
    return manifest


def validate_selection_lock(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LOCK_SCHEMA:
        raise ModelSelectionError("selection lock schema differs")
    claimed = payload.pop("lock_sha256", None)
    if claimed != canonical_hash(payload):
        raise ModelSelectionError("selection lock canonical hash differs")
    if (
        payload.get("outer_validation_targets_read") is not False
        or payload.get("test_targets_read") is not False
    ):
        raise ModelSelectionError("selection lock violates train-only boundary")
    for relative, record in payload.get("artifact_sha256", {}).items():
        artifact = path.parent / relative
        if (
            not artifact.is_file()
            or file_sha256(artifact) != record.get("sha256")
            or artifact.stat().st_size != record.get("bytes")
        ):
            raise ModelSelectionError(f"selection artifact changed: {relative}")
    return {**payload, "lock_sha256": claimed}
