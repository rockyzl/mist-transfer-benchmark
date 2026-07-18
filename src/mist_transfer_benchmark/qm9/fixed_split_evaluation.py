"""Repeated evaluation on the one fixed split used by the released QM9 MIST model.

Unlike :mod:`paper_evaluation`, this route never creates a new split.  It treats
MIST as a fixed inference artifact, repeats only the engineered comparators, and
closes one durable global selection gate before any test label is supplied.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import resource
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import TARGET_COLUMNS
from .phase2_metrics import native_metrics

CONFIG_SCHEMA = "qm9-fixed-mist-split-v2-config-v1"
SUMMARY_SCHEMA = "qm9-fixed-mist-split-v2-summary-v1"
MODEL_KEYS = (
    "engineered_ridge",
    "xgboost",
    "mlp",
    "traditional_ensemble",
    "all_model_ensemble",
)
EXPECTED_SEEDS = [20260713, 20260729, 20260811, 20260823, 20260907]
EXPECTED_FROZEN = {
    "feature_representation": "count_ecfp_plus_globals",
    "ridge_alpha": 10.0,
    "xgboost_max_depth": 10,
    "xgboost_learning_rate": 0.03,
    "xgboost_n_estimators": 1100,
    "xgboost_subsample": 0.85,
    "xgboost_colsample_bytree": 0.9,
    "xgboost_min_child_weight": 1.0,
    "xgboost_early_stopping_rounds": 35,
    "mlp_hidden_dims": [768, 384],
    "mlp_dropout": 0.02,
    "mlp_learning_rate": 0.0007,
    "mlp_weight_decay": 0.00001,
    "mlp_batch_size": 512,
    "mlp_max_epochs": 80,
    "mlp_patience": 10,
    "mlp_min_delta": 0.0001,
}
EXPECTED_MONITORING = {
    "validation_increase_mark_after": 2,
    "loss_nonfinite_is_abnormal": True,
    "restore_best_epoch": True,
}
CRITICAL_REVIEW_PLAN = (
    {
        "id": "input-boundary",
        "plan": "Verify immutable inputs, row order, split membership, and no test-label access.",
        "review": "Automated identity and leakage-boundary checks must pass before fitting.",
    },
    {
        "id": "selection-freeze",
        "plan": "Train all five seeds and freeze validation-only model and ensemble decisions.",
        "review": (
            "Review seed artifacts, loss anomalies, scalers, and prediction hashes before "
            "test access."
        ),
    },
    {
        "id": "test-unlock",
        "plan": "Authorize exactly one test-label read from the verified global freeze hash.",
        "review": (
            "Verify the global gate and event trail immediately before unlocking test labels."
        ),
    },
    {
        "id": "publication",
        "plan": "Generate the predetermined metrics, uncertainty, runtime, and subgroup artifacts.",
        "review": (
            "Independent scientific review is required before any headline or article update."
        ),
    },
)


def critical_review_plan(
    input_review_evidence: object | None = None,
) -> list[dict[str, object]]:
    """Return fresh, machine-readable Plan -> Execute -> Review checkpoints."""

    reviews = [
        {**step, "status": "planned", "evidence": None} for step in CRITICAL_REVIEW_PLAN
    ]
    if input_review_evidence is not None:
        _mark_review(
            reviews,
            "input-boundary",
            "automated-review-passed",
            input_review_evidence,
        )
    return reviews


def _mark_review(
    reviews: list[dict[str, object]], review_id: str, status: str, evidence: object
) -> None:
    for review in reviews:
        if review["id"] == review_id:
            review["status"] = status
            review["evidence"] = evidence
            return
    raise FixedSplitEvaluationError(f"unknown critical review checkpoint: {review_id}")


class FixedSplitEvaluationError(ValueError):
    """Raised when a fixed-split identity or leakage boundary is violated."""


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _peak_rss() -> dict[str, object]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "bytes": raw * (1 if os.uname().sysname == "Darwin" else 1024),
        "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "semantics": "cumulative-process-high-water-mark-not-model-isolated",
    }


def _atomic_npy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.save(temporary, np.asarray(value, dtype=np.float64), allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return file_sha256(path)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise FixedSplitEvaluationError(f"config schema must be {CONFIG_SCHEMA}")
    seeds = list(config.get("seeds", []))
    if seeds != EXPECTED_SEEDS:
        raise FixedSplitEvaluationError("the five paper seeds are frozen")
    if list(config.get("target_order", [])) != list(TARGET_COLUMNS):
        raise FixedSplitEvaluationError("target order differs from the fixed QM9 contract")
    if dict(config.get("frozen_winners", {})) != EXPECTED_FROZEN:
        raise FixedSplitEvaluationError("frozen v1 winner/training contract differs")
    if dict(config.get("monitoring", {})) != EXPECTED_MONITORING:
        raise FixedSplitEvaluationError("monitoring contract differs")
    if int(config.get("bootstrap_samples", 0)) != 2000:
        raise FixedSplitEvaluationError("bootstrap samples must be 2000")
    if float(config.get("bootstrap_confidence", 0)) != 0.95:
        raise FixedSplitEvaluationError("bootstrap confidence must be 0.95")
    if set(config) != {
        "schema_version",
        "seeds",
        "target_order",
        "bootstrap_samples",
        "bootstrap_confidence",
        "frozen_winners",
        "monitoring",
    }:
        raise FixedSplitEvaluationError("config contains missing or unknown fields")


def validate_partition(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray, rows: int
) -> None:
    joined = np.concatenate((train, validation, test)).astype(np.int64, copy=False)
    if not np.array_equal(np.sort(joined), np.arange(rows, dtype=np.int64)):
        raise FixedSplitEvaluationError("fixed split must be a disjoint cover in source-row space")


def monitor_curves(
    training_loss: Sequence[float],
    validation_loss: Sequence[float],
    *,
    increase_mark_after: int,
    max_epochs: int,
) -> dict[str, object]:
    train = np.asarray(training_loss, dtype=np.float64)
    validation = np.asarray(validation_loss, dtype=np.float64)
    abnormal: list[str] = []
    warnings: list[str] = []
    if len(train) == 0 or len(validation) == 0 or len(train) != len(validation):
        abnormal.append("missing-or-misaligned-curves")
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(validation)):
        abnormal.append("nonfinite-loss")
    best_epoch = int(np.argmin(validation)) + 1 if len(validation) else None
    maximum_run = 0
    current_run = 0
    for before, after in zip(validation, validation[1:], strict=False):
        current_run = current_run + 1 if after > before else 0
        maximum_run = max(maximum_run, current_run)
    if maximum_run >= increase_mark_after:
        warnings.append("validation-loss-increased-consecutively")
    if len(validation) and validation[-1] > validation.min():
        warnings.append("final-validation-loss-above-best")
    return {
        "status": "abnormal" if abnormal else "warning" if warnings else "normal",
        "training_loss": train.tolist(),
        "validation_loss": validation.tolist(),
        "best_epoch": best_epoch,
        "best_validation_loss": float(validation.min()) if len(validation) else None,
        "epochs_ran": int(len(train)),
        "maximum_consecutive_validation_increases": maximum_run,
        "abnormal_reasons": abnormal,
        "warning_reasons": warnings,
        "early_stop_reason": (
            "patience-exhausted" if len(train) < max_epochs else "max-epochs-reached"
        ),
    }


def _score(truth: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> float:
    return float(np.mean(np.mean(np.abs(prediction - truth), axis=0) / scale))


def select_weights(
    predictions: Mapping[str, np.ndarray], truth: np.ndarray, scale: np.ndarray
) -> dict:
    names = list(predictions)
    stack = np.stack([predictions[name] for name in names])

    def objective(weights: np.ndarray) -> float:
        return _score(truth, np.tensordot(weights, stack, axes=(0, 0)), scale)

    fitted = minimize(
        objective,
        np.full(len(names), 1.0 / len(names)),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    weights = np.clip(fitted.x, 0.0, 1.0)
    if not fitted.success:
        raise FixedSplitEvaluationError(f"ensemble SLSQP failed: {fitted.message}")
    weights /= weights.sum()
    return {
        "model_order": names,
        "weights": {name: float(weight) for name, weight in zip(names, weights, strict=True)},
        "validation_score": objective(weights),
        "optimizer_success": bool(fitted.success),
        "selection_uses": "fixed-validation-labels-only",
    }


def blend(selection: Mapping[str, Any], predictions: Mapping[str, np.ndarray]) -> np.ndarray:
    return sum(
        float(selection["weights"][name]) * predictions[name] for name in selection["model_order"]
    )


def _fit_smoke_models(
    x: np.ndarray | sparse.spmatrix,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    x_dense = np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float64)
    mean, scale = y[train].mean(axis=0), y[train].std(axis=0)
    normalized = (y - mean) / scale
    timings = {}
    started = time.perf_counter()
    ridge = Ridge(alpha=1.0).fit(x_dense[train], normalized[train])
    timings["engineered_ridge"] = time.perf_counter() - started
    started = time.perf_counter()
    trees = ExtraTreesRegressor(n_estimators=12, max_depth=5, random_state=seed, n_jobs=1).fit(
        x_dense[train], normalized[train]
    )
    timings["xgboost"] = time.perf_counter() - started
    started = time.perf_counter()
    mlp = MLPRegressor(
        hidden_layer_sizes=(12,),
        random_state=seed,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=5,
        tol=1e-4,
    ).fit(x_dense[train], normalized[train])
    timings["mlp"] = time.perf_counter() - started
    validation_error = [1.0 - float(value) for value in (mlp.validation_scores_ or [])]
    curves = monitor_curves(
        list(mlp.loss_curve_),
        validation_error,
        increase_mark_after=2,
        max_epochs=30,
    )
    predictions = {
        "engineered_ridge": ridge.predict(x_dense[validation]) * scale + mean,
        "xgboost": trees.predict(x_dense[validation]) * scale + mean,
        "mlp": mlp.predict(x_dense[validation]) * scale + mean,
    }
    state = {
        "models": (ridge, trees, mlp),
        "mean": mean,
        "scale": scale,
        "curves": curves,
        "model_selection_seconds": timings,
    }
    return predictions, state


def _predict_smoke_models(
    state: Mapping[str, Any], x: np.ndarray | sparse.spmatrix, test: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    values = np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float64)[test]
    ridge, trees, mlp = state["models"]
    predictions = {}
    timings = {}
    for name, model in (("engineered_ridge", ridge), ("xgboost", trees), ("mlp", mlp)):
        started = time.perf_counter()
        predictions[name] = model.predict(values) * state["scale"] + state["mean"]
        timings[name] = time.perf_counter() - started
    return predictions, timings


class TargetAccessGate:
    """Keep the test array inaccessible until the durable global gate exists."""

    def __init__(self, values: np.ndarray):
        self.__values = np.asarray(values, dtype=np.float64)
        self.authorized = False
        self.read_count = 0

    def authorize(self, gate_hash: str) -> None:
        if len(gate_hash) != 64:
            raise FixedSplitEvaluationError("test authorization requires a SHA-256 gate")
        self.authorized = True

    def read(self) -> np.ndarray:
        if not self.authorized:
            raise FixedSplitEvaluationError("test labels requested before global freeze")
        self.read_count += 1
        return self.__values.copy()


class LazyTestTargetGate(TargetAccessGate):
    """Load private test labels only when the durable gate authorizes the read."""

    def __init__(self, loader: Any):
        self._loader = loader
        self.authorized = False
        self.read_count = 0

    def read(self) -> np.ndarray:
        if not self.authorized:
            raise FixedSplitEvaluationError("test labels requested before global freeze")
        self.read_count += 1
        values = np.asarray(self._loader(), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(TARGET_COLUMNS):
            raise FixedSplitEvaluationError("lazy test labels have the wrong shape")
        return values


def _fit_real_models(
    config: Mapping[str, Any],
    x: sparse.csr_matrix,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Fit frozen v1 winners on fixed train rows and predict fixed validation rows."""

    frozen = config["frozen_winners"]
    train_scale = y_train.std(axis=0)
    train_mean = y_train.mean(axis=0)
    normalized = (y_train - train_mean) / train_scale
    model_timings = {}
    started = time.perf_counter()
    ridge = make_pipeline(
        StandardScaler(with_mean=False),
        Ridge(alpha=float(frozen["ridge_alpha"]), solver="lsqr", tol=1e-4, max_iter=10_000),
    ).fit(x[train], normalized)
    ridge_validation = ridge.predict(x[validation]) * train_scale + train_mean
    model_timings["engineered_ridge"] = time.perf_counter() - started

    from mist_transfer_benchmark.live_demo import _new_xgb, _xgb_device

    xgb_models = []
    xgb_columns = []
    xgb_rounds = []
    xgb_params = {
        "max_depth": int(frozen["xgboost_max_depth"]),
        "learning_rate": float(frozen["xgboost_learning_rate"]),
        "n_estimators": int(frozen["xgboost_n_estimators"]),
        "subsample": float(frozen["xgboost_subsample"]),
        "colsample_bytree": float(frozen["xgboost_colsample_bytree"]),
        "min_child_weight": float(frozen["xgboost_min_child_weight"]),
    }
    started = time.perf_counter()
    for target_position in range(len(TARGET_COLUMNS)):
        model = _new_xgb(
            xgb_params,
            seed=seed + target_position,
            device=_xgb_device(),
            early_stopping_rounds=int(frozen["xgboost_early_stopping_rounds"]),
        )
        model.fit(
            x[train],
            y_train[:, target_position],
            eval_set=[(x[validation], y_validation[:, target_position])],
            verbose=False,
        )
        xgb_models.append(model)
        xgb_columns.append(model.predict(x[validation]))
        xgb_rounds.append(int(getattr(model, "best_iteration", 0)) + 1)
    model_timings["xgboost"] = time.perf_counter() - started

    import torch

    from mist_transfer_benchmark.live_demo import _make_mlp, _predict_mlp, _set_seed

    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp_input_scaler = StandardScaler(with_mean=False).fit(x[train])
    local_x = sparse.vstack(
        (
            mlp_input_scaler.transform(x[train]),
            mlp_input_scaler.transform(x[validation]),
        ),
        format="csr",
    )
    local_y = np.vstack((y_train, y_validation))
    local_train = np.arange(len(y_train), dtype=np.int64)
    local_validation = np.arange(len(y_train), len(local_y), dtype=np.int64)
    mlp = _make_mlp(
        list(frozen["mlp_hidden_dims"]), float(frozen["mlp_dropout"]), input_dim=x.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        mlp.parameters(),
        lr=float(frozen["mlp_learning_rate"]),
        weight_decay=float(frozen["mlp_weight_decay"]),
    )
    batch_size = int(frozen["mlp_batch_size"])
    loss_fn = torch.nn.MSELoss()
    best_value = float("inf")
    best_state = None
    restored_epoch = None
    stale = 0
    training_curve: list[float] = []
    validation_curve: list[float] = []
    from mist_transfer_benchmark.live_demo import _batch_rows

    started = time.perf_counter()
    for epoch in range(int(frozen["mlp_max_epochs"])):
        mlp.train()
        total_loss = 0.0
        total_rows = 0
        order = np.random.default_rng(seed + epoch).permutation(local_train)
        for features, positions in _batch_rows(local_x, order, batch_size):
            feature_tensor = torch.from_numpy(features).to(device)
            target_tensor = torch.from_numpy(
                ((local_y[positions] - train_mean) / train_scale).astype(np.float32)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(mlp(feature_tensor), target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(positions)
            total_rows += len(positions)
        training_curve.append(total_loss / total_rows)
        predicted = _predict_mlp(
            mlp, local_x, local_validation, train_mean, train_scale, batch_size, device
        )
        score = _score(y_validation, predicted, train_scale)
        validation_curve.append(score)
        if score < best_value - float(frozen["mlp_min_delta"]):
            best_value = score
            restored_epoch = epoch + 1
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in mlp.state_dict().items()
            }
        else:
            stale += 1
            if stale >= int(frozen["mlp_patience"]):
                break
    if best_state is None:
        raise FixedSplitEvaluationError("MLP did not produce a finite best epoch")
    mlp.load_state_dict(best_state)
    mlp_validation = _predict_mlp(
        mlp, local_x, local_validation, train_mean, train_scale, batch_size, device
    )
    model_timings["mlp"] = time.perf_counter() - started
    curves = monitor_curves(
        training_curve,
        validation_curve,
        increase_mark_after=int(config["monitoring"]["validation_increase_mark_after"]),
        max_epochs=int(frozen["mlp_max_epochs"]),
    )
    curves["restored_best_epoch"] = restored_epoch
    curves["early_stopping"] = {
        "max_epochs": int(frozen["mlp_max_epochs"]),
        "patience": int(frozen["mlp_patience"]),
        "min_delta": float(frozen["mlp_min_delta"]),
        "selection_metric": "validation-mean-normalized-mae",
        "uses_test_labels": False,
    }
    if curves["status"] == "abnormal":
        raise FixedSplitEvaluationError(f"abnormal MLP curves for seed {seed}")
    return (
        {
            "engineered_ridge": ridge_validation,
            "xgboost": np.column_stack(xgb_columns),
            "mlp": mlp_validation,
        },
        {
            "models": (ridge, xgb_models, mlp),
            "mean": train_mean,
            "scale": train_scale,
            "curves": curves,
            "mlp_device": device,
            "mlp_batch_size": batch_size,
            "xgboost_best_rounds": xgb_rounds,
            "mlp_input_scaler": mlp_input_scaler,
            "model_selection_seconds": model_timings,
        },
    )


def _predict_real_models(
    state: Mapping[str, Any], x: sparse.csr_matrix, test: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    from mist_transfer_benchmark.live_demo import _predict_mlp

    ridge, xgb_models, mlp = state["models"]
    timings = {}
    started = time.perf_counter()
    ridge_prediction = ridge.predict(x[test]) * state["scale"] + state["mean"]
    timings["engineered_ridge"] = time.perf_counter() - started
    started = time.perf_counter()
    xgb_prediction = np.column_stack([model.predict(x[test]) for model in xgb_models])
    timings["xgboost"] = time.perf_counter() - started
    started = time.perf_counter()
    mlp_prediction = _predict_mlp(
        mlp,
        sparse.csr_matrix(state["mlp_input_scaler"].transform(x[test])),
        np.arange(len(test), dtype=np.int64),
        state["mean"],
        state["scale"],
        state["mlp_batch_size"],
        state["mlp_device"],
    )
    timings["mlp"] = time.perf_counter() - started
    return {
        "engineered_ridge": ridge_prediction,
        "xgboost": xgb_prediction,
        "mlp": mlp_prediction,
    }, timings


def paired_delta_bootstrap(
    truth: np.ndarray,
    candidate: np.ndarray,
    mist: np.ndarray,
    scale: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    per_target_row_delta = (np.abs(candidate - truth) - np.abs(mist - truth)) / scale
    row_delta = np.mean(per_target_row_delta, axis=1)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    target_draws = np.empty((samples, len(TARGET_COLUMNS)))
    for position in range(samples):
        rows = rng.integers(0, len(row_delta), len(row_delta))
        draws[position] = row_delta[rows].mean()
        target_draws[position] = per_target_row_delta[rows].mean(axis=0)
    alpha = (1.0 - confidence) / 2.0
    return {
        "definition": "candidate-minus-fixed-mist; negative-favors-candidate",
        "point": float(row_delta.mean()),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
        "samples": samples,
        "seed": seed,
        "per_target": {
            target: {
                "point": float(per_target_row_delta[:, index].mean()),
                "lower": float(np.quantile(target_draws[:, index], alpha)),
                "upper": float(np.quantile(target_draws[:, index], 1.0 - alpha)),
            }
            for index, target in enumerate(TARGET_COLUMNS)
        },
    }


def prediction_bootstrap(
    truth: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    normalized = np.abs(prediction - truth) / scale
    rng = np.random.default_rng(seed)
    target_draws = np.empty((samples, len(TARGET_COLUMNS)))
    for position in range(samples):
        rows = rng.integers(0, len(truth), len(truth))
        target_draws[position] = normalized[rows].mean(axis=0)
    aggregate = target_draws.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "paired-row-percentile-bootstrap",
        "samples": samples,
        "confidence": confidence,
        "aggregate": {
            "point": float(normalized.mean()),
            "lower": float(np.quantile(aggregate, alpha)),
            "upper": float(np.quantile(aggregate, 1.0 - alpha)),
        },
        "per_target": {
            target: {
                "point": float(normalized[:, index].mean()),
                "lower": float(np.quantile(target_draws[:, index], alpha)),
                "upper": float(np.quantile(target_draws[:, index], 1.0 - alpha)),
            }
            for index, target in enumerate(TARGET_COLUMNS)
        },
    }


def _loss_html(seed_payloads: Sequence[Mapping[str, Any]]) -> str:
    cards = []
    for payload in seed_payloads:
        curve = payload["validation"]["mlp_monitoring"]
        validation_points = " ".join(
            f"{index},{max(0.0, 100 - value * 80):.2f}"
            for index, value in enumerate(curve["validation_loss"])
        )
        training_points = " ".join(
            f"{index},{max(0.0, 100 - value * 80):.2f}"
            for index, value in enumerate(curve["training_loss"])
        )
        cards.append(
            f"<section><h2>Seed {int(payload['seed'])}</h2>"
            f"<p class='{html.escape(curve['status'])}'>{html.escape(curve['status'])}; "
            f"best epoch {curve['best_epoch']}; restored {curve.get('restored_best_epoch')}; "
            f"validation MNMAE {curve['best_validation_loss']:.6f}; "
            f"runtime {payload['runtime']['selection_train_validation_seconds']:.3f}s; "
            f"warnings {html.escape(str(curve['warning_reasons']))}</p>"
            f"<svg viewBox='0 0 100 110'><polyline class='train' "
            f"points='{training_points}'/><polyline class='validation' "
            f"points='{validation_points}'/></svg></section>"
        )
    style = """<style>
body{font:15px system-ui;max-width:1000px;margin:auto;padding:24px}
section{border:1px solid #ccc;padding:16px;margin:12px}
.warning{color:#a45b00}.abnormal{color:#b00020}.normal{color:#087830}
svg{width:100%;height:180px}polyline{fill:none;stroke-width:1}
.train{stroke:#087830}.validation{stroke:#e07800}
</style>"""
    heading = """<h1>Fixed-split MLP validation monitor</h1>
<p>Green: training loss. Orange: validation MNMAE. Rising runs are marked.</p>"""
    return (
        "<!doctype html><meta charset='utf-8'><title>QM9 loss monitor</title>"
        + style
        + heading
        + "".join(cards)
    )


def run_smoke_protocol(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    include_mist_validation: bool = True,
) -> dict[str, object]:
    """Run the full leakage/checkpoint/output contract on deterministic synthetic data."""

    validate_config(config)
    rng = np.random.default_rng(91)
    x = rng.normal(size=(90, 10))
    y = x @ rng.normal(size=(10, len(TARGET_COLUMNS))) + rng.normal(
        scale=0.08, size=(90, len(TARGET_COLUMNS))
    )
    train, validation, test = np.arange(60), np.arange(60, 75), np.arange(75, 90)
    mist_validation = y[validation] + rng.normal(scale=0.18, size=(15, 12))
    mist_test = y[test] + rng.normal(scale=0.18, size=(15, 12))
    return run_fixed_split(
        config,
        x,
        y[train],
        y[validation],
        TargetAccessGate(y[test]),
        train,
        validation,
        test,
        mist_validation if include_mist_validation else None,
        mist_test,
        output_dir,
        input_identity={
            "mode": "deterministic-public-smoke-v1",
            "x_sha256": canonical_hash(x.tolist()),
        },
        smoke=True,
    )


def run_fixed_split(
    config: Mapping[str, Any],
    x: np.ndarray | sparse.spmatrix,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    test_gate: TargetAccessGate,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    mist_validation: np.ndarray | None,
    mist_test: np.ndarray,
    output_dir: Path,
    *,
    input_identity: Mapping[str, Any],
    smoke: bool = False,
    structural_novelty_labels: np.ndarray | None = None,
) -> dict[str, object]:
    """Execute the fixed-split route.  Real backends are intentionally gated below."""

    run_started = time.perf_counter()
    validate_config(config)
    validate_partition(train, validation, test, x.shape[0])
    if not np.all(np.isfinite(y_train)) or not np.all(np.isfinite(y_validation)):
        raise FixedSplitEvaluationError("training/validation targets contain nonfinite values")
    if not np.all(np.isfinite(mist_test)):
        raise FixedSplitEvaluationError("fixed MIST test predictions contain nonfinite values")
    if mist_validation is not None and not np.all(np.isfinite(mist_validation)):
        raise FixedSplitEvaluationError(
            "fixed MIST validation predictions contain nonfinite values"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    config_hash = canonical_hash(config)
    identity_hash = canonical_hash(input_identity)
    code_hash = file_sha256(Path(__file__))
    manifest = {
        "schema_version": "qm9-fixed-mist-split-v2-manifest-v1",
        "config_sha256": config_hash,
        "code_sha256": code_hash,
        "input_identity_sha256": identity_hash,
        "input_identity": dict(input_identity),
        "selected_seeds": list(config["seeds"]),
        "selected_cells": [],
        "completed_seeds": [],
        "artifact_sha256": {},
        "artifact_bytes": {},
        "events": [],
        "critical_reviews": critical_review_plan(),
        "publication_ready": False,
        "complete": False,
        "test_access": {"authorized": False, "read_count": 0},
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("complete") is not True:
            raise FixedSplitEvaluationError(
                "incomplete output is not resumable; choose a new output directory"
            )
        if any(
            (
                old.get("config_sha256") != config_hash,
                old.get("code_sha256") != code_hash,
                old.get("input_identity_sha256") != identity_hash,
            )
        ):
            raise FixedSplitEvaluationError("complete output identity differs")
        for relative, expected in old.get("artifact_sha256", {}).items():
            path = output_dir / relative
            if not path.is_file() or file_sha256(path) != expected:
                raise FixedSplitEvaluationError(f"complete output artifact changed: {relative}")
            if path.stat().st_size != old.get("artifact_bytes", {}).get(relative):
                raise FixedSplitEvaluationError(
                    f"complete output artifact size changed: {relative}"
                )
        return old
    _atomic_json(manifest_path, manifest)

    def record(event: str, relative: str | None = None) -> None:
        manifest["events"].append(
            {"sequence": len(manifest["events"]) + 1, "event": event, "artifact": relative}
        )

    def register(path: Path) -> None:
        relative = str(path.relative_to(output_dir))
        manifest["artifact_sha256"][relative] = file_sha256(path)
        manifest["artifact_bytes"][relative] = path.stat().st_size

    record("run-created")
    _mark_review(
        manifest["critical_reviews"],
        "input-boundary",
        "automated-review-passed",
        {"input_identity_sha256": identity_hash, "test_label_reads": 0},
    )
    record("critical-review-input-boundary-passed")
    _atomic_json(manifest_path, manifest)
    states: dict[int, Mapping[str, Any]] = {}
    seed_payloads = []
    train_scale = np.std(y_train, axis=0)
    if np.any(train_scale <= 0):
        raise FixedSplitEvaluationError("training-only target scale has zero values")
    for seed in config["seeds"]:
        seed_started = time.perf_counter()
        if smoke:
            validation_predictions, state = _fit_smoke_models(
                x, np.vstack((y_train, y_validation)), train, validation, int(seed)
            )
        else:
            if not sparse.issparse(x):
                raise FixedSplitEvaluationError("real engineered features must be sparse")
            validation_predictions, state = _fit_real_models(
                config,
                sparse.csr_matrix(x),
                y_train,
                y_validation,
                train,
                validation,
                int(seed),
            )
        states[int(seed)] = state
        traditional = select_weights(validation_predictions, y_validation, train_scale)
        supplemental = None
        if mist_validation is not None:
            supplemental = select_weights(
                {**validation_predictions, "mist": mist_validation}, y_validation, train_scale
            )
        validation_hashes = {}
        for family, prediction in validation_predictions.items():
            if not np.all(np.isfinite(prediction)):
                raise FixedSplitEvaluationError(f"nonfinite {family} validation prediction")
            validation_path = output_dir / "validation_predictions" / f"{seed}-{family}.npy"
            _atomic_npy(validation_path, prediction)
            register(validation_path)
            validation_hashes[family] = file_sha256(validation_path)
        payload = {
            "schema_version": "qm9-fixed-mist-split-v2-seed-v1",
            "seed": int(seed),
            "validation": {
                "frozen_hyperparameters": dict(config["frozen_winners"]),
                "scores": {
                    name: _score(y_validation, value, train_scale)
                    for name, value in validation_predictions.items()
                },
                "mlp_monitoring": state["curves"],
                "prediction_sha256": validation_hashes,
                "xgboost_selected_rounds": state.get("xgboost_best_rounds"),
                "scaler_provenance": {
                    "input": "StandardScaler-fit-fixed-train-only",
                    "target": "per-target-mean-std-fit-fixed-train-only",
                    "test_or_validation_used_to_fit": False,
                },
                "traditional_ensemble": traditional,
                "all_model_ensemble": supplemental,
                "all_model_ensemble_status": "available"
                if supplemental
                else "omitted-no-fixed-mist-validation-predictions",
            },
            "test_metrics": None,
            "test_labels_read": False,
            "runtime": {
                "selection_train_validation_seconds": time.perf_counter() - seed_started,
                "model_selection_seconds": state["model_selection_seconds"],
                "test_prediction_seconds": None,
                "model_test_prediction_seconds": None,
                "process_peak_rss": _peak_rss(),
            },
        }
        path = output_dir / "seeds" / f"{seed}.json"
        _atomic_json(path, payload)
        manifest["selected_cells"].append(str(seed))
        register(path)
        record("seed-validation-selected", str(path.relative_to(output_dir)))
        _atomic_json(manifest_path, manifest)
        seed_payloads.append(payload)
    freeze = {
        "schema_version": "qm9-fixed-mist-split-v2-global-freeze-v1",
        "config_sha256": config_hash,
        "code_sha256": code_hash,
        "input_identity_sha256": identity_hash,
        "selected_seeds": manifest["selected_cells"],
        "selection_sha256": {
            str(payload["seed"]): canonical_hash(payload["validation"]) for payload in seed_payloads
        },
    }
    gate_path = output_dir / "global-freeze-gate.json"
    _atomic_json(gate_path, freeze)
    gate_hash = file_sha256(gate_path)
    register(gate_path)
    record("global-freeze-closed", str(gate_path.relative_to(output_dir)))
    _mark_review(
        manifest["critical_reviews"],
        "selection-freeze",
        "automated-review-passed",
        {"global_freeze_sha256": gate_hash, "selected_seed_count": len(seed_payloads)},
    )
    record("critical-review-selection-freeze-passed", str(gate_path.relative_to(output_dir)))
    _mark_review(
        manifest["critical_reviews"],
        "test-unlock",
        "automated-review-passed",
        {"authorized_read_count": 1, "global_freeze_sha256": gate_hash},
    )
    record("critical-review-test-unlock-passed", str(gate_path.relative_to(output_dir)))
    test_gate.authorize(gate_hash)
    manifest["test_access"] = {"authorized": True, "gate_sha256": gate_hash, "read_count": 0}
    _atomic_json(manifest_path, manifest)
    y_test = test_gate.read()
    per_seed_predictions: dict[str, list[np.ndarray]] = {key: [] for key in MODEL_KEYS}
    for payload in seed_payloads:
        seed = int(payload["seed"])
        prediction_started = time.perf_counter()
        predictions, model_prediction_seconds = (
            _predict_smoke_models(states[seed], x, test)
            if smoke
            else _predict_real_models(states[seed], sparse.csr_matrix(x), test)
        )
        predictions["traditional_ensemble"] = blend(
            payload["validation"]["traditional_ensemble"], predictions
        )
        supplemental = payload["validation"]["all_model_ensemble"]
        if supplemental is not None:
            predictions["all_model_ensemble"] = blend(
                supplemental, {**predictions, "mist": mist_test}
            )
        metrics = {}
        for family, prediction in predictions.items():
            path = output_dir / "predictions" / f"{seed}-{family}.npy"
            prediction_hash = _atomic_npy(path, prediction)
            register(path)
            if prediction_hash != manifest["artifact_sha256"][str(path.relative_to(output_dir))]:
                raise FixedSplitEvaluationError("prediction hash registration failed")
            metrics[family] = native_metrics(y_test, prediction, train_scale)
            per_seed_predictions[family].append(prediction)
        payload["test_metrics"] = metrics
        payload["test_labels_read"] = True
        payload["runtime"]["test_prediction_seconds"] = time.perf_counter() - prediction_started
        payload["runtime"]["model_test_prediction_seconds"] = model_prediction_seconds
        path = output_dir / "seeds" / f"{seed}.json"
        _atomic_json(path, payload)
        register(path)
        record("seed-test-completed", str(path.relative_to(output_dir)))
        manifest["completed_seeds"].append(seed)
        _atomic_json(manifest_path, manifest)
    method_summary = {}
    paired = {}
    per_target = {}
    for family, values in per_seed_predictions.items():
        if not values:
            continue
        seed_scores = [_score(y_test, value, train_scale) for value in values]
        mean_prediction = np.mean(values, axis=0)
        method_summary[family] = {
            "role": (
                "primary-traditional"
                if family == "traditional_ensemble"
                else "supplemental-includes-fixed-mist"
                if family == "all_model_ensemble"
                else "traditional-component"
            ),
            "seed_scores": seed_scores,
            "mean": float(np.mean(seed_scores)),
            "std": float(np.std(seed_scores, ddof=1)),
        }
        paired[family] = paired_delta_bootstrap(
            y_test,
            mean_prediction,
            mist_test,
            train_scale,
            samples=int(config["bootstrap_samples"]),
            seed=20260718,
            confidence=float(config["bootstrap_confidence"]),
        )
        metrics = native_metrics(y_test, mean_prediction, train_scale)
        per_target[family] = {}
        for target_index, target in enumerate(TARGET_COLUMNS):
            values_by_seed = [
                float(
                    np.mean(np.abs(prediction[:, target_index] - y_test[:, target_index]))
                    / train_scale[target_index]
                )
                for prediction in values
            ]
            per_target[family][target] = {
                "seed_values": values_by_seed,
                "mean": float(np.mean(values_by_seed)),
                "std": float(np.std(values_by_seed, ddof=1)),
                "seed_averaged_prediction_metrics": metrics["per_target"][target],
            }
    fixed_mist_metrics = native_metrics(y_test, mist_test, train_scale)
    fixed_mist_ci = prediction_bootstrap(
        y_test,
        mist_test,
        train_scale,
        samples=int(config["bootstrap_samples"]),
        seed=20260718,
        confidence=float(config["bootstrap_confidence"]),
    )
    supplemental_available = bool(per_seed_predictions["all_model_ensemble"])
    novelty: dict[str, object]
    if structural_novelty_labels is None:
        novelty = {
            "status": "not-estimated-in-synthetic-smoke" if smoke else "unavailable",
            "reason": (
                "synthetic rows have no molecular structures"
                if smoke
                else "no authenticated structural-novelty labels supplied"
            ),
        }
    else:
        labels = np.asarray(structural_novelty_labels).astype(str)
        if labels.shape != (len(y_test),):
            raise FixedSplitEvaluationError("novelty labels must align to fixed test rows")
        strata = {}
        for label in sorted(set(labels)):
            mask = labels == label
            strata[label] = {
                "rows": int(mask.sum()),
                "fixed_mist": native_metrics(y_test[mask], mist_test[mask], train_scale),
                "methods": {
                    family: native_metrics(y_test[mask], np.mean(values, axis=0)[mask], train_scale)
                    for family, values in per_seed_predictions.items()
                    if values
                },
            }
        novelty = {
            "status": "available",
            "method": "fixed-test-Bemis-Murcko-scaffold-seen-in-fixed-train",
            "strata": strata,
            "similarity_bins": {
                "status": "not-computed",
                "reason": "no separately authenticated nearest-train similarity cache supplied",
            },
        }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "fixed_mist": {
            "inference_only": True,
            "seed_repetitions": 1,
            "metrics": fixed_mist_metrics,
            "bootstrap_ci": fixed_mist_ci,
            "prediction_sha256": canonical_hash(mist_test.tolist()),
        },
        "methods": method_summary,
        "paired_delta_vs_mist": paired,
        "per_target": per_target,
        "reporting_roles": {
            "primary": "traditional_ensemble-excludes-mist",
            "supplemental": (
                "all_model_ensemble-includes-fixed-mist" if supplemental_available else "omitted"
            ),
            "supplemental_omission_reason": (
                None
                if supplemental_available
                else "no-authenticated-fixed-mist-validation-predictions"
            ),
        },
        "runtime": {
            "mode": "deterministic-smoke" if smoke else "real-fixed-split",
            "test_label_reads": test_gate.read_count,
            "total_wall_seconds": time.perf_counter() - run_started,
            "process_peak_rss": _peak_rss(),
            "per_seed": {str(payload["seed"]): payload["runtime"] for payload in seed_payloads},
            "historical_mist": input_identity.get(
                "mist_historical_runtime",
                {"status": "unavailable", "reason": "not supplied by input provenance"},
            ),
        },
        "structural_novelty_strata": novelty,
    }
    summary_path = output_dir / "summary.json"
    _atomic_json(summary_path, summary)
    loss_path = output_dir / "loss-monitor.html"
    _atomic_text(loss_path, _loss_html(seed_payloads))
    register(summary_path)
    register(loss_path)
    record("summary-completed", "summary.json")
    _mark_review(
        manifest["critical_reviews"],
        "publication",
        "independent-review-required",
        {"summary": "summary.json", "loss_monitor": "loss-monitor.html"},
    )
    record("critical-review-publication-pending", "summary.json")
    manifest["test_access"]["read_count"] = test_gate.read_count
    manifest["complete"] = len(manifest["completed_seeds"]) == len(config["seeds"])
    _atomic_json(manifest_path, manifest)
    return manifest
