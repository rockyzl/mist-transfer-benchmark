"""Small, deterministic ECFP regression baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .fingerprints import FingerprintConfig, ecfp_matrix, nearest_train_similarity

SUPPORTED_MODELS = ("dummy", "random_forest", "ridge", "tanimoto_1nn")


def _build_model(name: str, seed: int):
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=128,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"unsupported model {name!r}; choose from {SUPPORTED_MODELS}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    count = len(y_true)
    if count == 0:
        return {
            "n": 0,
            "mae": None,
            "median_ae": None,
            "rmse": None,
            "r2": None,
            "spearman": None,
        }
    r2: float | None = None
    spearman: float | None = None
    if count >= 2 and not np.isclose(np.var(y_true), 0.0):
        r2 = float(r2_score(y_true, y_pred))
        predicted_ranks = pd.Series(y_pred).rank(method="average").to_numpy()
        target_ranks = pd.Series(y_true).rank(method="average").to_numpy()
        if not np.isclose(np.var(predicted_ranks), 0.0):
            spearman = float(np.corrcoef(target_ranks, predicted_ranks)[0, 1])
    absolute_error = np.abs(y_true - y_pred)
    return {
        "n": count,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(np.median(absolute_error)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
        "spearman": spearman,
    }


def _similarity_summary(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def run_ecfp_baselines(
    frame: pd.DataFrame,
    assignments: pd.Series,
    model_names: Sequence[str],
    fingerprint_config: FingerprintConfig,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit requested regressors and return metrics plus row-level predictions."""

    unknown = sorted(set(model_names) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"unsupported models: {unknown}; choose from {SUPPORTED_MODELS}")
    if not model_names:
        raise ValueError("at least one model is required")

    matrix = ecfp_matrix(frame["canonical_smiles"].tolist(), fingerprint_config)
    y = frame["target_v"].to_numpy(dtype=float)
    split_values = assignments.to_numpy()
    train_positions = np.flatnonzero(split_values == "train")
    if len(train_positions) == 0:
        raise ValueError("the split has no training rows")

    train_record_ids = frame.iloc[train_positions]["record_id"].tolist()
    max_similarity, nearest_ids = nearest_train_similarity(
        matrix, train_positions, train_record_ids
    )
    train_targets_by_id = dict(zip(train_record_ids, y[train_positions], strict=True))

    prediction_tables: list[pd.DataFrame] = []
    metrics: dict[str, object] = {}
    breakdowns: dict[str, object] = {}
    for name in model_names:
        if name == "tanimoto_1nn":
            predicted = np.asarray([train_targets_by_id[record_id] for record_id in nearest_ids])
        else:
            model = _build_model(name, seed)
            model.fit(matrix[train_positions], y[train_positions])
            predicted = model.predict(matrix)
        model_metrics: dict[str, object] = {}
        for split_name in ("train", "validation", "test"):
            mask = split_values == split_name
            model_metrics[split_name] = _metrics(y[mask], predicted[mask])
        metrics[name] = model_metrics

        prediction_tables.append(
            pd.DataFrame(
                {
                    "record_id": frame["record_id"].to_numpy(),
                    "canonical_smiles": frame["canonical_smiles"].to_numpy(),
                    "split": split_values,
                    "model": name,
                    "target_v": y,
                    "prediction_v": predicted,
                    "absolute_error_v": np.abs(y - predicted),
                    "max_train_tanimoto": max_similarity,
                    "nearest_train_record_id": nearest_ids,
                    "chemical_family": frame["chemical_family"].to_numpy(),
                }
            )
        )

        held_out = split_values != "train"
        similarity_bins = pd.cut(
            max_similarity,
            bins=[-0.001, 0.3, 0.6, 0.8, 1.0],
            labels=["[0,0.3]", "(0.3,0.6]", "(0.6,0.8]", "(0.8,1.0]"],
        )
        model_breakdown: dict[str, object] = {"similarity_bins": {}, "chemical_family": {}}
        for label in similarity_bins.categories:
            mask = held_out & (similarity_bins == label)
            model_breakdown["similarity_bins"][str(label)] = _metrics(y[mask], predicted[mask])
        for family in sorted(frame.loc[held_out, "chemical_family"].astype(str).unique()):
            mask = held_out & (frame["chemical_family"].astype(str).to_numpy() == family)
            model_breakdown["chemical_family"][family] = _metrics(y[mask], predicted[mask])
        breakdowns[name] = model_breakdown

    similarity = {
        split_name: _similarity_summary(max_similarity[split_values == split_name])
        for split_name in ("train", "validation", "test")
    }
    return {"metrics": metrics, "similarity": similarity, "breakdowns": breakdowns}, pd.concat(
        prediction_tables, ignore_index=True
    )
