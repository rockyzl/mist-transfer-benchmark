"""Frozen train-only 12-target standardization and audit artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .io import canonical_hash


class ScalerContractError(ValueError):
    """Raised when target scaling differs from the frozen protocol."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def fit_frozen_scaler(
    config: dict[str, object], y_train: np.ndarray
) -> tuple[StandardScaler, dict[str, object]]:
    section = config["classical"]["target_standard_scaler"]
    expected = {
        "implementation": "sklearn.preprocessing.StandardScaler",
        "copy": True,
        "with_mean": True,
        "with_std": True,
        "fit_split": "train",
        "variance_ddof": 0,
        "zero_variance_policy": "stop",
        "get_params_sha256_required": True,
        "fitted_state_fields": [
            "mean_",
            "var_",
            "scale_",
            "n_samples_seen_",
            "n_features_in_",
        ],
    }
    for key, value in expected.items():
        if section.get(key) != value:
            raise ScalerContractError(f"target_standard_scaler.{key} differs")
    values = np.asarray(y_train, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12 or not np.all(np.isfinite(values)):
        raise ScalerContractError("training target matrix must be finite with 12 columns")
    scaler = StandardScaler(
        copy=section["copy"],
        with_mean=section["with_mean"],
        with_std=section["with_std"],
    )
    scaler.fit(values)
    if np.any(scaler.var_ == 0.0) or np.any(scaler.scale_ == 0.0):
        raise ScalerContractError("at least one training target has zero variance")
    params = {key: _jsonable(value) for key, value in scaler.get_params(deep=False).items()}
    state = {field: _jsonable(getattr(scaler, field)) for field in section["fitted_state_fields"]}
    artifact = {
        "schema_version": "qm9-target-scaler-v1",
        "fit_rows": int(values.shape[0]),
        "fit_columns": int(values.shape[1]),
        "fit_split": "train",
        "variance_ddof": 0,
        "parameters": params,
        "parameters_canonical_json_sha256": canonical_hash(params),
        "fitted_state": state,
        "fitted_state_canonical_json_sha256": canonical_hash(state),
    }
    return scaler, artifact
