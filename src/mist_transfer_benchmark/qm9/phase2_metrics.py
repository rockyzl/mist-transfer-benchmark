"""Native checkpoint-scale metrics for all twelve frozen QM9 targets."""

from __future__ import annotations

import numpy as np

from .constants import TARGET_COLUMNS


class MetricContractError(ValueError):
    """Raised when a metric input is incomplete or invalid."""


def native_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_target_scale: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    scale = np.asarray(train_target_scale, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] != len(TARGET_COLUMNS) or truth.shape != prediction.shape:
        raise MetricContractError("metric arrays must have identical [rows, 12] shapes")
    if len(truth) == 0 or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise MetricContractError("metric arrays must be nonempty and finite")
    if scale.shape != (len(TARGET_COLUMNS),) or np.any(scale <= 0) or not np.all(
        np.isfinite(scale)
    ):
        raise MetricContractError("training target scale must contain 12 positive values")
    residual = prediction - truth
    mae = np.mean(np.abs(residual), axis=0)
    rmse = np.sqrt(np.mean(np.square(residual), axis=0))
    denominator = np.sum(np.square(truth - np.mean(truth, axis=0)), axis=0)
    numerator = np.sum(np.square(residual), axis=0)
    r2 = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    r2 = 1.0 - r2
    normalized = mae / scale
    per_target = {
        name: {
            "mae": float(mae[position]),
            "rmse": float(rmse[position]),
            "r2": float(r2[position]),
            "mae_over_training_target_standard_deviation": float(normalized[position]),
        }
        for position, name in enumerate(TARGET_COLUMNS)
    }
    return {
        "rows": int(len(truth)),
        "target_order": list(TARGET_COLUMNS),
        "value_scale": "native-checkpoint-target-scale",
        "per_target": per_target,
        "mean_normalized_mae_across_12_targets": float(np.mean(normalized)),
    }
