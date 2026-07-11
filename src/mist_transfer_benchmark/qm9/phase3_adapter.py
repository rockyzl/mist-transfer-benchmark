"""Pure offline guards for pinned MIST batching and named prediction outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .constants import TARGET_COLUMNS
from .phase3_model import EXPECTED_UNITS


class Phase3AdapterError(ValueError):
    """Raised when a model batch violates the fixed prediction contract."""


def stack_named_outputs(
    output: Mapping[str, Mapping[str, Any]], *, expected_rows: int
) -> np.ndarray:
    """Stack checkpoint named outputs strictly in frozen config order."""

    if set(output) != set(TARGET_COLUMNS):
        raise Phase3AdapterError("named prediction channels differ from the frozen target set")
    columns: list[np.ndarray] = []
    for name, unit in zip(TARGET_COLUMNS, EXPECTED_UNITS, strict=True):
        record = output[name]
        if set(record) < {"value", "unit"} or record["unit"] != unit:
            raise Phase3AdapterError(f"prediction metadata differs for channel {name}")
        value = record["value"]
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (expected_rows,):
            raise Phase3AdapterError(
                f"prediction channel {name} has shape {array.shape}; expected {(expected_rows,)}"
            )
        columns.append(array)
    result = np.column_stack(columns)
    validate_prediction_matrix(result, expected_rows=expected_rows)
    return result


def validate_prediction_matrix(values: np.ndarray, *, expected_rows: int) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64)
    if prediction.shape != (expected_rows, len(TARGET_COLUMNS)):
        raise Phase3AdapterError(
            f"prediction shape {prediction.shape}; expected {(expected_rows, 12)}"
        )
    if not np.all(np.isfinite(prediction)):
        raise Phase3AdapterError("prediction matrix contains a non-finite value")
    return prediction


def batched_predict(
    smiles: Sequence[str],
    *,
    batch_size: int,
    predict_batch: Callable[[list[str]], np.ndarray],
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Run a fixed adapter in source order with no preprocessing or post-transform."""

    if type(batch_size) is not int or batch_size <= 0:
        raise Phase3AdapterError("batch size must be a positive integer")
    if not smiles or any(not isinstance(value, str) or not value for value in smiles):
        raise Phase3AdapterError("raw SMILES must be nonempty strings")
    chunks: list[np.ndarray] = []
    total = len(smiles)
    for start in range(0, total, batch_size):
        batch = list(smiles[start : start + batch_size])
        prediction = validate_prediction_matrix(
            predict_batch(batch), expected_rows=len(batch)
        )
        chunks.append(prediction)
        if progress is not None:
            progress(min(start + len(batch), total), total)
    result = np.concatenate(chunks, axis=0)
    return validate_prediction_matrix(result, expected_rows=total)


def assert_no_manual_inverse_transform(metadata: Mapping[str, Any]) -> None:
    """Fail if an adapter claims any post-model target transform."""

    if metadata.get("manual_inverse_transform_applied") is not False:
        raise Phase3AdapterError("manual inverse transformation is forbidden")
    if metadata.get("model_predict_returns_native_units") is not True:
        raise Phase3AdapterError("adapter did not attest native-unit model.predict output")
