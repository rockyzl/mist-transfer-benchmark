"""Index-scoped QM9 target loading used to enforce the validation/test boundary."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .constants import EXPECTED_HEADER, TARGET_COLUMNS
from .data import ValidatedQM9


class TargetLoadError(ValueError):
    """Raised when an index-scoped target read violates the source contract."""


def load_targets_for_indices(
    path: str | Path,
    indices: np.ndarray,
    data: ValidatedQM9,
) -> np.ndarray:
    """Read only requested label rows and return them in the caller's exact index order."""

    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim != 1 or len(np.unique(requested)) != len(requested):
        raise TargetLoadError("target indices must be a unique one-dimensional array")
    if np.any(requested < 0) or np.any(requested >= data.row_count):
        raise TargetLoadError("target indices contain an out-of-range source row")
    positions = {int(source_index): position for position, source_index in enumerate(requested)}
    matrix = np.empty((len(requested), len(TARGET_COLUMNS)), dtype=np.float64)
    seen = np.zeros(len(requested), dtype=bool)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise TargetLoadError("QM9 target source is empty") from error
        if header != EXPECTED_HEADER:
            raise TargetLoadError("QM9 target source header/order differs")
        offsets = [header.index(name) for name in TARGET_COLUMNS]
        mol_id_offset = header.index("mol_id")
        smiles_offset = header.index("smiles")
        for source_row_index, row in enumerate(reader):
            position = positions.get(source_row_index)
            if position is None:
                continue
            if row[mol_id_offset] != data.mol_ids[source_row_index]:
                raise TargetLoadError("mol_id changed during index-scoped target loading")
            if row[smiles_offset] != data.source_smiles[source_row_index]:
                raise TargetLoadError("SMILES changed during index-scoped target loading")
            matrix[position] = [float(row[offset]) for offset in offsets]
            seen[position] = True
    if not np.all(seen):
        raise TargetLoadError("index-scoped target read did not find every requested row")
    if not np.all(np.isfinite(matrix)):
        raise TargetLoadError("index-scoped target matrix contains a non-finite value")
    return matrix
