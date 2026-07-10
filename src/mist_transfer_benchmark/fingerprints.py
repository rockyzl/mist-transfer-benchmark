"""RDKit ECFP features and similarity diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


@dataclass(frozen=True)
class FingerprintConfig:
    radius: int = 2
    n_bits: int = 2048
    include_chirality: bool = True

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError("fingerprint radius must be non-negative")
        if self.n_bits <= 0:
            raise ValueError("fingerprint n_bits must be positive")


def ecfp_matrix(smiles: Sequence[str], config: FingerprintConfig) -> np.ndarray:
    """Create a dense binary Morgan/ECFP matrix."""

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.radius,
        fpSize=config.n_bits,
        includeChirality=config.include_chirality,
    )
    matrix = np.zeros((len(smiles), config.n_bits), dtype=np.uint8)
    for position, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid SMILES passed to fingerprinting: {value!r}")
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[position])
    return matrix


def nearest_train_similarity(
    matrix: np.ndarray,
    train_positions: np.ndarray,
    train_record_ids: Sequence[str],
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, list[str]]:
    """Return maximum binary Tanimoto and nearest training record for every row."""

    if len(train_positions) == 0:
        raise ValueError("at least one training row is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    train = matrix[train_positions]
    right = train.astype(np.int32, copy=False)
    right_sums = right.sum(axis=1)
    maxima = np.empty(len(matrix), dtype=float)
    nearest_positions = np.empty(len(matrix), dtype=int)
    for start in range(0, len(matrix), chunk_size):
        stop = min(start + chunk_size, len(matrix))
        left = matrix[start:stop].astype(np.int32, copy=False)
        intersections = left @ right.T
        unions = left.sum(axis=1)[:, None] + right_sums[None, :] - intersections
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions != 0,
        )
        chunk_nearest = similarities.argmax(axis=1)
        nearest_positions[start:stop] = chunk_nearest
        maxima[start:stop] = similarities[np.arange(stop - start), chunk_nearest]
    nearest_ids = [str(train_record_ids[position]) for position in nearest_positions]
    return maxima, nearest_ids
