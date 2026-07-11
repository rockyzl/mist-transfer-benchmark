"""Deterministic single-worker ECFP Tanimoto 1-NN control."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from rdkit import DataStructs
from scipy import sparse


def _explicit_bit_vectors(matrix: sparse.csr_matrix, indices: np.ndarray) -> list:
    result = []
    for source_index in indices:
        row = matrix.getrow(int(source_index))
        fingerprint = DataStructs.ExplicitBitVect(matrix.shape[1])
        fingerprint.SetBitsFromList([int(value) for value in row.indices])
        result.append(fingerprint)
    return result


def tanimoto_1nn_predict(
    matrix: sparse.csr_matrix,
    train_indices: np.ndarray,
    train_targets: np.ndarray,
    query_indices: np.ndarray,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict in ascending training-source order; first maximum is the frozen tie-break."""

    train_order = np.argsort(train_indices, kind="stable")
    sorted_train_indices = np.asarray(train_indices, dtype=np.int64)[train_order]
    sorted_targets = np.asarray(train_targets, dtype=np.float64)[train_order]
    query_order = np.argsort(query_indices, kind="stable")
    sorted_queries = np.asarray(query_indices, dtype=np.int64)[query_order]
    train_fingerprints = _explicit_bit_vectors(matrix, sorted_train_indices)
    query_fingerprints = _explicit_bit_vectors(matrix, sorted_queries)
    sorted_predictions = np.empty((len(sorted_queries), sorted_targets.shape[1]), dtype=np.float64)
    sorted_nearest = np.empty(len(sorted_queries), dtype=np.int64)
    sorted_similarity = np.empty(len(sorted_queries), dtype=np.float64)
    for position, fingerprint in enumerate(query_fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints)
        nearest_position = int(np.argmax(similarities))
        sorted_predictions[position] = sorted_targets[nearest_position]
        sorted_nearest[position] = sorted_train_indices[nearest_position]
        sorted_similarity[position] = similarities[nearest_position]
        if progress is not None and (position + 1) % 1_000 == 0:
            progress(f"Tanimoto 1-NN processed {position + 1:,}/{len(sorted_queries):,} queries")
    inverse = np.empty(len(query_order), dtype=np.int64)
    inverse[query_order] = np.arange(len(query_order), dtype=np.int64)
    return (
        sorted_predictions[inverse],
        sorted_nearest[inverse],
        sorted_similarity[inverse],
    )
