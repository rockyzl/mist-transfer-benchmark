from __future__ import annotations

import numpy as np
from scipy import sparse

from mist_transfer_benchmark.qm9.engineered_features import (
    GLOBAL_DESCRIPTOR_NAMES,
    build_count_ecfp4_plus_globals,
    engineered_feature_schema,
)


def test_count_ecfp_plus_globals_is_finite_sparse_and_deterministic():
    smiles = ["CC", "CCO", "c1ccccc1", "CC(=O)O"]
    first = build_count_ecfp4_plus_globals(smiles, fp_size=64)
    second = build_count_ecfp4_plus_globals(smiles, fp_size=64)
    assert sparse.isspmatrix_csr(first)
    assert first.shape == (len(smiles), 64 + len(GLOBAL_DESCRIPTOR_NAMES))
    assert np.all(np.isfinite(first.data))
    assert (first != second).nnz == 0
    assert first[:, :64].max() >= 1.0
    assert np.count_nonzero(first[:, 64:].toarray()) > 0


def test_feature_schema_declares_training_only_scaling():
    schema = engineered_feature_schema(fp_size=64)
    assert schema["columns"] == 64 + len(GLOBAL_DESCRIPTOR_NAMES)
    assert "training-rows-only" in schema["scaling"]
