from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from mist_transfer_benchmark.qm9.phase2_features import (
    FeatureContractError,
    build_ecfp4_csr,
    contract_from_config,
    csr_canonical_sha256,
    save_csr_atomic,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config():
    with (REPO_ROOT / "configs/qm9_28m.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_frozen_ecfp4_build_is_sparse_binary_ordered_and_deterministic(tmp_path):
    contract = contract_from_config(_config())
    smiles = ["CC", "O", "c1ccccc1"]
    first = build_ecfp4_csr(smiles, contract)
    second = build_ecfp4_csr(smiles, contract)

    assert sparse.isspmatrix_csr(first)
    assert first.shape == (3, 2048)
    assert first.dtype == np.float64
    assert set(first.data) == {1.0}
    assert (first != second).nnz == 0
    assert csr_canonical_sha256(first) == csr_canonical_sha256(second)
    record = save_csr_atomic(tmp_path / "features.npz", first)
    assert record["canonical_csr_sha256"] == csr_canonical_sha256(first)


def test_feature_contract_rejects_single_option_drift():
    config = copy.deepcopy(_config())
    config["features"]["include_chirality"] = False
    with pytest.raises(FeatureContractError, match="include_chirality"):
        contract_from_config(config)

    config = copy.deepcopy(_config())
    config["features"]["feature_matrix_dtype"] = "uint8"
    with pytest.raises(FeatureContractError, match="feature_matrix_dtype"):
        contract_from_config(config)


def test_invalid_smiles_stops_instead_of_dropping_a_row():
    with pytest.raises(FeatureContractError, match="may not be dropped"):
        build_ecfp4_csr(["CC", "not-a-smiles"], contract_from_config(_config()))
