from __future__ import annotations

import numpy as np
import pytest

from mist_transfer_benchmark.qm9.data import validate_qm9_csv
from mist_transfer_benchmark.qm9.duplicates import (
    CanonicalizationError,
    audit_duplicates,
    row_manifest,
)
from mist_transfer_benchmark.qm9.split import CandidateSplit

from .conftest import write_qm9_csv


def test_duplicate_audit_reports_train_test_overlap_and_clean_cohort(tmp_path):
    data = validate_qm9_csv(
        write_qm9_csv(tmp_path / "qm9.csv", ["C", "CC", "C", "O", "O", "N"]),
        expected_rows=6,
    )
    split = CandidateSplit(
        train=np.array([0, 1]), validation=np.array([3]), test=np.array([2, 4, 5])
    )

    audit = audit_duplicates(data, split)
    manifest = list(row_manifest(data, split, audit))

    assert audit.summary["train_test_overlap"] == {
        "identity_count": 1,
        "train_rows": 1,
        "test_rows": 1,
    }
    assert audit.summary["duplicate_clean_test"]["retained_rows"] == 1
    assert manifest[2]["duplicate_clean_exclusion_reason"] == "identity-in-train"
    assert manifest[4]["duplicate_clean_exclusion_reason"] == "identity-in-validation"
    assert manifest[5]["duplicate_clean_test"] is True
    assert "source_smiles" not in manifest[0]


def test_rdkit_parse_failure_is_not_silently_dropped(tmp_path):
    data = validate_qm9_csv(
        write_qm9_csv(tmp_path / "qm9.csv", ["C", "not a smiles"]), expected_rows=2
    )
    split = CandidateSplit(
        train=np.array([0]), validation=np.array([], dtype=np.int64), test=np.array([1])
    )
    with pytest.raises(CanonicalizationError, match="no row dropped"):
        audit_duplicates(data, split)
