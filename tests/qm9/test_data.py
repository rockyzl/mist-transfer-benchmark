from __future__ import annotations

import csv

import pytest

from mist_transfer_benchmark.qm9.constants import EXPECTED_HEADER
from mist_transfer_benchmark.qm9.data import QM9DataError, validate_qm9_csv

from .conftest import write_qm9_csv


def test_strict_qm9_validation_and_stable_hashes(tmp_path):
    path = write_qm9_csv(tmp_path / "qm9.csv", ["C", "CC", "O"])

    first = validate_qm9_csv(path, expected_rows=3)
    second = validate_qm9_csv(path, expected_rows=3)

    assert first.header == EXPECTED_HEADER
    assert first.mol_ids == ("gdb_1", "gdb_2", "gdb_3")
    assert first.row_identity_sha256 == second.row_identity_sha256
    assert len(first.raw_smiles_sha256) == 64
    assert first.record_id(1) == "qm9:000001:gdb_2"


def test_header_order_and_nonfinite_targets_are_hard_failures(tmp_path):
    wrong_header = write_qm9_csv(tmp_path / "wrong.csv", ["C"])
    rows = list(csv.reader(wrong_header.open(encoding="utf-8", newline="")))
    rows[0][0], rows[0][1] = rows[0][1], rows[0][0]
    with wrong_header.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    with pytest.raises(QM9DataError, match="header/order mismatch"):
        validate_qm9_csv(wrong_header, expected_rows=1)

    nonfinite = write_qm9_csv(tmp_path / "nonfinite.csv", ["C"])
    rows = list(csv.reader(nonfinite.open(encoding="utf-8", newline="")))
    rows[1][EXPECTED_HEADER.index("mu")] = "nan"
    with nonfinite.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    with pytest.raises(QM9DataError, match="not finite"):
        validate_qm9_csv(nonfinite, expected_rows=1)
