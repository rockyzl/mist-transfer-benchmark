import pandas as pd

from mist_transfer_benchmark.schema import (
    REQUIRED_COLUMNS,
    load_validated_csv,
    read_redox_csv,
    validate_redox_dataframe,
)

from .conftest import FIXTURE_CSV


def test_synthetic_fixture_satisfies_contract():
    frame, report = load_validated_csv(FIXTURE_CSV)

    assert report.is_valid
    assert not report.warnings
    assert len(frame) == 24
    assert set(REQUIRED_COLUMNS).issubset(frame.columns)
    assert "canonical_smiles" in frame.columns


def test_missing_required_column_is_rejected():
    frame = read_redox_csv(FIXTURE_CSV).drop(columns=["reference_electrode"])

    report = validate_redox_dataframe(frame)

    assert not report.is_valid
    assert any(issue.code == "missing_column" for issue in report.errors)


def test_formal_charge_must_match_smiles():
    frame = read_redox_csv(FIXTURE_CSV).iloc[:1].copy()
    frame.loc[:, "formal_charge"] = "1"

    report = validate_redox_dataframe(frame)

    assert any(issue.code == "charge_mismatch" for issue in report.errors)


def test_extension_column_is_preserved_with_warning():
    frame = read_redox_csv(FIXTURE_CSV).iloc[:1].copy()
    frame["replicate_id"] = "replicate-a"

    report = validate_redox_dataframe(frame)

    assert report.is_valid
    assert any(issue.code == "extension_column" for issue in report.warnings)


def test_duplicate_record_id_is_rejected():
    frame = read_redox_csv(FIXTURE_CSV).iloc[:2].copy()
    frame.loc[frame.index[1], "record_id"] = frame.iloc[0]["record_id"]

    report = validate_redox_dataframe(frame)

    assert any(issue.code == "duplicate_record_id" for issue in report.errors)


def test_read_preserves_missing_markers_as_strings():
    frame = pd.read_csv(FIXTURE_CSV, dtype=str, keep_default_na=False)

    assert frame.loc[0, "ph"] == "not_applicable"


def test_inchi_key_must_match_model_input_smiles():
    frame = read_redox_csv(FIXTURE_CSV).iloc[:1].copy()
    frame.loc[:, "inchi_key"] = "AAAAAAAAAAAAAA-BBBBBBBBSA-C"

    report = validate_redox_dataframe(frame)

    assert any(issue.code == "inchi_key_mismatch" for issue in report.errors)


def test_converted_target_requires_provenance():
    frame = read_redox_csv(FIXTURE_CSV).iloc[:1].copy()
    frame.loc[:, "target_v"] = "0.10"
    frame.loc[:, "conversion_provenance"] = "not_reported"

    report = validate_redox_dataframe(frame)

    assert any(issue.code == "missing_conversion_provenance" for issue in report.errors)
