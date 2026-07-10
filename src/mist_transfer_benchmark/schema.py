"""Strict, provenance-preserving redox data contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

DATA_CONTRACT_VERSION = "1.1"

REQUIRED_COLUMNS = (
    "record_id",
    "target_definition_id",
    "cohort_id",
    "redox_couple_id",
    "smiles",
    "model_input_role",
    "oxidized_smiles",
    "reduced_smiles",
    "inchi_key",
    "connectivity_key",
    "target_v",
    "target_v_original",
    "redox_direction",
    "potential_definition",
    "reference_electrode",
    "reference_electrode_original",
    "conversion_provenance",
    "n_electron",
    "n_proton",
    "oxidized_multiplicity",
    "reduced_multiplicity",
    "solvent",
    "supporting_electrolyte",
    "ph",
    "temperature_k",
    "protonation_state",
    "formal_charge",
    "measurement_method",
    "chemical_family",
    "group_id",
    "external_set",
    "source_type",
    "source_id",
    "source_url",
    "source_license",
)

REDOX_DIRECTIONS = {"oxidation", "reduction"}
POTENTIAL_DEFINITIONS = {
    "computed",
    "formal",
    "half_wave",
    "onset",
    "other",
    "peak_anodic",
    "peak_cathodic",
}
PROTONATION_STATES = {
    "deprotonated",
    "mixed",
    "neutral",
    "not_reported",
    "protonated",
    "zwitterionic",
}
SOURCE_TYPES = {"computation", "experiment", "synthetic"}
MISSING_MARKERS = {"not_applicable", "not_reported"}
MODEL_INPUT_ROLES = {"oxidized_state", "parent_connectivity", "reduced_state"}
TRUE_VALUES = {"1", "true"}
FALSE_VALUES = {"0", "false"}


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable contract violation or warning."""

    severity: str
    code: str
    message: str
    row: int | None = None
    column: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for one table."""

    row_count: int
    issues: tuple[ValidationIssue, ...]
    contract_version: str = DATA_CONTRACT_VERSION

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "row_count": self.row_count,
            "valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [asdict(issue) for issue in self.issues],
        }


class DataContractError(ValueError):
    """Raised when a redox table violates the contract."""

    def __init__(self, report: ValidationReport):
        self.report = report
        summary = "; ".join(issue.message for issue in report.errors[:3])
        super().__init__(
            f"redox data contract failed with {len(report.errors)} error(s): {summary}"
        )


def read_redox_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV without silently converting condition markers to missing values."""

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def canonicalize_smiles(smiles: str) -> str:
    """Return an isomeric canonical SMILES or raise for invalid input."""

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _issue(
    issues: list[ValidationIssue],
    severity: str,
    code: str,
    message: str,
    row: int | None = None,
    column: str | None = None,
) -> None:
    issues.append(ValidationIssue(severity, code, message, row, column))


def _require_clean_text(
    issues: list[ValidationIssue], value: str, column: str, csv_row: int
) -> bool:
    if not value:
        _issue(issues, "error", "empty_value", f"{column} must not be empty", csv_row, column)
        return False
    if value != value.strip():
        _issue(
            issues,
            "error",
            "surrounding_whitespace",
            f"{column} has surrounding whitespace",
            csv_row,
            column,
        )
        return False
    return True


def _parse_finite(
    value: str,
    column: str,
    csv_row: int,
    issues: list[ValidationIssue],
) -> float | None:
    try:
        number = float(value)
    except ValueError:
        _issue(issues, "error", "not_numeric", f"{column} must be numeric", csv_row, column)
        return None
    if not np.isfinite(number):
        _issue(issues, "error", "not_finite", f"{column} must be finite", csv_row, column)
        return None
    return number


def _parse_count_or_marker(
    value: str,
    column: str,
    csv_row: int,
    issues: list[ValidationIssue],
    *,
    minimum: int,
) -> None:
    if value == "not_reported":
        return
    try:
        number = int(value)
    except ValueError:
        _issue(
            issues,
            "error",
            "invalid_count",
            f"{column} must be an integer or not_reported",
            csv_row,
            column,
        )
        return
    if number < minimum:
        _issue(
            issues,
            "error",
            "invalid_count",
            f"{column} must be at least {minimum}",
            csv_row,
            column,
        )


def validate_redox_dataframe(frame: pd.DataFrame) -> ValidationReport:
    """Validate required fields, chemistry, conditions, and provenance."""

    issues: list[ValidationIssue] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    for column in missing:
        _issue(
            issues,
            "error",
            "missing_column",
            f"required column is missing: {column}",
            column=column,
        )
    if missing:
        return ValidationReport(len(frame), tuple(issues))

    extras = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
    for column in extras:
        _issue(
            issues,
            "warning",
            "extension_column",
            f"extension column is preserved but not validated: {column}",
            column=column,
        )

    if frame.empty:
        _issue(issues, "error", "empty_table", "the CSV contains no data rows")
        return ValidationReport(0, tuple(issues))

    duplicate_ids = frame["record_id"].duplicated(keep=False)
    for position in np.flatnonzero(duplicate_ids.to_numpy()):
        _issue(
            issues,
            "error",
            "duplicate_record_id",
            f"record_id is duplicated: {frame.iloc[position]['record_id']!r}",
            int(position) + 2,
            "record_id",
        )

    canonical_by_position: dict[int, str] = {}
    condition_keys: dict[tuple[str, ...], list[int]] = {}

    for position, (_, row) in enumerate(frame.iterrows()):
        csv_row = position + 2
        for column in REQUIRED_COLUMNS:
            _require_clean_text(issues, str(row[column]), column, csv_row)

        smiles = str(row["smiles"])
        molecule = None
        if smiles:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                _issue(
                    issues,
                    "error",
                    "invalid_smiles",
                    f"SMILES cannot be parsed: {smiles!r}",
                    csv_row,
                    "smiles",
                )
            else:
                canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
                canonical_by_position[position] = canonical

                expected_inchi_key = Chem.MolToInchiKey(molecule)
                supplied_inchi_key = str(row["inchi_key"])
                if supplied_inchi_key != expected_inchi_key:
                    _issue(
                        issues,
                        "error",
                        "inchi_key_mismatch",
                        "inchi_key does not match the model-input SMILES",
                        csv_row,
                        "inchi_key",
                    )
                expected_connectivity = expected_inchi_key.split("-")[0]
                if str(row["connectivity_key"]) != expected_connectivity:
                    _issue(
                        issues,
                        "error",
                        "connectivity_key_mismatch",
                        "connectivity_key must equal the first block of inchi_key",
                        csv_row,
                        "connectivity_key",
                    )

        role = str(row["model_input_role"])
        if role not in MODEL_INPUT_ROLES:
            _issue(
                issues,
                "error",
                "invalid_category",
                f"model_input_role must be one of {sorted(MODEL_INPUT_ROLES)}",
                csv_row,
                "model_input_role",
            )
        for state_column in ("oxidized_smiles", "reduced_smiles"):
            state_smiles = str(row[state_column])
            if state_smiles == "not_reported":
                continue
            with rdBase.BlockLogs():
                state_molecule = Chem.MolFromSmiles(state_smiles)
            if state_molecule is None:
                _issue(
                    issues,
                    "error",
                    "invalid_smiles",
                    f"{state_column} cannot be parsed or must be not_reported",
                    csv_row,
                    state_column,
                )

        target = _parse_finite(str(row["target_v"]), "target_v", csv_row, issues)
        original_target = _parse_finite(
            str(row["target_v_original"]), "target_v_original", csv_row, issues
        )

        converted = (
            target is not None
            and original_target is not None
            and (
                not np.isclose(target, original_target)
                or str(row["reference_electrode"])
                != str(row["reference_electrode_original"])
            )
        )
        conversion = str(row["conversion_provenance"])
        if converted and conversion in MISSING_MARKERS:
            _issue(
                issues,
                "error",
                "missing_conversion_provenance",
                "converted targets require explicit conversion_provenance",
                csv_row,
                "conversion_provenance",
            )
        if converted is False and conversion != "not_applicable":
            _issue(
                issues,
                "error",
                "unexpected_conversion_provenance",
                "unchanged target/reference must use conversion_provenance=not_applicable",
                csv_row,
                "conversion_provenance",
            )

        _parse_count_or_marker(
            str(row["n_electron"]), "n_electron", csv_row, issues, minimum=1
        )
        _parse_count_or_marker(str(row["n_proton"]), "n_proton", csv_row, issues, minimum=0)
        for multiplicity_column in ("oxidized_multiplicity", "reduced_multiplicity"):
            _parse_count_or_marker(
                str(row[multiplicity_column]),
                multiplicity_column,
                csv_row,
                issues,
                minimum=1,
            )

        direction = str(row["redox_direction"])
        if direction not in REDOX_DIRECTIONS:
            _issue(
                issues,
                "error",
                "invalid_category",
                f"redox_direction must be one of {sorted(REDOX_DIRECTIONS)}",
                csv_row,
                "redox_direction",
            )

        definition = str(row["potential_definition"])
        if definition not in POTENTIAL_DEFINITIONS:
            _issue(
                issues,
                "error",
                "invalid_category",
                f"potential_definition must be one of {sorted(POTENTIAL_DEFINITIONS)}",
                csv_row,
                "potential_definition",
            )

        ph = str(row["ph"])
        if ph not in MISSING_MARKERS:
            _parse_finite(ph, "ph", csv_row, issues)

        temperature = str(row["temperature_k"])
        if temperature not in MISSING_MARKERS:
            parsed_temperature = _parse_finite(temperature, "temperature_k", csv_row, issues)
            if parsed_temperature is not None and parsed_temperature <= 0:
                _issue(
                    issues,
                    "error",
                    "nonphysical_temperature",
                    "temperature_k must be greater than zero",
                    csv_row,
                    "temperature_k",
                )

        protonation = str(row["protonation_state"])
        if protonation not in PROTONATION_STATES and not protonation.startswith("other:"):
            _issue(
                issues,
                "error",
                "invalid_category",
                "protonation_state must use a controlled value or other:<description>",
                csv_row,
                "protonation_state",
            )

        charge_text = str(row["formal_charge"])
        try:
            charge = int(charge_text)
        except ValueError:
            _issue(
                issues,
                "error",
                "not_integer",
                "formal_charge must be an integer",
                csv_row,
                "formal_charge",
            )
        else:
            if molecule is not None:
                parsed_charge = Chem.GetFormalCharge(molecule)
                if charge != parsed_charge:
                    _issue(
                        issues,
                        "error",
                        "charge_mismatch",
                        f"formal_charge={charge} but the SMILES has net charge {parsed_charge}",
                        csv_row,
                        "formal_charge",
                    )

        external = str(row["external_set"]).lower()
        if external not in TRUE_VALUES | FALSE_VALUES:
            _issue(
                issues,
                "error",
                "invalid_boolean",
                "external_set must be true, false, 1, or 0",
                csv_row,
                "external_set",
            )

        source_type = str(row["source_type"])
        if source_type not in SOURCE_TYPES:
            _issue(
                issues,
                "error",
                "invalid_category",
                f"source_type must be one of {sorted(SOURCE_TYPES)}",
                csv_row,
                "source_type",
            )

        source_url = str(row["source_url"])
        if source_url:
            parsed_url = urlparse(source_url)
            if parsed_url.scheme not in {"doi", "http", "https", "synthetic"}:
                _issue(
                    issues,
                    "error",
                    "invalid_source_url",
                    "source_url must use doi, http, https, or synthetic scheme",
                    csv_row,
                    "source_url",
                )

        canonical = canonical_by_position.get(position)
        if canonical:
            key = (
                canonical,
                str(row["redox_couple_id"]),
                str(row["target_definition_id"]),
                str(row["redox_direction"]),
                str(row["potential_definition"]),
                str(row["reference_electrode"]),
                str(row["solvent"]),
                str(row["supporting_electrolyte"]),
                str(row["ph"]),
                str(row["temperature_k"]),
                str(row["protonation_state"]),
                str(row["formal_charge"]),
                str(row["measurement_method"]),
            )
            condition_keys.setdefault(key, []).append(csv_row)

    for rows in condition_keys.values():
        if len(rows) > 1:
            _issue(
                issues,
                "warning",
                "possible_replicate",
                f"rows {rows} share a molecule and measurement-condition key; "
                "preserve a replicate ID",
            )

    return ValidationReport(len(frame), tuple(issues))


def load_validated_csv(path: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Read and validate a redox CSV, raising on contract errors."""

    frame = read_redox_csv(path)
    report = validate_redox_dataframe(frame)
    if not report.is_valid:
        raise DataContractError(report)

    prepared = frame.copy()
    prepared["canonical_smiles"] = [canonicalize_smiles(value) for value in prepared["smiles"]]
    prepared["target_v"] = prepared["target_v"].astype(float)
    prepared["target_v_original"] = prepared["target_v_original"].astype(float)
    prepared["formal_charge"] = prepared["formal_charge"].astype(int)
    prepared["external_set"] = prepared["external_set"].str.lower().isin(TRUE_VALUES)
    return prepared, report
