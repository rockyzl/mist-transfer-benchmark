"""Comparability gates for molecular-only property baselines."""

from __future__ import annotations

import pandas as pd

COMPARABILITY_COLUMNS = (
    "target_definition_id",
    "cohort_id",
    "source_type",
    "redox_direction",
    "potential_definition",
    "reference_electrode",
    "solvent",
    "supporting_electrolyte",
    "ph",
    "temperature_k",
    "protonation_state",
    "formal_charge",
    "n_electron",
    "n_proton",
    "model_input_role",
    "oxidized_multiplicity",
    "reduced_multiplicity",
    "measurement_method",
)


def comparability_report(frame: pd.DataFrame) -> dict[str, object]:
    """Describe heterogeneous target/condition fields ignored by molecular-only models."""

    heterogeneous: dict[str, list[str]] = {}
    for column in COMPARABILITY_COLUMNS:
        values = sorted(frame[column].astype(str).unique().tolist())
        if len(values) > 1:
            heterogeneous[column] = values[:100]
    return {
        "comparable_for_condition_ignorant_model": not heterogeneous,
        "checked_columns": list(COMPARABILITY_COLUMNS),
        "heterogeneous_values": heterogeneous,
    }


def enforce_comparable_cohort(
    frame: pd.DataFrame,
    *,
    unsafe_allow_condition_ignorant_mixing: bool = False,
) -> dict[str, object]:
    """Reject heterogeneous targets unless the explicit unsafe override is supplied."""

    report = comparability_report(frame)
    source_types = sorted(frame["source_type"].astype(str).unique().tolist())
    if len(source_types) != 1:
        raise ValueError(
            "source_type mixing is forbidden for molecular-only baselines, including with the "
            f"unsafe override; found {source_types}"
        )
    heterogeneous = report["heterogeneous_values"]
    if heterogeneous and not unsafe_allow_condition_ignorant_mixing:
        fields = ", ".join(sorted(heterogeneous))
        raise ValueError(
            "molecular-only baseline requires one comparable target cohort; heterogeneous fields: "
            f"{fields}. Filter the CSV or deliberately pass "
            "--unsafe-allow-condition-ignorant-mixing"
        )
    report["unsafe_override_used"] = bool(
        heterogeneous and unsafe_allow_condition_ignorant_mixing
    )
    return report
