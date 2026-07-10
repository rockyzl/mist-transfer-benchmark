import pytest

from mist_transfer_benchmark.cohort import enforce_comparable_cohort


def test_comparability_gate_rejects_heterogeneous_measurement_method(redox_frame):
    frame = redox_frame.copy()
    frame.loc[frame.index[-1], "measurement_method"] = "different_protocol"

    with pytest.raises(ValueError, match="measurement_method"):
        enforce_comparable_cohort(frame)


@pytest.mark.parametrize(
    ("column", "different_value"),
    [
        ("supporting_electrolyte", "different_salt_1M"),
        ("ph", "7.0"),
        ("temperature_k", "313.15"),
    ],
)
def test_comparability_gate_rejects_ignored_electrochemical_conditions(
    redox_frame,
    column,
    different_value,
):
    frame = redox_frame.copy()
    frame.loc[frame.index[-1], column] = different_value

    with pytest.raises(ValueError, match=column):
        enforce_comparable_cohort(frame)


def test_explicit_unsafe_override_is_reported(redox_frame):
    frame = redox_frame.copy()
    frame.loc[frame.index[-1], "solvent"] = "different_solvent"

    report = enforce_comparable_cohort(
        frame,
        unsafe_allow_condition_ignorant_mixing=True,
    )

    assert report["unsafe_override_used"] is True
    assert "solvent" in report["heterogeneous_values"]


def test_source_type_mixing_is_never_allowed(redox_frame):
    frame = redox_frame.copy()
    frame.loc[frame.index[-1], "source_type"] = "experiment"

    with pytest.raises(ValueError, match="source_type mixing"):
        enforce_comparable_cohort(
            frame,
            unsafe_allow_condition_ignorant_mixing=True,
        )
