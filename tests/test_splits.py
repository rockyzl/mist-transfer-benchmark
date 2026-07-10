import pandas as pd
import pytest

from mist_transfer_benchmark.splits import (
    SplitConfig,
    bemis_murcko_scaffold,
    make_split,
    split_counts,
    split_overlap_report,
)


def test_scaffold_split_is_deterministic_and_has_no_scaffold_overlap(redox_frame):
    config = SplitConfig(strategy="scaffold", seed=42)

    first = make_split(redox_frame, config)
    second = make_split(redox_frame, config)

    pd.testing.assert_series_equal(first, second)
    scaffolds = redox_frame["canonical_smiles"].map(bemis_murcko_scaffold)
    audit = pd.DataFrame({"scaffold": scaffolds, "split": first})
    assert audit.groupby("scaffold")["split"].nunique().max() == 1
    assert all(value > 0 for value in split_counts(first).values())


def test_random_split_keeps_duplicate_molecule_together(redox_frame):
    duplicate = redox_frame.iloc[[0]].copy()
    duplicate.loc[:, "record_id"] = "synthetic-replicate"
    combined = pd.concat([redox_frame, duplicate], ignore_index=True)

    assignments = make_split(combined, SplitConfig(strategy="random", seed=17))

    duplicated_molecule = combined["canonical_smiles"] == combined.loc[0, "canonical_smiles"]
    assert assignments.loc[duplicated_molecule].nunique() == 1


def test_group_split_keeps_chemical_family_together(redox_frame):
    assignments = make_split(
        redox_frame,
        SplitConfig(strategy="group", group_column="chemical_family", seed=42),
    )

    audit = pd.DataFrame({"family": redox_frame["chemical_family"], "split": assignments})
    assert audit.groupby("family")["split"].nunique().max() == 1


def test_external_split_reserves_external_rows_for_test(full_redox_frame):
    assignments = make_split(full_redox_frame, SplitConfig(strategy="external", seed=42))

    assert (assignments.loc[full_redox_frame["external_set"]] == "test").all()
    assert (assignments.loc[~full_redox_frame["external_set"]] != "test").all()


@pytest.mark.parametrize("strategy", ["random", "scaffold", "group"])
def test_non_external_strategies_reject_marked_external_rows(full_redox_frame, strategy):
    with pytest.raises(ValueError, match="external_set=true"):
        make_split(full_redox_frame, SplitConfig(strategy=strategy, seed=42))


@pytest.mark.parametrize("column", ["source_id", "group_id"])
def test_external_split_rejects_provenance_overlap(full_redox_frame, column):
    frame = full_redox_frame.copy()
    internal_value = frame.loc[~frame["external_set"], column].iloc[0]
    external_index = frame.index[frame["external_set"]][0]
    frame.loc[external_index, column] = internal_value

    with pytest.raises(ValueError, match=column):
        make_split(frame, SplitConfig(strategy="external", seed=42))


def test_group_split_rejects_not_reported_group(redox_frame):
    frame = redox_frame.copy()
    frame.loc[frame.index[0], "chemical_family"] = "not_reported"

    with pytest.raises(ValueError, match="not_reported"):
        make_split(frame, SplitConfig(strategy="group", group_column="chemical_family"))


def test_imbalanced_groups_still_produce_nonempty_splits():
    groups = ["large"] * 50 + ["medium"] * 40 + ["small"] * 10
    frame = pd.DataFrame(
        {
            "canonical_smiles": [f"molecule-{index}" for index in range(100)],
            "connectivity_key": [f"key-{index}" for index in range(100)],
            "chemical_family": groups,
            "external_set": False,
        }
    )

    assignments = make_split(
        frame,
        SplitConfig(strategy="group", group_column="chemical_family", seed=42),
    )

    assert sorted(split_counts(assignments).values()) == [10, 40, 50]


def test_impossible_three_way_group_split_fails():
    frame = pd.DataFrame(
        {
            "canonical_smiles": ["molecule-a", "molecule-b"],
            "connectivity_key": ["key-a", "key-b"],
            "chemical_family": ["family-a", "family-b"],
            "external_set": False,
        }
    )

    with pytest.raises(ValueError, match="at least 3 distinct groups"):
        make_split(frame, SplitConfig(strategy="group", group_column="chemical_family"))


def test_overlap_report_exposes_family_and_scaffold(full_redox_frame):
    assignments = make_split(full_redox_frame, SplitConfig(strategy="external", seed=42))

    report = split_overlap_report(full_redox_frame, assignments)

    assert set(report) == {"validation", "test"}
    assert set(report["test"]) == {"scaffold", "chemical_family"}
