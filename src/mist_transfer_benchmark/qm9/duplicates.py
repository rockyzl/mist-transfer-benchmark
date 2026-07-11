"""RDKit identity and duplicate/leakage audit for the candidate QM9 split."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from .data import ValidatedQM9
from .io import canonical_hash, canonical_json_bytes
from .split import CandidateSplit, decimal_lines_sha256


class CanonicalizationError(ValueError):
    """Raised instead of silently dropping an RDKit parse/canonicalization failure."""


@dataclass(frozen=True)
class DuplicateAudit:
    canonical_sha256_by_row: tuple[str, ...]
    duplicate_clean_test_by_row: tuple[bool, ...]
    exclusion_reason_by_row: tuple[str | None, ...]
    summary: dict[str, object]
    events: tuple[dict[str, object], ...]


def _within_split(groups: dict[str, list[int]], split_names: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        counts = [
            sum(split_names[index] == split for index in indices)
            for indices in groups.values()
        ]
        duplicate_counts = [count for count in counts if count > 1]
        result[split] = {
            "identity_count": len(duplicate_counts),
            "rows_in_duplicate_identities": sum(duplicate_counts),
            "extra_rows_beyond_one_per_identity": sum(count - 1 for count in duplicate_counts),
        }
    return result


def _cross_split(
    groups: dict[str, list[int]], split_names: np.ndarray
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlapping = []
        for indices in groups.values():
            counts = Counter(str(split_names[index]) for index in indices)
            if counts[left] and counts[right]:
                overlapping.append(counts)
        result[f"{left}_{right}"] = {
            "identity_count": len(overlapping),
            f"{left}_rows": sum(item[left] for item in overlapping),
            f"{right}_rows": sum(item[right] for item in overlapping),
        }
    return result


def audit_duplicates(
    data: ValidatedQM9,
    split: CandidateSplit,
    *,
    progress: Callable[[str], None] | None = None,
) -> DuplicateAudit:
    """Canonicalize every row, report duplicates, and derive the fixed clean test cohort."""

    split_names, _ = split.assignment_arrays(data.row_count)
    canonical_values: list[str] = []
    canonical_hashes: list[str] = []
    groups: dict[str, list[int]] = defaultdict(list)
    canonical_stream = hashlib.sha256()
    for index, smiles in enumerate(data.source_smiles):
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
        if molecule is None:
            raise CanonicalizationError(
                f"RDKit failed to parse source row {index} ({data.mol_ids[index]}); no row dropped"
            )
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if not canonical:
            raise CanonicalizationError(
                f"RDKit returned empty canonical SMILES for row {index}; no row dropped"
            )
        canonical_values.append(canonical)
        canonical_hashes.append(canonical_hash(canonical))
        groups[canonical].append(index)
        canonical_stream.update(canonical_json_bytes(canonical))
        canonical_stream.update(b"\n")
        if progress is not None and (index + 1) % 10_000 == 0:
            progress(f"canonicalized {index + 1:,}/{data.row_count:,} rows")

    train_keys = {canonical_values[index] for index in split.train}
    validation_keys = {canonical_values[index] for index in split.validation}
    eligible_test: dict[str, list[int]] = defaultdict(list)
    reasons: list[str | None] = [None] * data.row_count
    clean: list[bool] = [False] * data.row_count
    for index in split.test:
        key = canonical_values[int(index)]
        in_train = key in train_keys
        in_validation = key in validation_keys
        if in_train and in_validation:
            reasons[int(index)] = "identity-in-train-and-validation"
        elif in_train:
            reasons[int(index)] = "identity-in-train"
        elif in_validation:
            reasons[int(index)] = "identity-in-validation"
        else:
            eligible_test[key].append(int(index))
    for indices in eligible_test.values():
        retained = min(indices)
        clean[retained] = True
        for index in indices:
            if index != retained:
                reasons[index] = "higher-source-index-duplicate-within-test"

    events: list[dict[str, object]] = []
    for canonical, indices in groups.items():
        if len(indices) < 2:
            continue
        counts = Counter(str(split_names[index]) for index in indices)
        overlap_types = [
            f"{left}_{right}"
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
            if counts[left] and counts[right]
        ]
        events.append(
            {
                "canonical_identity_sha256": canonical_hash(canonical),
                "source_row_indices": indices,
                "mol_ids": [data.mol_ids[index] for index in indices],
                "split_counts": dict(sorted(counts.items())),
                "cross_split_overlap_types": overlap_types,
            }
        )
    events.sort(key=lambda item: str(item["canonical_identity_sha256"]))
    within = _within_split(groups, split_names)
    cross = _cross_split(groups, split_names)
    excluded_overlap = sum(
        reason is not None and reason.startswith("identity-in-")
        for reason in (reasons[int(index)] for index in split.test)
    )
    excluded_within = sum(
        reasons[int(index)] == "higher-source-index-duplicate-within-test" for index in split.test
    )
    retained_test_indices = [int(index) for index in split.test if clean[int(index)]]
    excluded_test_indices = [int(index) for index in split.test if not clean[int(index)]]
    summary: dict[str, object] = {
        "rdkit_parse_failures": 0,
        "rows_canonicalized": data.row_count,
        "canonical_smiles_sequence_sha256": canonical_stream.hexdigest(),
        "unique_canonical_identities": len(groups),
        "duplicate_identity_count": len(events),
        "rows_in_duplicate_identities": sum(
            len(indices) for indices in groups.values() if len(indices) > 1
        ),
        "within_split": within,
        "cross_split": cross,
        "train_test_overlap": cross["train_test"],
        "duplicate_clean_test": {
            "primary_test_rows": len(split.test),
            "retained_rows": sum(clean),
            "excluded_for_train_or_validation_identity": excluded_overlap,
            "excluded_as_higher_source_index_test_duplicate": excluded_within,
            "retained_ordered_index_sha256": decimal_lines_sha256(retained_test_indices),
            "excluded_ordered_index_sha256": decimal_lines_sha256(excluded_test_indices),
        },
        "identity_artifacts_contain_raw_or_canonical_smiles": False,
    }
    return DuplicateAudit(
        canonical_sha256_by_row=tuple(canonical_hashes),
        duplicate_clean_test_by_row=tuple(clean),
        exclusion_reason_by_row=tuple(reasons),
        summary=summary,
        events=tuple(events),
    )


def row_manifest(
    data: ValidatedQM9,
    split: CandidateSplit,
    audit: DuplicateAudit,
) -> Iterator[dict[str, object]]:
    """Yield a source-ordered manifest without raw/canonical SMILES or target values."""

    split_names, split_positions = split.assignment_arrays(data.row_count)
    for index in range(data.row_count):
        yield {
            "record_id": data.record_id(index),
            "source_row_index": index,
            "mol_id": data.mol_ids[index],
            "source_smiles_sha256": canonical_hash(data.source_smiles[index]),
            "canonical_identity_sha256": audit.canonical_sha256_by_row[index],
            "split": str(split_names[index]),
            "split_position": int(split_positions[index]),
            "duplicate_clean_test": audit.duplicate_clean_test_by_row[index],
            "duplicate_clean_exclusion_reason": audit.exclusion_reason_by_row[index],
        }
