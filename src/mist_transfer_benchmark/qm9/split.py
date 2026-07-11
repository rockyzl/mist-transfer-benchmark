"""Candidate split reconstruction and independent Datasets-reference comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np

from .constants import (
    EXPECTED_ROW_COUNT,
    EXPECTED_SPLIT_COUNTS,
    FIRST_TEST_SIZE,
    SECOND_TEST_SIZE,
    SPLIT_SEED,
)


class SplitMismatchError(ValueError):
    """Raised when local split membership differs from Datasets 3.2.0."""


@dataclass(frozen=True)
class ReferenceContract:
    row_count: int
    seed: int
    counts: dict[str, int]
    python: str
    datasets: str
    numpy: str
    pyarrow: str
    pandas: str
    fsspec: str
    environment_freeze_canonical_json_sha256: str
    train_test_split_source_sha256: str
    reference_source_sha256: str
    executable: str
    prefix: str
    max_json_bytes: int


def decimal_lines_sha256(indices: np.ndarray | list[int], *, sort: bool = False) -> str:
    """Hash zero-based decimal indices, one per line with a final LF."""

    values = np.asarray(indices, dtype=np.int64)
    if sort:
        values = np.sort(values)
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def ordered_hashes(self) -> dict[str, str]:
        return {
            name: decimal_lines_sha256(getattr(self, name))
            for name in ("train", "validation", "test")
        }

    def membership_hashes(self) -> dict[str, str]:
        return {
            name: decimal_lines_sha256(getattr(self, name), sort=True)
            for name in ("train", "validation", "test")
        }

    def assignment_arrays(self, row_count: int) -> tuple[np.ndarray, np.ndarray]:
        names = np.empty(row_count, dtype=object)
        positions = np.empty(row_count, dtype=np.int64)
        for name in ("train", "validation", "test"):
            indices = getattr(self, name)
            names[indices] = name
            positions[indices] = np.arange(len(indices), dtype=np.int64)
        return names, positions


def reconstruct_candidate_split(
    row_count: int = EXPECTED_ROW_COUNT,
    *,
    seed: int = SPLIT_SEED,
) -> CandidateSplit:
    """Reproduce Datasets 3.2.0's two independent seeded train_test_split calls."""

    first_test_count = ceil(FIRST_TEST_SIZE * row_count)
    first_permutation = np.random.default_rng(seed).permutation(row_count)
    first_test = first_permutation[:first_test_count]
    train = first_permutation[first_test_count:]

    final_test_count = ceil(SECOND_TEST_SIZE * len(first_test))
    second_permutation = np.random.default_rng(seed).permutation(len(first_test))
    test = first_test[second_permutation[:final_test_count]]
    validation = first_test[second_permutation[final_test_count:]]
    result = CandidateSplit(train=train, validation=validation, test=test)
    all_indices = np.concatenate((train, validation, test))
    if len(all_indices) != row_count or not np.array_equal(
        np.sort(all_indices), np.arange(row_count, dtype=np.int64)
    ):
        raise SplitMismatchError("candidate split is not a disjoint cover of source rows")
    if row_count == EXPECTED_ROW_COUNT and result.counts() != EXPECTED_SPLIT_COUNTS:
        raise SplitMismatchError(
            f"candidate counts {result.counts()} differ from {EXPECTED_SPLIT_COUNTS}"
        )
    return result


def load_and_verify_datasets_reference(
    path: str | Path,
    local: CandidateSplit,
    *,
    contract: ReferenceContract,
) -> dict[str, object]:
    """Require exact ordered membership equality with an isolated Datasets reference."""

    reference_path = Path(path)
    if reference_path.stat().st_size > contract.max_json_bytes:
        raise SplitMismatchError("Datasets reference exceeds the configured JSON size limit")
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_top_level = {
        "kind",
        "schema_version",
        "row_count",
        "seed",
        "contract",
        "reference_source_sha256",
        "index_hash_serialization",
        "environment",
        "counts",
        "ordered_sha256",
        "membership_sha256",
        "ordered_indices",
    }
    if set(payload) != expected_top_level:
        raise SplitMismatchError("Datasets reference top-level schema is not exact")
    if payload["kind"] != "mist-transfer-benchmark-datasets-reference":
        raise SplitMismatchError("Datasets reference kind is invalid")
    if payload["schema_version"] != "datasets-reference-v1":
        raise SplitMismatchError("Datasets reference schema version is invalid")
    if type(payload["row_count"]) is not int or payload["row_count"] != contract.row_count:
        raise SplitMismatchError("Datasets reference row count is invalid")
    if type(payload["seed"]) is not int or payload["seed"] != contract.seed:
        raise SplitMismatchError("Datasets reference seed is invalid")
    if payload["reference_source_sha256"] != contract.reference_source_sha256:
        raise SplitMismatchError("Datasets reference implementation SHA-256 is invalid")
    if payload["index_hash_serialization"] != (
        "zero-based decimal integer per line, UTF-8, final LF"
    ):
        raise SplitMismatchError("Datasets reference index serialization is invalid")
    expected_split_contract = {
        "function": "Dataset.train_test_split",
        "first_test_size": FIRST_TEST_SIZE,
        "second_test_size": SECOND_TEST_SIZE,
        "shuffle": True,
        "independent_rng_with_same_seed_for_each_call": True,
    }
    if payload["contract"] != expected_split_contract:
        raise SplitMismatchError("Datasets reference split contract is invalid")
    if payload["counts"] != contract.counts or payload["counts"] != local.counts():
        raise SplitMismatchError("Datasets reference counts are invalid")

    environment = payload["environment"]
    expected_environment_keys = {
        "python",
        "executable",
        "prefix",
        "datasets",
        "numpy",
        "pyarrow",
        "pandas",
        "fsspec",
        "bit_generator",
        "environment_freeze",
        "environment_freeze_canonical_json_sha256",
        "train_test_split_source_sha256",
        "resource_usage",
    }
    if set(environment) != expected_environment_keys:
        raise SplitMismatchError("Datasets reference environment schema is not exact")
    expected_versions = {
        "python": contract.python,
        "datasets": contract.datasets,
        "numpy": contract.numpy,
        "pyarrow": contract.pyarrow,
        "pandas": contract.pandas,
        "fsspec": contract.fsspec,
        "bit_generator": "PCG64",
        "executable": contract.executable,
        "prefix": contract.prefix,
        "train_test_split_source_sha256": contract.train_test_split_source_sha256,
    }
    for key, expected in expected_versions.items():
        if environment[key] != expected:
            raise SplitMismatchError(f"Datasets reference environment {key} is invalid")
    freeze = environment["environment_freeze"]
    if not isinstance(freeze, list) or not all(isinstance(item, str) for item in freeze):
        raise SplitMismatchError("Datasets reference environment freeze is invalid")
    if freeze != sorted(freeze) or len(freeze) != len(set(freeze)):
        raise SplitMismatchError("Datasets reference environment freeze is not sorted and unique")
    freeze_hash = hashlib.sha256(
        json.dumps(
            freeze,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if freeze_hash != environment["environment_freeze_canonical_json_sha256"]:
        raise SplitMismatchError("Datasets reference freeze hash is internally inconsistent")
    if freeze_hash != contract.environment_freeze_canonical_json_sha256:
        raise SplitMismatchError("Datasets reference freeze differs from the approved environment")
    required_freeze = {
        f"datasets=={contract.datasets}",
        f"numpy=={contract.numpy}",
        f"pyarrow=={contract.pyarrow}",
        f"pandas=={contract.pandas}",
        f"fsspec=={contract.fsspec}",
    }
    normalized_freeze = {item.lower() for item in freeze}
    if not {item.lower() for item in required_freeze} <= normalized_freeze:
        raise SplitMismatchError("Datasets reference freeze lacks a required direct package")
    resource_usage = environment["resource_usage"]
    if set(resource_usage) != {"method", "semantics", "peak_rss_gib"}:
        raise SplitMismatchError("Datasets reference resource schema is invalid")
    if resource_usage["method"] != "resource.getrusage(RUSAGE_SELF).ru_maxrss":
        raise SplitMismatchError("Datasets reference resource method is invalid")
    if type(resource_usage["peak_rss_gib"]) not in {int, float} or not (
        0 < resource_usage["peak_rss_gib"] < 1024
    ):
        raise SplitMismatchError("Datasets reference peak RSS is invalid")

    split_keys = {"train", "validation", "test"}
    for field in ("counts", "ordered_sha256", "membership_sha256", "ordered_indices"):
        if set(payload[field]) != split_keys:
            raise SplitMismatchError(f"Datasets reference {field} keys are invalid")
    seen: set[int] = set()
    for name in ("train", "validation", "test"):
        raw_indices = payload["ordered_indices"][name]
        if not isinstance(raw_indices, list) or not all(type(item) is int for item in raw_indices):
            raise SplitMismatchError(f"{name} indices must be non-bool JSON integers")
        if len(raw_indices) != contract.counts[name]:
            raise SplitMismatchError(f"{name} reference length is invalid")
        if any(item < 0 or item >= contract.row_count for item in raw_indices):
            raise SplitMismatchError(f"{name} reference contains an out-of-range index")
        if len(set(raw_indices)) != len(raw_indices):
            raise SplitMismatchError(f"{name} reference contains duplicate indices")
        overlap = seen.intersection(raw_indices)
        if overlap:
            raise SplitMismatchError(f"{name} reference overlaps another split")
        seen.update(raw_indices)
        reference = np.asarray(raw_indices, dtype=np.int64)
        observed = getattr(local, name)
        ordered_hash = decimal_lines_sha256(reference)
        membership_hash = decimal_lines_sha256(reference, sort=True)
        if payload["ordered_sha256"][name] != ordered_hash:
            raise SplitMismatchError(f"{name} reference ordered hash is internally inconsistent")
        if payload["membership_sha256"][name] != membership_hash:
            raise SplitMismatchError(
                f"{name} reference membership hash is internally inconsistent"
            )
        if not np.array_equal(reference, observed):
            mismatch = (
                int(np.flatnonzero(reference != observed)[0])
                if len(reference) == len(observed)
                else None
            )
            raise SplitMismatchError(
                f"{name} membership/order differs from Datasets reference; "
                f"reference={len(reference)} local={len(observed)} first_mismatch={mismatch}"
            )
    if seen != set(range(contract.row_count)):
        raise SplitMismatchError("Datasets reference is not a complete row cover")
    return payload
