from __future__ import annotations

import copy

import pytest

from mist_transfer_benchmark.qm9.io import atomic_write_json, canonical_hash
from mist_transfer_benchmark.qm9.split import (
    ReferenceContract,
    SplitMismatchError,
    load_and_verify_datasets_reference,
    reconstruct_candidate_split,
)


def _authenticated_payload(split, row_count):
    freeze = sorted(
        [
            "datasets==3.2.0",
            "fsspec==2024.9.0",
            "numpy==2.5.1",
            "pandas==3.0.3",
            "pyarrow==25.0.0",
        ]
    )
    payload = {
        "kind": "mist-transfer-benchmark-datasets-reference",
        "schema_version": "datasets-reference-v1",
        "row_count": row_count,
        "seed": 42,
        "contract": {
            "function": "Dataset.train_test_split",
            "first_test_size": 0.2,
            "second_test_size": 0.5,
            "shuffle": True,
            "independent_rng_with_same_seed_for_each_call": True,
        },
        "reference_source_sha256": "a" * 64,
        "index_hash_serialization": "zero-based decimal integer per line, UTF-8, final LF",
        "environment": {
            "python": "3.12.12",
            "executable": "/tmp/reference/bin/python",
            "prefix": "/tmp/reference",
            "datasets": "3.2.0",
            "numpy": "2.5.1",
            "pyarrow": "25.0.0",
            "pandas": "3.0.3",
            "fsspec": "2024.9.0",
            "bit_generator": "PCG64",
            "environment_freeze": freeze,
            "environment_freeze_canonical_json_sha256": canonical_hash(freeze),
            "train_test_split_source_sha256": "b" * 64,
            "resource_usage": {
                "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "semantics": "isolated-reference-process peak RSS, not process-group RSS",
                "peak_rss_gib": 0.25,
            },
        },
        "counts": split.counts(),
        "ordered_sha256": split.ordered_hashes(),
        "membership_sha256": split.membership_hashes(),
        "ordered_indices": {
            name: getattr(split, name).tolist() for name in ("train", "validation", "test")
        },
    }
    contract = ReferenceContract(
        row_count=row_count,
        seed=42,
        counts=split.counts(),
        python="3.12.12",
        datasets="3.2.0",
        numpy="2.5.1",
        pyarrow="25.0.0",
        pandas="3.0.3",
        fsspec="2024.9.0",
        environment_freeze_canonical_json_sha256=canonical_hash(freeze),
        train_test_split_source_sha256="b" * 64,
        reference_source_sha256="a" * 64,
        executable="/tmp/reference/bin/python",
        prefix="/tmp/reference",
        max_json_bytes=100_000,
    )
    return payload, contract


def test_full_candidate_split_counts_and_frozen_hashes():
    split = reconstruct_candidate_split()

    assert split.counts() == {"train": 107108, "validation": 13388, "test": 13389}
    assert split.ordered_hashes() == {
        "train": "b523014c59c4845a7c89b150e85ddc1a53804dbcc3b328246d32a8db6aa727f0",
        "validation": "c0692f48fd5085e8764bdecee67bd44f8cc2b4cdf9454ec8ff5be5f5e2219084",
        "test": "736daeb02a8420eb58643e0c7f0b56f958f279fe29e818c62ffcb07700ea5a73",
    }


def test_reference_comparison_requires_exact_ordered_membership(tmp_path):
    split = reconstruct_candidate_split(row_count=20)
    payload, contract = _authenticated_payload(split, 20)
    path = tmp_path / "reference.json"
    atomic_write_json(path, payload)
    verified = load_and_verify_datasets_reference(path, split, contract=contract)
    assert verified["environment"]["datasets"] == "3.2.0"

    changed = copy.deepcopy(payload)
    changed["ordered_indices"]["train"][0], changed["ordered_indices"]["train"][1] = (
        changed["ordered_indices"]["train"][1],
        changed["ordered_indices"]["train"][0],
    )
    atomic_write_json(path, changed)
    with pytest.raises(SplitMismatchError):
        load_and_verify_datasets_reference(path, split, contract=contract)


@pytest.mark.parametrize("mutation", ["schema", "seed", "freeze", "bool-index", "source"])
def test_reference_rejects_forged_or_incomplete_evidence(tmp_path, mutation):
    split = reconstruct_candidate_split(row_count=20)
    payload, contract = _authenticated_payload(split, 20)
    if mutation == "schema":
        payload.pop("membership_sha256")
    elif mutation == "seed":
        payload["seed"] = 7
    elif mutation == "freeze":
        payload["environment"]["environment_freeze_canonical_json_sha256"] = "0" * 64
    elif mutation == "bool-index":
        payload["ordered_indices"]["train"][0] = True
    else:
        payload["reference_source_sha256"] = "0" * 64
    path = tmp_path / f"{mutation}.json"
    atomic_write_json(path, payload)
    with pytest.raises(SplitMismatchError):
        load_and_verify_datasets_reference(path, split, contract=contract)
