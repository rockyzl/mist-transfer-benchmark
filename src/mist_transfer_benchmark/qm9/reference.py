"""Generate exact split membership with an isolated datasets==3.2.0 environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import platform
import resource
import sys
from pathlib import Path

import datasets
import numpy as np
from datasets import Dataset

from .constants import EXPECTED_ROW_COUNT, FIRST_TEST_SIZE, SECOND_TEST_SIZE, SPLIT_SEED
from .io import atomic_write_json, canonical_hash
from .split import decimal_lines_sha256


def datasets_reference(row_count: int, seed: int) -> dict[str, object]:
    if datasets.__version__ != "3.2.0":
        raise RuntimeError(f"datasets==3.2.0 required; found {datasets.__version__}")
    if np.__version__ != "2.5.1":
        raise RuntimeError(f"numpy==2.5.1 required; found {np.__version__}")
    source = Dataset.from_dict({"source_row_index": list(range(row_count))})
    train_other = source.train_test_split(test_size=0.2, seed=seed)
    validation_test = train_other["test"].train_test_split(test_size=0.5, seed=seed)
    ordered = {
        "train": train_other["train"]["source_row_index"],
        "validation": validation_test["train"]["source_row_index"],
        "test": validation_test["test"]["source_row_index"],
    }
    freeze = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    method_source = inspect.getsource(Dataset.train_test_split).encode("utf-8")
    reference_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rss_divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
    return {
        "kind": "mist-transfer-benchmark-datasets-reference",
        "schema_version": "datasets-reference-v1",
        "row_count": row_count,
        "seed": seed,
        "contract": {
            "function": "Dataset.train_test_split",
            "first_test_size": FIRST_TEST_SIZE,
            "second_test_size": SECOND_TEST_SIZE,
            "shuffle": True,
            "independent_rng_with_same_seed_for_each_call": True,
        },
        "reference_source_sha256": reference_source_sha256,
        "index_hash_serialization": "zero-based decimal integer per line, UTF-8, final LF",
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "datasets": datasets.__version__,
            "numpy": np.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "pandas": importlib.metadata.version("pandas"),
            "fsspec": importlib.metadata.version("fsspec"),
            "bit_generator": np.random.default_rng(seed).bit_generator.__class__.__name__,
            "environment_freeze": freeze,
            "environment_freeze_canonical_json_sha256": canonical_hash(freeze),
            "train_test_split_source_sha256": hashlib.sha256(method_source).hexdigest(),
            "resource_usage": {
                "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "semantics": "isolated-reference-process peak RSS, not process-group RSS",
                "peak_rss_gib": peak_rss,
            },
        },
        "counts": {name: len(values) for name, values in ordered.items()},
        "ordered_sha256": {
            name: decimal_lines_sha256(values) for name, values in ordered.items()
        },
        "membership_sha256": {
            name: decimal_lines_sha256(values, sort=True) for name, values in ordered.items()
        },
        "ordered_indices": ordered,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=EXPECTED_ROW_COUNT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    atomic_write_json(args.output, datasets_reference(args.rows, args.seed), mode=0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
