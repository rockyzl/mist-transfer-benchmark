#!/usr/bin/env python3
"""Build deterministic, synthetic-only data for the static benchmark explorer.

This script deliberately uses the public baseline API and bundled fixtures. It
does not download data, load MIST, or make a network request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mist_transfer_benchmark import __version__
from mist_transfer_benchmark.baseline import run_ecfp_baselines
from mist_transfer_benchmark.fingerprints import FingerprintConfig
from mist_transfer_benchmark.schema import load_validated_csv
from mist_transfer_benchmark.splits import SplitConfig, make_split, split_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "site" / "demo-data.json"
MODEL_NAMES = ("dummy", "tanimoto_1nn", "ridge", "random_forest")

SPLIT_SPECS = {
    "random": {
        "fixture": "data/fixtures/redox_tiny_internal.csv",
        "strategy": "random",
        "title": "Grouped random",
        "description": (
            "Molecular identities stay together, but related families may span partitions."
        ),
    },
    "scaffold": {
        "fixture": "data/fixtures/redox_tiny_internal.csv",
        "strategy": "scaffold",
        "title": "Scaffold holdout",
        "description": "Bemis–Murcko scaffold groups move as indivisible units.",
    },
    "family": {
        "fixture": "data/fixtures/redox_tiny_internal.csv",
        "strategy": "group",
        "title": "Chemical-family holdout",
        "description": "Entire synthetic chemical-family labels are held out together.",
    },
    "external": {
        "fixture": "data/fixtures/redox_tiny.csv",
        "strategy": "external",
        "title": "External-source holdout",
        "description": "Rows marked as a separate synthetic source are test-only.",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_payload() -> dict[str, object]:
    fingerprint = FingerprintConfig(radius=2, n_bits=2048, include_chirality=True)
    seed = 42
    fixture_hashes = {
        spec["fixture"]: _sha256(REPO_ROOT / str(spec["fixture"]))
        for spec in SPLIT_SPECS.values()
    }
    provenance_core = {
        "benchmark_version": __version__,
        "seed": seed,
        "fingerprint": {
            "kind": "ECFP",
            "radius": fingerprint.radius,
            "n_bits": fingerprint.n_bits,
            "include_chirality": fingerprint.include_chirality,
        },
        "fixtures_sha256": fixture_hashes,
    }

    splits: dict[str, object] = {}
    for public_name, spec in SPLIT_SPECS.items():
        fixture_path = REPO_ROOT / str(spec["fixture"])
        frame, _ = load_validated_csv(fixture_path)
        config = SplitConfig(
            strategy=str(spec["strategy"]),
            seed=seed,
            group_column="chemical_family",
        )
        assignments = make_split(frame, config)
        result, predictions = run_ecfp_baselines(
            frame,
            assignments,
            MODEL_NAMES,
            fingerprint,
            seed,
        )

        test_mask = assignments == "test"
        test_frame = frame.loc[test_mask]
        first_model = predictions[
            (predictions["model"] == MODEL_NAMES[0]) & (predictions["split"] == "test")
        ]
        records = [
            {
                "record_id": prediction.record_id,
                "family": prediction.chemical_family,
                "target_v": round(float(prediction.target_v), 6),
                "max_train_tanimoto": round(float(prediction.max_train_tanimoto), 6),
                "nearest_train_record_id": prediction.nearest_train_record_id,
            }
            for prediction in first_model.itertuples(index=False)
        ]
        if len(records) != len(test_frame):
            raise RuntimeError("demo record alignment failed")

        models: dict[str, object] = {}
        for model_name in MODEL_NAMES:
            held_out = predictions[
                (predictions["model"] == model_name) & (predictions["split"] == "test")
            ]
            metric_values = result["metrics"][model_name]["test"]
            models[model_name] = {
                "metrics": {
                    key: round(float(value), 6) if isinstance(value, float) else value
                    for key, value in metric_values.items()
                },
                "predictions_v": [
                    round(float(row.prediction_v), 6)
                    for row in held_out.itertuples(index=False)
                ],
            }

        splits[public_name] = {
            "title": spec["title"],
            "description": spec["description"],
            "fixture": spec["fixture"],
            "counts": split_counts(assignments),
            "test_similarity": {
                key: round(float(value), 6) if isinstance(value, float) else value
                for key, value in result["similarity"]["test"].items()
            },
            "records": records,
            "models": models,
        }

    payload: dict[str, object] = {
        "schema_version": 1,
        "scientific_status": "synthetic-software-demo-only",
        "notice": (
            "All values are synthetic. These outputs test software behavior only and are not "
            "scientific benchmark results. MIST is not executed."
        ),
        "provenance": provenance_core,
        "model_labels": {
            "dummy": "Mean baseline",
            "tanimoto_1nn": "Tanimoto 1-NN",
            "ridge": "ECFP + ridge",
            "random_forest": "ECFP + random forest",
        },
        "splits": splits,
    }
    # The ID covers every displayed value and all embedded provenance. Compute it
    # before adding the ID itself so the operation is stable and reproducible.
    provenance_core["demo_run_id"] = _stable_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = build_payload()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
