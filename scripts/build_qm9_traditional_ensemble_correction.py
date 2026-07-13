#!/usr/bin/env python3
"""Build the post-specified traditional-only ensemble without test-based selection.

The original extended run blended MIST with the task-specific models.  That is
useful as a second-layer systems result, but it does not answer the cleaner
scientific question: how does an independently fine-tuned foundation model
compare with a traditional modeling pipeline?  This correction reconstructs
the frozen engineered features and models, selects Ridge/XGBoost/MLP blend
weights on validation only, and then reports the already established test
cohorts.  Because the reporting question changed after the first test report,
the output is explicitly labeled post-specified.
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import numpy as np
from run_qm9_extended_comparison import _feature_candidates, _select_ensemble_weights
from scipy import sparse
from sklearn.linear_model import Ridge

from mist_transfer_benchmark.live_demo import _make_mlp, _predict_mlp
from mist_transfer_benchmark.qm9.constants import TARGET_COLUMNS
from mist_transfer_benchmark.qm9.data import load_qm9_identities
from mist_transfer_benchmark.qm9.io import atomic_write_json, sha256_file
from mist_transfer_benchmark.qm9.phase2_contract import verify_phase1_evidence
from mist_transfer_benchmark.qm9.phase2_metrics import native_metrics
from mist_transfer_benchmark.qm9.phase2_targets import load_targets_for_indices


def run(output_dir: Path) -> dict:
    repo = Path(__file__).resolve().parents[1]
    output_dir.mkdir(parents=True, exist_ok=False)
    private_dir = repo / "data/private/qm9/traditional-ensemble-correction-v1"
    private_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    source = repo / "data/private/qm9/qm9.csv"
    with (repo / "configs/qm9_28m.toml").open("rb") as handle:
        protocol = tomllib.load(handle)
    evidence = verify_phase1_evidence(protocol, repo / "results/qm9-phase1-v2", source)
    data = load_qm9_identities(source)
    binary = sparse.load_npz(repo / "results/qm9-phase2-classical-v1/feature_matrix.npz")
    y_train = load_targets_for_indices(source, evidence.split.train, data)
    y_validation = load_targets_for_indices(source, evidence.split.validation, data)
    train_mean = y_train.mean(axis=0)
    train_scale = y_train.std(axis=0, ddof=0)

    feature_sets, descriptor_names = _feature_candidates(
        binary, list(data.source_smiles), evidence.split.train
    )
    selected_features = "count_ecfp_plus_globals"
    features = feature_sets[selected_features]
    x_train = features[evidence.split.train]
    x_validation = features[evidence.split.validation]

    ridge = Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=10_000)
    ridge.fit(x_train, (y_train - train_mean) / train_scale)
    ridge_validation = ridge.predict(x_validation) * train_scale + train_mean

    frozen = repo / "data/private/qm9/extended-comparison-v1"
    selection = json.loads((frozen / "selection_freeze.json").read_text())
    from xgboost import XGBRegressor

    xgb_models = []
    for target in TARGET_COLUMNS:
        model = XGBRegressor()
        model.load_model(frozen / f"xgboost_{target}.json")
        xgb_models.append(model)
    xgb_validation = np.column_stack(
        [model.predict(x_validation) for model in xgb_models]
    )

    import torch
    from safetensors.torch import load_file

    mlp_params = selection["mlp_selected"]["parameters"]
    mlp_device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = _make_mlp(
        list(mlp_params["hidden_dims"]),
        float(mlp_params["dropout"]),
        input_dim=features.shape[1],
    )
    mlp.load_state_dict(load_file(frozen / "mlp.safetensors"))
    mlp.to(mlp_device)
    scaler = np.load(frozen / "mlp_scaler.npz")
    mlp_validation = _predict_mlp(
        mlp,
        x_validation,
        np.arange(len(y_validation), dtype=np.int64),
        scaler["mean"],
        scaler["scale"],
        512,
        mlp_device,
    )

    traditional_selection = _select_ensemble_weights(
        {
            "engineered_ridge": ridge_validation,
            "xgboost": xgb_validation,
            "mlp": mlp_validation,
        },
        y_validation,
        train_scale,
    )
    print(
        "traditional-only validation score="
        f"{traditional_selection['validation_mean_normalized_mae']:.8f}",
        flush=True,
    )
    print(f"weights={traditional_selection['weights']}", flush=True)

    # No choice below this point uses test labels.  The test cohort was already
    # reported in v1; this is therefore a post-specified analysis, not a new
    # pristine one-shot experiment.
    y_test = load_targets_for_indices(source, evidence.split.test, data)
    x_test = features[evidence.split.test]
    ridge_test = ridge.predict(x_test) * train_scale + train_mean
    stored = np.load(frozen / "test_predictions.npz")
    xgb_test = stored["xgboost"]
    mlp_test = stored["mlp"]
    traditional_test = sum(
        traditional_selection["weights"][name] * values
        for name, values in {
            "engineered_ridge": ridge_test,
            "xgboost": xgb_test,
            "mlp": mlp_test,
        }.items()
    )

    clean_rows = {int(index) for index in evidence.duplicate_clean_test}
    clean_mask = np.asarray(
        [int(index) in clean_rows for index in evidence.split.test], dtype=bool
    )
    old = json.loads(
        (repo / "results/qm9-extended-comparison-v1/aggregate_metrics.json").read_text()
    )

    def score(prediction: np.ndarray) -> dict:
        return {
            "full_test": native_metrics(y_test, prediction, train_scale),
            "duplicate_clean_test": native_metrics(
                y_test[clean_mask], prediction[clean_mask], train_scale
            ),
        }

    methods = {
        "traditional_ensemble": score(traditional_test),
        "xgboost": old["methods"]["xgboost"],
        "mist": old["methods"]["mist"],
        "mlp": old["methods"]["mlp"],
        "engineered_ridge": score(ridge_test),
        "locked_binary_ecfp_ridge": old["methods"]["ridge"],
    }
    leaderboard = {}
    for cohort in ("full_test", "duplicate_clean_test"):
        rows = sorted(
            (
                {
                    "model": model,
                    "mean_normalized_mae": value[cohort][
                        "mean_normalized_mae_across_12_targets"
                    ],
                }
                for model, value in methods.items()
            ),
            key=lambda row: row["mean_normalized_mae"],
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        leaderboard[cohort] = rows

    np.savez_compressed(
        private_dir / "test_predictions.npz",
        source_row_index=evidence.split.test,
        engineered_ridge=ridge_test,
        traditional_ensemble=traditional_test,
    )
    result = {
        "schema_version": "qm9-traditional-ensemble-correction-v1",
        "status": "post-specified-conceptual-correction-after-initial-test-report",
        "scientific_question": (
            "traditional task-specific pipeline versus independently fine-tuned MIST"
        ),
        "selection_uses_test_labels": False,
        "weights_selected_from": "validation-only",
        "test_interpretation": (
            "Test outcomes were known before this reporting correction; treat as "
            "post-specified, not a pristine preregistered primary analysis."
        ),
        "rows": {
            "train": len(y_train),
            "validation": len(y_validation),
            "full_test": len(y_test),
            "duplicate_clean_test": int(clean_mask.sum()),
        },
        "selected_features": selected_features,
        "feature_dimensions": int(features.shape[1]),
        "global_descriptor_names": descriptor_names,
        "traditional_ensemble_selection": traditional_selection,
        "methods": methods,
        "leaderboard": leaderboard,
        "second_layer_all_model_ensemble": {
            "role": "supplemental systems result, not the primary scientific comparison",
            "validation_selection": old["selected"]["ensemble"],
            "metrics": old["methods"]["ensemble"],
        },
        "original_full_run_runtime_seconds": old["runtime_seconds"],
        "correction_runtime_seconds": time.monotonic() - started,
    }
    atomic_write_json(output_dir / "aggregate_metrics.json", result)
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "source_sha256": sha256_file(source),
            "original_aggregate_sha256": sha256_file(
                repo / "results/qm9-extended-comparison-v1/aggregate_metrics.json"
            ),
            "aggregate_metrics_sha256": sha256_file(output_dir / "aggregate_metrics.json"),
            "private_predictions_sha256": sha256_file(private_dir / "test_predictions.npz"),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/qm9-traditional-ensemble-correction-v1"
    )
    args = parser.parse_args()
    result = run(Path(args.output).resolve())
    print(json.dumps(result["leaderboard"], indent=2))


if __name__ == "__main__":
    main()
