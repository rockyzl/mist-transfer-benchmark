import json

import numpy as np

from mist_transfer_benchmark.baseline import run_ecfp_baselines
from mist_transfer_benchmark.fingerprints import (
    FingerprintConfig,
    ecfp_matrix,
    nearest_train_similarity,
)
from mist_transfer_benchmark.splits import SplitConfig, make_split


def test_all_baselines_return_metrics_and_predictions(redox_frame):
    assignments = make_split(redox_frame, SplitConfig(strategy="scaffold", seed=42))

    result, predictions = run_ecfp_baselines(
        redox_frame,
        assignments,
        ("dummy", "tanimoto_1nn", "ridge", "random_forest"),
        FingerprintConfig(radius=2, n_bits=256),
        seed=42,
    )

    assert set(result["metrics"]) == {
        "dummy",
        "tanimoto_1nn",
        "ridge",
        "random_forest",
    }
    assert len(predictions) == 4 * len(redox_frame)
    assert predictions["max_train_tanimoto"].between(0, 1).all()
    assert predictions["nearest_train_record_id"].notna().all()
    json.dumps(result, allow_nan=False)
    assert "median_ae" in result["metrics"]["tanimoto_1nn"]["test"]
    assert "spearman" in result["metrics"]["tanimoto_1nn"]["test"]
    assert "similarity_bins" in result["breakdowns"]["tanimoto_1nn"]


def test_ridge_predictions_are_deterministic(redox_frame):
    assignments = make_split(redox_frame, SplitConfig(strategy="random", seed=73))
    arguments = (
        redox_frame,
        assignments,
        ("ridge",),
        FingerprintConfig(radius=2, n_bits=128),
        73,
    )

    first_result, first_predictions = run_ecfp_baselines(*arguments)
    second_result, second_predictions = run_ecfp_baselines(*arguments)

    assert first_result == second_result
    np.testing.assert_allclose(
        first_predictions["prediction_v"], second_predictions["prediction_v"]
    )


def test_chunked_similarity_matches_single_chunk(redox_frame):
    matrix = ecfp_matrix(redox_frame["canonical_smiles"].tolist(), FingerprintConfig(n_bits=128))
    train_positions = np.array([0, 1, 2, 3])
    train_ids = redox_frame.iloc[train_positions]["record_id"].tolist()

    chunked = nearest_train_similarity(
        matrix,
        train_positions,
        train_ids,
        chunk_size=2,
    )
    single = nearest_train_similarity(
        matrix,
        train_positions,
        train_ids,
        chunk_size=len(redox_frame),
    )

    np.testing.assert_allclose(chunked[0], single[0])
    assert chunked[1] == single[1]
