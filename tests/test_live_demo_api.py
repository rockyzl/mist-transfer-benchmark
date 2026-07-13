from __future__ import annotations

import numpy as np
import pytest


def test_private_live_demo_api_endpoints_and_repeat_stability():
    """Exercise actual private models only in the private serving runtime."""

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("torch")
    pytest.importorskip("xgboost")
    from fastapi.testclient import TestClient

    from mist_transfer_benchmark.live_demo_service import app

    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["models"] == ["ridge", "xgboost", "mlp", "mist"]

        card = client.get("/v1/model-card")
        assert card.status_code == 200
        assert card.json()["scope"] == "local-private-live-demo-only"
        assert len(card.json()["targets"]) == 12

        first = client.post("/v1/predict", json={"smiles": "CCO"})
        second = client.post("/v1/predict", json={"smiles": "CCO"})
        assert first.status_code == second.status_code == 200
        first_mlp = np.array(
            [item["value"] for item in first.json()["predictions"]["mlp"].values()]
        )
        second_mlp = np.array(
            [item["value"] for item in second.json()["predictions"]["mlp"].values()]
        )
        assert np.array_equal(first_mlp, second_mlp)
        assert set(first.json()["predictions"]) == {"ridge", "xgboost", "mlp", "mist"}
        assert all(len(model) == 12 for model in first.json()["predictions"].values())

        assert client.post("/v1/predict", json={"smiles": "not a smiles"}).status_code == 422
        assert client.post("/v1/predict", json={"smiles": "C" * 513}).status_code == 422
        assert (
            client.post("/v1/predict", json={"smiles": "CCO", "model": "mist"}).status_code == 422
        )
