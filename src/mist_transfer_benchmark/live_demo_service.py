"""Private local FastAPI service for the fixed QM9 live-demo models.

The service accepts one SMILES at a time and has no request-controlled model,
file, device, or path parameter.  It never downloads, redistributes, or logs
the private MIST snapshot or prediction requests.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem, rdBase

from .live_demo import LiveDemoBundleError, _make_mlp, validate_bundle
from .qm9.constants import TARGET_COLUMNS
from .qm9.phase2_features import MorganFeatureContract, build_ecfp4_csr
from .qm9.phase3_adapter import stack_named_outputs
from .qm9.phase3_model import (
    MODEL_ID,
    MODEL_REVISION,
    build_model_audit,
    validate_channels,
    verify_snapshot,
)

MAX_SMILES_LENGTH = 512
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_DIR = _REPO_ROOT / "data/private/qm9/live-demo-v1"
_MIST_SNAPSHOT = _REPO_ROOT / "data/private/qm9/mist-phase3/model"


class LiveDemoServiceError(RuntimeError):
    """Raised when a fixed local serving artifact is absent or altered."""


class PredictRequest(BaseModel):
    """Intentionally narrow public request schema."""

    model_config = ConfigDict(extra="forbid")
    smiles: Annotated[str, Field(min_length=1, max_length=MAX_SMILES_LENGTH)]


def _validate_smiles(value: str) -> str:
    """Check RDKit parsability without emitting the user input to logs."""

    if not isinstance(value, str) or not value or len(value) > MAX_SMILES_LENGTH:
        raise ValueError("invalid SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value, sanitize=True)
    if molecule is None:
        raise ValueError("invalid SMILES")
    return value


def _format_row(values: np.ndarray, units: list[str]) -> dict[str, dict[str, float | str]]:
    row = np.asarray(values, dtype=np.float64)
    if row.shape != (len(TARGET_COLUMNS),) or not np.all(np.isfinite(row)):
        raise LiveDemoServiceError("a loaded predictor returned invalid values")
    return {
        target: {"value": float(value), "unit": unit}
        for target, value, unit in zip(TARGET_COLUMNS, row, units, strict=True)
    }


class PrivatePredictors:
    """One-time loader for fixed local ridge/XGB/MLP/MIST artifacts."""

    def __init__(self) -> None:
        self.manifest = validate_bundle(_BUNDLE_DIR)
        self.units = list(self.manifest["units"])
        self._lock = threading.Lock()
        self._load_classical()
        self._load_mist()

    def _load_classical(self) -> None:
        contract_values = self.manifest["ecfp_contract"]["parameters"]
        self.contract = MorganFeatureContract(
            radius=int(contract_values["radius"]),
            fp_size=int(contract_values["fp_size"]),
            count_simulation=bool(contract_values["count_simulation"]),
            include_chirality=bool(contract_values["include_chirality"]),
            use_bond_types=bool(contract_values["use_bond_types"]),
            only_nonzero_invariants=bool(contract_values["only_nonzero_invariants"]),
            include_ring_membership=bool(contract_values["include_ring_membership"]),
            count_bounds=None,
            atom_invariants_generator=None,
            bond_invariants_generator=None,
            include_redundant_environments=bool(contract_values["include_redundant_environments"]),
        )
        ridge = np.load(_BUNDLE_DIR / "ridge.npz", allow_pickle=False)
        self.ridge_coefficients = np.asarray(ridge["coefficients"], dtype=np.float64)
        self.ridge_intercept = np.asarray(ridge["intercept"], dtype=np.float64)
        if self.ridge_coefficients.shape != (len(TARGET_COLUMNS), self.contract.fp_size):
            raise LiveDemoServiceError("Ridge coefficient shape differs from bundle contract")

        from xgboost import XGBRegressor

        self.xgboost = []
        for target in TARGET_COLUMNS:
            model = XGBRegressor()
            model.load_model(_BUNDLE_DIR / f"xgboost_{target}.json")
            self.xgboost.append(model)

        architecture = json.loads((_BUNDLE_DIR / "mlp_architecture.json").read_text())
        if architecture.get("input_dim") != self.contract.fp_size:
            raise LiveDemoServiceError("MLP input width differs from ECFP contract")
        if architecture.get("output_dim") != len(TARGET_COLUMNS):
            raise LiveDemoServiceError("MLP output width differs from target contract")
        import torch
        from safetensors.torch import load_file

        self.mlp = _make_mlp(list(architecture["hidden_dims"]), float(architecture["dropout"]))
        self.mlp.load_state_dict(load_file(_BUNDLE_DIR / "mlp.safetensors", device="cpu"))
        self.mlp.eval()
        if self.mlp.training:
            raise LiveDemoServiceError("MLP eval mode did not persist")
        scaler = np.load(_BUNDLE_DIR / "mlp_scaler.npz", allow_pickle=False)
        self.mlp_mean = np.asarray(scaler["mean"], dtype=np.float64)
        self.mlp_scale = np.asarray(scaler["scale"], dtype=np.float64)
        if self.mlp_mean.shape != (len(TARGET_COLUMNS),) or np.any(self.mlp_scale <= 0):
            raise LiveDemoServiceError("MLP target scaler differs from target contract")
        self._torch = torch

    def _load_mist(self) -> None:
        # Fail closed before importing the reviewed remote-code model class.
        verify_snapshot(_MIST_SNAPSHOT)
        validate_channels(_MIST_SNAPSHOT)
        audit = build_model_audit(_MIST_SNAPSHOT)
        if audit.get("hard_gate_passed") is not True:
            raise LiveDemoServiceError("MIST snapshot did not pass the reviewed hard gate")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from smirk import SmirkTokenizerFast
        from transformers import AutoModel

        self.mist_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mist_tokenizer = SmirkTokenizerFast.from_pretrained(
            str(_MIST_SNAPSHOT), revision=MODEL_REVISION, local_files_only=True
        )
        self.mist = AutoModel.from_pretrained(
            str(_MIST_SNAPSHOT),
            revision=MODEL_REVISION,
            code_revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
        )
        if "tokenizer" not in inspect.signature(self.mist.predict).parameters:
            raise LiveDemoServiceError("reviewed MIST predict interface changed")
        self.mist.to(self.mist_device)
        self.mist.eval()
        if self.mist.training:
            raise LiveDemoServiceError("MIST eval mode did not persist")

    def predict(self, smiles: str) -> dict[str, np.ndarray]:
        """Predict all four fixed models; lock protects shared GPU model state."""

        with self._lock:
            features = build_ecfp4_csr([smiles], self.contract)
            ridge = features @ self.ridge_coefficients.T + self.ridge_intercept
            xgb = np.column_stack([model.predict(features) for model in self.xgboost])
            with self._torch.inference_mode():
                mlp = self.mlp(self._torch.from_numpy(features.toarray().astype(np.float32)))
                mlp = mlp.numpy() * self.mlp_scale + self.mlp_mean
                named = self.mist.predict([smiles], return_dict=True, tokenizer=self.mist_tokenizer)
            mist = stack_named_outputs(named, expected_rows=1)
        return {
            "ridge": np.asarray(ridge, dtype=np.float64)[0],
            "xgboost": np.asarray(xgb, dtype=np.float64)[0],
            "mlp": np.asarray(mlp, dtype=np.float64)[0],
            "mist": np.asarray(mist, dtype=np.float64)[0],
        }


def create_app() -> FastAPI:
    """Create an app whose startup loads only the fixed private artifacts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictors = PrivatePredictors()
        yield
        # Models own no files or processes requiring a teardown action.
        app.state.predictors = None

    app = FastAPI(title="Private QM9 live demo", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "ok", "models": ["ridge", "xgboost", "mlp", "mist"]}

    @app.get("/v1/model-card")
    def model_card() -> dict[str, object]:
        return {
            "scope": "local-private-live-demo-only",
            "targets": list(TARGET_COLUMNS),
            "units": list(app.state.predictors.units),
            "models": {
                "ridge": "ECFP4 + Ridge, alpha=10",
                "xgboost": "ECFP4 + 12 fixed XGBoost regressors",
                "mlp": "ECFP4 + fixed 12-output MLP",
                "mist": f"released fine-tuned checkpoint {MODEL_ID} at {MODEL_REVISION}",
            },
            "limitations": [
                "Predictions are QM9-property estimates, not experimental measurements.",
                "Demo predictions are not a new benchmark score.",
                "MIST weights remain private and are not redistributed by this service.",
            ],
        }

    @app.post("/v1/predict")
    def predict(request: PredictRequest) -> dict[str, object]:
        try:
            smiles = _validate_smiles(request.smiles)
        except ValueError:
            raise HTTPException(status_code=422, detail="invalid SMILES") from None
        try:
            outputs = app.state.predictors.predict(smiles)
        except (LiveDemoBundleError, LiveDemoServiceError, ValueError):
            # Do not expose filesystem paths, model details, or input in an error.
            raise HTTPException(status_code=503, detail="private predictor unavailable") from None
        return {
            "schema_version": "live-demo-prediction-v1",
            "targets": list(TARGET_COLUMNS),
            "units": list(app.state.predictors.units),
            "predictions": {
                name: _format_row(values, app.state.predictors.units)
                for name, values in outputs.items()
            },
        }

    return app


app = create_app()


def main() -> None:
    """Start a loopback-only server with access logging disabled."""

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
