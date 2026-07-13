# ruff: noqa: E501
"""Private, deterministic model-bundle builder for the interactive QM9 demo.

This module deliberately never accepts an evaluation/test index.  It only reads
the reconstructed ``train`` rows and creates an *internal* train/validation
partition for demo-model selection.  The resulting bundle is for interactive
predictions, not a new benchmark result.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from .qm9.constants import TARGET_COLUMNS
from .qm9.data import load_qm9_identities
from .qm9.io import canonical_json_bytes, sha256_file
from .qm9.phase2_features import MorganFeatureContract, build_ecfp4_csr
from .qm9.phase2_targets import load_targets_for_indices

BUNDLE_SCHEMA = "live-demo-v1-private-bundle"


class LiveDemoBundleError(ValueError):
    """Raised when an untrusted bundle or source violates the demo contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _runtime_versions(names: Iterable[str]) -> dict[str, str]:
    from importlib.metadata import version

    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in names:
        result[name] = version(name)
    return result


def _contract_from_config(config: dict[str, object]) -> MorganFeatureContract:
    section = config["features"]
    if section["representation"] != "binary Morgan fingerprint (ECFP4)":
        raise LiveDemoBundleError("unexpected ECFP representation")
    if int(section["radius"]) != 2 or int(section["fp_size"]) != 2048:
        raise LiveDemoBundleError("live demo requires the frozen ECFP4/2048 contract")
    return MorganFeatureContract(
        radius=2,
        fp_size=2048,
        count_simulation=False,
        include_chirality=bool(section["include_chirality"]),
        use_bond_types=bool(section["use_bond_types"]),
        only_nonzero_invariants=False,
        include_ring_membership=bool(section["include_ring_membership"]),
        count_bounds=None,
        atom_invariants_generator=None,
        bond_invariants_generator=None,
        include_redundant_environments=False,
    )


def _read_train_indices(split_path: Path, expected_rows: int) -> np.ndarray:
    rows: list[int] = []
    with split_path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        if header != "source_row_index\tsplit_name":
            raise LiveDemoBundleError("unexpected split manifest header")
        for expected_index, line in enumerate(handle):
            index_text, split_name = line.rstrip("\n").split("\t")
            if int(index_text) != expected_index:
                raise LiveDemoBundleError("split manifest index is not ordered")
            if split_name == "train":
                rows.append(expected_index)
            elif split_name not in {"validation", "test"}:
                raise LiveDemoBundleError("unknown reconstructed split name")
    result = np.asarray(rows, dtype=np.int64)
    if len(result) != expected_rows:
        raise LiveDemoBundleError("reconstructed train row count differs from contract")
    return result


def _mean_normalized_mae(
    predictions: np.ndarray, truth: np.ndarray, train_targets: np.ndarray
) -> float:
    scales = np.std(train_targets, axis=0, ddof=0)
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise LiveDemoBundleError("invalid target scales")
    return float(np.mean(np.mean(np.abs(predictions - truth), axis=0) / scales))


def _new_xgb(
    params: dict[str, object], *, seed: int, device: str, early_stopping_rounds: int | None = None
):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        n_jobs=4,
        random_state=seed,
        verbosity=0,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )


def _xgb_device() -> str:
    try:
        import xgboost as xgb

        # This small probe makes a CPU-only installation fall back without
        # silently changing a fitted model later.
        probe = xgb.XGBRegressor(n_estimators=1, tree_method="hist", device="cuda", verbosity=0)
        probe.fit(np.array([[0.0], [1.0]], dtype=np.float32), np.array([0.0, 1.0]))
        return "cuda"
    except Exception:
        return "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_mlp(hidden_dims: list[int], dropout: float, *, input_dim: int = 2048):
    import torch

    layers: list[object] = []
    width = input_dim
    for next_width in hidden_dims:
        layers.extend(
            [torch.nn.Linear(width, next_width), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        )
        width = next_width
    layers.append(torch.nn.Linear(width, len(TARGET_COLUMNS)))
    return torch.nn.Sequential(*layers)


def _batch_rows(matrix: sparse.csr_matrix, indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        yield matrix[batch].toarray().astype(np.float32, copy=False), batch


def _fit_mlp(
    matrix: sparse.csr_matrix,
    targets: np.ndarray,
    train_positions: np.ndarray,
    validation_positions: np.ndarray | None,
    *,
    config: dict[str, object],
    hidden_dims: list[int],
    dropout: float,
    epochs: int,
    seed: int,
    device: str,
) -> tuple[object, dict[str, np.ndarray], dict[str, object]]:
    import torch

    _set_seed(seed)
    section = config["mlp"]
    train_mean = targets[train_positions].mean(axis=0)
    train_scale = targets[train_positions].std(axis=0)
    if np.any(train_scale <= 0):
        raise LiveDemoBundleError("MLP target scaler has a zero scale")
    model = _make_mlp(hidden_dims, dropout, input_dim=matrix.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = int(section["batch_size"])
    best_state: dict[str, object] | None = None
    best_value = float("inf")
    stale = 0
    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train_positions)
        for features, positions in _batch_rows(matrix, order, batch_size):
            x = torch.from_numpy(features).to(device)
            y = torch.from_numpy(
                ((targets[positions] - train_mean) / train_scale).astype(np.float32)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            optimizer.step()
        if validation_positions is None:
            continue
        prediction = _predict_mlp(
            model, matrix, validation_positions, train_mean, train_scale, batch_size, device
        )
        score = _mean_normalized_mae(
            prediction, targets[validation_positions], targets[train_positions]
        )
        history.append(score)
        if score < best_value:
            best_value, stale = score, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= int(section["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return (
        model,
        {"mean": train_mean, "scale": train_scale},
        {
            "validation_scores": history,
            "best_normalized_mae": best_value if validation_positions is not None else None,
            "epochs_ran": epoch + 1,
            "best_epoch": int(np.argmin(history)) + 1 if history else epochs,
        },
    )


def _predict_mlp(model, matrix, positions, mean, scale, batch_size, device) -> np.ndarray:
    import torch

    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in _batch_rows(matrix, positions, batch_size):
            values = model(torch.from_numpy(features).to(device)).detach().cpu().numpy()
            chunks.append(values * scale + mean)
    return np.vstack(chunks)


def _save_ridge(path: Path, model: Ridge) -> None:
    np.savez_compressed(
        path,
        coefficients=np.asarray(model.coef_, dtype=np.float64),
        intercept=np.asarray(model.intercept_, dtype=np.float64),
    )


def validate_bundle(bundle_dir: str | Path) -> dict[str, object]:
    """Validate hashes/schema without unpickling any artifact."""

    directory = Path(bundle_dir).resolve(strict=True)
    manifest_path = directory / "bundle_manifest.json"
    if manifest_path.is_symlink():
        raise LiveDemoBundleError("manifest may not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("no_test_policy") is not True:
        raise LiveDemoBundleError("not a valid no-test live-demo bundle")
    for name, expected_hash in manifest.get("files", {}).items():
        path = directory / name
        if path.parent != directory or path.is_symlink() or not path.is_file():
            raise LiveDemoBundleError(f"unsafe bundle file {name!r}")
        if sha256_file(path) != expected_hash:
            raise LiveDemoBundleError(f"bundle file hash changed: {name}")
    if manifest.get("targets") != list(TARGET_COLUMNS):
        raise LiveDemoBundleError("target order differs from QM9 contract")
    return manifest


def predict_from_bundle(bundle_dir: str | Path, smiles: list[str]) -> dict[str, np.ndarray]:
    """Reload all safe bundle formats and return finite native-unit predictions.

    This is deliberately an offline helper; the future API may call it but it
    performs no network operation and does not read QM9 labels.
    """

    manifest = validate_bundle(bundle_dir)
    directory = Path(bundle_dir).resolve(strict=True)
    contract_values = manifest["ecfp_contract"]["parameters"]
    contract = MorganFeatureContract(
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
    features = build_ecfp4_csr(smiles, contract)
    ridge_npz = np.load(directory / "ridge.npz", allow_pickle=False)
    ridge = features @ ridge_npz["coefficients"].T + ridge_npz["intercept"]
    from xgboost import XGBRegressor

    xgb_columns = []
    for target in TARGET_COLUMNS:
        model = XGBRegressor()
        model.load_model(directory / f"xgboost_{target}.json")
        xgb_columns.append(model.predict(features))
    xgboost = np.column_stack(xgb_columns)
    architecture = json.loads((directory / "mlp_architecture.json").read_text(encoding="utf-8"))
    if (
        architecture.get("output_dim") != len(TARGET_COLUMNS)
        or architecture.get("input_dim") != 2048
    ):
        raise LiveDemoBundleError("MLP architecture differs from target contract")
    import torch
    from safetensors.torch import load_file

    mlp = _make_mlp(list(architecture["hidden_dims"]), float(architecture["dropout"]))
    mlp.load_state_dict(load_file(directory / "mlp.safetensors", device="cpu"))
    # This is a serving path.  Without eval(), Dropout remains active after a
    # safetensors reload and identical SMILES can receive different outputs.
    mlp.eval()
    scaler = np.load(directory / "mlp_scaler.npz", allow_pickle=False)
    with torch.inference_mode():
        mlp_values = mlp(torch.from_numpy(features.toarray().astype(np.float32))).numpy()
    mlp_values = mlp_values * scaler["scale"] + scaler["mean"]
    result = {"ridge": np.asarray(ridge), "xgboost": xgboost, "mlp": mlp_values}
    if not all(
        values.shape == (len(smiles), len(TARGET_COLUMNS)) and np.all(np.isfinite(values))
        for values in result.values()
    ):
        raise LiveDemoBundleError("bundle prediction is non-finite or has the wrong output shape")
    return result


def refresh_bundle_self_test(bundle_dir: str | Path) -> dict[str, object]:
    """Regenerate private smoke predictions via the deterministic reload path.

    It changes only ignored self-test provenance: no labels, training data, or
    model weights are read or changed.
    """

    directory = Path(bundle_dir).resolve(strict=True)
    manifest = validate_bundle(directory)
    self_test_path = directory / "self_test.json"
    previous = json.loads(self_test_path.read_text(encoding="utf-8"))
    smiles = previous.get("smiles")
    if not isinstance(smiles, list) or not all(isinstance(value, str) for value in smiles):
        raise LiveDemoBundleError("private self-test SMILES are invalid")
    predictions = predict_from_bundle(directory, smiles)
    payload = {
        "smiles": smiles,
        "targets": list(TARGET_COLUMNS),
        "ridge": predictions["ridge"].tolist(),
        "xgboost": predictions["xgboost"].tolist(),
        "mlp": predictions["mlp"].tolist(),
        "reload_mode": "eval-plus-torch-inference-mode",
    }
    _atomic_json(self_test_path, payload)
    manifest["files"]["self_test.json"] = sha256_file(self_test_path)
    manifest["self_test_sha256"] = sha256_file(self_test_path)
    _atomic_json(directory / "bundle_manifest.json", manifest)
    return validate_bundle(directory)


def refresh_bundle_runtime_manifest(bundle_dir: str | Path) -> dict[str, object]:
    """Record the actual isolated `uv pip freeze` output without retraining.

    This is intentionally separate from training because uv-managed venvs may
    omit a `pip` module.  It only changes ignored provenance metadata.
    """

    directory = Path(bundle_dir).resolve(strict=True)
    manifest = validate_bundle(directory)
    config_path = Path(__file__).resolve().parents[2] / "configs" / "live_demo_v1.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("target_order") != list(TARGET_COLUMNS):
        raise LiveDemoBundleError("refreshed config target order is invalid")
    units = config.get("target_units")
    if not isinstance(units, list) or len(units) != len(TARGET_COLUMNS):
        raise LiveDemoBundleError("refreshed config target units are invalid")
    result = subprocess.run(
        ["uv", "pip", "freeze", "--python", sys.executable],
        check=True,
        capture_output=True,
        text=True,
    )
    freeze_path = directory / "runtime.freeze.txt"
    freeze_path.write_text(result.stdout, encoding="utf-8")
    manifest["files"]["runtime.freeze.txt"] = sha256_file(freeze_path)
    manifest["runtime_freeze_sha256"] = sha256_file(freeze_path)
    manifest["units"] = units
    manifest["config_sha256"] = sha256_file(config_path)
    _atomic_json(directory / "bundle_manifest.json", manifest)
    return validate_bundle(directory)


def build_live_demo_bundle(
    *,
    config_path: str | Path = "configs/live_demo_v1.toml",
    bundle_dir: str | Path = "data/private/qm9/live-demo-v1",
    overwrite: bool = False,
    progress=print,
) -> dict[str, object]:
    """Train serving-only Ridge/XGBoost/MLP models using reconstructed train rows only."""

    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("target_order") != list(TARGET_COLUMNS):
        raise LiveDemoBundleError("live-demo target order differs from the frozen QM9 contract")
    target_units = config.get("target_units")
    if not isinstance(target_units, list) or len(target_units) != len(TARGET_COLUMNS):
        raise LiveDemoBundleError("live-demo target units differ from the frozen QM9 contract")
    source = repo_root / str(config["source"]["qm9_csv"])
    split_file = repo_root / str(config["source"]["split_assignments"])
    if sha256_file(source) != config["source"]["source_sha256"]:
        raise LiveDemoBundleError("QM9 source hash differs from frozen contract")
    destination = Path(bundle_dir)
    if not destination.is_absolute():
        destination = repo_root / destination
    if destination.exists() and not overwrite:
        raise FileExistsError(f"bundle destination exists: {destination}")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"staging directory exists: {staging}")
    staging.mkdir(parents=True)
    started = time.monotonic()
    try:
        contract = _contract_from_config(config)
        data = load_qm9_identities(source)
        train_indices = _read_train_indices(
            split_file, int(config["source"]["required_train_rows"])
        )
        # This call reads target cells for train indices only.  Validation/test
        # target rows are intentionally never handed to a loader.
        targets = load_targets_for_indices(source, train_indices, data)
        features = build_ecfp4_csr(
            [data.source_smiles[i] for i in train_indices], contract, progress=progress
        )
        all_positions = np.arange(len(train_indices), dtype=np.int64)
        internal_train, internal_validation = train_test_split(
            all_positions,
            test_size=float(config["internal_validation"]["validation_fraction"]),
            random_state=int(config["internal_validation"]["seed"]),
            shuffle=True,
        )
        seed = int(config["internal_validation"]["seed"])
        progress("training fixed-alpha Ridge on reconstructed train rows")
        ridge = Ridge(alpha=float(config["ridge"]["alpha"]), solver="lsqr", tol=1e-4)
        ridge.fit(features, targets)
        _save_ridge(staging / "ridge.npz", ridge)

        xgb_section = config["xgboost"]
        xgb_device = _xgb_device()
        xgb_validation: list[dict[str, object]] = []
        best_xgb: dict[str, object] | None = None
        for candidate in xgb_section["candidates"]:
            progress(
                f"selecting XGBoost candidate {candidate['id']} on internal validation ({xgb_device})"
            )
            params = {key: value for key, value in candidate.items() if key != "id"}
            predictions = np.empty(
                (len(internal_validation), len(TARGET_COLUMNS)), dtype=np.float64
            )
            rounds: list[int] = []
            for target_index in range(len(TARGET_COLUMNS)):
                model = _new_xgb(
                    params,
                    seed=seed + target_index,
                    device=xgb_device,
                    early_stopping_rounds=int(xgb_section["early_stopping_rounds"]),
                )
                model.fit(
                    features[internal_train],
                    targets[internal_train, target_index],
                    eval_set=[
                        (features[internal_validation], targets[internal_validation, target_index])
                    ],
                    verbose=False,
                )
                predictions[:, target_index] = model.predict(features[internal_validation])
                rounds.append(int(getattr(model, "best_iteration", params["n_estimators"] - 1)) + 1)
            score = _mean_normalized_mae(
                predictions, targets[internal_validation], targets[internal_train]
            )
            record = {
                "id": candidate["id"],
                "parameters": params,
                "normalized_mae": score,
                "per_target_rounds": rounds,
            }
            xgb_validation.append(record)
            if best_xgb is None or score < float(best_xgb["normalized_mae"]):
                best_xgb = record
        assert best_xgb is not None
        progress(
            f"training 12 final XGBoost models from all reconstructed train rows ({xgb_device})"
        )
        for target_index, target in enumerate(TARGET_COLUMNS):
            final_params = dict(best_xgb["parameters"])
            final_params["n_estimators"] = int(best_xgb["per_target_rounds"][target_index])
            model = _new_xgb(final_params, seed=seed + target_index, device=xgb_device)
            model.fit(features, targets[:, target_index], verbose=False)
            model.save_model(staging / f"xgboost_{target}.json")

        import torch
        from safetensors.torch import save_file

        mlp_section = config["mlp"]
        mlp_device = "cuda" if torch.cuda.is_available() else "cpu"
        mlp_validation: list[dict[str, object]] = []
        selected_mlp: dict[str, object] | None = None
        for candidate in mlp_section["candidates"]:
            progress(
                f"selecting MLP candidate {candidate['id']} on internal validation ({mlp_device})"
            )
            _, _, evidence = _fit_mlp(
                features,
                targets,
                internal_train,
                internal_validation,
                config=config,
                hidden_dims=list(candidate["hidden_dims"]),
                dropout=float(candidate["dropout"]),
                epochs=int(mlp_section["max_epochs"]),
                seed=seed,
                device=mlp_device,
            )
            record = {"id": candidate["id"], "parameters": dict(candidate), **evidence}
            mlp_validation.append(record)
            if selected_mlp is None or float(record["best_normalized_mae"]) < float(
                selected_mlp["best_normalized_mae"]
            ):
                selected_mlp = record
        assert selected_mlp is not None
        final_epochs = int(selected_mlp["best_epoch"])
        progress(
            f"training final MLP for {final_epochs} epochs on all reconstructed train rows ({mlp_device})"
        )
        final_mlp, scaler, final_evidence = _fit_mlp(
            features,
            targets,
            all_positions,
            None,
            config=config,
            hidden_dims=list(selected_mlp["parameters"]["hidden_dims"]),
            dropout=float(selected_mlp["parameters"]["dropout"]),
            epochs=final_epochs,
            seed=seed,
            device=mlp_device,
        )
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in final_mlp.state_dict().items()
            },
            staging / "mlp.safetensors",
        )
        _atomic_json(
            staging / "mlp_architecture.json",
            {
                "input_dim": 2048,
                "hidden_dims": selected_mlp["parameters"]["hidden_dims"],
                "dropout": selected_mlp["parameters"]["dropout"],
                "output_dim": len(TARGET_COLUMNS),
                "activation": "ReLU",
                "target_standardization": "(y - mean) / population_std",
            },
        )
        np.savez_compressed(staging / "mlp_scaler.npz", mean=scaler["mean"], scale=scaler["scale"])

        # Use fixed, known train rows as private reload smoke cases.
        self_positions = np.array([0, 1, 2], dtype=np.int64)
        self_smiles = [
            data.source_smiles[int(train_indices[position])] for position in self_positions
        ]
        ridge_values = features[self_positions] @ ridge.coef_.T + ridge.intercept_
        xgb_values = []
        from xgboost import XGBRegressor

        for target in TARGET_COLUMNS:
            loaded = XGBRegressor()
            loaded.load_model(staging / f"xgboost_{target}.json")
            xgb_values.append(loaded.predict(features[self_positions]))
        xgb_values = np.column_stack(xgb_values)
        mlp_values = _predict_mlp(
            final_mlp,
            features,
            self_positions,
            scaler["mean"],
            scaler["scale"],
            int(mlp_section["batch_size"]),
            mlp_device,
        )
        if not all(
            np.all(np.isfinite(values)) for values in (ridge_values, xgb_values, mlp_values)
        ):
            raise LiveDemoBundleError("model self-test produced a non-finite prediction")
        _atomic_json(
            staging / "self_test.json",
            {
                "smiles": self_smiles,
                "targets": list(TARGET_COLUMNS),
                "ridge": np.asarray(ridge_values).tolist(),
                "xgboost": xgb_values.tolist(),
                "mlp": mlp_values.tolist(),
            },
        )
        # uv-managed virtual environments need not expose `python -m pip`.
        freeze = subprocess.run(
            ["uv", "pip", "freeze", "--python", sys.executable],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (staging / "runtime.freeze.txt").write_text(freeze, encoding="utf-8")
        file_names = [
            "ridge.npz",
            "mlp.safetensors",
            "mlp_architecture.json",
            "mlp_scaler.npz",
            "self_test.json",
            "runtime.freeze.txt",
        ] + [f"xgboost_{target}.json" for target in TARGET_COLUMNS]
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "protocol_id": config["protocol_id"],
            "no_test_policy": True,
            "statement": "Serving-only bundle. Neither reconstructed validation nor test labels were read. This bundle must not be used as a new benchmark result.",
            "source": {
                "sha256": sha256_file(source),
                "train_rows": int(len(train_indices)),
                "split_manifest_sha256": sha256_file(split_file),
            },
            "internal_validation": {
                "seed": seed,
                "train_rows": int(len(internal_train)),
                "validation_rows": int(len(internal_validation)),
                "validation_fraction": config["internal_validation"]["validation_fraction"],
            },
            "ecfp_contract": contract.manifest(),
            "targets": list(TARGET_COLUMNS),
            "units": target_units,
            "ridge": {"alpha": config["ridge"]["alpha"]},
            "xgboost": {"device": xgb_device, "selection": xgb_validation, "selected": best_xgb},
            "mlp": {
                "device": mlp_device,
                "selection": mlp_validation,
                "selected": selected_mlp,
                "final": final_evidence,
            },
            "runtime": _runtime_versions(
                [
                    "numpy",
                    "scikit-learn",
                    "torch",
                    "xgboost",
                    "safetensors",
                    "rdkit",
                    "fastapi",
                    "uvicorn",
                ]
            ),
            "files": {name: sha256_file(staging / name) for name in file_names},
            "runtime_freeze_sha256": sha256_file(staging / "runtime.freeze.txt"),
            "build_seconds": time.monotonic() - started,
        }
        _atomic_json(staging / "bundle_manifest.json", manifest)
        if destination.exists():
            import shutil

            shutil.rmtree(destination)
        os.replace(staging, destination)
        validate_bundle(destination)
        return manifest
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
