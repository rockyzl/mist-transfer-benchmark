#!/usr/bin/env python3
"""Tune XGBoost/MLP on validation, freeze them, then score the common QM9 test once."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import tomllib
from pathlib import Path

# Must be set before the first CUDA/cuBLAS context is created. The completed
# v1 run exposed the missing setting as a warning; future reruns fail less open.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from mist_transfer_benchmark.live_demo import (
    _fit_mlp,
    _mean_normalized_mae,
    _new_xgb,
    _predict_mlp,
    _save_ridge,
    _xgb_device,
)
from mist_transfer_benchmark.qm9.constants import TARGET_COLUMNS
from mist_transfer_benchmark.qm9.data import load_qm9_identities
from mist_transfer_benchmark.qm9.io import atomic_write_json, sha256_file
from mist_transfer_benchmark.qm9.phase2_contract import verify_phase1_evidence
from mist_transfer_benchmark.qm9.phase2_metrics import native_metrics
from mist_transfer_benchmark.qm9.phase2_targets import load_targets_for_indices
from mist_transfer_benchmark.qm9.phase3_adapter import stack_named_outputs


def _count_ecfp(smiles: list[str], fp_size: int = 2048) -> sparse.csr_matrix:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=fp_size,
        includeChirality=True,
        useBondTypes=True,
        includeRingMembership=True,
    )
    rows, columns, values = [], [], []
    for row, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"RDKit could not parse source row {row}")
        for column, count in generator.GetCountFingerprint(molecule).GetNonzeroElements().items():
            rows.append(row)
            columns.append(column)
            values.append(float(count))
        if (row + 1) % 20_000 == 0:
            print(f"count ECFP: {row + 1:,}/{len(smiles):,}", flush=True)
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(smiles), fp_size))


def _global_descriptors(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    names = [
        "mol_wt", "heavy_atoms", "all_atoms", "bonds", "rings", "rotatable_bonds",
        "h_donors", "h_acceptors", "tpsa", "logp", "fraction_csp3", "formal_charge",
        "carbon", "nitrogen", "oxygen", "fluorine", "aromatic_atoms",
    ]
    matrix = np.empty((len(smiles), len(names)), dtype=np.float64)
    for row, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"RDKit could not parse source row {row}")
        atoms = list(molecule.GetAtoms())
        symbols = [atom.GetSymbol() for atom in atoms]
        matrix[row] = [
            Descriptors.MolWt(molecule),
            molecule.GetNumHeavyAtoms(),
            sum(1 + atom.GetTotalNumHs() for atom in atoms),
            molecule.GetNumBonds(),
            Lipinski.RingCount(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            Chem.GetFormalCharge(molecule),
            symbols.count("C"), symbols.count("N"), symbols.count("O"), symbols.count("F"),
            sum(atom.GetIsAromatic() for atom in atoms),
        ]
        if (row + 1) % 20_000 == 0:
            print(f"global descriptors: {row + 1:,}/{len(smiles):,}", flush=True)
    return matrix, names


def _feature_candidates(
    binary: sparse.csr_matrix,
    smiles: list[str],
    train_indices: np.ndarray,
) -> tuple[dict[str, sparse.csr_matrix], list[str]]:
    count = _count_ecfp(smiles)
    descriptors, names = _global_descriptors(smiles)
    scaler = StandardScaler().fit(descriptors[train_indices])
    scaled = sparse.csr_matrix(scaler.transform(descriptors))
    return {
        "binary_ecfp": binary,
        "count_ecfp": count,
        "binary_ecfp_plus_globals": sparse.hstack([binary, scaled], format="csr"),
        "count_ecfp_plus_globals": sparse.hstack([count, scaled], format="csr"),
    }, names


def _mist_predict(repo: Path, smiles: list[str], batch_size: int = 128) -> np.ndarray:
    """Run the fixed, audited MIST snapshot in source order and native units."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from smirk import SmirkTokenizerFast
    from transformers import AutoModel

    snapshot = repo / "data/private/qm9/mist-phase3/model"
    tokenizer = SmirkTokenizerFast.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True, use_safetensors=True
    ).to("cuda")
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start : start + batch_size]
            named = model.predict(batch, return_dict=True, tokenizer=tokenizer)
            chunks.append(stack_named_outputs(named, expected_rows=len(batch)))
    result = np.vstack(chunks)
    del model
    torch.cuda.empty_cache()
    return result


def _select_ensemble_weights(
    predictions: dict[str, np.ndarray], truth: np.ndarray, scale: np.ndarray
) -> dict:
    """Select nonnegative sum-to-one blend weights using validation labels only."""

    names = list(predictions)
    stack = np.stack([predictions[name] for name in names], axis=0)

    def objective(weights: np.ndarray) -> float:
        blended = np.tensordot(weights, stack, axes=(0, 0))
        return float(np.mean(np.mean(np.abs(blended - truth), axis=0) / scale))

    starts = [np.full(len(names), 1.0 / len(names))]
    starts.extend(np.eye(len(names)))
    best = None
    for start in starts:
        fitted = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(names),
            constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
            options={"maxiter": 300, "ftol": 1e-10},
        )
        weights = np.clip(fitted.x, 0.0, 1.0)
        weights /= weights.sum()
        score = objective(weights)
        if best is None or score < best["validation_mean_normalized_mae"]:
            best = {
                "model_order": names,
                "weights": {name: float(value) for name, value in zip(names, weights)},
                "validation_mean_normalized_mae": score,
                "optimizer_success": bool(fitted.success),
            }
    assert best is not None
    return best


def _load_existing_metrics(repo: Path) -> tuple[dict, dict]:
    classical = json.loads(
        (repo / "results/qm9-phase2-classical-v1/test_metrics.json").read_text()
    )
    mist = json.loads((repo / "results/qm9-phase3-mist-v1/mist_metrics.json").read_text())
    return classical, mist


def _mist_cohort(mist: dict, cohort: str) -> dict:
    if cohort in mist:
        return mist[cohort]
    if "cohorts" in mist and cohort in mist["cohorts"]:
        return mist["cohorts"][cohort]
    raise KeyError(f"MIST metrics missing cohort {cohort}")


def run(config_path: Path, output_dir: Path) -> dict:
    repo = Path(__file__).resolve().parents[1]
    config = tomllib.loads(config_path.read_text())
    if config["target_order"] != list(TARGET_COLUMNS):
        raise ValueError("target order differs from the frozen QM9 contract")
    output_dir.mkdir(parents=True, exist_ok=False)
    private_dir = repo / "data/private/qm9/extended-comparison-v1"
    if private_dir.exists():
        raise FileExistsError(f"private output already exists: {private_dir}")
    private_dir.mkdir(parents=True)
    started = time.monotonic()

    source = repo / "data/private/qm9/qm9.csv"
    with (repo / "configs/qm9_28m.toml").open("rb") as handle:
        protocol = tomllib.load(handle)
    evidence = verify_phase1_evidence(protocol, repo / "results/qm9-phase1-v2", source)
    data = load_qm9_identities(source)
    binary_features = sparse.load_npz(
        repo / "results/qm9-phase2-classical-v1/feature_matrix.npz"
    )
    y_train = load_targets_for_indices(source, evidence.split.train, data)
    y_validation = load_targets_for_indices(source, evidence.split.validation, data)
    train_scale = np.std(y_train, axis=0, ddof=0)
    seed = int(config["seed"])
    feature_sets, descriptor_names = _feature_candidates(
        binary_features, list(data.source_smiles), evidence.split.train
    )
    feature_screen = []
    selected_features = None
    ridge = None
    ridge_validation = None
    best_feature_score = float("inf")
    for name, matrix in feature_sets.items():
        candidate = Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=10000)
        candidate.fit(
            matrix[evidence.split.train],
            (y_train - y_train.mean(axis=0)) / train_scale,
        )
        prediction = (
            candidate.predict(matrix[evidence.split.validation]) * train_scale
            + y_train.mean(axis=0)
        )
        score = _mean_normalized_mae(prediction, y_validation, y_train)
        feature_screen.append({"id": name, "ridge_validation_mean_normalized_mae": score})
        print(f"feature screen {name}: {score:.8f}", flush=True)
        if score < best_feature_score:
            best_feature_score = score
            selected_features = name
            ridge = candidate
            ridge_validation = prediction
    assert selected_features is not None and ridge is not None and ridge_validation is not None
    features = feature_sets[selected_features]
    x_train = features[evidence.split.train]
    x_validation = features[evidence.split.validation]
    print(f"Selected feature representation: {selected_features}", flush=True)

    device = _xgb_device()
    xgb_records = []
    best_xgb = None
    for candidate in config["xgboost"]["candidates"]:
        print(f"XGBoost validation: {candidate['id']} ({device})", flush=True)
        params = {key: value for key, value in candidate.items() if key != "id"}
        predictions = np.empty_like(y_validation)
        rounds = []
        candidate_started = time.monotonic()
        for target_index, target in enumerate(TARGET_COLUMNS):
            model = _new_xgb(
                params,
                seed=seed + target_index,
                device=device,
                early_stopping_rounds=int(config["xgboost"]["early_stopping_rounds"]),
            )
            model.fit(
                x_train,
                y_train[:, target_index],
                eval_set=[(x_validation, y_validation[:, target_index])],
                verbose=False,
            )
            predictions[:, target_index] = model.predict(x_validation)
            rounds.append(int(getattr(model, "best_iteration", params["n_estimators"] - 1)) + 1)
        score = _mean_normalized_mae(predictions, y_validation, y_train)
        record = {
            "id": candidate["id"],
            "parameters": params,
            "per_target_rounds": rounds,
            "validation_mean_normalized_mae": score,
            "runtime_seconds": time.monotonic() - candidate_started,
        }
        xgb_records.append(record)
        if best_xgb is None or score < best_xgb["validation_mean_normalized_mae"]:
            best_xgb = record
        print(f"  score={score:.8f}", flush=True)
    assert best_xgb is not None

    print(f"Freezing XGBoost winner: {best_xgb['id']}", flush=True)
    xgb_models = []
    for target_index, target in enumerate(TARGET_COLUMNS):
        params = dict(best_xgb["parameters"])
        params["n_estimators"] = int(best_xgb["per_target_rounds"][target_index])
        model = _new_xgb(params, seed=seed + target_index, device=device)
        model.fit(x_train, y_train[:, target_index], verbose=False)
        model.save_model(private_dir / f"xgboost_{target}.json")
        xgb_models.append(model)
    xgb_validation = np.column_stack([model.predict(x_validation) for model in xgb_models])
    del xgb_models

    combined_features = sparse.vstack([x_train, x_validation], format="csr")
    combined_targets = np.vstack([y_train, y_validation])
    train_positions = np.arange(len(y_train), dtype=np.int64)
    validation_positions = np.arange(len(y_train), len(combined_targets), dtype=np.int64)
    mlp_records = []
    best_mlp = None
    mlp_device = "cuda"
    for candidate in config["mlp"]["candidates"]:
        print(f"MLP validation: {candidate['id']} ({mlp_device})", flush=True)
        local_config = {"mlp": copy.deepcopy(config["mlp"])}
        local_config["mlp"]["learning_rate"] = candidate["learning_rate"]
        local_config["mlp"]["weight_decay"] = candidate["weight_decay"]
        candidate_started = time.monotonic()
        _, _, evidence_record = _fit_mlp(
            combined_features,
            combined_targets,
            train_positions,
            validation_positions,
            config=local_config,
            hidden_dims=list(candidate["hidden_dims"]),
            dropout=float(candidate["dropout"]),
            epochs=int(config["mlp"]["max_epochs"]),
            seed=seed,
            device=mlp_device,
        )
        record = {
            "id": candidate["id"],
            "parameters": dict(candidate),
            **evidence_record,
            "runtime_seconds": time.monotonic() - candidate_started,
        }
        mlp_records.append(record)
        if best_mlp is None or record["best_normalized_mae"] < best_mlp["best_normalized_mae"]:
            best_mlp = record
        print(f"  score={record['best_normalized_mae']:.8f}", flush=True)
    assert best_mlp is not None

    print(f"Freezing MLP winner: {best_mlp['id']}", flush=True)
    mlp_config = {"mlp": copy.deepcopy(config["mlp"])}
    mlp_config["mlp"]["learning_rate"] = best_mlp["parameters"]["learning_rate"]
    mlp_config["mlp"]["weight_decay"] = best_mlp["parameters"]["weight_decay"]
    final_positions = np.arange(len(y_train), dtype=np.int64)
    mlp_model, mlp_scaler, _ = _fit_mlp(
        x_train,
        y_train,
        final_positions,
        None,
        config=mlp_config,
        hidden_dims=list(best_mlp["parameters"]["hidden_dims"]),
        dropout=float(best_mlp["parameters"]["dropout"]),
        epochs=int(best_mlp["best_epoch"]),
        seed=seed,
        device=mlp_device,
    )
    from safetensors.torch import save_file

    save_file(
        {key: value.detach().cpu().contiguous() for key, value in mlp_model.state_dict().items()},
        private_dir / "mlp.safetensors",
    )
    np.savez_compressed(
        private_dir / "mlp_scaler.npz", mean=mlp_scaler["mean"], scale=mlp_scaler["scale"]
    )

    mlp_validation = _predict_mlp(
        mlp_model,
        x_validation,
        np.arange(len(y_validation), dtype=np.int64),
        mlp_scaler["mean"],
        mlp_scaler["scale"],
        int(config["mlp"]["batch_size"]),
        mlp_device,
    )
    print("Running fixed MIST on validation for ensemble selection", flush=True)
    mist_validation = _mist_predict(
        repo, [data.source_smiles[int(index)] for index in evidence.split.validation]
    )
    ensemble = _select_ensemble_weights(
        {
            "ridge": ridge_validation,
            "xgboost": xgb_validation,
            "mlp": mlp_validation,
            "mist": mist_validation,
        },
        y_validation,
        train_scale,
    )
    print(f"Ensemble validation score={ensemble['validation_mean_normalized_mae']:.8f}", flush=True)

    # Scientific choices are now frozen. This is the single test-label read for
    # the extended comparison requested by the repository owner.
    freeze = {
        "protocol_id": config["protocol_id"],
        "source_sha256": sha256_file(source),
        "feature_matrix_sha256": sha256_file(
            repo / "results/qm9-phase2-classical-v1/feature_matrix.npz"
        ),
        "xgboost_selected": best_xgb,
        "mlp_selected": best_mlp,
        "ensemble_selected": ensemble,
        "feature_selection": feature_screen,
        "selected_features": selected_features,
        "global_descriptor_names": descriptor_names,
        "test_rows": int(len(evidence.split.test)),
    }
    atomic_write_json(private_dir / "selection_freeze.json", freeze, mode=0o600)
    print("Selection frozen; loading common test labels once", flush=True)
    y_test = load_targets_for_indices(source, evidence.split.test, data)
    x_test = features[evidence.split.test]
    from xgboost import XGBRegressor

    xgb_test_columns = []
    for target in TARGET_COLUMNS:
        model = XGBRegressor()
        model.load_model(private_dir / f"xgboost_{target}.json")
        xgb_test_columns.append(model.predict(x_test))
    xgb_test = np.column_stack(xgb_test_columns)
    mlp_test = _predict_mlp(
        mlp_model,
        x_test,
        np.arange(len(y_test), dtype=np.int64),
        mlp_scaler["mean"],
        mlp_scaler["scale"],
        int(config["mlp"]["batch_size"]),
        mlp_device,
    )
    ridge_test = ridge.predict(x_test) * train_scale + y_train.mean(axis=0)
    mist_test = np.vstack(
        [
            np.asarray(json.loads(line)["predicted"], dtype=np.float64)
            for line in (repo / "results/qm9-phase3-mist-v1/mist_predictions.jsonl")
            .read_text()
            .splitlines()
        ]
    )
    ensemble_test = sum(
        ensemble["weights"][name] * values
        for name, values in {
            "ridge": ridge_test,
            "xgboost": xgb_test,
            "mlp": mlp_test,
            "mist": mist_test,
        }.items()
    )
    clean = {int(index) for index in evidence.duplicate_clean_test}
    clean_mask = np.asarray([int(index) in clean for index in evidence.split.test], dtype=bool)
    np.savez_compressed(
        private_dir / "test_predictions.npz",
        source_row_index=evidence.split.test,
        xgboost=xgb_test,
        mlp=mlp_test,
        ensemble=ensemble_test,
    )
    classical, mist = _load_existing_metrics(repo)
    methods = {
        "ridge": classical["methods"]["ridge"],
        "xgboost": {
            "full_test": native_metrics(y_test, xgb_test, train_scale),
            "duplicate_clean_test": native_metrics(
                y_test[clean_mask], xgb_test[clean_mask], train_scale
            ),
        },
        "mlp": {
            "full_test": native_metrics(y_test, mlp_test, train_scale),
            "duplicate_clean_test": native_metrics(
                y_test[clean_mask], mlp_test[clean_mask], train_scale
            ),
        },
        "mist": {
            "full_test": _mist_cohort(mist, "full_test"),
            "duplicate_clean_test": _mist_cohort(mist, "duplicate_clean_test"),
        },
        "ensemble": {
            "full_test": native_metrics(y_test, ensemble_test, train_scale),
            "duplicate_clean_test": native_metrics(
                y_test[clean_mask], ensemble_test[clean_mask], train_scale
            ),
        },
    }
    leaderboard = {}
    for cohort in ("full_test", "duplicate_clean_test"):
        leaderboard[cohort] = sorted(
            [
                {
                    "rank": 0,
                    "model": name,
                    "mean_normalized_mae": values[cohort][
                        "mean_normalized_mae_across_12_targets"
                    ],
                }
                for name, values in methods.items()
            ],
            key=lambda item: item["mean_normalized_mae"],
        )
        for rank, item in enumerate(leaderboard[cohort], start=1):
            item["rank"] = rank
    result = {
        "schema_version": "qm9-extended-comparison-v1",
        "status": "preliminary-point-estimates-common-frozen-test",
        "selection_uses_test_labels": False,
        "test_evaluations_after_freeze": 1,
        "rows": {
            "train": len(y_train),
            "validation": len(y_validation),
            "full_test": len(y_test),
            "duplicate_clean_test": int(clean_mask.sum()),
        },
        "selection": {"xgboost": xgb_records, "mlp": mlp_records},
        "feature_selection": feature_screen,
        "selected_features": selected_features,
        "global_descriptor_names": descriptor_names,
        "selected": {"xgboost": best_xgb, "mlp": best_mlp, "ensemble": ensemble},
        "methods": methods,
        "leaderboard": leaderboard,
        "auxiliary_baselines": {
            key: classical["methods"].get(key)
            for key in ("training-target-means", "ecfp-tanimoto-1nn", "random_forest")
            if key in classical["methods"]
        },
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_write_json(output_dir / "aggregate_metrics.json", result)
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "config_sha256": sha256_file(config_path),
            "aggregate_metrics_sha256": sha256_file(output_dir / "aggregate_metrics.json"),
            "private_selection_freeze_sha256": sha256_file(
                private_dir / "selection_freeze.json"
            ),
            "private_predictions_sha256": sha256_file(private_dir / "test_predictions.npz"),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qm9_extended_comparison_v1.toml")
    parser.add_argument("--output", default="results/qm9-extended-comparison-v1")
    args = parser.parse_args()
    result = run(Path(args.config).resolve(), Path(args.output).resolve())
    print(json.dumps(result["leaderboard"], indent=2))


if __name__ == "__main__":
    main()
