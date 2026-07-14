from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from mist_transfer_benchmark.live_demo import BUNDLE_SCHEMA, LiveDemoBundleError, validate_bundle
from mist_transfer_benchmark.qm9.constants import TARGET_COLUMNS


def test_live_demo_config_freezes_ecfp_target_order_and_units():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "live_demo_v1.toml").open("rb") as handle:
        config = tomllib.load(handle)
    assert config["target_order"] == list(TARGET_COLUMNS)
    assert config["target_units"] == [
        "D",
        "bohr^3",
        "hartree",
        "hartree",
        "hartree",
        "bohr^2",
        "hartree",
        "hartree",
        "hartree",
        "hartree",
        "hartree",
        "cal/(mol K)",
    ]
    assert config["features"] == {
        "representation": "binary Morgan fingerprint (ECFP4)",
        "radius": 2,
        "fp_size": 2048,
        "include_chirality": True,
        "use_bond_types": True,
        "include_ring_membership": True,
    }


def _write_manifest(directory, files: dict[str, bytes]) -> None:
    for name, payload in files.items():
        (directory / name).write_bytes(payload)
    (directory / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "no_test_policy": True,
                "targets": list(TARGET_COLUMNS),
                "files": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
            }
        ),
        encoding="utf-8",
    )


def test_bundle_validator_accepts_hash_checked_private_shape(tmp_path):
    _write_manifest(tmp_path, {"ridge.npz": b"not-loaded-by-validator"})
    assert validate_bundle(tmp_path)["targets"] == list(TARGET_COLUMNS)


def test_bundle_validator_rejects_changed_file(tmp_path):
    _write_manifest(tmp_path, {"ridge.npz": b"before"})
    (tmp_path / "ridge.npz").write_bytes(b"after")
    with pytest.raises(LiveDemoBundleError, match="hash"):
        validate_bundle(tmp_path)


def test_bundle_validator_rejects_wrong_target_order(tmp_path):
    _write_manifest(tmp_path, {"ridge.npz": b"before"})
    manifest = json.loads((tmp_path / "bundle_manifest.json").read_text(encoding="utf-8"))
    manifest["targets"] = list(reversed(TARGET_COLUMNS))
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LiveDemoBundleError, match="target order"):
        validate_bundle(tmp_path)


def test_bundle_validator_rejects_path_traversal(tmp_path):
    _write_manifest(tmp_path, {"ridge.npz": b"before"})
    manifest = json.loads((tmp_path / "bundle_manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {"../outside": "00"}
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LiveDemoBundleError, match="unsafe"):
        validate_bundle(tmp_path)


def test_bundle_validator_requires_no_test_policy(tmp_path):
    _write_manifest(tmp_path, {"ridge.npz": b"before"})
    manifest = json.loads((tmp_path / "bundle_manifest.json").read_text(encoding="utf-8"))
    manifest["no_test_policy"] = False
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LiveDemoBundleError, match="no-test"):
        validate_bundle(tmp_path)


def test_private_mlp_reload_is_repeat_stable_and_matches_self_test():
    """Run only when the ignored serving runtime/bundle is available locally.

    CI neither downloads models nor accesses private artifacts, so it skips
    there.  In the local serving runtime this catches accidentally leaving
    Dropout in train mode after safetensors reload.
    """

    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("xgboost")
    root = Path(__file__).resolve().parents[1]
    bundle = root / "data" / "private" / "qm9" / "live-demo-v1"
    if not bundle.exists():
        pytest.skip("private demo bundle is intentionally unavailable")
    from mist_transfer_benchmark.live_demo import predict_from_bundle

    self_test = json.loads((bundle / "self_test.json").read_text(encoding="utf-8"))
    first = predict_from_bundle(bundle, self_test["smiles"])["mlp"]
    second = predict_from_bundle(bundle, self_test["smiles"])["mlp"]
    import numpy as np

    assert np.array_equal(first, second)
    assert np.allclose(first, np.asarray(self_test["mlp"]), rtol=0.0, atol=1e-7)
