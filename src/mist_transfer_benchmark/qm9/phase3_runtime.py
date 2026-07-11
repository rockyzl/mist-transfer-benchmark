"""Isolated, pinned MIST inference-runtime verification."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .io import canonical_hash


class Phase3RuntimeError(ValueError):
    """Raised when the isolated inference environment differs from the lock."""


DIRECT_DISTRIBUTIONS = {
    "datasets": "3.2.0",
    "numpy": "2.5.1",
    "pandas": "2.3.3",
    "rdkit": "2026.3.3",
    "scikit-learn": "1.7.2",
    "smirk": "0.2.0",
    "torch": "2.9.0",
    "transformers": "4.57.1",
}

_PROBE = r"""
import importlib.metadata
import json
import platform
import torch

names = [
    "datasets", "numpy", "pandas", "rdkit", "scikit-learn", "smirk", "torch",
    "transformers",
]
cuda = {
    "available": bool(torch.cuda.is_available()),
    "torch_cuda_version": torch.version.cuda,
    "device_count": int(torch.cuda.device_count()),
    "devices": [],
}
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    cuda["devices"].append({
        "index": index,
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "major": int(props.major),
        "minor": int(props.minor),
    })
print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "distributions": {name: importlib.metadata.version(name) for name in names},
    "torch_cuda": cuda,
}, sort_keys=True))
"""


def _offline_env(python: Path) -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": os.pathsep.join((str(python.parent), "/usr/bin", "/bin")),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TZ": "UTC",
    }


def verify_runtime(
    runtime_python: str | Path, config: dict[str, object]
) -> tuple[dict[str, object], list[str]]:
    """Probe direct versions and freeze the complete isolated environment."""

    python = Path(runtime_python).absolute()
    if not python.exists() or not os.access(python, os.X_OK):
        raise Phase3RuntimeError("isolated runtime Python is not executable")
    completed = subprocess.run(
        [str(python), "-I", "-c", _PROBE],
        check=False,
        capture_output=True,
        text=True,
        env=_offline_env(python),
        timeout=180,
    )
    if completed.returncode != 0:
        raise Phase3RuntimeError(f"runtime probe failed: {completed.stderr}")
    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise Phase3RuntimeError("runtime probe did not emit one JSON object") from error
    if probe.get("python") != "3.12.12":
        raise Phase3RuntimeError(f"runtime Python differs: {probe.get('python')}")
    if probe.get("distributions") != DIRECT_DISTRIBUTIONS:
        raise Phase3RuntimeError(
            f"runtime direct versions differ: {probe.get('distributions')}"
        )
    declared = config["mist_inference_environment"]
    configured = {
        "datasets": str(declared["datasets"]).removeprefix("=="),
        "numpy": str(declared["numpy"]).removeprefix("=="),
        "pandas": str(declared["pandas"]).removeprefix("=="),
        "rdkit": str(declared["rdkit_distribution"]).removeprefix("=="),
        "scikit-learn": str(declared["scikit_learn"]).removeprefix("=="),
        "smirk": str(declared["smirk"]).removeprefix("=="),
        "torch": str(declared["torch"]).removeprefix("=="),
        "transformers": str(declared["transformers"]).removeprefix("=="),
    }
    if configured != DIRECT_DISTRIBUTIONS:
        raise Phase3RuntimeError("TOML inference versions differ from the hard lock")
    uv = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(python)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if uv.returncode != 0:
        raise Phase3RuntimeError(f"uv freeze failed: {uv.stderr}")
    freeze = sorted({line.strip() for line in uv.stdout.splitlines() if line.strip()})
    if not freeze or len(freeze) != len(set(freeze)):
        raise Phase3RuntimeError("environment freeze is empty or contains duplicates")
    result = {
        "schema_version": "qm9-mist-runtime-v1",
        "python_executable": str(python),
        "python": probe["python"],
        "platform": probe["platform"],
        "direct_distributions": probe["distributions"],
        "torch_cuda": probe["torch_cuda"],
        "complete_freeze_canonical_sha256": canonical_hash(freeze),
        "complete_freeze_count": len(freeze),
        "historical_training_environment_claimed": False,
        "purpose": "project-selected-pinned-inference-runtime",
    }
    return result, freeze
