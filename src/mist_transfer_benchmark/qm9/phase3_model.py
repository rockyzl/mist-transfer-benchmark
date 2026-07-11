"""Pinned MIST-28M artifact, remote-code, channel, and tensor audit."""

from __future__ import annotations

import ast
import json
import math
import stat
import struct
from itertools import pairwise
from pathlib import Path

from .constants import TARGET_COLUMNS
from .io import canonical_hash, sha256_file

MODEL_ID = "mist-models/mist-26.9M-kkgx0omx-qm9"
MODEL_REVISION = "65ceeed479609e9dcaef04e687556e2b39e25f23"
WEIGHT_SHA256 = "f92e42f932c75e39a1dcb070fca8fd1c3fb3a4dcb763fb15447f035d770a9618"
WEIGHT_BYTES = 108_614_208

EXPECTED_FILES: dict[str, tuple[int, str]] = {
    ".gitattributes": (
        1519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "README.md": (
        7041,
        "ec01a4cc03024b8558f1d8011cf674651794e3ae153e02718598fb39dac3c1ca",
    ),
    "config.json": (
        4117,
        "7b94d0123e1c19c2f50f6c1d9925673caaafa9955b00ddeab1f08eac3b35bfec",
    ),
    "model.safetensors": (WEIGHT_BYTES, WEIGHT_SHA256),
    "modeling_mist_finetuned.py": (
        28062,
        "0dfd3573382ff9adcfc884acbab85d268278356cc6b8683759871fa7029da6ae",
    ),
    "notebook.ipynb": (
        927,
        "5461bfd358bf9bf12e4966e2beb738a179f2a4dd7f0e02d70f0521fc2c6e9637",
    ),
    "requirements.txt": (
        83,
        "667f75fed13abdbe64660012dc9d5f1bdd2066e789e75944a826332de8052158",
    ),
    "special_tokens_map.json": (
        971,
        "820899a5801411108e66c10c1b258363aa9a8e1576a525e76016a2dba7680ea7",
    ),
    "tokenizer.json": (
        5014,
        "fb362f00c2f02529b264e9f131ef9ca1ce9915fc5a82155e1a93d27bd764d356",
    ),
    "tokenizer_config.json": (
        1681,
        "ec0f9570f48db969f1872873e9d8b95cf25690abe17b96b073f04e8de80acd1c",
    ),
}

EXPECTED_UNITS = (
    "debye",
    "cubic bohr",
    "hartree",
    "hartree",
    "hartree",
    "square bohr",
    "hartree",
    "hartree",
    "hartree",
    "hartree",
    "hartree",
    "calorie / mole / kelvin",
)


class Phase3ModelError(ValueError):
    """Raised before unreviewed or altered model bytes can execute."""


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
        raise Phase3ModelError(f"model artifact must be a regular file: {path}")


def validate_model_config(config: dict[str, object]) -> dict[str, object]:
    model = config["model"]
    if model["id"] != MODEL_ID or model["revision"] != MODEL_REVISION:
        raise Phase3ModelError("model id/revision differs from the pinned contract")
    if model["tokenizer_id"] != MODEL_ID or model["tokenizer_revision"] != MODEL_REVISION:
        raise Phase3ModelError("tokenizer is not pinned to the model revision")
    if model["expected_output_count"] != 12 or model["trust_remote_code"] is not True:
        raise Phase3ModelError("model output/remote-code contract differs")
    observed = model["hugging_face_api_observation"]
    if observed["safetensors_bytes"] != WEIGHT_BYTES:
        raise Phase3ModelError("configured safetensors byte count differs")
    if observed["safetensors_sha256"] != WEIGHT_SHA256:
        raise Phase3ModelError("configured safetensors SHA-256 differs")
    if tuple(config["targets"]["ordered_names"]) != TARGET_COLUMNS:
        raise Phase3ModelError("configured target order differs")
    if tuple(config["targets"]["checkpoint_unit_strings"]) != EXPECTED_UNITS:
        raise Phase3ModelError("configured checkpoint units differ")
    return model


def verify_snapshot(snapshot_dir: str | Path) -> list[dict[str, object]]:
    """Require the exact reviewed ten-file snapshot and immutable bytes."""

    directory = Path(snapshot_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise Phase3ModelError("model snapshot must be a real directory")
    entries = {entry.name for entry in directory.iterdir()}
    if entries != set(EXPECTED_FILES):
        raise Phase3ModelError(
            f"model snapshot allowlist differs: {sorted(entries ^ set(EXPECTED_FILES))}"
        )
    manifest = []
    for name in sorted(EXPECTED_FILES):
        expected_bytes, expected_hash = EXPECTED_FILES[name]
        path = directory / name
        _regular_file(path)
        observed_hash = sha256_file(path)
        if path.stat().st_size != expected_bytes or observed_hash != expected_hash:
            raise Phase3ModelError(f"model artifact bytes/hash differ: {name}")
        manifest.append(
            {"path": name, "bytes": expected_bytes, "sha256": observed_hash}
        )
    return manifest


def validate_channels(snapshot_dir: str | Path) -> list[dict[str, str]]:
    payload = json.loads((Path(snapshot_dir) / "config.json").read_text(encoding="utf-8"))
    channels = payload.get("channels")
    if not isinstance(channels, list) or len(channels) != 12:
        raise Phase3ModelError("checkpoint does not declare exactly 12 channels")
    names = tuple(item.get("name") for item in channels)
    units = tuple(item.get("unit") for item in channels)
    if names != TARGET_COLUMNS or units != EXPECTED_UNITS:
        raise Phase3ModelError("checkpoint channel order/units differ from the contract")
    if payload.get("transform") != {"class": "Standardize", "num_outputs": 12}:
        raise Phase3ModelError("checkpoint output transform differs")
    if payload.get("task_network", {}).get("output_size") != 12:
        raise Phase3ModelError("checkpoint task head is not 12-output")
    return channels


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def audit_safetensors(path: str | Path) -> dict[str, object]:
    """Parse and validate the safetensors header without importing torch."""

    source = Path(path)
    _regular_file(source)
    with source.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise Phase3ModelError("safetensors header prefix is truncated")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 0 or header_bytes > 10_000_000:
            raise Phase3ModelError("safetensors header length is unsafe")
        raw_header = handle.read(header_bytes)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase3ModelError("safetensors header JSON is invalid") from error
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if len(tensors) != 145:
        raise Phase3ModelError(f"unexpected safetensors tensor count: {len(tensors)}")
    offsets: list[tuple[int, int, str]] = []
    dtype_counts: dict[str, int] = {}
    for name, record in tensors.items():
        if set(record) != {"dtype", "shape", "data_offsets"}:
            raise Phase3ModelError(f"invalid safetensors entry schema: {name}")
        dtype = record["dtype"]
        if dtype not in _DTYPE_BYTES:
            raise Phase3ModelError(f"unsupported safetensors dtype: {dtype}")
        shape = record["shape"]
        start, end = record["data_offsets"]
        expected = math.prod(shape) * _DTYPE_BYTES[dtype]
        if start < 0 or end < start or end - start != expected:
            raise Phase3ModelError(f"invalid safetensors extent: {name}")
        offsets.append((start, end, name))
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    offsets.sort()
    if not offsets or offsets[0][0] != 0:
        raise Phase3ModelError("safetensors data does not begin at offset zero")
    for prior, following in pairwise(offsets):
        if prior[1] != following[0]:
            raise Phase3ModelError("safetensors tensor extents overlap or contain gaps")
    if 8 + header_bytes + offsets[-1][1] != source.stat().st_size:
        raise Phase3ModelError("safetensors final extent differs from file size")
    required = {
        "task_network.final.weight": ("F32", [12, 512]),
        "task_network.final.bias": ("F32", [12]),
        "transform.mean": ("F32", [12]),
        "transform.std": ("F32", [12]),
    }
    for name, (dtype, shape) in required.items():
        record = tensors.get(name)
        if record is None or record["dtype"] != dtype or record["shape"] != shape:
            raise Phase3ModelError(f"required safetensors tensor differs: {name}")
    keys = sorted(tensors)
    return {
        "schema_version": "qm9-mist-safetensors-audit-v1",
        "file_bytes": source.stat().st_size,
        "file_sha256": sha256_file(source),
        "header_bytes": header_bytes,
        "tensor_count": len(tensors),
        "dtype_counts": dtype_counts,
        "tensor_keys_sha256": canonical_hash(keys),
        "metadata": header.get("__metadata__"),
        "contiguous_nonoverlapping_complete_extents": True,
        "required_output_and_transform_tensors_verified": True,
    }


def _call_name(node: ast.Call) -> str:
    value = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def audit_remote_code(snapshot_dir: str | Path) -> dict[str, object]:
    """Record reviewed capabilities without importing the custom module."""

    directory = Path(snapshot_dir)
    path = directory / "modeling_mist_finetuned.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path.name)
    forbidden_calls: list[dict[str, object]] = []
    network_fallbacks: list[dict[str, object]] = []
    write_capabilities: list[dict[str, object]] = []
    dynamic_install_or_shell: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"eval", "exec", "compile", "__import__", "open"}:
            forbidden_calls.append({"line": node.lineno, "call": name})
        if leaf == "from_pretrained":
            network_fallbacks.append({"line": node.lineno, "call": name})
        if leaf == "save_pretrained":
            write_capabilities.append({"line": node.lineno, "call": name})
        if name.startswith(("subprocess.", "os.system", "os.popen")):
            dynamic_install_or_shell.append({"line": node.lineno, "call": name})
    notebook = json.loads((directory / "notebook.ipynb").read_text(encoding="utf-8"))
    notebook_source = "".join(
        line
        for cell in notebook.get("cells", [])
        for line in cell.get("source", [])
    )
    if forbidden_calls or dynamic_install_or_shell:
        raise Phase3ModelError("reviewed custom Python contains a forbidden execution primitive")
    return {
        "schema_version": "qm9-mist-remote-code-audit-v1",
        "reviewed_revision": MODEL_REVISION,
        "reviewed_python": path.name,
        "reviewed_python_bytes": path.stat().st_size,
        "reviewed_python_sha256": sha256_file(path),
        "forbidden_eval_exec_compile_import_open_calls": forbidden_calls,
        "subprocess_shell_calls": dynamic_install_or_shell,
        "network_capable_from_pretrained_calls": network_fallbacks,
        "write_capable_save_pretrained_calls": write_capabilities,
        "notebook_contains_dynamic_pip_install": "!pip" in notebook_source,
        "notebook_execution_forbidden": True,
        "runtime_guards": {
            "explicit_same_revision_local_tokenizer_required": True,
            "tokenizer_none_is_hard_error": True,
            "model_local_files_only": True,
            "model_revision_and_code_revision_pinned": MODEL_REVISION,
            "use_safetensors_only": True,
            "offline_environment_required": True,
            "save_pretrained_forbidden": True,
            "manual_inverse_transform_forbidden": True,
        },
        "review_result": "pass-under-mandatory-runtime-guards",
    }


def build_model_audit(snapshot_dir: str | Path) -> dict[str, object]:
    files = verify_snapshot(snapshot_dir)
    channels = validate_channels(snapshot_dir)
    remote = audit_remote_code(snapshot_dir)
    safetensors = audit_safetensors(Path(snapshot_dir) / "model.safetensors")
    return {
        "schema_version": "qm9-mist-model-audit-v1",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "allowlisted_files": files,
        "allowlisted_files_canonical_sha256": canonical_hash(files),
        "channels": channels,
        "channel_order": list(TARGET_COLUMNS),
        "checkpoint_unit_strings": list(EXPECTED_UNITS),
        "remote_code": remote,
        "safetensors": safetensors,
        "license_boundary": {
            "metadata": "apache-2.0",
            "model_card_use_restrictions": [
                "research-use-only",
                "no-redistribution-without-permission",
                "no-commercial-use-without-licensing-agreement",
            ],
            "conflict_policy": "apply-the-stricter-model-card-restrictions",
            "weights_redistributed": False,
        },
        "hard_gate_passed": True,
    }
