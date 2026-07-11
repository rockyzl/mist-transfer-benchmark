"""Isolated offline worker for guarded released-MIST inference."""

from __future__ import annotations

import argparse
import inspect
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .io import atomic_write_json, canonical_hash, iter_jsonl, sha256_file
from .phase3_adapter import batched_predict, stack_named_outputs
from .phase3_model import (
    MODEL_REVISION,
    build_model_audit,
    validate_channels,
    verify_snapshot,
)


class Phase3WorkerError(ValueError):
    """Raised when a guarded worker invariant fails."""


def _rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return float(value) / divisor


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_inputs(path: Path) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    smiles: list[str] = []
    for expected_position, row in enumerate(iter_jsonl(path)):
        if set(row) != {"position", "source_row_index", "source_smiles"}:
            raise Phase3WorkerError("worker input row schema differs")
        if row["position"] != expected_position:
            raise Phase3WorkerError("worker input positions are not contiguous")
        if type(row["source_row_index"]) is not int:
            raise Phase3WorkerError("worker source index is not an integer")
        if not isinstance(row["source_smiles"], str) or not row["source_smiles"]:
            raise Phase3WorkerError("worker input contains an empty raw SMILES")
        indices.append(row["source_row_index"])
        smiles.append(row["source_smiles"])
    if not indices or len(set(indices)) != len(indices):
        raise Phase3WorkerError("worker inputs are empty or contain duplicate source indices")
    return indices, smiles


def run_worker(
    *,
    snapshot: Path,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    batch_size: int,
    requested_device: str,
) -> dict[str, object]:
    """Execute only the exact statically reviewed local snapshot."""

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise Phase3WorkerError(f"offline environment guard is missing: {name}")
    started = time.monotonic()
    verify_snapshot(snapshot)
    channels = validate_channels(snapshot)
    audit = build_model_audit(snapshot)
    if audit["hard_gate_passed"] is not True:
        raise Phase3WorkerError("model audit gate is not passed")
    indices, smiles = _load_inputs(input_path)

    import torch
    from smirk import SmirkTokenizerFast
    from transformers import AutoModel

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise Phase3WorkerError("CUDA was requested but is unavailable")
        device = torch.device("cuda:0")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        raise Phase3WorkerError("worker device must be explicit cpu or cuda")
    tokenizer = SmirkTokenizerFast.from_pretrained(
        str(snapshot),
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    if tokenizer is None:
        raise Phase3WorkerError("explicit same-revision tokenizer resolved to None")
    model = AutoModel.from_pretrained(
        str(snapshot),
        revision=MODEL_REVISION,
        code_revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
    )
    signature = inspect.signature(model.predict)
    if "tokenizer" not in signature.parameters:
        raise Phase3WorkerError("reviewed predict API cannot accept an explicit tokenizer")
    model.to(device)
    model.eval()
    if model.training:
        raise Phase3WorkerError("model.eval() did not disable training mode")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch_counter = 0

    def predict_batch(batch: list[str]) -> np.ndarray:
        nonlocal batch_counter
        if tokenizer is None:
            raise Phase3WorkerError("tokenizer unexpectedly became None")
        with torch.inference_mode():
            named = model.predict(batch, return_dict=True, tokenizer=tokenizer)
        batch_counter += 1
        return stack_named_outputs(named, expected_rows=len(batch))

    def progress(done: int, total: int) -> None:
        print(f"MIST_INFERENCE_PROGRESS {done}/{total}", flush=True)

    predictions = batched_predict(
        smiles,
        batch_size=batch_size,
        predict_batch=predict_batch,
        progress=progress,
    )
    _save_npy_atomic(output_path, predictions)
    gpu_peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    gpu_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )
    report = {
        "schema_version": "qm9-mist-worker-report-v1",
        "model_revision": MODEL_REVISION,
        "model_local_files_only": True,
        "code_revision_pinned": MODEL_REVISION,
        "use_safetensors": True,
        "explicit_same_revision_tokenizer": True,
        "tokenizer_none_fails_closed": True,
        "model_eval": not model.training,
        "torch_inference_mode": True,
        "manual_inverse_transform_applied": False,
        "model_predict_returns_native_units": True,
        "named_outputs_stacked_by_config_order": True,
        "channel_order": [channel["name"] for channel in channels],
        "channel_units": [channel["unit"] for channel in channels],
        "input_rows": len(smiles),
        "input_source_indices_sha256": canonical_hash(indices),
        "input_jsonl_sha256": sha256_file(input_path),
        "output_shape": list(predictions.shape),
        "output_dtype": str(predictions.dtype),
        "output_npy_sha256": sha256_file(output_path),
        "output_canonical_sha256": canonical_hash(predictions.tolist()),
        "batch_size": batch_size,
        "batch_count": batch_counter,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "gpu_peak_memory_allocated_bytes": gpu_peak,
        "gpu_peak_memory_reserved_bytes": gpu_reserved,
        "parent_peak_rss_gib": _rss_gib(),
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_write_json(report_path, report, mode=0o600)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(argv)
    run_worker(
        snapshot=args.snapshot,
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        batch_size=args.batch_size,
        requested_device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
