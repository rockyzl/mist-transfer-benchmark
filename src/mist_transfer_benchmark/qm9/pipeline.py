"""End-to-end Phase 1 QM9 source, split, and duplicate audit."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import tomllib

from .constants import (
    EXPECTED_CONTENT_TYPE,
    EXPECTED_ETAG,
    EXPECTED_HEADER,
    EXPECTED_LAST_MODIFIED,
    EXPECTED_ROW_COUNT,
    EXPECTED_SOURCE_BYTES,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SPLIT_COUNTS,
    FIRST_TEST_SIZE,
    IDENTITY_COLUMNS,
    QM9_URL,
    SECOND_TEST_SIZE,
    SPLIT_SEED,
    TARGET_COLUMNS,
)
from .data import validate_qm9_csv
from .download import (
    DownloadError,
    assert_source_unchanged,
    capture_source_snapshot,
    copy_validated_source,
    download_atomic,
)
from .duplicates import audit_duplicates, row_manifest
from .io import atomic_write_bytes, atomic_write_json, atomic_write_jsonl, sha256_file
from .output import (
    OutputSafetyError,
    discard_staging_workspace,
    finalize_output_workspace,
    prepare_output_workspace,
    write_owner_marker,
)
from .provenance import (
    ProvenanceError,
    assert_code_provenance_unchanged,
    capture_code_provenance,
)
from .split import (
    ReferenceContract,
    SplitMismatchError,
    load_and_verify_datasets_reference,
    reconstruct_candidate_split,
)


class QM9AuditError(ValueError):
    """Raised when a Phase 1 stop condition closes the pipeline."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit_distribution": _distribution_version("rdkit"),
            "rdkit_runtime": rdkit.__version__,
            "scipy": _distribution_version("scipy"),
            "scikit-learn": _distribution_version("scikit-learn"),
        },
    }


def _rss_gib(who: int) -> float:
    value = resource.getrusage(who).ru_maxrss
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return float(value) / divisor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QM9AuditError(message)


def _enforce_resource_limits(
    config: dict[str, object], *, parent_peak_rss_gib: float, child_peak_rss_gib: float
) -> None:
    resources = config["resource_budget"]
    if parent_peak_rss_gib > resources["max_parent_process_peak_rss_gib"]:
        raise QM9AuditError("parent audit process exceeded its per-process peak-RSS ceiling")
    if child_peak_rss_gib > resources["max_reference_child_peak_rss_gib"]:
        raise QM9AuditError("isolated reference exceeded its per-process peak-RSS ceiling")


def _validate_config(config: dict[str, object]) -> None:
    """Hard-fail code/config/runtime drift before cache or output I/O begins."""

    dataset = config["dataset"]
    head = dataset["http_head_observation"]
    upstream = dataset["upstream_declaration"]
    expectation = dataset["protocol_expectation"]
    local = dataset["local_verification"]
    reconstruction = config["reconstruction_environment"]
    split = config["split"]
    identity = config["identity_audit"]
    primary = config["cohorts"]["primary"]
    secondary = config["cohorts"]["secondary"]
    resources = config["resource_budget"]
    phase_one = config["phase_1_observation"]

    exact_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit_distribution": _distribution_version("rdkit"),
        "rdkit_runtime": rdkit.__version__,
        "scipy": _distribution_version("scipy"),
        "scikit_learn": _distribution_version("scikit-learn"),
    }
    for key, observed in exact_runtime.items():
        configured = str(reconstruction[key]).removeprefix("==")
        _require(
            observed == configured,
            f"runtime {key}={observed!r}; config requires {configured!r}",
        )

    _require(dataset["url"] == QM9_URL, "config source URL drifted")
    _require(head["content_length_bytes"] == EXPECTED_SOURCE_BYTES, "config source bytes drifted")
    _require(str(head["etag"]).strip('"') == EXPECTED_ETAG.strip('"'), "config ETag drifted")
    _require(local["content_bytes"] == EXPECTED_SOURCE_BYTES, "verified source bytes drifted")
    _require(local["content_sha256"] == EXPECTED_SOURCE_SHA256, "verified source SHA drifted")
    _require(local["etag"] == EXPECTED_ETAG, "verified source ETag drifted")
    _require(local["last_modified"] == EXPECTED_LAST_MODIFIED, "Last-Modified drifted")
    _require(local["content_type"] == EXPECTED_CONTENT_TYPE, "Content-Type drifted")
    _require(local["observed_header"] == list(EXPECTED_HEADER), "verified header drifted")
    _require(local["parsed_rows"] == EXPECTED_ROW_COUNT, "verified row count drifted")
    _require(upstream["row_count"] == EXPECTED_ROW_COUNT, "upstream row count drifted")
    _require(
        upstream["required_target_columns"] == list(TARGET_COLUMNS),
        "target column order drifted",
    )
    _require(
        expectation["identity_columns"] == list(IDENTITY_COLUMNS),
        "identity-column expectation drifted",
    )
    _require(
        expectation["identity_columns_upstream_declared_by_pinned_mist_sources"] is False,
        "mol_id/smiles provenance boundary drifted",
    )

    _require(split["library"] == "datasets==3.2.0", "split Datasets version drifted")
    _require(split["numpy"] == "==2.5.1", "split NumPy version drifted")
    _require(split["function"] == "Dataset.train_test_split", "split function drifted")
    _require(split["source_order_is_part_of_identity"] is True, "source-order rule drifted")
    _require(split["shuffle"] is True, "split shuffle rule drifted")
    _require(split["seed_first"] == SPLIT_SEED, "first split seed drifted")
    _require(split["first_test_size"] == FIRST_TEST_SIZE, "first split size drifted")
    _require(split["seed_second"] == SPLIT_SEED, "second split seed drifted")
    _require(split["second_test_size"] == SECOND_TEST_SIZE, "second split size drifted")
    _require(split["second_input"] == "first_test_partition", "second split input drifted")
    _require(split["second_train_name"] == "validation", "validation naming drifted")
    _require(split["second_test_name"] == "test", "test naming drifted")
    configured_counts = {
        "train": split["derived_train_rows_if_upstream_row_count_matches"],
        "validation": split["derived_validation_rows_if_upstream_row_count_matches"],
        "test": split["derived_test_rows_if_upstream_row_count_matches"],
    }
    _require(configured_counts == EXPECTED_SPLIT_COUNTS, "split counts drifted")

    for key in (
        "mutate_mist_input",
        "run_before_predictions",
        "audit_train_validation_overlap",
        "audit_train_test_overlap",
        "audit_validation_test_overlap",
        "audit_within_split_duplicates",
    ):
        expected = key != "mutate_mist_input"
        _require(identity[key] is expected, f"identity audit policy {key} drifted")
    _require(primary["id"] == "complete-candidate-reconstructed-test", "primary cohort drifted")
    _require(
        primary["derived_rows_if_reconstruction_matches"] == EXPECTED_SPLIT_COUNTS["test"],
        "primary cohort count drifted",
    )
    _require(
        secondary["exclude_if_identity_in"] == ["train", "validation"],
        "duplicate-clean exclusion policy drifted",
    )
    _require(secondary["deduplicate_within_test"] is True, "test deduplication rule drifted")
    _require(
        secondary["within_test_keep"] == "lowest-source-row-index",
        "within-test retention rule drifted",
    )
    _require(secondary["retrain_for_subset"] is False, "secondary retraining rule drifted")

    for key in (
        "max_parent_process_peak_rss_gib",
        "max_reference_child_peak_rss_gib",
        "reference_timeout_seconds",
        "max_reference_json_bytes",
    ):
        _require(type(resources[key]) is int and resources[key] > 0, f"resource {key} is invalid")
    _require(type(resources["cpu_worker_limit"]) is int, "CPU worker limit is invalid")
    _require(resources["cpu_worker_limit"] > 0, "CPU worker limit must be positive")
    _require(phase_one["reference_datasets"] == "3.2.0", "reference Datasets drifted")
    _require(phase_one["reference_numpy"] == "2.5.1", "reference NumPy drifted")
    _require(phase_one["reference_python"] == "3.12.12", "reference Python drifted")
    _require(phase_one["reference_pyarrow"] == "25.0.0", "reference PyArrow drifted")


def _validate_cache_dir(cache_dir: Path, repo_root: Path) -> Path:
    private_root = repo_root / "data" / "private"
    private_root.mkdir(parents=True, exist_ok=True)
    if ".." in cache_dir.parts:
        raise QM9AuditError("QM9 cache path traversal is not allowed")
    if private_root.is_symlink() or cache_dir.is_symlink():
        raise QM9AuditError("QM9 cache path must not use symlinks")
    requested = cache_dir if cache_dir.is_absolute() else repo_root / cache_dir
    resolved = requested.resolve(strict=False)
    if resolved.parent != private_root.resolve(strict=True):
        raise QM9AuditError("QM9 cache must be a direct child of data/private/")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _run_reference(
    python: Path,
    destination: Path,
    repo_root: Path,
    row_count: int,
    *,
    timeout_seconds: int,
) -> dict[str, object]:
    python = python.absolute()
    if not python.exists() or not os.access(python, os.X_OK):
        raise QM9AuditError(f"Datasets reference Python is not executable: {python}")
    command = [
        str(python),
        "-m",
        "mist_transfer_benchmark.qm9.reference",
        "--rows",
        str(row_count),
        "--seed",
        str(SPLIT_SEED),
        "--output",
        str(destination),
    ]
    with tempfile.TemporaryDirectory(prefix="qm9-reference-home-") as isolated_home:
        environment = {
            "HOME": isolated_home,
            "HF_HOME": str(Path(isolated_home) / "hf"),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(repo_root / "src"),
            "PATH": os.pathsep.join((str(python.parent), "/usr/bin", "/bin")),
            "TMPDIR": isolated_home,
            "TZ": "UTC",
        }
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            raise QM9AuditError(
                f"Datasets reference exceeded {timeout_seconds}s and was terminated"
            ) from error
    if process.returncode != 0:
        raise QM9AuditError(
            "isolated Datasets reference failed: "
            f"exit={process.returncode}\nstdout={stdout}\nstderr={stderr}"
        )
    return {
        "command": command,
        "cwd": str(repo_root),
        "environment_policy": "sanitized-offline-no-inherited-pythonpath",
        "timeout_seconds": timeout_seconds,
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "rusage_children_peak_rss_gib_after_wait": _rss_gib(resource.RUSAGE_CHILDREN),
    }


def _reference_contract(
    config: dict[str, object], datasets_python: Path, repo_root: Path
) -> ReferenceContract:
    phase_one = config["phase_1_observation"]
    resources = config["resource_budget"]
    reference_source = repo_root / "src/mist_transfer_benchmark/qm9/reference.py"
    return ReferenceContract(
        row_count=EXPECTED_ROW_COUNT,
        seed=SPLIT_SEED,
        counts=EXPECTED_SPLIT_COUNTS,
        python=phase_one["reference_python"],
        datasets=phase_one["reference_datasets"],
        numpy=phase_one["reference_numpy"],
        pyarrow=phase_one["reference_pyarrow"],
        pandas="3.0.3",
        fsspec="2024.9.0",
        environment_freeze_canonical_json_sha256=phase_one[
            "reference_environment_freeze_canonical_json_sha256"
        ],
        train_test_split_source_sha256=phase_one[
            "reference_train_test_split_source_sha256"
        ],
        reference_source_sha256=sha256_file(reference_source),
        executable=str(datasets_python.absolute()),
        prefix=str(datasets_python.absolute().parent.parent),
        max_json_bytes=resources["max_reference_json_bytes"],
    )


def _assignment_tsv(split_names: np.ndarray) -> bytes:
    lines = ["source_row_index\tsplit_name\n"]
    lines.extend(f"{index}\t{name}\n" for index, name in enumerate(split_names))
    return "".join(lines).encode("utf-8")


def run_phase1_audit(
    *,
    config_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    datasets_python: str | Path,
    force_download: bool = False,
    overwrite: bool = False,
    command: list[str] | None = None,
    progress=print,
) -> dict[str, object]:
    """Run Phase 1 with authenticated source, code, output, and reference boundaries."""

    started_at = _now()
    started_monotonic = time.monotonic()
    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    _validate_config(config)
    initial_code_provenance = capture_code_provenance(repo_root, config_path)
    workspace = prepare_output_workspace(output_dir, repo_root, overwrite=overwrite)
    cache_dir = _validate_cache_dir(Path(cache_dir), repo_root)
    staging = workspace.staging_dir
    try:
        source_path = cache_dir / "qm9.csv"
        cached_retrieval_path = cache_dir / "qm9.download.json"
        if cached_retrieval_path.is_symlink():
            raise QM9AuditError("retrieval record must not be a symlink")
        previous_retrieval = None
        if cached_retrieval_path.is_file() and source_path.is_file() and not force_download:
            previous_retrieval = json.loads(cached_retrieval_path.read_text(encoding="utf-8"))
        progress(f"retrieving or verifying local source cache: {source_path}")
        local = config["dataset"]["local_verification"]
        download = download_atomic(
            config["dataset"]["url"],
            source_path,
            expected_bytes=EXPECTED_SOURCE_BYTES,
            expected_sha256=EXPECTED_SOURCE_SHA256,
            expected_etag=local["etag"],
            expected_last_modified=local["last_modified"],
            expected_content_type=local["content_type"],
            force=force_download,
        )
        if download.retrieval_mode == "atomic-http-get":
            atomic_write_json(cached_retrieval_path, download.to_dict(), mode=0o600)
            retrieval_evidence = download.to_dict()
        elif previous_retrieval is not None:
            if previous_retrieval.get("sha256") != download.sha256:
                raise QM9AuditError("cached CSV hash differs from its original retrieval record")
            retrieval_evidence = {
                "original_retrieval": previous_retrieval,
                "cache_reuse": download.to_dict(),
            }
        else:
            retrieval_evidence = download.to_dict()

        cache_snapshot = capture_source_snapshot(
            source_path,
            expected_bytes=EXPECTED_SOURCE_BYTES,
            expected_sha256=EXPECTED_SOURCE_SHA256,
        )
        private_source_path = staging / "source.snapshot.csv"
        private_snapshot = copy_validated_source(
            source_path,
            private_source_path,
            cache_snapshot,
            expected_bytes=EXPECTED_SOURCE_BYTES,
            expected_sha256=EXPECTED_SOURCE_SHA256,
        )
        progress("validating exact CSV schema, row identities, and 12 finite target columns")
        data = validate_qm9_csv(private_source_path)
        progress("reconstructing the local two-stage seed=42 candidate split")
        split = reconstruct_candidate_split(data.row_count)
        reference_path = staging / "datasets_reference.json"
        datasets_python = Path(datasets_python)
        reference_command = _run_reference(
            datasets_python,
            reference_path,
            repo_root,
            data.row_count,
            timeout_seconds=config["resource_budget"]["reference_timeout_seconds"],
        )
        reference = load_and_verify_datasets_reference(
            reference_path,
            split,
            contract=_reference_contract(config, datasets_python, repo_root),
        )
        child_peak_rss_gib = reference["environment"]["resource_usage"]["peak_rss_gib"]
        _enforce_resource_limits(
            config, parent_peak_rss_gib=0.0, child_peak_rss_gib=child_peak_rss_gib
        )
        reference_freeze_path = staging / "datasets_environment.freeze.txt"
        freeze_bytes = (
            "\n".join(reference["environment"]["environment_freeze"]) + "\n"
        ).encode("utf-8")
        atomic_write_bytes(reference_freeze_path, freeze_bytes, mode=0o600)
        progress("Datasets 3.2.0 membership matches exactly; starting RDKit duplicate audit")
        duplicate_audit = audit_duplicates(data, split, progress=progress)

        assert_source_unchanged(
            private_source_path,
            private_snapshot,
            expected_bytes=EXPECTED_SOURCE_BYTES,
            expected_sha256=EXPECTED_SOURCE_SHA256,
        )
        assert_source_unchanged(
            source_path,
            cache_snapshot,
            expected_bytes=EXPECTED_SOURCE_BYTES,
            expected_sha256=EXPECTED_SOURCE_SHA256,
        )
        assert_code_provenance_unchanged(initial_code_provenance, repo_root, config_path)

        split_names, _ = split.assignment_arrays(data.row_count)
        assignment_path = staging / "split_assignments.tsv"
        assignment_bytes = _assignment_tsv(split_names)
        atomic_write_bytes(assignment_path, assignment_bytes, mode=0o600)
        assignment_rows_sha256 = hashlib.sha256(assignment_bytes.split(b"\n", 1)[1]).hexdigest()
        row_manifest_path = staging / "row_manifest.jsonl"
        manifest_rows = atomic_write_jsonl(
            row_manifest_path, row_manifest(data, split, duplicate_audit), mode=0o600
        )
        events_path = staging / "duplicate_events.jsonl"
        event_rows = atomic_write_jsonl(events_path, duplicate_audit.events, mode=0o600)
        duplicate_summary_path = staging / "duplicate_summary.json"
        atomic_write_json(duplicate_summary_path, duplicate_audit.summary, mode=0o600)
        provenance_path = staging / "code_provenance.json"
        atomic_write_json(provenance_path, initial_code_provenance, mode=0o600)

        source_manifest = {
            "schema_version": "qm9-source-manifest-v2",
            "source_path_as_invoked": str(source_path),
            "source_bytes": cache_snapshot.bytes,
            "source_sha256": cache_snapshot.sha256,
            "retrieval_evidence": retrieval_evidence,
            "validated_private_snapshot": private_snapshot.to_dict(),
            "final_cache_and_private_snapshot_equality_verified": True,
            "validation": data.metadata(),
            "protocol_config": str(config_path.relative_to(repo_root)),
            "protocol_config_sha256": sha256_file(config_path),
            "expected_header": list(EXPECTED_HEADER),
        }
        source_manifest_path = staging / "source_manifest.json"
        atomic_write_json(source_manifest_path, source_manifest, mode=0o600)
        private_source_path.unlink()

        artifact_paths = {
            "source_manifest": source_manifest_path,
            "code_provenance": provenance_path,
            "datasets_reference": reference_path,
            "datasets_environment_freeze": reference_freeze_path,
            "split_assignments": assignment_path,
            "row_manifest": row_manifest_path,
            "duplicate_events": events_path,
            "duplicate_summary": duplicate_summary_path,
        }
        artifact_manifest = {
            name: {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        }
        parent_peak_rss_gib = _rss_gib(resource.RUSAGE_SELF)
        _enforce_resource_limits(
            config,
            parent_peak_rss_gib=parent_peak_rss_gib,
            child_peak_rss_gib=child_peak_rss_gib,
        )
        run = {
            "schema_version": "qm9-phase1-run-v2",
            "scientific_status": "data-split-duplicate-audit-only-no-model-result",
            "candidate_split_not_publisher_certified": True,
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "runtime_seconds": time.monotonic() - started_monotonic,
            "resource_usage": {
                "semantics": "separate per-process peaks; not a process-group aggregate",
                "parent_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "parent_peak_rss_gib": parent_peak_rss_gib,
                "reference_child_reported_peak_rss_gib": child_peak_rss_gib,
                "rusage_children_peak_rss_gib_after_wait": reference_command[
                    "rusage_children_peak_rss_gib_after_wait"
                ],
            },
            "command": command,
            "source": source_manifest,
            "split": {
                "counts": split.counts(),
                "ordered_index_sha256": split.ordered_hashes(),
                "membership_sha256": split.membership_hashes(),
                "assignment_tsv_sha256": sha256_file(assignment_path),
                "assignment_rows_sha256": assignment_rows_sha256,
                "assignment_serialization": (
                    "UTF-8 TSV with header and final LF; assignment_rows_sha256 excludes header"
                ),
                "datasets_reference_environment": reference["environment"],
                "datasets_reference_command": reference_command,
                "exact_ordered_membership_match": True,
            },
            "duplicates": duplicate_audit.summary,
            "row_manifest_rows": manifest_rows,
            "duplicate_event_rows": event_rows,
            "code_provenance": {
                "aggregate_sha256": initial_code_provenance["aggregate_sha256"],
                "file_manifest_canonical_json_sha256": initial_code_provenance[
                    "file_manifest_canonical_json_sha256"
                ],
                "git_head": initial_code_provenance["git_head"],
                "git_dirty_in_scope": initial_code_provenance["git_dirty_in_scope"],
            },
            "environment": _runtime_metadata(),
            "artifacts": artifact_manifest,
            "stop_conditions_triggered": [],
            "next_phase_authorized": False,
        }
        _require(split.counts() == EXPECTED_SPLIT_COUNTS, "split counts changed unexpectedly")
        assert_code_provenance_unchanged(initial_code_provenance, repo_root, config_path)
        run_path = staging / "phase1_run.json"
        atomic_write_json(run_path, run, mode=0o600)
        atomic_write_bytes(
            staging / "phase1_run.sha256",
            f"{sha256_file(run_path)}  {run_path.name}\n".encode("ascii"),
            mode=0o600,
        )
        write_owner_marker(staging)
        finalize_output_workspace(workspace)
        run["phase1_run_sha256"] = sha256_file(workspace.output_dir / "phase1_run.json")
        run["output_dir"] = str(workspace.output_dir)
        return run
    except (
        DownloadError,
        OSError,
        OutputSafetyError,
        ProvenanceError,
        SplitMismatchError,
        subprocess.SubprocessError,
    ) as error:
        raise QM9AuditError(str(error)) from error
    finally:
        discard_staging_workspace(workspace)
