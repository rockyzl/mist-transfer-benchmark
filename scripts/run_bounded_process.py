#!/usr/bin/env python3
"""Run one command in a fail-closed systemd/cgroup-v2 resource envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_WALL_SECONDS = 60.0
MAX_RSS_BYTES = 1_610_612_736
MAX_POLL_SECONDS = 1.0
MAX_COMMAND_SECONDS = 55.0
FINAL_VERIFY_SECONDS = 0.1
MEDIUM_PROFILE = "medium-v1"
PUBLIC_FIXED_PROFILE = "public-fixed-v1"
MEDIUM_WALL_SECONDS = 2400.0
MEDIUM_RSS_BYTES = 4 * 1024**3
MEDIUM_COMMAND_SECONDS = 2280.0
MEDIUM_AUTH_SCHEMA = "qm9-v31-medium-profile-authorization-v1"
PUBLIC_FIXED_AUTH_SCHEMA = "qm9-v31-public-fixed-authorization-v1"
MEDIUM_PRODUCTION_RUN_ID = "qm9-v31-medium-profile-cpu-v1"
MEDIUM_PUBLIC_SYNTHETIC_RUN_ID = "qm9-v31-medium-profile-public-synthetic-v1"
PUBLIC_FIXED_RUN_ID = "qm9-v31-public-fixed-three-backend-cpu-v1"
PUBLIC_FIXED_PURPOSE = "public_synthetic_three_backend_engineering_seam"


class HarnessError(RuntimeError):
    """Fail-closed harness error."""


class SensorError(HarnessError):
    """The external process-group sensor is unavailable."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reserve_result(path: Path) -> int:
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o200)


def _write_result(descriptor: int, path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    try:
        view = memoryview(encoded)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    _fsync_parent(path)


def _write_reserved_failure(
    descriptor: int,
    path: Path,
    *,
    reason: str,
    error: BaseException,
) -> int:
    """Seal a minimal result after reservation and before command spawn."""
    _write_result(
        descriptor,
        path,
        {
            "schema_version": "bounded-process-result-v2",
            "state": "failed",
            "reason": reason,
            "error_type": type(error).__name__,
            "pid": None,
            "spawned": False,
            "finished_at_utc": _utc_now(),
        },
    )
    return 1


def _repo_code_state(path: Path) -> dict[str, object]:
    try:
        root = Path(
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise HarnessError("repository code state is unavailable") from error
    manifest = []
    for raw_relative in sorted(item for item in listed.split(b"\0") if item):
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        source = root / relative
        if source.is_file() and not source.is_symlink():
            manifest.append(
                {"path": relative, "bytes": source.stat().st_size, "sha256": _file_sha256(source)}
            )
    record: dict[str, object] = {
        "repository_root": str(root),
        "head": head,
        "file_manifest_sha256": _canonical_sha256(manifest),
        "file_count": len(manifest),
    }
    record["code_state_sha256"] = _canonical_sha256(record)
    return record


def _capability_preflight(
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, object]:
    try:
        if (Path("/sys/fs/cgroup").stat().st_mode & 0o170000) != 0o040000:
            raise HarnessError("cgroup root is unavailable")
        controllers = Path("/sys/fs/cgroup/cgroup.controllers").read_text().split()
        if "memory" not in controllers:
            raise HarnessError("cgroup v2 memory controller is unavailable")
        version = runner(
            ["systemctl", "--user", "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
        runner(
            ["systemctl", "--user", "show-environment"],
            check=True,
            capture_output=True,
            text=True,
        )
        probe = f"bounded-capability-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        runner(
            [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                f"--unit={probe}",
                "--property=MemoryMax=16777216",
                "--",
                "/usr/bin/true",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        raise HarnessError("systemd user cgroup capability is unavailable") from error
    return {
        "cgroup_filesystem": "cgroup2",
        "memory_controller": True,
        "systemd_version": version,
    }


def _read_proc_table() -> dict[int, dict[str, int]]:
    table: dict[int, dict[str, int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as error:
        raise SensorError("cannot enumerate /proc") from error
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            resident_pages = int((entry / "statm").read_text().split()[1])
            table[int(entry.name)] = {
                "ppid": int(fields[1]),
                "pgrp": int(fields[2]),
                "session": int(fields[3]),
                "starttime": int(fields[19]),
                "rss": resident_pages * page_size,
            }
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    return table


def _sample_group(pgid: int, session_id: int) -> tuple[list[int], int]:
    table = _read_proc_table()
    members = sorted(
        pid for pid, row in table.items() if row["pgrp"] == pgid and row["session"] == session_id
    )
    return members, sum(table[pid]["rss"] for pid in members)


def _unit_metrics(unit: str) -> dict[str, object]:
    try:
        output = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=MemoryCurrent,MemoryPeak,MemoryMax,ControlGroup,Result,ActiveState,LoadState",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if not values.get("ControlGroup") and values.get("MemoryMax") in (None, "", "infinity"):
        return {}
    result: dict[str, object] = {
        "result": values.get("Result") or None,
        "active_state": values.get("ActiveState") or None,
        "load_state": values.get("LoadState") or None,
    }
    for source, target in (
        ("MemoryCurrent", "memory_current_bytes"),
        ("MemoryPeak", "memory_peak_bytes"),
        ("MemoryMax", "memory_max_bytes"),
    ):
        try:
            result[target] = int(values[source])
        except (KeyError, ValueError):
            result[target] = None
    control_group = values.get("ControlGroup") or None
    result["control_group"] = control_group
    events: dict[str, int] = {}
    if control_group:
        event_path = Path("/sys/fs/cgroup") / control_group.lstrip("/") / "memory.events"
        try:
            events = {
                key: int(value)
                for key, value in (line.split() for line in event_path.read_text().splitlines())
            }
        except (OSError, ValueError):
            events = {}
    result["memory_events"] = events
    return result


def _cgroup_sample(control_group: str) -> tuple[list[int], int, int, int]:
    root = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    if not root.is_dir():
        raise SensorError("cgroup path is unavailable")
    try:
        members = sorted(
            {
                int(pid)
                for procs in [root / "cgroup.procs", *root.rglob("cgroup.procs")]
                for pid in procs.read_text().split()
            }
        )
        current = int((root / "memory.current").read_text())
        peak = int((root / "memory.peak").read_text())
        events = dict(line.split() for line in (root / "cgroup.events").read_text().splitlines())
        populated = int(events["populated"])
    except (OSError, ValueError, KeyError) as error:
        raise SensorError("cgroup membership/resource sensor is unavailable") from error
    return members, current, peak, populated


def _unit_signal(unit: str, signame: str) -> dict[str, object]:
    command = [
        "systemctl",
        "--user",
        "kill",
        "--kill-whom=all",
        f"--signal={signame}",
        unit,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=1)
    return {"timestamp_utc": _utc_now(), "signal": signame, "returncode": completed.returncode}


def _cgroup_kill(control_group: str | None) -> bool:
    if not control_group:
        return False
    path = Path("/sys/fs/cgroup") / control_group.lstrip("/") / "cgroup.kill"
    try:
        path.write_text("1\n")
    except OSError:
        return False
    return True


def _stop_unit(unit: str) -> int:
    return subprocess.run(
        ["systemctl", "--user", "stop", unit],
        check=False,
        capture_output=True,
        timeout=1,
    ).returncode


def _unit_inactive(unit: str) -> bool:
    metrics = _unit_metrics(unit)
    return (
        not metrics
        or metrics.get("active_state") in {"inactive", "failed"}
        or metrics.get("load_state") == "not-found"
    )


def _unit_collected_success(unit: str, child_exit_code: int | None) -> bool:
    if child_exit_code is None:
        return False
    metrics = _unit_metrics(unit)
    if not metrics:
        return True
    inactive = metrics.get("active_state") == "inactive" or metrics.get("load_state") == "not-found"
    return inactive and metrics.get("result") in (None, "success")


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def _spawn(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str] | None = None
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        list(command), cwd=cwd, env=environment, start_new_session=True
    )


def _status_payload(
    *,
    started: float,
    pid: int | None,
    pgid: int | None,
    members: Sequence[int],
    rss: int,
    peak_rss: int,
    state: str,
    reason: str,
) -> dict[str, object]:
    descendants = [member for member in members if member != pid]
    return {
        "timestamp": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "pid": pid,
        "pgid": pgid,
        "live_descendants": descendants,
        "live_descendant_count": len(descendants),
        "group_members": list(members),
        "aggregate_rss_bytes": rss,
        "peak_rss_bytes": peak_rss,
        "state": state,
        "reason": reason,
    }


def run_bounded(
    command: list[str],
    *,
    status_path: Path,
    result_path: Path,
    cwd: Path | None = None,
    wall_seconds: float = MAX_WALL_SECONDS,
    rss_bytes: int = MAX_RSS_BYTES,
    poll_seconds: float = 0.5,
    term_grace_seconds: float = 3.0,
    command_seconds: float | None = None,
    expected_command_sha256: str | None = None,
    expected_harness_sha256: str | None = None,
    expected_code_state_sha256: str | None = None,
    expected_cwd: Path | None = None,
    expected_bound_files: Mapping[Path, str] | None = None,
    profile: str = "default",
    expected_profile_path: Path | None = None,
    expected_profile_sha256: str | None = None,
    wrapper_argv: Sequence[str] | None = None,
    capability_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    if not command:
        raise ValueError("command must be nonempty")
    if profile not in {"default", MEDIUM_PROFILE, PUBLIC_FIXED_PROFILE}:
        raise ValueError("bounded profile mode differs")
    maximum_wall = MAX_WALL_SECONDS
    maximum_rss = MAX_RSS_BYTES
    maximum_command = MAX_COMMAND_SECONDS
    if profile in {MEDIUM_PROFILE, PUBLIC_FIXED_PROFILE}:
        if (
            wall_seconds != MEDIUM_WALL_SECONDS
            or rss_bytes != MEDIUM_RSS_BYTES
            or command_seconds != MEDIUM_COMMAND_SECONDS
            or expected_profile_path is None
            or expected_profile_sha256 is None
        ):
            raise ValueError("medium-v1 requires exact reviewed limits and profile binding")
        profile_path = expected_profile_path.resolve()
        if (
            profile_path.is_symlink()
            or not profile_path.is_file()
            or _file_sha256(profile_path) != expected_profile_sha256
        ):
            raise HarnessError("medium-v1 reviewed profile physical binding differs")
        try:
            reviewed_profile = json.loads(profile_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise HarnessError("medium-v1 reviewed profile cannot be loaded") from error
        unsigned_profile = dict(reviewed_profile)
        profile_semantic = unsigned_profile.pop("semantic_sha256", None)
        expected_run_id = PUBLIC_FIXED_RUN_ID if profile == PUBLIC_FIXED_PROFILE else (
            MEDIUM_PUBLIC_SYNTHETIC_RUN_ID
            if os.environ.get("QM9_MEDIUM_PUBLIC_SYNTHETIC_SEAM") == "1"
            else MEDIUM_PRODUCTION_RUN_ID
        )
        expected_purpose = (
            PUBLIC_FIXED_PURPOSE
            if profile == PUBLIC_FIXED_PROFILE
            else "train_only_medium_profile"
        )
        expected_schema = (
            PUBLIC_FIXED_AUTH_SCHEMA
            if profile == PUBLIC_FIXED_PROFILE
            else MEDIUM_AUTH_SCHEMA
        )
        if (
            reviewed_profile.get("schema_version") != expected_schema
            or reviewed_profile.get("run_id") != expected_run_id
            or reviewed_profile.get("purpose") != expected_purpose
            or reviewed_profile.get("one_use") is not True
            or profile_semantic != _canonical_sha256(unsigned_profile)
            or any(
                reviewed_profile.get(key) is not False
                for key in (
                    "outer_validation_targets_read", "test_targets_read",
                    "private_evaluation_labels_read", "protected_labels_read",
                    "ranking_allowed", "ensemble_allowed", "promotion_allowed", "w1_allowed",
                    "general_training_allowed", "scientific_result", "publication_ready",
                )
            )
        ):
            raise HarnessError("medium-v1 reviewed profile semantic scope differs")
        maximum_wall = MEDIUM_WALL_SECONDS
        maximum_rss = MEDIUM_RSS_BYTES
        maximum_command = MEDIUM_COMMAND_SECONDS
    elif expected_profile_path is not None or expected_profile_sha256 is not None:
        raise ValueError("default profile does not accept medium profile evidence")
    if not 0 < wall_seconds <= maximum_wall:
        raise ValueError(f"wall_seconds must be in (0, {maximum_wall}]")
    if not 0 < rss_bytes <= maximum_rss:
        raise ValueError(f"rss_bytes must be in (0, {maximum_rss}]")
    if not 0 < poll_seconds <= MAX_POLL_SECONDS:
        raise ValueError(f"poll_seconds must be in (0, {MAX_POLL_SECONDS}]")
    if not 0.05 <= term_grace_seconds < wall_seconds:
        raise ValueError("term grace must be >=0.05 seconds and less than wall limit")
    command_seconds = command_seconds or min(maximum_command, wall_seconds - 0.1)
    if not 0 < command_seconds <= maximum_command or command_seconds > wall_seconds - 0.1:
        raise ValueError("command deadline must leave result-finalization headroom")

    status_path = status_path.resolve()
    result_path = result_path.resolve()
    execution_cwd = (cwd or Path.cwd()).resolve()
    if status_path == result_path:
        raise ValueError("status and result paths must differ")
    if status_path.exists() or result_path.exists():
        raise FileExistsError("status and result paths must be fresh")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_descriptor = _reserve_result(result_path)

    try:
        harness_path = Path(__file__).resolve()
        harness_sha256 = _file_sha256(harness_path)
        command_sha256 = _canonical_sha256(command)
        code_state = _repo_code_state(harness_path.parent)
    except BaseException as error:
        return _write_reserved_failure(
            result_descriptor,
            result_path,
            reason="CODE_STATE_OR_PROVENANCE_UNAVAILABLE",
            error=error,
        )
    try:
        bound_files = {
            str(path.resolve()): {
                "expected_sha256": expected,
                "observed_sha256": _file_sha256(path.resolve()),
            }
            for path, expected in (expected_bound_files or {}).items()
        }
    except BaseException as error:
        return _write_reserved_failure(
            result_descriptor,
            result_path,
            reason="BOUND_FILE_UNAVAILABLE",
            error=error,
        )
    provenance = {
        "command": command,
        "command_sha256": command_sha256,
        "wrapper_argv": list(wrapper_argv) if wrapper_argv is not None else None,
        "wrapper_argv_sha256": (
            _canonical_sha256(list(wrapper_argv)) if wrapper_argv is not None else None
        ),
        "harness_path": str(harness_path),
        "harness_code_sha256": harness_sha256,
        "execution_cwd": str(execution_cwd),
        "repository": code_state,
        "bound_files": bound_files,
        "launch_parameters": {
            "profile": profile,
            "wall_seconds": wall_seconds,
            "memory_max_bytes": rss_bytes,
            "poll_seconds": poll_seconds,
            "term_grace_seconds": term_grace_seconds,
            "command_seconds": command_seconds,
            "status_path": str(status_path),
            "result_path": str(result_path),
        },
    }
    mismatches = {
        "command": expected_command_sha256 not in (None, command_sha256),
        "harness": expected_harness_sha256 not in (None, harness_sha256),
        "code_state": expected_code_state_sha256 not in (None, code_state["code_state_sha256"]),
        "cwd": expected_cwd is not None and expected_cwd.resolve() != execution_cwd,
        "bound_files": any(
            row["expected_sha256"] != row["observed_sha256"] for row in bound_files.values()
        ),
    }

    started = time.monotonic()
    started_utc = _utc_now()
    process: subprocess.Popen[bytes] | None = None
    pid: int | None = None
    pgid: int | None = None
    session_id: int | None = None
    leader_starttime: int | None = None
    peak_rss = 0
    reason = "RUNNING"
    requested_signal: int | None = None
    term_at: str | None = None
    kill_at: str | None = None
    term_survivors: list[int] = []
    kill_survivors: list[int] = []
    group_empty = False
    cgroup_empty = False
    action_evidence: list[dict[str, object]] = []
    child_exit_code: int | None = None
    unit_metrics: dict[str, object] = {}
    status_error: str | None = None
    unit = f"bounded-{os.getpid()}-{uuid.uuid4().hex[:12]}.scope"
    properties = {
        "MemoryMax": rss_bytes,
        "OOMPolicy": "kill",
        "RuntimeMaxSec": command_seconds,
    }
    capability: dict[str, object] = {}
    cgroup_verified = False
    control_group: str | None = None
    terminal_status_written = False

    def receive_signal(signum: int, _frame: Any) -> None:
        nonlocal requested_signal
        requested_signal = signum

    installed = {
        signum: signal.signal(signum, receive_signal) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def append_status(state: str, current_reason: str, members: Sequence[int], rss: int) -> None:
        nonlocal status_error, terminal_status_written
        if terminal_status_written:
            raise HarnessError("status append after terminal state")
        payload = _status_payload(
            started=started,
            pid=pid,
            pgid=pgid,
            members=members,
            rss=rss,
            peak_rss=peak_rss,
            state=state,
            reason=current_reason,
        )
        try:
            _append_jsonl(status_path, payload)
            if state in {"complete", "failed"}:
                terminal_status_written = True
        except BaseException as error:
            status_error = type(error).__name__
            raise HarnessError("status append failed") from error

    try:
        status_path.touch(mode=0o600, exist_ok=False)
        _fsync_parent(status_path)
        try:
            capability = _capability_preflight(runner=capability_runner)
        except BaseException:
            reason = "CAPABILITY_PREFLIGHT_FAILED"
            raise
        if any(mismatches.values()):
            reason = "PROVENANCE_MISMATCH"
            raise HarnessError(reason)
        wrapped = [
            "systemd-run",
            "--user",
            "--scope",
            "--collect",
            "--quiet",
            f"--unit={unit.removesuffix('.scope')}",
            f"--property=MemoryMax={rss_bytes}",
            "--property=OOMPolicy=kill",
            f"--property=RuntimeMaxSec={command_seconds}",
            "--",
            *command,
        ]
        child_environment = os.environ.copy()
        if profile in {MEDIUM_PROFILE, PUBLIC_FIXED_PROFILE}:
            child_environment["QM9_MEDIUM_OUTER_HARNESS_BINDING"] = str(
                expected_profile_sha256
            )
        if profile == PUBLIC_FIXED_PROFILE:
            child_environment["QM9_V31_PUBLIC_FIXED_BACKENDS"] = "1"
        process = _spawn(wrapped, cwd=execution_cwd, environment=child_environment)
        pid = process.pid
        readiness_deadline = min(started + command_seconds, time.monotonic() + 1.0)
        while time.monotonic() < readiness_deadline:
            unit_metrics = _unit_metrics(unit)
            control_group = unit_metrics.get("control_group") if unit_metrics else None
            if unit_metrics.get("memory_max_bytes") == rss_bytes and control_group:
                cgroup_verified = True
                break
            time.sleep(0.01)
        if not cgroup_verified or not control_group:
            reason = "CGROUP_SENSOR_UNAVAILABLE"
            raise SensorError(reason)
        pgid = os.getpgid(pid)
        session_id = os.getsid(pid)
        table = _read_proc_table()
        leader = table.get(pid)
        if leader is None or leader["pgrp"] != pgid or leader["session"] != session_id:
            reason = "SENSOR_UNAVAILABLE"
            raise SensorError("leader process identity is unavailable")
        leader_starttime = leader["starttime"]
        outer_deadline = started + wall_seconds
        hard_deadline = started + command_seconds
        soft_term_deadline = hard_deadline - min(term_grace_seconds, command_seconds / 2)
        kill_deadline = hard_deadline - FINAL_VERIFY_SECONDS
        next_status = started
        terminating = False
        leader_exit_observed_at: float | None = None

        while True:
            now = time.monotonic()
            if now >= hard_deadline:
                reason = "ABSOLUTE_HARD_DEADLINE"
                break
            try:
                members, aggregate_rss, cgroup_peak, populated = _cgroup_sample(control_group)
            except SensorError:
                child_exit_code = process.poll()
                if _unit_collected_success(unit, child_exit_code):
                    members, aggregate_rss, cgroup_peak, populated = [], 0, peak_rss, 0
                    group_empty = True
                    cgroup_empty = True
                    if not terminating:
                        reason = "CHILD_EXIT"
                    break
                raise
            peak_rss = max(peak_rss, cgroup_peak)
            unit_metrics = _unit_metrics(unit) or unit_metrics
            observed_memory_max = unit_metrics.get("memory_max_bytes")
            if observed_memory_max is not None:
                if observed_memory_max != rss_bytes:
                    reason = "CGROUP_LIMIT_MISMATCH"
                    raise HarnessError(reason)
                cgroup_verified = True
            memory_events = unit_metrics.get("memory_events")
            if (
                isinstance(memory_events, Mapping)
                and int(memory_events.get("oom_kill", 0)) > 0
                and not terminating
            ):
                reason = "CGROUP_OOM_KILL"
                terminating = True
            child_exit_code = process.poll()
            if not members:
                group_empty = True
                if child_exit_code is not None:
                    reason = "CHILD_EXIT" if not terminating else reason
                    break
            elif child_exit_code is not None and not terminating:
                leader_exit_observed_at = leader_exit_observed_at or now
                if now - leader_exit_observed_at >= poll_seconds:
                    reason = "LEADER_EXIT_WITH_SURVIVORS"
                    terminating = True
            if requested_signal is not None and not terminating:
                reason = f"EXTERNAL_{signal.Signals(requested_signal).name}"
                terminating = True
            if now >= soft_term_deadline and not terminating:
                reason = (
                    "LEADER_EXIT_WITH_SURVIVORS"
                    if leader_exit_observed_at is not None
                    else "WALL_LIMIT"
                )
                terminating = True
            if terminating and term_at is None:
                term_at = _utc_now()
                term_survivors = list(members)
                action_evidence.append(_unit_signal(unit, "TERM"))
            if terminating and members and now >= kill_deadline and kill_at is None:
                kill_at = _utc_now()
                kill_survivors = list(members)
                action_evidence.append(_unit_signal(unit, "KILL"))
                action_evidence.append(
                    {"timestamp_utc": _utc_now(), "cgroup_kill": _cgroup_kill(control_group)}
                )
            if now >= next_status or terminating:
                append_status(
                    "terminating" if terminating else "running", reason, members, aggregate_rss
                )
                next_status = now + poll_seconds
            wake = min(next_status, soft_term_deadline, kill_deadline, hard_deadline)
            time.sleep(max(0.005, min(poll_seconds, wake - time.monotonic())))

        if control_group is not None and not cgroup_empty:
            try:
                members, aggregate_rss, cgroup_peak, populated = _cgroup_sample(control_group)
            except SensorError:
                child_exit_code = process.poll()
                if not _unit_collected_success(unit, child_exit_code):
                    raise
                members, aggregate_rss, cgroup_peak, populated = [], 0, peak_rss, 0
                group_empty = True
                cgroup_empty = True
            peak_rss = max(peak_rss, cgroup_peak)
            if members and kill_at is None:
                kill_at = _utc_now()
                kill_survivors = list(members)
                action_evidence.append(_unit_signal(unit, "KILL"))
                action_evidence.append(
                    {"timestamp_utc": _utc_now(), "cgroup_kill": _cgroup_kill(control_group)}
                )
            while (members or populated) and time.monotonic() < outer_deadline:
                time.sleep(0.005)
                try:
                    members, aggregate_rss, cgroup_peak, populated = _cgroup_sample(control_group)
                except SensorError:
                    child_exit_code = process.poll()
                    if not _unit_collected_success(unit, child_exit_code):
                        raise
                    members, aggregate_rss, cgroup_peak, populated = [], 0, peak_rss, 0
                    break
            group_empty = not members
            cgroup_empty = not members and populated == 0
        if process.poll() is None:
            try:
                child_exit_code = process.wait(
                    timeout=max(0.0, started + wall_seconds - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                child_exit_code = None
        else:
            child_exit_code = process.returncode
        if not group_empty:
            reason = "ABSOLUTE_HARD_DEADLINE_SURVIVORS"
        append_status("finalizing", reason, [], 0)
    except BaseException as error:
        # A telemetry write failure is itself the terminal harness failure.  Do not let
        # a deadline classification assigned immediately before append_status() hide it;
        # under load those two events can occur in the same polling iteration.
        if status_error is not None:
            reason = f"HARNESS_{type(error).__name__.upper()}"
        elif reason == "RUNNING":
            reason = (
                "SENSOR_UNAVAILABLE"
                if isinstance(error, SensorError)
                else f"HARNESS_{type(error).__name__.upper()}"
            )
        if process is not None:
            term_at = term_at or _utc_now()
            action_evidence.append(_unit_signal(unit, "TERM"))
            time.sleep(min(0.05, max(0.0, started + wall_seconds - time.monotonic())))
            kill_at = kill_at or _utc_now()
            action_evidence.append(_unit_signal(unit, "KILL"))
            action_evidence.append(
                {"timestamp_utc": _utc_now(), "cgroup_kill": _cgroup_kill(control_group)}
            )
        if process is not None:
            child_exit_code = process.poll()
    finally:
        if process is not None:
            clean_success = (
                reason == "CHILD_EXIT"
                and child_exit_code == 0
                and cgroup_empty
                and _unit_inactive(unit)
            )
            if not clean_success:
                action_evidence.append(
                    {"timestamp_utc": _utc_now(), "stop_returncode": _stop_unit(unit)}
                )
            if control_group:
                try:
                    survivors, _, _, populated = _cgroup_sample(control_group)
                except SensorError:
                    survivors, populated = [], 0
                kill_survivors = survivors or kill_survivors
                cgroup_empty = not survivors and populated == 0
            cgroup_empty = cgroup_empty and _unit_inactive(unit)
            group_empty = cgroup_empty
        for signum, handler in installed.items():
            signal.signal(signum, handler)

    unit_metrics = _unit_metrics(unit) or unit_metrics
    if reason == "CHILD_EXIT" and (not cgroup_verified or not cgroup_empty):
        reason = "SUCCESS_CGROUP_CLEANUP_FAILED"
    try:
        append_status(
            "complete" if reason == "CHILD_EXIT" and child_exit_code == 0 else "failed",
            reason,
            [] if cgroup_empty else kill_survivors,
            0,
        )
    except BaseException:
        pass
    status_sha256 = _file_sha256(status_path) if status_path.is_file() else None
    raw_memory_events = unit_metrics.get("memory_events") or {}
    memory_events = {
        name: int(raw_memory_events.get(name, 0))
        for name in ("low", "high", "max", "oom", "oom_kill", "oom_group_kill")
    }
    result = {
        "schema_version": "bounded-process-result-v2",
        "provenance": provenance,
        "expected_provenance": {
            "command_sha256": expected_command_sha256,
            "harness_code_sha256": expected_harness_sha256,
            "code_state_sha256": expected_code_state_sha256,
            "cwd": str(expected_cwd.resolve()) if expected_cwd is not None else None,
            "bound_files": {
                str(path.resolve()): digest
                for path, digest in (expected_bound_files or {}).items()
            },
        },
        "provenance_mismatches": mismatches,
        "limits": {
            "absolute_wall_seconds": wall_seconds,
            "memory_max_bytes": rss_bytes,
            "poll_seconds": poll_seconds,
            "term_grace_seconds": term_grace_seconds,
            "command_runtime_max_seconds": command_seconds,
            "result_headroom_seconds": wall_seconds - command_seconds,
        },
        "cgroup": {
            "unit": unit,
            "properties": properties,
            "capability": capability,
            "memory_peak_bytes": unit_metrics.get("memory_peak_bytes"),
            "observed_memory_max_bytes": unit_metrics.get("memory_max_bytes"),
            "control_group": unit_metrics.get("control_group"),
            "memory_events": memory_events,
            "limit_verified": cgroup_verified if process is not None else False,
            "result": unit_metrics.get("result"),
            "oom_killed": (
                unit_metrics.get("result") == "oom-kill"
                or int((unit_metrics.get("memory_events") or {}).get("oom_kill", 0)) > 0
            ),
            "empty": cgroup_empty,
        },
        "started_at_utc": started_utc,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "pid": pid,
        "pgid": pgid,
        "session_id": session_id,
        "leader_starttime": leader_starttime,
        "peak_observed_rss_bytes": peak_rss,
        "child_exit_code": child_exit_code,
        "reason": reason,
        "state": "complete" if reason == "CHILD_EXIT" and child_exit_code == 0 else "failed",
        "term_action_at_utc": term_at,
        "term_survivors": term_survivors,
        "kill_action_at_utc": kill_at,
        "kill_survivors": kill_survivors,
        "group_empty": group_empty,
        "cgroup_empty": cgroup_empty,
        "action_evidence": action_evidence,
        "signal_received": signal.Signals(requested_signal).name if requested_signal else None,
        "status_file_sha256": status_sha256,
        "status_append_error": status_error,
        "harness_private_data_accessed": False,
    }
    _write_result(result_descriptor, result_path, result)
    if reason == "CHILD_EXIT" and child_exit_code is not None:
        return child_exit_code
    return child_exit_code if child_exit_code not in (None, 0) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-jsonl", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--wall-seconds", type=float, default=MAX_WALL_SECONDS)
    parser.add_argument("--rss-bytes", type=int, default=MAX_RSS_BYTES)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--term-grace-seconds", type=float, default=3.0)
    parser.add_argument("--command-seconds", type=float)
    parser.add_argument("--expected-command-sha256")
    parser.add_argument("--expected-harness-sha256")
    parser.add_argument("--expected-code-state-sha256")
    parser.add_argument("--expected-cwd", type=Path)
    parser.add_argument(
        "--profile",
        choices=("default", MEDIUM_PROFILE, PUBLIC_FIXED_PROFILE),
        default="default",
    )
    parser.add_argument("--expected-profile", type=Path)
    parser.add_argument("--expected-profile-sha256")
    parser.add_argument("--expected-bound-file", action="append", default=[], metavar="PATH=SHA256")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    bound_files: dict[Path, str] = {}
    for value in args.expected_bound_file:
        path, separator, digest = value.rpartition("=")
        if not separator or len(digest) != 64:
            raise SystemExit("--expected-bound-file requires PATH=SHA256")
        bound_files[Path(path)] = digest
    returncode = run_bounded(
        command,
        status_path=args.status_jsonl,
        result_path=args.result_json,
        cwd=args.cwd,
        wall_seconds=args.wall_seconds,
        rss_bytes=args.rss_bytes,
        poll_seconds=args.poll_seconds,
        term_grace_seconds=args.term_grace_seconds,
        command_seconds=args.command_seconds,
        expected_command_sha256=args.expected_command_sha256,
        expected_harness_sha256=args.expected_harness_sha256,
        expected_code_state_sha256=args.expected_code_state_sha256,
        expected_cwd=args.expected_cwd,
        profile=args.profile,
        expected_profile_path=args.expected_profile,
        expected_profile_sha256=args.expected_profile_sha256,
        expected_bound_files=bound_files,
        wrapper_argv=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    return returncode if returncode >= 0 else 128 + abs(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
