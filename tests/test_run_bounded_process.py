from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_bounded_process.py"
SPEC = importlib.util.spec_from_file_location("run_bounded_process", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _result(tmp_path: Path, command: list[str], **kwargs: object) -> tuple[int, dict]:
    status = tmp_path / "status.jsonl"
    result = tmp_path / "result.json"
    kwargs.setdefault("poll_seconds", 0.05)
    code = harness.run_bounded(
        command,
        status_path=status,
        result_path=result,
        **kwargs,
    )
    return code, json.loads(result.read_text())


def test_default_profile_limits_are_not_relaxed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="wall_seconds"):
        harness.run_bounded(
            [sys.executable, "-c", "pass"],
            status_path=tmp_path / "status.jsonl",
            result_path=tmp_path / "result.json",
            wall_seconds=61,
        )
    assert harness.MAX_WALL_SECONDS == 60.0
    assert harness.MAX_RSS_BYTES == 1_610_612_736


@pytest.mark.parametrize(
    ("wall", "rss", "command"),
    [
        (2399.0, 4 * 1024**3, 2280.0),
        (2400.0, 4 * 1024**3 - 1, 2280.0),
        (2400.0, 4 * 1024**3, 2279.0),
    ],
)
def test_medium_profile_rejects_any_nonfrozen_limit(
    tmp_path: Path, wall: float, rss: int, command: float
) -> None:
    with pytest.raises(ValueError, match="exact reviewed limits"):
        harness.run_bounded(
            [sys.executable, "-c", "pass"],
            status_path=tmp_path / "status.jsonl",
            result_path=tmp_path / "result.json",
            profile="medium-v1",
            wall_seconds=wall,
            rss_bytes=rss,
            command_seconds=command,
        )


def test_public_fixed_profile_binds_identity_and_child_environment(tmp_path: Path) -> None:
    false_boundaries = {
        "outer_validation_targets_read": False,
        "test_targets_read": False,
        "private_evaluation_labels_read": False,
        "protected_labels_read": False,
        "ranking_allowed": False,
        "ensemble_allowed": False,
        "promotion_allowed": False,
        "w1_allowed": False,
        "general_training_allowed": False,
        "scientific_result": False,
        "publication_ready": False,
    }
    unsigned = {
        "schema_version": harness.PUBLIC_FIXED_AUTH_SCHEMA,
        "run_id": harness.PUBLIC_FIXED_RUN_ID,
        "purpose": harness.PUBLIC_FIXED_PURPOSE,
        "one_use": True,
        **false_boundaries,
    }
    profile = tmp_path / "authorization.json"
    profile.write_text(
        json.dumps(
            {**unsigned, "semantic_sha256": harness._canonical_sha256(unsigned)},
            sort_keys=True,
        )
    )
    command = [
        sys.executable,
        "-c",
        "import os; assert os.environ['QM9_V31_PUBLIC_FIXED_BACKENDS']=='1'; "
        "assert os.environ['QM9_MEDIUM_OUTER_HARNESS_BINDING']",
    ]
    code, payload = _result(
        tmp_path,
        command,
        profile=harness.PUBLIC_FIXED_PROFILE,
        wall_seconds=harness.MEDIUM_WALL_SECONDS,
        rss_bytes=harness.MEDIUM_RSS_BYTES,
        command_seconds=harness.MEDIUM_COMMAND_SECONDS,
        term_grace_seconds=120.0,
        expected_profile_path=profile,
        expected_profile_sha256=harness._file_sha256(profile),
    )
    assert code == 0
    assert payload["state"] == "complete"
    assert payload["cgroup"]["limit_verified"] is True


def test_spawn_passes_binding_to_child_without_polluting_parent(tmp_path: Path) -> None:
    marker = "QM9_MEDIUM_OUTER_HARNESS_BINDING"
    os.environ.pop(marker, None)
    output = tmp_path / "child-binding.txt"
    environment = {**os.environ, marker: "a" * 64}
    process = harness._spawn(
        [
            sys.executable,
            "-c",
            f"import os; open({str(output)!r}, 'w').write(os.environ[{marker!r}])",
        ],
        cwd=tmp_path,
        environment=environment,
    )
    assert process.wait(timeout=5) == 0
    assert output.read_text() == "a" * 64
    assert marker not in os.environ


def test_success_binds_pre_spawn_provenance_cgroup_and_status_hash(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import time; time.sleep(.2); raise SystemExit(7)"]
    command_hash = harness._canonical_sha256(command)
    harness_hash = harness._file_sha256(SCRIPT)
    code_state = harness._repo_code_state(SCRIPT.parent)["code_state_sha256"]
    bound_paths = [
        Path("tests/test_run_bounded_process.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    bound_files = {path: harness._file_sha256(path) for path in bound_paths}
    code, payload = _result(
        tmp_path,
        command,
        expected_command_sha256=command_hash,
        expected_harness_sha256=harness_hash,
        expected_code_state_sha256=code_state,
        expected_cwd=Path.cwd(),
        expected_bound_files=bound_files,
    )
    assert code == payload["child_exit_code"] == 7
    assert payload["reason"] == "CHILD_EXIT"
    assert payload["pid"] == payload["pgid"] == payload["session_id"]
    assert payload["provenance"]["command_sha256"] == command_hash
    assert payload["provenance"]["harness_code_sha256"] == harness_hash
    assert not any(payload["provenance_mismatches"].values())
    assert payload["cgroup"]["properties"]["MemoryMax"] == harness.MAX_RSS_BYTES
    assert payload["cgroup"]["properties"]["RuntimeMaxSec"] <= 55
    assert payload["cgroup"]["limit_verified"] is True
    assert payload["cgroup_empty"] is True
    assert set(payload["provenance"]["bound_files"]) == {
        str(path.resolve()) for path in bound_paths
    }
    assert payload["group_empty"] is True
    assert payload["status_file_sha256"] == harness._file_sha256(tmp_path / "status.jsonl")
    assert (tmp_path / "result.json").stat().st_mode & 0o777 == 0o444


@pytest.mark.parametrize("sleep_seconds", [0, 0.15])
def test_collected_success_has_one_monotonic_terminal_and_no_cleanup_actions(
    tmp_path: Path, sleep_seconds: float
) -> None:
    code, payload = _result(
        tmp_path,
        [sys.executable, "-c", f"import time; time.sleep({sleep_seconds})"],
    )
    states = [row["state"] for row in _rows(tmp_path / "status.jsonl")]
    terminals = [state for state in states if state in {"complete", "failed"}]
    assert code == 0
    assert payload["reason"] == "CHILD_EXIT"
    assert terminals == ["complete"]
    assert "failed" not in states
    assert states[-1] == "complete"
    assert payload["term_action_at_utc"] is None
    assert payload["kill_action_at_utc"] is None
    assert payload["action_evidence"] == []


def test_cgroup_sensor_disappearing_while_unit_active_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_sample = harness._cgroup_sample
    calls = 0

    def disappear(control_group: str) -> tuple[list[int], int, int, int]:
        nonlocal calls
        calls += 1
        if calls > 0:
            raise harness.SensorError("injected active sensor loss")
        return original_sample(control_group)

    monkeypatch.setattr(harness, "_cgroup_sample", disappear)
    code, payload = _result(
        tmp_path,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        wall_seconds=0.6,
        term_grace_seconds=0.2,
    )
    states = [row["state"] for row in _rows(tmp_path / "status.jsonl")]
    assert code != 0
    assert payload["reason"] == "SENSOR_UNAVAILABLE"
    assert states[-1] == "failed"
    assert sum(state in {"complete", "failed"} for state in states) == 1
    assert payload["term_action_at_utc"] is not None


def test_cgroup_cleans_nested_setpgrp_and_setsid_escape_attempts(tmp_path: Path) -> None:
    pid_file = tmp_path / "children.txt"
    source = (
        "import os,subprocess,sys,time; "
        "a=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
        "preexec_fn=os.setpgrp); "
        "b=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
        "start_new_session=True); "
        f"open({str(pid_file)!r},'w').write(f'{{a.pid}} {{b.pid}}'); "
        "time.sleep(30)"
    )
    code, payload = _result(
        tmp_path,
        [sys.executable, "-c", source],
        wall_seconds=0.8,
        term_grace_seconds=0.2,
    )
    assert code != 0
    assert payload["cgroup_empty"] is True
    assert payload["group_empty"] is True
    assert payload["term_survivors"]
    assert any("signal" in row or "cgroup_kill" in row for row in payload["action_evidence"])
    final_status = _rows(tmp_path / "status.jsonl")[-1]
    assert final_status["group_members"] == []


def test_absolute_hard_deadline_includes_term_grace_and_kills_term_ignorer(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ]
    code, payload = _result(
        tmp_path,
        command,
        wall_seconds=1.2,
        term_grace_seconds=0.4,
    )
    assert code != 0
    assert payload["reason"] in {"WALL_LIMIT", "ABSOLUTE_HARD_DEADLINE"}
    assert payload["term_action_at_utc"] is not None
    assert payload["kill_action_at_utc"] is not None
    assert payload["term_survivors"]
    assert payload["kill_survivors"]
    assert payload["group_empty"] is True
    assert payload["elapsed_seconds"] < 1.6


def test_orphaned_command_descendant_is_not_lost_when_command_parent_exits_zero(
    tmp_path: Path,
) -> None:
    child = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"
    )
    code, payload = _result(
        tmp_path,
        [sys.executable, "-c", child],
        # The absolute wall clock includes capability probing and cgroup readiness.
        # Leave enough headroom for the command leader to actually run and exit when
        # this test shares the systemd user manager with the full test suite.
        wall_seconds=1.2,
        term_grace_seconds=0.2,
    )
    assert code != 0
    assert payload["reason"] == "LEADER_EXIT_WITH_SURVIVORS"
    assert payload["group_empty"] is True
    assert payload["cgroup_empty"] is True
    assert payload["term_survivors"]


def test_proc_group_sampling_is_independent_of_leader_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        harness,
        "_read_proc_table",
        lambda: {
            202: {"ppid": 1, "pgrp": 101, "session": 101, "starttime": 20, "rss": 7},
            303: {"ppid": 202, "pgrp": 101, "session": 101, "starttime": 30, "rss": 11},
        },
    )
    assert harness._sample_group(101, 101) == ([202, 303], 18)


def test_sensor_error_fails_closed_and_cleans_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness, "_read_proc_table", lambda: (_ for _ in ()).throw(harness.SensorError("sensor"))
    )
    code, payload = _result(
        tmp_path,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        wall_seconds=0.5,
        term_grace_seconds=0.2,
    )
    assert code != 0
    assert payload["reason"] == "SENSOR_UNAVAILABLE"
    assert payload["state"] == "failed"
    assert payload["term_action_at_utc"] is not None
    assert payload["kill_action_at_utc"] is not None


def test_getpgid_failure_still_cleans_authoritative_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(ProcessLookupError("injected getpgid failure")),
    )
    code, payload = _result(
        tmp_path,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        wall_seconds=0.6,
        term_grace_seconds=0.2,
    )
    assert code != 0
    assert payload["reason"] == "HARNESS_PROCESSLOOKUPERROR"
    assert payload["cgroup_empty"] is True
    assert payload["term_action_at_utc"] is not None
    assert payload["kill_action_at_utc"] is not None


def test_status_append_monitor_error_still_writes_failure_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = harness._append_jsonl
    calls = 0

    def fail_second(path: Path, payload: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected append failure")
        original(path, payload)

    monkeypatch.setattr(harness, "_append_jsonl", fail_second)
    code, payload = _result(
        tmp_path,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        wall_seconds=0.5,
        term_grace_seconds=0.2,
    )
    assert code != 0
    assert payload["reason"] == "HARNESS_HARNESSERROR"
    assert payload["status_append_error"] == "OSError"


def test_popen_error_produces_outer_fail_closed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness,
        "_spawn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected popen")),
    )
    code, payload = _result(tmp_path, [sys.executable, "-c", "pass"])
    assert code == 1
    assert payload["reason"] == "HARNESS_OSERROR"
    assert payload["pid"] is None
    assert payload["group_empty"] is False


def test_provenance_mismatch_prevents_spawn(tmp_path: Path) -> None:
    marker = tmp_path / "spawned"
    code, payload = _result(
        tmp_path,
        [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
        expected_command_sha256="0" * 64,
    )
    assert code == 1
    assert payload["reason"] == "PROVENANCE_MISMATCH"
    assert payload["pid"] is None
    assert not marker.exists()


@pytest.mark.parametrize("failure", ["missing_bound_file", "repo_code_state"])
def test_post_reservation_provenance_failure_is_sealed_without_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    spawned = False
    reserved_fd: int | None = None
    original_reserve = harness._reserve_result

    def reserve(path: Path) -> int:
        nonlocal reserved_fd
        reserved_fd = original_reserve(path)
        return reserved_fd

    def spawn(*_args: object, **_kwargs: object) -> None:
        nonlocal spawned
        spawned = True
        raise AssertionError("command must not spawn")

    monkeypatch.setattr(harness, "_reserve_result", reserve)
    monkeypatch.setattr(harness, "_spawn", spawn)
    kwargs: dict[str, object] = {}
    expected_reason = "BOUND_FILE_UNAVAILABLE"
    if failure == "missing_bound_file":
        kwargs["expected_bound_files"] = {tmp_path / "missing.lock": "0" * 64}
    else:
        expected_reason = "CODE_STATE_OR_PROVENANCE_UNAVAILABLE"
        monkeypatch.setattr(
            harness,
            "_repo_code_state",
            lambda _path: (_ for _ in ()).throw(harness.HarnessError("injected code state")),
        )

    code, payload = _result(tmp_path, [sys.executable, "-c", "pass"], **kwargs)
    result_path = tmp_path / "result.json"
    assert code == 1
    assert spawned is False
    assert result_path.stat().st_size > 0
    assert result_path.stat().st_mode & 0o777 == 0o444
    assert payload["schema_version"] == "bounded-process-result-v2"
    assert payload["state"] == "failed"
    assert payload["reason"] == expected_reason
    assert payload["spawned"] is False
    assert reserved_fd is not None
    with pytest.raises(OSError):
        harness.os.fstat(reserved_fd)


def test_result_collision_is_refused_before_spawn(tmp_path: Path) -> None:
    marker = tmp_path / "spawned"
    result = tmp_path / "result.json"
    result.write_text("existing\n")
    with pytest.raises(FileExistsError):
        harness.run_bounded(
            [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
            status_path=tmp_path / "status.jsonl",
            result_path=result,
        )
    assert result.read_text() == "existing\n"
    assert not marker.exists()


def test_status_cadence_never_exceeds_one_second(tmp_path: Path) -> None:
    _result(
        tmp_path,
        [sys.executable, "-c", "import time; time.sleep(.25)"],
        poll_seconds=0.08,
    )
    elapsed = [float(row["elapsed_seconds"]) for row in _rows(tmp_path / "status.jsonl")]
    assert elapsed == sorted(elapsed)
    assert (
        max((right - left for left, right in zip(elapsed, elapsed[1:], strict=False)), default=0)
        <= 1
    )


def test_capability_preflight_uses_injectable_runner() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        stdout = "systemd 255\n" if command[:3] == ["systemctl", "--user", "--version"] else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    capability = harness._capability_preflight(runner=runner)
    assert capability["memory_controller"] is True
    assert any(
        command[0] == "systemd-run" and "--property=MemoryMax=16777216" in command
        for command in commands
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("wall_seconds", 60.1), ("rss_bytes", 1_610_612_737), ("poll_seconds", 1.01)],
)
def test_limits_can_only_be_tightened(tmp_path: Path, field: str, value: float) -> None:
    kwargs = {
        "status_path": tmp_path / "status.jsonl",
        "result_path": tmp_path / "result.json",
        field: value,
    }
    with pytest.raises(ValueError):
        harness.run_bounded([sys.executable, "-c", "pass"], **kwargs)
