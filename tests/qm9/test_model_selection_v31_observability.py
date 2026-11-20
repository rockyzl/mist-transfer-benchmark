from __future__ import annotations

import json
import os
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mist_transfer_benchmark.qm9.model_selection_v31_guard import (
    GovernedStop,
    GuardState,
    GuardThresholds,
    RunGuard,
    SignalRelay,
)
from mist_transfer_benchmark.qm9.model_selection_v31_observability import (
    EventSink,
    HeartbeatEmitter,
    ObservabilityError,
    TrialIdentity,
    timed_stage,
    validate_event_stream,
)


class FakeClock:
    def __init__(self) -> None:
        self.ns = 0

    def monotonic_ns(self) -> int:
        return self.ns

    def advance(self, seconds: float) -> None:
        self.ns += int(seconds * 1_000_000_000)


class FakeResources:
    def __init__(self, **overrides: object) -> None:
        self.values = {
            "rss_bytes": 10,
            "gpu_allocated_bytes": 20,
            "gpu_reserved_bytes": 30,
            "gpu_temperature_c": 40.0,
            "disk_free_bytes": 1_000,
        }
        self.values.update(overrides)

    def sample(self) -> dict[str, object]:
        return dict(self.values)


def trial(run_id: str = "run-1") -> TrialIdentity:
    return TrialIdentity(
        run_id=run_id,
        wave="W1",
        strategy="identity",
        candidate_id="mlp|all_raw|compact",
        family="mlp",
        fold=0,
        seed=20260713,
        trial_index=1,
        trial_total=123,
    )


def make_sink(tmp_path: Path, clock: FakeClock | None = None) -> EventSink:
    active_clock = clock or FakeClock()
    return EventSink(
        tmp_path / "events.jsonl",
        "run-1",
        monotonic_ns=active_clock.monotonic_ns,
        utc_now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
    )


def thresholds(clock: FakeClock, **overrides: object) -> GuardThresholds:
    values: dict[str, object] = {
        "deadline_monotonic_ns": clock.ns + 100_000_000_000,
        "heartbeat_seconds": 10.0,
        "stale_after_seconds": 20.0,
        "sample_seconds": 5.0,
        "rss_max_bytes": 100,
        "gpu_allocated_max_bytes": 100,
        "gpu_reserved_max_bytes": 100,
        "gpu_temperature_max_c": 80.0,
        "disk_free_min_bytes": 100,
    }
    values.update(overrides)
    return GuardThresholds(**values)


def test_event_schema_chain_and_nonfinite_rejection(tmp_path: Path) -> None:
    sink = make_sink(tmp_path)
    first = sink.emit("run_start", trial=None, progress={"current": 0})
    second = sink.emit(
        "epoch_end",
        trial=trial(),
        progress={"current": 1, "total": 2},
        metrics={"train_loss": 0.5, "per_target_nmae": [0.1] * 12},
    )
    with pytest.raises(ObservabilityError, match="nonfinite"):
        sink.emit("heartbeat", trial=trial(), progress={"loss": float("nan")})
    sink.close()

    events = validate_event_stream(tmp_path / "events.jsonl")
    assert [row["sequence"] for row in events] == [1, 2]
    assert second["prev_event_sha256"] == first["event_sha256"]
    assert all(row["scientific_result"] is False for row in events)
    assert all(row["outer_validation_targets_read"] is False for row in events)
    assert all(row["test_targets_read"] is False for row in events)


def test_chain_tamper_and_truncated_final_line_fail_closed(tmp_path: Path) -> None:
    sink = make_sink(tmp_path)
    sink.emit("run_start", trial=None, progress={})
    sink.emit("heartbeat", trial=trial(), progress={})
    sink.close()
    path = tmp_path / "events.jsonl"
    original = path.read_bytes()

    path.write_bytes(original[:-3])
    with pytest.raises(ObservabilityError, match="truncated"):
        validate_event_stream(path)
    assert len(validate_event_stream(path, allow_truncated_final_line=True)) == 1

    lines = original.splitlines()
    payload = json.loads(lines[1])
    payload["progress"]["tampered"] = True
    path.write_bytes(lines[0] + b"\n" + json.dumps(payload).encode() + b"\n")
    with pytest.raises(ObservabilityError, match="hash differs"):
        validate_event_stream(path)


def test_reopen_quarantines_truncated_tail_and_rejects_cross_run(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with EventSink(path, "run-1") as sink:
        sink.emit("run_start", trial=None, progress={"binding": "fixture"})
        sink.emit("heartbeat", trial=trial(), progress={})
    raw = path.read_bytes()
    path.write_bytes(raw[:-5])
    with EventSink(path, "run-1") as recovered:
        recovered.emit("heartbeat", trial=trial(), progress={"recovered": True})
    events = validate_event_stream(path)
    assert [row["sequence"] for row in events] == [1, 2]
    quarantines = list(tmp_path.glob("events.jsonl.truncated-*.quarantine"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes()
    with pytest.raises(ObservabilityError, match="another run|another run-start"):
        EventSink(path, "run-2")


def test_strict_vocabulary_and_terminal_immutability(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = EventSink(path, "run-1")
    with pytest.raises(ObservabilityError, match="reviewed vocabulary"):
        sink.emit("invented_event", trial=None, progress={})
    sink.emit("failure", trial=None, progress={}, failure={"reason": "fixture"})
    with pytest.raises(ObservabilityError, match="terminal"):
        sink.emit("heartbeat", trial=trial(), progress={})
    sink.close()
    with pytest.raises(ObservabilityError, match="terminal"):
        EventSink(path, "run-1")


def test_single_writer_concurrency_preserves_json_and_sequence(tmp_path: Path) -> None:
    sink = make_sink(tmp_path)

    def producer(worker: int) -> None:
        for item in range(25):
            sink.emit("batch_end", trial=trial(), progress={"worker": worker, "item": item})

    threads = [threading.Thread(target=producer, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    sink.close()

    events = validate_event_stream(tmp_path / "events.jsonl")
    assert len(events) == 200
    assert events[-1]["sequence"] == 200


def test_forced_event_calls_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: calls.append(descriptor))
    sink = make_sink(tmp_path)
    before = len(calls)
    sink.emit("failure", trial=None, progress={}, failure={"code": "INJECTED"}, force_fsync=True)
    assert len(calls) == before + 1
    sink.close()
    assert len(calls) == before + 2


def test_timed_stage_records_exception_and_duration(tmp_path: Path) -> None:
    sink = make_sink(tmp_path)
    durations: dict[str, float] = {}
    with pytest.raises(RuntimeError, match="injected"):
        with timed_stage(sink, trial(), "to_dense", durations):
            raise RuntimeError("injected")
    sink.close()
    events = validate_event_stream(tmp_path / "events.jsonl")
    assert [row["event"] for row in events] == ["stage_start", "stage_end"]
    assert events[-1]["progress"]["failed"] is True
    assert durations["to_dense"] >= 0


def test_heartbeat_is_independent_and_interval_is_bounded(tmp_path: Path) -> None:
    with pytest.raises(ObservabilityError, match="interval"):
        HeartbeatEmitter(
            make_sink(tmp_path),
            interval_seconds=31,
            trial_provider=lambda: trial(),
            progress_provider=lambda: {},
            resource_provider=FakeResources(),
        )
    path = tmp_path / "live.jsonl"
    sink = EventSink(path, "run-1")
    emitter = HeartbeatEmitter(
        sink,
        interval_seconds=0.02,
        trial_provider=lambda: trial(),
        progress_provider=lambda: {"worker_alive": True, "progress_healthy": True},
        resource_provider=FakeResources(),
    )
    emitter.start()
    time.sleep(0.07)
    emitter.close()
    sink.close()
    events = validate_event_stream(path)
    assert len(events) >= 3
    assert {row["event"] for row in events} == {"heartbeat"}


@pytest.mark.parametrize(
    ("resources", "code"),
    [
        ({"rss_bytes": 101}, "RSS_LIMIT"),
        ({"gpu_allocated_bytes": 101}, "GPU_ALLOCATED_LIMIT"),
        ({"gpu_reserved_bytes": 101}, "GPU_RESERVED_LIMIT"),
        ({"gpu_temperature_c": 81.0}, "GPU_TEMPERATURE"),
        ({"disk_free_bytes": 99}, "DISK_FLOOR"),
    ],
)
def test_every_resource_gate_fails_closed(
    tmp_path: Path, resources: dict[str, object], code: str
) -> None:
    clock = FakeClock()
    sink = make_sink(tmp_path, clock)
    guard = RunGuard(
        thresholds(clock),
        sink,
        FakeResources(**resources),
        monotonic_ns=clock.monotonic_ns,
        trial_provider=lambda: trial(),
    )
    with pytest.raises(GovernedStop) as caught:
        guard.check("unit-test")
    assert caught.value.code == code
    assert guard.state == GuardState.HARD_STOP
    sink.close()
    events = validate_event_stream(tmp_path / "events.jsonl")
    assert events[-1]["event"] == "hard_stop_requested"
    assert events[-1]["failure"]["code"] == code


def test_deadline_stale_heartbeat_and_threshold_contract(tmp_path: Path) -> None:
    clock = FakeClock()
    sink = make_sink(tmp_path, clock)
    with pytest.raises(ValueError, match="two heartbeat"):
        thresholds(clock, stale_after_seconds=19.0)
    with pytest.raises(ValueError, match=r"\(0, 5\]"):
        thresholds(clock, sample_seconds=5.1)

    stale_guard = RunGuard(
        thresholds(clock), sink, FakeResources(), monotonic_ns=clock.monotonic_ns
    )
    clock.advance(20)
    with pytest.raises(GovernedStop) as stale:
        stale_guard.check("stale-test")
    assert stale.value.code == "STALE_HEARTBEAT"
    sink.close()

    deadline_clock = FakeClock()
    deadline_sink = EventSink(
        tmp_path / "deadline.jsonl", "run-1", monotonic_ns=deadline_clock.monotonic_ns
    )
    deadline_guard = RunGuard(
        thresholds(deadline_clock, deadline_monotonic_ns=10_000_000_000),
        deadline_sink,
        FakeResources(),
        monotonic_ns=deadline_clock.monotonic_ns,
    )
    deadline_clock.advance(10)
    with pytest.raises(GovernedStop) as deadline:
        deadline_guard.check("deadline-test")
    assert deadline.value.code == "WALL_DEADLINE"
    deadline_sink.close()


def test_soft_drain_blocks_new_work_but_not_current_work(tmp_path: Path) -> None:
    clock = FakeClock()
    sink = make_sink(tmp_path, clock)
    guard = RunGuard(
        thresholds(
            clock,
            soft_drain_monotonic_ns=5_000_000_000,
            deadline_monotonic_ns=10_000_000_000,
        ),
        sink,
        FakeResources(),
        monotonic_ns=clock.monotonic_ns,
    )
    clock.advance(5)
    guard.check("current-trial")
    assert guard.state == GuardState.SOFT_DRAIN
    assert guard.can_start_new_work is False
    with pytest.raises(GovernedStop) as stopped:
        guard.check("next-trial", starting_new_work=True)
    assert stopped.value.code == "SOFT_DRAIN"
    sink.close()


def test_signal_relay_sets_flag_and_self_pipe() -> None:
    relay = SignalRelay((signal.SIGUSR1,))
    previous = signal.getsignal(signal.SIGUSR1)
    try:
        relay.install()
        os.kill(os.getpid(), signal.SIGUSR1)
        assert relay.poll() == signal.SIGUSR1
        assert relay.received_signal == signal.SIGUSR1
    finally:
        relay.close()
    assert signal.getsignal(signal.SIGUSR1) == previous
