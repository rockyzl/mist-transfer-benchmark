"""Continuous runtime guards and minimal signal relay for QM9 v3.1."""

from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import FrameType
from typing import Any

from .model_selection_v31_observability import EventSink, ResourceProvider, TrialIdentity


class GuardState(StrEnum):
    RUNNING = "running"
    SOFT_DRAIN = "soft-drain"
    HARD_STOP = "hard-stop"
    CLOSED = "closed"


@dataclass(frozen=True)
class GuardThresholds:
    deadline_monotonic_ns: int
    heartbeat_seconds: float
    stale_after_seconds: float
    sample_seconds: float
    rss_max_bytes: int
    gpu_allocated_max_bytes: int
    gpu_reserved_max_bytes: int
    gpu_temperature_max_c: float
    disk_free_min_bytes: int
    soft_drain_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.heartbeat_seconds <= 30:
            raise ValueError("heartbeat_seconds must be in (0, 30]")
        if abs(self.stale_after_seconds - 2 * self.heartbeat_seconds) > 1e-9:
            raise ValueError("stale_after_seconds must equal two heartbeat intervals")
        if not 0 < self.sample_seconds <= 5:
            raise ValueError("sample_seconds must be in (0, 5]")
        for name in (
            "deadline_monotonic_ns",
            "rss_max_bytes",
            "gpu_allocated_max_bytes",
            "gpu_reserved_max_bytes",
            "disk_free_min_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.gpu_temperature_max_c <= 0:
            raise ValueError("gpu_temperature_max_c must be positive")
        if self.soft_drain_monotonic_ns is not None and not (
            0 < self.soft_drain_monotonic_ns < self.deadline_monotonic_ns
        ):
            raise ValueError("soft drain must be positive and earlier than hard deadline")


class GovernedStop(RuntimeError):
    """A predeclared continuous guard requested a fail-closed stop."""

    def __init__(
        self,
        code: str,
        *,
        observed: object,
        threshold: object,
        safe_point: str,
    ) -> None:
        super().__init__(f"{code} at {safe_point}: observed={observed}, threshold={threshold}")
        self.code = code
        self.observed = observed
        self.threshold = threshold
        self.safe_point = safe_point

    def payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "observed": self.observed,
            "threshold": self.threshold,
            "safe_point": self.safe_point,
        }


class RunGuard:
    """Sample hard limits continuously and surface them at backend safe points."""

    def __init__(
        self,
        thresholds: GuardThresholds,
        sink: EventSink,
        resource_provider: ResourceProvider,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        trial_provider: Callable[[], TrialIdentity | None] = lambda: None,
        heartbeat_ns_provider: Callable[[], int] | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.sink = sink
        self.resource_provider = resource_provider
        self.monotonic_ns = monotonic_ns
        self.trial_provider = trial_provider
        self.heartbeat_ns_provider = heartbeat_ns_provider or (
            lambda: sink.last_heartbeat_monotonic_ns
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = GuardState.RUNNING
        self._pending: GovernedStop | None = None
        self._last_resources: Mapping[str, object] = {}
        self._soft_event_emitted = False

    @property
    def state(self) -> GuardState:
        with self._lock:
            return self._state

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._pending is not None

    @property
    def can_start_new_work(self) -> bool:
        return self.state == GuardState.RUNNING

    def _trip(self, code: str, observed: object, threshold: object, safe_point: str) -> None:
        stop = GovernedStop(
            code, observed=observed, threshold=threshold, safe_point=safe_point
        )
        with self._lock:
            if self._pending is not None:
                return
            self._pending = stop
            self._state = GuardState.HARD_STOP
        self.sink.emit(
            "hard_stop_requested",
            trial=self.trial_provider(),
            progress={"safe_point": safe_point, "guard_state": GuardState.HARD_STOP},
            failure=stop.payload(),
            resources=self._last_resources,
            force_fsync=True,
        )

    def _sample(self, safe_point: str) -> None:
        now = self.monotonic_ns()
        thresholds = self.thresholds
        if (
            thresholds.soft_drain_monotonic_ns is not None
            and now >= thresholds.soft_drain_monotonic_ns
            and now < thresholds.deadline_monotonic_ns
        ):
            with self._lock:
                if self._state == GuardState.RUNNING:
                    self._state = GuardState.SOFT_DRAIN
                emit = not self._soft_event_emitted
                self._soft_event_emitted = True
            if emit:
                self.sink.emit(
                    "soft_drain_requested",
                    trial=self.trial_provider(),
                    progress={"safe_point": safe_point, "guard_state": GuardState.SOFT_DRAIN},
                    force_fsync=True,
                )
        if now >= thresholds.deadline_monotonic_ns:
            self._trip("WALL_DEADLINE", now, thresholds.deadline_monotonic_ns, safe_point)
            return
        heartbeat_age = (now - self.heartbeat_ns_provider()) / 1_000_000_000
        if heartbeat_age >= thresholds.stale_after_seconds:
            self._trip(
                "STALE_HEARTBEAT", heartbeat_age, thresholds.stale_after_seconds, safe_point
            )
            return
        resources = dict(self.resource_provider.sample())
        self._last_resources = resources
        gates = (
            (
                "RSS_LIMIT",
                "rss_bytes",
                thresholds.rss_max_bytes,
                lambda value, limit: value > limit,
            ),
            (
                "GPU_ALLOCATED_LIMIT",
                "gpu_allocated_bytes",
                thresholds.gpu_allocated_max_bytes,
                lambda value, limit: value > limit,
            ),
            (
                "GPU_RESERVED_LIMIT",
                "gpu_reserved_bytes",
                thresholds.gpu_reserved_max_bytes,
                lambda value, limit: value > limit,
            ),
            (
                "GPU_TEMPERATURE",
                "gpu_temperature_c",
                thresholds.gpu_temperature_max_c,
                lambda value, limit: value > limit,
            ),
            (
                "DISK_FLOOR",
                "disk_free_bytes",
                thresholds.disk_free_min_bytes,
                lambda value, limit: value < limit,
            ),
        )
        for code, field, threshold, breached in gates:
            observed = resources.get(field)
            if observed is not None and breached(observed, threshold):
                self._trip(code, observed, threshold, safe_point)
                return

    def sample_once(self, safe_point: str = "guard-sampler") -> None:
        self._sample(safe_point)

    def _run(self) -> None:
        while not self._stop_event.wait(self.thresholds.sample_seconds):
            try:
                self._sample("guard-sampler")
            except BaseException as error:
                self._trip(
                    "GUARD_FAILURE",
                    type(error).__name__,
                    "healthy sampler",
                    "guard-sampler",
                )
                return
            if self.stop_requested:
                return

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("run guard was already started")
        self._sample("guard-start")
        if self.stop_requested:
            return
        self._thread = threading.Thread(target=self._run, name="qm9-v31-run-guard", daemon=True)
        self._thread.start()

    def check(self, safe_point: str, *, starting_new_work: bool = False) -> None:
        self._sample(safe_point)
        with self._lock:
            pending = self._pending
            soft_drain = self._state == GuardState.SOFT_DRAIN
        if pending is not None:
            raise pending
        if starting_new_work and soft_drain:
            raise GovernedStop(
                "SOFT_DRAIN",
                observed=self.monotonic_ns(),
                threshold=self.thresholds.soft_drain_monotonic_ns,
                safe_point=safe_point,
            )

    def request_stop(
        self, reason: str, detail: Mapping[str, object], *, safe_point: str = "external-request"
    ) -> None:
        self._trip(reason, dict(detail), "no external stop", safe_point)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            pending = self._pending.payload() if self._pending is not None else None
            state = self._state
        return {
            "state": state,
            "stop_requested": pending is not None,
            "pending_stop": pending,
            "resources": dict(self._last_resources),
            "deadline_monotonic_ns": self.thresholds.deadline_monotonic_ns,
            "soft_drain_monotonic_ns": self.thresholds.soft_drain_monotonic_ns,
        }

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.thresholds.sample_seconds + 0.5))
        with self._lock:
            if self._state != GuardState.HARD_STOP:
                self._state = GuardState.CLOSED


class SignalRelay:
    """Translate SIGINT/SIGTERM into a flag and self-pipe byte.

    The installed handler performs no JSON, locks, CUDA, or arbitrary file I/O.
    Normal supervisor code calls :meth:`poll` and persists the governed failure.
    """

    def __init__(self, signals: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)) -> None:
        self.signals = signals
        self.read_fd, self.write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self.write_fd, False)
        self.received_signal: int | None = None
        self._previous: dict[int, Any] = {}
        self._installed = False

    def _handler(self, signum: int, _frame: FrameType | None) -> None:
        self.received_signal = signum
        try:
            os.write(self.write_fd, bytes((signum & 0xFF,)))
        except BlockingIOError:
            pass

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("signal relay was already installed")
        for signum in self.signals:
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)
        self._installed = True

    def poll(self) -> int | None:
        try:
            while os.read(self.read_fd, 4096):
                pass
        except BlockingIOError:
            pass
        return self.received_signal

    def close(self) -> None:
        if self._installed:
            for signum, previous in self._previous.items():
                signal.signal(signum, previous)
            self._installed = False
        os.close(self.read_fd)
        os.close(self.write_fd)

    def __enter__(self) -> SignalRelay:
        self.install()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
