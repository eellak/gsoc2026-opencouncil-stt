"""Deterministic tests for Claude CLI diagnostics and shared retry cooldown."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from eval import fix_call
from eval.fix_call import FixCallError, RetryPolicy, SerializedRetryGate


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_retry_uses_backoff_then_succeeds():
    clock = FakeClock()
    gate = SerializedRetryGate(
        RetryPolicy(backoff_seconds=(5, 15), max_attempts=3),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        jitter=lambda seconds: seconds,
    )
    outcomes = iter([FixCallError("throttled"), FixCallError("throttled"), "ok"])

    def flaky_call() -> str:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert gate.call(flaky_call) == "ok"
    assert clock.sleeps == [5, 15]


def test_terminal_failure_cools_down_next_caller():
    clock = FakeClock()
    gate = SerializedRetryGate(
        RetryPolicy(backoff_seconds=(7,), max_attempts=2),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        jitter=lambda seconds: seconds,
    )

    with pytest.raises(FixCallError, match="2 serialized attempts"):
        gate.call(lambda: (_ for _ in ()).throw(FixCallError("exit 1")))

    assert gate.call(lambda: "recovered") == "recovered"
    assert clock.sleeps == [7, 7]


def test_gate_serializes_concurrent_callers():
    gate = SerializedRetryGate(
        RetryPolicy(max_attempts=1),
        jitter=lambda seconds: seconds,
    )
    first_started = threading.Event()
    release_first = threading.Event()
    second_waiting = threading.Event()
    second_entered = threading.Event()

    def first_call() -> str:
        first_started.set()
        assert release_first.wait(timeout=1)
        return "first"

    def second_task() -> str:
        second_waiting.set()

        def second_call() -> str:
            second_entered.set()
            return "second"

        return gate.call(second_call)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(gate.call, first_call)
        assert first_started.wait(timeout=1)
        second = executor.submit(second_task)
        assert second_waiting.wait(timeout=1)
        assert not second_entered.is_set()
        release_first.set()
        assert first.result(timeout=1) == "first"
        assert second.result(timeout=1) == "second"


def test_empty_stderr_exit_one_has_actionable_diagnostic(monkeypatch):
    monkeypatch.setattr(
        fix_call.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr="",
            stdout="",
        ),
    )

    with pytest.raises(FixCallError, match="OAuth throttling"):
        fix_call.fix_call("system", "user")
