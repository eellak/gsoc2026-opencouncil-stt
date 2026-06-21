"""Thin wrapper around the on-box `claude` CLI (print mode).

No ANTHROPIC_API_KEY here — `claude -p` uses the existing Claude Code OAuth.
We fully override the system prompt (clean output, no default CC preamble),
disable tools, and read raw text. NOT using --bare (it forces API-key auth).
"""
from __future__ import annotations

import random
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

CLAUDE_BIN = "claude"
DEFAULT_MODEL = "sonnet"

# matches a numbered line "<n>. <text>"
_NUM_RE = re.compile(r"^\s*(\d+)\.\s?(.*)$")


class FixCallError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    backoff_seconds: tuple[float, ...] = (5, 15, 30, 60, 90)
    max_attempts: int = 6


class SerializedRetryGate:
    """Serialize Claude CLI calls and share cooldown across worker threads."""

    def __init__(
        self,
        policy: RetryPolicy = RetryPolicy(),
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        self._policy = policy
        self._sleep = sleep
        self._monotonic = monotonic
        self._jitter = jitter or (
            lambda seconds: random.uniform(seconds * 0.8, seconds * 1.2)
        )
        self._lock = threading.Lock()
        self._not_before = 0.0

    def _delay_for_failure(self, failure_index: int) -> float:
        backoffs = self._policy.backoff_seconds
        if not backoffs:
            return 0.0
        base = backoffs[min(failure_index, len(backoffs) - 1)]
        return max(0.0, self._jitter(base))

    def _wait_for_cooldown(self) -> None:
        remaining = self._not_before - self._monotonic()
        if remaining > 0:
            self._sleep(remaining)

    def call(self, fn: Callable[[], str]) -> str:
        """Call ``fn`` with serialized retries and a shared terminal cooldown."""
        attempts = max(1, self._policy.max_attempts)
        last_error: FixCallError | None = None

        # Keep the lock while backing off: queued workers must wait instead of
        # starting their own retry loops and amplifying OAuth throttling.
        with self._lock:
            self._wait_for_cooldown()
            for attempt in range(attempts):
                try:
                    result = fn()
                except FixCallError as exc:
                    last_error = exc
                    delay = self._delay_for_failure(attempt)
                    self._not_before = self._monotonic() + delay
                    if attempt < attempts - 1:
                        self._wait_for_cooldown()
                        continue
                    raise FixCallError(
                        f"{exc} (failed after {attempts} serialized attempts; "
                        f"next call cooled down for {delay:.1f}s)"
                    ) from exc
                else:
                    self._not_before = 0.0
                    return result

        assert last_error is not None  # pragma: no cover - loop always returns/raises
        raise last_error


_CLAUDE_GATE = SerializedRetryGate()


def fix_call(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL,
             timeout: int = 180) -> str:
    """Run one fix-call; return raw stdout text."""
    cmd = [
        CLAUDE_BIN, "-p",
        "--model", model,
        "--system-prompt", system_prompt,
        "--allowedTools", "",
        "--output-format", "text",
    ]
    try:
        proc = subprocess.run(
            cmd, input=user_prompt, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise FixCallError(f"timeout after {timeout}s") from e
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr or stdout
        if not detail:
            detail = (
                "no stderr/stdout (possible Claude Code OAuth throttling or "
                "authentication failure)"
            )
        elif not stderr:
            detail = f"empty stderr; stdout: {detail}"
        raise FixCallError(f"claude -p exit {proc.returncode}: {detail[:300]}")
    return proc.stdout.strip()


def fix_call_with_retry(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 180,
) -> str:
    """Run one process-wide serialized Claude call with retry/cooldown."""
    return _CLAUDE_GATE.call(
        lambda: fix_call(system_prompt, user_prompt, model=model, timeout=timeout)
    )


def parse_numbered(text: str, n_expected: int) -> list[str] | None:
    """Parse model output into n_expected utterance strings.

    Returns the list on a clean parse (exactly n_expected numbered lines,
    contiguous 1..n), else None.
    """
    out: list[str] = []
    for line in text.splitlines():
        m = _NUM_RE.match(line)
        if m:
            out.append(m.group(2).rstrip())
    if len(out) != n_expected:
        return None
    return out
