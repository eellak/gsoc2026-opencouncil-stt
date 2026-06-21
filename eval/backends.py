"""Pluggable inference backends for fix-calls.

Lets the eval run on cheaper models for broad sweeps, keeping sonnet for final
validation. Unified entry point:

    generate(system_prompt, user_prompt, backend="claude", model=None) -> str

Backends:
  - claude : on-box `claude -p` over OAuth (model: sonnet | haiku | ...).
             Uses the shared serialized retry gate (OAuth throttle protection).
  - codex  : OpenAI models via the codex-bridge (separate quota from Claude).
             Serialized through the bridge worker — fine for small sweeps, slow
             for thousands of calls.
  - gemini : Google Gemini (e.g. flash). Requires a `gemini` CLI or
             GEMINI_API_KEY; raises a clear error if neither is present.

All backends return raw text; callers parse numbered lines themselves.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from eval.fix_call import FixCallError, fix_call_with_retry

CODEX_CLIENT = "/home/harold/codex-bridge/codex_client.py"

# default model per backend
DEFAULT_MODELS = {
    "claude": "sonnet",
    "codex": "gpt-5.4-mini",
    "gemini": "gemini-flash-3.5",
}


def generate(system_prompt: str, user_prompt: str, backend: str = "claude",
             model: str | None = None, timeout: int = 180) -> str:
    model = model or DEFAULT_MODELS.get(backend)
    if backend == "claude":
        return fix_call_with_retry(system_prompt, user_prompt, model=model, timeout=timeout)
    if backend == "codex":
        return _codex_generate(system_prompt, user_prompt, model=model, timeout=timeout)
    if backend == "gemini":
        return _gemini_generate(system_prompt, user_prompt, model=model, timeout=timeout)
    raise FixCallError(f"unknown backend: {backend}")


def _codex_generate(system_prompt: str, user_prompt: str, model: str, timeout: int) -> str:
    """Run one fix via the codex-bridge (enqueue + wait). Low reasoning effort."""
    if not os.path.exists(CODEX_CLIENT):
        raise FixCallError("codex bridge client not found")
    prompt = (
        f"{system_prompt}\n\n"
        "Apply the above. Output ONLY the corrected numbered lines, nothing else.\n\n"
        f"{user_prompt}"
    )
    enq = subprocess.run(
        ["python3", CODEX_CLIENT, "enqueue", "exec",
         "-c", "model_reasoning_effort=low", "-c", f"model={model}", prompt],
        capture_output=True, text=True, timeout=60,
    )
    if enq.returncode != 0:
        raise FixCallError(f"codex enqueue failed: {enq.stderr.strip()[:200]}")
    try:
        job_id = json.loads(enq.stdout)["job_id"]
    except Exception as e:
        raise FixCallError(f"codex enqueue bad response: {enq.stdout.strip()[:200]}") from e
    wait = subprocess.run(
        ["python3", CODEX_CLIENT, "wait", job_id, str(timeout)],
        capture_output=True, text=True, timeout=timeout + 30,
    )
    try:
        res = json.loads(wait.stdout)
    except Exception as e:
        raise FixCallError(f"codex wait bad response: {wait.stdout.strip()[:200]}") from e
    status = res.get("status")
    if status != "completed":
        raise FixCallError(f"codex status={status}")
    return (res.get("output") or "").strip()


def _gemini_generate(system_prompt: str, user_prompt: str, model: str, timeout: int) -> str:
    cli = shutil.which("gemini")
    if cli:
        proc = subprocess.run(
            [cli, "-m", model, "-p", f"{system_prompt}\n\n{user_prompt}"],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise FixCallError(f"gemini exit {proc.returncode}: {proc.stderr.strip()[:200]}")
        return proc.stdout.strip()
    raise FixCallError(
        "gemini backend unavailable: no `gemini` CLI on PATH and no API wiring. "
        "Install the Gemini CLI or provide GEMINI_API_KEY + a client to enable."
    )
