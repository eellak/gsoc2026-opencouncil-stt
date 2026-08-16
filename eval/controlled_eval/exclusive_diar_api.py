#!/usr/bin/env python3
"""Shared pyannoteAI client for the exclusive-diarization experiment.

Same call pattern as `precision2_compare.py`: curl via subprocess (the API sits
behind Cloudflare and urllib gets bounced), presigned PUT upload to
`media://oc-overlap/...`, POST /diarize, poll /jobs/{id}.

The one new thing here is `exclusive`: when true the job output carries an extra
`exclusiveDiarization` key alongside the regular `diarization` one.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
API = "https://api.pyannote.ai/v1"
MODEL = "precision-2"


def api_key() -> str:
    k = os.environ.get("PYANNOTE_API_KEY")
    if k:
        return k
    return (SC / "pyannote_api_key").read_text().strip()


def log(*a):
    print(*a, flush=True)


def curl(args: list[str]) -> dict:
    r = subprocess.run(["curl", "-sS", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed: {r.stderr[:300]}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"non-JSON response: {r.stdout[:300]}")


def upload(path: Path, key: str, name: str) -> str:
    media = f"media://oc-overlap/{name}"
    r = curl(["-X", "POST", f"{API}/media/input",
              "-H", f"Authorization: Bearer {key}",
              "-H", "Content-Type: application/json",
              "-d", json.dumps({"url": media})])
    put = r.get("url")
    if not put:
        raise RuntimeError(f"no presigned url: {r}")
    p = subprocess.run(["curl", "-sS", "-X", "PUT", put, "--data-binary", f"@{path}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"upload failed: {p.stderr[:300]}")
    return media


def diarize(media: str, key: str, exclusive: bool = False, model: str = MODEL,
            transcription: bool = False, stt: str | None = None) -> str:
    body = {"url": media, "model": model}
    if exclusive:
        body["exclusive"] = True
    if transcription:
        # Adds wordLevelTranscription + turnLevelTranscription to the job output,
        # from the default STT (Nvidia Parakeet-TDT-0.6b-v3). Billed separately
        # from diarization; see wayfinder issue #17 for the cost check.
        body["transcription"] = True
        if stt:
            # `transcriptionConfig` is only read when `transcription` is true.
            body["transcriptionConfig"] = {"model": stt}
    elif stt:
        raise ValueError("transcriptionConfig without transcription=True is ignored")
    r = curl(["-X", "POST", f"{API}/diarize",
              "-H", f"Authorization: Bearer {key}",
              "-H", "Content-Type: application/json",
              "-d", json.dumps(body)])
    if "jobId" not in r:
        raise RuntimeError(f"no jobId: {r}")
    return r["jobId"]


def wait_job(job_id: str, key: str, timeout: float = 300) -> dict:
    """Poll GET /v1/jobs/{id}.

    A successful ~2-min precision-2 job settles in 10-20 s. One job during the
    Phase 0 probe sat in `running` for over half an hour with `updatedAt` frozen
    two seconds after creation, so a bounded timeout is load-bearing, not
    decoration: the caller retries rather than blocking the run.
    """
    t0 = time.time()
    delay = 2.0
    while time.time() - t0 < timeout:
        r = curl([f"{API}/jobs/{job_id}", "-H", f"Authorization: Bearer {key}"])
        st = r.get("status")
        if st in ("succeeded", "failed", "canceled"):
            return r
        if st is None:
            return {"status": "bad_response", "jobId": job_id, "raw": r}
        time.sleep(delay)
        delay = min(delay * 1.4, 15)
    return {"status": "timeout", "jobId": job_id}


def run_one(path: Path, key: str, name: str, exclusive: bool,
            attempts: int = 2, transcription: bool = False,
            timeout: float = 300, stt: str | None = None) -> dict:
    """Upload + diarize + wait, retrying once on a stuck/timed-out job."""
    for i in range(attempts):
        media = upload(path, key, f"{name}__try{i}.wav" if i else f"{name}.wav")
        job = diarize(media, key, exclusive=exclusive, transcription=transcription,
                      stt=stt)
        r = wait_job(job, key, timeout=timeout)
        if r.get("status") == "succeeded":
            return r
        log(f"    {name} excl={exclusive} attempt {i + 1}: {r.get('status')}")
    return r
