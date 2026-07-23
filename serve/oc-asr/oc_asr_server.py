#!/usr/bin/env python3
"""Self-hosted Greek council ASR endpoint (faster-whisper + fine-tuned LoRA).

Serves the fine-tuned `whisper-large-v3` model (LoRA merged, converted to
CTranslate2 int8) as an HTTP endpoint returning the OpenCouncil tasks-server
`Transcript` schema, so it is a drop-in alternative to the ElevenLabs Scribe
transcriber. CPU-only. Binds 127.0.0.1 and is gated by an API key; put a
Cloudflare Tunnel in front for public access.

Env: OC_ASR_MODEL_DIR, OC_ASR_API_KEY (required), OC_ASR_COMPUTE (int8),
     OC_ASR_LANGUAGE (el), OC_ASR_MAX_BYTES.

Run: OC_ASR_API_KEY=... uvicorn oc_asr_server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import ipaddress
import math
import os
import secrets
import socket
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

MODEL_DIR = os.environ.get("OC_ASR_MODEL_DIR", "/home/harold/oc-asr-serve/ct2")
API_KEY = os.environ.get("OC_ASR_API_KEY", "")
COMPUTE = os.environ.get("OC_ASR_COMPUTE", "int8")
LANGUAGE = os.environ.get("OC_ASR_LANGUAGE", "el")
MAX_BYTES = int(os.environ.get("OC_ASR_MAX_BYTES", str(500 * 1024 * 1024)))
# Use all CPU cores by default: CTranslate2's default doesn't saturate them, and
# a slow transcription can blow past Cloudflare's ~100s edge timeout (HTTP 524).
CPU_THREADS = int(os.environ.get("OC_ASR_CPU_THREADS", str(os.cpu_count() or 8)))
BEAM = int(os.environ.get("OC_ASR_BEAM", "5"))
# Hard wall-clock cap per transcription: if decoding ever runs away (e.g. a
# repetition loop), stop consuming segments instead of pegging the CPU forever.
MAX_INFER_SEC = float(os.environ.get("OC_ASR_MAX_INFER_SEC", "150"))
CHUNK = 1 << 20

_ALLOWED_SCHEMES = {"http", "https"}

app = FastAPI(title="OpenCouncil self-hosted ASR", version="1.0")
_model = None
# One transcription at a time: the model is CPU-bound (extra parallelism wouldn't
# add throughput) and a single shared WhisperModel consumed by concurrent lazy
# generators could interleave. The lock keeps requests correct; they queue.
_infer_lock = threading.Lock()
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(MODEL_DIR, device="cpu", compute_type=COMPUTE,
                                      cpu_threads=CPU_THREADS)
    return _model


def _require_key(x_api_key: str | None) -> None:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="server missing OC_ASR_API_KEY")
    # Constant-time compare so a wrong key can't be recovered by timing.
    if not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _validate_url(url: str) -> None:
    """SSRF guard: http(s) only, and reject non-public resolved addresses."""
    p = urlparse(url)
    if p.scheme not in _ALLOWED_SCHEMES:
        raise HTTPException(status_code=400, detail=f"blocked URL scheme: {p.scheme}")
    host = p.hostname
    if not host:
        raise HTTPException(status_code=400, detail="URL has no host")
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise HTTPException(status_code=400, detail=f"cannot resolve host: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        # `is_global` is the complete "publicly reachable" test: it also rejects
        # CGNAT (100.64.0.0/10) and other special ranges the individual flags miss.
        if not ip.is_global:
            raise HTTPException(status_code=400,
                                detail=f"blocked non-public address: {ip}")


class _ValidatingRedirect(urllib.request.HTTPRedirectHandler):
    """Re-run the SSRF guard on every redirect hop.

    Without this, an allowed public URL could 302 to http://169.254.169.254 or a
    private host and the guard would be bypassed.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Empty ProxyHandler: never route through an env proxy, which could resolve the
# host differently and slip past the SSRF check above.
_opener = urllib.request.build_opener(_ValidatingRedirect, urllib.request.ProxyHandler({}))


def _download(url: str) -> Path:
    _validate_url(url)
    suffix = Path(urlparse(url).path).suffix or ".mp3"
    fd, tmp = tempfile.mkstemp(prefix="oc_asr_", suffix=suffix)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "oc-asr-server"})
        with _opener.open(req, timeout=180) as r, os.fdopen(fd, "wb") as f:
            fd = None  # fd now owned/closed by f
            read = 0
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                read += len(chunk)
                if read > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="audio exceeds size limit")
                f.write(chunk)
    except BaseException:
        if fd is not None:
            os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


def _transcribe_file(path: str, language: str, word_timestamps: bool = True) -> dict:
    model = _get_model()
    t0 = time.time()
    utterances = []
    full = []
    # Hold the lock across the whole lazy generator: segments are produced on
    # demand, so the model is in use until iteration finishes.
    with _infer_lock:
        # condition_on_previous_text=False is the key guard against runaway
        # repetition loops (greedy/low-beam decoding can otherwise loop forever
        # on some audio, pegging the CPU and blocking every other request).
        segments, info = model.transcribe(path, language=language,
                                          word_timestamps=word_timestamps, beam_size=BEAM,
                                          condition_on_previous_text=False)
        for seg in segments:
            if time.time() - t0 > MAX_INFER_SEC:
                # Runaway guard: return what we have rather than hang the server.
                full.append("[truncated: transcription exceeded time limit]")
                break
            words = []
            for w in (seg.words or []):
                words.append({
                    "word": w.word,
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    # faster-whisper gives a real per-word probability (0..1).
                    "confidence": round(float(w.probability), 4),
                })
            seg_conf = round(math.exp(seg.avg_logprob), 4) if seg.avg_logprob is not None else 0.0
            text = seg.text.strip()
            full.append(text)
            utterances.append({
                "text": text,
                "language": info.language or language,
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "confidence": seg_conf,
                "channel": 0,   # single channel
                "speaker": 0,   # diarization is downstream (pyannote)
                "drift": 0,
                "words": words,
            })
    elapsed = time.time() - t0
    return {
        "metadata": {
            "audio_duration": round(float(info.duration), 3),
            "number_of_distinct_channels": 1,
            "billing_time": 0,
            "transcription_time": round(elapsed, 3),
            "notes": ("self-hosted whisper-large-v3+LoRA (CTranslate2 int8); "
                      "speaker/channel/drift=0 (diarization downstream/pyannote)"),
        },
        "transcription": {
            "languages": [info.language or language],
            "full_transcript": " ".join(full).strip(),
            "utterances": utterances,
        },
    }


class TranscribeBody(BaseModel):
    audioUrl: str | None = None
    language: str | None = None


_PROTECTED = ("/transcribe", "/v1/audio/transcriptions")


def _extract_key(request) -> str:
    """Accept the key as X-API-Key or as an Authorization: Bearer token (the
    latter is what OpenAI-compatible clients like the benchmark send)."""
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:]
    return request.headers.get("x-api-key", "") or ""


@app.middleware("http")
async def _auth_and_size_gate(request, call_next):
    """Check the API key and declared size BEFORE the body is read, so an
    unauthenticated or oversized request can't spool a large upload to disk first.
    """
    if any(request.url.path.startswith(p) for p in _PROTECTED):
        if not API_KEY:
            return JSONResponse({"detail": "server missing OC_ASR_API_KEY"}, status_code=503)
        if not secrets.compare_digest(_extract_key(request), API_KEY):
            return JSONResponse({"detail": "invalid or missing API key"}, status_code=401)
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BYTES:
            return JSONResponse({"detail": "request too large"}, status_code=413)
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "healthy", "model_dir": MODEL_DIR, "compute": COMPUTE,
            "model_loaded": _model is not None}


async def _run(src: str, language: str | None, word_timestamps: bool = True) -> dict:
    # Whisper transcription is CPU-bound and synchronous; run it off the event
    # loop so /health and other requests stay responsive.
    return await run_in_threadpool(_transcribe_file, src, language or LANGUAGE, word_timestamps)


@app.post("/transcribe")
async def transcribe(body: TranscribeBody,
                     x_api_key: str | None = Header(default=None)):
    """Primary path: JSON {audioUrl, language}. The audio should already be a
    segment (the OpenCouncil pipeline sends pre-cut segment URLs), not a whole
    meeting file — this transcribes the entire audio it is given."""
    _require_key(x_api_key)
    if not body.audioUrl:
        raise HTTPException(status_code=400, detail="audioUrl is required")
    # Download off the event loop: DNS/connect/reads must not block other requests.
    tmp = await run_in_threadpool(_download, body.audioUrl)
    try:
        return await _run(str(tmp), body.language)
    finally:
        tmp.unlink(missing_ok=True)


async def _spool_upload(file: UploadFile) -> Path:
    fd, tmp_path = tempfile.mkstemp(prefix="oc_asr_up_",
                                    suffix=Path(file.filename or "").suffix or ".mp3")
    tmp = Path(tmp_path)
    read = 0
    with os.fdopen(fd, "wb") as f:
        while True:
            chunk = await file.read(CHUNK)
            if not chunk:
                break
            read += len(chunk)
            if read > MAX_BYTES:
                tmp.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="upload exceeds size limit")
            f.write(chunk)
    return tmp


@app.post("/transcribe/upload")
async def transcribe_upload(file: UploadFile = File(...),
                            language: str | None = Form(default=None),
                            x_api_key: str | None = Header(default=None)):
    """Alternate path: multipart file upload (a single audio clip)."""
    tmp = await _spool_upload(file)
    try:
        return await _run(str(tmp), language)
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(file: UploadFile = File(...),
                                model: str | None = Form(default=None),
                                language: str | None = Form(default=None),
                                response_format: str | None = Form(default=None)):
    """OpenAI-compatible STT endpoint: multipart {file, model, language} ->
    {"text": ...}. Lets the OpenCouncil benchmark register this as an
    `openai-compatible` provider (POST {baseURL}/audio/transcriptions)."""
    tmp = await _spool_upload(file)
    try:
        # This route only returns text, so skip word timestamps: the alignment
        # pass roughly doubles CPU time and would push long clips past
        # Cloudflare's ~100s edge timeout.
        result = await _run(str(tmp), language, word_timestamps=False)
        return {"text": result["transcription"]["full_transcript"]}
    finally:
        tmp.unlink(missing_ok=True)
