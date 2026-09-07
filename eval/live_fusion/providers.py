# -*- coding: utf-8 -*-
"""Live calls to the three ASR systems, one function each.

Every response is cached on disk under LIVE_CACHE keyed by
(window id, provider, params sha), because a five-hour unattended run that dies
at window 20 must not pay ElevenLabs twice for windows 1-19. The cache key
contains the request parameters: change a parameter and the old answer is not
reused, which is the whole reason the sha is in the key rather than beside it.

Never prints transcript text. Diagnostics carry ids, counts and hashes only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

LIVE_CACHE = Path(os.environ.get("LIVE_FUSION_CACHE")
                  or Path.home() / ".cache/oc-public/live-fusion-2026-09")

# Mirrors src/lib/fusion/providers/scribe.ts SCRIBE_REQUEST_PARAMS.
SCRIBE_PARAMS = {
    "model_id": "scribe_v2",
    "no_verbatim": True,
    "tag_audio_events": False,
    "timestamps_granularity": "word",
    "diarize": False,
    "language_code": "ell",
}

# Mirrors src/lib/fusion/providers/soniox.ts SONIOX_REQUEST_PARAMS. stt-async-v5
# is the PAID async model; the free stt-rt-* realtime models are a different
# system and mixing them has already invalidated one measurement here.
SONIOX_MODEL = "stt-async-v5"
SONIOX_PARAMS = {
    "model": SONIOX_MODEL,
    "language_hints": ["el"],
    "enable_speaker_diarization": False,
    "enable_language_identification": False,
}

# The canonical CPU decode of eval/results_train_wer_v2.json, plus
# word_timestamps: fusion needs word spans and confidences, and the frozen
# config was measured with word_timestamps off. That is an addition to the
# frozen config and is recorded as such.
OURS_DECODE = {
    "language": "el",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "word_timestamps": True,
    "vad_filter": False,
    "task": "transcribe",
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
}
OURS_MODEL_DIR = Path(os.environ.get("OURS_MODEL_DIR", "/home/harold/oc-asr-serve/ct2-v2"))
OURS_MODEL_SHA = "0c6976f120f12f7c6dc487f86dcb3649b80c78adb6c08e31142ec55f01e535b4"
OURS_RUNTIME = {"device": "cpu", "compute_type": "int8", "cpu_threads": 4}


RETRYABLE = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3


def with_retry(call, what: str):
    """Retry only transport failures, never a response we dislike.

    The accepted attempt is the first structurally valid one. Retrying because a
    transcript scores badly would be cherry-picking, so nothing here can see a
    score. `Retry-After` is honoured when the server sends it.
    """
    import random as _random
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = call()
            if r.status_code in RETRYABLE and attempt < MAX_ATTEMPTS:
                wait = float(r.headers.get("Retry-After") or 0) or (
                    2 ** attempt + _random.uniform(0, 1))
                print(f"[retry] {what} HTTP {r.status_code}, waiting {wait:.0f}s",
                      file=__import__("sys").stderr)
                time.sleep(wait)
                last = RuntimeError(f"{what} HTTP {r.status_code}")
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
            if attempt == MAX_ATTEMPTS:
                break
            wait = 2 ** attempt + _random.uniform(0, 1)
            print(f"[retry] {what} {type(exc).__name__}, waiting {wait:.0f}s",
                  file=__import__("sys").stderr)
            time.sleep(wait)
    raise last if last else RuntimeError(f"{what} failed")


def params_sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _cache_path(window_id: str, provider: str, sha: str) -> Path:
    d = LIVE_CACHE / provider
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{window_id}.{sha}.json"


def cached(window_id: str, provider: str, sha: str):
    p = _cache_path(window_id, provider, sha)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def store(window_id: str, provider: str, sha: str, value) -> None:
    p = _cache_path(window_id, provider, sha)
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def logprob_to_conf(logprob):
    """Mirror of logprobToConfidence in src/lib/ScribeTranscribe.ts."""
    if logprob is None:
        return None
    import math
    return max(0.0, min(1.0, math.exp(logprob)))


# --------------------------------------------------------------- scribe

def scribe_words(window_id: str, wav: Path, api_key: str, timeout=600):
    sha = params_sha(SCRIBE_PARAMS)
    hit = cached(window_id, "scribe", sha)
    if hit is None:
        data = {
            "model_id": SCRIBE_PARAMS["model_id"],
            "language_code": SCRIBE_PARAMS["language_code"],
            "tag_audio_events": "false",
            "timestamps_granularity": "word",
            "diarize": "false",
        }

        def post():
            # Reopened per attempt: an exhausted handle would upload zero bytes
            # and the retry would transcribe different audio from the first try.
            with open(wav, "rb") as fh:
                return requests.post("https://api.elevenlabs.io/v1/speech-to-text",
                                     headers={"xi-api-key": api_key}, data=data,
                                     files={"file": (wav.name, fh, "audio/wav")},
                                     timeout=timeout)

        hit = with_retry(post, f"scribe {window_id}").json()
        store(window_id, "scribe", sha, hit)
    words = [{"raw": w["text"], "start": w.get("start"), "end": w.get("end"),
              "conf": logprob_to_conf(w.get("logprob"))}
             for w in hit.get("words", []) if w.get("type") == "word"]
    return words, sha


# --------------------------------------------------------------- soniox

def _soniox_join(tokens):
    """Sub-word tokens -> words. Mirror of normalizeSonioxTokens in soniox.ts:
    a word starts at a token whose text begins with whitespace; its confidence
    is the minimum over the run, because the weakest piece decides."""
    words, cur = [], None

    def flush():
        nonlocal cur
        if cur is not None:
            raw = cur["text"].strip()
            if raw:
                words.append({"raw": raw, "start": cur["start"],
                              "end": cur["end"], "conf": cur["conf"]})
            cur = None

    for t in tokens or []:
        text = t.get("text") or ""
        if not text:
            continue
        start = None if t.get("start_ms") is None else t["start_ms"] / 1000.0
        end = None if t.get("end_ms") is None else t["end_ms"] / 1000.0
        conf = t.get("confidence")
        if cur is None or text[:1].isspace():
            flush()
            cur = {"text": text, "start": start, "end": end, "conf": conf}
        else:
            cur["text"] += text
            if end is not None:
                cur["end"] = end
            if conf is not None:
                cur["conf"] = conf if cur["conf"] is None else min(cur["conf"], conf)
    flush()
    return words


def soniox_words(window_id: str, wav: Path, api_key: str, timeout=900):
    """Upload the bytes, transcribe by file_id, then delete both.

    By file_id rather than audio_url because no public bucket credential is
    available here. That is a transport change: the model, the language hints
    and every decode flag are the frozen ones, and the audio bytes are
    identical. The account allows 100 pending / 2000 stored transcriptions, so
    the deletes run in a finally block or the next run starts broken.
    """
    sha = params_sha(SONIOX_PARAMS)
    hit = cached(window_id, "soniox", sha)
    if hit is None:
        base = os.environ.get("SONIOX_API_BASE", "https://api.soniox.com/v1")
        auth = {"Authorization": f"Bearer {api_key}"}
        file_id = tid = None
        try:
            def upload():
                with open(wav, "rb") as fh:
                    return requests.post(f"{base}/files", headers=auth,
                                         files={"file": (wav.name, fh, "audio/wav")},
                                         timeout=timeout)

            file_id = with_retry(upload, f"soniox upload {window_id}").json()["id"]

            body = dict(SONIOX_PARAMS)
            body["file_id"] = file_id
            tid = with_retry(
                lambda: requests.post(f"{base}/transcriptions", headers=auth,
                                      json=body, timeout=120),
                f"soniox create {window_id}").json()["id"]

            deadline = time.time() + timeout
            while True:
                r = requests.get(f"{base}/transcriptions/{tid}", headers=auth, timeout=60)
                r.raise_for_status()
                status = r.json().get("status")
                if status == "completed":
                    break
                if status == "error":
                    raise RuntimeError(f"soniox error: {r.json().get('error_message')}")
                if time.time() > deadline:
                    raise TimeoutError(f"soniox did not finish within {timeout}s")
                time.sleep(2)

            r = requests.get(f"{base}/transcriptions/{tid}/transcript",
                             headers=auth, timeout=120)
            r.raise_for_status()
            hit = r.json()
        finally:
            if tid:
                requests.delete(f"{base}/transcriptions/{tid}", headers=auth,
                                timeout=60).close()
            if file_id:
                requests.delete(f"{base}/files/{file_id}", headers=auth,
                                timeout=60).close()
        store(window_id, "soniox", sha, hit)
    return _soniox_join(hit.get("tokens")), sha


# --------------------------------------------------------------- ours

_MODEL = None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_ours():
    """Load the CT2 build, refusing anything that is not the recorded artifact.

    An output whose producing model is unknown is not evidence, so the hash is
    checked before the first decode rather than asserted in the report.
    """
    global _MODEL
    if _MODEL is None:
        got = _sha256_file(OURS_MODEL_DIR / "model.bin")
        if got != OURS_MODEL_SHA:
            raise RuntimeError(
                f"model.bin is {got[:16]}, expected artifact-ct2-cleanpack-cont-s47 "
                f"{OURS_MODEL_SHA[:16]}")
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel(str(OURS_MODEL_DIR), device=OURS_RUNTIME["device"],
                              compute_type=OURS_RUNTIME["compute_type"],
                              cpu_threads=OURS_RUNTIME["cpu_threads"])
    return _MODEL


def ours_words(window_id: str, wav: Path):
    sha = params_sha({"decode": OURS_DECODE, "runtime": OURS_RUNTIME,
                      "model_sha": OURS_MODEL_SHA[:16]})
    hit = cached(window_id, "ours", sha)
    if hit is None:
        model = load_ours()
        cfg = dict(OURS_DECODE)
        cfg["temperature"] = tuple(cfg["temperature"])
        segments, _info = model.transcribe(str(wav), **cfg)
        words = []
        for seg in segments:
            for w in (seg.words or []):
                raw = (w.word or "").strip()
                if raw:
                    words.append({"raw": raw, "start": w.start, "end": w.end,
                                  "conf": w.probability})
        hit = {"words": words}
        store(window_id, "ours", sha, hit)
    return hit["words"], sha
