#!/usr/bin/env python3
"""OpenCouncil-shaped inference harness for the fine-tuned Whisper adapter.

Purpose
-------
Run our fine-tuned `whisper-large-v3` + LoRA adapter on a single audio clip and
emit the transcript in the *exact* JSON shape the OpenCouncil tasks server
produces, so this is a drop-in alternative to the current ElevenLabs Scribe
transcriber (`schemalabz/opencouncil-tasks`, `src/lib/ScribeTranscribe.ts` →
`src/tasks/transcribe.ts`). That server's `scribeTranscriber.transcribe(
{audioUrl, language})` returns a `Transcript`:

    Transcript {
      metadata: { audio_duration, number_of_distinct_channels,
                  billing_time, transcription_time },
      transcription: {
        languages: string[],
        full_transcript: string,
        utterances: Utterance[] } }
    Utterance { text, language, start, end, confidence, channel,
                speaker, drift, words: Word[] }
    Word { word, start, end, confidence }

Honest divergences from Scribe (documented, not hidden)
-------------------------------------------------------
* **speaker / channel / drift = 0.** Scribe runs with `diarize:false`; speakers
  are assigned *downstream* by pyannote in the OC pipeline. A standalone ASR pass
  has no diarization, so we emit fixed zeros rather than inventing speakers.
* **confidence = 0.0 (placeholder).** whisper-large-v3 does not expose a
  calibrated per-word confidence the way Scribe returns logprobs. We surface a
  placeholder and flag it in `metadata.notes` — downstream must NOT read these as
  Scribe-comparable scores.
* **billing_time = 0.** Self-hosted; there is no per-request billing meter.

Cost
----
CPU-only for a single short clip (base weights are ~3 GB and cached locally) is
free but slow. For real throughput, the same code runs as a RunPod *serverless*
handler (see `handler()` at the bottom) — pay per inference-second, scales to
zero. This script never starts a paid pod on its own.

Usage
-----
    python eval/oc_inference_harness.py \
        --audio https://data.opencouncil.gr/.../clip.mp3 \
        --adapter /home/harold/oc-train-checkpoints/adapter \
        --reference "human corrected text" \
        --out /tmp/transcript.json
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import re
import socket
import sys
import time
import unicodedata
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

MODEL_ID = "openai/whisper-large-v3"
LANGUAGE, TASK = "greek", "transcribe"
SR = 16000

# Utterance segmentation mirrors ScribeTranscribe.ts so downstream (pyannote
# speaker-merge) sees the same granularity: split on a pause, on sentence-final
# punctuation, or at a hard duration cap. Kept deliberately simple — this is a
# test/eval harness, not the production segmenter.
UTTERANCE_PAUSE_SECONDS = 1.0
UTTERANCE_MAX_DURATION_SECONDS = 30.0
# Greek '·'? No — sentence-final set matches Scribe: '.', '!', and both question
# marks (';' U+003B and ';' U+037E). Abbreviations ending in '.' are not split.
_SENT_FINAL = re.compile(r"[.!?;;]$")
_KNOWN_ABBR = {"δηλ", "βλ", "σελ", "αριθ", "κεφ", "λεωφ", "τηλ", "κ", "αρ", "οδ", "π.χ"}

CONF_PLACEHOLDER = 0.0  # whisper exposes no calibrated confidence — see module docstring

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def gnorm(s: str) -> str:
    """Greek-normalise for WER_norm: lowercase, strip accents, ς→σ, drop punct.

    Identical to the training script's `gnorm` so numbers are comparable.
    """
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFC", s).replace("ς", "σ")
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s)).strip()


def _ends_sentence(text: str) -> bool:
    if not _SENT_FINAL.search(text):
        return False
    core = text[:-1]
    if core.lower() in _KNOWN_ABBR or (text.endswith(".") and len(core) <= 2):
        return False
    return True


_ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url: str) -> None:
    """Reject non-public fetch targets (SSRF guard for the serverless handler).

    `handler()` takes an attacker-controlled `audioUrl`, so before fetching we
    require an http(s) scheme and resolve the host, rejecting loopback, private,
    link-local (incl. 169.254.169.254 cloud-metadata), reserved, and multicast
    addresses. Best-effort: this does not close the DNS-rebinding window between
    resolve and fetch, which is acceptable for a self-hosted test harness.
    """
    p = urlparse(url)
    if p.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"blocked URL scheme: {p.scheme!r}")
    host = p.hostname
    if not host:
        raise ValueError("URL has no host")
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve host {host!r}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"blocked non-public address for {host!r}: {ip}")


def _load_audio(src: str) -> "tuple[object, float]":
    """Return (float32 mono 16 kHz waveform, duration_seconds). Accepts URL or path."""
    import librosa

    tmp = None
    path = src
    if src.startswith("http://") or src.startswith("https://"):
        _validate_url(src)
        # Unique per download so concurrent requests can't share/overwrite a path.
        tmp = Path("/tmp") / f"oc_infer_{uuid.uuid4().hex}{Path(urlparse(src).path).suffix or '.mp3'}"
        # data.opencouncil.gr needs no auth; a plain GET is enough.
        req = urllib.request.Request(src, headers={"User-Agent": "oc-inference-harness"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
        path = str(tmp)
    wav, _ = librosa.load(path, sr=SR, mono=True)
    dur = float(len(wav)) / SR
    if tmp is not None:
        try:
            tmp.unlink()
        except OSError:
            pass
    return wav, dur


# Loading base weights + merging the adapter costs several seconds and a few GB;
# a warm serverless worker (or the FastAPI endpoint) must not pay that per request.
# Cache the built pipeline keyed by (adapter, device, dtype).
_PIPELINE_CACHE: dict = {}


def _build_pipeline(adapter: str, device: str | None, dtype_name: str | None):
    import torch
    from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                              pipeline)
    from peft import PeftModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype_name is None:
        dtype_name = "float16" if device == "cuda" else "float32"

    cache_key = (adapter, device, dtype_name)
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dtype = getattr(torch, dtype_name)

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype)
    # merge_and_unload → a plain WhisperForConditionalGeneration the ASR pipeline
    # handles natively (PeftModel wrappers confuse pipeline generate plumbing).
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.to(device)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        torch_dtype=dtype,
        device=device,
    )
    _PIPELINE_CACHE[cache_key] = asr
    return asr


def _words_to_utterances(words: list[dict]) -> list[dict]:
    """Group timestamped words into OC Utterances by pause / punctuation / cap."""
    utterances: list[dict] = []
    cur: list[dict] = []

    def flush():
        if not cur:
            return
        text = "".join(w["word"] for w in cur).strip()
        utterances.append({
            "text": text,
            "language": LANGUAGE,
            "start": cur[0]["start"],
            "end": cur[-1]["end"],
            "confidence": CONF_PLACEHOLDER,
            "channel": 0,
            "speaker": 0,
            "drift": 0,
            "words": [dict(w) for w in cur],
        })
        cur.clear()

    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            span = w["end"] - cur[0]["start"]
            if (gap >= UTTERANCE_PAUSE_SECONDS
                    or _ends_sentence(cur[-1]["word"].strip())
                    or span >= UTTERANCE_MAX_DURATION_SECONDS):
                flush()
        cur.append(w)
    flush()
    return utterances


def transcribe_to_oc(audio: str, adapter: str, device: str | None = None,
                     dtype_name: str | None = None) -> dict:
    """Transcribe `audio` (URL/path) and return the OpenCouncil `Transcript` dict."""
    asr = _build_pipeline(adapter, device, dtype_name)
    wav, duration = _load_audio(audio)

    t0 = time.time()
    out = asr({"array": wav, "sampling_rate": SR},
              return_timestamps="word",
              generate_kwargs={"language": LANGUAGE, "task": TASK})
    elapsed = time.time() - t0

    words: list[dict] = []
    for ch in out.get("chunks", []):
        ts = ch.get("timestamp") or (None, None)
        start, end = ts[0], ts[1]
        # Keep every chunk's text so utterances stay aligned with full_transcript:
        # fall back to the previous word's end (or 0.0) rather than dropping it.
        if start is None:
            start = words[-1]["end"] if words else 0.0
        if end is None:
            end = start
        words.append({
            "word": ch["text"],
            "start": float(start),
            "end": float(end),
            "confidence": CONF_PLACEHOLDER,
        })

    utterances = _words_to_utterances(words)
    return {
        "metadata": {
            "audio_duration": duration,
            "number_of_distinct_channels": 1,
            "billing_time": 0,
            "transcription_time": round(elapsed, 3),
            "notes": ("self-hosted whisper-large-v3+LoRA; speaker/channel/drift=0 "
                      "(diarization is downstream/pyannote); confidence is a "
                      "placeholder (whisper exposes no calibrated logprob)"),
        },
        "transcription": {
            "languages": [LANGUAGE],
            "full_transcript": out.get("text", "").strip(),
            "utterances": utterances,
        },
    }


def _score(hyp: str, ref: str) -> dict:
    import evaluate
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    hn, rn = gnorm(hyp), gnorm(ref)
    return {
        "wer": round(100 * wer.compute(predictions=[" ".join(hyp.split())],
                                       references=[" ".join(ref.split())]), 2),
        "wer_norm": round(100 * wer.compute(predictions=[hn], references=[rn]), 2),
        "cer": round(100 * cer.compute(predictions=[hyp.strip()],
                                       references=[ref.strip()]), 2),
    }


# --- RunPod serverless handler (documented drop-in; import-safe) ----------------
# Deploy this file as a RunPod serverless worker to get pay-per-second GPU ASR.
# event["input"] = {"audioUrl": str, "language"?: str, "reference"?: str}
# returns the OC Transcript (plus "scores" if a reference is supplied). The OC
# tasks server would wrap this endpoint as a `scribeTranscriber`-shaped provider.
def handler(event):  # pragma: no cover - runs only inside RunPod
    import os
    inp = event.get("input", {})
    adapter = os.environ.get("ADAPTER_PATH", "/workspace/adapter")
    result = transcribe_to_oc(inp["audioUrl"], adapter)
    ref = inp.get("reference")
    if ref:
        result["scores"] = _score(result["transcription"]["full_transcript"], ref)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="audio URL or local path")
    ap.add_argument("--adapter", default="/home/harold/oc-train-checkpoints/adapter",
                    help="path or HF id of the LoRA adapter")
    ap.add_argument("--reference", default=None, help="reference text → prints WER/CER")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--dtype", default=None, help="float16|float32 (default: auto)")
    ap.add_argument("--out", default=None, help="write the Transcript JSON here")
    args = ap.parse_args()

    result = transcribe_to_oc(args.audio, args.adapter, args.device, args.dtype)
    if args.reference:
        result["scores"] = _score(result["transcription"]["full_transcript"],
                                   args.reference)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
