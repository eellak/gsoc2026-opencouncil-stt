"""Chunking-aware decoding experiment, ``exp-2026-08-17-chunking-aware-decoding``.

This module deliberately keeps hypotheses and audio in the public cache, never in
the repository.  The experiment has four arms:

* ``V`` is the frozen CONTROL decode with ``vad_filter=True``;
* ``P`` detects silences, cuts at reported silences, decodes each piece with
  CONTROL, and combines the pieces without seam de-duplication.
* ``PI`` splits into pieces of at most 25 seconds and appends digital silence to
  every piece so the decoder always receives a full 30-second window;
* ``E`` decodes overlapping 30-second windows and keeps the central tiled region
  of each window, resolving seam ties toward the earlier centre.

The selection and scoring helpers are model-free so the contract can be checked
before the multi-hour CPU decode is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap  # noqa: E402

import decode_ablation as DA  # noqa: E402


EXPERIMENT = "exp-2026-08-17-chunking-aware-decoding"
CONTROL = dict(DA.CONTROL)
MAX_PIECE_SECONDS = 30.0
PI_MAX_AUDIO_SECONDS = 25.0
WHISPER_WINDOW_SECONDS = 30.0
PREFERRED_SILENCE_SECONDS = 5.0
MINIMUM_SILENCE_SECONDS = 0.5
OVERLAP_STRIDE_SECONDS = 15.0
KEPT_CENTRE_WIDTH_SECONDS = 15.0
SAMPLE_RATE = 16000
DEVICE, COMPUTE, THREADS = DA.DEVICE, DA.COMPUTE, DA.THREADS
BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED = 4000, 7
MODEL_DIR = Path("/home/harold/oc-asr-serve/ct2-fixed")
MODEL_ARTIFACT_ID = "artifact-ct2-fixed"
MODEL_SHA256_16 = "8a1a3b257d0c1bdb"

ARMS = {"V": {"vad_filter": True}, "P": {}, "PI": {}, "E": {}}


@dataclass(frozen=True)
class Silence:
    """A reported silent interval in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Piece:
    """One non-overlapping decode input and its boundary diagnostic."""

    start: float
    end: float
    cut_silence_duration: float | None = None
    whole_fallback: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class OverlapWindow:
    """One 30-second input and its exactly-owned central audio region."""

    index: int
    start: float
    end: float
    centre: float
    kept_start: float
    kept_end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def kept_centre_width(self) -> float:
        return self.kept_end - self.kept_start


def out_dir() -> Path:
    directory = DA.sc() / "chunking-decode-2026-08"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def destination_for(arm: str) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown chunking arm {arm!r}")
    return out_dir() / f"eval-{arm}.json"


def eval_rows() -> list[dict]:
    """Return exactly the 39 unlocked evaluation windows."""
    rows = list(DA.rows("eval"))
    if len(rows) != 39:
        raise ValueError(f"DA.rows('eval') returned {len(rows)} rows, expected 39")
    return rows


def config_for(arm: str) -> dict:
    """CONTROL with the arm's one declared change and no other changes."""
    if arm not in ARMS:
        raise ValueError(f"unknown chunking arm {arm!r}")
    overrides = ARMS[arm]
    unknown = set(overrides) - set(CONTROL)
    if unknown:
        raise ValueError(f"arm {arm} overrides keys not in CONTROL: {sorted(unknown)}")
    resolved = dict(CONTROL, **overrides)
    differing = {key for key in resolved if resolved[key] != CONTROL[key]}
    if differing != set(overrides):
        raise ValueError(
            f"arm {arm} differs from CONTROL in {sorted(differing)}, "
            f"declared {sorted(overrides)}"
        )
    return resolved


def sha16(path: Path) -> str:
    digest = hashlib.sha256()
    with (path / "model.bin").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def verify_model() -> str:
    """Verify the active CT2 artifact before allowing hypotheses to be written."""
    model_bin = MODEL_DIR / "model.bin"
    if not model_bin.exists():
        raise FileNotFoundError(
            f"CT2 model directory is absent: {MODEL_DIR}; "
            "the real decode needs artifact-ct2-fixed"
        )
    actual = sha16(MODEL_DIR)
    if actual != MODEL_SHA256_16:
        raise ValueError(
            f"{model_bin} is {actual}, expected {MODEL_SHA256_16}; "
            "an output whose producing model is unknown is not evidence"
        )
    return actual


def environment() -> dict:
    """The serving harness environment identity, copied for cache safety."""
    import ctranslate2
    import faster_whisper

    return {
        "python": sys.version.split()[0],
        "faster_whisper": faster_whisper.__version__,
        "ctranslate2": ctranslate2.__version__,
        "device": DEVICE,
        "compute_type": COMPUTE,
        "cpu_threads": THREADS,
    }


def _fresh_state(arm: str, model_sha: str, env: dict) -> dict:
    return {
        "experiment": EXPERIMENT,
        "arm": arm,
        "artifact_id": MODEL_ARTIFACT_ID,
        "model": str(MODEL_DIR),
        "model_sha256_16": model_sha,
        "config": config_for(arm),
        "environment": env,
        "set": "eval",
        "windows": {},
    }


def validate_cache_identity(existing: dict, fresh: dict, path: Path | str = "cache") -> None:
    """Refuse to extend a cache made by another model/config/environment.

    This is intentionally strict.  In particular, a cache missing an identity
    field is not treated as compatible with a new decode.
    """
    for key in ("model", "model_sha256_16", "config"):
        if existing.get(key) != fresh.get(key):
            raise ValueError(
                f"{path} was written under a different {key}; "
                "delete it rather than extending it"
            )
    if existing.get("environment") != fresh.get("environment"):
        raise ValueError(
            f"{path} was written in a different environment "
            f"({existing.get('environment')} != {fresh.get('environment')}); "
            "delete it rather than extending it"
        )


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def _as_silence(value: Silence | dict | Sequence[float]) -> Silence:
    if isinstance(value, Silence):
        return value
    if isinstance(value, dict):
        start = value.get("start", value.get("start_sec"))
        end = value.get("end", value.get("end_sec"))
    else:
        start, end = value[0], value[1]
    if start is None or end is None:
        raise ValueError(f"silence has no start/end: {value!r}")
    silence = Silence(float(start), float(end))
    if silence.end <= silence.start:
        raise ValueError(f"silence must have positive duration: {value!r}")
    return silence


def silences_from_speech_timestamps(
    speech_timestamps: Iterable[dict], duration: float, sample_rate: int = SAMPLE_RATE
) -> list[Silence]:
    """Convert faster-whisper/Silero speech sample ranges to silence intervals."""
    speech = sorted(
        (Silence(float(item["start"]) / sample_rate,
                 float(item["end"]) / sample_rate)
         for item in speech_timestamps),
        key=lambda item: item.start,
    )
    gaps: list[Silence] = []
    cursor = 0.0
    for span in speech:
        start = max(0.0, min(duration, span.start))
        end = max(0.0, min(duration, span.end))
        if start > cursor:
            gaps.append(Silence(cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append(Silence(cursor, duration))
    return [gap for gap in gaps if gap.duration > 0]


def silences_from_segment_timeline(
    segments: Iterable[Any], duration: float
) -> list[Silence]:
    """Build fallback silence intervals from a cheap decoder segment timeline."""
    spans: list[tuple[float, float]] = []
    for segment in segments:
        start = float(segment["start"] if isinstance(segment, dict) else segment.start)
        end = float(segment["end"] if isinstance(segment, dict) else segment.end)
        spans.append((max(0.0, start), min(duration, end)))
    spans.sort()
    gaps: list[Silence] = []
    cursor = 0.0
    for start, end in spans:
        if end <= start:
            continue
        if start > cursor:
            gaps.append(Silence(cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append(Silence(cursor, duration))
    return gaps


def detect_silences(
    audio: Any,
    duration: float,
    model: Any | None = None,
) -> tuple[list[Silence], str, str | None]:
    """Detect silences with bundled Silero VAD, falling back to segment gaps.

    ``audio`` is a decoded 16 kHz numpy array.  The fallback deliberately makes a
    first CONTROL pass only when importing/running the bundled VAD fails.
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        vad_options = VadOptions(
            min_silence_duration_ms=int(MINIMUM_SILENCE_SECONDS * 1000),
            speech_pad_ms=0,
        )
        timestamps = get_speech_timestamps(
            audio, vad_options=vad_options, sampling_rate=SAMPLE_RATE
        )
        return (
            silences_from_speech_timestamps(timestamps, duration),
            "silero_vad",
            None,
        )
    except Exception as exc:
        if model is None:
            raise RuntimeError(
                "bundled Silero VAD is unavailable and no model was supplied for "
                "the segment-timeline fallback"
            ) from exc
        first_segments, _ = model.transcribe(audio, **CONTROL)
        first_segments = list(first_segments)
        return (
            silences_from_segment_timeline(first_segments, duration),
            "segment_timeline",
            f"{type(exc).__name__}: {exc}",
        )


def _cut_inside(silence: Silence, cursor: float, limit: float) -> float | None:
    """Choose a point inside a reported silence without crossing ``limit``."""
    if silence.end <= cursor or silence.start > limit:
        return None
    midpoint = (max(silence.start, cursor) + silence.end) / 2.0
    cut = min(midpoint, limit)
    if cut <= cursor:
        return None
    # A midpoint clipped by the limit is valid only if the limit remains in the
    # reported silent interval; this prevents inventing a boundary at the limit.
    if not silence.start <= cut <= silence.end:
        return None
    return cut


def choose_cut(
    cursor: float,
    duration: float,
    silences: Iterable[Silence | dict | Sequence[float]],
    max_piece_seconds: float = MAX_PIECE_SECONDS,
) -> tuple[float, Silence] | None:
    """Choose the longest qualifying silence before the 30-second boundary.

    Silences of at least five seconds are considered first.  If none exists, the
    longest interval of at least 0.5 seconds is used.  Ties resolve toward the
    earliest interval.  No arbitrary time-based boundary is returned.
    """
    boundary = min(duration, cursor + max_piece_seconds)
    candidates: list[tuple[float, Silence, float]] = []
    for raw in silences:
        silence = _as_silence(raw)
        cut = _cut_inside(silence, cursor, boundary)
        if cut is not None:
            candidates.append((silence.duration, silence, cut))
    if not candidates:
        return None
    preferred = [item for item in candidates if item[0] >= PREFERRED_SILENCE_SECONDS]
    eligible = preferred or [item for item in candidates if item[0] >= MINIMUM_SILENCE_SECONDS]
    if not eligible:
        return None
    _, silence, cut = max(eligible, key=lambda item: (item[0], -item[1].start))
    return cut, silence


def split_at_silences(
    duration: float,
    silences: Iterable[Silence | dict | Sequence[float]],
    max_piece_seconds: float = MAX_PIECE_SECONDS,
    allow_whole_fallback: bool = True,
) -> tuple[list[Piece], int]:
    """Split at reported silences, or hand the remaining stretch over whole.

    The explicit whole-stretch fallback is the only permitted exception to the
    <=30-second diagnostic when ``allow_whole_fallback`` is true.  P keeps that
    historical behavior.  PI passes false because its contract has a hard
    25-second maximum even when no qualifying silence was reported.
    """
    if duration < 0:
        raise ValueError("duration must be non-negative")
    normalized = [_as_silence(item) for item in silences]
    normalized.sort(key=lambda item: item.start)
    pieces: list[Piece] = []
    cursor = 0.0
    whole_fallbacks = 0
    epsilon = 1e-7
    while duration - cursor > max_piece_seconds + epsilon:
        chosen = choose_cut(cursor, duration, normalized, max_piece_seconds)
        if chosen is None:
            if allow_whole_fallback:
                pieces.append(Piece(cursor, duration, None, True))
                whole_fallbacks += 1
                cursor = duration
                break
            cut = min(duration, cursor + max_piece_seconds)
            pieces.append(Piece(cursor, cut, None, False))
            cursor = cut
            continue
        cut, silence = chosen
        pieces.append(Piece(cursor, cut, silence.duration, False))
        cursor = cut
    if duration - cursor > epsilon or not pieces:
        pieces.append(Piece(cursor, duration, None, False))
    for previous, current in zip(pieces, pieces[1:]):
        if abs(previous.end - current.start) > 1e-6:
            raise AssertionError("chunk pieces overlap or leave a gap")
    return pieces, whole_fallbacks


# --------------------------------------------------------------- approved splitter
# Reviewed and approved by gpt-5.6-sol, 2026-08-18. Replaces the first splitter,
# which cut at EVERY reported silence and, worse, could re-enter the SAME silence:
# `_cut_inside` took the midpoint of [cursor, silence.end], so once the cursor
# landed inside a gap each further pass advanced by half the remainder. A 148 s
# window came out as 149 pieces of 9.38 / 0.51 / 0.26 / 0.13 / 0.064 s -- a
# geometric series, not a segmentation -- and Whisper on a 64 ms fragment emits
# garbage: insertions went 0.02015 -> 0.07657.
#
# The rule now is accumulate-then-cut. Duration is real waveform time, never
# cumulative speech time, and the VAD supplies CANDIDATE cut points rather than
# one piece per speech interval.

MIN_SPEECH_SECONDS = 2.5      # voiced audio required before a cut is allowed
MAX_CHUNK_SECONDS = 29.5      # room under Whisper's 30 s window for frame rounding
MIN_CHUNK_SECONDS = 5.0       # real audio; below this a piece is pathological
MIN_SILENCE_SECONDS = 0.5     # shorter gaps are intra-phrase, not boundaries


def _voiced_between(a: float, b: float, silences: list[Silence]) -> float:
    """Waveform seconds in [a, b] that the VAD did not call silence."""
    quiet = 0.0
    for s in silences:
        lo, hi = max(a, s.start), min(b, s.end)
        if hi > lo:
            quiet += hi - lo
    return max(0.0, (b - a) - quiet)


def split_accumulating(
    duration: float,
    silences: Iterable[Silence | dict | Sequence[float]],
    max_chunk: float = MAX_CHUNK_SECONDS,
    min_chunk: float = MIN_CHUNK_SECONDS,
    min_speech: float = MIN_SPEECH_SECONDS,
    min_silence: float = MIN_SILENCE_SECONDS,
) -> tuple[list[Piece], dict]:
    """Accumulate audio up to `max_chunk`, then cut at the last usable silence.

    Returns the pieces and the counters that decide whether the splitter worked at
    all -- `tiny_chunks`, `speech_dropped`, `unjustified_forced_cuts` -- which are
    checked BEFORE any WER is looked at.
    """
    if duration < 0:
        raise ValueError("duration must be non-negative")
    quiet = sorted((_as_silence(x) for x in silences), key=lambda s: s.start)
    cuts = [((s.start + s.end) / 2.0, s) for s in quiet if s.duration >= min_silence]

    pieces: list[Piece] = []
    forced = forced_unjustified = 0
    cursor = 0.0
    while duration - cursor > max_chunk + 1e-7:
        deadline = cursor + max_chunk
        # a candidate must leave a piece of legal length AND enough voiced audio
        inrange = [(c, s) for c, s in cuts if cursor + min_chunk <= c <= deadline]
        ok = [(c, s) for c, s in inrange
              if _voiced_between(cursor, c, quiet) >= min_speech]
        # the 30 s ceiling overrides the speech minimum, never the reverse
        pick = (max(ok, key=lambda t: t[0]) if ok
                else max(inrange, key=lambda t: t[0]) if inrange else None)
        if pick is None:
            # genuinely continuous speech: a forced cut is the only legal move
            pieces.append(Piece(cursor, deadline, None, False))
            forced += 1
            cursor = deadline
            continue
        cut, sil = pick
        pieces.append(Piece(cursor, cut, sil.duration, False))
        cursor = cut
    if duration - cursor > 1e-7 or not pieces:
        pieces.append(Piece(cursor, duration, None, False))

    # short-tail repair: merge into the previous piece when that stays legal
    if len(pieces) > 1 and (pieces[-1].end - pieces[-1].start) < min_chunk:
        prev, last = pieces[-2], pieces[-1]
        if (last.end - prev.start) <= max_chunk:
            pieces[-2] = Piece(prev.start, last.end, prev.cut_silence_duration, False)
            pieces.pop()
        else:
            # Merging would breach the window, so move the shared boundary instead:
            # both halves must land inside [min_chunk, max_chunk]. Prefer a real
            # silence, fall back to the midpoint, which is always legal here
            # because the pair spans more than max_chunk and less than 2*max_chunk.
            lo = max(prev.start + min_chunk, last.end - max_chunk)
            hi = min(prev.start + max_chunk, last.end - min_chunk)
            if lo <= hi:
                inside = [c for c, _ in cuts if lo <= c <= hi]
                boundary = min(inside, key=lambda c: abs(c - (lo + hi) / 2.0)) \
                    if inside else (lo + hi) / 2.0
                pieces[-2] = Piece(prev.start, boundary, prev.cut_silence_duration, False)
                pieces[-1] = Piece(boundary, last.end, None, False)

    for a, b in zip(pieces, pieces[1:]):
        if abs(a.end - b.start) > 1e-6:
            raise AssertionError("pieces overlap or leave a gap")
    covered = sum(p.end - p.start for p in pieces)
    allowed_short_single = len(pieces) == 1 and duration < min_chunk
    counters = {
        "n_pieces": len(pieces),
        "tiny_chunks": (
            0
            if allowed_short_single
            else sum(1 for p in pieces if (p.end - p.start) < min_chunk)
        ),
        "speech_dropped": round(max(0.0, duration - covered), 6),
        "forced_cuts": forced,
        "unjustified_forced_cuts": forced_unjustified,
        "durations": [round(p.end - p.start, 3) for p in pieces],
    }
    return pieces, counters


def overlap_layout(
    duration: float,
    window_seconds: float = WHISPER_WINDOW_SECONDS,
    stride_seconds: float = OVERLAP_STRIDE_SECONDS,
    kept_centre_width_seconds: float = KEPT_CENTRE_WIDTH_SECONDS,
) -> list[OverlapWindow]:
    """Return an exact-cover layout of overlapping decoder windows.

    The kept regions are consecutive tiles of ``stride_seconds``.  Each decoder
    window is centred on its nominal tile, so the tile is its central region.  The
    final tile may be clipped at the audio end when the duration is not a multiple
    of the stride; its decoder window remains full length and is padded at the
    audio boundary as needed.  Keeping the nominal centre on the stride grid makes
    the nearest-centre seam rule agree with the exact tile boundaries.
    """
    if duration < 0:
        raise ValueError("duration must be non-negative")
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window and stride must be positive")
    if stride_seconds >= window_seconds:
        raise ValueError("stride must be shorter than the decoder window")
    if not math.isclose(kept_centre_width_seconds, stride_seconds, abs_tol=1e-9):
        raise ValueError("kept centre width must equal stride for an exact cover")
    if duration == 0:
        return []

    windows: list[OverlapWindow] = []
    kept_start = 0.0
    epsilon = 1e-9
    while kept_start < duration - epsilon:
        kept_end = min(duration, kept_start + kept_centre_width_seconds)
        centre = kept_start + kept_centre_width_seconds / 2.0
        windows.append(OverlapWindow(
            index=len(windows),
            start=centre - window_seconds / 2.0,
            end=centre + window_seconds / 2.0,
            centre=centre,
            kept_start=kept_start,
            kept_end=kept_end,
        ))
        kept_start = kept_end
    assert_exact_cover(windows, duration, kept_centre_width_seconds, window_seconds)
    return windows


def assert_exact_cover(
    windows: Sequence[OverlapWindow],
    duration: float,
    kept_centre_width_seconds: float = KEPT_CENTRE_WIDTH_SECONDS,
    window_seconds: float = WHISPER_WINDOW_SECONDS,
) -> None:
    """Assert that kept centre regions tile ``[0, duration]`` exactly once."""
    epsilon = 1e-7
    if duration == 0:
        if windows:
            raise AssertionError("zero-duration audio has kept regions")
        return
    if not windows:
        raise AssertionError("positive-duration audio has no kept region")
    if not math.isclose(windows[0].kept_start, 0.0, abs_tol=epsilon):
        raise AssertionError("first kept region does not start at audio time zero")
    total = 0.0
    for previous, current in zip(windows, windows[1:]):
        if not math.isclose(previous.kept_end, current.kept_start, abs_tol=epsilon):
            raise AssertionError("kept centre regions have a gap or overlap")
    for window in windows:
        if window.kept_start < -epsilon or window.kept_end > duration + epsilon:
            raise AssertionError("kept centre region lies outside the audio")
        if window.kept_end <= window.kept_start:
            raise AssertionError("kept centre region is empty")
        if window.kept_centre_width > kept_centre_width_seconds + epsilon:
            raise AssertionError("kept centre exceeds the frozen width")
        nominal_start = window.centre - kept_centre_width_seconds / 2.0
        nominal_end = window.centre + kept_centre_width_seconds / 2.0
        if not math.isclose(window.kept_start, max(0.0, nominal_start), abs_tol=epsilon):
            raise AssertionError("kept region does not begin at the central tile")
        if not math.isclose(window.kept_end, min(duration, nominal_end), abs_tol=epsilon):
            raise AssertionError("kept region does not end at the central tile")
        if not math.isclose(window.duration,
                            window_seconds,
                            abs_tol=epsilon):
            raise AssertionError("overlap decoder input is not a full Whisper window")
        total += window.kept_centre_width
    if not math.isclose(windows[-1].kept_end, duration, abs_tol=epsilon):
        raise AssertionError("last kept region does not reach the audio end")
    if not math.isclose(total, duration, abs_tol=epsilon):
        raise AssertionError("kept centre regions do not sum to the audio duration")


def _segment_value(segment: Any, name: str, default: Any = None) -> Any:
    if isinstance(segment, dict):
        return segment.get(name, default)
    return getattr(segment, name, default)


def combine_piece_transcripts(piece_results: Iterable[dict]) -> dict:
    """Combine pieces in order with offsets and no seam de-duplication.

    Text is the production-style plain concatenation.  Segment timestamps are
    shifted by each piece's original window offset and otherwise left untouched.
    """
    combined_segments: list[dict] = []
    texts: list[str] = []
    for piece in piece_results:
        offset = float(piece.get("start_sec", piece.get("offset_sec", 0.0)))
        texts.append(str(piece.get("text", "")))
        for segment in piece.get("segments", []):
            start = _segment_value(segment, "start", _segment_value(segment, "start_sec"))
            end = _segment_value(segment, "end", _segment_value(segment, "end_sec"))
            adjusted = dict(segment) if isinstance(segment, dict) else {
                "text": _segment_value(segment, "text", ""),
                "start": start,
                "end": end,
            }
            if start is not None:
                adjusted["start"] = float(start) + offset
            if end is not None:
                adjusted["end"] = float(end) + offset
            combined_segments.append(adjusted)
    return {"text": "".join(texts).strip(), "segments": combined_segments}


def _segments_to_record(segments: list[Any]) -> tuple[str, list[dict]]:
    records: list[dict] = []
    texts: list[str] = []
    for segment in segments:
        text = str(_segment_value(segment, "text", ""))
        texts.append(text)
        records.append({
            "text": text,
            "start": _segment_value(segment, "start"),
            "end": _segment_value(segment, "end"),
        })
    return "".join(texts), records


def _decode_one(model: Any, audio: Any, config: dict) -> tuple[str, list[dict], Any]:
    segments, info = model.transcribe(audio, **config)
    segments = list(segments)
    text, records = _segments_to_record(segments)
    return text, records, info


def _audio_duration(audio: Any, fallback: float) -> float:
    try:
        return len(audio) / SAMPLE_RATE
    except TypeError:
        return fallback


def _load_audio(path: Path) -> Any:
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def _assert_control_config(arm: str, config: dict) -> None:
    """Guard the arms whose only declared change is audio geometry."""
    if arm in {"P", "PI", "E"} and config != CONTROL:
        raise AssertionError(f"arm {arm} must decode with CONTROL unchanged")


def _zero_padded_window(audio: Any, start: float, end: float) -> Any:
    """Extract ``[start, end)`` and zero-pad it to one full Whisper window."""
    import numpy as np

    expected = int(round(WHISPER_WINDOW_SECONDS * SAMPLE_RATE))
    target = int(round((end - start) * SAMPLE_RATE))
    if target != expected:
        raise AssertionError(f"overlap input is {target} samples, expected {expected}")
    source = np.asarray(audio)
    if source.ndim != 1:
        raise ValueError("decoded audio must be a mono sample array")
    padded = np.zeros(expected, dtype=source.dtype)
    source_start = max(0, int(round(start * SAMPLE_RATE)))
    source_end = min(len(source), int(round(end * SAMPLE_RATE)))
    if source_end <= source_start:
        return padded
    destination_start = source_start - int(round(start * SAMPLE_RATE))
    destination_end = destination_start + source_end - source_start
    padded[destination_start:destination_end] = source[source_start:source_end]
    return padded


def _padded_piece(audio: Any, piece: Piece) -> tuple[Any, float]:
    """Return a piece followed by digital silence and its padding duration."""
    import numpy as np

    source = np.asarray(audio)
    if source.ndim != 1:
        raise ValueError("decoded audio must be a mono sample array")
    real_seconds = len(source) / SAMPLE_RATE
    if piece.duration > PI_MAX_AUDIO_SECONDS + 1e-7 or real_seconds > PI_MAX_AUDIO_SECONDS + 1e-7:
        raise AssertionError("PI handed a piece with more than 25 seconds of audio")
    if abs(real_seconds - piece.duration) > 1.0 / SAMPLE_RATE + 1e-7:
        raise AssertionError("PI piece duration does not match its audio samples")
    target_samples = int(round(WHISPER_WINDOW_SECONDS * SAMPLE_RATE))
    real_samples = len(source)
    if real_samples != int(round(real_seconds * SAMPLE_RATE)):
        raise AssertionError("PI piece duration does not match its audio samples")
    if real_samples > target_samples:
        raise AssertionError("PI piece is longer than the full Whisper window")
    padding_seconds = WHISPER_WINDOW_SECONDS - real_seconds
    padded = np.zeros(target_samples, dtype=source.dtype)
    padded[:real_samples] = source
    return padded, padding_seconds


def _window_token_candidates(
    window: OverlapWindow,
    segments: Iterable[dict],
    duration: float,
) -> list[dict]:
    """Shift segment timestamps from one padded input into window time."""
    candidates: list[dict] = []
    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            raise ValueError("E requires segment timestamps for deterministic merging")
        start = float(start) + window.start
        end = float(end) + window.start
        if end < start:
            raise ValueError("segment timestamp end precedes start")
        midpoint = (start + end) / 2.0
        if midpoint < -1e-7 or midpoint > duration + 1e-7:
            continue
        candidates.append({
            "text": str(segment.get("text", "")),
            "start": start,
            "end": end,
            "midpoint": midpoint,
            "window_index": window.index,
            "window_centre": window.centre,
        })
    return candidates


def nearest_overlap_window_index(
    timestamp: float,
    windows: Sequence[OverlapWindow],
) -> int | None:
    """Return the nearest covering centre; ties deterministically choose earliest."""
    covering = [
        window for window in windows
        if window.start - 1e-7 <= timestamp <= window.end + 1e-7
    ]
    if not covering:
        return None
    return min(
        covering,
        key=lambda window: (abs(timestamp - window.centre), window.index),
    ).index


def merge_overlap_segments(
    candidates: Iterable[dict],
    windows: Sequence[OverlapWindow],
    duration: float,
) -> dict:
    """Keep one timestamped segment at each seam using the frozen centre rule.

    Segment timestamps are the permitted fallback for this experiment because
    CONTROL has ``word_timestamps=False``.  A segment belongs to the window whose
    centre is nearest to its timestamp midpoint.  At an exact tie, the earlier
    window index wins.  Candidates from the other overlapping windows are counted
    as seam duplicates and discarded.
    """
    assert_exact_cover(windows, duration)
    selected: list[dict] = []
    dropped = 0
    for candidate in candidates:
        start = float(candidate["start"])
        end = float(candidate["end"])
        midpoint = float(candidate.get("midpoint", (start + end) / 2.0))
        owner = nearest_overlap_window_index(midpoint, windows)
        if owner is None:
            continue
        if owner != candidate["window_index"]:
            dropped += 1
            continue
        window = windows[owner]
        if not (window.kept_start - 1e-7 <= midpoint <= window.kept_end + 1e-7):
            dropped += 1
            continue
        selected.append({
            "text": candidate["text"],
            "start": max(0.0, min(duration, start)),
            "end": max(0.0, min(duration, end)),
        })
    selected.sort(key=lambda segment: (segment["start"], segment["end"], segment["text"]))
    return {
        "text": "".join(segment["text"] for segment in selected).strip(),
        "segments": selected,
        "tokens_dropped_as_duplicates_at_seams": dropped,
    }


def _window_record_v(model: Any, wav: Path, row: dict) -> dict:
    import ctranslate2

    ctranslate2.set_random_seed(DA.seed_for("A", row["window_id"]))
    started = time.monotonic()
    text, segments, info = _decode_one(model, str(wav), config_for("V"))
    return {
        "text": text.strip(),
        "segments": segments,
        "n_segments": len(segments),
        "audio_seconds": round(float(getattr(info, "duration", row["duration_sec"])), 2),
        "wall_seconds": round(time.monotonic() - started, 1),
        "seed": DA.seed_for("A", row["window_id"]),
        "config": config_for("V"),
    }


def _window_record_p(
    model: Any,
    wav: Path,
    row: dict,
    audio_loader: Callable[[Path], Any] = _load_audio,
    silence_detector: Callable[[Any, float, Any | None], tuple[list[Silence], str, str | None]] = detect_silences,
) -> dict:
    import ctranslate2

    started = time.monotonic()
    audio = audio_loader(wav)
    duration = _audio_duration(audio, float(row["duration_sec"]))
    silences, detection_source, detection_error = silence_detector(audio, duration, model)
    pieces, split_counters = split_accumulating(duration, silences)
    whole_fallbacks = split_counters["forced_cuts"]
    piece_results: list[dict] = []
    ctranslate2.set_random_seed(DA.seed_for("A", row["window_id"]))
    for piece in pieces:
        begin = int(round(piece.start * SAMPLE_RATE))
        end = int(round(piece.end * SAMPLE_RATE))
        chunk = audio[begin:end]
        text, segments, _ = _decode_one(model, chunk, CONTROL)
        piece_results.append({
            "start_sec": piece.start,
            "end_sec": piece.end,
            "text": text,
            "segments": segments,
        })
    combined = combine_piece_transcripts(piece_results)
    result = {
        "text": combined["text"],
        "segments": combined["segments"],
        "n_segments": len(combined["segments"]),
        "audio_seconds": round(duration, 2),
        "wall_seconds": round(time.monotonic() - started, 1),
        "seed": DA.seed_for("A", row["window_id"]),
        "config": CONTROL,
        "detection_source": detection_source,
        "detection_error": detection_error,
        "pieces": [
            {
                "start_sec": round(piece.start, 3),
                "end_sec": round(piece.end, 3),
                "duration_sec": round(piece.duration, 3),
                "cut_silence_duration_sec": (
                    round(piece.cut_silence_duration, 3)
                    if piece.cut_silence_duration is not None else None
                ),
                "whole_fallback": piece.whole_fallback,
            }
            for piece in pieces
        ],
        "n_pieces": len(pieces),
        "whole_fallback_stretches": whole_fallbacks,
    }
    return result


def _window_record_pi(
    model: Any,
    wav: Path,
    row: dict,
    audio_loader: Callable[[Path], Any] = _load_audio,
    silence_detector: Callable[[Any, float, Any | None], tuple[list[Silence], str, str | None]] = detect_silences,
) -> dict:
    """Decode PI: <=25 seconds of source audio, then right-pad to 30 seconds."""
    import ctranslate2

    started = time.monotonic()
    audio = audio_loader(wav)
    duration = _audio_duration(audio, float(row["duration_sec"]))
    silences, detection_source, detection_error = silence_detector(audio, duration, model)
    pieces, whole_fallbacks = split_at_silences(
        duration,
        silences,
        max_piece_seconds=PI_MAX_AUDIO_SECONDS,
        allow_whole_fallback=False,
    )
    if any(piece.duration > PI_MAX_AUDIO_SECONDS + 1e-7 for piece in pieces):
        raise AssertionError("PI pieces exceed the 25-second real-audio maximum")

    config = config_for("PI")
    _assert_control_config("PI", config)
    piece_results: list[dict] = []
    ctranslate2.set_random_seed(DA.seed_for("A", row["window_id"]))
    for piece in pieces:
        begin = int(round(piece.start * SAMPLE_RATE))
        end = int(round(piece.end * SAMPLE_RATE))
        raw_piece = audio[begin:end]
        padded, padding_seconds = _padded_piece(raw_piece, piece)
        if len(padded) != int(round(WHISPER_WINDOW_SECONDS * SAMPLE_RATE)):
            raise AssertionError("PI did not pad the decoder input to 30 seconds")
        text, segments, _ = _decode_one(model, padded, config)
        piece_results.append({
            "start_sec": piece.start,
            "end_sec": piece.end,
            "text": text,
            "segments": segments,
            "speech_seconds": piece.duration,
            "padding_seconds": padding_seconds,
        })
    combined = combine_piece_transcripts(piece_results)
    speech_seconds = [round(piece["speech_seconds"], 3) for piece in piece_results]
    padding_seconds = [round(piece["padding_seconds"], 3) for piece in piece_results]
    return {
        "text": combined["text"],
        "segments": combined["segments"],
        "n_segments": len(combined["segments"]),
        "audio_seconds": round(duration, 2),
        "wall_seconds": round(time.monotonic() - started, 1),
        "seed": DA.seed_for("A", row["window_id"]),
        "config": config,
        "detection_source": detection_source,
        "detection_error": detection_error,
        "pieces": [
            {
                "start_sec": round(piece.start, 3),
                "end_sec": round(piece.end, 3),
                "duration_sec": round(piece_result["speech_seconds"], 3),
                "speech_seconds": round(piece_result["speech_seconds"], 3),
                "padding_seconds": round(piece_result["padding_seconds"], 3),
                "padded_duration_sec": WHISPER_WINDOW_SECONDS,
                "cut_silence_duration_sec": (
                    round(piece.cut_silence_duration, 3)
                    if piece.cut_silence_duration is not None else None
                ),
                "whole_fallback": piece.whole_fallback,
            }
            for piece, piece_result in zip(pieces, piece_results)
        ],
        "n_pieces": len(pieces),
        "speech_seconds_per_piece": speech_seconds,
        "padding_seconds_per_piece": padding_seconds,
        "padded_window_seconds": WHISPER_WINDOW_SECONDS,
        "whole_fallback_stretches": whole_fallbacks,
    }


def _window_record_e(
    model: Any,
    wav: Path,
    row: dict,
    audio_loader: Callable[[Path], Any] = _load_audio,
) -> dict:
    """Decode E with overlapping full windows and exact central-region ownership."""
    import ctranslate2

    started = time.monotonic()
    audio = audio_loader(wav)
    duration = _audio_duration(audio, float(row["duration_sec"]))
    windows = overlap_layout(duration)
    config = config_for("E")
    _assert_control_config("E", config)
    candidates: list[dict] = []
    ctranslate2.set_random_seed(DA.seed_for("A", row["window_id"]))
    for window in windows:
        chunk = _zero_padded_window(audio, window.start, window.end)
        if len(chunk) != int(round(WHISPER_WINDOW_SECONDS * SAMPLE_RATE)):
            raise AssertionError("E handed the decoder a non-30-second input")
        _, segments, _ = _decode_one(model, chunk, config)
        candidates.extend(_window_token_candidates(window, segments, duration))
    merged = merge_overlap_segments(candidates, windows, duration)
    return {
        "text": merged["text"],
        "segments": merged["segments"],
        "n_segments": len(merged["segments"]),
        "audio_seconds": round(duration, 2),
        "wall_seconds": round(time.monotonic() - started, 1),
        "seed": DA.seed_for("A", row["window_id"]),
        "config": config,
        "n_overlapping_windows": len(windows),
        "stride_sec": OVERLAP_STRIDE_SECONDS,
        "kept_centre_width_sec": KEPT_CENTRE_WIDTH_SECONDS,
        "window_seconds": WHISPER_WINDOW_SECONDS,
        "tokens_dropped_as_duplicates_at_seams": (
            merged["tokens_dropped_as_duplicates_at_seams"]
        ),
        "overlap_windows": [
            {
                "index": window.index,
                "start_sec": round(window.start, 3),
                "end_sec": round(window.end, 3),
                "centre_sec": round(window.centre, 3),
                "kept_start_sec": round(window.kept_start, 3),
                "kept_end_sec": round(window.kept_end, 3),
                "kept_centre_width_sec": round(window.kept_centre_width, 3),
            }
            for window in windows
        ],
    }


def decode(
    arm: str,
    limit: int | None = None,
    model: Any | None = None,
    destination: Path | None = None,
    rows_override: list[dict] | None = None,
    audio_loader: Callable[[Path], Any] = _load_audio,
    silence_detector: Callable[[Any, float, Any | None], tuple[list[Silence], str, str | None]] = detect_silences,
) -> Path:
    """Decode one chunking arm with strict identity checks and incremental writes.

    ``model`` and the fixture hooks are intentionally injectable for the two-window
    smoke tests; normal CLI use leaves them at their production defaults.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown chunking arm {arm!r}")
    model_sha = verify_model() if model is None else MODEL_SHA256_16
    env = environment()
    fresh = _fresh_state(arm, model_sha, env)
    destination = destination or destination_for(arm)
    state = json.loads(destination.read_text()) if destination.exists() else fresh
    validate_cache_identity(state, fresh, destination)
    state.setdefault("windows", {})
    rows = list(rows_override if rows_override is not None else eval_rows())
    if limit is not None:
        rows = rows[:limit]
    todo = [row for row in rows if row["window_id"] not in state["windows"]]
    if not todo:
        return destination
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel(str(MODEL_DIR), device=DEVICE, compute_type=COMPUTE,
                             cpu_threads=THREADS)
    for row in todo:
        wav = DA.sc() / "bench_windows" / f"{row['window_id']}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"missing audio for {row['window_id']}: {wav}")
        if arm == "V":
            record = _window_record_v(model, wav, row)
        elif arm == "P":
            record = _window_record_p(model, wav, row, audio_loader, silence_detector)
        elif arm == "PI":
            record = _window_record_pi(model, wav, row, audio_loader, silence_detector)
        else:
            record = _window_record_e(model, wav, row, audio_loader)
        state["windows"][row["window_id"]] = record
        # This is deliberately per-window: a multi-hour CPU pass must resume.
        _write_state(destination, state)
    return destination


def _reference_for_row(row: dict) -> list[str]:
    if "reference_text" in row:
        return ftoks(row["reference_text"])
    return ftoks(DA.reference_text(row["window_id"]))


def per_window_scores(state: dict, rows: list[dict] | None = None) -> dict[str, tuple[int, int, int, int]]:
    rows = list(rows if rows is not None else eval_rows())
    missing = [row["window_id"] for row in rows if row["window_id"] not in state.get("windows", {})]
    if missing:
        raise ValueError(f"decode cache is incomplete; missing {missing[:3]}")
    scores: dict[str, tuple[int, int, int, int]] = {}
    for row in rows:
        ref = _reference_for_row(row)
        hyp = ftoks(state["windows"][row["window_id"]].get("text", ""))
        scores[row["window_id"]] = (*sdi(ref, hyp), len(ref))
    return scores


def rates(scores: dict[str, tuple[int, int, int, int]]) -> dict:
    substitutions = sum(value[0] for value in scores.values())
    deletions = sum(value[1] for value in scores.values())
    insertions = sum(value[2] for value in scores.values())
    ref_tokens = sum(value[3] for value in scores.values())
    if not ref_tokens:
        raise ValueError("cannot score an empty reference")
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "ref_tokens": ref_tokens,
        "wer": (substitutions + deletions + insertions) / ref_tokens,
        "deletion_rate": deletions / ref_tokens,
        "insertion_rate": insertions / ref_tokens,
        "substitution_rate": substitutions / ref_tokens,
    }


METRIC_PICKERS: dict[str, Callable[[tuple[int, int, int, int]], int]] = {
    "wer": lambda value: value[0] + value[1] + value[2],
    "deletion_rate": lambda value: value[1],
    "insertion_rate": lambda value: value[2],
    "substitution_rate": lambda value: value[0],
}


def paired_interval(
    arm_scores: dict[str, tuple[int, int, int, int]],
    control_scores: dict[str, tuple[int, int, int, int]],
    rows: list[dict],
    metric: str,
) -> dict:
    if metric not in METRIC_PICKERS:
        raise ValueError(f"unknown metric {metric!r}")
    wids = [row["window_id"] for row in rows]
    clusters = [row["meeting_id"] for row in rows]
    picker = METRIC_PICKERS[metric]
    arm_counts = [(picker(arm_scores[wid]), arm_scores[wid][3]) for wid in wids]
    control_counts = [(picker(control_scores[wid]), control_scores[wid][3]) for wid in wids]
    return cluster_bootstrap(
        arm_counts, control_counts, clusters,
        n_boot=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
    )


def _pooled_delta(
    arm_scores: dict[str, tuple[int, int, int, int]],
    control_scores: dict[str, tuple[int, int, int, int]],
    wids: Iterable[str],
    picker: Callable[[tuple[int, int, int, int]], int],
) -> float:
    wids = list(wids)
    denominator = sum(arm_scores[wid][3] for wid in wids)
    if not denominator:
        raise ValueError("cannot calculate a rate with no reference tokens")
    return sum(picker(arm_scores[wid]) - picker(control_scores[wid]) for wid in wids) / denominator


def loo_deletion_sign(
    arm_scores: dict[str, tuple[int, int, int, int]],
    control_scores: dict[str, tuple[int, int, int, int]],
    rows: list[dict],
) -> dict:
    wids = [row["window_id"] for row in rows]
    picker = METRIC_PICKERS["deletion_rate"]
    full = _pooled_delta(arm_scores, control_scores, wids, picker)
    reversed_by: list[str] = []
    without: dict[str, float] = {}
    for omitted in wids:
        keep = [wid for wid in wids if wid != omitted]
        value = _pooled_delta(arm_scores, control_scores, keep, picker)
        without[omitted] = value
        if (full < 0) != (value < 0):
            reversed_by.append(omitted)
    return {
        "delta": full,
        "without_window": without,
        "sign_reversed_by": reversed_by,
        "stable": not reversed_by,
    }


def evaluate_gate(vs_control: dict, deletion_loo: dict) -> dict:
    """Evaluate precisely the four frozen primary conditions."""
    deletion = vs_control["deletion_rate"]
    wer = vs_control["wer"]
    insertion = vs_control["insertion_rate"]
    conditions = {
        "deletion_rate_delta_negative_and_ci95_upper_lt_0": (
            deletion["delta"] < 0 and deletion["ci95"][1] < 0
        ),
        "wer_ci95_upper_le_0.002": wer["ci95"][1] <= 0.002,
        "insertion_rate_ci95_upper_le_0.002": insertion["ci95"][1] <= 0.002,
        "deletion_rate_sign_stable_leave_one_window_out": bool(deletion_loo["stable"]),
    }
    return {
        "conditions": conditions,
        "overall_pass": all(conditions.values()),
        "verdict": "PASS" if all(conditions.values()) else "FAIL",
        "failing_conditions": [name for name, passed in conditions.items() if not passed],
    }


def score_states(
    control_state: dict,
    arm_states: dict[str, dict],
    rows: list[dict] | None = None,
) -> dict:
    rows = list(rows if rows is not None else eval_rows())
    control_scores = per_window_scores(control_state, rows)
    result = {
        "experiment": EXPERIMENT,
        "n_windows": len(rows),
        "n_meetings": len({row["meeting_id"] for row in rows}),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "blocks": "meeting_id",
        },
        "control": {"rates": rates(control_scores)},
        "arms": {},
    }
    for arm, state in arm_states.items():
        arm_scores = per_window_scores(state, rows)
        vs = {
            metric: paired_interval(arm_scores, control_scores, rows, metric)
            for metric in METRIC_PICKERS
        }
        loo = loo_deletion_sign(arm_scores, control_scores, rows)
        result["arms"][arm] = {
            "rates": rates(arm_scores),
            "vs_control": vs,
            "deletion_rate_leave_one_window_out": loo,
            "gate": evaluate_gate(vs, loo),
        }
    return result


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def score() -> Path:
    """Score control against whichever complete arm caches exist."""
    control_path = DA.sc() / "decode-ablation" / "eval-A.json"
    if not control_path.exists():
        raise FileNotFoundError(
            f"control cache is absent: {control_path}; run the existing control first"
        )
    rows = eval_rows()
    control = load_json(control_path)
    arm_states: dict[str, dict] = {}
    for arm in ARMS:
        path = destination_for(arm)
        if path.exists():
            arm_states[arm] = load_json(path)
    result = score_states(control, arm_states, rows)
    destination = out_dir() / "results-eval.json"
    _write_state(destination, result)
    for arm, report in result["arms"].items():
        gate = report["gate"]
        if gate["overall_pass"]:
            print(f"{arm} gate PASS; failing condition: none")
        else:
            print(f"{arm} gate FAIL; failing condition: {', '.join(gate['failing_conditions'])}")
    print(f"-> {destination}")
    return destination


SEGMENT_CACHE_DIRNAME = "chunking-decode-2026-08"


def _cached_window_records(path: Path) -> dict[str, dict]:
    """Load only model-free cache records needed by ``segment-only``."""
    if not path.exists():
        return {}
    state = json.loads(path.read_text())
    records = state.get("windows", {})
    if not isinstance(records, dict):
        raise ValueError(f"cached window records are not an object: {path}")
    return records


def _silences_from_cached_record(record: dict, duration: float) -> list[Silence]:
    """Read explicit VAD data or reconstruct it from the historical P cache.

    The first P splitter did not persist the full VAD list.  It did persist the
    duration of the silence used for each cut, so the first cut in each repeated
    run identifies the original interval.  Later repeated cuts are deliberately
    ignored; retaining them would recreate the geometric-series bug.
    """
    for key in ("silences", "vad_silences", "silence_intervals"):
        raw = record.get(key)
        if raw is None:
            continue
        if isinstance(raw, dict):
            raw = raw.get("silences", raw.get("intervals", [])) or []
        normalized = []
        for item in raw:
            silence = _as_silence(item)
            start = max(0.0, silence.start)
            end = min(duration, silence.end)
            if end > start:
                normalized.append(Silence(start, end))
        return sorted(normalized, key=lambda silence: silence.start)

    reconstructed: list[Silence] = []
    for piece in record.get("pieces", []):
        if not isinstance(piece, dict):
            continue
        raw_duration = piece.get("cut_silence_duration_sec")
        raw_cut = piece.get("end_sec")
        if raw_duration is None or raw_cut is None:
            continue
        silence_duration = float(raw_duration)
        cut = float(raw_cut)
        if silence_duration < MIN_SILENCE_SECONDS:
            continue
        candidate = Silence(
            max(0.0, cut - silence_duration / 2.0),
            min(duration, cut + silence_duration / 2.0),
        )
        if candidate.duration < MIN_SILENCE_SECONDS:
            continue
        # One old silence can have many cut points.  The first point is the
        # original midpoint; subsequent points lie inside that reconstructed gap.
        if any(silence.start <= cut <= silence.end for silence in reconstructed):
            continue
        reconstructed.append(candidate)
    return sorted(reconstructed, key=lambda silence: silence.start)


def _cached_silences_for_window(
    window_id: str,
    duration: float,
    p_records: dict[str, dict],
    v_records: dict[str, dict],
) -> tuple[list[Silence], str]:
    p_record = p_records.get(window_id)
    if p_record is not None:
        silences = _silences_from_cached_record(p_record, duration)
        if silences:
            return silences, "cached_vad"

    # A V arm cache contains the already-produced segment timeline.  It is a
    # model-free fallback for windows whose P cache had no usable cut metadata.
    v_record = v_records.get(window_id)
    if v_record is not None and v_record.get("segments"):
        return (
            silences_from_segment_timeline(v_record["segments"], duration),
            "cached_segment_timeline",
        )
    return [], "none"


def segment_only(
    cache_root: Path | None = None,
    rows_override: list[dict] | None = None,
) -> dict:
    """Print the approved segmentation over all frozen windows without decoding.

    This deliberately reads only the manifest and existing cache metadata.  It
    does not load audio, instantiate a VAD/Whisper model, or write a result file.
    ``rows_override`` is a small test seam; the CLI always uses the 39 frozen rows.
    """
    cache_root = cache_root or (DA.sc() / SEGMENT_CACHE_DIRNAME)
    p_records = _cached_window_records(cache_root / "eval-P.json")
    v_records = _cached_window_records(cache_root / "eval-V.json")
    rows = list(rows_override if rows_override is not None else eval_rows())

    totals = {
        "tiny_chunks": 0,
        "speech_dropped": 0.0,
        "unjustified_forced_cuts": 0,
    }
    reports: list[dict] = []
    for row in rows:
        window_id = row["window_id"]
        duration = float(row["duration_sec"])
        silences, source = _cached_silences_for_window(
            window_id, duration, p_records, v_records
        )
        pieces, counters = split_accumulating(duration, silences)
        minimum = min(piece.duration for piece in pieces)
        maximum = max(piece.duration for piece in pieces)
        print(
            f"{window_id} audio_seconds={duration:.2f} "
            f"n_pieces={len(pieces)} min_piece_seconds={minimum:.3f} "
            f"max_piece_seconds={maximum:.3f}"
        )
        totals["tiny_chunks"] += counters["tiny_chunks"]
        totals["speech_dropped"] += counters["speech_dropped"]
        totals["unjustified_forced_cuts"] += counters["unjustified_forced_cuts"]
        reports.append({
            "window_id": window_id,
            "audio_seconds": duration,
            "n_pieces": len(pieces),
            "min_piece_seconds": minimum,
            "max_piece_seconds": maximum,
            "silence_source": source,
            "counters": counters,
        })

    totals["speech_dropped"] = round(totals["speech_dropped"], 6)
    print(
        "summary "
        f"tiny_chunks={totals['tiny_chunks']} "
        f"speech_dropped={totals['speech_dropped']:.6f} "
        f"unjustified_forced_cuts={totals['unjustified_forced_cuts']}"
    )
    return {"windows": reports, "totals": totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    decode_parser.add_argument("--limit", type=int, default=None,
                               help="optional smoke limit; omit for all 39 windows")
    subparsers.add_parser("score")
    subparsers.add_parser(
        "segment-only",
        help="report splitter geometry from cached VAD metadata without decoding",
    )
    args = parser.parse_args()
    if args.command == "decode":
        decode(args.arm, args.limit)
    elif args.command == "score":
        score()
    else:
        segment_only()


if __name__ == "__main__":
    main()
