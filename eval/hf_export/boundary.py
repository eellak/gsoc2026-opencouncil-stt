"""Boundary-quality classification for utterance clips.

Closes the 2026-07-03 known issue in docs/decisions/audio.md: raw CSV
utterance spans can cut mid-syllable or bracket neighbouring speech. Each clip
is sliced with MARGIN_S extra context per side; the label is force-aligned
inside that extended clip and VAD provides speech segments. This module holds
the pure geometry; the model calls live in the boundary stage of build.py.
"""
from __future__ import annotations

from dataclasses import dataclass

MARGIN_S = 1.0        # extra context sliced on each side of the raw span
EDGE_TOL_S = 0.15     # how far a word may poke past the raw edge before "cut"
BLEED_MIN_S = 0.25    # min un-labelled VAD speech inside the span -> bleed
PAD_S = 0.2           # padding added around the aligned span
OK_SHIFT_S = 0.30     # max per-edge adjustment still called "ok"
MIN_MEAN_SCORE = 0.15 # below this mean alignment score -> align_failed


@dataclass
class BoundaryResult:
    status: str        # ok|adjusted|suspect_cut_start|suspect_cut_end|suspect_bleed_in|align_failed
    start_off: float   # proposed start, seconds relative to the EXTENDED clip
    end_off: float
    mean_score: float


def _speech_overlap(vad: list[dict], lo: float, hi: float) -> float:
    """Total VAD speech seconds inside [lo, hi]."""
    return sum(max(0.0, min(seg["end"], hi) - max(seg["start"], lo))
               for seg in vad)


def _sane_words(words: list[dict], clip_dur: float) -> list[dict]:
    """Drop star/NaN/zero-length/out-of-clip words; sort by start (Codex #8)."""
    import math
    out = []
    for w in words:
        s, e = w.get("start"), w.get("end")
        if s is None or e is None:
            continue
        if isinstance(s, float) and math.isnan(s):
            continue
        if isinstance(e, float) and math.isnan(e):
            continue
        if not (0.0 <= s < e <= clip_dur + 1e-6):
            continue
        if w.get("text") == "<star>":
            continue
        out.append(w)
    return sorted(out, key=lambda w: (w["start"], w["end"]))


def classify_boundary(words: list[dict], vad: list[dict], *, raw_dur: float,
                      clip_dur: float, margin_s: float = MARGIN_S) -> BoundaryResult:
    raw_lo = margin_s
    raw_hi = min(margin_s + raw_dur, clip_dur)   # clamp: span may outrun audio
    fallback = BoundaryResult("align_failed", raw_lo, raw_hi, 0.0)
    words = _sane_words(words, clip_dur)
    if not words:
        return fallback
    scores = [w.get("score") for w in words if w.get("score") is not None]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    if mean_score < MIN_MEAN_SCORE:
        return BoundaryResult("align_failed", raw_lo, raw_hi, mean_score)

    a_lo = min(w["start"] for w in words)
    a_hi = max(w["end"] for w in words)
    # plausibility: aligned span wildly shorter/longer than the raw span means
    # the alignment cannot be trusted (Codex #9)
    if (a_hi - a_lo) < 0.3 * (raw_hi - raw_lo) or (a_hi - a_lo) > clip_dur:
        return BoundaryResult("align_failed", raw_lo, raw_hi, mean_score)
    start_off = max(0.0, a_lo - PAD_S)
    end_off = min(clip_dur, a_hi + PAD_S)

    if a_lo < raw_lo - EDGE_TOL_S:
        status = "suspect_cut_start"
    elif a_hi > raw_hi + EDGE_TOL_S:
        status = "suspect_cut_end"
    elif (_speech_overlap(vad, raw_lo, a_lo) >= BLEED_MIN_S
          or _speech_overlap(vad, a_hi, raw_hi) >= BLEED_MIN_S):
        status = "suspect_bleed_in"
    elif max(abs(start_off - raw_lo), abs(end_off - raw_hi)) <= OK_SHIFT_S:
        status = "ok"
    else:
        status = "adjusted"
    return BoundaryResult(status, start_off, end_off, mean_score)
