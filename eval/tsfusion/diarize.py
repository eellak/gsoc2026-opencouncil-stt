#!/usr/bin/env python3
"""One pyannote precision-2 call over a PADDED span, cropped to the page.

Two things this module is careful about.

PADDING. A diarizer given exactly the 299 s of interest has no context for the turn
that starts before it and the turn that ends after it, and its speaker labels at the
edges are the least reliable it produces. So it is fed `PAD` seconds either side and
the turns are cropped afterwards. The padding is never displayed: it exists to make
the central 299 s better, not to be looked at.

EXCLUSIVE. `exclusive=true` returns two segmentations. `diarization` is the honest
one: overlapping turns, so a moment can have two speakers. `exclusiveDiarization`
assigns every moment to at most one speaker, which is what a single display lane
needs. During overlap that assignment is a DISPLAY CONVENIENCE and not evidence that
one person spoke; `multiplicity_at` exists so the page can say so.

MONEY. pyannote bills per audio-hour (EUR 0.112/h as recorded by
`exclusive_cost_check.py`). The padded span is 479 s, about EUR 0.0149. A 30 s smoke
call is about EUR 0.0009. Both numbers are printed before anything is spent.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "controlled_eval"))

from exclusive_diar_api import api_key, run_one          # noqa: E402

EUR_PER_HOUR = 0.112
USD_PER_EUR = 1.10
PAD = 90.0                       # seconds of context each side, not displayed


def cost_eur(seconds: float) -> float:
    return seconds / 3600.0 * EUR_PER_HOUR


def cut(source: Path, out: Path, start: float, duration: float) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-accurate_seek",
                    "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)],
                   check=True)
    return out


def turns(job: dict, key: str = "diarization") -> list[dict]:
    out = job.get("output", {}) or {}
    return list(out.get(key) or [])


def shift_and_crop(segs: list[dict], t_offset: float, lo: float, hi: float) -> list[dict]:
    """Move turns from clip time into absolute time, then crop to [lo, hi).

    A turn that only partly overlaps the page is KEPT and truncated, with
    `clipped_start` / `clipped_end` recording that its real extent runs past the edge.
    Dropping it would invent silence at the boundary.
    """
    out = []
    for s in segs:
        a, b = s["start"] + t_offset, s["end"] + t_offset
        if b <= lo or a >= hi:
            continue
        out.append({
            "speaker": s.get("speaker"),
            "start": max(a, lo), "end": min(b, hi),
            "clipped_start": a < lo, "clipped_end": b > hi,
            "raw_start": a, "raw_end": b,
        })
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def multiplicity_at(segs: list[dict], lo: float, hi: float) -> int:
    """How many REGULAR-diarization turns touch [lo, hi). >1 means overlap."""
    return sum(1 for s in segs if s["end"] > lo and s["start"] < hi)


def run(clip: Path, name: str, out_json: Path) -> dict:
    key = api_key()
    job = run_one(clip, key, name, exclusive=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(job, ensure_ascii=False, indent=1))
    return job
