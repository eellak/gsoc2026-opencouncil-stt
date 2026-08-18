#!/usr/bin/env python3
"""Attach a speaker state to a time interval, without inventing certainty.

The rule is maximum TEMPORAL OVERLAP, not the midpoint. A midpoint rule silently
hands a word that straddles a handover entirely to whichever speaker happens to own
its centre, which is exactly the case this page exists to look at.

Six states, and five of them are not a name:

  named          one turn dominates the interval
  ambiguous      two turns each carry a real share of it: a straddling word
  overlap        the regular (non-exclusive) diarization has >1 turn here, so the
                 exclusive lane's single answer is a display convenience
  non_speech     no turn touches the interval at all
  no_diarization there is no diarization for this stretch
  unresolved     the interval itself is not trustworthy enough to ask the question
                 (set by `timing.py`, passed through here)

Forcing every word onto a named speaker would destroy the uncertainty the instrument
is for, so nothing here ever falls back to "closest speaker".
"""
from __future__ import annotations

from dataclasses import dataclass, field

# A second turn holding at least this share of the interval makes it ambiguous.
AMBIGUOUS_SHARE = 0.25


@dataclass
class SpeakerCall:
    state: str
    speaker: str | None = None
    overlap_fraction: float = 0.0          # share of the interval the winner covers
    runner_up: str | None = None
    runner_up_fraction: float = 0.0
    multiplicity: int = 0                  # regular-diarization turns touching it
    shares: dict = field(default_factory=dict)


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign(lo: float, hi: float, exclusive: list[dict], regular: list[dict],
           unresolved: bool = False) -> SpeakerCall:
    """Speaker state for the interval [lo, hi)."""
    if unresolved:
        return SpeakerCall(state="unresolved")
    if not exclusive:
        return SpeakerCall(state="no_diarization")
    width = hi - lo
    if width <= 0:
        # A zero-width interval cannot be apportioned; ask about the instant instead.
        width = 1e-6
        hi = lo + width
    shares: dict[str, float] = {}
    for s in exclusive:
        ov = _overlap(lo, hi, s["start"], s["end"])
        if ov > 0:
            shares[s["speaker"]] = shares.get(s["speaker"], 0.0) + ov
    mult = sum(1 for s in regular if s["end"] > lo and s["start"] < hi)
    if not shares:
        return SpeakerCall(state="non_speech", multiplicity=mult, shares={})
    ranked = sorted(shares.items(), key=lambda kv: -kv[1])
    top, top_ov = ranked[0]
    top_f = top_ov / width
    second, second_f = (ranked[1][0], ranked[1][1] / width) if len(ranked) > 1 \
        else (None, 0.0)
    if second_f >= AMBIGUOUS_SHARE:
        state = "ambiguous"
    elif mult > 1:
        state = "overlap"
    else:
        state = "named"
    return SpeakerCall(state=state, speaker=top, overlap_fraction=round(top_f, 4),
                       runner_up=second, runner_up_fraction=round(second_f, 4),
                       multiplicity=mult,
                       shares={k: round(v / width, 4) for k, v in ranked})


def handovers(exclusive: list[dict]) -> list[float]:
    """Times where the exclusive lane changes speaker, sorted."""
    out = []
    prev = None
    for s in sorted(exclusive, key=lambda x: x["start"]):
        if prev is not None and s["speaker"] != prev["speaker"]:
            out.append((prev["end"] + s["start"]) / 2.0)
        prev = s
    return out


def crosses_handover(lo: float, hi: float, exclusive: list[dict]) -> bool:
    return any(lo < h < hi for h in handovers(exclusive))


def silence_gaps(exclusive: list[dict], min_gap: float = 1.0) -> list[tuple]:
    """Stretches with no exclusive turn at all, at least `min_gap` long."""
    out, prev_end = [], None
    for s in sorted(exclusive, key=lambda x: x["start"]):
        if prev_end is not None and s["start"] - prev_end >= min_gap:
            out.append((prev_end, s["start"]))
        prev_end = max(prev_end or s["end"], s["end"])
    return out


def crosses_silence(lo: float, hi: float, exclusive: list[dict],
                    min_gap: float = 1.0) -> bool:
    return any(_overlap(lo, hi, g0, g1) > 0 for g0, g1 in
               silence_gaps(exclusive, min_gap))
