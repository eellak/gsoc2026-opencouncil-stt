#!/usr/bin/env python3
"""Faithful port of opencouncil-tasks' speaker-assignment logic.

Source: `src/lib/DiarizationManager.ts` + `src/tasks/applyDiarization.ts` at commit
5ff16a3c20968d6a5610d3584322b9a0059ad482 (pinned by the preregistration).

Faithful means bug-for-bug, because the point is to replay what production would
actually do with each timeline, not what a better algorithm would do:

- the fast path fires only when *exactly one* diarization segment touches the
  utterance envelope, and it returns drift 0 without looking at words at all;
- `findClosestDiarizationForSpeaker` is `Array.prototype.reduce` with **no initial
  value**, so the first segment of that speaker seeds the accumulator and iteration
  starts at index 1; a segment that fully contains the word wins immediately but can
  still be displaced by a later, closer, non-containing segment;
- ties in the final `reduce` keep the **first** candidate, and candidate order is JS
  `Set` insertion order — word order, then segment array order;
- `maxDriftCost` is `+Infinity` in the shipped config, so the drift cap never fires;
- returning `None` means `applyDiarization` drops the utterance ("SKIPPING").

Parity against the real TypeScript is enforced by `test_oc_merge_port.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

INF = float("inf")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Result:
    speaker: str
    drift: float
    branch: str  # "direct" | "single" | "guess"


def _segments_containing(diarization: list[Segment], start: float, end: float):
    return [d for d in diarization if d.start <= start and d.end >= end]


def _closest_for_speaker(diarization: list[Segment], speaker: str,
                         w_start: float, w_end: float) -> Segment:
    """`reduce` with no initial value, exactly as the TypeScript has it."""
    segs = [d for d in diarization if d.speaker == speaker]
    closest = segs[0]
    for current in segs[1:]:
        if current.start <= w_start and current.end >= w_end:
            closest = current
            continue
        closest_diff = abs(closest.start - w_start) + abs(closest.end - w_end)
        current_diff = abs(current.start - w_start) + abs(current.end - w_end)
        if current_diff < closest_diff:
            closest = current
    return closest


def _speakers_with_drift(diarization: list[Segment],
                         words: list[tuple[float, float]]) -> list[tuple[str, float]]:
    full_coverage: dict[str, None] = {}  # dict = insertion-ordered set, like JS Set
    for w_start, w_end in words:
        for d in _segments_containing(diarization, w_start, w_end):
            full_coverage.setdefault(d.speaker, None)

    out = []
    for speaker in full_coverage:
        total = 0.0
        for w_start, w_end in words:
            sd = _closest_for_speaker(diarization, speaker, w_start, w_end)
            if sd.start <= w_start and sd.end >= w_end:
                continue
            total += abs(sd.start - w_start) ** 2 + abs(sd.end - w_end) ** 2
        out.append((speaker, math.sqrt(total)))
    return out


def find_best_speaker(diarization: list[Segment], u_start: float, u_end: float,
                      words: list[tuple[float, float]],
                      max_drift_cost: float = INF) -> Result | None:
    overlapping = [d for d in diarization
                   if (d.start <= u_end and d.start >= u_start)
                   or (d.end <= u_end and d.end >= u_start)
                   or (d.start <= u_start and d.end >= u_end)]

    if len(overlapping) == 1:
        return Result(overlapping[0].speaker, 0.0, "direct")

    cands = _speakers_with_drift(diarization, words)
    if not cands:
        return None
    if len(cands) == 1:
        return Result(cands[0][0], cands[0][1], "single")

    best = cands[0]
    for cur in cands[1:]:
        if cur[1] < best[1]:
            best = cur
    if best[1] > max_drift_cost:
        return None
    return Result(best[0], best[1], "guess")


def replay(diarization: list[Segment], utterances: list[dict]) -> list[dict]:
    """`applyDiarization`: assign or drop each utterance, in order."""
    out = []
    for u in utterances:
        words = [(float(w["start"]), float(w["end"])) for w in u["words"]]
        r = find_best_speaker(diarization, float(u["start"]), float(u["end"]), words)
        out.append({"start": u["start"], "end": u["end"],
                    "speaker": None if r is None else r.speaker,
                    "drift": None if r is None else r.drift,
                    "branch": "drop" if r is None else r.branch})
    return out


def segments(raw) -> list[Segment]:
    return [Segment(float(d["start"]), float(d["end"]), d["speaker"]) for d in raw]
