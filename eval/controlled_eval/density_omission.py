#!/usr/bin/env python3
"""A per-speaker omission rule that can see one lost speaker inside overlap.

Preregistered in `docs/specs/2026-08-16-overlap-speaker-arms-prereg.md` §3, revised
once on Codex review `5851725675b5` before anything was run.

THE PROBLEM. The shipped rule flags an interval of asserted speech >= 1.5 s inside
which our transcript has NO words. Round 2 of `exp-2026-08-16-pyannote-transcription`
showed its structural blind spot: in an interval with active set {A, B}, one recognised
word of A blocks the flag even if the whole of B's turn was lost - precisely the case
the rule is wanted for.

WHY THE OBVIOUS FIX IS WRONG. "Flag when words-per-speaker-second drops below half the
corpus rate" provably cannot fire on that case: two speakers, one transcribed normally
at rate p, gives p*d / 2d = 0.5p exactly, and a strict `< 0.5p` does not fire. Three
speakers with one lost gives 2p/3, further away. The quantity must be a COUNT of
missing speakers, not a density fraction.

THE RULE, hybrid because the two regimes are different problems:

  outside overlap (|S| == 1)   flag iff dur >= 1.5 s and no token of ours lands inside
  inside overlap  (|S| >= 2)   missing(I) = |S| - obs(I) / (rho_single * dur(I))
                               flag iff missing(I) >= 1.0   (inclusive, so the exact
                               one-lost-speaker case fires at the boundary)

`rho_single` is the single-speaker token rate, estimated off single-speaker intervals
only and cross-fitted leave-one-city-out, so no meeting sets its own threshold and the
very omissions being hunted cannot depress it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.exp_speaker_fusion import active_intervals   # noqa: E402

MIN_SPEECH_SEC = 1.5        # wall-clock eligibility, single-speaker regime (unchanged)
MIN_DEL_RUN = 3             # a truth event is a run of >= 3 deletions
MISSING_THRESHOLD = 1.0     # primary; sensitivity 0.75 / 1.25
CALIPER = 0.25              # duration caliper of the matched-random null


# ------------------------------------------------------------------ eligibility
def eligible(reg_segs, rho_single: float) -> list[tuple[float, float, int]]:
    """(start, end, n_speakers) for every interval a flag may be raised on.

    Single-speaker intervals keep the old wall-clock rule. Overlap intervals are
    eligible when ONE lost speaker would account for at least `MIN_DEL_RUN` words -
    the eligibility constant is tied to the ground-truth definition rather than being
    a new free number.
    """
    out = []
    for s, e, sp in active_intervals(reg_segs):
        n, dur = len(sp), e - s
        if n == 1:
            if dur >= MIN_SPEECH_SEC:
                out.append((s, e, 1))
        elif rho_single * dur >= MIN_DEL_RUN:
            out.append((s, e, n))
    return out


def count_in(times: list[float], s: float, e: float) -> int:
    """Half-open [s, e), matching `active_intervals`."""
    return sum(1 for t in times if s <= t < e)


def missing_speakers(n: int, obs: int, dur: float, rho_single: float) -> float:
    if rho_single <= 0 or dur <= 0:
        return 0.0
    return n - obs / (rho_single * dur)


# ------------------------------------------------------------------------ flags
def observed(reg_segs, our_times, rho_single) -> list[tuple[float, float, int, int]]:
    """(start, end, n_speakers, our token count) for every eligible interval.

    Computed once per window; it does not depend on the threshold, so the budget
    calibration of the duration-only comparator can scan thresholds without redoing it.
    """
    return [(s, e, n, count_in(our_times, s, e))
            for s, e, n in eligible(reg_segs, rho_single)]


def flags_from(obs_rows, rho_single, threshold=MISSING_THRESHOLD,
               force_single=False) -> list[tuple[float, float]]:
    """Unmerged flags from precomputed eligible intervals.

    `force_single=True` is the duration-only comparator: it treats every eligible
    interval as if one speaker were active, so the only information it can use is
    duration and our own word count.
    """
    out = []
    for s, e, n, obs in obs_rows:
        if force_single:
            if missing_speakers(1, obs, e - s, rho_single) >= threshold:
                out.append((s, e))
        elif n == 1:
            if obs == 0:
                out.append((s, e))
        elif missing_speakers(n, obs, e - s, rho_single) >= threshold:
            out.append((s, e))
    return out


def raw_flags(reg_segs, our_times, rho_single, threshold=MISSING_THRESHOLD,
              force_single=False) -> list[tuple[float, float]]:
    return flags_from(observed(reg_segs, our_times, rho_single), rho_single,
                      threshold, force_single)


def old_rule_flags(reg_segs, our_times) -> list[tuple[float, float]]:
    """The shipped rule, unchanged: >= 1.5 s of asserted speech with none of our words.

    Applied to every active interval regardless of how many speakers it holds, which
    is what round 2 measured.
    """
    return [(s, e) for s, e, _ in active_intervals(reg_segs)
            if e - s >= MIN_SPEECH_SEC and count_in(our_times, s, e) == 0]


def merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Touching or overlapping flags become one maximal span.

    Without this, one omission split across three adjacent active intervals would count
    as three true positives and inflate precision.
    """
    out: list[list[float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


# ---------------------------------------------------------------- truth + matching
def truth_events(ops: list[str], ref_times: list[float]) -> list[tuple[float, float]]:
    """Maximal runs of >= MIN_DEL_RUN consecutive deletions, timed by their tokens."""
    runs, i = [], 0
    while i < len(ops):
        if ops[i] == "D":
            j = i
            while j < len(ops) and ops[j] == "D":
                j += 1
            if j - i >= MIN_DEL_RUN:
                runs.append((ref_times[i], ref_times[j - 1]))
            i = j
        else:
            i += 1
    return runs


def _ov(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def match(flags: list[tuple[float, float]],
          truths: list[tuple[float, float]]) -> tuple[int, list[int], list[int]]:
    """One-to-one greedy matching by earliest flag start. Each truth is consumed once.

    Returns (n matched, indices of matched flags, indices of matched truths).
    """
    used_t: set[int] = set()
    mf, mt = [], []
    for fi, f in sorted(enumerate(flags), key=lambda kv: kv[1][0]):
        for ti, t in enumerate(truths):
            if ti in used_t or not _ov(f, t):
                continue
            used_t.add(ti)
            mf.append(fi)
            mt.append(ti)
            break
    return len(mf), sorted(mf), sorted(mt)


def calibrate_budget(per_window, target: int, lo=0.0, hi=1.0, steps=401) -> float:
    """Threshold for the duration-only comparator that emits `target` merged flags.

    Scans a fixed grid; ties are broken towards FEWER flags, so the comparator is never
    given a budget advantage. `per_window` yields (obs_rows, rho_single).
    """
    best, best_key = None, None
    for k in range(steps):
        thr = lo + (hi - lo) * k / (steps - 1)
        n = sum(len(merge(flags_from(rows, rho, thr, force_single=True)))
                for rows, rho in per_window)
        key = (abs(n - target), n)
        if best_key is None or key < best_key:
            best, best_key = thr, key
    return best
