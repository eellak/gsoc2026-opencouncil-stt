"""Move a cut to the quietest nearby moment, so it stops clipping syllables.

The first human check of spliced packs came back 10/12 unsure on boundaries: cutting on a
database timestamp lands mid-syllable often enough to hear. Diarization cannot place a
word edge — it only knows who is speaking — so the edge is found acoustically instead:
inside a small search window around the nominal cut, take the frame with the lowest RMS.

Two safeguards. The search never crosses into another speaker's turn (the caller passes
bounds from `pad_bounds`), and when nothing in the window is actually quiet the cut moves
outward to the safest end rather than staying tight: extra silence costs nothing, a
half-eaten word costs a training label.
"""
from __future__ import annotations

import numpy as np

FRAME_SEC = 0.01
QUIET_RATIO = 0.25          # "quiet" = a quarter of the window's loudest frame


def rms_frames(samples: np.ndarray, sr: int, frame_sec: float = FRAME_SEC) -> np.ndarray:
    size = max(1, int(round(sr * frame_sec)))
    usable = len(samples) - len(samples) % size
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    block = samples[:usable].reshape(-1, size).astype(np.float32)
    return np.sqrt((block ** 2).mean(axis=1))


def snap(samples: np.ndarray, sr: int, target_sec: float, lo_sec: float, hi_sec: float,
         side: str, frame_sec: float = FRAME_SEC) -> float:
    """Quietest instant in [lo_sec, hi_sec], as a time in the same clock as target_sec.

    `side` is "start" or "end": it decides which way to fall back when the whole window is
    loud — outward, never inward.
    """
    if hi_sec <= lo_sec:
        return target_sec
    lo_i, hi_i = int(round(lo_sec * sr)), int(round(hi_sec * sr))
    window = samples[max(0, lo_i):max(0, hi_i)]
    frames = rms_frames(window, sr, frame_sec)
    if frames.size == 0:
        return target_sec
    threshold = QUIET_RATIO * float(frames.max())
    quiet = frames <= threshold
    if not quiet.any():
        return lo_sec if side == "start" else hi_sec
    # Centre of the LONGEST quiet run, not the first quiet frame: a cut in the middle of
    # a pause survives a few milliseconds of timing error in either direction.
    best_start = best_len = run_start = run_len = 0
    for i, is_quiet in enumerate(quiet):
        if is_quiet:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    centre = lo_sec + (best_start + best_len / 2) * frame_sec
    return round(min(max(centre, lo_sec), hi_sec), 3)
