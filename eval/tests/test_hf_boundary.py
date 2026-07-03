"""Tests for boundary classification (pure geometry, no audio/models)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.hf_export.boundary import classify_boundary

MARGIN = 1.0


def _w(start, end, score=0.9):
    return {"text": "λέξη", "start": start, "end": end, "score": score}


def test_clean_clip_is_ok():
    # raw span [1.0, 4.0] in extended clip; words snugly inside; VAD agrees
    words = [_w(1.05, 2.0), _w(2.1, 3.9)]
    vad = [{"start": 1.0, "end": 3.95}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "ok"
    assert r.start_off < 1.05 and r.end_off > 3.9  # padded outward


def test_cut_start_detected():
    # aligner finds the first word starting well BEFORE the raw start
    words = [_w(0.55, 1.4), _w(1.5, 3.9)]
    vad = [{"start": 0.5, "end": 3.95}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "suspect_cut_start"


def test_cut_end_detected():
    words = [_w(1.05, 2.0), _w(2.1, 4.6)]  # last word ends past raw end 4.0
    vad = [{"start": 1.0, "end": 4.7}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "suspect_cut_end"


def test_bleed_in_detected():
    # words occupy [2.0, 3.9] but VAD hears speech from 1.0 — someone else
    # talks inside the raw span before the label starts
    words = [_w(2.0, 3.0), _w(3.1, 3.9)]
    vad = [{"start": 1.0, "end": 3.95}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "suspect_bleed_in"


def test_no_words_is_align_failed():
    r = classify_boundary([], [{"start": 1.0, "end": 4.0}],
                          raw_dur=3.0, clip_dur=5.0)
    assert r.status == "align_failed"


def test_low_score_is_align_failed():
    words = [_w(1.05, 3.9, score=0.05)]
    r = classify_boundary(words, [{"start": 1.0, "end": 3.95}],
                          raw_dur=3.0, clip_dur=5.0)
    assert r.status == "align_failed"


def test_large_shift_is_adjusted():
    # aligned span sits deep inside the raw span -> big inward shift, no flags
    words = [_w(1.9, 2.5), _w(2.6, 3.0)]
    vad = [{"start": 1.9, "end": 3.0}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "adjusted"


def test_invalid_and_unsorted_words_are_sanitized():
    # unsorted + zero-length + NaN words must not corrupt the span
    words = [_w(3.0, 3.9), _w(2.0, 2.0), _w(float("nan"), 1.5), _w(1.05, 1.9)]
    vad = [{"start": 1.0, "end": 3.95}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "ok"          # effective span = [1.05, 3.9]


def test_raw_span_truncated_by_clip_end_is_clamped():
    # raw span extends past the decoded audio: raw_hi clamps to clip_dur
    words = [_w(1.05, 3.4)]
    vad = [{"start": 1.0, "end": 3.5}]
    r = classify_boundary(words, vad, raw_dur=4.0, clip_dur=3.5)
    assert r.status in ("ok", "adjusted")  # must not flag a phantom cut_end
