"""What Pool 2 has frozen, and what happens when the world moves under it.

Three things in this module are decisions rather than code, and each of them fails
silently if it is left to be recomputed:

  * the 67 measurement meetings. "The queue minus the training side" is a derivation
    whose right-hand side GROWS: build one more training arm and a meeting quietly
    leaves the evaluation design, in the direction that makes a result easier to get.
  * the eligibility rule. `del >= 1` was written down before a single hypothesis was
    decoded, and the severity grid is reporting rather than selection.
  * the decode provenance. A hypothesis whose model is unknown is not evidence.

There is also a brute-force check of the scheduler, because "largest non-overlapping
set" is the one piece of real algorithm here and a subtle bug in it silently shrinks
the pool.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import cut_pool2_clips as P2  # noqa: E402


# ------------------------------------------------------------------ frozen sets

def test_the_eligibility_rule_is_directional_and_the_grid_is_only_reporting():
    assert P2.MIN_WITNESS_ONLY_WORDS == 1
    assert P2.DISAGREEMENT_GRID[0] == 0.0, (
        "the grid starts at 0.0 because the frozen rule, not a threshold, is what "
        "selects; the row at 0.0 is the pool")
    assert list(P2.DISAGREEMENT_GRID) == sorted(P2.DISAGREEMENT_GRID)


def test_cut_refuses_without_an_explicit_human_decision_about_size(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cut_pool2_clips", "cut"])
    with pytest.raises(SystemExit) as e:
        P2.main()
    assert "i-have-chosen-the-size" in str(e.value)


@pytest.mark.skipif(not P2.MEASUREMENT_SET.exists(),
                    reason="the measurement set has not been frozen on this machine")
def test_the_frozen_measurement_set_matches_its_pinned_hash():
    raw = P2.MEASUREMENT_SET.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == P2.FROZEN_MEASUREMENT_SHA256
    keys = json.loads(raw.decode("utf-8"))["meeting_keys"]
    assert len(keys) == 67, (
        "wayfinder #60 measured 67 untouched meetings holding 449 queue clips; a "
        "different number means the design moved and somebody has to say so")


@pytest.mark.skipif(not P2.MEASUREMENT_SET.exists(),
                    reason="the measurement set has not been frozen on this machine")
def test_the_measurement_set_and_the_training_side_are_disjoint():
    """The whole point. A meeting cannot be training material and evaluation material."""
    assert not (P2.load_measurement_set() & P2.training_side_meetings())


@pytest.mark.skipif(not P2.MEASUREMENT_SET.exists(),
                    reason="the measurement set has not been frozen on this machine")
def test_a_tampered_measurement_set_is_refused(tmp_path):
    keys = json.loads(P2.MEASUREMENT_SET.read_text(encoding="utf-8"))["meeting_keys"]
    p = tmp_path / "measurement.json"
    p.write_text(json.dumps({"meeting_keys": keys[:-1]}), encoding="utf-8")
    with pytest.raises(ValueError):
        P2.load_measurement_set(p, expect_sha=P2.FROZEN_MEASUREMENT_SHA256)


def test_an_empty_measurement_set_is_refused(tmp_path):
    p = tmp_path / "measurement.json"
    body = json.dumps({"meeting_keys": []})
    p.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        P2.load_measurement_set(
            p, expect_sha=hashlib.sha256(body.encode()).hexdigest(), cross_check=False)


def test_the_listening_queue_is_pinned_to_a_generation_not_to_the_current_symlink():
    """`current` is a mutable symlink; the 67 are defined by exact bytes."""
    assert "generations" in str(P2.LISTEN_QUEUE)
    assert len(P2.LISTEN_QUEUE_SHA256) == 64


# ------------------------------------------------------------------ decode contract

def test_the_decode_order_is_frozen_and_independent_of_input_order():
    cands = [{"origin_row_id": f"row{i}"} for i in range(20)]
    a = [c["origin_row_id"] for c in P2.decode_order(cands)]
    b = [c["origin_row_id"] for c in P2.decode_order(list(reversed(cands)))]
    assert a == b
    assert a != [c["origin_row_id"] for c in cands], (
        "an ordering that happens to be the input order would make a pilot a prefix of "
        "whatever sorted first, which for this pool is athens only")


def test_the_decode_config_is_the_one_the_v2_training_wer_run_used():
    """Two decodes of the same adapter under different options are two measurements."""
    from eval.controlled_eval import train_wer_v2  # noqa: F401
    cfg = P2.DECODE_CONFIG
    assert cfg["language"] == "el" and cfg["beam_size"] == 5
    assert cfg["condition_on_previous_text"] is False
    assert cfg["vad_filter"] is False
    assert P2.EXPECT_ARTIFACT == "artifact-ct2-cleanpack-cont-s47"


def test_wav_seconds_is_the_wav_arithmetic_and_not_a_guess(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"\0" * (44 + 16000 * 2 * 3))     # 3 s of 16 kHz mono 16-bit
    assert P2.wav_seconds(p) == pytest.approx(3.0)
    assert P2.wav_seconds(tmp_path / "missing.wav") is None


def test_the_decode_cost_counts_the_wavs_it_cannot_find(tmp_path):
    good = tmp_path / "a.wav"
    good.write_bytes(b"\0" * (44 + 16000 * 2 * 10))
    cost = P2.decode_cost([{"gap_verify_clip": str(good)},
                           {"gap_verify_clip": str(tmp_path / "nope.wav")},
                           {"gap_verify_clip": None}])
    assert cost["spans"] == 3
    assert cost["cached_wav_missing"] == 2
    assert cost["audio_hours"] == round(10 / 3600, 3)


# ------------------------------------------------------------------ the scheduler

def _brute_force_max(intervals):
    """The largest set of pairwise non-overlapping intervals, by exhaustion."""
    best = 0
    n = len(intervals)
    for r in range(n, -1, -1):
        for combo in itertools.combinations(range(n), r):
            ok = all(not (intervals[a][0] < intervals[b][1]
                          and intervals[b][0] < intervals[a][1])
                     for a, b in itertools.combinations(combo, 2))
            if ok:
                return r
    return best


def test_the_scheduler_finds_the_largest_set_on_random_instances():
    rng = random.Random(20260906)
    for _ in range(60):
        n = rng.randint(1, 8)
        spans = []
        for i in range(n):
            s = rng.randrange(0, 40) * 1000
            spans.append((s, s + rng.choice([5000, 8000, 12000])))
        rows = [{"origin_row_id": f"r{i}",
                 "geometry": P2.Geometry(start_ms=s, end_ms=e, pad_left_ms=0,
                                         pad_right_ms=0, reuse_gap_verify=True),
                 "disagreement": {"rate": 0.0}}
                for i, (s, e) in enumerate(spans)]
        chosen, lost = P2._schedule(rows)
        assert len(chosen) == _brute_force_max(spans)
        assert len(chosen) + lost == n
        picked = sorted((c["geometry"].start_ms, c["geometry"].end_ms) for c in chosen)
        for a, b in zip(picked, picked[1:]):
            assert a[1] <= b[0], "the scheduler returned an overlapping pair"
