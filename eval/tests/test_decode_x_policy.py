"""The deployable form of arm X must be the same policy the experiment measured."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from serve import decode_x as DX
from eval import seam_repair as SR


def test_the_policy_fits_one_whisper_window():
    DX._assert_fits_one_window()
    total = (DX.POLICY["segmentation"]["max_core_seconds"]
             + DX.POLICY["context"]["lookback_seconds"]
             + DX.POLICY["context"]["lookahead_seconds"])
    assert total == pytest.approx(29.5)


def test_the_policy_matches_the_experiment_constants():
    """A drift between the served policy and the measured arm is a silent lie."""
    seg, ctx = DX.POLICY["segmentation"], DX.POLICY["context"]
    assert seg["max_core_seconds"] == SR.X_MAX_CHUNK_SECONDS
    assert seg["valley_slack_seconds"] == SR.X_SLACK_SECONDS
    assert seg["valley_prob_threshold"] == SR.VALLEY_PROB_THRESHOLD
    assert seg["valley_min_frames"] == SR.MIN_VALLEY_FRAMES
    assert ctx["lookback_seconds"] == SR.X_LOOKBACK_SECONDS
    assert ctx["lookahead_seconds"] == SR.X_LOOKAHEAD_SECONDS
    assert ctx["prompt_max_tokens"] == SR.PROMPT_MAX_TOKENS
    assert DX.POLICY["decode"] == dict(SR.CONTROL, word_timestamps=True)
    assert DX.POLICY["ownership"] == SR.ARM_MANIFESTS["X"]["merge_x"]


def test_the_policy_hash_is_stable_and_changes_with_the_policy():
    first = DX.policy_sha256()
    assert first == DX.policy_sha256()
    original = DX.POLICY["context"]["lookahead_seconds"]
    DX.POLICY["context"]["lookahead_seconds"] = original + 0.5
    try:
        assert DX.policy_sha256() != first
    finally:
        DX.POLICY["context"]["lookahead_seconds"] = original
    assert DX.policy_sha256() == first


def test_transcript_record_carries_its_provenance():
    t = DX.Transcript(text="γεια", words=[DX.Word(" γεια", 0.0, 0.4)],
                      boundaries=[DX.Boundary(27.0, SR.PROBABILITY_VALLEY)],
                      audio_seconds=30.0, policy_sha256=DX.policy_sha256(),
                      counters={"blind_cuts": 0})
    record = t.as_dict()
    assert record["policy_id"] == DX.POLICY_ID
    assert record["policy_sha256"] == DX.policy_sha256()
    assert record["boundaries"][0]["kind"] == SR.PROBABILITY_VALLEY


def test_the_served_policy_reproduces_the_measured_arm_on_a_cached_window():
    """The strongest available check without a GPU: same audio, same text."""
    cache = Path(
        "/home/harold/.cache/oc-public/seam-repair-2026-08/pilot-X.json")
    if not cache.exists():
        pytest.skip("the arm X pilot cache is not on this machine")
    raw = cache.read_bytes()
    if not raw.strip(b"\x00"):
        # Measured 2026-08-28: the file on this machine is 532 KB of NUL bytes, a
        # truncated write from the pilot run. Recorded as a caveat and a next_action on
        # exp-2026-08-20-seam-repair; it needs a re-decode, not a test edit.
        pytest.skip("the arm X pilot cache is zero-filled; see exp-2026-08-20-seam-repair")
    state = json.loads(raw.decode("utf-8"))
    record = next(iter(state["windows"].values()))
    # the policy's segmentation must reproduce the arm's, boundary for boundary
    import sys
    sys.path.insert(0, "/home/harold/opencouncil-fine-tuning/notebooks")
    import decode_ablation as DA
    from eval import chunking_decode as CD

    window_id = next(iter(state["windows"]))
    wav = DA.sc() / "bench_windows" / f"{window_id}.wav"
    if not wav.exists():
        pytest.skip("window audio is not on this machine")
    audio = CD._load_audio(wav)
    duration = len(audio) / SR.SAMPLE_RATE
    silences, _, _ = CD.detect_silences(audio, duration, None)
    probs = SR.speech_probabilities(audio)
    energies = SR.frame_energies(audio, probs.shape[0])
    pieces, _ = SR.split_with_valley_repair(
        duration, silences, probs, energies,
        max_chunk=DX.POLICY["segmentation"]["max_core_seconds"],
        slack=DX.POLICY["segmentation"]["valley_slack_seconds"])
    cached = [(round(p["start_sec"], 3), round(p["end_sec"], 3), p["end_kind"])
              for p in record["pieces"]]
    served = [(round(p.start, 3), round(p.end, 3), p.end_kind) for p in pieces]
    assert served == cached
