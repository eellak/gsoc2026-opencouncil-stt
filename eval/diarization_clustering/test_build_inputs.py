#!/usr/bin/env python3
"""Tests that must pass before any paid pyannote job is submitted.

These are the subset of Astra's list that guard the two failure modes which
would silently corrupt the result: a slicing rule that loses or double-counts
audio, and a sliced arm whose timestamps do not map back to the original
timebase. Run with ``python3 -m pytest -q eval/diarization_clustering``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_inputs import build_transcript, human_utterances, merge_runs, topic_runs  # noqa: E402
from run_arms import (  # noqa: E402
    ensure_succeeded,
    identification_timeline,
    link_by_overlap,
    localise,
    namespace_speakers,
    slice_offsets,
)


def utt(start, end, text="x", tag="T1", subject="S1"):
    return {"start": start, "end": end, "text": text, "tag": tag, "subject": subject}


# --- slicing ---------------------------------------------------------------

def test_topic_runs_group_only_contiguous_same_subject():
    utts = [utt(0, 1, subject="A"), utt(1, 2, subject="A"),
            utt(2, 3, subject="B"), utt(3, 4, subject="A")]
    runs = topic_runs(utts)
    assert [r["subject"] for r in runs] == ["A", "B", "A"]
    assert runs[0]["utterances"] == 2


def test_merge_reaches_minimum_and_covers_the_whole_timeline():
    runs = [{"subject": str(i), "start": i * 60.0, "end": (i + 1) * 60.0,
             "utterances": 1} for i in range(10)]
    slices = merge_runs(runs, min_slice=300.0)
    assert slices[0]["start"] == 0.0
    assert slices[-1]["end"] == 600.0
    # contiguous, no gaps, no overlaps
    for a, b in zip(slices, slices[1:]):
        assert a["end"] == b["start"]
    assert all(s["duration"] >= 300.0 for s in slices)


def test_short_tail_is_absorbed_not_shipped_as_a_runt():
    runs = [{"subject": "A", "start": 0.0, "end": 310.0, "utterances": 1},
            {"subject": "B", "start": 310.0, "end": 350.0, "utterances": 1}]
    slices = merge_runs(runs, min_slice=300.0)
    assert len(slices) == 1
    assert slices[0]["end"] == 350.0


def test_meeting_shorter_than_one_slice_yields_exactly_one_slice():
    runs = [{"subject": "A", "start": 0.0, "end": 100.0, "utterances": 1}]
    slices = merge_runs(runs, min_slice=300.0)
    assert len(slices) == 1  # caller must mark this meeting slicing_is_null


# --- transcript ------------------------------------------------------------

def test_transcript_preserves_every_utterance_and_its_bounds():
    utts = [utt(0, 1.5, "a"), utt(2, 3.25, "b")]
    t = build_transcript(utts)
    got = t["transcription"]["utterances"]
    assert len(got) == 2
    assert [u["start"] for u in got] == [0, 2]
    assert [u["end"] for u in got] == [1.5, 3.25]
    assert t["metadata"]["audio_duration"] == 3.25


def test_human_utterances_are_time_ordered_across_segments():
    meeting = {"transcript": [
        {"speakerTagId": "T2", "utterances": [
            {"startTimestamp": 10, "endTimestamp": 11, "text": "late"}]},
        {"speakerTagId": "T1", "utterances": [
            {"startTimestamp": 1, "endTimestamp": 2, "text": "early"}]},
    ]}
    utts = human_utterances(meeting)
    assert [u["text"] for u in utts] == ["early", "late"]
    assert [u["tag"] for u in utts] == ["T1", "T2"]


# --- sliced arm: the part that silently breaks ----------------------------

def test_slice_timestamps_map_back_to_the_original_timebase():
    slices = [{"index": 0, "start": 0.0, "end": 300.0},
              {"index": 1, "start": 300.0, "end": 600.0}]
    # pyannote returns times relative to the slice it was given
    per_slice = [
        [{"start": 5.0, "end": 9.0, "speaker": "SPEAKER_00"}],
        [{"start": 5.0, "end": 9.0, "speaker": "SPEAKER_00"}],
    ]
    merged = slice_offsets(slices, per_slice)
    assert merged[0]["start"] == 5.0
    assert merged[1]["start"] == 305.0  # NOT 5.0
    assert merged[1]["end"] == 309.0


def test_speakers_are_namespaced_per_slice_so_identity_never_leaks_across():
    slices = [{"index": 0, "start": 0.0, "end": 300.0},
              {"index": 1, "start": 300.0, "end": 600.0}]
    per_slice = [
        [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}],
    ]
    merged = slice_offsets(slices, per_slice)
    speakers = {s["speaker"] for s in merged}
    assert speakers == {"s0:SPEAKER_00", "s1:SPEAKER_00"}, speakers


def test_a0_local_control_partitions_the_baseline_the_same_way():
    """A0-local is what makes A3 readable; it must cut identically and for free."""
    slices = [{"index": 0, "start": 0.0, "end": 300.0},
              {"index": 1, "start": 300.0, "end": 600.0}]
    whole = [
        {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_00"},
        {"start": 400.0, "end": 410.0, "speaker": "SPEAKER_00"},
    ]
    local = localise(whole, slices)
    speakers = {s["speaker"] for s in local}
    # the same voice in two slices becomes two voices, exactly as in A3
    assert speakers == {"s0:SPEAKER_00", "s1:SPEAKER_00"}
    assert len(local) == 2


def test_segment_crossing_a_boundary_is_split_once_and_not_duplicated():
    slices = [{"index": 0, "start": 0.0, "end": 300.0},
              {"index": 1, "start": 300.0, "end": 600.0}]
    whole = [{"start": 290.0, "end": 310.0, "speaker": "SPEAKER_00"}]
    local = localise(whole, slices)
    assert len(local) == 2
    total = sum(s["end"] - s["start"] for s in local)
    assert abs(total - 20.0) < 1e-9  # duration conserved, nothing counted twice


def test_namespace_is_stable_and_reversible():
    assert namespace_speakers([{"start": 0, "end": 1, "speaker": "X"}], 3)[0]["speaker"] == "s3:X"


# --- a failed job must not survive on disk as if it were data ---------------

def test_linking_gives_sliced_segments_one_global_identity():
    """The c3 arm: slice boundaries, whole-meeting identity.

    Two slices that each invented their own name for the same person must come
    out carrying the single global label.
    """
    local = [{"start": 0.0, "end": 5.0, "speaker": "s0:SPEAKER_00"},
             {"start": 310.0, "end": 315.0, "speaker": "s1:SPEAKER_00"}]
    globl = [{"start": 0.0, "end": 600.0, "speaker": "GLOBAL_A"}]
    out = link_by_overlap(local, globl)
    assert {s["speaker"] for s in out} == {"GLOBAL_A"}
    assert len(out) == 2  # boundaries preserved


def test_linking_sums_overlap_per_global_speaker():
    """Same rule as assignment: totals per speaker, not the widest single segment."""
    local = [{"start": 0.0, "end": 10.0, "speaker": "s0:X"}]
    globl = [{"start": 0.0, "end": 3.0, "speaker": "A"},
             {"start": 3.0, "end": 7.0, "speaker": "B"},
             {"start": 7.0, "end": 10.0, "speaker": "A"}]  # A = 6s, B = 4s
    assert link_by_overlap(local, globl)[0]["speaker"] == "A"


def test_linking_uses_max_overlap_not_first_hit():
    local = [{"start": 0.0, "end": 10.0, "speaker": "s0:X"}]
    globl = [{"start": 0.0, "end": 3.0, "speaker": "SHORT"},
             {"start": 3.0, "end": 10.0, "speaker": "LONG"}]
    assert link_by_overlap(local, globl)[0]["speaker"] == "LONG"


def test_linking_keeps_unmatched_segments_instead_of_dropping_them():
    """Dropping them would shrink the denominator and flatter the arm."""
    local = [{"start": 900.0, "end": 905.0, "speaker": "s2:SPEAKER_03"}]
    globl = [{"start": 0.0, "end": 100.0, "speaker": "GLOBAL_A"}]
    out = link_by_overlap(local, globl)
    assert len(out) == 1
    assert out[0]["speaker"] == "s2:SPEAKER_03"


def test_identified_people_keep_one_label_across_slices():
    """The reason voiceprints can rescue slicing: identity survives the cut.

    If this ever namespaces a matched personId, every sliced arm silently turns
    one person into N people again and the measurement is worthless.
    """
    out = {"identification": [
        {"start": 1.0, "end": 2.0, "speaker": "person_A", "diarizationSpeaker": "SPEAKER_00"},
    ]}
    known = {"person_A"}
    a = identification_timeline(out, 0, known)
    b = identification_timeline(out, 7, known)
    assert a[0]["speaker"] == "person_A"
    assert b[0]["speaker"] == "person_A"  # same person, different slice, same label


def test_unmatched_speech_is_namespaced_per_slice():
    """SPEAKER_00 in slice 1 and slice 2 are not known to be the same person."""
    out = {"identification": [
        {"start": 1.0, "end": 2.0, "speaker": None, "match": None,
         "diarizationSpeaker": "SPEAKER_00"},
    ]}
    known = {"person_A"}
    assert identification_timeline(out, 0, known)[0]["speaker"] == "unmatched:s0:SPEAKER_00"
    assert identification_timeline(out, 3, known)[0]["speaker"] == "unmatched:s3:SPEAKER_00"
    # whole-meeting call has no slice, so no slice namespace
    assert identification_timeline(out, None, known)[0]["speaker"] == "unmatched:SPEAKER_00"


def test_identification_falls_back_to_match_field():
    out = {"identification": [
        {"start": 0.0, "end": 1.0, "speaker": None, "match": "person_B",
         "diarizationSpeaker": "SPEAKER_01"},
    ]}
    assert identification_timeline(out, 0, {"person_B"})[0]["speaker"] == "person_B"


def test_empty_identification_yields_empty_timeline():
    assert identification_timeline({}, 0, set()) == []


def test_speaker_label_not_among_our_voiceprints_is_namespaced():
    """pyannote does not return null for unmatched speech.

    Measured on chania: 2 voiceprints sent, 13 distinct labels returned — the
    extra 11 were `SPEAKER_xx`, in the SAME `speaker` field as real personIds.
    Trusting that field made two slices' anonymous speakers look like one person.
    """
    out = {"identification": [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "match": None,
         "diarizationSpeaker": "SPEAKER_00"},
    ]}
    known = {"person_A"}
    assert identification_timeline(out, 0, known)[0]["speaker"] == "unmatched:s0:SPEAKER_00"
    assert identification_timeline(out, 1, known)[0]["speaker"] == "unmatched:s1:SPEAKER_00"
    # and a real personId in the same field is left alone
    out2 = {"identification": [
        {"start": 0.0, "end": 1.0, "speaker": "person_A", "match": "person_A",
         "diarizationSpeaker": "SPEAKER_00"},
    ]}
    assert identification_timeline(out2, 0, known)[0]["speaker"] == "person_A"


def test_succeeded_job_passes_through():
    r = {"status": "succeeded", "output": {"diarization": []}}
    assert ensure_succeeded(r, "tag") is r


def test_failed_or_timed_out_job_is_rejected_even_when_cached():
    """A persisted failure reused on the next run would score a partial timeline
    as a real one; rejection must not depend on the job being fresh."""
    for status in ("failed", "timeout", "canceled", "bad_response", None):
        try:
            ensure_succeeded({"status": status}, "tag")
        except RuntimeError:
            continue
        raise AssertionError(f"status {status!r} was accepted")
