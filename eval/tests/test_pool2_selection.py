"""Tests for the Pool 2 cutter (spec section 4.1, pool 2).

Pool 2 selects training clips where our adapter and the Soniox witness disagree, on
meetings that are already council training material. The three things that can go
irreversibly wrong are all meeting-level, and each has its own test here:

  * a clip from an excluded meeting is training on the test set;
  * a clip from a MEASUREMENT meeting makes one meeting both training material and
    evaluation material, which is exactly the defect the 2026-09-06 redesign removed;
  * a clip that overlaps an existing training or evaluation clip supersedes a label
    that already exists.

The fourth is provenance: a hypothesis whose producing model is unknown, or that came
from a different adapter, must not enter the pool.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import cut_pool2_clips as P2  # noqa: E402
from eval.controlled_eval.exclusion_manifest import Manifest  # noqa: E402


def manifest(meetings=(), cities=()) -> Manifest:
    return Manifest(meeting_keys=frozenset(meetings), city_keys=frozenset(cities))


class FakeTranscripts:
    """The live utterance spans, as `TranscriptSpans` would return them.

    `spans` maps `(city, meeting, utterance_id) -> (start, end)`; anything absent has
    moved or vanished since Soniox ran and must fail closed.
    """

    def __init__(self, spans=None):
        self.spans = spans or {}

    def span(self, city, meeting, rid):
        return self.spans.get((city, meeting, rid))


def unmoved(cands):
    """Transcript spans that agree with every candidate, i.e. nothing was edited."""
    return FakeTranscripts({(c["city"], c["meeting_id"], c["origin_row_id"]):
                            (c["row_start"], c["row_start"] + c["row_dur"])
                            for c in cands})


def witness_row(rid, city, meeting, start, end, text="ένα δύο τρία τέσσερα",
                tier="VERIFIED", url="https://x/a.mp3"):
    return {
        "id": rid,
        "witness": {"id": rid, "soniox_text": text, "tier": tier, "error": None,
                    "clip_padded": None},
        "api": {"city_id": city, "meeting_id": meeting, "start": start, "end": end,
                "audio_url": url},
    }


# ------------------------------------------------------------------ geometry

def test_geometry_matches_pool1_exactly():
    """The two pools must live in one coordinate system or the overlap rule is a lie."""
    from eval.controlled_eval import cut_listening_clips as P1
    assert (P2.PAD_MS, P2.MIN_CLIP_MS, P2.MAX_CLIP_MS) == \
           (P1.PAD_MS, P1.MIN_CLIP_MS, P1.MAX_CLIP_MS)
    for start, dur in [(0.0, 0.14), (100.0, 3.0), (7.0, 9.0), (12.0, 14.999),
                       (0.5, 2.0), (3.0, 15.0)]:
        a, b = P2.clip_geometry(start, dur), P1.clip_geometry(start, dur)
        assert (a is None) == (b is None)
        if a is not None:
            assert (a.start_ms, a.end_ms) == (b.start_ms, b.end_ms)


def test_a_row_longer_than_the_ceiling_is_dropped_not_truncated():
    assert P2.clip_geometry(10.0, 15.001) is None


def test_the_pad_budget_moves_to_the_right_at_the_start_of_a_recording():
    g = P2.clip_geometry(0.0, 0.14)
    assert g.start_ms == 0
    assert g.end_ms - g.start_ms >= P2.MIN_CLIP_MS
    assert not g.reuse_gap_verify      # only a symmetric 3 s pad may reuse the cache


# ------------------------------------------------------------------ disagreement

def test_disagreement_separates_the_two_directions():
    """Words the witness has and we do not are not the same finding as the reverse."""
    d = P2.disagreement("ένα δύο", "ένα δύο τρία τέσσερα")
    assert d["del"] == 2 and d["ins"] == 0
    assert d["del_rate"] == pytest.approx(0.5)
    d2 = P2.disagreement("ένα δύο τρία τέσσερα πέντε", "ένα δύο τρία τέσσερα")
    assert d2["ins"] == 1 and d2["del"] == 0
    assert d2["del_rate"] == 0.0
    # ...but both raise the overall rate, which is why the rate alone cannot select.
    assert d["rate"] > 0 and d2["rate"] > 0


def test_identical_text_is_zero_disagreement():
    d = P2.disagreement("ένα δύο τρία", "ένα δύο τρία")
    assert d["rate"] == 0.0 and d["del"] == d["ins"] == d["sub"] == 0


def test_an_empty_witness_yields_no_rate_rather_than_zero():
    d = P2.disagreement("ένα δύο", "")
    assert d["rate"] is None and d["n_witness_tokens"] == 0


# ------------------------------------------------------------------ meeting sets

def test_measurement_meetings_are_the_queue_minus_the_training_side(tmp_path):
    q = tmp_path / "queue.json"
    q.write_text(json.dumps({"clips": [
        {"city": "athens", "meeting_id": "a_2026"},
        {"city": "athens", "meeting_id": "b_2026"},
        {"city": "chania", "meeting_id": "a_2026"},   # same slug, different city
    ]}), encoding="utf-8")
    got = P2.derive_measurement_meetings(q, frozenset({("athens", "a_2026")}))
    assert got == frozenset({("athens", "b_2026"), ("chania", "a_2026")})


def test_meeting_id_alone_is_not_the_key(tmp_path):
    """`a_2026` in athens being training material says nothing about `a_2026` in chania."""
    q = tmp_path / "queue.json"
    q.write_text(json.dumps({"clips": [{"city": "chania", "meeting_id": "a_2026"}]}),
                 encoding="utf-8")
    got = P2.derive_measurement_meetings(q, frozenset({("athens", "a_2026")}))
    assert ("chania", "a_2026") in got


def test_an_empty_queue_is_an_error_not_an_empty_measurement_set(tmp_path):
    q = tmp_path / "queue.json"
    q.write_text(json.dumps({"clips": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        P2.derive_measurement_meetings(q, frozenset())


# ------------------------------------------------------------------ candidates

TRAINING_SIDE = frozenset({("athens", "a_2026"), ("chania", "b_2026"),
                           ("athens", "gone_2026")})
MEASUREMENT = frozenset({("athens", "m_2026")})


def test_every_meeting_level_filter_fires_and_is_counted():
    rows = [
        witness_row("keep", "athens", "a_2026", 100.0, 104.0),
        witness_row("excl", "athens", "gone_2026", 100.0, 104.0),
        witness_row("meas", "athens", "m_2026", 100.0, 104.0),
        witness_row("offside", "athens", "z_2026", 100.0, 104.0),
        witness_row("nowitness", "athens", "a_2026", 200.0, 204.0, text="  "),
    ]
    rows.append({"id": "noapi", "witness": rows[0]["witness"], "api": None})
    cands, drops, _ = P2.candidates(
        rows, manifest([("athens", "gone_2026")]), TRAINING_SIDE, MEASUREMENT, {})
    assert [c["origin_row_id"] for c in cands] == ["keep"]
    assert drops["excluded_meeting"] == 1
    assert drops["measurement_meeting"] == 1
    assert drops["not_training_side"] == 1
    assert drops["no_witness_text"] == 1
    assert drops["no_api_provenance"] == 1


def test_a_whole_city_exclusion_removes_every_meeting_in_it():
    """The five confirmation cities are city-level, which is what kills cutclips.py."""
    rows = [witness_row("x", "argos", "a_2026", 10.0, 14.0)]
    cands, drops, _ = P2.candidates(
        rows, manifest(cities=["argos"]), frozenset({("argos", "a_2026")}),
        frozenset(), {})
    assert cands == [] and drops["excluded_meeting"] == 1


# ------------------------------------------------------------------ provenance

def test_an_adapter_hypothesis_without_an_artifact_is_refused(tmp_path):
    (tmp_path / "row1.json").write_text(json.dumps({"text": "ένα"}), encoding="utf-8")
    with pytest.raises(P2.ProvenanceError):
        P2.load_adapter_hyps(tmp_path)


def test_hypotheses_from_two_adapters_are_never_pooled(tmp_path):
    (tmp_path / "row1.json").write_text(
        json.dumps({"text": "ένα", "artifact": P2.EXPECT_ARTIFACT}), encoding="utf-8")
    (tmp_path / "row2.json").write_text(
        json.dumps({"text": "δύο", "artifact": "artifact-ct2-fixed"}), encoding="utf-8")
    with pytest.raises(P2.ProvenanceError):
        P2.load_adapter_hyps(tmp_path)


def test_a_missing_hypothesis_directory_is_empty_not_an_error(tmp_path):
    assert P2.load_adapter_hyps(tmp_path / "nope") == {}


def test_the_witness_is_labelled_with_the_served_model_not_the_string_we_sent():
    assert P2.WITNESS == "soniox-stt-rt-v5"


# ------------------------------------------------------------------ witness span

def test_a_span_edited_since_soniox_ran_fails_the_provenance_check():
    """The comparison is against the LIVE transcript, not against the same api record.

    Pool 2 takes its span from `gap-verify/api/<id>.json`, so comparing that file to
    itself would pass every row and detect nothing. The third record -- the utterance as
    OpenCouncil holds it now -- is what can have moved.
    """
    cand = {"audio_url": "https://x/a.mp3", "row_start": 100.0, "row_dur": 4.0}
    assert P2.witness_matches(cand, (100.0, 104.0))
    assert P2.witness_matches(cand, (100.005, 104.005))    # 5 ms of serialization slack
    assert not P2.witness_matches(cand, (113.25, 117.25))  # same duration, moved 13.25 s
    assert not P2.witness_matches(cand, (100.0, 103.0))
    assert not P2.witness_matches(cand, None)              # utterance is gone


def test_a_vanished_utterance_is_dropped_rather_than_assumed_unchanged():
    c = scored("r", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    rows, drops, _ = P2.select([c], manifest(), frozenset(), {}, FakeTranscripts())
    assert rows == [] and drops["witness_span_mismatch"] == 1


# ------------------------------------------------------------------ selection

def scored(rid, city, meeting, start, dur, adapter, witness, url="https://x/a.mp3"):
    return {
        "origin_row_id": rid, "city": city, "meeting_id": meeting,
        "meeting_date": None, "audio_url": url,
        "row_start": start, "row_dur": dur, "tier": "VERIFIED",
        "witness_text": witness, "adapter_text": adapter, "gap_verify_clip": None,
        "geometry": P2.clip_geometry(start, dur),
    }


def api_for(cands):
    """Kept as the name every test reads; now returns the live-transcript stand-in."""
    return unmoved(cands)


def test_a_span_with_no_adapter_hypothesis_is_counted_not_scored():
    c = scored("r", "athens", "a_2026", 100.0, 4.0, None, "ένα δύο τρία")
    rows, drops, _ = P2.select([c], manifest(), frozenset(), {}, api_for([c]))
    assert rows == [] and drops["no_adapter_hypothesis"] == 1


def test_the_frozen_rule_is_directional_and_needs_no_threshold():
    """`del >= 1`: the witness has words we do not. An adapter-only insertion does not
    qualify, and neither does an identical pair."""
    lost = scored("lost", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    added = scored("added", "athens", "a_2026", 300.0, 4.0,
                   "ένα δύο τρία τέσσερα πέντε", "ένα δύο τρία τέσσερα")
    same = scored("same", "athens", "a_2026", 500.0, 4.0,
                  "ένα δύο τρία τέσσερα", "ένα δύο τρία τέσσερα")
    cands = [lost, added, same]
    got, drops, _ = P2.select(cands, manifest(), frozenset(), {}, api_for(cands))
    assert [r["origin_row_id"] for r in got] == ["lost"]
    assert drops["no_witness_only_words"] == 2


def test_the_severity_threshold_is_an_extra_filter_never_a_substitute():
    hi = scored("hi", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    mild = scored("mild", "athens", "a_2026", 300.0, 4.0,
                  "ένα δύο τρία", "ένα δύο τρία τέσσερα")
    cands = [hi, mild]
    both, _, _ = P2.select(cands, manifest(), frozenset(), {}, api_for(cands))
    assert len(both) == 2
    strict, drops, _ = P2.select(cands, manifest(), frozenset(), {}, api_for(cands),
                                 min_disagreement=0.5)
    assert [r["origin_row_id"] for r in strict] == ["hi"]
    assert drops["below_severity_threshold"] == 1


def test_a_row_whose_clip_would_be_narrower_than_the_witness_span_is_dropped():
    """Above 9 s the pad budget shrinks, so the clip stops being the audio Soniox heard."""
    c = scored("r", "athens", "a_2026", 100.0, 10.0, "ένα", "ένα δύο τρία τέσσερα")
    assert c["geometry"] is not None and not c["geometry"].reuse_gap_verify
    rows, drops, _ = P2.select([c], manifest(), frozenset(), {}, api_for([c]))
    assert rows == [] and drops["clip_narrower_than_witness_span"] == 1


def test_a_two_word_witness_is_not_thrown_away_for_being_short():
    """A short witness cannot estimate a WER, but it can still hold a word we lost."""
    c = scored("r", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο")
    rows, _, _ = P2.select([c], manifest(), frozenset(), {}, api_for([c]))
    assert len(rows) == 1


def test_a_meeting_whose_overlap_cannot_be_checked_is_dropped_whole():
    c = scored("r", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    rows, drops, _ = P2.select([c], manifest(), frozenset(), {}, api_for([c]),
                               overlap_checkable=frozenset({("chania", "b_2026")}))
    assert rows == [] and drops["overlap_not_checkable"] == 1


def test_a_clip_overlapping_an_existing_training_row_is_dropped():
    c = scored("r", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    g = c["geometry"]
    occupied = {("athens", "a_2026"): [(g.start_ms + 500, g.end_ms + 500, "pack:x")]}
    rows, drops, ledger = P2.select([c], manifest(), frozenset(), occupied,
                                    api_for([c]))
    assert rows == [] and drops["existing_clip_overlap"] == 1
    assert any(e.get("with") == ["pack:x"] for e in ledger)


def test_an_adjacent_but_not_overlapping_training_row_does_not_block():
    c = scored("r", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    g = c["geometry"]
    occupied = {("athens", "a_2026"): [(g.end_ms, g.end_ms + 5000, "pack:x")]}
    rows, _, _ = P2.select([c], manifest(), frozenset(), occupied, api_for([c]))
    assert len(rows) == 1


def test_overlap_is_scoped_to_the_meeting_not_the_slug():
    c = scored("r", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    g = c["geometry"]
    occupied = {("chania", "a_2026"): [(g.start_ms, g.end_ms, "pack:x")]}
    rows, _, _ = P2.select([c], manifest(), frozenset(), occupied, api_for([c]))
    assert len(rows) == 1


def test_selected_clips_never_overlap_each_other():
    a = scored("a", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    b = scored("b", "athens", "a_2026", 101.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    rows, drops, _ = P2.select([a, b], manifest(), frozenset(), {}, api_for([a, b]))
    assert len(rows) == 1 and drops["queue_overlap"] == 1


def test_scheduling_keeps_the_larger_set_not_the_greedy_one():
    """A-B, B-C, A and C disjoint: a pairwise greedy pass keeps one, the answer is two.

    With the 3 s pads the clips are 97-107, 105-115 and 113-123, so A touches B, B
    touches C, and A and C do not touch.
    """
    a = scored("a", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    b = scored("b", "athens", "a_2026", 108.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    c = scored("c", "athens", "a_2026", 116.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    rows, _, _ = P2.select([a, b, c], manifest(), frozenset(), {}, api_for([a, b, c]))
    assert [r["origin_row_id"] for r in rows] == ["a", "c"]


def test_selection_is_deterministic_under_input_order():
    a = scored("a", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    b = scored("b", "athens", "a_2026", 108.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    c = scored("c", "athens", "a_2026", 116.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    first, _, _ = P2.select([a, b, c], manifest(), frozenset(), {}, api_for([a, b, c]))
    second, _, _ = P2.select([c, b, a], manifest(), frozenset(), {}, api_for([a, b, c]))
    assert [r["clip_id"] for r in first] == [r["clip_id"] for r in second]


# ------------------------------------------------------------------ hard stops

def test_an_excluded_meeting_reaching_the_output_raises_rather_than_asserts():
    """`python -O` deletes asserts; this check stands between a clip and the benchmark.

    `candidates` already drops excluded meetings, so reaching `select` with one is the
    regression this second check exists to catch. It is tested by doing exactly that.
    """
    c = scored("r", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    with pytest.raises(P2.ExclusionViolation):
        P2.select([c], manifest([("athens", "a_2026")]), frozenset(), {},
                  api_for([c]))


def test_a_measurement_meeting_reaching_the_output_raises():
    c = scored("r", "athens", "m_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    with pytest.raises(P2.ExclusionViolation):
        P2.select([c], manifest(), frozenset({("athens", "m_2026")}), {},
                  api_for([c]))


def test_the_queue_row_carries_no_transcript_text():
    """Blindness is a property of the pipeline, not of the renderer."""
    c = scored("r", "athens", "a_2026", 100.0, 4.0, "ένα", "ένα δύο τρία τέσσερα")
    c["disagreement"] = P2.disagreement(c["adapter_text"], c["witness_text"])
    row = P2.queue_row(c)
    blob = json.dumps(row, ensure_ascii=False)
    assert "ένα" not in blob and "τέσσερα" not in blob
    assert row["witness"] == P2.WITNESS and row["adapter"] == P2.EXPECT_ARTIFACT


def test_the_yield_curve_is_frozen_and_monotonic():
    cands = [
        {"city": "athens", "meeting_id": "a_2026", "row_dur": 4.0,
         "adapter_text": "ένα", "witness_text": "ένα δύο τρία τέσσερα",
         "disagreement": P2.disagreement("ένα", "ένα δύο τρία τέσσερα")},
        {"city": "athens", "meeting_id": "a_2026", "row_dur": 4.0,
         "adapter_text": "ένα δύο τρία τέσσερα", "witness_text": "ένα δύο τρία τέσσερα",
         "disagreement": P2.disagreement("ένα δύο τρία τέσσερα",
                                         "ένα δύο τρία τέσσερα")},
    ]
    curve = P2.yield_curve(cands)
    assert [c["threshold"] for c in curve] == list(P2.DISAGREEMENT_GRID)
    counts = [c["spans"] for c in curve]
    assert counts == sorted(counts, reverse=True)


# ------------------------------------------------------------------ why not cutclips

def test_cutclips_rows_are_all_excluded_by_construction():
    """Spec section 4.1's first reason for not reusing `chooser/cutclips.py`.

    `chooser/deletions.py:20` keeps only windows whose city is in `A.CONFIRM_CITIES`,
    and every one of those five cities is a WHOLE-CITY key of the exclusion manifest.
    The yield of that path is therefore exactly zero, not merely small. Asserted against
    the live constants rather than quoted, so a change to either side fails here.
    """
    from eval.controlled_eval.exclusion_manifest import (
        CONFIRM_CITIES, build_manifest)
    import eval.controlled_eval.chooser.autoprompt as A
    assert set(A.CONFIRM_CITIES) == set(CONFIRM_CITIES)
    live = build_manifest(with_audio=False)
    for city in A.CONFIRM_CITIES:
        assert live.is_excluded(city, "any_meeting_2026")


def test_cutclips_positions_a_clip_by_column_fraction_not_by_time():
    """Spec section 4.1's second reason: `cutclips.py:45` reads

        start = it["startSec"] + o["pos"] * it["durationSec"]

    and `deletions.py:44` defines `pos` as `island_column / n_columns`. A column index
    is not a timestamp -- speakers do not talk at a constant rate -- so a pool built
    that way cannot carry an overlap rule stated in seconds. Asserted on the source so
    the claim cannot rot into folklore.
    """
    root = Path(__file__).resolve().parents[2] / "eval/controlled_eval/chooser"
    cut = (root / "cutclips.py").read_text(encoding="utf-8")
    dele = (root / "deletions.py").read_text(encoding="utf-8")
    assert 'o["pos"] * it["durationSec"]' in cut
    assert '"pos": i["s"] / max(1, len(cols))' in dele
