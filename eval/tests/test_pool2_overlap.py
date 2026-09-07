"""Tests for what Pool 2 counts as already-occupied time (spec section 4.2).

"Clips overlapping in time with any existing training or evaluation clip are dropped"
is one sentence, and satisfying it needs three sources that are held differently:

  * the stage-2 training rows, which carry `start` and `end`;
  * the council packs, which DO NOT -- their manifests carry `span_sec` only, so the
    spans have to be reconstructed and cached, and a meeting whose diarization job is
    corrupt has no reconstructable spans at all;
  * the published pool-1 listening queue.

A source that is silently absent looks exactly like a meeting nobody has used, and the
difference between those two is whether the rule ran. These tests are about that
difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import cut_pool2_clips as P2  # noqa: E402


def write_train(tmp_path, rows):
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def write_packs(tmp_path, packs):
    p = tmp_path / "packs.json"
    p.write_text(json.dumps({"packs": packs}), encoding="utf-8")
    return p


def write_queue(tmp_path, clips):
    p = tmp_path / "queue.json"
    p.write_text(json.dumps({"clips": clips}), encoding="utf-8")
    return p


def test_all_three_sources_land_in_one_index(tmp_path):
    train = write_train(tmp_path, [
        {"id": "t1", "city_id": "athens", "meeting_id": "a_2026",
         "start": 10.0, "end": 14.0}])
    packs = write_packs(tmp_path, [
        {"pack_id": "p1", "arm": "contiguous", "city_id": "athens",
         "meeting_id": "a_2026", "start": 100.0, "end": 128.0, "published": True}])
    queue = write_queue(tmp_path, [
        {"clip_id": "athens__a_2026__200000", "city": "athens",
         "meeting_id": "a_2026", "clip_start": 200.0, "clip_end": 209.0}])
    index, notes = P2.occupied_spans(packs, train, queue)
    got = sorted(index[("athens", "a_2026")])
    assert [i for _, _, i in got] == ["train:t1", "pack:p1",
                                      "listen:athens__a_2026__200000"]
    assert ("athens", "a_2026") in notes["pack_spans_available_for"]


def test_a_spliced_pack_occupies_its_utterances_not_the_gaps_between_them(tmp_path):
    """A spliced pack is glued with synthetic silence; the silence is not training audio.

    Collapsing it to first-start..last-end would block Pool 2 out of real free time.
    """
    train = write_train(tmp_path, [])
    packs = write_packs(tmp_path, [
        {"pack_id": "s1", "arm": "spliced", "city_id": "athens",
         "meeting_id": "a_2026", "start": 10.0, "end": 14.0, "published": True},
        {"pack_id": "s1", "arm": "spliced", "city_id": "athens",
         "meeting_id": "a_2026", "start": 300.0, "end": 305.0, "published": True},
        {"pack_id": "s9", "arm": "spliced", "city_id": "athens",
         "meeting_id": "a_2026", "start": 140.0, "end": 160.0, "published": False},
    ])
    queue = write_queue(tmp_path, [])
    index, notes = P2.occupied_spans(packs, train, queue)
    spans = [(s, e) for s, e, _ in index[("athens", "a_2026")]]
    # the 140-160 s interval was reconstructed but never cut, so it occupies nothing
    assert spans == [(10000, 14000), (300000, 305000)]
    assert notes["unpublished_packs_ignored"] == 1
    # the 286 s between them is free, and a clip there is not blocked
    c = {"origin_row_id": "r", "city": "athens", "meeting_id": "a_2026",
         "meeting_date": None, "audio_url": "https://x/a.mp3", "row_start": 150.0,
         "row_dur": 4.0, "tier": "VERIFIED", "witness_text": "ένα δύο τρία τέσσερα",
         "adapter_text": "ένα", "gap_verify_clip": None,
         "geometry": P2.clip_geometry(150.0, 4.0)}
    from eval.controlled_eval.exclusion_manifest import Manifest

    class T:
        def span(self, city, meeting, rid):
            return (150.0, 154.0)

    rows, _, _ = P2.select(
        [c], Manifest(meeting_keys=frozenset(), city_keys=frozenset()), frozenset(),
        index, T())
    assert len(rows) == 1


def test_a_missing_pack_span_cache_is_reported_not_treated_as_no_packs(tmp_path):
    train = write_train(tmp_path, [])
    queue = write_queue(tmp_path, [])
    index, notes = P2.occupied_spans(tmp_path / "absent.json", train, queue)
    assert notes["pack_spans_available_for"] == set()
    assert index == {} or all(not v for v in index.values())


def test_the_index_is_keyed_on_city_and_meeting_together(tmp_path):
    train = write_train(tmp_path, [
        {"id": "t1", "city_id": "athens", "meeting_id": "a_2026",
         "start": 10.0, "end": 14.0},
        {"id": "t2", "city_id": "chania", "meeting_id": "a_2026",
         "start": 10.0, "end": 14.0}])
    index, _ = P2.occupied_spans(tmp_path / "absent.json", train,
                                 write_queue(tmp_path, []))
    assert len(index[("athens", "a_2026")]) == 1
    assert len(index[("chania", "a_2026")]) == 1


def test_spans_are_sorted_so_the_scheduler_sees_a_stable_index(tmp_path):
    train = write_train(tmp_path, [
        {"id": "b", "city_id": "athens", "meeting_id": "a_2026",
         "start": 100.0, "end": 104.0},
        {"id": "a", "city_id": "athens", "meeting_id": "a_2026",
         "start": 10.0, "end": 14.0}])
    index, _ = P2.occupied_spans(tmp_path / "absent.json", train,
                                 write_queue(tmp_path, []))
    starts = [s for s, _, _ in index[("athens", "a_2026")]]
    assert starts == sorted(starts)


@pytest.mark.skipif(not P2.PACK_SPANS.exists(),
                    reason="pack spans have not been rebuilt on this machine")
def test_the_real_pack_span_cache_names_the_meetings_it_could_not_rebuild():
    """Six diarization jobs on this machine are zero-filled; their meetings have no
    reconstructable pack spans and the cache has to say so out loud."""
    payload = json.loads(P2.PACK_SPANS.read_text(encoding="utf-8"))
    assert "corrupt_diarization_jobs" in payload
    assert "meetings_without_reconstructable_spans" in payload
    for city, meeting in payload["meetings_without_reconstructable_spans"]:
        assert isinstance(city, str) and isinstance(meeting, str)
