"""Tests for the listening-clip cutter (spec section 4.1-4.2).

Record: exp-2026-08-26-training-round-2.

The failure this file exists to prevent is not a crash. It is a clip cut from an
evaluation meeting, which is training on the test set and is not reversible. The
exclusion tests therefore come first and are written against the canonical
`(city, meeting)` key, never the meeting slug alone.

The second failure it exists to prevent is subtler and was found by measurement, not by
reasoning: a row whose span has been edited since Soniox ran on it still carries the old
witness text, and the cached wav is a different clip. 71 of the surviving rows are in
that state, one of them off by 13.25 s.

Everything here runs on stub rows. No audio is read, no ffmpeg is invoked, and no
transcript text appears in this file: the fixtures carry ids and seconds only.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.exclusion_manifest import Manifest  # noqa: E402
from eval.controlled_eval import cut_listening_clips as CL  # noqa: E402
from eval.controlled_eval.cut_listening_clips import (  # noqa: E402
    MAX_CLIP, MIN_CLIP, PAD, POOL, SOURCE, WITNESS, ExclusionViolation,
    GeometryError, candidates, clip_geometry, queue_row, select, witness_matches)


# --------------------------------------------------------------------- fixtures

def train_row(rid, city="athens", meeting="mar23_2026", start=100.0, dur=4.0,
              bucket=POOL):
    return {
        "id": rid, "city_id": city, "meeting_id": meeting,
        "meeting_date": "2026-03-23 17:00:00",
        "audio_url": f"https://data.opencouncil.gr/audio/{city}.mp3",
        "start": f"{start}", "end": f"{start + dur}", "dur": f"{dur}",
        "text": "TRAINING TARGET", "source": "correction", "bucket": bucket,
    }


def gv_row(rid, n_added=5, tier="VERIFIED", witness="WITNESS"):
    return {
        "id": rid, "tier": tier, "found_frac": "0.9", "n_added": f"{n_added}",
        "soniox_text": witness,
        "clip_padded": f"/nowhere/{rid}.pad.wav",
        "clip_exact": f"/nowhere/{rid}.exact.wav",
    }


def manifest(meetings=(), cities=()):
    return Manifest(frozenset(meetings), frozenset(cities))


def prov_from(rows):
    """Provenance that agrees with every row, i.e. nothing was edited since."""
    return {r["id"]: {"start": float(r["start"]), "end": float(r["end"]),
                      "audio_url": r["audio_url"]} for r in rows}


def spans_from(rows):
    idx = collections.defaultdict(list)
    for r in rows:
        idx[(r["city_id"], r["meeting_id"])].append(
            (int(round(float(r["start"]) * 1000)),
             int(round(float(r["end"]) * 1000)), r["id"]))
    return idx


def run(rows, gv, man=None, spans=None, prov=None, **kw):
    """select() over the stub rows, with every guard supplied by default."""
    return select(candidates(rows, gv), man or manifest(),
                  spans if spans is not None else spans_from(rows),
                  prov if prov is not None else prov_from(rows), **kw)


# ------------------------------------------------------------------- exclusions

def test_an_excluded_meeting_never_reaches_the_queue():
    rows = [train_row("a", city="athens", meeting="mar23_2026")]
    kept, drops, _ = run(rows, {"a": gv_row("a")},
                         man=manifest(meetings=[("athens", "mar23_2026")]))
    assert kept == []
    assert drops["excluded"] == 1


def test_an_excluded_city_excludes_every_one_of_its_meetings():
    rows = [train_row("a", city="sparta", meeting="jan1_2026"),
            train_row("b", city="sparta", meeting="feb2_2026")]
    kept, drops, _ = run(rows, {"a": gv_row("a"), "b": gv_row("b")},
                         man=manifest(cities=["sparta"]))
    assert kept == []
    assert drops["excluded"] == 2


def test_the_key_is_city_and_meeting_not_the_meeting_slug():
    """Meeting slugs repeat across cities. A meeting-only key would silently drop real
    training material, and built the other way round would let an excluded meeting
    through under a different city."""
    rows = [train_row("a", city="athens", meeting="mar23_2026"),
            train_row("b", city="chania", meeting="mar23_2026")]
    kept, _, _ = run(rows, {"a": gv_row("a"), "b": gv_row("b")},
                     man=manifest(meetings=[("athens", "mar23_2026")]))
    assert [r["origin_row_id"] for r in kept] == ["b"]


def test_the_final_exclusion_check_raises_rather_than_asserts():
    """`python -O` deletes assert statements. The last guard between a typo in the
    filter and a permanently contaminated benchmark may not be one of them."""
    rows = [train_row("a", city="athens", meeting="mar23_2026")]
    with pytest.raises(ExclusionViolation):
        run(rows, {"a": gv_row("a")},
            man=manifest(meetings=[("athens", "mar23_2026")]),
            _skip_exclusion_filter_for_test=True)


# ---------------------------------------------------------------- witness gate

def test_a_row_without_a_witness_is_not_a_candidate():
    """The witnessless rows are the earlier human-reviewed stratum. Filtering on the
    witness also selects the unreviewed remainder, which is what §4.1 asks for -- so
    this is one filter doing two jobs, and it must not be relaxed."""
    rows = [train_row("a"), train_row("b")]
    assert [c["origin_row_id"] for c in candidates(rows, {"a": gv_row("a")})] == ["a"]


def test_only_the_deletion_hard_bucket_is_mined():
    rows = [train_row("a", bucket="backbone"), train_row("b", bucket=POOL)]
    gv = {"a": gv_row("a"), "b": gv_row("b")}
    assert [c["origin_row_id"] for c in candidates(rows, gv)] == ["b"]


def test_every_tier_but_verified_fails_closed():
    """MIDDLE and REJECT are verdicts, and an unknown tier is a schema change."""
    rows = [train_row(x) for x in "abcd"]
    gv = {"a": gv_row("a", tier="REJECT"), "b": gv_row("b", tier="MIDDLE"),
          "c": gv_row("c", tier="SOMETHING_NEW"), "d": gv_row("d", tier="VERIFIED")}
    assert [c["origin_row_id"] for c in candidates(rows, gv)] == ["d"]


def test_an_empty_witness_string_is_not_a_witness():
    assert candidates([train_row("a")], {"a": gv_row("a", witness="")}) == []


# ------------------------------------------- the witness must describe THIS span

def test_a_row_edited_since_the_witness_ran_is_dropped():
    """Measured: 71 surviving rows differ from the span Soniox was run on, one by
    13.25 s. Their `soniox_text` and their cached wav belong to other audio."""
    rows = [train_row("a", start=100.0, dur=4.0)]
    prov = {"a": {"start": 100.0, "end": 117.25,
                  "audio_url": rows[0]["audio_url"]}}
    kept, drops, _ = run(rows, {"a": gv_row("a")}, prov=prov)
    assert kept == []
    assert drops["witness_span_mismatch"] == 1


def test_the_same_duration_at_a_different_offset_is_not_a_match():
    """A shifted clip has the right length and the wrong content, so length is not
    what gets compared."""
    cand = candidates([train_row("a", start=100.0, dur=4.0)], {"a": gv_row("a")})[0]
    shifted = {"start": 108.0, "end": 112.0, "audio_url": cand["audio_url"]}
    assert witness_matches(cand, shifted) is False
    exact = {"start": 100.0, "end": 104.0, "audio_url": cand["audio_url"]}
    assert witness_matches(cand, exact) is True


def test_a_witness_from_another_recording_is_not_a_match():
    cand = candidates([train_row("a")], {"a": gv_row("a")})[0]
    assert witness_matches(cand, {"start": 100.0, "end": 104.0,
                                  "audio_url": "https://elsewhere/x.mp3"}) is False


def test_a_row_with_no_provenance_snapshot_is_dropped_not_trusted():
    rows = [train_row("a")]
    kept, drops, _ = run(rows, {"a": gv_row("a")}, prov={})
    assert kept == []
    assert drops["witness_span_mismatch"] == 1


# ------------------------------------------------------------------- geometry

def test_a_short_row_is_padded_to_the_gap_verify_geometry_and_reuses_its_wav():
    g = clip_geometry(start=100.0, dur=3.64)
    assert g.pad == PAD
    assert (g.start, round(g.end, 2)) == (97.0, 106.64)
    assert MIN_CLIP <= g.dur <= MAX_CLIP
    assert g.reuse_gap_verify is True


def test_a_long_row_shrinks_its_pad_to_the_ceiling_and_cannot_reuse():
    g = clip_geometry(start=100.0, dur=12.0)
    assert g.dur == pytest.approx(MAX_CLIP)
    assert g.pad == pytest.approx(1.5)
    assert g.reuse_gap_verify is False


def test_a_row_longer_than_the_ceiling_is_dropped_rather_than_truncated():
    """A truncated clip asks the listener to label speech that runs off the edge."""
    assert clip_geometry(start=100.0, dur=MAX_CLIP + 0.01) is None
    assert clip_geometry(start=100.0, dur=MAX_CLIP) is not None


def test_the_start_of_a_recording_clamps_at_zero_and_still_reuses():
    """gap2_verify_audio.py:253 clamps the same way (`max(0.0, start - PAD)`), so the
    clamped geometry is still byte-identical to the wav already on disk."""
    g = clip_geometry(start=1.0, dur=4.0)
    assert g.start == 0.0
    assert g.reuse_gap_verify is False, \
        "one side of the pad was cut short, so this is not the cached geometry"


def test_padding_lost_at_the_start_moves_to_the_other_side():
    """Without redistribution a 0.14 s row at t=0 makes a 3.14 s clip and breaks the
    5 s floor silently. The budget is what is fixed, not the margin on each side."""
    g = clip_geometry(start=0.0, dur=0.14)
    assert g.start == 0.0
    assert g.dur == pytest.approx(6.14)
    assert (g.pad_left_ms, g.pad_right_ms) == (0, 6000)


def test_every_emitted_clip_lands_inside_the_frozen_bounds():
    for start in (0.0, 0.5, 2.9, 3.0, 500.0):
        for dur in (0.14, 1.0, 3.64, 8.9, 9.0, 9.1, 12.0, 15.0):
            g = clip_geometry(start=start, dur=dur)
            assert g is not None
            assert MIN_CLIP - 1e-9 <= g.dur <= MAX_CLIP + 1e-9


def test_a_non_finite_or_inconsistent_row_is_an_error_not_a_guess():
    with pytest.raises(GeometryError):
        clip_geometry(start=float("nan"), dur=1.0)
    bad = train_row("a")
    bad["dur"] = "99.0"                       # disagrees with end - start
    with pytest.raises(GeometryError):
        candidates([bad], {"a": gv_row("a")})
    rev = train_row("b")
    rev["end"] = rev["start"]
    with pytest.raises(GeometryError):
        candidates([rev], {"b": gv_row("b")})


# -------------------------------------------------------------------- overlap

def test_a_clip_overlapping_another_training_row_is_dropped():
    """Applied literally, and affordable: measured, 724 of 899 selected clips touch no
    other training row and only 175 touch one or more. An earlier draft of this module
    argued the literal rule was unusable; the measurement says otherwise, and dropping
    those rows is what stops a new human label from contradicting an existing target
    for the same audio."""
    rows = [train_row("a", start=100.0, dur=4.0),        # clip 97.0 -> 107.0
            train_row("neighbour", start=108.0, dur=2.0, bucket="backbone")]
    kept, drops, _ = run(rows, {"a": gv_row("a")})
    assert [r["origin_row_id"] for r in kept] == ["a"]

    rows[1] = train_row("neighbour", start=106.0, dur=2.0, bucket="backbone")
    kept, drops, _ = run(rows, {"a": gv_row("a")})
    assert kept == []
    assert drops["training_overlap"] == 1


def test_two_overlapping_clips_in_one_meeting_keep_the_richer_witness():
    rows = [train_row("a", start=100.0, dur=4.0),
            train_row("b", start=102.0, dur=4.0)]
    gv = {"a": gv_row("a", n_added=2), "b": gv_row("b", n_added=9)}
    kept, drops, _ = run(rows, gv, spans={})
    assert [r["origin_row_id"] for r in kept] == ["b"]
    assert drops["queue_overlap"] == 1


def test_an_overlap_chain_keeps_the_most_clips_not_the_greediest_pick():
    """A-B and B-C overlap while A and C do not. Taking B first costs a clip."""
    rows = [train_row("a", start=100.0, dur=4.0),     #  97 -> 107
            train_row("b", start=106.0, dur=4.0),     # 103 -> 113
            train_row("c", start=112.0, dur=4.0)]     # 109 -> 119
    gv = {"a": gv_row("a", n_added=1), "b": gv_row("b", n_added=99),
          "c": gv_row("c", n_added=1)}
    kept, drops, _ = run(rows, gv, spans={})
    assert [r["origin_row_id"] for r in kept] == ["a", "c"]
    assert drops["queue_overlap"] == 1


def test_the_same_span_in_two_different_meetings_is_not_an_overlap():
    rows = [train_row("a", meeting="m1", start=100.0, dur=4.0),
            train_row("b", meeting="m2", start=100.0, dur=4.0)]
    kept, drops, _ = run(rows, {"a": gv_row("a"), "b": gv_row("b")})
    assert len(kept) == 2
    assert drops["queue_overlap"] == 0


def test_the_same_span_in_two_cities_sharing_a_meeting_slug_is_not_an_overlap():
    rows = [train_row("a", city="athens", meeting="mar23_2026", start=100.0),
            train_row("b", city="chania", meeting="mar23_2026", start=100.0)]
    kept, _, _ = run(rows, {"a": gv_row("a"), "b": gv_row("b")})
    assert len(kept) == 2


def test_adjacent_but_disjoint_clips_both_survive():
    rows = [train_row("a", start=100.0, dur=4.0),   # 97.0 -> 107.0
            train_row("b", start=110.0, dur=4.0)]   # 107.0 -> 117.0
    kept, drops, _ = run(rows, {"a": gv_row("a"), "b": gv_row("b")}, spans={})
    assert len(kept) == 2
    assert drops["queue_overlap"] == 0


def test_selection_is_independent_of_input_order():
    rows = [train_row(x, start=s, dur=4.0) for x, s in
            (("a", 100.0), ("b", 102.0), ("c", 130.0), ("d", 131.0))]
    gv = {r["id"]: gv_row(r["id"], n_added=i) for i, r in enumerate(rows)}
    fwd, _, _ = run(rows, gv, spans={})
    rev, _, _ = run(list(reversed(rows)), gv, spans={})
    assert [r["clip_id"] for r in fwd] == [r["clip_id"] for r in rev]


def test_a_tie_on_witness_richness_is_broken_deterministically_by_id():
    rows = [train_row("b", start=100.0, dur=4.0),
            train_row("a", start=102.0, dur=4.0)]
    gv = {"a": gv_row("a", n_added=5), "b": gv_row("b", n_added=5)}
    kept, _, _ = run(rows, gv, spans={})
    assert [r["origin_row_id"] for r in kept] == ["a"]


# ------------------------------------------------------ provenance and blindness

def test_the_queue_row_carries_the_provenance_section_4_2_requires():
    row = queue_row(candidates([train_row("a")], {"a": gv_row("a")})[0])
    for field in ("clip_id", "city", "meeting_id", "meeting_date", "audio_url",
                  "clip_start", "clip_end", "clip_dur", "pool", "origin_row_id",
                  "source", "witness", "reused_gap_verify_clip"):
        assert field in row, field
    assert row["source"] == SOURCE
    assert row["witness"] == WITNESS
    assert row["pool"] == POOL


def test_no_queue_row_carries_transcript_text_of_any_kind():
    """The queue is what the listening page loads. If the witness or the current
    training target travels inside it, the page cannot be blind however it is
    written -- blindness has to be a property of the file, not of the renderer."""
    row = queue_row(candidates([train_row("a")], {"a": gv_row("a")})[0])
    blob = repr(row)
    assert "WITNESS" not in blob
    assert "TRAINING TARGET" not in blob
    for banned in ("text", "soniox_text", "target", "hypothesis", "added_tokens"):
        assert banned not in row, banned


def test_the_clip_id_is_stable_and_names_its_meeting():
    a = queue_row(candidates([train_row("a")], {"a": gv_row("a")})[0])
    b = queue_row(candidates([train_row("a")], {"a": gv_row("a")})[0])
    assert a["clip_id"] == b["clip_id"]
    assert a["clip_id"].startswith("athens__mar23_2026__")


def test_a_clip_id_that_is_not_path_safe_is_refused():
    bad = train_row("a", city="../etc")
    with pytest.raises(GeometryError):
        queue_row(candidates([bad], {"a": gv_row("a")})[0])


def test_the_ledger_records_a_reason_for_every_candidate():
    """The private ledger is how a dropped row is ever explained. Aggregate counts in
    the repo cannot answer 'why is this meeting missing'."""
    rows = [train_row("a", city="sparta"), train_row("b"),
            train_row("c", meeting="other_2026", start=100.0, dur=20.0)]
    gv = {r["id"]: gv_row(r["id"]) for r in rows}
    kept, drops, ledger = run(rows, gv, man=manifest(cities=["sparta"]))
    assert {e["row"] for e in ledger} == {"a", "b", "c"}
    reasons = {e["row"]: e["drop"] for e in ledger}
    assert reasons["a"] == "excluded"
    assert reasons["c"] == "too_long"
    assert reasons["b"] is None


# ------------------------------------------------------------- what reaches git

def test_the_repo_side_result_is_counts_and_hashes_only():
    """Transcript text and audio never go in git (CLAUDE.md, the 2026-07-21 purge).
    The queue itself lives under ~/.cache; what lands in the repo is the funnel."""
    import json as _json
    p = ROOT / "eval" / "results_listening_queue.json"
    if not p.exists():
        pytest.skip("queue not cut on this machine")
    d = _json.loads(p.read_text(encoding="utf-8"))
    assert "clips" not in d and "queue" not in d
    for key in ("n_clips", "drops", "clip_hours", "cities", "queue_sha256"):
        assert key in d, key
    blob = _json.dumps(d, ensure_ascii=False)
    # A URL, not the four letters: `http-fetch-failed` is a legitimate failure CLASS
    # (cut_listening_clips.classify_failure), and a test that banned the substring would
    # be weakened by whoever hit it next rather than tightened.
    assert "://" not in blob, "a source URL is not a count"
    # No clip id, and so no meeting id, travels into the repo.
    assert "__" not in blob


def test_the_expected_clip_format_is_the_one_both_sources_produce():
    """ffmpeg is invoked with `-ac 1 -ar 16000` and the reused gap-verify wavs were cut
    the same way, so anything else in the queue came from somewhere unexpected."""
    from eval.controlled_eval.cut_listening_clips import EXPECT_STREAM
    assert EXPECT_STREAM == {"codec_name": "pcm_s16le", "sample_rate": "16000",
                             "channels": 1}


def test_an_excluded_recording_is_caught_under_a_second_meeting_id():
    """`is_excluded` compares ids. The same recording uploaded under a second meeting
    id would pass that and be the same audio, so the recording itself is compared too."""
    from eval.controlled_eval.cut_listening_clips import excluded_recordings
    rows = [train_row("a", city="athens", meeting="mar23_2026"),
            train_row("b", city="athens", meeting="mar23_2026_reupload")]
    rows[1]["audio_url"] = rows[0]["audio_url"]      # the same recording, renamed
    banned = excluded_recordings(rows, manifest(meetings=[("athens", "mar23_2026")]))
    assert rows[1]["audio_url"] in banned


# ==========================================================================
# What may be written to the repository (CodeRabbit, 2026-08-28)
# ==========================================================================

def test_a_failure_reason_written_to_git_carries_no_url():
    """ffmpeg puts the input URL in its own error text, and the failure summary of a
    cut is written to eval/results_listening_queue.json, which is in git."""
    stderr = ("https://townhall-recordings.example.com/chania__mar23_2026.mp3: "
              "Server returned 403 Forbidden")
    reason = CL.classify_failure(stderr)
    assert reason == "http-fetch-failed"
    assert "http" not in reason.replace("http-fetch-failed", "")
    assert "example.com" not in reason and "chania" not in reason


def test_an_unrecognised_ffmpeg_failure_still_classifies():
    assert CL.classify_failure("something nobody has seen before") == "ffmpeg-error"
    assert CL.classify_failure("") == "ffmpeg-error"
    assert CL.classify_failure(None) == "ffmpeg-error"
