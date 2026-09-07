"""Tests for the ordinary-sample cutter v2 (wayfinder #59 design, #64 build).

Record: exp-2026-08-26-training-round-2. Frame replaced 2026-09-06.

The failure this file exists to prevent is not a crash. It is a sample that LOOKS
ordinary and is not. v1 had two ways to be that, and v2 fixes both, so both are tested:

1. Positions that are, somewhere up the chain, a function of where the incumbent failed.
   The strongest check is positive noninterference: replace every field of every meeting
   file except the four whitelisted ones -- INCLUDING the published transcript -- shuffle
   the files, and the plan must not move.

2. A frame the round being judged has trained on. `eligible_meetings` removes every
   meeting any council training source touched, and every meeting the exclusion manifest
   blocks AT BOTH LEVELS. The city level has its own test because skipping it is exactly
   the mistake that turned 15 eligible meetings into 33.

Everything here runs on stub payloads unless a test is explicitly marked as reading the
cache. No audio is fetched, no ffmpeg is invoked, and the only "transcript" text in this
file is an obvious placeholder that must never come out the other side.
"""
from __future__ import annotations

import ast
import collections
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import cut_ordinary_sample as ORD  # noqa: E402
from eval.controlled_eval.cut_ordinary_sample import (  # noqa: E402
    CLIP_MS, CLIPS_PER_MEETING, END_MARGIN_MS, N_CLIPS, ExclusionViolation,
    FrameError, GeometryError, IndependenceViolation, InputNotAllowed, Meeting,
    allocation, build_rows, check_frame, check_sample, clip_id, eligible_meetings,
    meetings_from_payloads, positions, sample_row)
from eval.controlled_eval.exclusion_manifest import Manifest  # noqa: E402

# The placeholder that must never survive a projection. If it ever appears in a plan,
# a row, a manifest or a clip id, published council text has reached the cutter.
TRANSCRIPT = "PUBLISHED TRANSCRIPT TEXT THAT MUST NEVER BE READ"


# --------------------------------------------------------------------- fixtures

def payload(city="athens", meeting="jul13_2026", **extra):
    """A meeting file with every field the real cache carries, transcript included."""
    d = {
        "meeting": {
            "id": meeting, "cityId": city,
            "name": "Συνεδρίαση Δημοτικού Συμβουλίου",
            "name_en": "Council meeting",
            "dateTime": "2026-07-13T15:00:00.000Z",
            "audioUrl": f"https://data.opencouncil.gr/audio/{city}_{meeting}.mp3",
            "videoUrl": f"https://data.opencouncil.gr/video/{city}_{meeting}.mp4",
            "youtubeUrl": "https://www.youtube.com/watch?v=xxxx",
            "agendaUrl": "https://example.invalid/agenda.pdf",
            "muxPlaybackId": "abc123", "administrativeBodyId": "cm8y9ijh7",
            "released": True, "createdAt": "2026-07-01", "updatedAt": "2026-07-14",
        },
        "transcript": [{"text": TRANSCRIPT, "start": 12.0, "end": 18.0}],
        "subjects": [{"name": TRANSCRIPT}],
        "speakerTags": [{"label": TRANSCRIPT}],
        "taskStatus": {"transcribe": "done"},
        "people": [], "parties": [], "city": {"id": city},
        "transcriptHiddenForReview": False,
    }
    d.update(extra)
    return d


def payloads_for(keys):
    return [payload(city=c, meeting=m) for c, m in keys]


def manifest(meetings=(), cities=()):
    return Manifest(frozenset(meetings), frozenset(cities))


def durations_for(meetings, seconds=3600.0):
    return {m.key: int(seconds * 1000) for m in meetings}


def plan_rows(keys, seconds=3600.0, per_meeting=CLIPS_PER_MEETING):
    meetings = meetings_from_payloads(payloads_for(keys))
    alloc = allocation(meetings, per_meeting, per_meeting * len(meetings))
    return build_rows(meetings, alloc, durations_for(meetings, seconds))


# ==========================================================================
# Independence: the published transcript must never reach a position
# ==========================================================================

def test_the_published_transcript_never_survives_the_projection():
    """A meeting file carries OpenCouncil's published text, which is partly ASR output.
    Reading it to place a clip would reopen the leak this whole design exists to close."""
    (m,), = [meetings_from_payloads([payload()])]
    assert TRANSCRIPT not in repr(m)
    for banned in ("transcript", "subjects", "speakerTags", "taskStatus", "people"):
        assert not hasattr(m, banned), banned
    rows = plan_rows([("athens", "jul13_2026")])
    blob = json.dumps(rows, ensure_ascii=False)
    assert TRANSCRIPT not in blob
    assert "transcript" not in blob and "subjects" not in blob


def test_a_meeting_contributes_four_fields_and_nothing_else():
    (m,) = meetings_from_payloads([payload()])
    assert (m.city, m.meeting_id) == ("athens", "jul13_2026")
    assert set(Meeting.__slots__) == set(ORD.MEETING_FIELDS)
    for banned in ("videoUrl", "youtubeUrl", "muxPlaybackId", "released", "transcript"):
        assert not hasattr(m, banned), banned


def test_a_meeting_cannot_be_given_a_forbidden_field_afterwards():
    """`slots` is the enforcement. The exception type is not fixed: a frozen+slots
    dataclass raises `TypeError` on some CPython versions and `FrozenInstanceError` on
    others. What matters is that the assignment does not succeed."""
    (m,) = meetings_from_payloads([payload()])
    assert Meeting.__slots__ == ORD.MEETING_FIELDS
    with pytest.raises((AttributeError, TypeError)):
        m.transcript = TRANSCRIPT              # type: ignore[attr-defined]
    assert not hasattr(m, "transcript")


def test_the_plan_does_not_move_when_every_other_field_is_replaced():
    """The positive noninterference check. Everything outside the four whitelisted
    fields is poisoned in the second run -- transcript, subjects, speaker tags, video
    URLs -- and the files are reversed. If any of it reached a position or an ordering,
    this fails."""
    keys = [("athens", "jul13_2026"), ("athens", "jul20_2026"),
            ("chania", "jul13_2026"), ("thira", "aug1_2026")]
    base = payloads_for(keys)
    poisoned = []
    for i, p in enumerate(reversed(base)):
        bad = json.loads(json.dumps(p))
        for k in list(bad):
            if k != "meeting":
                bad[k] = f"POISON-{i}"
        for k in list(bad["meeting"]):
            if k in ("cityId", "id", "dateTime", "audioUrl"):
                continue
            bad["meeting"][k] = f"POISON-{i}"
        poisoned.append(bad)

    def run(ps):
        meetings = meetings_from_payloads(ps)
        return build_rows(meetings, allocation(meetings, 3, 3 * len(meetings)),
                          durations_for(meetings))

    assert run(base) == run(poisoned)


def test_the_plan_does_not_depend_on_the_order_files_are_read_in():
    keys = [("athens", f"jul{i}_2026") for i in range(1, 6)]
    a = plan_rows(keys)
    b = plan_rows(list(reversed(keys)))
    assert a == b


def test_the_module_may_only_open_the_manifests_and_the_meetings_cache(tmp_path):
    assert {Path(p).name for p in ORD.ALLOWED_FILES} == {"manifest.jsonl",
                                                         "manifest.json"}
    assert ORD.ALLOWED_DIRS == (ORD.MEETINGS_DIR,)
    with pytest.raises(InputNotAllowed):
        ORD._open_input(Path.home() / ".cache" / "oc-public" / "gap-verify" /
                        "manifest.jsonl")
    with pytest.raises(InputNotAllowed):
        ORD._open_input(ORD.MEETINGS_DIR / "sub" / "deep.json")
    with pytest.raises(InputNotAllowed):
        ORD._open_input(ORD.MEETINGS_DIR / "notes.txt")
    with pytest.raises(InputNotAllowed):
        ORD._open_input(tmp_path / "anything.json")


def test_the_listening_queue_is_no_longer_an_input_at_all():
    """v2's frame has no connection to the failure-selected queue, and the code must not
    be able to reach it even by accident."""
    src = Path(ORD.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "listen-2026-08" not in node.value or _is_docstring(node.value, tree)
    assert "cut_challenge_sample" not in src


def _is_docstring(value, tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) == value:
                return True
    return False


def test_the_module_source_never_names_a_model_output():
    """A grep over the parsed source, not over the file. Docstrings and comments
    legitimately discuss the transcript and the deletion pool; what must not exist is a
    string literal or an attribute that could reach one."""
    tree = ast.parse(Path(ORD.__file__).read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                docstrings.add(d)
    banned = ("gap-verify", "gap_verify", "soniox", "witness", "hypothes", "bench_",
              "deletion_hard", "labels.json", "current_target", "train-screens",
              "correction", "transcript", "subjects", "speakertags")
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue
            low = node.value.lower()
            offenders += [(b, node.value[:60]) for b in banned if b in low]
        if isinstance(node, ast.Attribute):
            offenders += [(b, node.attr) for b in banned if b in node.attr.lower()]
        if isinstance(node, ast.Name):
            offenders += [(b, node.id) for b in banned if b in node.id.lower()]
    assert offenders == [], offenders


# ==========================================================================
# The frame
# ==========================================================================

def test_the_city_level_exclusion_is_applied_not_only_the_meeting_level(tmp_path):
    """This is the mistake that turned 15 eligible meetings into 33: applying only
    `meeting_keys` leaves the five confirmation cities of the category-chooser
    experiment in the frame."""
    for c, m in [("sparta", "jul1_2026"), ("athens", "jul2_2026"),
                 ("chania", "jul3_2026")]:
        (tmp_path / f"{c}__{m}.json").write_text(json.dumps(payload(c, m)))
    man = manifest(meetings=[("chania", "jul3_2026")], cities=["sparta"])
    ORD.ALLOWED_DIRS = (tmp_path,)
    try:
        kept, drops = eligible_meetings(man, tmp_path)
    finally:
        ORD.ALLOWED_DIRS = (ORD.MEETINGS_DIR,)
    assert [m.key for m in kept] == [("athens", "jul2_2026")]
    assert drops == {"trained_on": 0, "excluded_meeting": 1, "excluded_city": 1}


def test_the_key_is_city_and_meeting_not_the_meeting_slug():
    ms = meetings_from_payloads(payloads_for([("athens", "jul13_2026"),
                                              ("chania", "jul13_2026")]))
    assert len(ms) == 2


def test_one_recording_under_two_meeting_ids_is_an_error():
    a, b = payload("athens", "jul13_2026"), payload("athens", "jul13_2026_reupload")
    b["meeting"]["audioUrl"] = a["meeting"]["audioUrl"]
    with pytest.raises(FrameError, match="same recording"):
        meetings_from_payloads([a, b])


def test_a_meeting_with_two_different_urls_is_an_error_not_a_guess():
    a, b = payload(), payload()
    b["meeting"]["audioUrl"] = "https://data.opencouncil.gr/audio/other.mp3"
    with pytest.raises(FrameError, match="two different identities"):
        meetings_from_payloads([a, b])


def test_a_file_without_a_meeting_object_or_an_audio_url_is_refused():
    with pytest.raises(IndependenceViolation, match="no `meeting` object"):
        meetings_from_payloads([{"transcript": TRANSCRIPT}])
    bad = payload()
    del bad["meeting"]["audioUrl"]
    with pytest.raises(IndependenceViolation, match="audio_url"):
        meetings_from_payloads([bad])


def test_a_drifted_frame_is_refused_rather_than_cut():
    """A cache that has gained an eligible meeting is a different sample wearing the
    same name."""
    assert check_frame("anything", expect="") == ORD._sha256_bytes(b"anything")
    with pytest.raises(FrameError, match="different sample wearing the same name"):
        check_frame("anything", expect="0" * 64)


# ==========================================================================
# The allocation is fully determined
# ==========================================================================

def test_every_meeting_supplies_exactly_three_clips():
    meetings = meetings_from_payloads(
        payloads_for([("athens", f"jul{i}_2026") for i in range(1, 16)]))
    alloc = allocation(meetings)
    assert set(alloc.values()) == {3}
    assert len(alloc) == 15
    assert sum(alloc.values()) == N_CLIPS == 45


def test_there_is_no_city_quota_step_left_to_get_wrong():
    """v1 apportioned 80 clips over uneven city pools and needed largest-remainder. v2
    does not, and the function is gone rather than kept as a no-op."""
    assert not hasattr(ORD, "city_quotas")
    assert not hasattr(ORD, "allocate")
    assert CLIPS_PER_MEETING * ORD.N_MEETINGS == N_CLIPS


def test_a_frame_that_does_not_multiply_out_is_an_error_not_a_reallocation():
    meetings = meetings_from_payloads(
        payloads_for([("athens", f"jul{i}_2026") for i in range(1, 15)]))
    with pytest.raises(FrameError, match="not the frozen 45"):
        allocation(meetings)


def test_the_estimand_says_what_this_does_and_not_the_stronger_thing():
    assert "meetings drawn evenly" in ORD.ESTIMAND
    assert "NOT a uniformly random second of council audio" in ORD.ESTIMAND
    assert "independent of every model output" in ORD.ESTIMAND
    assert "both the meeting and the city level" in ORD.ESTIMAND


def test_the_precision_note_states_the_cluster_cost_in_plain_words():
    assert "15 meeting clusters" in ORD.PRECISION_NOTE
    assert "67" in ORD.PRECISION_NOTE
    assert "different order of precision" in ORD.PRECISION_NOTE


# ==========================================================================
# Positions
# ==========================================================================

def test_positions_are_inside_the_recording_with_the_tail_margin_respected():
    m = Meeting("athens", "jul13_2026", None, "https://x/y.mp3")
    dur = 600_000
    for start in positions(m, 3, dur):
        assert 0 <= start <= dur - CLIP_MS - END_MARGIN_MS


def test_three_clips_of_one_meeting_never_overlap_even_in_the_shortest_meeting():
    """The shortest meeting in the real frame is 141.5 s, leaving 91.5 s of usable span
    for three 20 s clips."""
    for i in range(200):
        m = Meeting("chania", f"jul{i}_2026", None, "https://x/y.mp3")
        xs = positions(m, 3, 141_500)
        assert len(xs) == 3
        assert all(xs[j + 1] - xs[j] >= CLIP_MS for j in range(2))


def test_positions_are_a_function_of_the_seed_and_the_meeting_only():
    m = Meeting("athens", "jul13_2026", None, "https://x/y.mp3")
    other = Meeting("athens", "jul13_2026", "2026-07-13", "https://elsewhere/z.mp3")
    assert positions(m, 3, 600_000) == positions(other, 3, 600_000)
    assert positions(m, 3, 600_000) == positions(m, 3, 600_000)


def test_a_recording_too_short_for_a_clip_is_an_error_not_a_truncation():
    m = Meeting("athens", "jul13_2026", None, "https://x/y.mp3")
    with pytest.raises(GeometryError, match="too short"):
        positions(m, 1, CLIP_MS + END_MARGIN_MS)


def test_positions_spread_over_the_recording_rather_than_clustering():
    dur = 3_600_000
    usable = dur - CLIP_MS - END_MARGIN_MS
    seen = set()
    for i in range(400):
        m = Meeting("chania", f"jul{i}_2026", None, "https://x/y.mp3")
        for x in positions(m, 3, dur):
            seen.add(int(x / usable * 10))
    assert seen == set(range(10))


# ==========================================================================
# Rows, ids and the last guard
# ==========================================================================

def test_the_clip_id_accepts_a_hyphenated_city():
    """`papagos-cholargos` is in the v2 frame. The v1 regex would have rejected it."""
    m = Meeting("papagos-cholargos", "jul30_2026", None, "https://x/y.mp3")
    assert clip_id(m, 97000) == "ord__papagos-cholargos__jul30_2026__97000"


def test_a_clip_id_that_is_not_path_safe_is_refused():
    with pytest.raises(GeometryError):
        clip_id(Meeting("../etc", "m_2026", None, "https://x/y.mp3"), 1)


def test_the_sample_row_carries_provenance_and_no_text_of_any_kind():
    row = sample_row(Meeting("athens", "jul13_2026", "2026-07-13", "https://x/y.mp3"), 0)
    for field in ("clip_id", "sample", "design_version", "city", "meeting_id",
                  "meeting_date", "audio_url", "clip_start", "clip_end", "clip_dur",
                  "source", "selection", "seed"):
        assert field in row, field
    assert row["sample"] == "ordinary" and row["design_version"] == "v2"
    for banned in ("text", "target", "transcript", "witness", "pool", "origin_row_id"):
        assert banned not in row, banned


def test_an_excluded_or_trained_on_meeting_raises_rather_than_asserts():
    """`python -O` deletes assert statements."""
    rows = plan_rows([("sparta", "jul1_2026")] * 1)
    with pytest.raises(ExclusionViolation):
        check_sample(rows, manifest(cities=["sparta"]), {})
    with pytest.raises(ExclusionViolation):
        check_sample(rows, manifest(), {("sparta", "jul1_2026"): ["clean-pack-arm"]})


def test_a_meeting_supplying_the_wrong_number_of_clips_is_refused():
    m = Meeting("athens", "jul13_2026", None, "https://x/y.mp3")
    rows = [sample_row(m, i * 40_000) for i in range(2)]
    with pytest.raises(ExclusionViolation, match="supplies exactly 3"):
        check_sample(rows, manifest(), {})


def test_a_row_labelled_as_the_other_sample_is_refused():
    rows = plan_rows([("athens", "jul13_2026")])
    rows[0]["sample"] = "challenge"
    with pytest.raises(ExclusionViolation, match="never summed"):
        check_sample(rows, manifest(), {})


# ==========================================================================
# What the cut checks, and what reaches git
# ==========================================================================

def test_a_twenty_second_clip_is_exactly_320000_frames():
    assert ORD.EXPECT_FRAMES == 320_000
    assert ORD.EXPECT_WAV == {"nchannels": 1, "sampwidth": 2, "framerate": 16000,
                              "nframes": 320_000}


def test_a_failure_reason_written_to_git_carries_no_url():
    stderr = ("https://data.opencouncil.gr/audio/thira_aug1.mp3: "
              "Server returned 403 Forbidden")
    assert ORD.classify_failure(stderr) == "http-fetch-failed"
    assert ORD.classify_failure("something nobody has seen") == "ffmpeg-error"
    assert ORD.classify_failure("") == "ffmpeg-error"
    assert ORD.classify_failure(None) == "ffmpeg-error"


def test_the_repo_side_result_is_counts_and_hashes_only():
    p = ROOT / "eval" / "results_ordinary_sample.json"
    if not p.exists():
        pytest.skip("sample not cut on this machine")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert "clips" not in d
    blob = json.dumps(d, ensure_ascii=False)
    assert "://" not in blob, "a source URL is not a count"
    assert "__" not in blob, "a clip id names its meeting"
    for key in ("n_clips", "estimand", "precision_note", "n_meeting_clusters",
                "sample_sha256", "frame_sha256", "design_version"):
        assert key in d, key
    assert d["design_version"] == "v2"
    assert d["n_meeting_clusters"] == 15


# ==========================================================================
# Against the real cache (skipped where it is absent)
# ==========================================================================

@pytest.fixture(scope="module")
def real_plan():
    if not ORD.MEETINGS_DIR.exists() or not ORD.DURATIONS_CACHE.exists():
        pytest.skip("the meetings cache or the duration cache is not on this machine")
    return ORD.plan(durations=ORD.load_durations())


def test_the_measured_frame_is_15_meetings_in_7_cities(real_plan):
    s = real_plan["summary"]
    assert s["n_released_meetings"] == 151
    assert s["n_eligible_meetings"] == 15
    assert s["meeting_drops"] == {"trained_on": 0, "excluded_meeting": 75,
                                  "excluded_city": 61}
    assert s["meetings_by_city"] == {"athens": 4, "chalandri": 4, "chania": 1,
                                     "papagos-cholargos": 1, "thira": 1,
                                     "vrilissia": 2, "zografou": 2}


def test_the_real_plan_is_45_clips_three_from_every_meeting(real_plan):
    s = real_plan["summary"]
    assert s["n_clips"] == 45
    assert s["n_meeting_clusters"] == 15
    assert s["v1_meeting_clusters"] == 67
    counts = collections.Counter((r["city"], r["meeting_id"])
                                 for r in real_plan["clips"])
    assert set(counts.values()) == {3}
    assert s["clips_by_city"] == {"athens": 12, "chalandri": 12, "chania": 3,
                                  "papagos-cholargos": 3, "thira": 3,
                                  "vrilissia": 6, "zografou": 6}


def test_the_real_plan_carries_no_published_text(real_plan):
    blob = json.dumps(real_plan["clips"], ensure_ascii=False)
    for banned in ("transcript", "subjects", "speakerTags", "utterance"):
        assert banned not in blob, banned


def test_the_real_plan_is_reproducible(real_plan):
    again = ORD.plan(durations=ORD.load_durations())
    assert again["clips"] == real_plan["clips"]
    assert again["summary"]["frame_sha256"] == real_plan["summary"]["frame_sha256"]
