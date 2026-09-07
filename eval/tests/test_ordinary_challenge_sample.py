"""Tests for the challenge-sample cutter (wayfinder #59 design, #64 build).

Record: exp-2026-08-26-training-round-2.

This sample is selected BECAUSE the incumbent already failed there, so its risk is the
mirror image of the ordinary sample's: not that failure leaks in, but that its result
leaks OUT and gets quoted as "the model is better". The tests therefore concentrate on
the three things that keep the two apart -- separate ids, separate manifests, and a
scorer that refuses to decide on this one -- and on the capacity arithmetic, because a
quota that cannot be met silently changes the frozen city mixture.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import cut_challenge_sample as CHL  # noqa: E402
from eval.controlled_eval import cut_ordinary_sample as ORD  # noqa: E402
from eval.controlled_eval.cut_challenge_sample import (  # noqa: E402
    ExclusionViolation, GeometryError, QuotaError, challenge_clip_id, check_sample,
    eligible, sample_row, select)
from eval.controlled_eval.exclusion_manifest import Manifest  # noqa: E402


# --------------------------------------------------------------------- fixtures

def queue_row(city="athens", meeting="mar23_2026", start_ms=97000, dur=9.0):
    return {
        "clip_id": f"{city}__{meeting}__{start_ms}",
        "city": city, "meeting_id": meeting,
        "meeting_date": "2026-03-23 15:00:00",
        "audio_url": f"https://data.opencouncil.gr/audio/{city}_{meeting}.mp3",
        "clip_start": start_ms / 1000, "clip_end": start_ms / 1000 + dur,
        "clip_dur": dur, "pool": "deletion_hard",
        "origin_row_id": f"row-{city}-{meeting}-{start_ms}",
        "source": "listen-2026-08", "witness": "soniox-stt-rt-v4",
        "reused_gap_verify_clip": True, "audio_sha256": "ab" * 32,
    }


def rows_for(spec):
    out = []
    for (city, meeting), n in spec.items():
        for i in range(n):
            out.append(queue_row(city=city, meeting=meeting, start_ms=10_000 * (i + 1)))
    return out


def manifest(meetings=(), cities=()):
    return Manifest(frozenset(meetings), frozenset(cities))


# ==========================================================================
# The two samples must be impossible to merge
# ==========================================================================

def test_the_two_samples_use_different_clip_id_prefixes():
    chl = challenge_clip_id(queue_row())
    ordn = ORD.clip_id(ORD.Meeting("athens", "mar23_2026", None, "https://x/y.mp3"),
                       97000)
    assert chl.startswith("chl__") and ordn.startswith("ord__")
    assert not CHL._CLIP_ID_RE.match(ordn)
    assert not ORD._CLIP_ID_RE.match(chl)


def test_the_two_samples_write_to_different_directories_and_result_files():
    assert CHL.OUT_DIR != ORD.OUT_DIR
    assert CHL.COUNTS_PATH != ORD.COUNTS_PATH
    assert CHL.SEED != ORD.SEED


def test_a_row_labelled_as_the_other_sample_is_refused():
    row = sample_row(queue_row())
    row["sample"] = "ordinary"
    with pytest.raises(ExclusionViolation, match="never summed"):
        check_sample([row], manifest(), {})


def test_the_challenge_row_keeps_the_failure_provenance_it_was_selected_by():
    """The ordinary sample must not carry this. This one must: it IS the selection."""
    row = sample_row(queue_row())
    assert row["sample"] == "challenge"
    assert row["selection"] == "incumbent-failed-here"
    assert row["pool"] == "deletion_hard"
    assert row["origin_row_id"].startswith("row-")
    assert row["origin_clip_id"] == "athens__mar23_2026__97000"
    for banned in ("text", "witness_text", "target"):
        assert banned not in row, banned


# ==========================================================================
# Eligibility and capacity
# ==========================================================================

def test_an_excluded_or_trained_on_meeting_supplies_nothing():
    rows = rows_for({("athens", "a_2026"): 2, ("sparta", "b_2026"): 2,
                     ("chania", "c_2026"): 2})
    kept = eligible(rows, manifest(cities=["sparta"]),
                    {("chania", "c_2026"): ["clean-pack-arm-2026-08"]})
    assert {(r["city"], r["meeting_id"]) for r in kept} == {("athens", "a_2026")}


def test_capacity_counts_what_each_meeting_actually_holds_not_two_times_meetings():
    """A meeting with one eligible clip supplies one. `2 x meetings` would say two, the
    deal would come up short, and the frozen city mixture would change silently."""
    rows = rows_for({("athens", "a_2026"): 1, ("athens", "b_2026"): 1})
    assert len(select(rows, {"athens": 2})) == 2
    with pytest.raises(QuotaError, match="at most 2"):
        select(rows, {"athens": 3})


def test_a_quota_over_the_two_per_meeting_cap_fails_loudly():
    rows = rows_for({("athens", "a_2026"): 9})
    with pytest.raises(QuotaError, match="at most 2"):
        select(rows, {"athens": 3})


def test_selection_spreads_over_meetings_before_taking_a_second_from_any():
    rows = rows_for({("chania", f"m{i}_2026"): 4 for i in range(5)})
    picked = select(rows, {"chania": 6})
    per = collections.Counter((r["city"], r["meeting_id"]) for r in picked)
    assert sorted(per.values()) == [1, 1, 1, 1, 2]


def test_selection_is_deterministic_and_independent_of_input_order():
    rows = rows_for({("chania", f"m{i}_2026"): 3 for i in range(6)})
    a = [r["clip_id"] for r in select(rows, {"chania": 8})]
    b = [r["clip_id"] for r in select(list(reversed(rows)), {"chania": 8})]
    assert sorted(a) == sorted(b)


def test_the_same_slug_in_two_cities_is_two_clusters():
    rows = rows_for({("athens", "mar23_2026"): 2, ("chania", "mar23_2026"): 2})
    picked = select(rows, {"athens": 1, "chania": 1})
    assert len({(r["city"], r["meeting_id"]) for r in picked}) == 2


def test_more_than_two_clips_from_one_meeting_is_refused_by_the_guard():
    rows = [sample_row(queue_row(start_ms=1000 * i)) for i in (1, 2, 3)]
    with pytest.raises(ExclusionViolation, match="cap is 2"):
        check_sample(rows, manifest(), {})


def test_a_queue_clip_id_that_is_not_path_safe_is_refused():
    bad = queue_row()
    bad["clip_id"] = "../../etc/passwd"
    with pytest.raises(GeometryError):
        challenge_clip_id(bad)


# ==========================================================================
# Against the real cache (skipped where it is absent)
# ==========================================================================

@pytest.fixture(scope="module")
def real_plan():
    if not CHL.QUEUE_PATH.exists():
        pytest.skip("the listening queue is not on this machine")
    return CHL.plan()


def test_the_real_plan_is_40_clips_in_the_frozen_city_mix(real_plan):
    s = real_plan["summary"]
    assert s["n_pool_clips"] == 449
    assert s["n_pool_meetings"] == 67
    assert s["n_clips"] == CHL.N_CLIPS == 40
    assert s["clips_by_city"] == {"athens": 9, "chalandri": 8, "chania": 20,
                                  "vrilissia": 3}
    assert s["city_quotas"] == s["clips_by_city"]
    assert s["decides"] is False


def test_the_real_plan_puts_every_clip_in_its_own_meeting(real_plan):
    per = collections.Counter((r["city"], r["meeting_id"])
                              for r in real_plan["clips"])
    assert max(per.values()) <= CHL.MAX_PER_MEETING
    assert len(per) == 40, "40 clusters is the most this sample can have"


def test_the_real_plan_is_reproducible(real_plan):
    assert CHL.plan()["clips"] == real_plan["clips"]


def test_the_repo_side_result_is_counts_and_hashes_only():
    p = ROOT / "eval" / "results_challenge_sample.json"
    if not p.exists():
        pytest.skip("sample not cut on this machine")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert "clips" not in d
    blob = json.dumps(d, ensure_ascii=False)
    assert "://" not in blob
    assert "__" not in blob
    assert d["decides"] is False
