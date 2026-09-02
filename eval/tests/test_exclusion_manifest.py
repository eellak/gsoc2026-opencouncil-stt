"""Tests for the exclusion manifest (spec section 4.2).

Nothing may be cut for the listening batch until this manifest exists and is hashed,
because a clip from an evaluation meeting is training on the test set and is not
reversible.

The load-bearing test in this file is `test_bench_windows_is_not_the_391_source`. The
spec used to name `~/.cache/oc-public/bench_windows` as the source of the 391 held-out
windows. It is not: that directory holds the 260 wavs of the 2026-06-10 / 2026-08-10
run and matches 0 of the 391. A builder written from the old line would have excluded
the wrong benchmark and let the 391's meetings straight into training.

Tests that need the caches skip when they are absent, so the file still runs on a
machine that has the repo but not the data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import exclusion_manifest as em  # noqa: E402
from eval.controlled_eval.exclusion_manifest import (  # noqa: E402
    SOURCES, Manifest, build_manifest, parse_gold_cell_id)

SC = Path.home() / ".cache" / "oc-public"


def _have(name: str) -> bool:
    return Path(SOURCES[name].path).exists()


@pytest.fixture(scope="module")
def manifest():
    missing = [n for n in SOURCES if SOURCES[n].kind != "city" and not _have(n)]
    if missing:
        pytest.skip(f"caches absent on this machine: {missing}")
    return build_manifest()


# ==========================================================================
# The defect this manifest exists to prevent
# ==========================================================================

def test_bench_windows_is_not_the_391_source():
    """The wav directory belongs to the 260-window run and matches none of the 391."""
    assert "bench_windows" not in str(SOURCES["held_out_391"].path), \
        "the 391 meeting set comes from the post-june report, not from a wav directory"

    wavs = SC / "bench_windows"
    report = Path(SOURCES["held_out_391"].path)
    if not (wavs.exists() and report.exists()):
        pytest.skip("caches absent on this machine")

    wav_ids = {p.stem for p in wavs.iterdir() if p.suffix == ".wav"}
    ids_391 = {it["itemId"] for it in json.loads(report.read_text())["items"]}
    assert wav_ids & ids_391 == set(), \
        "bench_windows must not be mistaken for the 391 held-out windows"


def test_the_391_report_contributes_117_meetings_over_11_cities(manifest):
    keys = manifest.sources["held_out_391"].keys
    assert len(keys) == 117
    assert len({city for city, _ in keys}) == 11


# ==========================================================================
# Key normalisation across four different id shapes
# ==========================================================================

def test_gold_cell_ids_parse_including_underscored_meeting_slugs():
    """Meeting slugs carry underscores and digits, so the offset is the last segment
    and everything between the city and it is the meeting."""
    assert parse_gold_cell_id("gc_chalandri_jul29_2026_2020000") == ("chalandri", "jul29_2026")
    assert parse_gold_cell_id("gc_argos_apr30_2_2026_1264428") == ("argos", "apr30_2_2026")
    for bad in ("chalandri_jul29_2026_2020000", "gc_argos_apr30_2_2026", "gc_argos"):
        with pytest.raises(ValueError):
            parse_gold_cell_id(bad)


def test_gold_set_resolves_to_its_six_meetings(manifest):
    assert manifest.sources["gold_set"].keys == frozenset({
        ("chalandri", "jul29_2026"), ("chania", "jun24_2026"),
        ("samothraki", "jul6_2026"), ("sparta", "jul1_2026"),
        ("xylokastro", "jun29_2026"), ("zografou", "jun18_2026"),
    })


def test_the_39_validation_windows_resolve_to_32_meetings(manifest):
    assert len(manifest.sources["validation_39"].keys) == 32


def test_the_canonical_key_is_city_and_meeting_never_meeting_alone():
    """Meeting slugs repeat across cities and collapse 117 real meetings into 66
    (exp_composition_postjune.py:212-216), so a meeting-only key silently over-excludes."""
    m = Manifest(
        meeting_keys=frozenset({("alpha", "jun1_2026"), ("beta", "jun1_2026")}),
        city_keys=frozenset(),
        sources={},
    )
    assert m.is_excluded("alpha", "jun1_2026")
    assert m.is_excluded("beta", "jun1_2026")
    assert not m.is_excluded("gamma", "jun1_2026")


def test_confirmation_cities_exclude_every_meeting_in_the_city(manifest):
    """That source is city-level with no meeting list, so any meeting id must match."""
    assert manifest.sources["confirm_cities"].kind == "city"
    for city in ("argos", "samothraki", "sparta", "xylokastro", "argithea"):
        assert manifest.is_excluded(city, "a_meeting_that_does_not_exist_2099")
    assert not manifest.is_excluded("athens", "a_meeting_that_does_not_exist_2099")


# ==========================================================================
# The calibration decision of 2026-08-27
# ==========================================================================

def test_the_260_window_calibration_set_is_excluded_from_clip_cutting(manifest):
    """USER DECISION 2026-08-27: the five-witness benchmark windows are
    calibration-only and never supply a training clip.

    203 meetings, less the 33 released by the amendment of 2026-08-28.
    """
    src = manifest.sources["calibration_260"]
    assert src.kind == "meeting"
    assert len(src.keys) == 170
    assert src.keys <= manifest.meeting_keys


# ==========================================================================
# The amendment of 2026-08-28: two cities released, at a priced cost
# ==========================================================================

def test_the_released_cities_are_absent_from_the_calibration_source(manifest):
    """The release is the whole mechanism, so it is asserted rather than assumed."""
    assert em.CALIBRATION_RELEASED_CITIES == ("chalandri", "vrilissia")
    released = {k for k in manifest.sources["calibration_260"].keys
                if k[0] in em.CALIBRATION_RELEASED_CITIES}
    assert released == set()


def test_a_city_that_was_not_released_stays_fully_excluded(manifest):
    """Zografou was measured as the worst clips-per-window trade and was NOT released.

    Without this, a later edit that widens the tuple looks like a no-op in the tests.
    """
    kept = {k for k in manifest.sources["calibration_260"].keys if k[0] == "zografou"}
    assert len(kept) >= 10
    assert kept <= manifest.meeting_keys


def test_a_released_meeting_named_by_another_source_stays_excluded(manifest):
    """The release drops keys from ONE source; the union still decides.

    Two of the 33 released meetings are also held-out or gold meetings. If the
    release leaked past the union, those two would become eligible for clip cutting
    and the held-out set would be contaminated.
    """
    others = set()
    for name, s in manifest.sources.items():
        if name != "calibration_260" and s.kind == "meeting":
            others |= set(s.keys)
    overlap = {k for k in others if k[0] in em.CALIBRATION_RELEASED_CITIES}
    assert overlap, "fixture drift: no released city appears in another source"
    for k in overlap:
        assert manifest.is_excluded(*k)


def test_the_release_is_part_of_the_hashed_payload(manifest):
    """A silent edit of the release list must move the hash, not just the code."""
    payload = json.loads(manifest.to_json())
    assert payload["calibration_released_cities"] == ["chalandri", "vrilissia"]
    widened = Manifest(
        meeting_keys=manifest.meeting_keys,
        city_keys=manifest.city_keys,
        released_cities=frozenset(em.CALIBRATION_RELEASED_CITIES) | {"zografou"},
    )
    same = Manifest(meeting_keys=manifest.meeting_keys, city_keys=manifest.city_keys,
                    released_cities=frozenset(em.CALIBRATION_RELEASED_CITIES))
    assert widened.sha256 != same.sha256


def test_the_release_price_in_evaluation_windows_is_asserted():
    """The cost is 38 of the 137 five-witness windows and 36 of the 247 fusion
    substrate windows. It is asserted here so that widening the release cannot be
    done without restating the price."""
    bench = Path(SOURCES["calibration_260"].path)
    substrate = SC / "composition-rt-2026-08" / "manifest.json"
    if not bench.exists() or not substrate.exists():
        pytest.skip("benchmark or substrate cache absent on this machine")
    rel = set(em.CALIBRATION_RELEASED_CITIES)
    windows = json.loads(substrate.read_text())["windows"]
    assert sum(1 for w in windows if w["city"] in rel) == 36
    assert len(windows) == 247

    m = build_manifest()
    other = set()
    for name, s in m.sources.items():
        if name not in ("calibration_260", "confirm_cities"):
            other |= set(s.keys)
    items = json.loads(bench.read_text())["items"]
    five_witness = [w for w in items
                    if w["cityId"] not in m.city_keys
                    and (w["cityId"], w["meetingId"]) not in other]
    assert len(five_witness) == 137
    assert sum(1 for w in five_witness if w["cityId"] in rel) == 38


# ==========================================================================
# The hash is what goes in the ledger record
# ==========================================================================

def test_manifest_is_deterministic_and_sorted(manifest):
    again = build_manifest()
    assert again.sha256 == manifest.sha256
    payload = json.loads(manifest.to_json())
    assert payload["meeting_keys"] == sorted(payload["meeting_keys"])
    assert payload["city_keys"] == sorted(payload["city_keys"])


def test_hash_changes_when_a_key_changes(manifest):
    mutated = Manifest(
        meeting_keys=manifest.meeting_keys | {("nowhere", "jan1_2099")},
        city_keys=manifest.city_keys,
        sources=manifest.sources,
    )
    assert mutated.sha256 != manifest.sha256


def test_every_source_is_attributed_in_the_payload(manifest):
    payload = json.loads(manifest.to_json())
    assert set(payload["sources"]) == set(SOURCES)
    for name, src in payload["sources"].items():
        assert src["path"] and src["kind"] in ("meeting", "city")
        assert src["n_keys"] >= 1


def test_a_meeting_outside_every_source_is_not_excluded(manifest):
    assert not manifest.is_excluded("athens", "jan1_2099")


# ==========================================================================
# Source audio hashes (section 4.2: "canonical meeting IDs AND source audio hashes")
# ==========================================================================

def test_audio_hashing_is_off_by_default(manifest):
    """Kept off for tests and quick checks; the operational build turns it on."""
    assert manifest.audio_sha256 == {}
    assert manifest.audio_missing == frozenset()


def test_audio_hashing_reports_missing_files_instead_of_skipping(tmp_path):
    """A meeting with no local recording cannot be checked for renaming, so it is
    named rather than silently absent from the map."""
    from eval.controlled_eval.exclusion_manifest import hash_audio

    (tmp_path / "alpha__jun1_2026.mp3").write_bytes(b"audio-bytes")
    keys = {("alpha", "jun1_2026"), ("beta", "jul2_2026")}
    hashes, missing = hash_audio(keys, audio_dir=tmp_path)

    assert set(hashes) == {("alpha", "jun1_2026")}
    assert missing == frozenset({("beta", "jul2_2026")})
    assert hashes[("alpha", "jun1_2026")] == \
        __import__("hashlib").sha256(b"audio-bytes").hexdigest()

    again, _ = hash_audio(keys, audio_dir=tmp_path)
    assert again == hashes


def test_audio_hashes_enter_the_manifest_hash(tmp_path):
    """The point of hashing the audio is that a renamed or swapped recording changes
    the manifest sha256; if the hashes sat outside the payload it would not."""
    base = Manifest(meeting_keys=frozenset({("alpha", "jun1_2026")}),
                    city_keys=frozenset(), sources={})
    with_audio = Manifest(meeting_keys=base.meeting_keys, city_keys=frozenset(),
                          sources={}, audio_sha256={("alpha", "jun1_2026"): "deadbeef"})
    swapped = Manifest(meeting_keys=base.meeting_keys, city_keys=frozenset(),
                       sources={}, audio_sha256={("alpha", "jun1_2026"): "cafebabe"})
    assert len({base.sha256, with_audio.sha256, swapped.sha256}) == 3


# ------------------------------------------------------------------ loading it

def test_load_manifest_rejects_a_file_whose_hash_moved(tmp_path):
    """A drifted manifest is not the manifest the ledger record vouches for. The
    loader must refuse it rather than quietly excluding a different set of meetings."""
    p = tmp_path / "manifest.json"
    p.write_text('{"meeting_keys": [], "city_keys": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        em.load_manifest(p, expect_sha="0" * 64, cross_check=False)


def test_load_manifest_round_trips_the_keys(tmp_path, manifest):
    built = manifest
    p = tmp_path / "manifest.json"
    p.write_text(built.to_json(), encoding="utf-8")
    got = em.load_manifest(p, expect_sha=built.sha256, cross_check=False)
    assert got.meeting_keys == built.meeting_keys
    assert got.city_keys == built.city_keys


def test_the_frozen_manifest_on_disk_still_verifies():
    """The operational check: the file recorded in exp-2026-08-26-training-round-2
    must still hash to its recorded sha AND still agree with the live sources."""
    if not em.MANIFEST_PATH.exists():
        pytest.skip("manifest not built on this machine")
    man = em.load_manifest()
    assert len(man.meeting_keys) == 283   # 314 before the 2026-08-28 release
    assert len(man.city_keys) == 5
    assert man.released_cities == frozenset(em.CALIBRATION_RELEASED_CITIES)


def test_a_loaded_manifest_refuses_to_rehash_itself(tmp_path):
    """CodeRabbit 2026-08-28: a manifest read from disk cannot rebuild its sources or
    audio hashes, so `sha256` would return a digest matching no file. Recording that
    in the ledger is exactly the failure the hash exists to prevent."""
    built = build_manifest()
    p = tmp_path / "manifest.json"
    p.write_text(built.to_json(), encoding="utf-8")
    got = em.load_manifest(p, expect_sha=built.sha256, cross_check=False)
    assert got.loaded_sha256 == built.sha256
    with pytest.raises(ValueError, match="read from disk"):
        got.sha256
