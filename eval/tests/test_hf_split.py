"""Tests for the seeded speaker split."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.hf_export.split import SplitError, assign_splits


def _row(uid, city, spk, dur):
    return {"utterance_id": uid, "city_id": city, "speaker_id": spk,
            "duration_s": float(dur)}


def test_heldout_cities_always_validation():
    rows = [_row("a", "orestiada", "s1", 10), _row("b", "argos", None, 10),
            _row("c", "athens", "s2", 80)]
    res = assign_splits(rows, target_frac=0.20, window=(0.10, 0.30), seed=1)
    assert res.splits[0] == "validation"
    assert res.splits[1] == "validation"
    assert res.splits[2] == "train"


def test_speakers_added_until_target_and_disjoint():
    # held-out = 10s of 200s total (5%) -> needs ~30s more from speakers
    rows = [_row("v", "argos", "sv", 10)]
    rows += [_row(f"a{i}", "athens", "spkA", 10) for i in range(4)]   # 40s
    rows += [_row(f"b{i}", "sparta", "spkB", 10) for i in range(4)]   # 40s
    rows += [_row(f"c{i}", "chania", "spkC", 11) for i in range(10)]  # 110s
    res = assign_splits(rows, target_frac=0.20, window=(0.18, 0.45), seed=7,
                        floor_s=30.0)  # fixture speakers are 40-110s
    val_spk = {rows[i]["speaker_id"] for i, s in enumerate(res.splits)
               if s == "validation"} - {"sv"}
    train_spk = {rows[i]["speaker_id"] for i, s in enumerate(res.splits)
                 if s == "train"}
    assert val_spk, "at least one train-city speaker moved to validation"
    assert not (val_spk & train_spk), "speaker-disjoint violated"
    assert res.val_share >= 0.18


def test_floor_excludes_short_speakers():
    rows = [_row("v", "argos", "sv", 100)]
    rows += [_row("a", "athens", "tiny", 5)]           # below 180s floor
    rows += [_row(f"c{i}", "chania", "big", 60) for i in range(6)]  # 360s
    res = assign_splits(rows, target_frac=0.20, window=(0.15, 0.35),
                        seed=1, floor_s=180.0)
    assert "tiny" not in res.val_speakers


def test_null_speaker_rows_stay_in_train():
    rows = [_row("v", "argos", "sv", 20),
            _row("n", "athens", None, 80)]
    res = assign_splits(rows, target_frac=0.20, window=(0.18, 0.22), seed=1)
    assert res.splits[1] == "train"


def test_same_seed_same_split_different_seed_may_differ():
    rows = [_row("v", "argos", "sv", 10)]
    rows += [_row(f"u{i}", "athens", f"s{i}", 30) for i in range(20)]
    # floor_s below the 30s fixture speakers so the seeded walk actually runs
    # (default 180s floor would exclude all of them -> share stays 1.6%)
    a = assign_splits(rows, target_frac=0.20, window=(0.10, 0.40), seed=42,
                      floor_s=10.0)
    b = assign_splits(rows, target_frac=0.20, window=(0.10, 0.40), seed=42,
                      floor_s=10.0)
    assert a.splits == b.splits and a.val_speakers == b.val_speakers


def test_out_of_window_raises():
    # held-out cities alone are 50% of hours -> way past 22% cap
    rows = [_row("v", "argos", "sv", 100), _row("t", "athens", "s1", 100)]
    with pytest.raises(SplitError):
        assign_splits(rows, target_frac=0.20, window=(0.18, 0.22), seed=1)
