"""Tests for the v2 training-WER harness. Written while the arms were still decoding."""
from __future__ import annotations

import pytest

from eval.controlled_eval import train_wer_v2 as T


def make_rows(specs: list[tuple[str, str, str, int, float, float]]) -> list[dict]:
    """(key, city, meeting, n_utterances, speech_sec, span_sec) -> harness rows."""
    return [
        {
            "key": key, "city_id": city, "meeting_id": meeting,
            "person_id": f"p-{key}", "n_utterances": turns,
            "speech_sec": speech, "span_sec": span,
            "clip": f"/nowhere/{key}.wav", "reference": "x",
        }
        for key, city, meeting, turns, speech, span in specs
    ]


# ---------------------------------------------------------------------- slices

def test_n_utterances_bins_are_the_frozen_boundaries():
    rows = make_rows([
        ("a", "athens", "m1", 1, 10.0, 12.0),
        ("b", "athens", "m1", 5, 10.0, 12.0),
        ("c", "athens", "m1", 6, 10.0, 12.0),
        ("d", "athens", "m1", 9, 10.0, 12.0),
        ("e", "athens", "m1", 10, 10.0, 12.0),
    ])
    bins = T.slices(rows)["n_utterances"]
    assert bins["1-5"] == ["a", "b"]
    assert bins["6-9"] == ["c", "d"]
    assert bins["10+"] == ["e"]


def test_occupancy_terciles_order_by_speech_over_span_not_by_duration():
    # b has the most speech but the worst occupancy; the ordering must follow the
    # ratio, otherwise the axis silently becomes "long packs" again.
    rows = make_rows([
        ("a", "athens", "m1", 3, 9.0, 10.0),    # 0.90
        ("b", "athens", "m1", 3, 20.0, 40.0),   # 0.50
        ("c", "athens", "m1", 3, 7.0, 10.0),    # 0.70
    ])
    terciles = T.slices(rows)["occupancy_tercile"]
    assert terciles["t1"] == ["b"]
    assert terciles["t2"] == ["c"]
    assert terciles["t3"] == ["a"]


def test_every_row_lands_in_exactly_one_bucket_per_axis():
    rows = make_rows([
        (f"k{i}", "athens" if i % 2 else "chania", f"m{i % 3}", i % 12 + 1,
         float(i + 1), float(i + 2))
        for i in range(12)
    ])
    for axis, groups in T.slices(rows).items():
        placed = [key for members in groups.values() for key in members]
        assert sorted(placed) == sorted(row["key"] for row in rows), axis
        assert len(placed) == len(set(placed)), axis


# ------------------------------------------------------------------- dominance

def test_contrast_dominance_separates_net_share_from_positive_share():
    """One row helps hugely, another hurts, and the net is small.

    The absolute-over-net share degenerates here: it reports 3.0, which reads as
    "one row is 300% of the effect". The positive share is the honest statement.
    """
    rows = make_rows([
        ("good", "athens", "m1", 3, 10.0, 12.0),
        ("bad", "athens", "m2", 3, 10.0, 12.0),
        ("flat", "chania", "m3", 3, 10.0, 12.0),
    ])
    base = {"good": (9, 0, 0, 10), "bad": (0, 0, 0, 10), "flat": (2, 0, 0, 10)}
    v2 = {"good": (0, 0, 0, 10), "bad": (6, 0, 0, 10), "flat": (2, 0, 0, 10)}
    out = T.contrast_dominance(base, v2, rows)

    assert out["net_error_difference"] == 3
    assert out["rows_better"] == 1 and out["rows_worse"] == 1 and out["rows_tied"] == 1
    assert out["largest_row_key"] == "good"
    assert out["largest_row_share_of_net"] == pytest.approx(3.0)
    assert out["largest_gaining_row_key"] == "good"
    assert out["largest_gaining_row_share_of_positive"] == pytest.approx(1.0)


def test_leave_one_out_reports_the_sign_reversal_it_finds():
    rows = make_rows([
        ("a", "athens", "m1", 3, 10.0, 12.0),
        ("b", "chania", "m2", 3, 10.0, 12.0),
    ])
    # Dropping "a" flips the net from +1 to -4, so the effect is one row.
    base = {"a": (5, 0, 0, 10), "b": (0, 0, 0, 10)}
    v2 = {"a": (0, 0, 0, 10), "b": (4, 0, 0, 10)}
    out = T.contrast_dominance(base, v2, rows)
    assert out["net_error_difference"] == 1
    assert out["leave_one_row_out"]["sign_reversal"] is True
    assert out["leave_one_city_out"]["sign_reversal"] is True


def test_dominance_survives_a_zero_net_contrast():
    rows = make_rows([("a", "athens", "m1", 3, 10.0, 12.0)])
    counts = {"a": (3, 0, 0, 10)}
    out = T.contrast_dominance(counts, counts, rows)
    assert out["net_error_difference"] == 0
    assert out["largest_row_share_of_net"] is None
    assert out["leave_one_row_out"]["sign_reversal"] is None


# ----------------------------------------------------------------------- rates

def test_rates_sum_the_three_components_into_wer():
    counts = {"a": (2, 1, 3, 10), "b": (0, 0, 1, 10)}
    out = T.rates(counts)
    assert out["ref_tokens"] == 20 and out["n_rows"] == 2
    assert out["wer"] == pytest.approx(7 / 20)
    assert out["sub_rate"] + out["del_rate"] + out["ins_rate"] == pytest.approx(out["wer"])


def test_rates_on_an_empty_slice_does_not_divide_by_zero():
    assert T.rates({})["wer"] is None


# ------------------------------------------------------------------ guardrails

def test_the_over_cap_pack_is_named_and_excluded_from_the_training_population():
    assert T.EXCLUDED_OVER_CAP == "31cba9e6d6dc28a7"
    assert T.EXECUTED_TRAINING_ROWS == 2475


def test_the_base_arm_is_the_local_build_not_the_systran_conversion():
    """Pairing a locally converted v2 against someone else's conversion of the base
    would measure the converter, not the adapter."""
    assert T.ARMS["base"]["model"].name == "ct2-base"
    assert T.ARMS["base"]["model_sha256_16"] == "bba445638b80555f"
    assert "huggingface" not in str(T.ARMS["base"]["model"])


# ------------------------------------- regressions from the 2026-08-25 Codex review

def test_largest_gaining_row_is_chosen_among_rows_that_helped():
    """Contributions +5, +5, -9: the abs-largest row is the one that HURT."""
    rows = make_rows([
        ("up1", "athens", "m1", 3, 10.0, 12.0),
        ("up2", "athens", "m2", 3, 10.0, 12.0),
        ("down", "chania", "m3", 3, 10.0, 12.0),
    ])
    base = {"up1": (5, 0, 0, 10), "up2": (5, 0, 0, 10), "down": (0, 0, 0, 10)}
    v2 = {"up1": (0, 0, 0, 10), "up2": (0, 0, 0, 10), "down": (9, 0, 0, 10)}
    out = T.contrast_dominance(base, v2, rows)

    assert out["net_error_difference"] == 1
    assert out["largest_row_key"] == "down"          # abs-largest, and it hurt
    assert out["largest_gaining_row_key"] in {"up1", "up2"}
    assert out["largest_gaining_row_errors"] == 5
    assert out["largest_gaining_row_share_of_positive"] == pytest.approx(0.5)


def test_sign_reversal_is_judged_the_same_way_for_positive_and_negative_nets():
    """Dropping a row that leaves exactly zero is a lost sign, not a reversal."""
    rows = make_rows([
        ("a", "athens", "m1", 3, 10.0, 12.0),
        ("b", "chania", "m2", 3, 10.0, 12.0),
    ])
    # net = +3, and dropping "a" leaves exactly 0.
    positive = T.contrast_dominance(
        {"a": (3, 0, 0, 10), "b": (0, 0, 0, 10)},
        {"a": (0, 0, 0, 10), "b": (0, 0, 0, 10)}, rows)
    # The mirror image: net = -3, dropping "a" also leaves exactly 0.
    negative = T.contrast_dominance(
        {"a": (0, 0, 0, 10), "b": (0, 0, 0, 10)},
        {"a": (3, 0, 0, 10), "b": (0, 0, 0, 10)}, rows)

    assert positive["net_error_difference"] == 3
    assert negative["net_error_difference"] == -3
    for out in (positive, negative):
        assert out["leave_one_row_out"]["sign_reversal"] is False
        assert out["leave_one_row_out"]["sign_lost"] is True


def test_check_stack_rejects_arms_decoded_with_different_thread_counts():
    good = {"config": T.DA.CONTROL, "model_sha256": "0c6976f120f12f7cdeadbeef",
            "cpu_threads": 4, "device": "cpu", "compute_type": "int8"}
    other = dict(good, model_sha256="bba445638b80555fdeadbeef", cpu_threads=16)
    with pytest.raises(SystemExit, match="cpu_threads"):
        T.check_stack({"v2": good, "base": other})


def test_check_stack_rejects_an_arm_decoded_with_the_wrong_model():
    wrong = {"config": T.DA.CONTROL, "model_sha256": "ffffffffffffffffdeadbeef",
             "cpu_threads": 4, "device": "cpu", "compute_type": "int8"}
    with pytest.raises(SystemExit, match="the record names"):
        T.check_stack({"v2": wrong})


def test_check_stack_accepts_one_matched_pair():
    v2 = {"config": T.DA.CONTROL, "model_sha256": "0c6976f120f12f7cdeadbeef",
          "cpu_threads": 4, "device": "cpu", "compute_type": "int8"}
    base = dict(v2, model_sha256="bba445638b80555fdeadbeef")
    assert T.check_stack({"v2": v2, "base": base}) == {
        "cpu_threads": 4, "device": "cpu", "compute_type": "int8"}
