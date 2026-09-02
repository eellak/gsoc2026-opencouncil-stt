from eval.controlled_eval.training_wer import (
    choose_sample,
    contrast_dominance,
    rates,
    row_rank,
    usable_span,
)


def test_sample_is_deterministic_and_outcome_blind():
    rows = [
        {"utterance_id": str(i), "text": f"outcome {10 - i}"}
        for i in range(10)
    ]
    expected = sorted(rows, key=lambda row: row_rank(row["utterance_id"]))[:4]
    assert choose_sample(rows, 4) == expected
    mutated = [{**row, "text": "changed"} for row in rows]
    assert [r["utterance_id"] for r in choose_sample(mutated, 4)] == [
        r["utterance_id"] for r in expected
    ]


def test_usable_span_prefers_adjusted_bounds_and_adds_training_pad():
    assert usable_span({"start": 2.0, "end": 3.0, "start_adj": 1.5,
                        "end_adj": 3.5}) == (1.3, 3.7)
    assert usable_span({"start": 0.1, "end": 1.0, "start_adj": None,
                        "end_adj": None}) == (0.0, 1.2)


def test_rates_and_contrast_dominance_report_all_components():
    fixed = {"a": (1, 2, 0, 10), "b": (0, 1, 1, 10)}
    july = {"a": (2, 3, 0, 10), "b": (0, 1, 2, 10)}
    total = rates(fixed)
    assert total["wer"] == 5 / 20
    assert total["sub_rate"] == 1 / 20
    assert total["del_rate"] == 3 / 20
    assert total["ins_rate"] == 1 / 20
    influence = contrast_dominance(july, fixed)
    assert influence["net_error_difference"] == 3
    assert influence["largest_row_error_difference"] == 2
    assert influence["leave_one_out_sign_reversal"] is False
