from eval.controlled_eval.seed_calibration import normal_approx_seed_count, rates


def test_rates_reports_all_error_components_and_length_ratio():
    result = rates({"a": (2, 1, 3, 10), "b": (1, 2, 0, 10)})
    assert result["wer"] == 9 / 20
    assert result["sub_rate"] == 3 / 20
    assert result["del_rate"] == 3 / 20
    assert result["ins_rate"] == 3 / 20
    assert result["recall_error_rate"] == 6 / 20
    assert result["hyp_ref_ratio"] == 1.0


def test_normal_approx_seed_count_matches_frozen_rule():
    assert normal_approx_seed_count(1.05) == 9
    assert normal_approx_seed_count(2.1) == 35
