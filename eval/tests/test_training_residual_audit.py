from eval.controlled_eval.training_residual_audit import adjustment_bin, categories, duration_bin


def test_duration_bins_are_half_open():
    assert [duration_bin(value) for value in (1.99, 2, 3.99, 4, 7.99, 8)] == [
        "<2", "2-4", "2-4", "4-8", "4-8", ">=8"]


def test_adjustment_bins_and_missing_values():
    base = {"start": 1.0, "end": 3.0, "start_adj": 1.0, "end_adj": 3.0}
    assert adjustment_bin(base) == "none"
    assert adjustment_bin({**base, "start_adj": 1.1}) == "<=0.25"
    assert adjustment_bin({**base, "start_adj": 1.3}) == "0.25-1"
    assert adjustment_bin({**base, "start_adj": 2.1}) == ">1"
    assert adjustment_bin({**base, "start_adj": float("nan")}) == "none"


def test_categories_accepts_vector_like_values():
    class Vector:
        def tolist(self):
            return ["name", "number"]

    assert categories(Vector()) == ["name", "number"]
    assert categories([]) == ["uncategorized"]
