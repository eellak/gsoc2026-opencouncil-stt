from eval.controlled_eval.dense_insertion_analysis import align, window_analysis
from eval.controlled_eval.exp_same_stack import sdi


def test_alignment_decomposition_matches_frozen_sdi():
    ref = ["α", "β", "γ"]
    hyp = ["α", "χ", "β", "δ"]
    operations = align(ref, hyp)
    got = tuple(sum(op["op"] == name for op in operations)
                for name in ("sub", "del", "ins"))
    assert got == sdi(ref, hyp)


def test_window_analysis_marks_neighbor_repetition_and_reference_occurrence():
    result = window_analysis(["α", "β"], [" α α β"])
    assert result["ins"] == 1
    assert result["insertions"] == [{
        "token": "α",
        "hyp_index": 0,
        "repeated_neighbor": True,
        "occurs_in_reference": True,
        "numeric": False,
    }]
