from eval.controlled_eval.score_training_listening_audit import score


HIDDEN = [
    {"review_id": "t1", "kind": "training_clip", "dimension": "adjustment", "stratum": ">1"},
    {"review_id": "t2", "kind": "training_clip", "dimension": "adjustment", "stratum": ">1"},
    {"review_id": "w1", "kind": "validation_window", "window_id": "private-1"},
    {"review_id": "w2", "kind": "validation_window", "window_id": "private-2"},
]


def test_incomplete_answers_do_not_force_decision():
    result = score({"t1": {"label_faithful": "yes", "boundaries_usable": "yes"}}, HIDDEN)
    assert result["status"] == "incomplete"
    assert "unresolved" in result["frozen_interpretation"]


def test_complete_answers_route_defects_and_omitted_reference():
    answers = {
        "t1": {"label_faithful": "yes", "boundaries_usable": "yes", "notes": "private"},
        "t2": {"label_faithful": "no", "boundaries_usable": "yes"},
        "w1": {"reference_completeness": "complete"},
        "w2": {"reference_completeness": "minor_omission"},
    }
    result = score(answers, HIDDEN)
    assert result["status"] == "complete"
    assert result["training"]["overall"]["n_routed_to_protected_correction"] == 1
    assert "re-reference" in result["frozen_interpretation"]
    assert "private" not in str(result)
    assert "private-1" not in str(result)
