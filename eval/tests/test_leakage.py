"""Self-test #2 — leakage control.

Two-fold split with a glossary term that appears ONLY in the held-out fold.
The glossary builder (train-only) must exclude it, and an eval prompt's
glossary block must provably never contain it.
"""
from eval.glossary import mine_glossary, build_glossary_block
from eval.splits import split_by_meeting


def _chain(uid, meeting, city, before, after, by="user"):
    return {
        "utterance_id": uid,
        "meeting_id": meeting,
        "city_id": city,
        "input_raw": before,
        "gold_final": after,
        "edited_by_seq": [by],
        "has_task": by == "task",
        "has_user": by == "user",
        "task_then_user": False,
        "chain_type": "pure_correction",
        "links_ok": True,
    }


def _corpus():
    # ΚΕΔΕ appears across two TRAIN meetings (mineable, global).
    # ΔΕΥΑΞ appears ONLY in the held-out EVAL meeting (must never be mined).
    return [
        _chain("u1", "m_train_a", "c1", "η κεδε αποφασισε", "η ΚΕΔΕ αποφάσισε"),
        _chain("u2", "m_train_b", "c1", "συμφωνα με την κεδε", "σύμφωνα με την ΚΕΔΕ"),
        _chain("u3", "m_train_a", "c1", "ο δημος", "ο δήμος"),
        _chain("u4", "m_eval", "c1", "η δευαξ ειπε", "η ΔΕΥΑΞ είπε"),
    ]


def test_split_partitions_by_meeting():
    chains = _corpus()
    train, ev = split_by_meeting(chains, eval_meetings={"m_eval"})
    train_meetings = {c["meeting_id"] for c in train}
    eval_meetings = {c["meeting_id"] for c in ev}
    assert "m_eval" not in train_meetings
    assert eval_meetings == {"m_eval"}
    # no utterance appears in both folds
    assert not ({c["utterance_id"] for c in train} & {c["utterance_id"] for c in ev})


def test_heldout_term_not_mined():
    chains = _corpus()
    train, _ = split_by_meeting(chains, eval_meetings={"m_eval"})
    gloss = mine_glossary(train)
    all_terms = set(gloss["global"]) | {
        t for terms in gloss["per_city"].values() for t in terms
    }
    assert "ΚΕΔΕ" in all_terms          # mineable train term is present
    assert "ΔΕΥΑΞ" not in all_terms      # held-out-only term is excluded


def test_eval_prompt_never_contains_heldout_term():
    chains = _corpus()
    train, _ = split_by_meeting(chains, eval_meetings={"m_eval"})
    gloss = mine_glossary(train)
    block = build_glossary_block(gloss, city_id="c1")
    assert "ΔΕΥΑΞ" not in block
