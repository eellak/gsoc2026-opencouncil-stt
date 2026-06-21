"""Self-test #1 — chain integrity.

Synthetic CSV with task-only, user-only, task+user, and mixed/semantic chains.
Reconstruction must yield correct first-before / final-after, preserve
edited_by tags, and bucket mixed/semantic per policy.
"""
import pandas as pd

from eval.chains import reconstruct_chains


def _row(edit_id, utterance_id, ts, before, after, by, meeting="m1", city="c1"):
    return {
        "edit_id": edit_id,
        "utterance_id": utterance_id,
        "edit_timestamp": ts,
        "edit_updated_at": ts,
        "before_text": before,
        "after_text": after,
        "edited_by": by,
        "utterance_start": 0.0,
        "utterance_end": 1.0,
        "audio_url": "",
        "youtube_url": "",
        "meeting_name": meeting,
        "meeting_date": "2026-01-01",
        "meeting_id": meeting,
        "city_id": city,
    }


def _build_df():
    rows = [
        # task-only pure correction (accent fix)
        _row("e1", "u_task", 1, "ο δημος", "ο δήμος", "task"),
        # user-only pure correction (name fix)
        _row("e2", "u_user", 1, "Ρημάκης μίλησε", "Δημάκης μίλησε", "user"),
        # task+user chain: task fixes accent, user fixes name; sequential (before==prev after)
        _row("e3", "u_both", 1, "ο ρημακης", "ο Ρημάκης", "task"),
        _row("e4", "u_both", 2, "ο Ρημάκης", "ο Δημάκης", "user"),
        # mixed chain: correction + added content
        _row("e5", "u_mixed", 1, "το εργο θα γινη", "Το έργο θα γίνει σύντομα", "user"),
        # semantic rewrite: meaning changed
        _row("e6", "u_sem", 1, "πιστεύω ότι είναι σωστό", "συμφωνώ απόλυτα", "user"),
    ]
    return pd.DataFrame(rows)


def _by_id(chains):
    return {c["utterance_id"]: c for c in chains}


def test_first_before_and_final_after():
    chains = _by_id(reconstruct_chains(_build_df()))
    both = chains["u_both"]
    assert both["input_raw"] == "ο ρημακης"
    assert both["gold_final"] == "ο Δημάκης"


def test_edited_by_sequence_and_flags():
    chains = _by_id(reconstruct_chains(_build_df()))
    both = chains["u_both"]
    assert both["edited_by_seq"] == ["task", "user"]
    assert both["has_task"] is True
    assert both["has_user"] is True
    assert both["task_then_user"] is True

    task_only = chains["u_task"]
    assert task_only["edited_by_seq"] == ["task"]
    assert task_only["has_user"] is False
    assert task_only["task_then_user"] is False

    user_only = chains["u_user"]
    assert user_only["has_task"] is False
    assert user_only["has_user"] is True


def test_chain_type_classification():
    chains = _by_id(reconstruct_chains(_build_df()))
    assert chains["u_task"]["chain_type"] == "pure_correction"
    assert chains["u_user"]["chain_type"] == "pure_correction"
    assert chains["u_both"]["chain_type"] == "pure_correction"
    assert chains["u_mixed"]["chain_type"] == "mixed"
    assert chains["u_sem"]["chain_type"] == "semantic_rewrite"


def test_chain_link_integrity_flag():
    # The task+user chain has before==prev after at every link; the others are
    # single-link. reconstruct_chains should mark link integrity.
    chains = _by_id(reconstruct_chains(_build_df()))
    assert chains["u_both"]["links_ok"] is True
