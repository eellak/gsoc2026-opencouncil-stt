import json
from pathlib import Path

import pytest

from eval.controlled_eval.score_reference_repair import score, window_status


FINGERPRINT = "abc123"


def sources(published_a="μια δυο τρια", published_b="τεσσερα πεντε"):
    return {
        "source_fingerprint": FINGERPRINT,
        "windows": [
            {"review_id": "r_a", "window_id": "w_a", "published_tokens": published_a.split()},
            {"review_id": "r_b", "window_id": "w_b", "published_tokens": published_b.split()},
        ],
    }


def answer(text, **over):
    row = {
        "repaired_text": text,
        "listened_to_full_audio": True,
        "finalized": True,
        "source_fingerprint": FINGERPRINT,
        "t": 1,
    }
    row.update(over)
    return row


def test_unchanged_finalized_repair_needs_adjudication():
    status, reason = window_status(answer("μια δυο τρια"), ["μια", "δυο", "τρια"], FINGERPRINT)
    assert status == "needs_adjudication"
    assert "unchanged" in reason


def test_unfinalized_or_stale_or_unresolved_answers_are_incomplete():
    published = ["μια", "δυο", "τρια"]
    assert window_status(answer("μια δυο", finalized=False), published, FINGERPRINT)[0] == "incomplete"
    assert window_status(answer("μια δυο", listened_to_full_audio=False), published,
                         FINGERPRINT)[0] == "incomplete"
    assert window_status(answer("μια δυο", source_fingerprint="old"), published,
                         FINGERPRINT)[0] == "incomplete"
    assert window_status(answer("   "), published, FINGERPRINT)[0] == "incomplete"
    assert window_status(answer("μια δυο", t=None), published, FINGERPRINT)[0] == "incomplete"
    assert window_status({}, published, FINGERPRINT)[0] == "incomplete"


def test_uncertain_spans_are_counted_but_do_not_block():
    published = ["μια", "δυο", "τρια"]
    status, _ = window_status(answer("μια δυο [?] τεσσερα"), published, FINGERPRINT)
    assert status == "complete"
    rows = {row["window_id"]: row for row in score(
        {"r_a": answer("μια δυο [?] τεσσερα"), "r_b": answer("τεσσερα εξι")}, sources())["windows"]}
    assert rows["w_a"]["uncertain_spans"] == 1
    assert rows["w_a"]["repaired_tokens"] == 3          # [?] is not a token


def test_aggregate_counts_and_arithmetic_identity():
    answers = {
        "r_a": answer("μια δυο τρια τεσσερα"),      # one token added
        "r_b": answer("τεσσερα εξι"),               # one token substituted
    }
    result = score(answers, sources())
    rows = {row["window_id"]: row for row in result["windows"]}
    a, b = rows["w_a"], rows["w_b"]
    assert (a["published_tokens"], a["repaired_tokens"]) == (3, 4)
    assert (a["substitutions"], a["deletions"], a["insertions"]) == (0, 0, 1)
    assert a["net_token_delta"] == 1
    assert (b["substitutions"], b["deletions"], b["insertions"]) == (1, 0, 0)
    assert b["net_token_delta"] == 0
    for row in result["windows"]:
        assert row["repaired_tokens"] == row["published_tokens"] - row["deletions"] + row["insertions"]
    assert result["status"] == "complete"


def test_status_is_incomplete_until_both_windows_finalize():
    answers = {"r_a": answer("μια δυο τρια τεσσερα")}
    assert score(answers, sources())["status"] == "incomplete"


def test_aggregate_carries_no_transcript_or_notes():
    answers = {
        "r_a": answer("μια δυο τρια ΜΥΣΤΙΚΟ", notes="ιδιωτικη σημειωση"),
        "r_b": answer("τεσσερα πεντε εξι"),
    }
    blob = json.dumps(score(answers, sources()), ensure_ascii=False)
    assert "ΜΥΣΤΙΚΟ" not in blob
    assert "ιδιωτικη" not in blob
    assert "μια" not in blob
