"""Tests for the blind adjudication of tempo-arm disagreements.

The load-bearing property is that the judge cannot tell which arm produced a word. If
he could, the adjudication would measure his expectations rather than the audio, and the
whole point of the pilot's descriptive-unmatched-words rule would be lost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import adjudicate_tempo as AJ  # noqa: E402
from eval.controlled_eval import tempo_probe as TP  # noqa: E402

BLOCKS = [
    {"id": "p1", "s": 1.0, "e": 4.0, "text": "να πω κάτι", "ov_with": ["p2"]},
    {"id": "p2", "s": 3.0, "e": 6.0, "text": "όχι τώρα", "ov_with": ["p1"]},
]


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """A fake token cache: control hears the gold words, t085 also hears one more."""
    monkeypatch.setattr(TP, "OUT", tmp_path)
    monkeypatch.setattr(AJ, "OUT", tmp_path)
    monkeypatch.setattr(AJ, "EPISODE_CLIPS", tmp_path / "episode-audio")

    def write(cell, arm, cycle, words):
        p = tmp_path / "tokens" / f"{cell}__{arm}__c{cycle}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        # Shaped like the wire: pieces carry a confidence, and a word boundary is a
        # LEADING SPACE. Without the space `group_words` would glue the list into one
        # long word, which is the same class of bug this fixture exists to guard.
        p.write_text(json.dumps({
            "tempo": TP.ARMS[arm].tempo,
            "tokens": [{"text": ("" if i == 0 else " ") + w,
                        "start_ms": 2000 + 100 * i, "end_ms": 2100 + 100 * i,
                        "is_final": True, "confidence": 0.9}
                       for i, w in enumerate(words)],
        }, ensure_ascii=False))

    for cyc in range(TP.CYCLES):
        write("c1", "proc10", cyc, ["να", "πω", "κάτι"])
        write("c1", "t085", cyc, ["να", "πω", "κάτι", "παρακαλώ"])
        write("c1", "t070", cyc, ["να", "πω", "κάτι"])
        write("c1", "raw", cyc, ["να", "πω", "κάτι"])
        write("c1", "pace070", cyc, ["να", "πω", "κάτι"])
    # a one-cycle apparition, which must NOT become a candidate
    write("c1", "t070", 0, ["να", "πω", "κάτι", "φάντασμα"])
    return {"c1": BLOCKS}


def test_a_word_only_the_slowed_arm_heard_becomes_a_candidate(probe):
    cands = AJ.candidates(probe)
    assert [(c["arm"], c["word"]) for c in cands] == [("t085", "παρακαλω")]


def test_a_word_heard_in_only_one_cycle_is_noise_not_a_candidate(probe):
    """Streaming output is not deterministic; adjudicating a one-off would spend the
    scarcest resource in this project on nothing."""
    assert all(c["word"] != "φαντασμα" for c in AJ.candidates(probe))


def test_a_word_the_human_already_wrote_is_not_adjudicated(probe):
    """Gold words are counted by the recall metric; only NEW words need a human."""
    assert all(c["word"] not in ("να", "πω", "κατι") for c in AJ.candidates(probe))


def test_the_page_never_names_the_arm(probe):
    items = AJ.page_items(AJ.candidates(probe))
    for it in items:
        assert set(it) == {"token", "word", "dur"}
        assert not any(a in json.dumps(it, ensure_ascii=False) for a in TP.ARMS)


def test_the_same_word_from_two_arms_is_asked_once(probe, monkeypatch):
    cands = AJ.candidates(probe)
    twice = cands + [{**cands[0], "arm": "t070"}]
    assert len(AJ.page_items(twice)) == 1
    assert AJ.tally(twice, {cands[0]["token"]: {"verdict": "supported"}}) == {
        "t085": {"supported": 1, "unsupported": 0, "cannot_tell": 0,
                 "judged": 1, "supported_rate": 1.0, "is_null_class": False},
        "t070": {"supported": 1, "unsupported": 0, "cannot_tell": 0,
                 "judged": 1, "supported_rate": 1.0, "is_null_class": False},
    }


def test_an_unknown_verdict_is_refused(probe):
    cands = AJ.candidates(probe)
    with pytest.raises(ValueError):
        AJ.save_verdict({}, cands, cands[0]["token"], "maybe", "T0")


def test_a_verdict_for_something_that_is_not_a_candidate_is_refused(probe):
    with pytest.raises(ValueError):
        AJ.save_verdict({}, AJ.candidates(probe), "deadbeefdeadbeef",
                        "supported", "T0")


def test_cannot_tell_does_not_count_in_the_supported_rate(probe):
    cands = AJ.candidates(probe)
    t = AJ.tally(cands, {cands[0]["token"]: {"verdict": "cannot_tell"}})
    assert t["t085"]["judged"] == 1 and t["t085"]["supported_rate"] is None


def test_the_control_disagreeing_with_itself_supplies_the_null_class(probe, tmp_path):
    """Same audio, same settings, different session: these words are noise by
    construction. Without them a supported-rate has no yardstick."""
    p = tmp_path / "tokens" / "c1__proc10__c1.json"
    p.write_text(json.dumps({
        "tempo": 1.0,
        "tokens": [{"text": ("" if i == 0 else " ") + w, "start_ms": 2000 + 100 * i,
                    "end_ms": 2100 + 100 * i, "is_final": True, "confidence": 0.9}
                   for i, w in enumerate(["να", "πω", "κάτι", "ψίθυρος"])]},
        ensure_ascii=False))
    churn = AJ.control_churn_candidates(probe)
    assert [c["word"] for c in churn] == ["ψιθυροσ"]
    assert all(c["arm"] == AJ.CHURN_CLASS for c in churn)


def test_the_null_class_is_invisible_on_the_page(probe, tmp_path):
    (tmp_path / "tokens" / "c1__proc10__c1.json").write_text(json.dumps({
        "tempo": 1.0,
        "tokens": [{"text": ("" if i == 0 else " ") + w, "start_ms": 2000 + 100 * i,
                    "end_ms": 2100 + 100 * i, "is_final": True, "confidence": 0.9}
                   for i, w in enumerate(["να", "πω", "κάτι", "ψίθυρος"])]},
        ensure_ascii=False))
    items = AJ.page_items(AJ.candidates(probe) + AJ.control_churn_candidates(probe))
    assert all(AJ.CHURN_CLASS not in json.dumps(it, ensure_ascii=False) for it in items)
    assert len(items) == 2


def test_the_tally_marks_which_row_is_the_null_class(probe, tmp_path):
    (tmp_path / "tokens" / "c1__proc10__c1.json").write_text(json.dumps({
        "tempo": 1.0,
        "tokens": [{"text": ("" if i == 0 else " ") + w, "start_ms": 2000 + 100 * i,
                    "end_ms": 2100 + 100 * i, "is_final": True, "confidence": 0.9}
                   for i, w in enumerate(["να", "πω", "κάτι", "ψίθυρος"])]},
        ensure_ascii=False))
    cands = AJ.candidates(probe) + AJ.control_churn_candidates(probe)
    v = {c["token"]: {"verdict": "supported"} for c in cands}
    t = AJ.tally(cands, v)
    assert t[AJ.CHURN_CLASS]["is_null_class"] is True
    assert t["t085"]["is_null_class"] is False
