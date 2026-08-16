"""Locks the frozen definitions of `exp-2026-08-16-soniox-confidence`.

These are not behaviour tests, they are freeze tests: if word grouping, the
confidence aggregates or the AUROC tie handling change, the 0.8167 in
`docs/reports/2026-08-16-soniox-confidence-probe.md` stops meaning what it says.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.soniox_confidence_probe import (  # noqa: E402
    auroc, average_precision, group_words, quantile_edges, word_units,
)


def tok(text, conf, start, is_final=True):
    return {"text": text, "confidence": conf, "start_ms": start,
            "end_ms": start + 60, "is_final": is_final}


def test_words_cut_on_whitespace_and_min_excludes_punctuation_only_for_lex():
    # "Όμως, στις" -> two words. The comma sits at 0.10 inside the first word.
    words, stats = group_words([
        tok("Ό", 0.90, 300), tok("μ", 0.95, 360), tok("ως", 1.0, 420),
        tok(",", 0.10, 480), tok(" στις", 0.64, 540),
    ])
    assert [w["raw"] for w in words] == ["Όμως,", "στις"]
    assert words[0]["conf_min"] == 0.10          # preregistered: all runes
    assert words[0]["conf_min_lex"] == 0.90      # production: lexical runes only
    assert words[0]["start"] == 0.300
    assert words[1]["conf_min"] == 0.64
    assert stats == {"words_without_timestamp": 0, "words_without_confidence": 0,
                     "words_without_lexical_rune": 0}


def test_non_final_tokens_are_ignored():
    words, _ = group_words([tok(" ναι", 0.9, 0), tok(" οχι", 0.1, 100, is_final=False)])
    assert [w["raw"] for w in words] == ["ναι"]


def test_word_without_timestamp_is_counted_not_placed_at_zero():
    words, stats = group_words([{"text": " ναι", "confidence": 0.9, "is_final": True}])
    assert words == []
    assert stats["words_without_timestamp"] == 1


def test_punctuation_only_word_yields_no_scored_token():
    words, _ = group_words([tok(" ...", 0.5, 0)])
    units, dropped, split = word_units(words)
    assert units == [] and dropped == 1 and split == 0


def test_auroc_ties_are_worth_half():
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert auroc([1.0, 0.0], [1, 0]) == 1.0
    assert auroc([0.0, 1.0], [1, 0]) == 0.0
    assert auroc([1.0, 1.0], [1, 1]) is None      # no negatives -> undefined


def test_average_precision_perfect_and_prevalence():
    assert average_precision([1.0, 0.0], [1, 0]) == 1.0
    # one positive ranked last out of four -> AP = 1/4
    assert average_precision([0.4, 0.3, 0.2, 0.1], [0, 0, 0, 1]) == 0.25


def test_quantile_edges_are_deduplicated_and_span_the_data():
    e = quantile_edges([0.0] * 9 + [1.0], q=10)
    assert e[0] == 0.0 and e[-1] == 1.0
    assert e == sorted(set(e))
