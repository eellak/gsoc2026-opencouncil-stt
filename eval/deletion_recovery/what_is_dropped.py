# -*- coding: utf-8 -*-
"""Step 2 of `exp-2026-09-16-scribe-mode-product-choice`: WHAT does the clean read drop?

`exp-2026-09-16-deletion-recovery` established that 56.2% of the words the clean mode
drops and the verbatim mode keeps are words the human annotator actually heard. It did
not establish what those words ARE, and the whole product decision turns on that: a
dropped hesitation is the feature working, a dropped negation is meaning destroyed.

This sorts the recovered words into frozen categories and prints counts. It is
DESCRIPTIVE. The category lists below are closed, small and written down here rather
than tuned, but they are still a rule of thumb applied to single tokens with no context
-- «μη» inside «μη κερδοσκοπικός» is not a negation, and this script cannot tell. Treat
every count as an upper bound on its category and a pointer to where a human should look,
never as an adjudication.

Counts only. The words themselves stay in the cache -- they are PII.

    .venv-eval/bin/python -m eval.deletion_recovery.what_is_dropped
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.deletion_recovery.gold_score import (                      # noqa: E402
    BASE_DIR, DONOR_DIR, _key, cell_streams,
)
from eval.deletion_recovery.screen import _backtrace, _blind_stream  # noqa: E402
from eval.gold_set_score import load                                 # noqa: E402

OUT = ROOT / "eval" / "results_what_is_dropped.json"
REGION = "core_envelope"

# Frozen, and deliberately tiny. These are the tokens whose loss changes what a
# council meeting recorded, which is the only reason this categorisation exists.
CRITICAL = {"δεν", "μην", "μη", "οχι", "όχι", "ναι", "ουτε", "ούτε", "χωρις", "χωρίς"}

# A closed list of Greek function words: articles, common prepositions, conjunctions,
# particles and clitic pronouns. Losing one usually costs grammar, not content.
FUNCTION = {
    "ο", "η", "το", "οι", "τα", "του", "της", "των", "τον", "την", "τους", "τις",
    "ενας", "ένας", "μια", "μία", "ενα", "ένα", "ενος", "ενός", "μιας",
    "και", "κι", "αλλα", "αλλά", "η'", "ειτε", "είτε", "ομως", "όμως", "λοιπον",
    "λοιπόν", "αρα", "άρα", "γιατι", "γιατί", "οτι", "ότι", "που", "ποu", "πως",
    "πώς", "αν", "οταν", "όταν", "ενω", "ενώ", "καθως", "καθώς", "ωστε", "ώστε",
    "σε", "στο", "στη", "στην", "στον", "στους", "στις", "στα", "απο", "από",
    "για", "με", "χωρίς'", "προς", "παρα", "παρά", "κατα", "κατά", "μετα", "μετά",
    "πριν", "μεχρι", "μέχρι", "εως", "έως", "επι", "επί", "υπο", "υπό",
    "μου", "σου", "μας", "σας", "τoυ", "τη", "θα", "να", "ας", "δε", "μεν",
    "ειναι", "είναι", "ηταν", "ήταν", "εχει", "έχει", "εχουμε", "έχουμε",
    "αυτο", "αυτό", "αυτη", "αυτή", "αυτος", "αυτός", "εδω", "εδώ", "εκει", "εκεί",
}

NUMBER_WORDS = {
    "ενα", "ένα", "δυο", "δύο", "τρια", "τρία", "τεσσερα", "τέσσερα", "πεντε",
    "πέντε", "εξι", "έξι", "εφτα", "επτά", "εφτά", "οκτω", "οχτώ", "οκτώ", "εννεα",
    "εννέα", "δεκα", "δέκα", "εκατο", "εκατό", "χιλια", "χίλια", "χιλιαδες",
    "χιλιάδες", "εκατομμυρια", "εκατομμύρια", "πρωτο", "πρώτο", "δευτερο", "δεύτερο",
    "τριτο", "τρίτο", "τοις", "εκατο'",
}


def categorise(token: str, neighbours: set[str]) -> str:
    """One token -> one category. Order matters and is frozen here."""
    if token in CRITICAL:
        return "critical_negation_or_vote"
    if any(ch.isdigit() for ch in token) or token in NUMBER_WORDS:
        return "number"
    if token in neighbours:
        return "repetition_of_a_neighbour"
    if len(token) == 1:
        return "single_character"
    if token in FUNCTION:
        return "function_word"
    return "content_word"


def main() -> int:
    _man, ans, _cells, sel = load()
    scored = [c for c in _man["scored_cell_ids"]
              if (BASE_DIR / f"{c}.json").exists() and (DONOR_DIR / f"{c}.json").exists()]
    heard = Counter()
    unheard = Counter()
    n_heard = n_unheard = 0

    for cid in scored:
        got = cell_streams(cid, ans, REGION)
        if got is None:
            continue
        seq, base, donor = got
        # "The human wrote this word down" means the restored word ALIGNS TO a gold
        # word at its own position -- not that the token appears somewhere in the
        # cell. The multiset version of this test overstated support and was
        # corrected in screen.restore on the same day; the same error here would
        # inflate exactly the categories this script exists to count.
        base_toks = [t for t in (_key(w) for w in base) if t]
        donor_toks = [t for t in (_key(w) for w in donor) if t]
        stream, added = _blind_stream(base_toks, donor_toks)
        matched = {j for op, _i, j in _backtrace(seq, stream) if op == "match"}
        neighbours = set(base_toks)
        for idx in added:
            cat = categorise(stream[idx], neighbours)
            if idx in matched:
                heard[cat] += 1
                n_heard += 1
            else:
                unheard[cat] += 1
                n_unheard += 1

    cats = sorted(set(heard) | set(unheard))
    result = {
        "protocol": "what-is-dropped-2026-09-16",
        "experiment": "exp-2026-09-16-scribe-mode-product-choice",
        "region": REGION, "cells": len(scored),
        "heard_by_category": dict(heard), "not_heard_by_category": dict(unheard),
        "totals": {"heard": n_heard, "not_heard": n_unheard},
        "caveat": ("single-token categorisation without context; every category count is "
                   "an upper bound and a pointer for a human, not an adjudication"),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{len(scored)} cells, region {REGION}")
    print(f"words the clean read dropped and the donor kept: "
          f"{n_heard + n_unheard}  (heard by the human: {n_heard})\n")
    print(f"{'category':32s} {'HEARD':>6} {'not heard':>10}")
    for c in cats:
        print(f"{c:32s} {heard.get(c, 0):6d} {unheard.get(c, 0):10d}")
    crit = heard.get("critical_negation_or_vote", 0)
    cont = heard.get("content_word", 0) + heard.get("number", 0)
    print(f"\nof the {n_heard} words the human heard: {crit} critical, {cont} content or "
          f"number, {n_heard - crit - cont} hesitation-shaped or grammatical")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
