#!/usr/bin/env python3
"""Arms C (per-character vote) and H (homophones) on top of W - wayfinder #24.

READ THE CENSUS FIRST (`column_census.py`, results_column_census.json). It was run
before either arm was built, because #18's lesson is that an arbiter can look neutral
merely because it was handed 2.6% of the decisions. The counts:

  80,659 columns. Only 2,066 (2.56%) are unresolved - the same tie set #18's LLM saw.
  After protecting token majorities and quarantining token-boundary disagreements:
      arm C eligible: 136 columns (0.17% of all columns)
      arm H eligible:  34 columns (0.042%)   - loose variant adds one
  Columns where W's vote differs from the alignment-conditional column oracle: 5,915.
  Only 23 of those are H-eligible and 77 are C-eligible.

So arm H is a machine for 34 decisions with a hindsight ceiling of 23 tokens out of
74,917, i.e. 0.031 WER POINTS if it were an oracle. It is NOT BUILT: no KenLM, no
LLM. Its ceiling is computed here instead, by the same scorer, so the decision rests
on a measured number and not on an argument. Arm C is cheap and text-only, so it IS
built and measured out-of-fold, even though its own ceiling is 0.103 points.

ARM C, as frozen: on a column with three distinct candidates, no strict-homophone
relation, max pairwise character distance <= 2 and no split/merge suspicion, align
the three candidate strings CHARACTER-wise with the same exact 3-way DP and vote with
the same hierarchical rule. Accept the composite only if it equals one of the
candidates or appears in the closed lexicon. Otherwise keep W's token. C never emits
epsilon and never consumes a column W dropped, so it cannot change the token count.

LEXICON, fitted leave-one-city-out: tokens appearing >= 5 times in the reference text
of the NINE TRAINING CITIES, plus the held-out city's frozen term list and this
meeting's roster surnames. The term list is a declared contamination: it was mined
from data overlapping this benchmark, so the `C_common_only` sensitivity arm drops it.

Writes results_char_homophone.json (aggregates only, never transcript text).

Env: SC (cache dir), N_BOOT (10000), WORKERS
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.column_classes import (column_class,          # noqa: E402
                                                 eligibility)
from eval.controlled_eval.fusion_lab import (Idea, N_BOOT,             # noqa: E402
                                             evaluate, load_substrate, log,
                                             summary_line)
from eval.controlled_eval.msa import align3, compose, oracle_select    # noqa: E402
from eval.controlled_eval.roster_lexicon import (load_city_terms,      # noqa: E402
                                                 load_rosters)
from eval.controlled_eval.scoring import wtoks                         # noqa: E402
from scripts.serving_stack.name_repair import rnorm                    # noqa: E402

OUT = Path(__file__).with_name("results_char_homophone.json")
COMMON_FREQ = 5           # verbatim from name_repair.COMMON_FREQ_THRESHOLD
CHAR_BAND = 12            # candidates are single tokens; 12 characters is generous


def char_vote(cands: list[str], pivot: int) -> str:
    """Per-character hierarchical vote over three candidate strings."""
    a, b, c = (list(cands[0]), list(cands[1]), list(cands[2]))
    band = max(CHAR_BAND, max(len(a), len(b), len(c)) - min(len(a), len(b), len(c)) + 4)
    cols = align3(a, b, c, band=band)
    chars, _ = compose(cols, pivot=pivot)
    return "".join(chars)


class ArmC(Idea):
    """Per-character composition inside three-way disagreement columns."""

    fitted = True

    def __init__(self, name="C", use_terms=True):
        self.name = name
        self.use_terms = use_terms
        self.city_terms = load_city_terms() if use_terms else {}
        self.rosters = load_rosters() if use_terms else {}
        self.acct = Counter()

    def fit(self, train):
        freq = Counter()
        for w in train:
            freq.update(w.ref)
        return {t for t, n in freq.items() if n >= COMMON_FREQ}

    def _lexicon(self, w, common: set[str]) -> set[str]:
        lex = set(common)
        if not self.use_terms:
            return lex
        for t in self.city_terms.get(w.city, []):
            for alias in t.get("aliases", []):
                lex.add(rnorm(alias))
            lex.add(rnorm(t["canonical"]))
        for entry in self.rosters.get(f"{w.city}/{w.meeting}", []):
            for part in rnorm(entry).split():
                lex.add(part)
        return lex

    def apply(self, w, params):
        elig = eligibility(w.cols)
        lex = self._lexicon(w, params or set())
        out: list[str] = []
        for d in w.decisions:
            tok = d["token"]
            i = d["col"]
            if tok is not None and elig.get(i) == "C":
                col = w.cols[i]
                comp = char_vote([col[0], col[1], col[2]], w.pivot)
                cands = {e for e in col if e is not None}
                self.acct["fired_columns"] += 1
                if comp == tok:
                    self.acct["composite_equals_W"] += 1
                elif comp in cands:
                    self.acct["accepted_other_candidate"] += 1
                    tok = comp
                elif comp in lex:
                    self.acct["accepted_novel_in_lexicon"] += 1
                    tok = comp
                else:
                    self.acct["rejected_off_lexicon"] += 1
            if tok is not None:
                out.append(tok)
        return out


class CeilingArm(Idea):
    """W with the column oracle's own entry in every column eligible for `arms`.

    HINDSIGHT: it reads the reference. It is here to price an arm's eligible set with
    the frozen scorer instead of arguing about it, and it is never a result.
    """

    fitted = False

    def __init__(self, arms: tuple[str, ...] = (), loose=False, name=None,
                 classes: tuple[str, ...] = ()):
        self.arms = arms
        self.classes = classes          # whole column CLASSES, ignoring eligibility
        self.loose = loose
        self.name = name or f"ceiling[{'+'.join(arms or classes)}]"

    def apply(self, w, params):
        elig = ({i: "*" for i, col in enumerate(w.cols)
                 if column_class(col) in self.classes} if self.classes
                else eligibility(w.cols, loose=self.loose))
        arms = ("*",) if self.classes else self.arms
        # per-COLUMN oracle choice, read off the oracle DP's own backtrace. An
        # earlier version matched the oracle's TOKEN LIST back onto the columns by
        # membership, which mis-attributes a token whenever the same word is a
        # candidate in two nearby columns (CodeRabbit, this change).
        choice = oracle_select(w.cols, w.ref)
        out = []
        for d in w.decisions:
            tok = d["token"]
            i = d["col"]
            if elig.get(i) in arms and choice[i] is not None:
                tok = choice[i]
            if tok is not None:
                out.append(tok)
        return out


def main():
    sub = load_substrate()
    log(json.dumps(sub.meta, indent=1))
    n_boot = N_BOOT

    res = {"substrate": sub.meta, "n_boot": n_boot,
           "census": "eval/controlled_eval/results_column_census.json",
           "arms": {}, "accounting": {}}

    c_full = ArmC("C", use_terms=True)
    c_common = ArmC("C_common_only", use_terms=False)
    ideas = [
        c_full,
        c_common,
        CeilingArm(("C",), name="ceiling_C"),
        CeilingArm(("H",), name="ceiling_H"),
        CeilingArm(("H",), loose=True, name="ceiling_H_loose"),
        CeilingArm(("C", "H"), name="ceiling_C+H"),
        # the honest outer bound on ANY per-column identity arbiter restricted to
        # these three systems: perfect hindsight on EVERY unresolved column, not
        # only the ones C and H are allowed to touch
        CeilingArm(classes=("unresolved_two", "unresolved_three"),
                   name="ceiling_all_unresolved"),
        # and the two classes no identity arbiter can reach, for the decomposition
        CeilingArm(classes=("singleton", "two_present_same"),
                   name="ceiling_occupancy_columns"),
        CeilingArm(classes=("exact_2_of_3",), name="ceiling_token_majorities"),
    ]
    for idea in ideas:
        r = evaluate(idea, sub, fold="city", n_boot=n_boot)
        res["arms"][idea.name] = r
        log(summary_line(r))
    res["accounting"]["C"] = dict(c_full.acct)
    res["accounting"]["C_common_only"] = dict(c_common.acct)

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
