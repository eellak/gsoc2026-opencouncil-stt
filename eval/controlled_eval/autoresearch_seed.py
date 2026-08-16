#!/usr/bin/env python3
"""The first, small run of the autoresearch harness — six ideas, one A/A control.

This is a SMOKE TEST OF THE HARNESS. It is not a result about the model. Its job is
to prove that registration precedes results, that the journal cannot be rewritten,
that the dedup guard fires on a cosmetic variant, and that the multiplicity machinery
produces a number with its denominator attached.

WHERE THE IDEAS COME FROM. The wayfinder #24 diagnostic decomposed the 5.30-point gap
between W (0.1005) and the alignment-conditional column oracle (0.0475) by replaying
hindsight into named column classes:

    exact_2_of_3 token majorities   25.0% of the gap
    occupancy columns               14.2%  (fails the insertion gate even in hindsight)
    all unresolved columns          12.7%
    agree columns                    0.0%  (62,919 columns, oracle never disagrees)

So the mass is where the three systems agree, or agree 2-to-1, and are wrong
together. Ideas 1-3 attack the majority class, idea 4 the occupancy class, idea 6 the
insertion side of occupancy.

DECLARED POOR BET. Idea 5 re-arbitrates unresolved columns, which is the family that
already returned ~zero: the char vote of `exp-2026-08-16-char-vote-homophones` had
136 eligible columns of 80,659 and gave dWER -0.00008 with a CI spanning zero. It is
seeded anyway, as a negative control for the harness rather than as a live candidate,
and it is registered with `poor_bet` set so the journal says so.

Run:  .venv-eval/bin/python -m eval.controlled_eval.autoresearch_seed
Env:  N_BOOT_SEARCH (2000), R_WILD (9999), SC, WORKERS
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import autoresearch as A                     # noqa: E402
from eval.controlled_eval.column_classes import column_class           # noqa: E402
from eval.controlled_eval.fusion_lab import Idea, load_substrate, log  # noqa: E402

OUT = Path(__file__).with_name("results_autoresearch.json")
COMMON_FREQ = 5            # identical to exp_char_homophone.py / name_repair


def _lexicon(train) -> Counter:
    freq: Counter = Counter()
    for w in train:
        freq.update(w.ref)
    return freq


class _ColumnRule(Idea):
    """Rewrite W's stream by revisiting single columns of the frozen alignment.

    Every subclass sees exactly what W saw — the column's three entries and W's own
    choice — plus a reference-word frequency table fitted on the training fold. None
    of them may look at the held-out reference.
    """
    fitted = True

    def fit(self, train):
        return _lexicon(train)

    def decide(self, col, tok, freq) -> str | None:
        raise NotImplementedError

    def apply(self, w, params):
        freq = params or Counter()
        out = []
        for d in w.decisions:
            tok = self.decide(w.cols[d["col"]], d["token"], freq)
            if tok is not None:
                out.append(tok)
        return out


def _pair(col):
    """(majority token, minority token) of an exact_2_of_3 column."""
    c = Counter(e for e in col if e is not None)
    (maj, _), (mino, _) = c.most_common(2)
    return maj, mino


class MajorityOOVOverride(_ColumnRule):
    """Override a 2-of-3 majority when only the minority is a word we have seen."""
    name = "majority_oov_override"

    def decide(self, col, tok, freq):
        if tok is None or column_class(col) != "exact_2_of_3":
            return tok
        maj, mino = _pair(col)
        if tok != maj:
            return tok
        if freq.get(maj, 0) == 0 and freq.get(mino, 0) >= COMMON_FREQ:
            return mino
        return tok


class MajorityFreqRatioOverride(_ColumnRule):
    """Override a 2-of-3 majority when the minority is far more frequent in training."""
    name = "majority_freq_ratio"
    RATIO = 10

    def decide(self, col, tok, freq):
        if tok is None or column_class(col) != "exact_2_of_3":
            return tok
        maj, mino = _pair(col)
        if tok != maj:
            return tok
        fm, fn = freq.get(maj, 0), freq.get(mino, 0)
        if fn >= COMMON_FREQ and fn >= self.RATIO * max(fm, 1) and fm < COMMON_FREQ:
            return mino
        return tok


class MajorityShortFunctionWord(_ColumnRule):
    """Override a 2-of-3 majority only for short, very common function words."""
    name = "majority_function_word"
    FLOOR = 200

    def decide(self, col, tok, freq):
        if tok is None or column_class(col) != "exact_2_of_3":
            return tok
        maj, mino = _pair(col)
        if tok != maj or len(mino) > 4:
            return tok
        if freq.get(mino, 0) >= self.FLOOR and freq.get(maj, 0) < self.FLOOR:
            return mino
        return tok


class OccupancyRestore(_ColumnRule):
    """Restore text at two-present-same columns W dropped, when the word is common."""
    name = "occupancy_restore"

    def decide(self, col, tok, freq):
        if tok is not None or column_class(col) != "two_present_same":
            return tok
        cand = next(e for e in col if e is not None)
        return cand if freq.get(cand, 0) >= COMMON_FREQ else None


class UnresolvedLexiconPick(_ColumnRule):
    """On a tied column, take the one candidate the training text has actually seen."""
    name = "unresolved_lexicon_pick"

    def decide(self, col, tok, freq):
        if tok is None or column_class(col) not in ("unresolved_two",
                                                    "unresolved_three"):
            return tok
        seen = [e for e in {x for x in col if x is not None}
                if freq.get(e, 0) >= COMMON_FREQ]
        return seen[0] if len(seen) == 1 else tok


class SingletonOOVDrop(_ColumnRule):
    """Drop a lone token only one system heard that the training text never saw."""
    name = "singleton_oov_drop"

    def decide(self, col, tok, freq):
        if tok is None or column_class(col) != "singleton":
            return tok
        return None if freq.get(tok, 0) == 0 else tok


class OccupancyRestoreSingleton(_ColumnRule):
    """Restore a common word only one system heard, which the vote dropped.

    ROUND 2. Round 1's `occupancy_restore` fired on ZERO columns: the hierarchical
    vote already keeps every [x, x, eps] column, so there was nothing there to
    restore. The occupancy mass the diagnostic priced sits in the [x, eps, eps]
    singletons the vote drops. The harness caught the mis-targeting for free, which
    is what a firing-set-of-zero is for.
    """
    name = "occupancy_restore_singleton"
    FLOOR = COMMON_FREQ

    def decide(self, col, tok, freq):
        if tok is not None or column_class(col) != "singleton":
            return tok
        cand = next(e for e in col if e is not None)
        return cand if freq.get(cand, 0) >= self.FLOOR else None


class OccupancyRestoreSingletonStrict(OccupancyRestoreSingleton):
    """The same restoration behind a much higher frequency floor."""
    name = "occupancy_restore_singleton_strict"
    FLOOR = 50


class TwoPresentOOVDrop(_ColumnRule):
    """Drop a word two systems heard when the training text has never seen it."""
    name = "two_present_oov_drop"

    def decide(self, col, tok, freq):
        if tok is None or column_class(col) != "two_present_same":
            return tok
        return None if freq.get(tok, 0) == 0 else tok


class MajorityOOVOverrideRestyled(_ColumnRule):
    """Deliberate cosmetic variant of `majority_oov_override`, to test the guard.

    Same rule, different code: the class test is spelled with occupancy counts
    instead of `column_class`, and the majority/minority pair is recovered by sorting
    rather than by `Counter.most_common`. The firing set is identical, so the dedup
    guard must refuse it. If it ever does not, the guard is broken.
    """
    name = "majority_oov_override_restyled"

    def decide(self, col, tok, freq):
        present = [e for e in col if e is not None]
        if tok is None or len(present) != 3 or len(set(present)) != 2:
            return tok
        ranked = sorted(set(present), key=lambda t: -present.count(t))
        maj, mino = ranked[0], ranked[1]
        if tok != maj:
            return tok
        if freq.get(maj, 0) == 0 and freq.get(mino, 0) >= COMMON_FREQ:
            return mino
        return tok


SEEDS = [
    ("null_identity",
     "A/A control: an idea that changes nothing must show exactly zero and p=1.",
     lambda: Idea(), None),
    (MajorityOOVOverride.name,
     "A 2-of-3 majority that is out-of-vocabulary while the minority is common is a "
     "shared ASR guess, and the minority is the real word.",
     MajorityOOVOverride, None),
    (MajorityFreqRatioOverride.name,
     "The same override driven by a frequency ratio rather than a hard OOV test "
     "should fire on a strictly larger, mostly overlapping set of columns.",
     MajorityFreqRatioOverride, None),
    (MajorityShortFunctionWord.name,
     "Jointly wrong majorities concentrate in short Greek function words where two "
     "systems share the same acoustic confusion.",
     MajorityShortFunctionWord, None),
    (OccupancyRestore.name,
     "Where two systems heard a common word and the vote dropped it, the word was "
     "spoken and restoring it lowers deletions more than it raises insertions.",
     OccupancyRestore,
     "hindsight replay of this class already FAILED the insertion gate (14.2% of the "
     "gap, ins 0.0374 -> 0.0391), so this is expected to fail"),
    (UnresolvedLexiconPick.name,
     "On a three-way tie, the candidate the training references have seen is the "
     "word that was said.",
     UnresolvedLexiconPick,
     "text-only arbitration restricted to disagreement columns already returned "
     "~zero (char vote: 136 of 80,659 columns, dWER -0.00008, CI spans zero)"),
    (SingletonOOVDrop.name,
     "A token only one system heard and the training text never saw is an insertion, "
     "and dropping it lowers WER without touching deletions.",
     SingletonOOVDrop, None),
    # ---- round 2, registered after round 1's numbers were seen. That ordering is
    # legitimate BECAUSE it happens on the search partition; confirmation is still
    # untouched. It is also exactly the adaptivity the split exists to absorb.
    (OccupancyRestoreSingleton.name,
     "The occupancy mass is in the singletons the vote drops, not the two-present "
     "columns it keeps, so restoring common singleton words lowers deletions.",
     OccupancyRestoreSingleton,
     "the hindsight replay of this class raised insertions 0.0374 -> 0.0391 and "
     "failed the frozen insertion gate, so this is expected to fail it too"),
    (OccupancyRestoreSingletonStrict.name,
     "Restricting the same restoration to very frequent words trades recall for "
     "precision and may clear the insertion gate the loose version fails.",
     OccupancyRestoreSingletonStrict, None),
    (TwoPresentOOVDrop.name,
     "A word two systems heard that the training text never saw is a shared "
     "hallucination, and dropping it lowers insertions without costing deletions.",
     TwoPresentOOVDrop, None),
    (MajorityOOVOverrideRestyled.name,
     "Harness control: a rewritten copy of an already-evaluated rule must be refused "
     "by the behavioural dedup guard rather than scored a second time.",
     MajorityOOVOverrideRestyled,
     "registered on purpose as a cosmetic variant; a PASS here would be a harness bug"),
]


def main() -> int:
    sub = load_substrate()
    A.assert_partition(sub)
    search = A.search_partition(sub)
    confirm = A.confirm_partition(sub)
    log(f"search  {search.meta['n_cities']} cities, {search.meta['n_windows']} windows, "
        f"{search.meta['n_meetings']} meetings, {search.meta['ref_tokens']} ref tokens")
    log(f"confirm {confirm.meta['n_cities']} cities, {confirm.meta['n_windows']} windows, "
        f"{confirm.meta['n_meetings']} meetings, {confirm.meta['ref_tokens']} ref tokens "
        "— SEALED until a batch is frozen")

    reg = A.Registry()
    n_boot = int(os.environ.get("N_BOOT_SEARCH", "2000"))
    weights = A.rademacher(search.meta["n_meetings"])

    handles, results = {}, {}
    for name, hyp, factory, poor in SEEDS:
        h = reg.register(name, hyp, factory, poor_bet=poor, resume=True)
        handles[name] = h
        log(f"registered {name} [{h.idea_key}] — hypothesis on record before any number")

    done = reg.searched()
    prior = {r["idea_key"]: r for r in reg.journal.records()
             if r["type"] == A.SEARCH_RESULT}
    for name, *_ in SEEDS:
        h = handles[name]
        if h.idea_key in done:
            s = prior[h.idea_key]                 # already scored; never re-scored
        else:
            s = reg.run_search(h, search, n_boot=n_boot, weights=weights)
        results[name] = s
        log(A.summary_line(s))
        if s.get("duplicate_of"):
            log(f"  REFUSED as a cosmetic variant of {s['duplicate_of']}")

    board = reg.leaderboard()
    survivors = [n for n, s in results.items()
                 if s["screen"]["pass"] and not s.get("duplicate_of")]
    log(json.dumps(board["denominator"], indent=1))
    log(f"survivors of the search screen: {survivors or 'NONE'}")
    if survivors:
        log("confirmation batch NOT frozen automatically — freezing the batch is the "
            "one-way door and stays a deliberate act")

    OUT.write_text(json.dumps({"leaderboard": board, "search": results,
                               "survivors": survivors}, ensure_ascii=False, indent=1))
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
