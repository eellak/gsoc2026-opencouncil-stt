#!/usr/bin/env python3
"""DS-WER: word error rate restricted to domain-critical terms.

Proposal Milestone 2 (`docs/reference/gsoc-proposal.md:100`) asks for Levenshtein
restricted to domain-critical words - council member names and local acronyms. This
is that metric, with the counting rules written down rather than left to the
alignment to decide.

The algorithm, fixed in `docs/specs/2026-08-11-endgame-handoff-plan.md`:

1. Normalize reference, hypothesis and terms with the project's frozen scorer
   (`eval.controlled_eval.scoring.wtoks` plus the frozen filler strip in
   `eval.controlled_eval.eval_freeze`). The matcher and the scorer share a
   normalizer by construction, not by coincidence.
2. Collapse term matches into atomic symbols, longest-match-leftmost.
3. Unit-cost Levenshtein over the whole transformed sequence, backtracking with a
   fixed preference: match > substitution > deletion > insertion.
4. Count only the operations that touch a term:
   - `D` reference term aligned to nothing,
   - `S` reference term aligned to anything else - including a *different* term,
     which is one substitution and never a substitution plus an insertion,
   - `I` hypothesis term aligned to nothing, or to a reference token that is not
     itself a term.
5. `N` = reference term occurrences. `DS-WER = (S+D+I)/N`, and **NA when N is 0** -
   a window with no names in it has no domain accuracy, and reporting 0 there would
   award a perfect score for saying nothing.

The term lists are frozen separately, before any scoring, by
`scripts/build_ds_wer_terms.py`.

    .venv-eval/bin/python -m pytest tests/test_ds_wer.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402

MATCH, SUB, DEL, INS = "match", "sub", "del", "ins"


class TermList:
    """Normalized aliases -> term id, with an alias belonging to exactly one term."""

    def __init__(self, terms: list[dict]):
        self.terms = terms
        self.by_alias: dict[tuple[str, ...], str] = {}
        self.canonical: dict[str, str] = {}
        for t in terms:
            self.canonical[t["id"]] = t.get("canonical", t["id"])
            for alias in t["aliases"]:
                key = tuple(ftoks(alias))
                if not key:
                    continue
                if key in self.by_alias and self.by_alias[key] != t["id"]:
                    raise ValueError(
                        f"duplicate alias {' '.join(key)!r}: claimed by "
                        f"{self.by_alias[key]} and {t['id']}")
                self.by_alias[key] = t["id"]
        self.max_len = max((len(k) for k in self.by_alias), default=0)

    def __len__(self) -> int:
        return len(self.terms)

    @classmethod
    def load(cls, path: str | Path) -> "TermList":
        import json
        return cls(json.loads(Path(path).read_text())["terms"])


def tag(tokens: list[str], terms: TermList) -> list[tuple[bool, str]]:
    """(is_term, symbol) per position, term matches collapsed into one symbol.

    Longest match wins and ties go leftmost, so a term list containing both
    "Δήμος Άργους" and "Άργους" cannot count the same words twice.
    """
    out: list[tuple[bool, str]] = []
    i, n = 0, len(tokens)
    while i < n:
        hit = None
        for span in range(min(terms.max_len, n - i), 0, -1):
            tid = terms.by_alias.get(tuple(tokens[i:i + span]))
            if tid is not None:
                hit = (span, tid)
                break
        if hit:
            out.append((True, hit[1]))
            i += hit[0]
        else:
            out.append((False, tokens[i]))
            i += 1
    return out


def align(ref: list, hyp: list) -> list[tuple[str, int | None, int | None]]:
    """Unit-cost Levenshtein, backtracked match > substitution > deletion > insertion.

    Same tie-break as `eval/controlled_eval/exp_same_stack.py::sdi`, so a DS-WER
    number and a WER number from this repo describe the same alignment.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))

    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((MATCH if ref[i - 1] == hyp[j - 1] else SUB, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append((DEL, i - 1, None))
            i -= 1
        else:
            ops.append((INS, None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def ds_wer(reference: str, hypothesis: str, terms: TermList) -> dict:
    ref = tag(ftoks(reference), terms)
    hyp = tag(ftoks(hypothesis), terms)
    ref_sym = [s for _, s in ref]
    hyp_sym = [s for _, s in hyp]

    per_term: dict[str, dict[str, int]] = {}

    def bump(tid: str, field: str) -> None:
        per_term.setdefault(tid, {"N": 0, "S": 0, "D": 0, "I": 0})[field] += 1

    for is_term, sym in ref:
        if is_term:
            bump(sym, "N")

    S = D = I = 0
    for op, i, j in align(ref_sym, hyp_sym):
        ref_is_term = i is not None and ref[i][0]
        hyp_is_term = j is not None and hyp[j][0]
        if op == MATCH:
            continue
        if op == SUB:
            if ref_is_term:
                # One substitution. Never S on the reference side plus I on the
                # hypothesis side for the same operation.
                S += 1
                bump(ref[i][1], "S")
            elif hyp_is_term:
                I += 1
                bump(hyp[j][1], "I")
        elif op == DEL:
            if ref_is_term:
                D += 1
                bump(ref[i][1], "D")
        else:
            if hyp_is_term:
                I += 1
                bump(hyp[j][1], "I")

    N = sum(1 for is_term, _ in ref if is_term)
    return {"S": S, "D": D, "I": I, "N": N,
            "errors": S + D + I,
            "ds_wer": (S + D + I) / N if N else None,
            "per_term": per_term}


def aggregate(rows: list[dict]) -> dict:
    """Corpus DS-WER: errors and denominators summed, not per-window rates averaged."""
    S = sum(r["S"] for r in rows)
    D = sum(r["D"] for r in rows)
    I = sum(r["I"] for r in rows)
    N = sum(r["N"] for r in rows)
    return {"S": S, "D": D, "I": I, "N": N, "errors": S + D + I,
            "ds_wer": (S + D + I) / N if N else None}
