#!/usr/bin/env python3
"""Arm E: post-hoc roster name repair (shadow mode). Step 1 of
docs/specs/2026-08-11-name-repair-plan.md, under the arm-E prereg of
docs/specs/2026-08-12-serving-stack-plan.md.

The rule is the one the Step-0 feasibility audit implemented and PASSED
(scripts/analysis/name_lexicon_audit.py, results in
data/reports/finetune-research/name-lexicon-audit-2026-08-12.json). Its
normalization (NFD strip combining marks, lowercase, fold final sigma, NFC),
its OSA Damerau-Levenshtein, its length/distance budget, and all of its
abstention conditions are reused VERBATIM from that audit, because the audit's
gate numbers were computed with exactly this behavior:

  length < 6           : never
  length 6-9           : distance exactly 1
  length >= 10         : distance <= 2 AND distance/length <= 0.20

plus, jointly: unique best candidate; runner-up at >= 2 more edits; the term
identifies exactly ONE person in this meeting's roster; the hypothesis token is
not itself a valid alias of a roster term; the hypothesis token is not a common
Greek word (frequency >= 5 in seen-city references, supplied by the caller); at
distance 2, an independent signal is additionally required (capital letter off
sentence start, adjacent first name, or adjacent honorific).

PREREGISTERED AMENDMENT (frozen 2026-08-12, BEFORE any WER number was computed
for this arm). The audit's only two correct->wrong firings were one failure
mode: a correctly transcribed out-of-roster feminine surname variant
('μιχαηλιδου') was pulled at distance 2 onto the in-roster masculine form
('μιχαηλιδη'). Guard, applied ONLY at distance 2, before the signal check:

  if the hypothesis token ends in a productive Greek surname suffix
  (checked longest first: 'οπουλου', 'ιδου', 'ακου', 'ακη', 'ου')
  AND it differs from the chosen candidate alias only in that suffix region
  (i.e. the candidate alias starts with the token's pre-suffix stem),
  then ABSTAIN (decision 'abstain_suffix_family').

Rationale: such a token is a plausible standalone inflected/feminine variant of
a surname family that is simply not in the roster; rewriting it is an identity
swap, not a repair. Distance-1 behavior is untouched. This guard is
deterministic, was frozen from the audit's failure rows alone, and is not
tuned afterwards; it is accepted that it also blocks the audit's one
distance-2 feminine wrong->correct case ('τσομπανιδου'->'τσομπανιδη').

Deviations from the plan text, documented rather than silently resolved:
  - The plan's "surname appearing in >=5% of meetings is never corrected on
    its own" clause was not operationalized by the Step-0 audit (no
    per-meeting surname frequency table exists in its inputs) and is likewise
    not implemented here; the audit's common-word abstention is the frozen
    stand-in. Same rule as audited, no more, no less.
  - Replacements are written with the term file's stored alias surface form
    (ftoks-normalized: lowercase, accent-free, final 'ς' preserved), with the
    original token's casing shape re-applied (Title/UPPER). Accents are NOT
    reconstructed: every scorer in this repo strips combining marks before
    comparing, so accent reconstruction cannot change any measured number and
    would be an un-audited cosmetic heuristic.

Contract (arm-E prereg): deterministic; exact no-op on an empty/missing
roster; idempotent (a written alias is a valid roster form, so a second pass
lands in 'abstain_already_valid'); punctuation and everything outside replaced
tokens preserved byte-for-byte.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

COMMON_FREQ_THRESHOLD = 5  # verbatim from the Step-0 audit

# Amendment guard: productive Greek surname suffixes, longest first.
SURNAME_SUFFIXES = ("οπουλου", "ιδου", "ακου", "ακη", "ου")


# ---------- normalization (verbatim from name_lexicon_audit.rnorm) ----------

def rnorm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFC", s).lower().replace("ς", "σ")
    return s


# ---------- OSA Damerau-Levenshtein (verbatim from the audit) ----------

def dl(a: str, b: str, cap: int = 10) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[n][m]


def rule_allows(token_len: int, dist: int) -> bool:
    """Conservative length rule from the plan (length = hypothesis token length)."""
    if token_len < 6:
        return False
    if 6 <= token_len <= 9:
        return dist == 1
    return dist <= 2 and dist / token_len <= 0.20


# ---------- roster context ----------

@dataclass
class RosterContext:
    """Everything repair() needs about one meeting. Built once per meeting."""
    present: dict = field(default_factory=dict)        # term id -> {term, persons_in_meeting}
    first_names: set = field(default_factory=set)      # rnorm first names of roster persons
    valid_aliases: set = field(default_factory=set)    # rnorm aliases of present terms
    alias_surface: dict = field(default_factory=dict)  # (term id, rnorm alias) -> stored alias
    seen_freq: Counter = field(default_factory=Counter)


def meeting_roster_terms(city_terms: list[dict], roster_entries: list[str]) -> dict:
    """Term ids present in this meeting's roster (verbatim from the audit)."""
    roster_norm = [rnorm(e) for e in roster_entries]
    roster_words = set()
    for e in roster_norm:
        roster_words.update(re.findall(r"\w+", e))
    present = {}
    for t in city_terms:
        surname = rnorm(t["canonical"])
        covered_present = []
        for cov in t.get("covers", []):
            cn = rnorm(cov)
            if any(cn == e or cn in roster_norm for e in roster_norm):
                covered_present.append(cov)
        if surname in roster_words or covered_present:
            present[t["id"]] = {
                "term": t,
                "persons_in_meeting": covered_present or t.get("covers", []),
            }
    return present


def build_context(city_terms: list[dict], roster_entries: list[str],
                  seen_freq: Counter | None = None) -> RosterContext:
    ctx = RosterContext(seen_freq=seen_freq or Counter())
    if not roster_entries:
        return ctx  # empty context -> repair() is an exact no-op
    ctx.present = meeting_roster_terms(city_terms, roster_entries)
    for tid, info in ctx.present.items():
        for cov in info["persons_in_meeting"]:
            parts = cov.split()
            if len(parts) >= 2:
                ctx.first_names.add(rnorm(parts[0]))
        for alias in info["term"]["aliases"]:
            an = rnorm(alias)
            ctx.valid_aliases.add(an)
            ctx.alias_surface[(tid, an)] = alias
    return ctx


# ---------- distance to a term over its aliases (verbatim) ----------

def term_distance(token_norm: str, term: dict):
    best = (99, None)
    for alias in term["aliases"]:
        a = rnorm(alias)
        d0 = dl(token_norm, a)
        if d0 < best[0]:
            best = (d0, a)
    return best


# ---------- independent signal for distance-2 (verbatim) ----------

def has_signal(raw_hyp: str, token_norm: str, first_names: set[str]) -> dict:
    """Capital off sentence-start, adjacent 'κ.'/κύριος/κυρία, or adjacent first name."""
    words = re.findall(r"[\w.]+", raw_hyp)
    sigs = {"capital_mid": False, "kyrios": False, "first_name": False}
    for i, w in enumerate(words):
        if rnorm(re.sub(r"\.$", "", w)) != token_norm:
            continue
        if w[:1].isupper():
            prev = words[i - 1] if i > 0 else ""
            if i > 0 and not re.search(r"[.;!?]$", prev):
                sigs["capital_mid"] = True
        for j in (i - 1, i + 1):
            if 0 <= j < len(words):
                nb = rnorm(words[j])
                if nb in ("κ.", "κ", "κυριοσ", "κυρια", "κυριε"):
                    sigs["kyrios"] = True
                if nb in first_names:
                    sigs["first_name"] = True
    sigs["any"] = any(sigs.values())
    return sigs


# ---------- amendment guard (frozen 2026-08-12, see module docstring) ----------

def suffix_family_abstain(token_norm: str, alias_norm: str) -> bool:
    """True when the token is a plausible other-inflection surname form whose
    difference from the candidate lies entirely in a productive suffix."""
    for suf in SURNAME_SUFFIXES:
        if token_norm.endswith(suf):
            stem = token_norm[: len(token_norm) - len(suf)]
            if stem and alias_norm.startswith(stem):
                return True
    return False


# ---------- the selection rule ----------

def select(token_norm: str, ctx: RosterContext, raw_hyp: str) -> dict:
    """Apply the full conservative rule to one hypothesis token.

    decision in {fire, abstain_tie, abstain_margin, abstain_multi_person,
    abstain_common, abstain_already_valid, abstain_no_signal,
    abstain_suffix_family, no_candidate}."""
    out = {"token": token_norm, "decision": "no_candidate", "candidates": []}
    if token_norm in ctx.valid_aliases:
        out["decision"] = "abstain_already_valid"
        return out
    scored = []
    for tid, info in ctx.present.items():
        d0, alias = term_distance(token_norm, info["term"])
        scored.append((d0, tid, alias, info))
    if not scored:
        return out
    scored.sort(key=lambda x: (x[0], x[1]))
    best_d, best_tid, best_alias, best_info = scored[0]
    if not rule_allows(len(token_norm), best_d):
        return out
    out["candidates"] = [(t, d) for d, t, _, _ in scored[:3]]
    ties = [s for s in scored if s[0] == best_d]
    if len(ties) > 1:
        out["decision"] = "abstain_tie"
        return out
    second_d = scored[1][0] if len(scored) > 1 else 99
    if second_d < best_d + 2:
        out["decision"] = "abstain_margin"
        return out
    if token_norm in ctx.seen_freq and ctx.seen_freq[token_norm] >= COMMON_FREQ_THRESHOLD:
        out["decision"] = "abstain_common"
        out["common_freq"] = ctx.seen_freq[token_norm]
        return out
    persons = best_info["persons_in_meeting"]
    out["n_persons"] = len(persons)
    if len(persons) != 1:
        out["decision"] = "abstain_multi_person"
        out["selected"] = {"term": best_tid, "alias": best_alias, "dist": best_d}
        return out
    if best_d == 2:
        if suffix_family_abstain(token_norm, best_alias):
            out["decision"] = "abstain_suffix_family"
            out["selected"] = {"term": best_tid, "alias": best_alias, "dist": best_d}
            return out
        sigs = has_signal(raw_hyp, token_norm, ctx.first_names)
        out["signals"] = sigs
        if not sigs["any"]:
            out["decision"] = "abstain_no_signal"
            out["selected"] = {"term": best_tid, "alias": best_alias, "dist": best_d}
            return out
    out["decision"] = "fire"
    out["selected"] = {"term": best_tid, "alias": best_alias, "dist": best_d,
                       "person": persons[0]}
    return out


# ---------- text-level repair ----------

@dataclass
class RepairResult:
    text: str
    changes: list  # dicts: start, end, original, replacement, term, person, dist, decision_alias


def _match_case(original: str, replacement: str) -> str:
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def repair(text: str, ctx: RosterContext | None) -> RepairResult:
    """Pure, deterministic post-hoc roster repair of one window's text.

    Exact no-op (byte-identical text, zero changes) when ctx is None or the
    roster produced no present terms. Everything outside replaced tokens is
    preserved byte-for-byte."""
    if ctx is None or not ctx.present:
        return RepairResult(text=text, changes=[])
    pieces = []
    changes = []
    last = 0
    for m in re.finditer(r"\w+", text):
        tok = m.group(0)
        tn = rnorm(tok)
        res = select(tn, ctx, text)
        if res["decision"] != "fire":
            continue
        sel = res["selected"]
        surface = ctx.alias_surface[(sel["term"], sel["alias"])]
        repl = _match_case(tok, surface)
        if repl == tok:
            continue  # nothing to change (already the surface form)
        pieces.append(text[last:m.start()])
        pieces.append(repl)
        last = m.end()
        changes.append({
            "start": m.start(), "end": m.end(), "original": tok,
            "replacement": repl, "term": sel["term"], "person": sel["person"],
            "dist": sel["dist"], "alias": sel["alias"],
        })
    pieces.append(text[last:])
    return RepairResult(text="".join(pieces), changes=changes)
