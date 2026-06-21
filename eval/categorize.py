"""Heuristic, text-only error categorizer for correction pairs.

No audio, no NER — so this is approximate and merges the UI's fine labels into
text-decidable buckets (person/place/org -> named_entity;
verb/noun/article -> morph_grammar). It exists to stratify the eval sample and
the routing report. Treated as triage, not ground truth.

Output labels:
  accent_tonos, final_sigma, punctuation_capitalization, word_boundary,
  number_date, acronym_abbreviation, named_entity, homophone, morph_grammar,
  insertion_deletion, other_lexical, no_change
"""
from __future__ import annotations

import re
import unicodedata

_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΆΈΉΊΌΎΏΪΫ"
_ACRONYM_RE = re.compile(rf"[{_UPPER}]{{2,}}")
_DIGIT_RE = re.compile(r"\d")


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s)


def _lower(s: str) -> str:
    return s.strip().lower()


def _no_punct(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _canon(s: str) -> str:
    """Full canonical form (accent/sigma/punct/case/space-insensitive)."""
    t = _strip_accents(_lower(s)).replace("ς", "σ")
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return _collapse(t)


def _phonetic_skeleton(tok: str) -> str:
    """Collapse Greek homophone classes so misspellings of the same sound match."""
    t = _strip_accents(_lower(tok)).replace("ς", "σ")
    t = _no_punct(t)
    # digraphs first
    t = t.replace("αι", "ε")
    for dg in ("ει", "οι", "υι"):
        t = t.replace(dg, "ι")
    # vowel classes
    t = re.sub(r"[ηιυ]", "ι", t)
    t = t.replace("ω", "ο")
    # collapse doubled letters
    t = re.sub(r"(.)\1+", r"\1", t)
    return t


def categorize(before: str, after: str) -> str:
    if before is None:
        before = ""
    if after is None:
        after = ""

    cb, ca = _canon(before), _canon(after)

    # --- normalised-equal: formatting-level differences ---
    if cb == ca:
        if before.strip() == after.strip():
            return "no_change"
        # case+punct-normalised forms (accents and final sigma still present)
        b1, a1 = _no_punct(_lower(before)), _no_punct(_lower(after))
        b1, a1 = _collapse(b1), _collapse(a1)
        if b1 != a1:
            if _strip_accents(b1) == _strip_accents(a1):
                return "accent_tonos"            # only accents differ
            if b1.replace("ς", "σ") == a1.replace("ς", "σ"):
                return "final_sigma"             # only σ/ς differ
            return "accent_tonos"                # accent-ish mix
        return "punctuation_capitalization"      # only punctuation/case/space

    # --- real lexical change ---
    from collections import Counter
    tb, ta = cb.split(), ca.split()
    cnt_b, cnt_a = Counter(tb), Counter(ta)

    # number/date: the set of digit groups changed (added, removed, or altered)
    if re.findall(r"\d+", before) != re.findall(r"\d+", after):
        return "number_date"

    # acronym introduced in after (before word-boundary: «δ ε υ α»->«ΔΕΥΑ»)
    after_acr = set(_ACRONYM_RE.findall(after))
    before_acr = set(_ACRONYM_RE.findall(before))
    if after_acr - before_acr:
        return "acronym_abbreviation"

    # word boundary: identical letters once spaces are removed (join/split only)
    if cb.replace(" ", "") == ca.replace(" ", ""):
        return "word_boundary"

    # named entity: a new capitalised proper noun appears in `after`
    # (checked before insertion_deletion so adding/removing a name isn't masked)
    after_caps = re.findall(rf"\b[{_UPPER}][\wά-ώΆ-Ώ]+", after)
    before_caps = re.findall(rf"\b[{_UPPER}][\wά-ώΆ-Ώ]+", before)
    if set(after_caps) - set(before_caps):
        return "named_entity"

    # token-count change with one-sided difference -> insertion/deletion
    only_b, only_a = cnt_b - cnt_a, cnt_a - cnt_b
    if abs(len(tb) - len(ta)) >= 1 and (not only_b or not only_a):
        return "insertion_deletion"

    # homophone / morphology: compare the multiset of changed tokens
    changed_b = list((cnt_b - cnt_a).elements())
    changed_a = list((cnt_a - cnt_b).elements())
    if len(changed_b) == len(changed_a) and changed_b:
        if all(_phonetic_skeleton(x) == _phonetic_skeleton(y)
               for x, y in zip(changed_b, changed_a)):
            return "homophone"
        if all(x[:3] == y[:3] for x, y in zip(changed_b, changed_a)):
            return "morph_grammar"

    return "other_lexical"
