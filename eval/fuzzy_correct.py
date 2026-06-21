"""Deterministic fuzzy corrector for names/toponyms (no LLM).

Hypothesis under test: name/toponym ASR errors can be fixed in CODE by matching
tokens against the mined glossary, recovering most named-entity errors WITHOUT
the cross-category overcorrection that injecting the glossary into the LLM prompt
caused. This module is the standalone corrector; the eval lives in
`eval.fuzzy_eval`.

Design follows the Codex review (2026-06-21):
  - separate MATCH normalisation (case/accent/final-sigma-insensitive) from the
    scorer's normalisation; always emit the glossary's canonical spelling;
  - length-dependent edit-distance limits, not a single similarity cutoff;
  - a candidate margin (best must beat second-best) to avoid ambiguous pulls;
  - never replace a token already match-equivalent to a glossary term;
  - longest multi-token phrases first, then single tokens, no overlaps;
  - "name-like" typing proxy (proper-noun = leading uppercase) to keep ordinary
    lowercase glossary words out of single-token fuzzy replacement.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import DamerauLevenshtein

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _match_norm(s: str) -> str:
    """Case/accent/final-sigma-insensitive form, for MATCHING only."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s).replace("ς", "σ")


def _max_edit(n: int) -> int:
    """Allowed edit distance by (match-normalised) token length."""
    if n <= 5:
        return 1
    if n <= 9:
        return 2
    return 3


def _name_like(term: str) -> bool:
    """Typing proxy: a proper noun / acronym starts with an uppercase letter.
    Excludes ordinary lowercase glossary words from fuzzy single-token swaps."""
    return term[:1].isupper()


def _best_two(norm_tok: str, choices: list[tuple[str, str]]) -> tuple:
    """Return (best_canonical, best_dist, second_dist) over (norm, canonical)."""
    best_c, best_d, second_d = None, 10**9, 10**9
    n = len(norm_tok)
    for cn, canon in choices:
        if abs(len(cn) - n) > 3:
            continue
        d = DamerauLevenshtein.distance(norm_tok, cn)
        if d < best_d:
            best_c, second_d, best_d = canon, best_d, d
        elif d < second_d:
            second_d = d
    return best_c, best_d, second_d


def _prep(terms, name_like_only):
    singles, phrases = [], []
    seen_s, seen_p = set(), set()
    for t in terms:
        t = t.strip()
        if not t:
            continue
        parts = t.split()
        if len(parts) == 1:
            if name_like_only and not _name_like(t):
                continue
            cn = _match_norm(t)
            if len(cn) >= 4 and cn not in seen_s:
                seen_s.add(cn)
                singles.append((cn, t))
        else:
            if name_like_only and not _name_like(t):
                continue
            key = " ".join(_match_norm(p) for p in parts)
            if key not in seen_p:
                seen_p.add(key)
                phrases.append((key, t, len(parts)))
    # longest phrases first so the greedy span match prefers them
    phrases.sort(key=lambda x: -x[2])
    return singles, phrases


def fuzzy_correct(text: str, terms, *, name_like_only: bool = True,
                  phrases: bool = True, margin: int = 1,
                  max_dist_cap: int | None = None, min_len: int = 4,
                  allowed_spans: list | None = None) -> str:
    """Return `text` with name/toponym tokens snapped to glossary spellings.

    `margin`: the best candidate's edit distance must be at least `margin`
    better than the second-best (1 = strictly unique best) — guards against a
    token sitting equally close to two names.
    `max_dist_cap`: hard ceiling on the allowed edit distance (e.g. 1 = only
    single-character slips), on top of the length-dependent limit.
    `min_len`: minimum (match-normalised) token length to consider.
    `allowed_spans`: if given, a list of (char_start, char_end) ranges (e.g. NER
    entity spans); only tokens overlapping one of them may be replaced. This is
    the NER gate — it stops the corrector touching non-entity words.
    """
    def edit_limit(n: int) -> int:
        lim = _max_edit(n)
        return min(lim, max_dist_cap) if max_dist_cap is not None else lim

    def _allowed(i: int) -> bool:
        if allowed_spans is None:
            return True
        s, e = pieces[i][2], pieces[i][3]
        return any(s < ae and a0 < e for (a0, ae) in allowed_spans)

    singles, phrase_list = _prep(terms, name_like_only)

    # tokenise into (text, is_word, start, end) pieces; offsets drive the NER gate
    pieces, last = [], 0
    for m in _WORD_RE.finditer(text):
        if m.start() > last:
            pieces.append((text[last:m.start()], False, last, m.start()))
        pieces.append((m.group(0), True, m.start(), m.end()))
        last = m.end()
    if last < len(text):
        pieces.append((text[last:], False, last, len(text)))

    word_idx = [i for i, p in enumerate(pieces) if p[1]]
    replaced = [False] * len(pieces)

    # phrase pass (greedy, longest first), over consecutive word tokens
    if phrases and phrase_list:
        for pkey, canon, plen in phrase_list:
            for wi in range(len(word_idx) - plen + 1):
                idxs = word_idx[wi:wi + plen]
                if any(replaced[i] for i in idxs):
                    continue
                if not all(_allowed(i) for i in idxs):
                    continue  # NER gate: whole phrase must be inside an entity span
                span = " ".join(_match_norm(pieces[i][0]) for i in idxs)
                if span == pkey:
                    continue  # already correct
                if DamerauLevenshtein.distance(span, pkey) <= edit_limit(len(pkey)):
                    s0, e0 = pieces[idxs[0]][2], pieces[idxs[-1]][3]
                    pieces[idxs[0]] = (canon, True, s0, e0)
                    for i in idxs[1:]:
                        pieces[i] = ("", True, pieces[i][2], pieces[i][3])
                    for i in idxs:
                        replaced[i] = True

    # single-token pass
    if singles:
        for i in word_idx:
            if replaced[i] or not _allowed(i):
                continue
            tok = pieces[i][0]
            nt = _match_norm(tok)
            if len(nt) < min_len:
                continue
            best_c, best_d, second_d = _best_two(nt, singles)
            if best_c is None or best_d == 0:
                continue  # no candidate, or already match-equivalent (correct)
            if best_d <= edit_limit(len(nt)) and (second_d - best_d) >= margin:
                pieces[i] = (best_c, True, pieces[i][2], pieces[i][3])
                replaced[i] = True

    # reassemble, collapsing the blanks left by phrase replacements
    out = "".join(p[0] for p in pieces)
    return re.sub(r"\s+", " ", out).strip()
