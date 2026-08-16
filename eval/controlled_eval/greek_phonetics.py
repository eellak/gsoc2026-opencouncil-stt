#!/usr/bin/env python3
"""Greek phonemic skeletons, for deciding when two spellings sound the same.

Written for wayfinder #24 arm H. The question it has to answer is narrow: given two
tokens that two ASR systems proposed for the SAME position in the SAME audio, could
the audio itself have distinguished them? If not, the column is a spelling choice and
an acoustic model can never settle it — that is exactly the residual
`exp-2026-08-11-wer-levers-research` called "homophone orthography the audio cannot
decide".

TWO maps, deliberately, because the honest answer depends on how much you fold:

  STRICT — only the equivalences that modern Greek orthography genuinely leaves
  ambiguous to a listener:
      /i/  ι η υ ει οι υι
      /o/  ο ω
      /e/  ε αι
      /u/  ου
      final sigma ς = σ
      doubled consonants collapse (λλ=λ, μμ=μ, σσ=σ, ...)
  Everything else is preserved. Two tokens with the same STRICT key are homophones
  under any reasonable Greek phonology; a listener cannot tell them apart.

  LOOSE — STRICT plus the mappings where the *spelling convention* differs but the
  sound is close rather than identical:
      μπ→b  ντ→d  γκ/γγ→g  τσ→c  τζ→j
      αυ/ευ/ηυ → af/av, ef/ev, if/iv by the voicing of what follows
      ξ→ks  ψ→ps
  LOOSE over-merges on purpose (μπ→b erases the /mb/–/b/ distinction, ντ→d erases
  /nd/). It is reported beside STRICT so the census shows what the folding buys, and
  STRICT is the preregistered primary definition.

Neither map is a pronunciation dictionary and neither knows about stress: the
project's scorer already strips diacritics, so stress is invisible upstream of here.
"""
from __future__ import annotations

import re
import unicodedata

VOICELESS = set("θκξπστφχψ")

# strict: only the orthographic ambiguities of modern Greek
_STRICT_DIGRAPHS = {
    "ου": "u",
    "αι": "e",
    "ει": "i", "οι": "i", "υι": "i",
}
_STRICT_SINGLES = {
    "η": "i", "ι": "i", "υ": "i",
    "ω": "o", "ο": "o",
    "ε": "e", "α": "a",
    "ς": "σ",
}

_LOOSE_DIGRAPHS = {
    "μπ": "b", "ντ": "d", "γκ": "g", "γγ": "g", "τσ": "c", "τζ": "j",
}
_LOOSE_SINGLES = {"ξ": "ks", "ψ": "ps"}

# Consonants are transliterated in BOTH maps. This is pure relabelling for STRICT
# (no two Greek consonants share a latin symbol here), and it is what makes the
# LOOSE map comparable at all: αυ->"af" has to be able to match α+φ.
_CONS = {
    "β": "v", "γ": "g", "δ": "D", "ζ": "z", "θ": "8", "κ": "k", "λ": "l",
    "μ": "m", "ν": "n", "π": "p", "ρ": "r", "σ": "s", "τ": "t", "φ": "f",
    "χ": "x", "ξ": "3", "ψ": "4",
}


def _base(s: str) -> str:
    """Lowercase, diacritic-free — the same shape `scoring.norm` produces."""
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s).lower().replace("ς", "σ")


GREEK_CONSONANTS = "βγδζθκλμνξπρσςτφχψ"


def _collapse_source(s: str) -> str:
    """Collapse doubled CONSONANTS in the source string only: λλ -> λ, σσ -> σ.

    Codex job 55293f6b found the bug this replaces. Collapsing runs in the PRODUCED
    key merges vowel positions that different source graphemes created: ποιητης maps
    to p-i-i-t-i-s, and collapsing the two /i/ makes it collide with πιτης, which is a
    different word with a different number of syllables. Doubled consonants are a
    genuine orthographic-only distinction in Greek; doubled vowels are not.
    """
    return re.sub(r"([" + GREEK_CONSONANTS + r"])\1+", r"\1", s)


def phon(token: str, loose: bool = False) -> str:
    """Phonemic skeleton of one token. Equal keys == 'the audio cannot decide'."""
    s = _collapse_source(_base(token))
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        pair = s[i:i + 2]
        if loose and pair in ("αυ", "ευ", "ηυ"):
            nxt = s[i + 2] if i + 2 < n else ""
            v = "f" if (nxt in VOICELESS or nxt == "") else "v"
            out.append({"αυ": "a", "ευ": "e", "ηυ": "i"}[pair] + v)
            i += 2
            continue
        if pair in ("αυ", "ευ", "ηυ"):
            # STRICT keeps these opaque rather than letting the υ fall through to
            # /i/: ευα and εια are not homophones, and folding the υ would claim
            # they are.
            out.append({"αυ": "aY", "ευ": "eY", "ηυ": "iY"}[pair])
            i += 2
            continue
        if pair in _STRICT_DIGRAPHS:
            out.append(_STRICT_DIGRAPHS[pair])
            i += 2
            continue
        if loose and pair in _LOOSE_DIGRAPHS:
            out.append(_LOOSE_DIGRAPHS[pair])
            i += 2
            continue
        ch = s[i]
        if ch in _STRICT_SINGLES:
            v = _STRICT_SINGLES[ch]
            out.append(_CONS.get(v, v))
        elif loose and ch in _LOOSE_SINGLES:
            out.append(_LOOSE_SINGLES[ch])
        else:
            out.append(_CONS.get(ch, ch))
        i += 1
    return "".join(out)


def homophones(tokens, loose: bool = False) -> bool:
    """True when every token shares one phonemic key AND they are not all identical."""
    ts = [t for t in tokens if t]
    if len(ts) < 2 or len(set(ts)) < 2:
        return False
    return len({phon(t, loose=loose) for t in ts}) == 1
