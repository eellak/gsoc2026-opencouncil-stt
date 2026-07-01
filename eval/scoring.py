"""Layered, Greek-normalised scoring for the fix-task eval harness.

Spec: docs/specs/fix-task-eval-harness.md §5.

Normalise (accent-, punctuation-, final-sigma-, whitespace-, case-insensitive)
then compute:
  - edit_application — was the targeted before->after correction span applied?
  - normalized_exact  — canonical-form exact match (sanity).
  - overcorrection    — tokens gold left unchanged that the model changed (harm).
  - surface_fidelity  — un-normalised similarity (punctuation/casing/digits).
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter

from rapidfuzz.distance import Levenshtein

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def greek_normalize(s: str) -> str:
    """Canonical form: lowercase, accents stripped, final sigma unified,
    punctuation removed, whitespace collapsed."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    # strip combining diacritics (τόνοι / διαλυτικά)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFC", s)
    # final sigma -> medial sigma so «όπως»/«οπωσ» compare equal
    s = s.replace("ς", "σ")
    # punctuation -> space, then collapse whitespace
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _norm_tokens(s: str) -> list[str]:
    n = greek_normalize(s)
    return n.split() if n else []


def cer(hyp: str, ref: str) -> float:
    """Character error rate of `hyp` against reference `ref`, on the Greek-
    normalised char form (accent/sigma/punct/case/whitespace-insensitive).

    Denominator is len(normalised ref). Edge cases are explicit so callers never
    divide by zero:
      - ref and hyp both normalise to empty -> 0.0 (nothing to transcribe, got nothing)
      - ref empty, hyp non-empty            -> 1.0 (any output is fully spurious)
    For the faithfulness pipeline empty-ref rows are dropped upstream, so this
    only guards degenerate inputs.
    """
    nh, nr = greek_normalize(hyp), greek_normalize(ref)
    if not nr:
        return 0.0 if not nh else 1.0
    return Levenshtein.distance(nh, nr) / len(nr)


def wer(hyp: str, ref: str) -> float:
    """Word error rate of `hyp` vs `ref` on Greek-normalised tokens (token-level
    Levenshtein / len(ref tokens)). Same zero-ref guard as `cer`."""
    th, tr = _norm_tokens(hyp), _norm_tokens(ref)
    if not tr:
        return 0.0 if not th else 1.0
    return Levenshtein.distance(th, tr) / len(tr)


def score_pair(input_raw: str, model_output: str, gold: str) -> dict:
    """Score one corrected line against the gold reference.

    Multiset-based (robust to duplicate tokens; no positional aliasing). The
    gold "edit" is the set of tokens it adds and removes relative to the input;
    the model gets credit only when it actually achieves each — adds the needed
    token AND removes the spurious one. Harm counts every token the model
    introduced or dropped that gold did not (catches hallucinated insertions and
    over-deletion), not merely unchanged tokens it failed to echo.
    """
    ni, nm, ng = _norm_tokens(input_raw), _norm_tokens(model_output), _norm_tokens(gold)
    ci, cm, cg = Counter(ni), Counter(nm), Counter(ng)

    normalized_exact = nm == ng

    # what gold changes relative to input
    gold_adds = cg - ci      # tokens gold introduces
    gold_removes = ci - cg   # tokens gold deletes
    needed = sum(gold_adds.values()) + sum(gold_removes.values())

    if needed == 0:
        edit_application = 1.0  # nothing to fix => trivially satisfied
    else:
        # achieved add: the needed token is present in the model output
        achieved_add = sum(min(n, cm.get(t, 0)) for t, n in gold_adds.items())
        # achieved remove: the spurious token is gone from the model output
        achieved_remove = 0
        for t, n in gold_removes.items():
            remaining = cm.get(t, 0)
            # gold wants this token's count reduced from ci[t] to cg[t]
            target_removed = n
            actual_removed = max(0, ci[t] - remaining)
            achieved_remove += min(target_removed, actual_removed)
        edit_application = (achieved_add + achieved_remove) / needed

    # harm: tokens the model added or dropped that gold did not
    model_adds = cm - ci
    model_removes = ci - cm
    harmful_adds = model_adds - gold_adds          # invented / overcorrected
    harmful_removes = model_removes - gold_removes  # over-deleted
    harm = sum(harmful_adds.values()) + sum(harmful_removes.values())
    overcorrection = harm / max(1, len(ni))

    # Surface fidelity: un-normalised char similarity (casing/punct/digits visible).
    surface_fidelity = difflib.SequenceMatcher(
        None, (model_output or "").strip(), (gold or "").strip()
    ).ratio()

    # Word error rate of the corrected output vs the human-final gold (token-level,
    # on the normalised forms). Lower is better; this is the ASR-style error metric.
    if ng:
        wer = Levenshtein.distance(nm, ng) / len(ng)
    else:
        wer = 0.0 if not nm else 1.0

    return {
        "normalized_exact": normalized_exact,
        "edit_application": edit_application,
        "overcorrection": overcorrection,
        "surface_fidelity": surface_fidelity,
        "wer": wer,
    }
