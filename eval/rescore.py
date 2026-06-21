"""Recompute category + scores from stored model outputs.

Scoring and categorisation are pure functions of (input, output, gold), all of
which the A/B runner persists per row. Recomputing at report time means metric
or categorizer fixes apply uniformly to every row — including rows scored under
an earlier version — without re-calling the LLM.
"""
from __future__ import annotations

from eval.categorize import categorize
from eval.scoring import score_pair


def enrich(rec: dict) -> dict:
    inp = rec.get("input_raw", "")
    gold = rec.get("gold_final", "")
    out = dict(rec)
    out["category"] = categorize(inp, gold)
    for arm in ("baseline", "glossary"):
        if arm in rec and isinstance(rec[arm], dict):
            s = score_pair(inp, rec[arm].get("output", ""), gold)
            out[arm] = {**rec[arm], **s}
    return out
