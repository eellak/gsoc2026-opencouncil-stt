"""Train/eval splits with leakage control.

Spec: docs/specs/fix-task-eval-harness.md §2.

Split by meeting_id (never by utterance) so the glossary is mined from training
meetings only and eval rows never contribute glossary terms. A second split by
city_id gives a true zero-shot-city eval.
"""
from __future__ import annotations

import hashlib


def _stable_hash(value: str) -> int:
    return int(hashlib.md5(str(value).encode("utf-8")).hexdigest(), 16)


def split_by_meeting(chains, eval_meetings=None, eval_frac: float = 0.2):
    """Partition chains into (train, eval) by meeting_id.

    If `eval_meetings` is given, those meetings form the eval fold; otherwise a
    deterministic hash assigns ~eval_frac of meetings to eval.
    """
    if eval_meetings is None:
        meetings = sorted({c["meeting_id"] for c in chains})
        bucket = max(1, round(1 / eval_frac)) if eval_frac > 0 else 0
        eval_meetings = {m for m in meetings if bucket and _stable_hash(m) % bucket == 0}
    else:
        eval_meetings = set(eval_meetings)

    train = [c for c in chains if c["meeting_id"] not in eval_meetings]
    ev = [c for c in chains if c["meeting_id"] in eval_meetings]
    return train, ev


def split_by_city(chains, eval_cities=None, eval_frac: float = 0.2):
    """Partition chains into (train, eval) by city_id for a zero-shot-city eval."""
    if eval_cities is None:
        cities = sorted({c["city_id"] for c in chains})
        bucket = max(1, round(1 / eval_frac)) if eval_frac > 0 else 0
        eval_cities = {c for c in cities if bucket and _stable_hash(c) % bucket == 0}
    else:
        eval_cities = set(eval_cities)

    train = [c for c in chains if c["city_id"] not in eval_cities]
    ev = [c for c in chains if c["city_id"] in eval_cities]
    return train, ev


def assert_no_leakage(train, ev, key="meeting_id"):
    """Guardrail: no key value appears in both folds."""
    overlap = {c[key] for c in train} & {c[key] for c in ev}
    if overlap:
        raise AssertionError(f"leakage on {key}: {sorted(overlap)[:5]}")
