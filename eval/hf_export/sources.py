"""Helpers for combining the three dataset sources into one row set.

Sources (all mapped to the rows.parquet schema, spec docs/specs/hf-dataset-export.md):
  - includes:  review-UI include rows (live export)          source="correction"
  - leftover:  NB2 audio-verified balanced edits not included source="correction"
  - backbone:  trusted no-edit ASR from review-exposed mtgs   source="no_edit"

This module holds the pure, unit-tested glue: date-from-slug parsing and
span-level dedupe. The I/O loaders live in build.py.
"""
from __future__ import annotations

import re

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
     "nov", "dec"])}
_SLUG = re.compile(r"^([a-z]{3})(\d{1,2})")
_YEAR = re.compile(r"(20\d\d)")


def slug_date(meeting_id: str) -> str | None:
    """`aug14_2025` / `oct30_2_2025` / `apr28_2025_2` -> `2025-08-14 00:00:00`.

    OpenCouncil meeting slugs are date-based; a trailing `_N` (before or after
    the year) is a same-day meeting counter. Returns None if unparseable.
    """
    if not meeting_id:
        return None
    m = _SLUG.match(meeting_id)
    y = _YEAR.search(meeting_id)
    if not m or not y:
        return None
    mon, day = m.group(1), int(m.group(2))
    if mon not in _MONTHS or not (1 <= day <= 31):
        return None
    return f"{y.group(1)}-{_MONTHS[mon]:02d}-{day:02d} 00:00:00"


def span_key(row: dict) -> tuple:
    """Canonical clip identity: audio file + start/end rounded to 20ms.

    Used to dedupe across sources by the actual audio span, not just the
    utterance_id (Codex review: a no-edit candidate must never re-publish a
    span already present as a correction)."""
    return (row["audio_url"], round(float(row["start"]), 2),
            round(float(row["end"]), 2))


def dedupe_by_span(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep one row per audio span. Priority = each row's `_rank` (lower wins):
    includes 0 < leftover 1 < backbone 2, so a correction always beats a
    no-edit on the same span, and an include beats a leftover. Preserves input
    order among kept rows. Returns (kept, n_dropped)."""
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    for r in rows:
        k = span_key(r)
        if k not in best:
            best[k] = r
            order.append(k)
        elif r["_rank"] < best[k]["_rank"]:
            best[k] = r
    kept = [best[k] for k in order]
    return kept, len(rows) - len(kept)
