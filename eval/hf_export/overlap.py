"""Reviewer-note overlap convention.

The reviewer marks overlapping speech (someone else talking over the
utterance) with a standalone Latin letter C in `reviewer_notes`. Standalone =
not adjacent to another Latin letter, so "C", "c, κακό" match but "check"
does not. Greek text around it is fine. See docs/specs/hf-dataset-export.md.
"""
from __future__ import annotations

import re

_STANDALONE_C = re.compile(r"(?<![A-Za-z])[cC](?![A-Za-z])")


def has_overlap_marker(note: str | None) -> bool:
    if not note:
        return False
    return bool(_STANDALONE_C.search(note))


def notes_report_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition rows with non-empty reviewer_notes into (matched, unmatched).

    Feeds the human-eyeball report published before the dataset: every note
    the rule flagged as overlap, and every note it did not.
    """
    matched, unmatched = [], []
    for r in rows:
        note = r.get("reviewer_notes")
        if not note or not str(note).strip():
            continue
        (matched if has_overlap_marker(note) else unmatched).append(r)
    return matched, unmatched
