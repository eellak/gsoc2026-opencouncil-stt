"""Canonical reader for the unreviewed-meeting denylist.

Single source of truth: data/exclusions/unreviewed_meetings.json (the 13 meetings
with human-edit fraction <5% as of 2026-06-23 — essentially never reviewed). Every
Python consumer that builds "our set" should drop these, keyed strictly by
(city_id, meeting_id) since meeting_id slugs collide across cities.

Disable for an opt-in raw build with DISABLE_UNREVIEWED_MEETING_EXCLUSIONS=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "exclusions" / "unreviewed_meetings.json"

_TRUTHY_OFF = {"", "0", "false", "no", "off"}


def exclusions_disabled() -> bool:
    return os.environ.get("DISABLE_UNREVIEWED_MEETING_EXCLUSIONS", "").strip().lower() \
        not in _TRUTHY_OFF


def load_excluded_keys(path: Path | str = DEFAULT_PATH) -> set[tuple[str, str]]:
    """Set of (city_id, meeting_id). Empty if disabled or file missing."""
    if exclusions_disabled():
        return set()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    return {(m["city_id"], m["meeting_id"]) for m in data.get("meetings", [])}


def is_excluded(city_id: str, meeting_id: str,
                keys: set[tuple[str, str]] | None = None) -> bool:
    if keys is None:
        keys = load_excluded_keys()
    return (city_id, meeting_id) in keys
