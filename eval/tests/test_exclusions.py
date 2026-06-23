"""Tests for the unreviewed-meeting denylist reader."""
import importlib
import os

from eval import exclusions


def test_loads_thirteen_meetings():
    keys = exclusions.load_excluded_keys()
    assert len(keys) == 13


def test_rhodes_excluded_but_same_slug_other_city_not():
    # rhodes/jul17_2025 is denylisted; the same meeting_id slug in another city
    # must NOT be — slugs collide across cities, so we key by (city, meeting).
    keys = exclusions.load_excluded_keys()
    assert exclusions.is_excluded("rhodes", "jul17_2025", keys)
    assert not exclusions.is_excluded("athens", "jul17_2025", keys)
    assert not exclusions.is_excluded("rhodes", "some_other_meeting", keys)


def test_thessaloniki_apr1_excluded():
    assert exclusions.is_excluded("thessaloniki", "apr1_2026")


def test_env_disable(monkeypatch):
    monkeypatch.setenv("DISABLE_UNREVIEWED_MEETING_EXCLUSIONS", "1")
    importlib.reload(exclusions)
    try:
        assert exclusions.load_excluded_keys() == set()
        assert not exclusions.is_excluded("rhodes", "jul17_2025")
    finally:
        monkeypatch.delenv("DISABLE_UNREVIEWED_MEETING_EXCLUSIONS", raising=False)
        importlib.reload(exclusions)
