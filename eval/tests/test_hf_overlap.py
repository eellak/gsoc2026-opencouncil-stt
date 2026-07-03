"""Tests for the reviewer-note overlap convention (standalone Latin C)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.hf_export.overlap import has_overlap_marker, notes_report_rows


def test_bare_c_matches():
    assert has_overlap_marker("C") is True
    assert has_overlap_marker("c") is True
    assert has_overlap_marker("  C  ") is True


def test_standalone_c_inside_text_matches():
    assert has_overlap_marker("C και θόρυβος") is True
    assert has_overlap_marker("κακό κόψιμο, C") is True
    assert has_overlap_marker("overlap (C) στο τέλος") is True


def test_c_inside_latin_word_does_not_match():
    assert has_overlap_marker("check this") is False
    assert has_overlap_marker("ASCII") is False
    assert has_overlap_marker("cut") is False


def test_none_and_empty_do_not_match():
    assert has_overlap_marker(None) is False
    assert has_overlap_marker("") is False
    assert has_overlap_marker("καθαρό") is False


def test_notes_report_partitions_matched_unmatched():
    rows = [
        {"utterance_id": "u1", "reviewer_notes": "C"},
        {"utterance_id": "u2", "reviewer_notes": "κάτι άλλο"},
        {"utterance_id": "u3", "reviewer_notes": None},
        {"utterance_id": "u4", "reviewer_notes": "c θόρυβος"},
    ]
    matched, unmatched = notes_report_rows(rows)
    assert [r["utterance_id"] for r in matched] == ["u1", "u4"]
    assert [r["utterance_id"] for r in unmatched] == ["u2"]  # u3 empty -> excluded
