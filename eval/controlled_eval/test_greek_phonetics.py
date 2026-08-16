"""Frozen tests for the Greek phonemic skeletons of wayfinder #24.

The homophone arm is only as honest as this map. Two failure directions matter and
are tested separately:

  UNDER-MERGE  two spellings a Greek listener genuinely cannot tell apart get
               different keys, so the arm never sees the column.
  OVER-MERGE   two spellings that sound different get one key, so the arm is asked
               to choose between words the audio COULD have decided - which is
               exactly the kind of licence that turned an LLM arbiter harmful in
               wayfinder #18.

    .venv-eval/bin/python -m pytest eval/controlled_eval/test_greek_phonetics.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.controlled_eval.greek_phonetics import homophones, phon  # noqa: E402


# --------------------------------------------------------------- must merge
MERGE_STRICT = [
    ("παιδί", "πεδί"),            # αι = ε
    ("είναι", "ίνε"),             # ει = ι, αι = ε
    ("κόστος", "κώστος"),         # ο = ω
    ("υγεία", "ιγία"),            # υ = ι, ει = ι
    ("άλλος", "άλος"),            # doubled consonant
    ("φοιτητής", "φυτητίς"),      # οι = ι = η  (φυ- via υ->i)
    ("τέλος", "τέλοσ"),           # final sigma
    ("οικονομία", "ικονομία"),
]

# things only the LOOSE map may merge
MERGE_LOOSE_ONLY = [
    ("αυτό", "αφτό"),
    ("εύκολο", "έφκολο"),
    ("μπάλα", "bάλα"),
    ("ντύνω", "dύνω"),
]

# --------------------------------------------------------------- must NOT merge
KEEP_APART_STRICT = [
    ("καλός", "κακός"),           # different consonant
    ("πόλη", "πύλη"),             # ο vs υ -> /o/ vs /i/, genuinely different
    ("ευα", "εια"),               # the reason αυ/ευ stay opaque
    ("αυτό", "αϊτό"),
    ("δήμος", "δίνος"),
    ("θέμα", "τέμα"),             # θ vs τ
]

KEEP_APART_LOOSE = [
    ("καλός", "κακός"),
    ("θέμα", "τέμα"),
    ("πόλη", "πύλη"),
]


def test_strict_merges_the_real_ambiguities():
    for a, b in MERGE_STRICT:
        assert phon(a) == phon(b), (a, b, phon(a), phon(b))
        assert homophones([a, b])


def test_strict_keeps_distinct_sounds_apart():
    for a, b in KEEP_APART_STRICT:
        assert phon(a) != phon(b), (a, b, phon(a))


def test_loose_only_merges_are_not_strict():
    for a, b in MERGE_LOOSE_ONLY:
        assert phon(a) != phon(b), ("strict must not merge", a, b)
        assert phon(a, loose=True) == phon(b, loose=True), (a, b)


def test_loose_still_keeps_real_differences_apart():
    for a, b in KEEP_APART_LOOSE:
        assert phon(a, loose=True) != phon(b, loose=True), (a, b)


def test_identical_tokens_are_not_homophones():
    # the class is "sounds the same, spelled differently"; identical spellings are
    # an agreement column, not a decision
    assert not homophones(["δήμος", "δήμος"])
    assert not homophones(["δήμος"])
    assert not homophones([])


def test_diacritics_and_case_are_already_invisible():
    assert phon("ΔΉΜΟΣ") == phon("δημος") == phon("δήμος")


def test_voicing_rule_of_the_loose_map():
    # ευ before a voiceless consonant is /ef/, before a voiced one /ev/
    assert phon("ευτυχία", loose=True).startswith("ef")
    assert phon("ευγενής", loose=True).startswith("ev")
    assert phon("αυτός", loose=True).startswith("af")
    assert phon("αύριο", loose=True).startswith("av")


def test_only_source_consonants_collapse():
    # doubled CONSONANTS are an orthography-only distinction and must merge
    assert phon("Ελλάδα") == phon("Ελάδα")
    assert phon("θάλασσα") == phon("θάλασα")
    assert phon("συνεννόηση") == phon("συνενόησι")


def test_produced_vowels_must_not_collapse():
    # Codex job 55293f6b: collapsing the KEY merges vowel positions that different
    # source graphemes created. ποιητής has three vowel slots, πίτης has two.
    assert phon("ποιητής") != phon("πίτης")
    assert phon("αα") != phon("α")


def test_loose_expansion_is_internally_consistent():
    # ξ = κσ and ψ = πσ under the LOOSE expansion; the key alphabet is latin
    # throughout, so an expanded digraph can meet the letters it expands to
    assert phon("έξω", loose=True) == phon("έκσω", loose=True)
    assert phon("ψάρι", loose=True) == phon("πσάρι", loose=True)


def test_diaeresis_is_lost_upstream_and_is_declared_not_hidden():
    # the frozen scorer strips combining marks, so the diaeresis that separates
    # αϋ from αυ is already gone before this module sees the token. STRICT keeps
    # αυ opaque, which limits but does not remove the damage.
    from eval.controlled_eval.scoring import norm
    assert norm("αϋπνία") == norm("αυπνία")
