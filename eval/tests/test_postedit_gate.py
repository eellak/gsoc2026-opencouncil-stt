"""The output-validity gate has to catch the failure mode and leave real edits alone.

Both halves matter. A gate that rejects everything suspicious would also throw away
the corrections that make post-editing worth doing, and would look fine in any metric
that only counts disasters avoided.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.exp_postedit_gate import gate, repair  # noqa: E402

SRC = ("ο δήμαρχος ανέφερε ότι το έργο της ανάπλασης στην οδό Καραολή "
       "θα ξεκινήσει τον Σεπτέμβριο")


def test_accepts_a_genuine_correction():
    """Fixing a name and an ending is exactly what the editor is for."""
    ok, reason = gate(SRC, SRC.replace("Καραολή", "Καραολή και Δημητρίου"))
    assert ok, reason


def test_accepts_an_unchanged_output():
    ok, reason = gate(SRC, SRC)
    assert ok, reason


def test_rejects_meta_commentary():
    ok, reason = gate(SRC, "Το διορθωμένο κείμενο είναι:\n\n" + SRC)
    assert not ok
    assert reason in ("meta_marker", "multi_paragraph")


def test_rejects_a_refusal():
    ok, reason = gate(SRC, "Δεν μπορώ να διορθώσω αυτό το κείμενο χωρίς περισσότερα συμφραζόμενα.")
    assert not ok


def test_rejects_an_explanation_appended_after_the_text():
    """The observed failure: a correct transcript, then the model keeps talking."""
    ok, reason = gate(SRC, SRC + "\n\nΣημείωση: άλλαξα το όνομα της οδού επειδή ...")
    assert not ok
    assert reason == "multi_paragraph"


def test_rejects_empty_output():
    assert not gate(SRC, "")[0]
    assert not gate(SRC, "   ")[0]


def test_rejects_truncation():
    ok, reason = gate(SRC, "ο δήμαρχος ανέφερε")
    assert not ok
    assert reason == "length_ratio"


def test_rejects_a_paraphrase_that_keeps_the_length():
    """Same length, different words: the tell is edit distance, not token count."""
    para = ("ο δήμαρχος δήλωσε πως οι εργασίες αναμόρφωσης επί της λεωφόρου Κύπρου "
            "πρόκειται να αρχίσουν εντός του φθινοπώρου")
    ok, reason = gate(SRC, para)
    assert not ok
    assert reason == "rewrote_too_much"


def test_rejects_the_documented_disaster():
    """The 2026-07-29 case: a 17-word utterance answered with 60+ edits of commentary."""
    src = " ".join(f"λέξη{i}" for i in range(17))
    out = ("Παρατήρησα ότι το κείμενο περιέχει αρκετά λάθη αναγνώρισης. "
           "Ακολουθεί η διορθωμένη εκδοχή με σχόλια για κάθε αλλαγή που έκανα, "
           "ώστε να είναι σαφές τι άλλαξε και γιατί σε κάθε σημείο του κειμένου.")
    assert not gate(src, out)[0]


def test_a_long_but_faithful_edit_survives():
    """Many small fixes across a long utterance must not trip the rewrite threshold."""
    src = " ".join(["ο δημαρχος ειπε οτι"] * 5)
    out = " ".join(["ο δήμαρχος είπε ότι"] * 5)
    ok, reason = gate(src, out)
    assert ok, reason


def test_repair_keeps_the_first_paragraph():
    assert repair(SRC + "\n\nΣημείωση: ...") == SRC
    assert repair("") == ""


@pytest.mark.parametrize("bad", ["```\n" + SRC + "\n```", '"' + SRC + '"', "- " + SRC])
def test_rejects_markup_wrappers(bad):
    assert not gate(SRC, bad)[0]
