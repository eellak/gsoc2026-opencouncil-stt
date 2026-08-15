"""Prereg guarantees of the arm D N-best selector, model-free.

What must hold before the one scoring pass is trusted:
- lambda=0 reproduces the beam top-1 identically (identity of the selector);
- dedup after frozen normalization keeps the best original beam rank;
- ties go to the smaller lambda, then to the lower beam rank;
- an alternative is taken only on a STRICTLY positive combined score;
- selection and LM training are deterministic;
- a missing KenLM file is a hard error, never a silent fallback.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.serving_stack.d_selector import (  # noqa: E402
    GRID, dedup_candidates, fold_scales, load_lm, pick_lambda, rms,
    select_chunk, select_window, train_kn_arpa)


def hyps(*pairs):
    return [{"text": t, "score": s} for t, s in pairs]


FLAT_LM = lambda text: 0.0                       # noqa: E731  LM contributes nothing


# ------------------------------------------------------------ lambda=0 identity
def test_lambda0_reproduces_top1():
    # alternatives have better LM scores but (as CT2 guarantees) worse A
    chunks = [
        dedup_candidates(hyps((" a b", -0.10), (" a c", -0.20), (" a d", -0.30))),
        dedup_candidates(hyps((" e f", -0.50), (" e g", -0.55))),
    ]
    lm = lambda text: 100.0 if "c" in text or "g" in text else -100.0  # noqa: E731
    text, ranks = select_window(chunks, 0.0, 1.0, 1.0, lm)
    assert ranks == [0, 0]
    assert text == "a b e f"                      # pipeline join + strip


def test_lambda0_keeps_top1_even_on_score_tie():
    cands = dedup_candidates(hyps((" x", -0.3), (" y", -0.3)))
    assert select_chunk(cands, 0.0, 1.0, 1.0, FLAT_LM)["rank"] == 0


# ----------------------------------------------------------------------- dedup
def test_dedup_keeps_best_rank():
    # ranks 1 and 3 normalize identically; the group keeps rank 1's identity
    cands = dedup_candidates(hyps(
        (" alpha", -0.1), (" beta!", -0.2), (" gamma", -0.3), (" beta", -0.4)))
    assert [c["rank"] for c in cands] == [0, 1, 2]
    kept = cands[1]
    assert kept["text"] == " beta!" and kept["A"] == -0.2


def test_dedup_top1_group_absorbs_duplicates():
    cands = dedup_candidates(hyps((" a", -0.1), (" a.", -0.2), (" b", -0.3)))
    assert [c["rank"] for c in cands] == [0, 2]
    assert cands[0]["A"] == -0.1


# ----------------------------------------------------------------- tie-breaking
def test_lambda_tie_goes_to_smaller():
    assert pick_lambda({lam: 10 for lam in GRID}) == 0.0
    totals = {lam: (5 if lam in (0.5, 2.0) else 10) for lam in GRID}
    assert pick_lambda(totals) == 0.5


def test_candidate_tie_goes_to_lower_rank():
    # both alternatives beat top1 by exactly the same combined score
    cands = dedup_candidates(hyps((" t", -0.5), (" u", -0.4), (" v", -0.4)))
    assert select_chunk(cands, 0.0, 1.0, 1.0, FLAT_LM)["rank"] == 1


# ------------------------------------------------------------ strict positivity
def test_zero_combined_score_keeps_top1():
    # A tie + LM tie -> combined score exactly 0 -> top1 stays
    cands = dedup_candidates(hyps((" p", -0.4), (" q", -0.4)))
    assert select_chunk(cands, 4.0, 1.0, 1.0, FLAT_LM)["rank"] == 0


def test_strictly_positive_score_switches():
    cands = dedup_candidates(hyps((" p", -0.4), (" q", -0.4)))
    lm = lambda text: 1.0 if "q" in text else 0.0        # noqa: E731
    assert select_chunk(cands, 0.25, 1.0, 1.0, lm)["rank"] == 1
    # ...but not when the acoustic loss exactly cancels the LM gain
    cands2 = dedup_candidates(hyps((" p", -0.4), (" q", -0.65)))
    assert select_chunk(cands2, 0.25, 1.0, 1.0, lm)["rank"] == 0


# ---------------------------------------------------------------- determinism
def test_selection_and_training_are_deterministic():
    chunks = [dedup_candidates(hyps((" a b", -0.1), (" a c", -0.15)))]
    lm = lambda text: -len(text)                          # noqa: E731
    assert (select_window(chunks, 2.0, 0.5, 3.0, lm)
            == select_window(chunks, 2.0, 0.5, 3.0, lm))
    corpus = [["το", "νερό", "της", "πόλης"], ["το", "νερό", "κόπηκε"],
              ["η", "πόλη", "ψήφισε"]]
    assert train_kn_arpa(corpus) == train_kn_arpa(corpus)


def test_scales_are_positive_and_stable():
    assert rms([]) == 1.0
    assert rms([0.0, 0.0]) == 1.0                # degenerate fold -> unit scale
    chunks = [dedup_candidates(hyps((" a", -0.1), (" b", -0.3)))]
    s_a, s_l = fold_scales([chunks], FLAT_LM)
    assert s_a == pytest.approx(0.2) and s_l == 1.0


# ------------------------------------------------------------- missing LM file
def test_missing_kenlm_file_is_a_hard_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="KenLM model missing"):
        load_lm(tmp_path / "nope.arpa")
