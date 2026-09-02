"""Tests for the retrieval-biased hotwords policy (spec section 3.4).

Record: exp-2026-08-27-retrieval-hotwords. The closed static arm
(exp-2026-07-25-hotwords) stays closed; this policy differs from it by making the
prompt CONDITIONAL on the unbiased pass-1 output.

The first test in this file is the policy contract and is deliberately written
before any implementation exists. It fixes the three things that decide whether the
arm measures what it claims:

  1. pass 2 differs from pass 1 by the `hotwords` kwarg and by NOTHING else;
  2. the window-only seed is identical across both passes (common random numbers,
     so the contrast is the prompt and not the sampling);
  3. an empty retrieved set produces no second decode at all, and the pass-1 text
     survives byte for byte as an exact zero-delta observation.

Everything else in the file guards the retrieval parameters that section 3.4
requires to be frozen before any decode runs.

No model and no tokenizer is loaded; both are stubbed.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.retrieval_hotwords import (  # noqa: E402
    RetrievalConfig, retrieve, run_policy)


# --------------------------------------------------------------------------
# Fixtures. The near miss is real: the pending v2 alias conflict recorded in
# exp-2026-08-11-name-repair. Measured skeleton distances under both folds are
# Adraktas 1, Drakos 4, Antoniou 7, Papadopoulos 8 -- so a threshold of 1 or 2
# separates the target from every distractor without tuning.
# --------------------------------------------------------------------------
NEAR_MISS = "Αδραχτάς"
TARGET = "Αδρακτάς"
DISTRACTORS = ["Παπαδόπουλος", "Αντωνίου", "Δράκος"]
ROSTER = [TARGET] + DISTRACTORS

PASS1_WITH_NAME = f"ο κύριος {NEAR_MISS} έχει τον λόγο"
PASS1_NO_NAME = "προχωρούμε στο επόμενο θέμα της ημερήσιας διάταξης"

WINDOW_ID = "win_argos_apr30_2_2026_1264428"

# The frozen CONTROL of notebooks/decode_ablation.py:44-62, minus the model handle.
CONTROL_KWARGS = dict(
    language="el", task="transcribe", beam_size=5, best_of=5, patience=1,
    condition_on_previous_text=False, word_timestamps=False, vad_filter=False,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    no_speech_threshold=0.6, log_prob_threshold=-1.0,
    compression_ratio_threshold=2.4,
    initial_prompt=None, prefix=None,
)

CFG = RetrievalConfig(
    loose=False, max_distance=1, min_skeleton_len=4, k_max=3,
    token_budget=160, include_exact=False,
)


class FakeTok:
    """Stub tokenizer: one id per whitespace chunk, matching the real interface."""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False, \
            "budget must be counted the way faster-whisper encodes (no specials)"
        return SimpleNamespace(ids=[0] * len(text.split()))


class Recorder:
    """Fake decoder plus seed sink. Records every decode call in order."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = []      # list of (audio, kwargs)
        self.seeds = []      # every value passed to set_seed, in order

    def decode(self, audio, **kwargs):
        self.calls.append((audio, dict(kwargs)))
        return self._texts[len(self.calls) - 1]

    def set_seed(self, value):
        self.seeds.append(value)


def seed_for(window_id):
    """Window-only seed: no arm term, so both passes get the same draw."""
    import hashlib
    return int(hashlib.sha256(window_id.encode()).hexdigest()[:4], 16)


def _run(recorder, pass1_text=None, roster=ROSTER, cfg=CFG, wid=WINDOW_ID):
    return run_policy(
        window_id=wid,
        audio=f"/audio/{wid}.wav",
        lexicon=roster,
        cfg=cfg,
        decoder=recorder.decode,
        set_seed=recorder.set_seed,
        seed_for=seed_for,
        base_kwargs=CONTROL_KWARGS,
        tokenizer=FakeTok(),
        pass1_text=pass1_text,
    )


# ==========================================================================
# THE POLICY CONTRACT -- the first red test
# ==========================================================================

def test_two_pass_contract_differs_only_by_hotwords():
    """Pass 1 unbiased, retrieval finds the roster form, pass 2 identical but for
    `hotwords`, same seed both times, and the pass-2 text is what the policy returns."""
    rec = Recorder([PASS1_WITH_NAME, f"ο κύριος {TARGET} έχει τον λόγο"])
    res = _run(rec, pass1_text=None)

    # (4) the same audio is decoded twice
    assert len(rec.calls) == 2, "expected an unbiased pass 1 and a biased pass 2"
    assert rec.calls[0][0] == rec.calls[1][0]

    # (5) call 1 is exactly the frozen control, with no biasing of any kind
    assert rec.calls[0][1] == CONTROL_KWARGS
    for forbidden in ("hotwords", "initial_prompt", "prefix"):
        assert rec.calls[0][1].get(forbidden) is None

    # (3) retrieval returned exactly the target, with its diagnostics
    assert [r.surface for r in res.retrieved] == [TARGET]
    got = res.retrieved[0]
    assert got.distance == 1
    assert got.rank == 0
    assert got.matched_token == NEAR_MISS
    assert got.skeleton and isinstance(got.skeleton, str)

    # (7) call 2 == call 1 plus exactly one key
    c1, c2 = rec.calls[0][1], rec.calls[1][1]
    assert set(c2) - set(c1) == {"hotwords"}
    assert {k: v for k, v in c2.items() if k != "hotwords"} == c1
    assert c2["hotwords"] == TARGET
    assert c2["prefix"] is None, \
        "faster-whisper ignores hotwords entirely when prefix is set"

    # (6) identical window-only seed before each decode
    assert rec.seeds == [seed_for(WINDOW_ID), seed_for(WINDOW_ID)]

    # (8) the policy returns the pass-2 text
    assert res.text == f"ο κύριος {TARGET} έχει τον λόγο"
    assert res.pass2_ran is True


def test_empty_retrieval_skips_pass_two_and_preserves_pass_one():
    """(9) A window with nothing to retrieve is an exact zero-delta observation."""
    rec = Recorder([PASS1_NO_NAME])
    res = _run(rec, pass1_text=None)

    assert res.retrieved == []
    assert res.pass2_ran is False
    assert len(rec.calls) == 1, "no second decode may be issued"
    assert res.text == PASS1_NO_NAME, "pass-1 text must survive byte for byte"


def test_cached_pass_one_issues_only_the_biased_decode():
    """The real arm reuses ~/.cache/oc-public/decode-ablation/eval-A.json rather than
    re-deriving pass 1, because exp-2026-08-16-adapter-confidence withdrew the
    bit-exact reproducibility claim. Supplying it must skip call 1, not duplicate it."""
    rec = Recorder([f"ο κύριος {TARGET} έχει τον λόγο"])
    res = _run(rec, pass1_text=PASS1_WITH_NAME)

    assert len(rec.calls) == 1
    assert rec.calls[0][1]["hotwords"] == TARGET
    assert rec.seeds == [seed_for(WINDOW_ID)]
    assert res.pass1_from_cache is True


def test_cached_empty_retrieval_issues_no_decode_at_all():
    rec = Recorder([])
    res = _run(rec, pass1_text=PASS1_NO_NAME)

    assert rec.calls == []
    assert rec.seeds == []
    assert res.text == PASS1_NO_NAME
    assert res.pass2_ran is False


# ==========================================================================
# Retrieval parameters that section 3.4 requires frozen
# ==========================================================================

def test_retrieval_cannot_see_the_reference():
    """Structural guard: retrieval takes the hypothesis and the lexicon, nothing else.
    A reference or scorer argument would make the arm reference-informed."""
    params = set(inspect.signature(retrieve).parameters)
    assert params == {"pass1_text", "lexicon", "cfg"}
    for banned in ("reference", "ref", "gold", "scorer", "sdi", "truth"):
        assert banned not in params


def test_exact_match_is_not_retrieved_when_include_exact_is_false():
    """A token the model already spells correctly needs no prompt; retrieving it
    spends budget and prev-context for nothing."""
    out = retrieve(f"ο κύριος {TARGET} μίλησε", ROSTER, CFG)
    assert out == []


def test_distance_threshold_is_a_boundary_not_a_ranking():
    """Distractors sit at 4, 7 and 8. Nothing beyond max_distance may be returned,
    however few candidates survive."""
    out = retrieve(PASS1_WITH_NAME, ROSTER, CFG)
    assert [r.surface for r in out] == [TARGET]
    assert all(r.distance <= CFG.max_distance for r in out)

    tight = RetrievalConfig(**{**CFG.__dict__, "max_distance": 0})
    assert retrieve(PASS1_WITH_NAME, ROSTER, tight) == []


def test_k_max_is_a_hard_cap():
    """'Typically 0-3 names' is not a constraint; K <= 3 is (Codex review a471616b)."""
    crowd = [TARGET, "Αδραχτής", "Αδραχτός", "Αδρεχτάς", "Αδραχτάν"]
    out = retrieve(PASS1_WITH_NAME, crowd, RetrievalConfig(
        **{**CFG.__dict__, "max_distance": 2}))
    assert len(out) <= CFG.k_max


def test_token_budget_counted_with_the_real_tokenizer_whole_entries_only():
    rec = Recorder([PASS1_WITH_NAME, "x"])
    res = _run(rec, cfg=RetrievalConfig(**{**CFG.__dict__, "token_budget": 0}))
    assert res.retrieved == [], "nothing fits in a zero budget"
    assert res.pass2_ran is False


def test_short_skeletons_are_not_eligible():
    """Two- and three-letter skeletons collide with ordinary Greek and would fire
    everywhere. min_skeleton_len is frozen, not tuned after seeing a number."""
    out = retrieve("και να πω ότι ναι", ["Νάι", "Πω"], CFG)
    assert out == []


def test_retrieval_is_deterministic_and_order_independent():
    a = retrieve(PASS1_WITH_NAME, ROSTER, CFG)
    b = retrieve(PASS1_WITH_NAME, list(reversed(ROSTER)), CFG)
    key = lambda rs: [(r.surface, r.distance, r.rank) for r in rs]  # noqa: E731
    assert key(a) == key(b)
    assert key(a) == key(retrieve(PASS1_WITH_NAME, ROSTER, CFG))


def test_strict_and_loose_folds_are_not_interchangeable():
    """greek_phonetics.phon has two modes and they disagree on exactly the cases
    that matter. The spec requires the choice to be frozen in writing, so the config
    must carry it rather than defaulting silently."""
    assert "loose" in inspect.signature(RetrievalConfig).parameters
    with pytest.raises(TypeError):
        RetrievalConfig(max_distance=1, min_skeleton_len=4, k_max=3,
                        token_budget=160, include_exact=False)


def test_k_max_cannot_be_widened_past_the_preregistered_cap():
    """K <= 3 is frozen in the spec. A config that widens it must fail at
    construction, not silently at the slice (CodeRabbit, 2026-08-27)."""
    for bad in (4, 10, -1):
        with pytest.raises(ValueError):
            RetrievalConfig(**{**CFG.__dict__, "k_max": bad})
    with pytest.raises(TypeError):
        RetrievalConfig(**{**CFG.__dict__, "k_max": "3"})
    for ok in (0, 1, 2, 3):
        assert RetrievalConfig(**{**CFG.__dict__, "k_max": ok}).k_max == ok
