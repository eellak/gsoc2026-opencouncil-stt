"""Correctness of the served-config / July-adapter harness, model-free.

`exp-2026-08-17-served-config-and-july-adapter`. Everything here runs without a
model, an audio file or a decode, and every assertion is something that had to hold
before the multi-hour decode was allowed to start:

- the primary text representation is production's per-segment join, and it really
  does differ from the fused `"".join` that `decode_ablation` wrote;
- an arm's config differs from CONTROL in exactly the keys it declares;
- WER = (S+D+I)/n, count deltas telescope, and the difference-of-differences is
  computed as declared rather than inferred from the telescoping;
- domination shares are direct contributions and are *not* the same statistic as
  leave-one-out sensitivity;
- the arm/contrast tables are internally consistent.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402

import decode_ablation as DA  # noqa: E402
import served_config_and_july as M  # noqa: E402


# ------------------------------------------------------- text assembly (blocking)
# faster-whisper does not always prefix a segment with a space. The fixture below is
# the shape that produced 505 fused boundaries in exp-2026-08-16-adapter-confidence.
FUSED = [" ο πρόεδρος είπε", "ναι στο δημοτικό", " συμβούλιο"]


def test_per_segment_tokens_match_the_server_join():
    """The primary representation is what the server's `" ".join(strip)` produces."""
    server_text = " ".join(s.strip() for s in FUSED).strip()
    assert M.tokens_per_segment(FUSED) == ftoks(server_text)


def test_legacy_join_fuses_words_across_a_boundary():
    """The bug this harness exists to avoid, demonstrated rather than asserted."""
    legacy = M.tokens_legacy("".join(FUSED).strip())
    primary = M.tokens_per_segment(FUSED)
    assert legacy != primary
    # the frozen normalizer strips accents, so the fused token is "ειπεναι"
    assert "ειπεναι" in legacy and "ειπεναι" not in primary
    assert len(legacy) == len(primary) - 1


def test_leading_space_segments_are_unaffected():
    segs = [" ένα δύο", " τρία τέσσερα"]
    assert M.tokens_per_segment(segs) == M.tokens_legacy("".join(segs).strip())


# ---------------------------------------------------------------- arm definitions
def test_every_arm_differs_from_control_in_exactly_its_declared_keys():
    for arm, (_, over, _) in M.ARMS.items():
        kw = M.config_for(arm)
        assert {k for k in kw if kw[k] != DA.CONTROL[k]} == set(over), arm
        assert {k: v for k, v in kw.items() if k not in over} == \
               {k: v for k, v in DA.CONTROL.items() if k not in over}, arm


def test_served_arms_encode_the_deployment():
    """asr.env sets OC_ASR_BEAM=2; the product routes pass word_timestamps=True."""
    assert M.config_for("S1")["beam_size"] == 2
    assert M.config_for("S1")["word_timestamps"] is False
    assert M.config_for("S2")["beam_size"] == 2
    assert M.config_for("S2")["word_timestamps"] is True
    assert M.config_for("R") == DA.CONTROL


def test_an_arm_cannot_override_a_key_control_does_not_have(monkeypatch):
    monkeypatch.setitem(M.ARMS, "X", (M.CT2_FIXED, {"nonesuch": 1}, "bad arm"))
    with pytest.raises(SystemExit):
        M.config_for("X")


def test_tables_are_consistent():
    assert set(M.PRIMARY_ARMS) <= set(M.ARMS)
    assert set(M.LEGACY) <= set(M.ARMS)
    assert not (set(M.PRIMARY_ARMS) & set(M.LEGACY)), \
        "a cached legacy-representation arm may not be primary"
    for a, b, _, cls in M.CONTRASTS:
        assert a in M.ARMS and b in M.ARMS
        assert cls in ("primary", "exploratory")
        if cls == "primary":
            assert a in M.PRIMARY_ARMS and b in M.PRIMARY_ARMS
    for model in {v[0] for v in M.ARMS.values()}:
        assert model in M.MODEL_SHA16
    # The July artifact's status must not be contingent on a number.
    assert "KNOWN_BROKEN regardless" in M.JULY_GUARD


# ---------------------------------------------------------------------- arithmetic
def _counts(**per_window):
    """window -> (S, D, I, n)."""
    return dict(per_window)


A = _counts(w1=(10, 5, 2, 100), w2=(4, 1, 1, 100), w3=(0, 0, 0, 100))
B = _counts(w1=(12, 4, 1, 100), w2=(4, 2, 1, 100), w3=(1, 0, 0, 100))
C = _counts(w1=(9, 6, 3, 100), w2=(5, 1, 0, 100), w3=(0, 1, 0, 100))
WIDS = ["w1", "w2", "w3"]
UNITS = ["m1", "m1", "m2"]


def test_rates_are_micro_averages():
    r = DA.rates(A)
    assert r["ref_tokens"] == 300
    assert r["wer"] == pytest.approx((10 + 5 + 2 + 4 + 1 + 1) / 300)
    assert r["wer"] == pytest.approx(r["sub_rate"] + r["del_rate"] + r["ins_rate"])


def test_count_deltas_telescope():
    """(C-A) == (C-B) + (B-A) exactly, because the denominator never changes."""
    pick = M.PICKS["wer"]
    n = sum(v[3] for v in A.values())

    def d(x, y):
        return (sum(pick(x[w]) for w in WIDS) - sum(pick(y[w]) for w in WIDS)) / n

    assert d(C, A) == pytest.approx(d(C, B) + d(B, A))


def test_dod_point_estimate_is_the_difference_of_differences():
    pick = M.PICKS["wer"]
    n = sum(v[3] for v in A.values())
    out = M.dod_ci(C, B, B, A, WIDS, UNITS, pick)

    def d(x, y):
        return (sum(pick(x[w]) for w in WIDS) - sum(pick(y[w]) for w in WIDS)) / n

    assert out["delta"] == pytest.approx(d(C, B) - d(B, A))
    assert out["ci95"][0] <= out["delta"] <= out["ci95"][1]


def test_dod_of_an_arm_against_itself_is_zero():
    out = M.dod_ci(C, A, C, A, WIDS, UNITS, M.PICKS["wer"])
    assert out["delta"] == pytest.approx(0.0)


def test_level_ci_brackets_the_point_estimate():
    out = M.level_ci(A, WIDS, UNITS, M.PICKS["wer"])
    assert out["value"] == pytest.approx(DA.rates(A)["wer"])
    assert out["ci95"][0] <= out["value"] <= out["ci95"][1]


# ---------------------------------------------------------------------- domination
def test_domination_shares_are_direct_contributions_not_loo_shifts():
    """One window carries the whole net move while another moves twice as far."""
    base = _counts(w1=(0, 0, 0, 100), w2=(0, 0, 0, 100), w3=(0, 0, 0, 100))
    arm = _counts(w1=(0, 10, 0, 100), w2=(0, 0, 0, 100), w3=(0, 0, 0, 100))
    out = M.domination(arm, base, WIDS, WIDS, M.PICKS["del_rate"])
    assert out["top_unit"] == "w1"
    assert out["top_unit_error_change"] == 10
    assert out["top_unit_signed_share"] == pytest.approx(1.0)
    assert out["top_unit_gross_share"] == pytest.approx(1.0)
    # The LOO shift is a different number from the share, and must not be quoted
    # as one: dropping w1 removes the whole effect but also shrinks the denominator.
    assert out["max_loo_shift"] != pytest.approx(out["top_unit_signed_share"])


def test_domination_detects_cancellation():
    """Net zero, gross 40 - the shape the near-zero deletion delta warning is about."""
    a = _counts(w1=(0, 20, 0, 100), w2=(0, 0, 0, 100), w3=(0, 0, 0, 100))
    b = _counts(w1=(0, 0, 0, 100), w2=(0, 20, 0, 100), w3=(0, 0, 0, 100))
    out = M.domination(a, b, WIDS, WIDS, M.PICKS["del_rate"])
    ch = M.churn(a, b, WIDS, M.PICKS["del_rate"])
    assert out["net_error_change"] == 0
    assert out["gross_error_movement"] == 40
    assert ch["net"] == 0 and ch["abs_sum"] == 40
    assert ch["windows_up"] == 1 and ch["windows_down"] == 1
    assert ch["windows_unchanged"] == 1


def test_domination_meeting_units_aggregate_windows():
    a = _counts(w1=(0, 3, 0, 100), w2=(0, 4, 0, 100), w3=(0, 0, 0, 100))
    b = _counts(w1=(0, 0, 0, 100), w2=(0, 0, 0, 100), w3=(0, 0, 0, 100))
    out = M.domination(a, b, WIDS, UNITS, M.PICKS["del_rate"])
    assert out["top_unit"] == "m1"           # w1 + w2 together
    assert out["top_unit_error_change"] == 7


# ------------------------------------------------------------------- sdi agreement
def test_sdi_totals_match_the_wer_definition():
    ref = ftoks("το δημοτικό συμβούλιο εγκρίνει το θέμα")
    hyp = ftoks("το δημοτικό συμβούλιο εγκρίνει θέμα δύο")
    s, d, i = sdi(ref, hyp)
    assert (s + d + i) / len(ref) == pytest.approx(M.PICKS["wer"]((s, d, i, len(ref)))
                                                   / len(ref))


# --------------------------------------------------------------------- the freeze
def test_the_substrate_is_the_frozen_39_and_excludes_the_sealed_windows():
    rows = DA.rows("eval")
    assert len(rows) == 39
    assert sum(r["ref_tokens"] for r in rows) == 11911
    assert len({r["meeting_id"] for r in rows}) == 31
    sealed = {w["window_id"] for w in DA.manifest()["holdout_windows"]}
    assert len(sealed) == 7
    assert not ({r["window_id"] for r in rows} & sealed)


def test_city_meeting_blocks_are_finer_than_the_frozen_meeting_blocks():
    """The frozen key merges two different cities' apr7_2026 meetings into one block."""
    rows = DA.rows("eval")
    assert len({f"{r['city']}/{r['meeting_id']}" for r in rows}) == 32
