"""Unit tests for `eval/tse_overlap.py` — the pure parts.

These guard the arithmetic that the preregistration's gates depend on. The
sign of G2 in particular was wrong in the first draft of the spec (it would have
passed when the wrong enrollment helped MORE than the right one), so it gets an
explicit adversarial case here.
"""
import numpy as np
import pytest

from eval import tse_overlap as T


def _tone(n, f=200.0, sr=T.SR, amp=1.0, phase=0.0):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)


# ------------------------------------------------------------------ levels
def test_active_rms_ignores_silence():
    """A block that is half silence must not read as half as loud — otherwise
    '0 dB SIR' would not mean equal loudness where both people are talking."""
    loud = _tone(T.SR)
    padded = np.concatenate([loud, np.zeros(T.SR, np.float32)])
    assert T.rms(padded) < 0.75 * T.rms(loud)
    assert T.active_rms(padded) == pytest.approx(T.active_rms(loud), rel=1e-3)


def test_active_rms_falls_back_on_short_input():
    x = _tone(100)
    assert T.active_rms(x) == pytest.approx(T.rms(x))


def test_mix_at_sir_sets_the_ratio_and_never_rescales_the_target():
    tgt = _tone(T.SR * 2, f=200)
    msk = _tone(T.SR * 2, f=350, amp=0.05)
    for sir in (0.0, 5.0, -3.0):
        t, m, tiled = T.mix_at_sir(tgt, msk, sir)
        assert np.array_equal(t, tgt), "the target must never be rescaled"
        got = 20 * np.log10(T.active_rms(t) / T.active_rms(m))
        assert got == pytest.approx(sir, abs=0.1)
        assert tiled is False


def test_mix_at_sir_tiles_only_a_short_masker():
    tgt = _tone(T.SR * 2)
    _, m_short, tiled_short = T.mix_at_sir(tgt, _tone(T.SR // 2, f=350), 0.0)
    assert tiled_short is True and m_short.size == tgt.size
    rng = np.random.default_rng(0)
    _, m_long, tiled_long = T.mix_at_sir(tgt, _tone(T.SR * 5, f=350), 0.0, rng)
    assert tiled_long is False and m_long.size == tgt.size


def test_mix_at_sir_survives_a_silent_masker():
    tgt = _tone(T.SR)
    t, m, _ = T.mix_at_sir(tgt, np.zeros(T.SR, np.float32), 0.0)
    assert np.array_equal(t, tgt) and not m.any()


def test_peak_norm_matches_wesep_output_norm():
    x = _tone(1000, amp=0.3)
    y = T.peak_norm(x)
    assert np.max(np.abs(y)) == pytest.approx(T.OUT_PEAK, abs=1e-6)
    assert not T.peak_norm(np.zeros(10, np.float32)).any()


def test_common_attenuation_is_shared_and_only_attenuates():
    quiet = {"a": _tone(100, amp=0.1)}
    assert T.common_attenuation(quiet) == 1.0
    loud = {"a": _tone(100, amp=2.0), "b": _tone(100, amp=0.5)}
    att = T.common_attenuation(loud)
    assert np.max(np.abs(loud["a"] * att)) == pytest.approx(T.PEAK_CEIL, abs=1e-6)


# ------------------------------------------------------------------ si-sdr
def test_si_sdr_is_scale_invariant_and_ranks_the_right_source():
    a, b = _tone(T.SR, f=200), _tone(T.SR, f=900)
    assert T.si_sdr(a * 0.01, a) > 100
    # scale invariance, checked on an imperfect estimate: a perfect one lands at
    # ~150 dB where float32 rounding is larger than any tolerance worth setting
    est = (a + 0.1 * b).astype(np.float32)
    assert T.si_sdr(est * 0.01, a) == pytest.approx(T.si_sdr(est * 7.0, a), abs=1e-4)
    assert T.si_sdr(a + b, a) > T.si_sdr(a + b, _tone(T.SR, f=1500))
    assert np.isnan(T.si_sdr(np.zeros(0, np.float32), a))
    assert np.isnan(T.si_sdr(a, np.zeros(T.SR, np.float32)))


# ------------------------------------------------------------- item selection
def _blocks():
    return [
        {"id": "b0", "s": 0.0, "e": 4.0, "spk": "A", "text": "a b c d e f", "ov_with": []},
        {"id": "b1", "s": 5.0, "e": 9.0, "spk": "A", "text": "g h i j k l", "ov_with": []},
        {"id": "b2", "s": 10.0, "e": 12.0, "spk": "B", "text": "m n o p q", "ov_with": ["b3"]},
        {"id": "b3", "s": 11.0, "e": 13.0, "spk": "C", "text": "r s t u v",
         "ov_with": ["b2"], "text_unc": True},
    ]


def test_is_clean_block_excludes_overlap_and_uncertain():
    b = _blocks()
    assert T.is_clean_block(b[0]) and T.is_clean_block(b[1])
    assert not T.is_clean_block(b[2]), "overlap-participating is not clean"
    assert not T.is_clean_block(b[3]), "uncertain text is not clean"


def test_enrollment_span_holds_out_the_target_and_enforces_the_minimum():
    b = _blocks()
    # b1 is held out as the target, leaving b0's 4 s -> above the 3 s minimum
    assert T.enrollment_span(b, "A", "b1") == [(0.0, 4.0)]
    # holding out both of A's blocks leaves nothing
    assert T.enrollment_span([b[0]], "A", "b0") == []
    # B's only block participates in overlap, so it is not enrollable at all
    assert T.enrollment_span(b, "B", None) == []


def test_enrollment_span_caps_at_the_frozen_maximum():
    long = [{"id": f"b{i}", "s": i * 20.0, "e": i * 20.0 + 20.0, "spk": "A",
             "text": "x", "ov_with": []} for i in range(3)]
    segs = T.enrollment_span(long, "A", None)
    assert sum(e - s for s, e in segs) == pytest.approx(T.MAX_ENROLL_SEC)


def test_arm_files_tags_only_the_level_dependent_arms():
    f = T.arm_files("i007", "0")
    assert set(f) == set(T.STAGE1_ARMS)
    for a in T.SIR_FREE:
        assert f[a] == f"i007.{a}"
    assert f["MIX_NORM"] == "i007.MIX_NORM.0"
    assert f["TSE"] == "i007.TSE.0"


# ------------------------------------------------------------------- gates
def _res(tse, mix_norm, wrong, clean_norm, tse_clean, del_tse=0.10, del_mix=0.10,
         loo=None, top_share=0.1):
    def arm(w, d=0.1):
        return {"wer": {"point": w, "ci95_meeting_cluster": [w - .1, w + .1]},
                "del_rate": {"point": d}, "ins_rate": {"point": 0.0}}
    return {
        "arms": {"TSE": arm(tse, del_tse), "MIX_NORM": arm(mix_norm, del_mix),
                 "TSE_WRONG": arm(wrong), "CLEAN_NORM": arm(clean_norm),
                 "TSE_CLEAN": arm(tse_clean)},
        "paired": {"TSE-MIX_NORM": {"leave_one_meeting_out":
                                    loo if loo is not None else {"m1": -0.1, "m2": -0.1}}},
        "domination": {"top_share": top_share},
    }


def test_gates_pass_when_tse_separates_and_targets():
    g = T.gates(_res(0.20, 0.40, 0.38, 0.20, 0.21), primary=True)
    assert g["all_pass"] is True
    assert g["G1_separates"]["d_right"] == pytest.approx(0.20)


def test_g2_fails_when_the_wrong_enrollment_helps_as_much():
    """The adversarial case the first draft of the spec would have passed:
    generic enhancement, not target extraction."""
    g = T.gates(_res(0.20, 0.40, 0.20, 0.20, 0.21), primary=True)
    assert g["G2_targets_not_enhances"]["pass"] is False
    assert g["all_pass"] is False


def test_g2_fails_when_the_wrong_enrollment_helps_more():
    g = T.gates(_res(0.30, 0.40, 0.10, 0.20, 0.21), primary=True)
    assert g["G2_targets_not_enhances"]["contrast_C"] > 0
    assert g["G2_targets_not_enhances"]["pass"] is False


def test_g1_fails_on_a_sign_flip_in_one_meeting():
    g = T.gates(_res(0.20, 0.40, 0.38, 0.20, 0.21, loo={"m1": -0.2, "m2": +0.01}),
                primary=True)
    assert g["G1_separates"]["pass"] is False


def test_g1_fails_when_one_item_supplies_most_of_the_gain():
    g = T.gates(_res(0.20, 0.40, 0.38, 0.20, 0.21, top_share=0.67), primary=True)
    assert g["G1_separates"]["pass"] is False


def test_g3_fails_when_tse_damages_clean_audio():
    g = T.gates(_res(0.20, 0.40, 0.38, 0.20, 0.30), primary=True)
    assert g["G3_safe_off_target"]["pass"] is False
    assert g["G3_safe_off_target"]["cost_on_clean"] == pytest.approx(0.10)


def test_g4_fails_when_wer_is_bought_with_deletions():
    g = T.gates(_res(0.20, 0.40, 0.38, 0.20, 0.21, del_tse=0.20, del_mix=0.10),
                primary=True)
    assert g["G4_no_deletion_purchase"]["pass"] is False


def test_gates_are_not_evaluated_at_the_secondary_sir():
    g = T.gates(_res(0.20, 0.40, 0.38, 0.20, 0.21), primary=False)
    assert g["evaluated"] is False and "all_pass" not in g


def test_gates_report_missing_arms_rather_than_passing():
    res = _res(0.20, 0.40, 0.38, 0.20, 0.21)
    del res["arms"]["TSE_WRONG"]
    g = T.gates(res, primary=True)
    assert "error" in g and "all_pass" not in g


# -------------------------------------------------------------- domination
def test_domination_finds_the_largest_contributor_to_a_gain():
    per = {"a": {"TSE": {"err": 0}, "MIX_NORM": {"err": 8}},
           "b": {"TSE": {"err": 3}, "MIX_NORM": {"err": 4}},
           "c": {"TSE": {"err": 5}, "MIX_NORM": {"err": 5}}}
    d = T.domination(per, "TSE", "MIX_NORM")
    assert d["total_err_diff"] == -9 and d["top_item"] == "a"
    assert d["top_share"] == pytest.approx(8 / 9)


def test_domination_is_defined_when_nothing_changed():
    per = {"a": {"TSE": {"err": 2}, "MIX_NORM": {"err": 2}}}
    assert T.domination(per, "TSE", "MIX_NORM")["top_share"] is None
