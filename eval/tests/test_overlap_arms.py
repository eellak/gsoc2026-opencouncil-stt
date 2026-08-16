"""Tests 1 and 2 of `docs/specs/2026-08-16-overlap-speaker-arms-prereg.md` §5.

They are the evaluation criterion for the arm implementation: the mask must be
cut-independent, the splice must conserve every token outside it, and the placebo must
be dose-matched, deterministic across processes, and must refuse rather than under-dose.
"""
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import overlap_arms as OA          # noqa: E402
from eval.controlled_eval.fusion_lab import TRIO, Window     # noqa: E402


def seg(spk, a, b):
    return {"speaker": spk, "start": a, "end": b}


# A timeline with one overlap ({A} -> {A,B} -> {B}), then two clean turns well after it.
REG = [seg("A", 0.0, 5.0), seg("B", 4.5, 10.0), seg("C", 12.0, 15.0),
       seg("A", 16.0, 18.0)]

ANCHOR_TOKENS = ["a1", "a2", "a3", "a4", "a5", "s1", "s2", "z1"]
ANCHOR_TIMES = [1.0, 2.0, 3.0, 6.0, 7.0, 13.0, 13.5, 17.0]
AUDIO_END = 18.0

W_TOKENS = list(ANCHOR_TOKENS)
SCRIBE = list(ANCHOR_TOKENS)
SONIOX = ["a1", "a2", "X", "a4", "a5", "s1", "s2", "z1"]
ADAPTER = ["a1", "a2", "X", "a4", "a5", "s1", "s2", "z1"]
OUTSIDE = ["s1", "s2", "z1"]          # every token whose time is outside the mask


def fake_stream(item_id, reg=None):
    return (ANCHOR_TOKENS, ANCHOR_TIMES, reg if reg is not None else REG, [], AUDIO_END)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(OA, "_turbo_stream", fake_stream)
    OA._CTX.clear()
    yield
    OA._CTX.clear()


def window(item_id="win_test_1"):
    return Window(item_id=item_id, city="testcity", meeting="m1", ref=list(W_TOKENS),
                  hyps=[SCRIBE, SONIOX, ADAPTER], pivot=0, cols=[], decisions=[],
                  w_tokens=list(W_TOKENS), v_tokens=list(SCRIBE), in_training=False)


# --------------------------------------------------------------------------- mask
def test_mask_is_overlap_plus_its_two_neighbouring_turns(patched):
    assert OA.OverlapArm.mask(REG) == [(0.0, 10.0)]


def test_mask_ignores_a_timeline_with_no_overlap(patched):
    clean = [seg("A", 0.0, 4.0), seg("B", 5.0, 9.0)]
    assert OA.OverlapArm.mask(clean) == []


def test_mask_is_the_same_for_every_arm(patched):
    """It must not depend on the cut source - that is the whole point."""
    masks = {c.name: c.mask(REG) for c in OA.ARMS.values()}
    assert len(set(map(tuple, masks.values()))) == 1


# ------------------------------------------------------ test 1: splice conservation
@pytest.mark.parametrize("arm_name", sorted(OA.ARMS))
def test_tokens_outside_the_mask_survive_exactly_once_and_in_order(patched, arm_name):
    out, info = OA.ARMS[arm_name]().build(OA.context(window()))
    assert info["matched"] is True
    assert [t for t in out if t in OUTSIDE] == OUTSIDE
    # and W's own inside-mask tokens are the ones that were replaced
    assert info["replaced"] == 5


def test_speaker_and_placebo_replace_the_same_region(patched):
    ctx = OA.context(window())
    spans = OA.OverlapArm.mask(ctx.reg)
    a, ia = OA.TurnSelect().build(ctx)
    b, ib = OA.TurnSelectPlacebo().build(ctx)
    assert ia["replaced"] == ib["replaced"]
    assert a[-len(OUTSIDE):] == b[-len(OUTSIDE):] == OUTSIDE
    assert spans == [(0.0, 10.0)]


def test_select_arm_emits_the_cell_local_winner(patched):
    out, _ = OA.TurnSelect().build(OA.context(window()))
    # cut at 5.0 splits the mask; in [0,5) two of three systems say "X" for "a3"
    assert out == ["a1", "a2", "X", "a4", "a5", "s1", "s2", "z1"]


def test_mask_arm_has_no_interior_cut(patched):
    ctx = OA.context(window())
    _, info = OA.MaskSelect().build(ctx)
    assert info["dose"] == 0
    _, sinfo = OA.TurnSelect().build(ctx)
    assert sinfo["dose"] == 1          # the {A}->{A,B}->{B} handover at t=5.0


def test_no_overlap_means_output_is_W_untouched(patched, monkeypatch):
    clean = [seg("A", 0.0, 4.0), seg("B", 5.0, 9.0)]
    monkeypatch.setattr(OA, "_turbo_stream", lambda i: fake_stream(i, clean))
    OA._CTX.clear()
    for cls in OA.ARMS.values():
        out, info = cls().build(OA.context(window("win_clean")))
        assert out == W_TOKENS
        assert info["replaced"] == 0


# ---------------------------------------------- test 2: placebo matching + determinism
def test_placebo_seed_is_sha256_not_the_runtime_hash():
    expect = int.from_bytes(
        hashlib.sha256(b"win_x|placebo|1").digest()[:8], "big")
    assert OA.OverlapArm.placebo_seed("win_x", "placebo", 1) == expect


def test_placebo_dose_equals_the_speaker_dose(patched):
    ctx = OA.context(window())
    spans = OA.OverlapArm.mask(ctx.reg)
    from eval.controlled_eval.exp_speaker_fusion import handover_cuts
    k = len(OA.OverlapArm.interior(handover_cuts(ctx.reg)[0], spans))
    for draw in (1, 2, 7):
        pc = OA.OverlapArm.placebo_cuts(ctx, spans, k, draw)
        assert pc is not None and len(pc) == k == 1
        assert len(set(pc)) == len(pc)
        assert all(0.0 < t < 10.0 for t in pc)


def test_placebo_is_identical_across_processes(patched):
    """Same draw index, same cuts - the seed may not depend on PYTHONHASHSEED."""
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from eval.controlled_eval.overlap_arms import OverlapArm as O;"
        "print(O.placebo_seed('win_test_1','placebo',3))" % str(ROOT))
    outs = set()
    for hs in ("0", "1", "12345"):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env={"PYTHONHASHSEED": hs, "PATH": "/usr/bin",
                                           "HOME": str(Path.home())})
        outs.add(r.stdout.strip())
    assert len(outs) == 1 and outs != {""}


def test_short_pool_refuses_rather_than_under_dosing(patched, monkeypatch):
    """A placebo that quietly takes fewer cuts under-doses the control in exactly the
    hardest windows. It must return None so the window is excluded and counted."""
    ctx = OA.context(window())
    spans = OA.OverlapArm.mask(ctx.reg)
    assert OA.OverlapArm.placebo_cuts(ctx, spans, 99, 1) is None
    arm = OA.TurnSelectPlacebo()
    monkeypatch.setattr(OA.OverlapArm, "cuts_for", lambda self, c, s: None)
    out, info = arm.build(ctx)
    assert out is None and info["matched"] is False


def test_unmatched_window_falls_back_to_W_in_apply(patched, monkeypatch):
    monkeypatch.setattr(OA.OverlapArm, "cuts_for", lambda self, c, s: None)
    assert OA.TurnSelectPlacebo().apply(window(), None) == W_TOKENS
