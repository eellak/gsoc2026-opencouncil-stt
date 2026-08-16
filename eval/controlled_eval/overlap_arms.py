#!/usr/bin/env python3
"""Overlap-restricted speaker-conditioned arms on top of W.

Preregistered in `docs/specs/2026-08-16-overlap-speaker-arms-prereg.md`, revised once
on Codex review `5851725675b5` before anything was run.

WHAT THIS IS. Round 2 of `exp-2026-08-16-pyannote-transcription` measured, with a
placebo control, that per-turn selection beats a placebo partition ONLY inside detected
overlap (turn minus placebo -0.00558 [-0.00888, -0.00228]); overall the placebo wins,
so speaker cuts are worse than random cuts except where people talk over each other.
That positive was never carried onto W, the per-column composition that displaced
whole-window selection as the fusion arm. These arms carry it.

THE ONE DESIGN DECISION THAT MATTERS. The treated region is fixed BEFORE any cut set
exists and is identical for every arm:

    M = union over maximal detected-overlap intervals O of
        [start of the active interval preceding O, end of the active interval after O]

If the region were instead "cells that contain overlap", it would depend on where the
cuts fall - and a speaker cut sits next to a handover by construction while a placebo
cut does not, so the placebo would replace a systematically different amount of
non-overlap text and could lose without speaker information doing any work. Every arm
here is token-for-token identical to W outside M; the endpoints of M are edges of every
partition; cut sets only change how M's interior is segmented.

The decision logic lives on `OverlapArm` (and its static methods) rather than in module
functions, because `autoresearch.impl_fingerprint` pins a factory's whole MRO but NOT
free functions in the caller's module.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.exp_speaker_fusion import (            # noqa: E402
    active_intervals, cells, handover_cuts, other_boundary_times,
    overlap_intervals, pick_by_similarity, split_tokens)
from eval.controlled_eval.exp_parakeet_voter import token_times  # noqa: E402
from eval.controlled_eval.fusion_lab import Idea, Window, TRIO   # noqa: E402
from eval.controlled_eval.msa import align3, compose             # noqa: E402
from eval.controlled_eval.scoring import wtoks                   # noqa: E402

N_PLACEBO = int(os.environ.get("N_PLACEBO", "20"))
BAND_FLOOR = 40


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def turbo_response(item_id: str) -> dict:
    """The cached pyannoteAI precision-2 response. Aborts if it is not on disk.

    No API fallback: a silent re-fetch would put a different timeline under a number
    the preregistration says is byte-identical between rounds.
    """
    p = sc() / "whisper_turbo" / f"{item_id}.json"
    if not p.exists():
        raise SystemExit(f"missing cached diarization for {item_id} ({p}); "
                         "the preregistration forbids an API fallback")
    return json.loads(p.read_text(encoding="utf-8"))["output"]


@lru_cache(maxsize=None)
def _turbo_stream(item_id: str):
    """(anchor tokens, anchor times, regular segments, audio_end) for one window."""
    d = turbo_response(item_id)
    toks, times = [], []
    for w in d["wordLevelTranscription"]:
        tt = wtoks(w["text"])
        if not tt:
            continue
        mid = (float(w["start"]) + float(w["end"])) / 2
        for t in tt:
            toks.append(t)
            times.append(mid)
    reg, exc = d["diarization"], d["exclusiveDiarization"]
    audio_end = max([float(s["end"]) for s in reg + exc]
                    + ([times[-1]] if times else []) + [0.0])
    return toks, times, reg, exc, audio_end


class Context:
    """Everything one window contributes, computed once and shared by every arm.

    Pure data preparation: token time axes and the diarization timeline. It contains
    no decision - the mask, the cells and the arm outputs are all on `OverlapArm`.
    """

    __slots__ = ("item_id", "reg", "exc", "audio_end", "toks", "times",
                 "w_tokens", "w_times", "ref", "ref_times", "anchors")

    def __init__(self, w: Window):
        pk_toks, pk_times, reg, exc, audio_end = _turbo_stream(w.item_id)
        self.item_id, self.reg, self.exc, self.audio_end = w.item_id, reg, exc, audio_end
        self.toks, self.times, self.anchors = {}, {}, {}
        for i, name in enumerate(TRIO):
            tk = list(w.hyps[i])
            tm, na = token_times(tk, pk_toks, pk_times, audio_end)
            self.toks[name], self.times[name], self.anchors[name] = tk, tm, na
        self.w_tokens = list(w.w_tokens)
        self.w_times, self.anchors["W"] = token_times(
            self.w_tokens, pk_toks, pk_times, audio_end)
        self.ref = list(w.ref)
        self.ref_times, self.anchors["REF"] = token_times(
            self.ref, pk_toks, pk_times, audio_end)


_CTX: dict[str, Context] = {}


def context(w: Window) -> Context:
    c = _CTX.get(w.item_id)
    if c is None:
        c = _CTX[w.item_id] = Context(w)
    return c


class OverlapArm(Idea):
    """Base of every arm: mask, cells, and the two ways of filling a mask cell.

    Subclasses set `cut_source` and `fill`, nothing else. `fitted` stays False - none
    of these arms fits a parameter, so leave-one-city-out is vacuous for them and
    `fusion_lab.evaluate` says so in `fold_note`.
    """
    name = "ov_base"
    fitted = False
    cut_source = "none"        # none | speaker | placebo
    fill = "select"            # select | compose
    draw = 1                   # placebo draw index

    # ---------------------------------------------------------------- the mask
    @staticmethod
    def mask(reg_segs) -> list[tuple[float, float]]:
        """Overlap plus its two neighbouring turns, merged. Cut-independent."""
        iv = active_intervals(reg_segs)
        spans = []
        for i, (s, e, sp) in enumerate(iv):
            if len(sp) < 2:
                continue
            lo = iv[i - 1][0] if i > 0 else s
            hi = iv[i + 1][1] if i + 1 < len(iv) else e
            spans.append((lo, hi))
        spans.sort()
        out: list[tuple[float, float]] = []
        for lo, hi in spans:
            if out and lo <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], hi))
            else:
                out.append((lo, hi))
        return out

    @staticmethod
    def inside(t: float, spans: list[tuple[float, float]]) -> bool:
        return any(lo <= t < hi for lo, hi in spans)

    # ---------------------------------------------------------------- the cuts
    @staticmethod
    def interior(cuts: list[float], spans: list[tuple[float, float]]) -> list[float]:
        """Cut times strictly inside a mask span. A cut on a span edge is already an
        edge of the partition and would be a duplicate, not a subdivision."""
        return sorted({t for t in cuts if any(lo < t < hi for lo, hi in spans)})

    @staticmethod
    def placebo_seed(item_id: str, name: str, draw: int) -> int:
        """SHA-256, never the runtime's `hash`: the draws must survive a new process."""
        h = hashlib.sha256(f"{item_id}|{name}|{draw}".encode()).digest()
        return int.from_bytes(h[:8], "big")

    @classmethod
    def placebo_cuts(cls, ctx: Context, spans, k: int, draw: int) -> list[float] | None:
        """k cuts drawn without replacement from the placebo pool INSIDE the mask.

        Returns None when the pool cannot supply k - the window is then unmatched and
        is excluded from the primary difference of differences rather than being given
        a smaller placebo, which would under-dose the control in exactly the hardest
        windows.
        """
        import numpy as np
        _, spans_h = handover_cuts(ctx.reg)
        pool = cls.interior(other_boundary_times(ctx.reg, spans_h), spans)
        if k == 0:
            return []
        if len(pool) < k:
            return None
        rng = np.random.default_rng(cls.placebo_seed(ctx.item_id, "placebo", draw))
        return sorted(float(x) for x in rng.choice(pool, size=k, replace=False))

    def cuts_for(self, ctx: Context, spans) -> list[float] | None:
        speaker = self.interior(handover_cuts(ctx.reg)[0], spans)
        if self.cut_source == "none":
            return []
        if self.cut_source == "speaker":
            return speaker
        return self.placebo_cuts(ctx, spans, len(speaker), self.draw)

    # ---------------------------------------------------------------- the output
    @classmethod
    def partition(cls, spans, cuts):
        """Half-open cells tiling the window; every mask endpoint is an edge."""
        edges = sorted({t for sp in spans for t in sp} | set(cuts))
        return cells(edges)

    @classmethod
    def fill_cell(cls, slices: dict[str, list[str]], how: str) -> list[str]:
        order = list(TRIO)
        winner = pick_by_similarity(slices, order)
        if how == "select":
            return list(slices[winner])
        a, b, c = (slices[p] for p in order)
        band = max(BAND_FLOOR,
                   max(len(a), len(b), len(c)) - min(len(a), len(b), len(c)) + 20)
        toks, _ = compose(align3(a, b, c, band=band), pivot=order.index(winner))
        return toks

    def build(self, ctx: Context) -> tuple[list[str], dict] | tuple[None, dict]:
        spans = self.mask(ctx.reg)
        if not spans:
            return list(ctx.w_tokens), {"matched": True, "dose": 0, "replaced": 0,
                                        "mask_spans": 0}
        cuts = self.cuts_for(ctx, spans)
        if cuts is None:
            return None, {"matched": False, "dose": len(self.interior(
                handover_cuts(ctx.reg)[0], spans)), "replaced": 0,
                "mask_spans": len(spans)}
        cs = self.partition(spans, cuts)
        w_slices = split_tokens(ctx.w_tokens, ctx.w_times, cs)
        per = {p: split_tokens(ctx.toks[p], ctx.times[p], cs) for p in TRIO}
        out, replaced = [], 0
        for i, (lo, hi) in enumerate(cs):
            mid = ((lo if lo != float("-inf") else 0.0)
                   + (hi if hi != float("inf") else ctx.audio_end)) / 2
            if not self.inside(mid, spans):
                out.extend(w_slices[i])
                continue
            got = self.fill_cell({p: per[p][i] for p in TRIO}, self.fill)
            replaced += len(w_slices[i])
            out.extend(got)
        return out, {"matched": True, "dose": len(cuts), "replaced": replaced,
                     "mask_spans": len(spans)}

    def apply(self, w: Window, params) -> list[str]:
        out, _ = self.build(context(w))
        return list(w.w_tokens) if out is None else out


class MaskSelect(OverlapArm):
    name = "ov_mask_select"
    cut_source = "none"
    fill = "select"


class TurnSelect(OverlapArm):
    name = "ov_turn_select"
    cut_source = "speaker"
    fill = "select"


class TurnSelectPlacebo(OverlapArm):
    name = "ov_turn_select_placebo"
    cut_source = "placebo"
    fill = "select"


class TurnCompose(OverlapArm):
    name = "ov_turn_compose"
    cut_source = "speaker"
    fill = "compose"


class TurnComposePlacebo(OverlapArm):
    name = "ov_turn_compose_placebo"
    cut_source = "placebo"
    fill = "compose"


ARMS = {c.name: c for c in (MaskSelect, TurnSelect, TurnSelectPlacebo,
                            TurnCompose, TurnComposePlacebo)}


# ------------------------------------------------------------- mechanistic estimand
def mask_rows(ctx: Context, out_tokens: list[str]) -> tuple[float, int]:
    """(edit distance inside the mask, reference tokens inside the mask).

    The scoring partition is the MASK ITSELF, identical for every arm - deliberately
    not each arm's own cells, because a finer partition constrains the alignment and
    would penalise whichever arm carries more cuts.
    """
    from eval.controlled_eval.scoring import edist
    spans = OverlapArm.mask(ctx.reg)
    if not spans:
        return 0.0, 0
    edges = sorted({t for sp in spans for t in sp})
    cs = cells(edges)
    keep = [i for i, (lo, hi) in enumerate(cs)
            if OverlapArm.inside(((lo if lo != float("-inf") else 0.0)
                                  + (hi if hi != float("inf") else ctx.audio_end)) / 2,
                                 spans)]
    rslice = split_tokens(ctx.ref, ctx.ref_times, cs)
    out_times, _ = token_times(out_tokens, *_anchor(ctx))
    oslice = split_tokens(out_tokens, out_times, cs)
    e = sum(edist(rslice[i], oslice[i]) for i in keep)
    n = sum(len(rslice[i]) for i in keep)
    return float(e), n


def _anchor(ctx: Context):
    toks, times, _, _, audio_end = _turbo_stream(ctx.item_id)
    return toks, times, audio_end
