#!/usr/bin/env python3
"""Give every MSA column an INTERVAL and say how it was obtained.

There is no such thing as "the time of a column". Two of the three systems carry
timestamps, from decodes that are not the decodes whose text was scored, and the
third (Scribe) carries none at all. So no column here ever gets a point timestamp.
Every column gets `[time_start, time_end)`, a `time_method`, an uncertainty, the
per-system source intervals and the disagreement between them.

    observed      at least one system supplied a stable interval for this column.
                  Display representative = median of the available midpoints; the
                  stored interval is the UNION, so it can only be too wide, never
                  too narrow. `time_conflict` is set when two systems' intervals do
                  not overlap within TOLERANCE, which nearly always means the MSA
                  put two different utterances of a repeated word in one column.
    bracketed     nobody supplied an interval: the column lies between the previous
                  anchor's end and the next anchor's start. Runs of such columns
                  divide that gap monotonically by rank, purely for layout.
    extrapolated  the column is outside the outermost anchor. Allowed to reach one
                  local median word duration beyond it, no further.
    unplaced      nothing above applies. The column goes to a gutter and claims
                  nothing about when it happened.

A bracketed column is additionally marked `unresolved` when its bracket is wider
than MAX_BRACKET, or crosses a speaker handover, an overlap region, or a long
silence. Unresolved means the page must not put a speaker on it either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

TOLERANCE = 0.30            # s. Two source intervals further apart than this conflict.
MAX_BRACKET = 2.0           # s. A wider bracket says nothing useful; mark unresolved.
MAX_EXTRAPOLATION = 1.0     # s. Hard cap on reaching past the outermost anchor.


@dataclass
class ColumnTime:
    time_start: float | None = None
    time_end: float | None = None
    time_method: str = "unplaced"
    time_uncertainty: float | None = None      # width of the interval, seconds
    time_conflict: bool = False
    conflict_gap: float | None = None          # seconds between disjoint sources
    unresolved: bool = False
    unresolved_reason: str = ""
    sources: dict = field(default_factory=dict)   # system -> {start,end,provenance}
    representative: float | None = None           # a point ONLY for seeking audio


def _from_sources(sources: dict) -> ColumnTime:
    """Fold one or more observed source intervals into a column interval."""
    items = [(k, v) for k, v in sources.items()
             if v and v.get("start") is not None]
    if not items:
        return ColumnTime(sources=sources)
    starts = [v["start"] for _, v in items]
    ends = [v["end"] for _, v in items]
    mids = [(v["start"] + v["end"]) / 2 for _, v in items]
    conflict, gap = False, None
    if len(items) > 1:
        (_, a), (_, b) = items[0], items[1]
        overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
        if overlap < -TOLERANCE:
            conflict, gap = True, round(-overlap, 3)
    lo, hi = min(starts), max(ends)
    return ColumnTime(time_start=lo, time_end=hi, time_method="observed",
                      time_uncertainty=round(hi - lo, 3),
                      time_conflict=conflict, conflict_gap=gap,
                      sources=sources, representative=median(mids))


def place(columns_sources: list[dict], exclusive: list[dict],
          bounds: tuple[float, float]) -> list[ColumnTime]:
    """Assign an interval and a method to every column, in order.

    `columns_sources[i]` maps a system name to `{start, end, provenance}` or to None.
    `bounds` is the absolute span of the page, used to cap extrapolation.
    """
    from eval.tsfusion.speakers import crosses_handover, crosses_silence

    n = len(columns_sources)
    out = [_from_sources(s) for s in columns_sources]
    anchors = [i for i, c in enumerate(out) if c.time_method == "observed"]

    # a local sense of how long a word lasts here, for the extrapolation cap only
    widths = [out[i].time_end - out[i].time_start for i in anchors]
    typical = median(widths) if widths else 0.3

    if not anchors:
        return out

    first, last = anchors[0], anchors[-1]
    # --- interior runs: bracket between the surrounding anchors, divided by rank
    for a, b in zip(anchors, anchors[1:]):
        run = list(range(a + 1, b))
        if not run:
            continue
        lo, hi = out[a].time_end, out[b].time_start
        if hi <= lo:                      # anchors overlap: nothing to divide
            lo = hi = (lo + hi) / 2
        step = (hi - lo) / len(run)
        wide = (hi - lo) > MAX_BRACKET
        for k, i in enumerate(run):
            s, e = lo + k * step, lo + (k + 1) * step
            c = out[i]
            c.time_start, c.time_end, c.time_method = lo, hi, "bracketed"
            c.time_uncertainty = round(hi - lo, 3)
            c.representative = (s + e) / 2
            reasons = []
            if wide:
                reasons.append(f"bracket {hi - lo:.2f}s > {MAX_BRACKET}s")
            if crosses_handover(lo, hi, exclusive):
                reasons.append("crosses a speaker handover")
            if crosses_silence(lo, hi, exclusive):
                reasons.append("crosses a silence")
            if sum(1 for s2 in exclusive
                   if s2["end"] > lo and s2["start"] < hi) > 1:
                reasons.append("more than one turn in the bracket")
            if reasons:
                c.unresolved = True
                c.unresolved_reason = "; ".join(reasons)

    # --- the two ends: limited extrapolation, then the unplaced gutter
    lead = list(range(0, first))
    if lead:
        edge = out[first].time_start
        reach = min(MAX_EXTRAPOLATION, typical * len(lead))
        lo = max(bounds[0], edge - reach)
        if len(lead) * typical > MAX_EXTRAPOLATION:
            for i in lead:
                out[i].time_method = "unplaced"
                out[i].unresolved = True
                out[i].unresolved_reason = (
                    f"{len(lead)} columns before the first anchor, beyond the "
                    f"{MAX_EXTRAPOLATION}s extrapolation cap")
        else:
            step = (edge - lo) / len(lead)
            for k, i in enumerate(lead):
                out[i].time_start, out[i].time_end = lo, edge
                out[i].time_method = "extrapolated"
                out[i].time_uncertainty = round(edge - lo, 3)
                out[i].representative = lo + (k + 0.5) * step
                out[i].unresolved = True
                out[i].unresolved_reason = "before the first observed anchor"

    tail = list(range(last + 1, n))
    if tail:
        edge = out[last].time_end
        reach = min(MAX_EXTRAPOLATION, typical * len(tail))
        hi = min(bounds[1], edge + reach)
        if len(tail) * typical > MAX_EXTRAPOLATION:
            for i in tail:
                out[i].time_method = "unplaced"
                out[i].unresolved = True
                out[i].unresolved_reason = (
                    f"{len(tail)} columns after the last anchor, beyond the "
                    f"{MAX_EXTRAPOLATION}s extrapolation cap")
        else:
            step = (hi - edge) / len(tail)
            for k, i in enumerate(tail):
                out[i].time_start, out[i].time_end = edge, hi
                out[i].time_method = "extrapolated"
                out[i].time_uncertainty = round(hi - edge, 3)
                out[i].representative = edge + (k + 0.5) * step
                out[i].unresolved = True
                out[i].unresolved_reason = "after the last observed anchor"
    return out
