#!/usr/bin/env python3
"""Align each system to the published reference INDEPENDENTLY, and keep the ties.

Aligning only the fused output W to the reference tells you how W did. It does not
tell you how each MODEL diverges, which is the thing this page is for. So Scribe,
Soniox, our Whisper and W are each aligned to the reference on their own, and the
four edit mappings are projected onto the display separately.

Repeated words admit several equally optimal alignments. Presenting one arbitrary
backtrace as fact is the same error `msa.oracle_select`'s docstring warns about, so
every operation here is checked against the FULL optimal lattice and marked
`ambiguous` when more than one optimal path disagrees about it.

A DELETION IS NOT A COLUMN. It is an edit event that exists only once an output is
compared to the reference: a reference word with no counterpart. It therefore lives
BETWEEN columns, spanning an interval, and is modelled separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EditOp:
    op: str                     # equal | sub | delete | insert
    ref_index: int | None
    hyp_index: int | None
    ref_word: str | None = None
    hyp_word: str | None = None
    ambiguous: bool = False     # another equally optimal alignment disagrees


@dataclass
class RefAlignment:
    system: str
    ops: list[EditOp] = field(default_factory=list)
    n_ref: int = 0
    n_hyp: int = 0
    distance: int = 0
    counts: dict = field(default_factory=dict)

    def deletions(self) -> list[EditOp]:
        return [o for o in self.ops if o.op == "delete"]


def _dp(ref: list[str], hyp: list[str]):
    n, m = len(ref), len(hyp)
    f = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        f[i][0] = i
    for j in range(m + 1):
        f[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            f[i][j] = min(f[i - 1][j - 1] + (ri != hyp[j - 1]),
                          f[i - 1][j] + 1, f[i][j - 1] + 1)
    return f


def align_to_reference(system: str, ref: list[str], hyp: list[str]) -> RefAlignment:
    """One frozen backtrace, plus an ambiguity flag from the full optimal lattice.

    The backtrace prefers equal/substitution, then deletion, then insertion. That
    order is arbitrary and frozen; where it mattered, `ambiguous` says so.
    """
    n, m = len(ref), len(hyp)
    fwd = _dp(ref, hyp)
    bwd = _dp(ref[::-1], hyp[::-1])
    total = fwd[n][m]

    def suf(i, j):
        return bwd[n - i][m - j]

    def optimal_ops(i, j):
        """Which incoming edges at (i, j) lie on some optimal path."""
        got = set()
        if i > 0 and j > 0:
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            if fwd[i - 1][j - 1] + c + suf(i, j) == total:
                got.add("equal" if c == 0 else "sub")
        if i > 0 and fwd[i - 1][j] + 1 + suf(i, j) == total:
            got.add("delete")
        if j > 0 and fwd[i][j - 1] + 1 + suf(i, j) == total:
            got.add("insert")
        return got

    ops: list[EditOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        choices = optimal_ops(i, j)
        amb = len(choices) > 1
        if "equal" in choices:
            pick = "equal"
        elif "sub" in choices:
            pick = "sub"
        elif "delete" in choices:
            pick = "delete"
        else:
            pick = "insert"
        if pick in ("equal", "sub"):
            ops.append(EditOp(pick, i - 1, j - 1, ref[i - 1], hyp[j - 1], amb))
            i, j = i - 1, j - 1
        elif pick == "delete":
            ops.append(EditOp("delete", i - 1, None, ref[i - 1], None, amb))
            i -= 1
        else:
            ops.append(EditOp("insert", None, j - 1, None, hyp[j - 1], amb))
            j -= 1
    ops.reverse()
    counts = {k: sum(1 for o in ops if o.op == k)
              for k in ("equal", "sub", "delete", "insert")}
    counts["ambiguous"] = sum(1 for o in ops if o.ambiguous)
    return RefAlignment(system=system, ops=ops, n_ref=n, n_hyp=m,
                        distance=total, counts=counts)


# ------------------------------------------------------------- deletion placement
@dataclass
class DeletionEvent:
    system: str
    ref_index: int
    word: str
    after_column: int | None        # display gutter: between this column and the next
    time_start: float | None = None
    time_end: float | None = None
    method: str = "unplaced"        # proposal | bracketed | open_left | open_right
    open_ended: bool = False
    label: str = "asr_deletion"     # or reference_only_no_speech_detected
    note: str = ""
    ambiguous: bool = False


def place_deletions(system: str, alignment: RefAlignment,
                    ref_anchor_time: dict, proposals: dict,
                    exclusive: list[dict], bounds: tuple) -> list[DeletionEvent]:
    """Give each deleted reference word an interval, by a fixed hierarchy.

    1. another system emitted this word: use that proposal's observed interval;
    2. else bracket between the nearest reference words that both matched somewhere
       AND carry a time;
    3. else, with only one anchor, an open-ended interval on that side.

    `ref_anchor_time[ref_index] = (start, end)` for reference words that some system
    matched and timed. `proposals[ref_index] = {system: (start, end)}` for reference
    words another system did emit.

    If no system emitted the word and its interval contains no detected speech, the
    event is relabelled `reference_only_no_speech_detected`. That is deliberately not
    "not audible": nobody listened. It records what the diarizer found, nothing more.
    """
    out = []
    anchors = sorted(ref_anchor_time)
    for op in alignment.deletions():
        ri = op.ref_index
        ev = DeletionEvent(system=system, ref_index=ri, word=op.ref_word,
                           after_column=None, ambiguous=op.ambiguous)
        prop = proposals.get(ri) or {}
        if prop:
            src, (s, e) = sorted(prop.items())[0]
            ev.time_start, ev.time_end, ev.method = s, e, "proposal"
            ev.note = f"emitted by {src}"
        else:
            left = [a for a in anchors if a < ri]
            right = [a for a in anchors if a > ri]
            if left and right:
                ev.time_start = ref_anchor_time[left[-1]][1]
                ev.time_end = ref_anchor_time[right[0]][0]
                ev.method = "bracketed"
            elif left:
                ev.time_start = ref_anchor_time[left[-1]][1]
                ev.time_end = bounds[1]
                ev.method, ev.open_ended = "open_right", True
            elif right:
                ev.time_start = bounds[0]
                ev.time_end = ref_anchor_time[right[0]][0]
                ev.method, ev.open_ended = "open_left", True
        if ev.time_start is not None and ev.time_end is not None:
            if ev.time_end < ev.time_start:
                ev.time_start, ev.time_end = ev.time_end, ev.time_start
            speechy = any(s["end"] > ev.time_start and s["start"] < ev.time_end
                          for s in exclusive)
            if not prop and not speechy:
                ev.label = "reference_only_no_speech_detected"
                ev.note = ("no system emitted it and the diarizer reports no speech "
                           "in the candidate interval")
        out.append(ev)
    return out
