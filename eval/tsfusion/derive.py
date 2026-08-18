#!/usr/bin/env python3
"""Facts derived at RENDER time from the bundle. The pipeline is not touched.

`build.py` writes `data.json` and stays exactly as it was. Everything here reads
that file and computes what the page needs on top of it, so a reader can rebuild
the page from an existing bundle without re-running any decode, any diarization
call or any alignment.

Three things live here that the page cannot do without.

ANCHORS. `data.json` carries `page_t`, the MIDPOINT of a column's interval. When a
column's two source intervals conflict, that interval is very wide and its midpoint
lands past the midpoint of the next column, so the page showed words going backwards
in time. A column is instead anchored on a single interval that some system really
observed, and the conflict is kept as a separate span. The order is:

  1. observed_source   the source interval with the EARLIEST start among the
                       column's `sources`. Its own end is the anchor end, so the
                       word's duration is one system's duration and not a union.
  2. bracketed         no source at all, but `timing.place` put the column inside a
                       bracket. The anchor point is the column's representative
                       (`page_t`), which within a bracketed run is monotone by
                       construction. There is no duration.
  3. extrapolated      the same, past the outermost anchor.
  4. none              nothing is claimed.

Only `observed_source` is `reliable`, and `anchors()` still FLAGS a regression
against the running maximum rather than hiding it.

POSITIONS. What the page displays is not the anchor. `positions()` takes the
columns that are `observed` AND unconflicted as fixed points and spreads every
run of columns between two of them evenly across the interval the two leave, so
the displayed time is non-decreasing over the whole page and a conflicted column
can no longer be shown before the column in front of it. Interpolated columns are
marked as such on the page; their time is an estimate and says nothing about
where the word really was.

OVERLAP. `speakers.assign` calls a column `overlap` whenever the regular
diarization has more than one turn touching it, which includes a turn that only
grazes the word by a few milliseconds. A stricter test is applied here and the
original state is preserved beside it.

REFERENCE OMISSION. An insertion in a column that at least two of the three
systems occupied is more likely to be missing from the published text than
invented by us. The test is occupancy, not agreement on the word: two decoders
hearing speech where the published text has none is the evidence, whether or not
they heard the same word. It is flagged, never subtracted.
"""
from __future__ import annotations

import itertools
import math

SYSTEMS = ("scribe", "soniox", "whisper")

# A second speaker must hold this much of the word before the column is overlap.
MIN_OVERLAP_SECONDS = 0.30
MIN_OVERLAP_FRACTION = 0.30

# A published token this short that no system produced anywhere on the page is an
# abbreviation or an initial, not something an ASR could have emitted.
ABBREVIATION_MAX_LEN = 2

# A named turn is cut into a new card when the audio goes quiet for this long.
TURN_GAP_SECONDS = 2.0

# An interval this wide cannot place a word inside its own duration, so the page
# says so. Below it, saying so would mark almost every word: the uncertainty on
# this page has median 0.36 s and p90 0.72 s.
WIDE_UNCERTAINTY_SECONDS = 1.0

# Two clocks for the same column that differ by more than this are disagreeing,
# not rounding: it is about two average word durations.
CLOCK_DISAGREEMENT_SECONDS = 0.30

EPS = 1e-6


def _num(x):
    """A finite float, or None. Guards against nulls and NaN in the bundle."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _span(start, end, t0):
    """A page-relative [start, end], or None when either end is missing."""
    a, b = _num(start), _num(end)
    return None if a is None or b is None else [a - t0, b - t0]


# ------------------------------------------------------------------- anchors
def anchor_of(row: dict, page_start_abs: float) -> dict:
    """One trustworthy position for a column. Never a conflicted midpoint."""
    best = None
    for name in SYSTEMS:
        src = (row.get("sources") or {}).get(name)
        if not src:
            continue
        s, e = _num(src.get("start")), _num(src.get("end"))
        if s is None:
            continue
        if e is None or e < s:
            e = None
        if best is None or s < best[1]:
            best = (name, s, e)
    if best is not None:
        name, s, e = best
        return {
            "kind": "observed_source", "source": name,
            "abs_start": s, "abs_end": e,
            "page_start": s - page_start_abs,
            "page_end": None if e is None else e - page_start_abs,
            "duration": None if e is None else round(e - s, 3),
            "reliable": e is not None and e > s,
        }
    page_t = _num(row.get("page_t"))
    method = row.get("time_method")
    if page_t is not None and method in ("bracketed", "extrapolated"):
        return {
            "kind": method, "source": None,
            "abs_start": page_t + page_start_abs, "abs_end": None,
            "page_start": page_t, "page_end": None,
            "duration": None, "reliable": False,
        }
    return {"kind": "none", "source": None, "abs_start": None, "abs_end": None,
            "page_start": None, "page_end": None, "duration": None,
            "reliable": False}


def anchors(rows: list, page_start_abs: float) -> list:
    """Anchors in row order, each flagged against the running maximum."""
    out, running = [], None
    for row in rows:
        a = dict(anchor_of(row, page_start_abs))
        p = a["page_start"]
        a["backwards"] = False
        a["regression"] = None
        if p is not None:
            if running is not None and p < running - EPS:
                a["backwards"] = True
                a["regression"] = round(running - p, 3)
            running = p if running is None else max(running, p)
        out.append(a)
    return out


# ----------------------------------------------------------------- positions
# A word is never shown for longer than this, however far the next one is.
MAX_WORD_SECONDS = 2.5
MIN_WORD_SECONDS = 0.05


def is_anchor_row(row: dict, anchor: dict) -> bool:
    """A column whose own time nobody disputes.

    `time_method == "observed"` means some system really timed this column, and
    `time_conflict == False` means the systems that timed it agree about where.
    Everything else (bracketed, extrapolated, conflicted) is positioned by its
    neighbours instead of by itself.
    """
    return bool(row.get("time_method") == "observed"
                and not row.get("time_conflict")
                and anchor.get("page_start") is not None)


def positions(rows: list, anchor_list: list, page_duration=None) -> list:
    """One displayed time per column, non-decreasing over the whole page.

    Anchors keep their own observed start, clamped up to the running maximum on
    the rare occasion two undisputed anchors are out of order. Every run of
    non-anchor columns between two anchors is spread evenly across the interval
    the two anchors leave, so a conflicted column can no longer appear before
    the column in front of it. Columns before the first anchor are spread from
    the start of the page, columns after the last from the last anchor to the
    end of the page.

    Returns, per column: `t` (displayed seconds into the page), `end` (when the
    karaoke highlight lets go), `anchor`, `interpolated`, `clamped`.
    """
    n = len(rows)
    if n == 0:
        return []
    flag = [is_anchor_row(r, a) for r, a in zip(rows, anchor_list)]
    idx = [k for k in range(n) if flag[k]]
    t: list = [None] * n
    clamped = [False] * n

    running = None
    for k in idx:
        v = anchor_list[k]["page_start"]
        if running is not None and v < running - EPS:
            v, clamped[k] = running, True
        running = v
        t[k] = v

    if not idx:
        span = _num(page_duration) or 0.0
        for k in range(n):
            t[k] = span * k / (n - 1) if n > 1 and span else 0.0
    else:
        first, last = idx[0], idx[-1]
        for k in range(first):                       # before the first anchor
            t[k] = t[first] * (k + 1) / (first + 1)
        for a_i, b_i in zip(idx, idx[1:]):           # between two anchors
            gap = b_i - a_i
            if gap < 2:
                continue
            lo, hi = t[a_i], t[b_i]
            for j in range(a_i + 1, b_i):
                t[j] = lo + (hi - lo) * (j - a_i) / gap
        tail = _num(page_duration)
        tail = t[last] if tail is None else max(tail, t[last])
        for j in range(last + 1, n):                 # after the last anchor
            t[j] = t[last] + (tail - t[last]) * (j - last) / (n - last)

    out = []
    for k in range(n):
        nxt = t[k + 1] if k + 1 < n else None
        dur = anchor_list[k].get("duration")
        if nxt is not None:
            end = min(nxt, t[k] + MAX_WORD_SECONDS)
        elif dur:
            end = t[k] + dur
        else:
            end = t[k] + MIN_WORD_SECONDS
        end = max(end, t[k] + MIN_WORD_SECONDS)
        out.append({"t": round(t[k], 3), "end": round(end, 3),
                    "anchor": flag[k], "interpolated": not flag[k],
                    "clamped": clamped[k]})
    return out


# ------------------------------------------------------------------- overlap
def _merge(spans: list) -> list:
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def overlap_call(row: dict, anchor: dict, regular: list) -> dict:
    """Does a second speaker really hold this word, or does it only graze it?

    The other speakers' turns are merged before measuring, so two diarization
    fragments of the same stretch are not counted twice.
    """
    if row.get("speaker_state") != "overlap":
        return {"evidence": "not_applicable", "seconds": None, "fraction": None,
                "speaker": None}
    lo, hi = anchor["abs_start"], anchor["abs_end"]
    if lo is None or hi is None or hi <= lo or not row.get("speaker"):
        return {"evidence": "unassessable", "seconds": None, "fraction": None,
                "speaker": None}
    width = hi - lo
    per: dict[str, list] = {}
    for t in regular:
        spk = t.get("speaker")
        if spk is None or spk == row.get("speaker"):
            continue
        a, b = _num(t.get("start")), _num(t.get("end"))
        if a is None or b is None:
            continue
        a, b = max(a, lo), min(b, hi)
        if b > a:
            per.setdefault(spk, []).append([a, b])
    if not per:
        return {"evidence": "not_confirmed", "seconds": 0.0, "fraction": 0.0,
                "speaker": None}
    spk, secs = max(((k, sum(b - a for a, b in _merge(v))) for k, v in per.items()),
                    key=lambda kv: kv[1])
    frac = secs / width
    ok = secs >= MIN_OVERLAP_SECONDS - EPS and frac >= MIN_OVERLAP_FRACTION - EPS
    return {"evidence": "confirmed" if ok else "not_confirmed",
            "seconds": round(secs, 3), "fraction": round(frac, 4), "speaker": spk}


# ------------------------------------------------------------------ warnings
def warnings_for(row: dict, overlap: dict, pos: dict) -> list:
    """Every reason this column is marked, with the numbers behind each one.

    Nothing may be marked without an entry here: the page renders one line of
    explanation per entry, on hover and in the popover. A mark the reader cannot
    interrogate is noise, and this page had 40 of them.
    """
    out = []
    if overlap.get("evidence") == "confirmed":
        out.append({"k": "overlap", "seconds": overlap.get("seconds"),
                    "fraction": overlap.get("fraction"),
                    "speaker": overlap.get("speaker")})
    if row.get("speaker_state") == "ambiguous":
        out.append({"k": "straddle", "speaker": row.get("speaker"),
                    "runner_up": row.get("speaker_runner_up"),
                    "coverage": _num(row.get("overlap_fraction"))})
    if row.get("unresolved"):
        out.append({"k": "unresolved",
                    "reason": row.get("unresolved_reason") or ""})
    u = _num(row.get("time_uncertainty"))
    if u is not None and u >= WIDE_UNCERTAINTY_SECONDS:
        out.append({"k": "wide", "seconds": round(u, 2)})
    if row.get("time_conflict"):
        out.append({"k": "conflict", "seconds": _num(row.get("conflict_gap"))})
    if pos.get("interpolated"):
        out.append({"k": "interpolated", "method": row.get("time_method")})
    return out


# --------------------------------------------------------- reference omission
def ref_omission_suspect(row: dict) -> bool:
    """An insertion two systems proposed on their own is a suspect omission.

    The test is OCCUPANCY: at least two of the three systems put a token in this
    column, independently of one another and of what the vote then chose. Two
    decoders hearing speech where the published text has none is evidence about
    the published text, and it stays evidence even when the two disagree about
    which word it was.

    Evidence, not a verdict: three systems can also be wrong together. It never
    changes a WER figure, it only lets the page count it separately.
    """
    ref = row.get("ref") or {}
    if ref.get("op") != "insert":
        return False
    if not row.get("w"):
        return False
    return sum(1 for s in SYSTEMS if row.get(s)) >= 2


# --------------------------------------------------------------- error census
def system_vocabulary(rows: list) -> set:
    """Every token any of the three systems produced anywhere on the page."""
    return {row[s] for row in rows for s in SYSTEMS if row.get(s)}


def is_publication_convention(ref_word, vocabulary: set) -> bool:
    """A published token no system produced, short enough to be an abbreviation.

    The tokens on this page are already normalised (accents stripped, lowercased,
    punctuation gone), so case and punctuation conventions are invisible here and
    this test can only see the surviving kind: «κ» for «κύριος», an initial before
    a surname. It is a suspicion drawn from the data, not a proof about what an
    ASR could emit.
    """
    if not ref_word:
        return False
    return len(ref_word) <= ABBREVIATION_MAX_LEN and ref_word not in vocabulary


def error_census(data: dict, rows: list) -> dict:
    """Every charged W edit, in exactly one of three buckets.

    Precedence: convention, then reference omission, then ours. The three counts
    reconcile to the edit distance the independent alignment reports.
    """
    vocab = system_vocabulary(rows)
    buckets = {"convention": [], "ref_omission": [], "ours": []}

    for row in rows:
        ref = row.get("ref") or {}
        op = ref.get("op")
        if op == "sub":
            item = {"kind": "sub", "i": row["i"], "w": row.get("w"),
                    "ref": ref.get("word"), "reason": row.get("w_reason")}
            if is_publication_convention(ref.get("word"), vocab):
                buckets["convention"].append(item)
            else:
                buckets["ours"].append(item)
        elif op == "insert":
            item = {"kind": "insert", "i": row["i"], "w": row.get("w"),
                    "ref": None, "reason": row.get("w_reason")}
            if ref_omission_suspect(row):
                buckets["ref_omission"].append(item)
            else:
                buckets["ours"].append(item)

    for d in data.get("deletions", []):
        if d.get("system") != "W":
            continue
        item = {"kind": "delete", "i": d.get("after_column"), "w": None,
                "ref": d.get("word"), "reason": d.get("note") or d.get("method")}
        if is_publication_convention(d.get("word"), vocab):
            buckets["convention"].append(item)
        else:
            buckets["ours"].append(item)

    n = {k: len(v) for k, v in buckets.items()}
    n["total"] = sum(n.values())
    return {"counts": n, "items": buckets}


# ---------------------------------------------------------------- structures
def drift_zones(view_rows: list) -> list:
    """Maximal runs of consecutive columns whose timestamps conflict."""
    zones, run = [], []
    for r in view_rows:
        if r["time_conflict"]:
            run.append(r)
        elif run:
            zones.append(run)
            run = []
    if run:
        zones.append(run)
    out = []
    for k, run in enumerate(zones):
        gaps = [r["conflict_gap"] for r in run if r["conflict_gap"] is not None]
        pairs = set()
        for r in run:
            pairs.add(tuple(sorted(r["source_names"])))
        starts = [r["pos"]["t"] for r in run if r["pos"]["t"] is not None]
        out.append({
            "id": k,
            "first": run[0]["i"], "last": run[-1]["i"], "n": len(run),
            "page_start": min(starts) if starts else None,
            "page_end": max(starts) if starts else None,
            "max_gap": max(gaps) if gaps else None,
            "systems": sorted({s for p in pairs for s in p}),
            "words": [r["w"] for r in run if r["w"]],
        })
    return out


def speaker_runs(view_rows: list) -> list:
    """The speaker each column is READ under, carried across gaps in the calls.

    A column the diarization could not resolve, or one that straddles a handover,
    does not start a new card. Cutting on those states shattered one handover
    into eleven cards of one word each. The state is kept, and shown on the word
    itself; the card stays with the speaker who is talking.
    """
    out, current = [], None
    for r in view_rows:
        spk = r.get("speaker")
        if spk:
            current = spk
        out.append(current)
    # a run with no call at all yet inherits the first speaker that appears
    first = next((s for s in out if s), None)
    return [s or first for s in out]


def turns(view_rows: list) -> list:
    """Consecutive columns of one speaker, cut at windows and at long silences."""
    out = []
    runs = speaker_runs(view_rows)
    for r, spk in zip(view_rows, runs):
        r["card_speaker"] = spk
    for (window, key), group in itertools.groupby(
            view_rows, key=lambda r: (r["window"], ("named", r["card_speaker"]))):
        block = list(group)
        cut = [[block[0]]]
        for prev, cur in zip(block, block[1:]):
            a = prev["pos"]["t"]
            b = cur["pos"]["t"]
            if a is not None and b is not None and b - a > TURN_GAP_SECONDS:
                cut.append([cur])
            else:
                cut[-1].append(cur)
        for piece in cut:
            starts = [r["pos"]["t"] for r in piece if r["pos"]["t"] is not None]
            ends = [r["pos"]["end"] for r in piece
                    if r["pos"]["end"] is not None]
            out.append({
                "id": len(out), "window": window,
                "state": key[0], "speaker": key[1],
                "first": piece[0]["i"], "last": piece[-1]["i"],
                "page_start": min(starts) if starts else None,
                "page_end": max(ends) if ends else (max(starts) if starts else None),
                "rows": [r["i"] for r in piece],
            })
    return out


# --------------------------------------------------------- per system census
def system_error_rows(row: dict) -> dict:
    """Which systems are wrong in this column, and which of them is right.

    Read off `sys_op`, the per-system alignment against the published text that
    `build.py` projected onto the MSA columns. Each system was aligned on its
    own, so this is not one alignment reinterpreted four times.
    """
    ref = row.get("ref") or {}
    ref_word = ref.get("word")
    wrong, right = [], []
    for s in SYSTEMS:
        op = ((row.get("sys_op") or {}).get(s) or {}).get("op")
        if op in ("sub", "insert"):
            wrong.append(s)
        if ref_word and row.get(s) == ref_word:
            right.append(s)
    w_wrong = ref.get("op") in ("sub", "insert")
    return {
        "wrong": wrong, "right": right, "w_wrong": bool(w_wrong),
        # The class the user cares about: the vote lost a word one of its own
        # voters had. Only decidable where the published text names a word.
        "selection_loss": bool(w_wrong and ref.get("op") == "sub" and right),
    }


def per_system_table(data: dict, rows: list) -> list:
    """S, D, I, (S+D)/N and WER for each system and for the vote.

    `(S+D)/N` is the rate that cannot be lowered by writing less; it is the
    figure to read next to the deletion count. The insertion-adjusted WER on the
    right drops insertions whose column two systems occupied, which is a
    property of the published text, not of the system: it is reported beside the
    primary WER and never in place of it.
    """
    names = {"scribe": "Scribe v2", "soniox": "Soniox",
             "whisper": "Το μοντέλο μας", "W": "W (σύνθεση)"}
    ps = data["per_system"]
    ref_tokens = (ps["W"]["counts"]["equal"] + ps["W"]["counts"]["sub"]
                  + ps["W"]["counts"]["delete"])
    occupied = {row["i"] for row in rows
                if sum(1 for s in SYSTEMS if row.get(s)) >= 2}
    out = []
    for key in ("scribe", "soniox", "whisper", "W"):
        c = ps[key]["counts"]
        if key == "W":
            susp = sum(1 for row in rows if ref_omission_suspect(row))
        else:
            susp = sum(1 for row in rows
                       if ((row.get("sys_op") or {}).get(key) or {})
                       .get("op") == "insert" and row["i"] in occupied)
        susp = min(susp, c["insert"])
        dist = c["sub"] + c["delete"] + c["insert"]
        out.append({
            "key": key, "name": names[key],
            "S": c["sub"], "D": c["delete"], "I": c["insert"],
            "ambiguous": c["ambiguous"], "n_hyp": ps[key]["n_hyp"],
            "ref_tokens": ref_tokens,
            "sd_rate": (c["sub"] + c["delete"]) / ref_tokens if ref_tokens else None,
            "wer": dist / ref_tokens if ref_tokens else None,
            "suspect_insertions": susp,
            "wer_excl_suspect": ((dist - susp) / ref_tokens
                                 if ref_tokens else None),
        })
    return out


# -------------------------------------------------------------- error ledger
LEDGER_SYSTEMS = ("scribe", "soniox", "whisper", "W")
CONTEXT_WORDS = 4


def _window_blocks(rows: list) -> list:
    """Rows grouped into the benchmark windows, in page order."""
    out = []
    for r in rows:
        if not out or out[-1][0] != r["window"]:
            out.append((r["window"], []))
        out[-1][1].append(r)
    return out


def reference_by_window(data: dict) -> dict:
    """The published text, rebuilt from W's own alignment plus W's deletions.

    Every reference token is either matched by a column (`ref.ref_index`) or is a
    W deletion carrying its own `ref_index`, so the two sources together are the
    whole text. `ref_index` restarts per window, which is why this is per window
    and not one list.
    """
    rows = data["rows"]
    win_of = {r["i"]: r["window"] for r in rows}
    present: dict = {name: {} for name, _ in _window_blocks(rows)}
    for r in rows:
        ref = r.get("ref") or {}
        if ref.get("ref_index") is not None:
            present[r["window"]][ref["ref_index"]] = ref.get("word") or ""
    for d in data.get("deletions", []):
        if d.get("system") != "W" or d.get("ref_index") is None:
            continue
        ac = d.get("after_column")
        ac = -1 if ac is None else ac
        w = win_of.get(ac)
        nxt = win_of.get(ac + 1)
        # `after_column` is the column the deletion FOLLOWS, and build.py writes
        # `base - 1` when it precedes a window's first column, so the neighbour
        # on the right decides whenever the left one already holds this index.
        if w is None or d["ref_index"] in present.get(w, {}):
            if nxt is not None and d["ref_index"] not in present.get(nxt, {}):
                w = nxt
        if w is None:
            continue
        present[w][d["ref_index"]] = d.get("word") or ""
    out = {}
    for name, _ in _window_blocks(rows):
        idx = present[name]
        out[name] = [idx.get(k, "") for k in range(max(idx) + 1 if idx else 0)]
    return out


def ledger(data: dict, view_rows: list) -> list:
    """Every charged edit of every system, as one readable row.

    Each system is aligned to the published text on its own (the same frozen
    `refalign` the rest of the page uses), so the counts here reconcile with the
    per-system table. Deletions are taken from the bundle's own deletion events,
    which is where their borrowed placement and its caveat already live.
    """
    from eval.tsfusion import refalign as RA

    rows = data["rows"]
    by_i = {r["i"]: r for r in view_rows}
    ref_by_window = reference_by_window(data)

    ref_all: list = []
    offset: dict = {}
    for name, _ in _window_blocks(rows):
        offset[name] = len(ref_all)
        ref_all += ref_by_window.get(name, [])

    def context(g: int) -> tuple:
        lo = max(0, g - CONTEXT_WORDS)
        return (" ".join(ref_all[lo:g]),
                " ".join(ref_all[g + 1:g + 1 + CONTEXT_WORDS]))

    # global reference index of every deletion event, and the events by system
    dels: dict = {}
    for d in data.get("deletions", []):
        if d.get("ref_index") is None:
            continue
        ac = -1 if d.get("after_column") is None else d["after_column"]
        row = by_i.get(ac) or by_i.get(ac + 1)
        win = row["window"] if row else next(iter(offset))
        g = offset.get(win, 0) + d["ref_index"]
        dels.setdefault(d.get("system"), {})[g] = d

    op_at: dict = {}
    out: list = []
    for sysname in LEDGER_SYSTEMS:
        cols, hyp = [], []
        for r in rows:
            tok = r.get("w") if sysname == "W" else r.get(sysname)
            if tok:
                cols.append(r["i"])
                hyp.append(tok)
        al = RA.align_to_reference(sysname, ref_all, hyp)
        op_at[sysname] = {o.ref_index: o.op for o in al.ops
                          if o.ref_index is not None}
        for o in al.ops:
            if o.op == "equal":
                continue
            col = cols[o.hyp_index] if o.hyp_index is not None else None
            row = by_i.get(col) if col is not None else None
            ev = dels.get(sysname, {}).get(o.ref_index) if o.op == "delete" \
                else None
            if o.op == "delete":
                # placed by the fixed hierarchy in `refalign.place_deletions`,
                # never by a clock of its own
                t = (ev or {}).get("page_t")
                near = None
                if col is None and ev is not None:
                    ac = -1 if ev.get("after_column") is None \
                        else ev["after_column"]
                    near = by_i.get(ac) or by_i.get(ac + 1)
                if t is None and near is not None:
                    t = near["pos"]["t"]
                borrowed = True
                method = (ev or {}).get("method") or "unplaced"
                anchor_i = near["i"] if near else None
            else:
                t = row["pos"]["t"] if row else None
                borrowed = bool(row and row["pos"]["interpolated"])
                method = (row or {}).get("time_method") or ""
                anchor_i = col
            g = o.ref_index
            left, right = context(g) if g is not None else ("", "")
            if g is None and o.hyp_index is not None:
                # an insertion sits between two reference words; show the text
                # around the reference position it interrupts
                before = sum(1 for x in al.ops
                             if x.ref_index is not None
                             and x.hyp_index is not None
                             and x.hyp_index < o.hyp_index)
                left = " ".join(ref_all[max(0, before - CONTEXT_WORDS):before])
                right = " ".join(ref_all[before:before + CONTEXT_WORDS])
            out.append({
                "system": sysname,
                "type": {"sub": "S", "delete": "D", "insert": "I"}[o.op],
                "ref_index": g, "ref_word": o.ref_word,
                "hyp_word": o.hyp_word,
                "column": col, "anchor_column": anchor_i,
                "t": None if t is None else round(t, 2),
                "time_borrowed": borrowed, "time_method": method,
                "time_note": (ev or {}).get("note") or "",
                "speaker": (row or near or {}).get("card_speaker")
                if (row or (o.op == "delete" and near)) else None,
                "left": left, "right": right,
                "tokens": ({s: row["systems"].get(s) for s in SYSTEMS}
                           if row else {}),
                "w": row["w"] if row else None,
                "reason": row["w_reason"] if row else None,
                "ambiguous": bool(o.ambiguous),
                "right_systems": [],
            })

    for e in out:
        g = e["ref_index"]
        if g is None:
            continue
        e["right_systems"] = [s for s in LEDGER_SYSTEMS
                              if op_at.get(s, {}).get(g) == "equal"]
    out.sort(key=lambda e: (e["t"] if e["t"] is not None else 1e9,
                            e["ref_index"] if e["ref_index"] is not None else 0))
    return out


# ------------------------------------------------------- clocks vs diarization
def _median(xs: list):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else (xs[k - 1] + xs[k]) / 2.0


def _nearest(x: float, points: list) -> float:
    return min((abs(x - p) for p in points), default=None)


def clock_report(data: dict, view_rows: list) -> dict:
    """Which timed system deserves to be the page's clock, measured not asserted.

    Both candidates are re-decodes: Soniox's times come from `stt-rt-v4`, which
    is NOT the `stt-async-v5` text W is composed from, and our own times come
    from a `word_timestamps=True` decode that is not the benchmark decode. The
    question here is only which clock the diarization joins onto best.
    """
    ex = sorted(data.get("diar", {}).get("exclusive", []),
                key=lambda t: t["page_start"])
    edges = sorted({round(t[k], 3) for t in ex
                    for k in ("page_start", "page_end")})
    changes = []
    for a, b in zip(ex, ex[1:]):
        if a.get("speaker") != b.get("speaker"):
            changes.append((a["page_end"] + b["page_start"]) / 2.0)

    per = {}
    for sysname in ("soniox", "whisper"):
        words = [(v["start"], v["end"]) for r in view_rows
                 for k, v in [(sysname, (r["source_detail"] or {}).get(sysname))]
                 if v and v.get("start") is not None and v.get("end") is not None
                 and v["end"] > v["start"]]
        straddle = sum(1 for a, b in words
                       if any(a + EPS < e < b - EPS for e in edges))
        inside = sum(1 for c in changes
                     if any(a + EPS < c < b - EPS for a, b in words))
        per[sysname] = {
            "n_words": len(words),
            "coverage": len(words) / len(view_rows) if view_rows else None,
            "median_duration": _median([b - a for a, b in words]),
            "straddling": straddle,
            "straddling_rate": straddle / len(words) if words else None,
            "changes_inside_a_word": inside,
            "median_word_edge_to_turn_edge": _median(
                [_nearest(x, edges) for a, b in words for x in (a, b)]),
            "median_turn_edge_to_word_edge": _median(
                [_nearest(e, [x for a, b in words for x in (a, b)])
                 for e in edges]),
        }

    both, diffs = 0, []
    for r in view_rows:
        sd = r["source_detail"] or {}
        a, b = sd.get("soniox"), sd.get("whisper")
        if not a or not b or a.get("start") is None or b.get("start") is None:
            continue
        both += 1
        diffs.append(abs(a["start"] - b["start"]))
    disagree = sum(1 for x in diffs if x > CLOCK_DISAGREEMENT_SECONDS)

    def better(key, lower=True):
        x, y = per["soniox"][key], per["whisper"][key]
        if x is None or y is None or x == y:
            return None
        return ("soniox" if (x < y) == lower else "whisper")

    votes = [better("straddling_rate"), better("median_turn_edge_to_word_edge"),
             better("coverage", lower=False)]
    counted = [v for v in votes if v]
    favours = None
    if counted:
        top = max(set(counted), key=counted.count)
        favours = top if counted.count(top) > len(counted) / 2 else "mixed"
    return {
        "systems": per,
        "n_turns": len(ex), "n_turn_edges": len(edges),
        "n_speaker_changes": len(changes),
        "n_columns": len(view_rows),
        "both_timed": both,
        "disagree_over_threshold": disagree,
        "disagree_rate": disagree / both if both else None,
        "median_start_gap": _median(diffs),
        "threshold": CLOCK_DISAGREEMENT_SECONDS,
        "favours": favours,
        "votes": votes,
    }


# ------------------------------------------------------------------ assembly
def build_view(data: dict) -> dict:
    """The whole view model. Pure: `data` is read and never mutated."""
    rows = data["rows"]
    t0 = data["manifest"]["page_start_abs"]
    regular = data.get("diar", {}).get("regular", [])
    anch = anchors(rows, t0)
    pos = positions(rows, anch, data["manifest"].get("page_duration"))

    view_rows, deletions_at = [], {}
    for d in data.get("deletions", []):
        deletions_at.setdefault(d.get("after_column"), []).append(d)

    for row, a, p in zip(rows, anch, pos):
        ov = overlap_call(row, a, regular)
        state = row.get("speaker_state")
        display = state
        if state == "overlap" and ov["evidence"] != "confirmed":
            display = "named" if row.get("speaker") else "unresolved"
        ref = row.get("ref") or {}
        warn = warnings_for(row, ov, p)
        suspect = ref_omission_suspect(row)
        errs = system_error_rows(row)
        proposals = {s: row.get(s) for s in SYSTEMS if row.get(s)}
        marks = []
        if not row.get("agree"):
            marks.append("disagree")
        if row.get("w") and ref.get("op") in ("sub", "insert"):
            marks.append("suspect_ref" if suspect else "vs_ref")
        if row.get("time_conflict"):
            marks.append("conflict")
        if not row.get("w") and proposals:
            marks.append("dropped")
        for wr in warn:
            if wr["k"] not in marks:
                marks.append(wr["k"])
        view_rows.append({
            "i": row["i"], "window": row["window"], "w": row.get("w"),
            "w_reason": row.get("w_reason"),
            "systems": {s: row.get(s) for s in SYSTEMS},
            "ref_word": ref.get("word"), "ref_op": ref.get("op"),
            "ref_ambiguous": bool(ref.get("ambiguous")),
            "agree": bool(row.get("agree")),
            "anchor": a, "pos": p,
            "sys_wrong": errs["wrong"], "sys_right": errs["right"],
            "w_wrong": errs["w_wrong"], "selection_loss": errs["selection_loss"],
            "time_method": row.get("time_method"),
            "time_uncertainty": _num(row.get("time_uncertainty")),
            "time_conflict": bool(row.get("time_conflict")),
            "conflict_gap": _num(row.get("conflict_gap")),
            "conflict_span": _span(row.get("time_start"), row.get("time_end"),
                                   t0) if row.get("time_conflict") else None,
            "unresolved": bool(row.get("unresolved")),
            "unresolved_reason": row.get("unresolved_reason") or "",
            "speaker": row.get("speaker"),
            # `overlap_fraction` in the bundle is the share of the interval the
            # ASSIGNED speaker covers, not any measure of overlap. Renamed here
            # so nothing downstream can read it as one.
            "speaker_coverage": _num(row.get("overlap_fraction")),
            "speaker_state": state,
            "display_speaker_state": display,
            "overlap": ov,
            "warnings": warn,
            "ref_omission_suspect": suspect,
            "marks": marks,
            "source_names": sorted((row.get("sources") or {}).keys()),
            "source_detail": {
                k: {"start": _num(v.get("start")) - t0
                    if _num(v.get("start")) is not None else None,
                    "end": _num(v.get("end")) - t0
                    if _num(v.get("end")) is not None else None,
                    "conf": _num(v.get("conf")),
                    "provenance": v.get("provenance"),
                    "match": v.get("match")}
                for k, v in (row.get("sources") or {}).items() if v},
            "source_intervals": {
                k: [v["start"] - t0, v["end"] - t0]
                for k, v in (row.get("sources") or {}).items()
                if v and _num(v.get("start")) is not None
                and _num(v.get("end")) is not None},
            "in_seam": bool(row.get("in_seam")),
        })

    view_by_i = {r["i"]: r for r in view_rows}
    zones = drift_zones(view_rows)
    for z in zones:
        for i in range(z["first"], z["last"] + 1):
            if i in view_by_i:
                view_by_i[i]["zone"] = z["id"]
    for r in view_rows:
        r.setdefault("zone", None)

    ps = data["per_system"]["W"]
    table = per_system_table(data, rows)
    ref_tokens = ps["counts"]["equal"] + ps["counts"]["sub"] + ps["counts"]["delete"]
    census = error_census(data, rows)
    n_overlap_before = sum(1 for r in rows if r.get("speaker_state") == "overlap")
    n_overlap_after = sum(1 for r in view_rows
                          if r["overlap"]["evidence"] == "confirmed")
    n_fraction_only = sum(
        1 for r in view_rows
        if r["overlap"]["fraction"] is not None
        and r["overlap"]["fraction"] >= MIN_OVERLAP_FRACTION - EPS)

    turn_list = turns(view_rows)
    ledger_rows = ledger(data, view_rows)
    clocks = clock_report(data, view_rows)
    return {
        "rows": view_rows,
        "turns": turn_list,
        "ledger": ledger_rows,
        "clocks": clocks,
        "zones": zones,
        "deletions_at": deletions_at,
        "per_system": table,
        "summary": {
            "ref_tokens": ref_tokens,
            "interpolated": sum(1 for p in pos if p["interpolated"]),
            "clamped": sum(1 for p in pos if p["clamped"]),
            "selection_loss": sum(1 for r in view_rows if r["selection_loss"]),
            "w_tokens": ps["n_hyp"],
            "correct": ps["counts"]["equal"],
            "wer": ps["wer"],
            "distance": ps["distance"],
            "insertions": ps["counts"]["insert"],
            "insertions_suspect": sum(1 for r in view_rows
                                      if r["ref_omission_suspect"]),
            "census": census,
            "wer_without_suspects": (
                (ps["distance"] - sum(1 for r in view_rows
                                      if r["ref_omission_suspect"])) / ref_tokens
                if ref_tokens else None),
            "overlap_before": n_overlap_before,
            "overlap_after": n_overlap_after,
            "overlap_fraction_only": n_fraction_only,
            "partial_coverage": sum(1 for r in view_rows
                                    if r["speaker_coverage"] is not None
                                    and r["speaker_coverage"] < 1.0),
            "backwards": sum(1 for a in anch if a["backwards"]),
            "unreliable": sum(1 for a in anch if not a["reliable"]),
            "n_zones": len(zones),
            "n_turns": len(turn_list),
            "marked": {k: sum(1 for r in view_rows
                              if any(w["k"] == k for w in r["warnings"]))
                       for k in ("overlap", "straddle", "unresolved", "wide",
                                 "conflict", "interpolated")},
        },
    }
