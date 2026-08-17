#!/usr/bin/env python3
"""How much of what we score as an INSERTION was really said?
(`exp-2026-08-17-insertion-fidelity`)

THE QUESTION. Every WER this project ships is agreement-with-OpenCouncil: the
reference is our own published transcript. The frozen gold set of
`exp-2026-08-16-gold-set` showed that reference omits real speech. This script
asks the mirror question, on the same frozen audio: of the words a system emits
that the published reference does NOT have -- the words the metric charges as
insertions -- how many are supported by the human who listened, and how many of
those are genuinely absent from the published text rather than an artefact of
where the alignment happened to put them?

WHAT THIS COVERS AND WHAT IT DOES NOT. W, the three-way fusion, CANNOT be run on
this audio: it needs ElevenLabs Scribe v2 and there is no credential here.
Covered: ADP (`artifact-adapter-fixed`, local CPU, per-word timestamps) and SNX
(Soniox `stt-rt-v4`, the free realtime path). NOT covered: W, Scribe v2, the paid
`stt-async-v5`. PUB, the published text, is the reference of alignment A and has
no insertions against itself; its own gold-unsupported tokens are reported apart
and never merged into the same number.

THE THREE ALIGNMENTS, AND WHY THREE
  A  ref = PUB view, hyp = system tokens. Its `I` labels are the population:
     exactly the tokens the shipped metric charges as insertions.
  C  ref = gold certain tokens, hyp = PUB view. Says which gold occurrences the
     published text represents at all.
  B  ref = gold certain tokens, hyp = system tokens. A conservative, order
     sensitive support definition, kept as SENSITIVITY only.
Support for the PRIMARY comes from an occurrence-level, temporally local,
INJECTIVE matching between system tokens and individual certain gold
occurrences, whose objective is blind to alignment A -- so a token's support
cannot depend on whether the metric happened to call it an insertion.

CLASSES, frozen before any outcome was computed (the evidence is the Codex design
review, job 847c449f, which ran before this file produced a number)
  GOLD_SUPPORTED      the token is injectively matched to a certain gold
                      occurrence in a temporally admissible block. Split in two:
      pub_unmatched_forced    PUB matches that gold occurrence in NO minimum-cost
                              alignment C. Reference-omission-CONSISTENT: the
                              annotator's certain transcript has the word, the
                              published text has no way to be reading it.
      pub_unmatched_possible  PUB fails to match it in at least one minimum-cost
                              alignment C. The looser end of the band.
      (matched)               PUB matches that occurrence in every minimum-cost
                              alignment; the insertion is a duplication or an
                              alignment artefact, NOT reference omission.
    The label is deliberately mechanical. "Attributable" would sound causal and a
    global alignment can attach a repeated word to the wrong occurrence, which is
    why the verdict is a forced/possible band and not a point.
  UNDECIDABLE         no such match AND the token's plausible time window is not
                      wholly inside exhaustively transcribed, text-certain gold
                      coverage. Proximity to a block is not evidence that the
                      surrounding time was annotated: the human transcribed
                      blocks, not the gaps between them.
  NOT_SUPPORTED       no match, and the whole plausible window lies inside
                      certain coverage. Named NOT_SUPPORTED, never "not said":
                      one anchored listener cannot establish non-occurrence.

FROZEN CHOICES, none selected after the fact
  region      core_envelope PRIMARY; core_strict and clip always reported beside
              it. They are separate estimands and are never summed.
  tau         0.5 s, dilating the system word interval. Declared as an
              operational rule, NOT a validated timestamp-error bound; the same
              tau is used for both systems, which gives a common decision rule
              and not equal validity. Grid {0.25, 0.5, 1.0} reported.
  tau_w       1.5 s for cross-system corroboration, grid {0.75, 1.5, 3.0}.
  PUB view    the frozen midpoint rule of `eval/gold_set_score.py` is PRIMARY
              because the target is literally the shipped metric; `any_overlap`
              and `wholly_contained` are reported as boundary sensitivities.
  alignment   the frozen S>D>I tie-break is primary; all 6 op priorities x
              forward/reversed are a sensitivity envelope over alignments A and
              B. Alignment C is computed over ALL minimum-cost paths and is
              tie-break-invariant by construction -- the envelope reaches it only
              through which tokens A labels `I`.
  overlap     a token is overlap-INTERSECTING only if its window meets the
              actual pairwise intersection of two gold blocks; windows that
              straddle that boundary are counted separately as ambiguous rather
              than assigned.

Writes aggregates only. Transcript text never leaves the cache.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.scoring import wtoks                       # noqa: E402
from eval.gold_set_score import (                                    # noqa: E402
    GOLD, REGIONS, cluster_ci, load, masked_intervals, pub_utterances,
    region_blocks, region_span, in_intervals, wilson,
)
from eval.soniox_confidence_probe import _align, group_words, word_units  # noqa: E402

TOKDIR = GOLD / "hyp" / "soniox-tokens"
ADPDIR = GOLD / "hyp" / "adapter"
OUT = ROOT / "eval" / "results_insertion_fidelity.json"

PRIMARY_REGION = "core_envelope"
TAU = 0.5
TAU_GRID = (0.25, 0.5, 1.0)
TAU_W = 1.5
TAU_W_GRID = (0.75, 1.5, 3.0)
SYSTEMS = ("ADP", "SNX")
PUB_RULES = ("midpoint", "any_overlap", "wholly_contained")

SUPPORTED, UNDEC, UNSUPPORTED = "gold_supported", "undecidable", "not_supported"


# --------------------------------------------------------------------- intervals
def merge(ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(ivs):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]):
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append((s, e))
    return merge(out)


def contains(cov: list[tuple[float, float]], s: float, e: float) -> bool:
    """Is [s, e] wholly inside the merged coverage?"""
    if e <= s:
        e = s + 1e-9
    cur = s
    for a, b in cov:
        if b <= cur:
            continue
        if a > cur:
            return False
        cur = b
        if cur >= e:
            return True
    return cur >= e


def meets(cov: list[tuple[float, float]], s: float, e: float) -> bool:
    return any(b > s and a < max(e, s + 1e-9) for a, b in cov)


def pairwise_overlap_regions(blocks: list[dict]) -> list[tuple[float, float]]:
    """Where two gold blocks genuinely sound at the same time.

    `ov_with` marks a block as participating in an overlap; it does not say that
    every instant of that block is simultaneous speech. Only the intersection of
    the two intervals is.
    """
    out = []
    by_id = {b["id"]: b for b in blocks}
    for b in blocks:
        for other in b.get("ov_with") or []:
            o = by_id.get(other)
            if not o:
                continue
            s, e = max(b["s"], o["s"]), min(b["e"], o["e"])
            if e > s:
                out.append((s, e))
    return merge(out)


# ------------------------------------------------------------------ system views
def adp_units(cid: str, ivs) -> list[dict]:
    d = json.loads((ADPDIR / f"{cid}.json").read_text())
    words = []
    for u in d["transcription"]["utterances"]:
        for w in u["words"]:
            words.append((float(w["start"]), float(w.get("end", w["start"])), w["word"]))
    words.sort()
    out = []
    for s, e, raw in words:
        if not in_intervals(s, ivs):
            continue
        for t in wtoks(raw):
            out.append({"tok": t, "start": s, "end": max(e, s)})
    return out


def snx_units(cid: str, ivs) -> tuple[list[dict], dict]:
    """Soniox realtime words, assembled exactly as in the confidence probe."""
    p = TOKDIR / f"{cid}.json"
    if not p.exists():
        return [], {"missing_cells": 1}
    words, stats = group_words(json.loads(p.read_text())["tokens"])
    words = [w for w in words if in_intervals(w["start"], ivs)]
    us, dropped, split = word_units(words)
    return ([{"tok": u["tok"], "start": u["start"], "end": max(u["end"], u["start"])}
             for u in us],
            dict(stats, punct_only_dropped=dropped, words_split=split))


def pub_units(city, meeting, clip_start: float, ivs, rule: str) -> list[str]:
    """Published tokens under one boundary rule. `midpoint` is the frozen one."""
    seq = []
    for u in pub_utterances(city, meeting):
        s, e = u["s"] - clip_start, u["e"] - clip_start
        if rule == "midpoint":
            keep = in_intervals((s + e) / 2.0, ivs)
        elif rule == "any_overlap":
            keep = meets(ivs, s, e)
        else:
            keep = contains(ivs, s, e)
        if keep:
            seq += wtoks(u["text"])
    return seq


# ------------------------------------------------------------------- the matching
def local_support_matching(units: list[dict], gold: list[dict], tau: float):
    """Injective system-token -> certain-gold-occurrence matching, A-blind.

    Edge iff the tokens are equal and the system word's window, dilated by `tau`,
    meets the gold occurrence's BLOCK interval. Every system token competes, not
    only the ones alignment A called insertions -- otherwise a duplicate could
    claim a gold occurrence another token already represents.

    MAXIMUM CARDINALITY, by augmenting paths, with each token's candidates ordered
    by (temporal distance, gold index). A greedy pass is not enough: if token 1
    can take gold A or B while token 2 can only take A, greedy gives token 1 A and
    leaves token 2 unmatched, so the supported COUNT would depend on processing
    order. Cardinality here does not. Which of two equally admissible gold
    occurrences a token receives still can, and that residual is why the
    PUB-unmatched verdict is reported as a forced/possible band rather than a
    point.
    """
    adj: dict[int, list[int]] = {}
    for j, u in enumerate(units):
        s, e = u["start"] - tau, u["end"] + tau
        cand = []
        for i, g in enumerate(gold):
            if g["tok"] == u["tok"] and g["e"] > s and g["s"] < e:
                cand.append((abs(u["start"] - (g["s"] + g["e"]) / 2.0), i))
        cand.sort()
        adj[j] = [i for _d, i in cand]

    gold2tok: dict[int, int] = {}

    def augment(j: int, seen: set[int]) -> bool:
        for i in adj[j]:
            if i in seen:
                continue
            seen.add(i)
            if i not in gold2tok or augment(gold2tok[i], seen):
                gold2tok[i] = j
                return True
        return False

    for j in sorted(adj, key=lambda x: (len(adj[x]), x)):
        augment(j, set())
    return {j: i for i, j in gold2tok.items()}


def pub_match_band(gold_toks: list[str], pseq: list[str]) -> tuple[set[int], set[int]]:
    """Which gold occurrences PUB matches, over ALL minimum-cost alignments.

    A single backtrace picks one optimal alignment out of many, and for a repeated
    word it can attach the match to the wrong occurrence. So the verdict "PUB does
    not have this occurrence" is computed as a band instead: forward and backward
    optimal costs mark every lattice cell that lies on some minimum-cost path,
    then row `i` is inspected for which edge types are available on such paths.

    Returns (matched_in_every_optimal, matched_in_some_optimal). A gold occurrence
    is FORCED-unmatched when it is in neither, and POSSIBLY-unmatched when it is
    only in the second.
    """
    n, m = len(gold_toks), len(pseq)
    inf = float("inf")
    f = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        f[i][0] = i
    for j in range(1, m + 1):
        f[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            f[i][j] = min(f[i - 1][j - 1] + (gold_toks[i - 1] != pseq[j - 1]),
                          f[i - 1][j] + 1, f[i][j - 1] + 1)
    g = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        g[i][m] = n - i
    for j in range(m - 1, -1, -1):
        g[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            g[i][j] = min(g[i + 1][j + 1] + (gold_toks[i] != pseq[j]),
                          g[i + 1][j] + 1, g[i][j + 1] + 1)
    best = f[n][m]
    on = [[f[i][j] + g[i][j] == best for j in range(m + 1)] for i in range(n + 1)]

    some_m, other_edge = set(), set()
    for i in range(1, n + 1):
        for j in range(m + 1):
            if not on[i][j]:
                continue
            if j and on[i - 1][j - 1] and f[i - 1][j - 1] + (
                    gold_toks[i - 1] != pseq[j - 1]) == f[i][j]:
                (some_m if gold_toks[i - 1] == pseq[j - 1] else other_edge).add(i - 1)
            if on[i - 1][j] and f[i - 1][j] + 1 == f[i][j]:
                other_edge.add(i - 1)          # a deletion consumes this gold token
    every_m = {i for i in some_m if i not in other_edge}
    return every_m, some_m


def corroborate(rows: list[dict], other: list[dict], tau_w: float, key: str) -> None:
    """Mark insertions ECHOED by the other system.

    The counterpart may be ANY of the other system's tokens in the region, not
    only its own insertions, so the relation is deliberately not symmetric and the
    two directions have different counts. Injective on the other side: one echo
    supports at most one insertion.
    """
    edges = []
    for r_i, r in enumerate(rows):
        for o_i, o in enumerate(other):
            if o["tok"] == r["tok"] and abs(o["start"] - r["start"]) <= tau_w:
                edges.append((abs(o["start"] - r["start"]), o_i, r_i))
    edges.sort()
    used_o: set[int] = set()
    hit: set[int] = set()
    for _d, o_i, r_i in edges:
        if o_i in used_o or r_i in hit:
            continue
        used_o.add(o_i)
        hit.add(r_i)
    for r_i, r in enumerate(rows):
        r[key] = r_i in hit


# ------------------------------------------------------------------------- a cell
def build_cell(cid: str, ans: dict, sel: dict, region: str, pub_rule: str = "midpoint"):
    """Everything one cell contributes, before any classification."""
    row = sel[cid]
    blocks = sorted(region_blocks(ans[cid]["b"]["blocks"], region),
                    key=lambda b: (b["s"], b["id"]))
    span = region_span(blocks, region)
    if span is None:
        return None
    outside = [b for b in ans[cid]["b"]["blocks"]
               if b["id"] not in {x["id"] for x in blocks}]
    ivs = masked_intervals(blocks, span, outside)

    gold: list[dict] = []
    for b in blocks:
        if b.get("text_unc"):
            continue
        for t in wtoks(b["text"]):
            gold.append({"tok": t, "block": b["id"], "s": b["s"], "e": b["e"],
                         "ov": bool(b.get("ov_with"))})
    if not gold:
        return None

    certain_cov = intersect(merge([(b["s"], b["e"]) for b in blocks
                                   if not b.get("text_unc")]), merge(ivs))
    ov_regions = intersect(pairwise_overlap_regions(
        [b for b in blocks if not b.get("text_unc")]), merge(ivs))

    pseq = pub_units(row["city_id"], row["meeting_id"], float(row["clip_start"]),
                     ivs, pub_rule)
    adp = adp_units(cid, ivs)
    snx, snx_stats = snx_units(cid, ivs)
    return {"cell": cid, "meeting": row["meeting_id"], "city": row["city_id"],
            "stream": row["draw_stream"], "gold": gold, "pseq": pseq,
            "blocks": blocks,
            "units": {"ADP": adp, "SNX": snx}, "certain_cov": certain_cov,
            "ov_regions": ov_regions, "snx_stats": snx_stats}


def classify(cell: dict, system: str, tau: float = TAU,
             priority=("S", "D", "I"), reverse=False) -> list[dict]:
    units, gold, pseq = cell["units"][system], cell["gold"], cell["pseq"]
    gtoks = [g["tok"] for g in gold]
    hyp = [u["tok"] for u in units]

    lab_a = {j: op for op, _i, j in _align(pseq, hyp, priority, reverse) if j >= 0}
    lab_b = {j: op for op, _i, j in _align(gtoks, hyp, priority, reverse) if j >= 0}
    pub_every, pub_some = pub_match_band(gtoks, pseq)
    tok2gold = local_support_matching(units, gold, tau)

    out = []
    for j, u in enumerate(units):
        if lab_a.get(j) != "I":
            continue
        s, e = u["start"] - tau, u["end"] + tau
        r = {"tok": u["tok"], "start": u["start"],
             "overlap": bool(meets(cell["ov_regions"], u["start"], u["end"])),
             "overlap_ambiguous": bool(meets(cell["ov_regions"], s, e)
                                       and not contains(cell["ov_regions"], s, e)),
             "align_b_match": lab_b.get(j) == "M",
             # the LOOSER, block-level notion: the token sits in (or beside) a
             # block that participates in an overlap somewhere. Reported beside
             # the strict intersection because the strict one turns out to have
             # almost no mass; it is adjacency, not simultaneity.
             "overlap_associated": any(
                 (b["s"] - tau) <= u["start"] < (b["e"] + tau)
                 for b in cell["blocks"] if b.get("ov_with"))}
        gi = tok2gold.get(j)
        if gi is not None:
            r["cls"] = SUPPORTED
            # FORCED: PUB matches this gold occurrence in no minimum-cost
            # alignment. POSSIBLE: there is at least one in which it does not.
            r["pub_unmatched_forced"] = gi not in pub_some
            r["pub_unmatched_possible"] = gi not in pub_every
            r["gold_overlap_block"] = gold[gi]["ov"]
        else:
            r["pub_unmatched_forced"] = r["pub_unmatched_possible"] = None
            r["cls"] = UNDEC if not contains(cell["certain_cov"], s, e) else UNSUPPORTED
        out.append(r)
    return out


# ------------------------------------------------------------------- aggregation
def share(rows):
    r = cluster_ci(rows)
    r["naive_token_ci95"] = wilson(r["num"], r["den"])
    return r


def _rows(cells, numf, denf):
    """Rows for a POOLED ratio: sum(numerator) / sum(denominator).

    Cells with a zero denominator are KEPT. Dropping them would have removed the
    cell whose scored region has no published tokens at all -- precisely the case
    most suggestive of reference omission -- and would have quietly shrunk the
    numerator too. Codex caught exactly that arithmetic in review (job
    cea10e78): the first version reported 69/905 for a system with 76
    insertions.
    """
    return [(c["meeting"], numf(c), denf(c)) for c in cells]


def summarise(cells: list[dict], label: str) -> dict:
    """Pooled micro-ratios with meeting-clustered intervals.

    Never a product of two separately averaged quantities: every rate below is
    sum(numerator) / sum(denominator) over the resampled meetings, and the
    matching that produced the rows travels with the cell it belongs to, so a
    meeting resample recomputes the whole thing.
    """
    n_ins = lambda c: len(c["rows"])                                    # noqa: E731
    n_pub = lambda c: c["n_pub"]                                        # noqa: E731
    cnt = lambda k: (lambda c: sum(1 for r in c["rows"] if r["cls"] == k))   # noqa: E731
    forced = lambda c: sum(1 for r in c["rows"]                         # noqa: E731
                           if r["pub_unmatched_forced"] is True)
    possible = lambda c: sum(1 for r in c["rows"]                       # noqa: E731
                             if r["pub_unmatched_possible"] is True)
    matched = lambda c: sum(1 for r in c["rows"]                        # noqa: E731
                            if r["cls"] == SUPPORTED
                            and r["pub_unmatched_possible"] is False)
    out = {
        "label": label,
        "n_cells": len(cells),
        "n_insertions": sum(len(c["rows"]) for c in cells),
        "n_pub_tokens": sum(c["n_pub"] for c in cells),
        "n_system_tokens": sum(c["n_sys"] for c in cells),
        "n_gold_tokens": sum(c["n_gold"] for c in cells),
        "counts": {
            SUPPORTED: sum(cnt(SUPPORTED)(c) for c in cells),
            "supported_pub_unmatched_forced": sum(forced(c) for c in cells),
            "supported_pub_unmatched_possible": sum(possible(c) for c in cells),
            "supported_pub_matched_in_every_optimal": sum(matched(c) for c in cells),
            UNDEC: sum(cnt(UNDEC)(c) for c in cells),
            UNSUPPORTED: sum(cnt(UNSUPPORTED)(c) for c in cells),
        },
    }
    if not out["n_insertions"]:
        return out
    for k in (SUPPORTED, UNDEC, UNSUPPORTED):
        out[f"share_{k}"] = share(_rows(cells, cnt(k), n_ins))
    out["share_pub_unmatched_forced"] = share(_rows(cells, forced, n_ins))
    out["share_pub_unmatched_possible"] = share(_rows(cells, possible, n_ins))
    out["share_supported_but_pub_matched"] = share(_rows(cells, matched, n_ins))
    out["share_align_b_match_SENSITIVITY"] = share(_rows(
        cells, lambda c: sum(1 for r in c["rows"] if r["align_b_match"]), n_ins))
    out["insertion_rate_vs_pub"] = share(_rows(cells, n_ins, n_pub))
    out["pub_unmatched_component_of_insertion_rate"] = share(
        _rows(cells, forced, n_pub))
    out["pub_unmatched_component_possible"] = share(_rows(cells, possible, n_pub))
    out["pub_unmatched_component_upper_incl_undecidable"] = share(_rows(
        cells, lambda c: possible(c) + cnt(UNDEC)(c), n_pub))
    out["gold_supported_component_of_insertion_rate"] = share(
        _rows(cells, cnt(SUPPORTED), n_pub))
    # splits: genuine simultaneous speech vs the rest, ambiguous windows apart
    for name, pred in (("overlap", lambda r: r["overlap"]),
                       ("non_overlap", lambda r: not r["overlap"]),
                       ("overlap_boundary_ambiguous", lambda r: r["overlap_ambiguous"]),
                       ("overlap_associated_LOOSE", lambda r: r["overlap_associated"]),
                       ("not_overlap_associated_LOOSE",
                        lambda r: not r["overlap_associated"])):
        sub = [{**c, "rows": [r for r in c["rows"] if pred(r)]} for c in cells]
        n = sum(len(c["rows"]) for c in sub)
        d = {"n_insertions": n}
        if n:
            for k in (SUPPORTED, UNDEC, UNSUPPORTED):
                d[f"share_{k}"] = share(_rows(sub, cnt(k), n_ins))
            d["share_pub_unmatched_forced"] = share(_rows(sub, forced, n_ins))
        out[f"split_{name}"] = d
    # single-item domination is checked at the CELL, because the cell is the item
    # this set draws; one window has supplied 67% of a headline effect here before.
    out["per_cell"] = sorted(
        [{"cell": c["cell"], "meeting": c["meeting"], "stream": c["stream"],
          "n_insertions": len(c["rows"]), "n_pub": c["n_pub"],
          "n_gold_supported": cnt(SUPPORTED)(c),
          "n_pub_unmatched_forced": forced(c),
          "n_undecidable": cnt(UNDEC)(c), "n_not_supported": cnt(UNSUPPORTED)(c)}
         for c in cells], key=lambda r: -r["n_insertions"])
    top = out["per_cell"][0] if out["per_cell"] else None
    out["largest_cell_share_of_insertions"] = (
        top["n_insertions"] / out["n_insertions"] if top else None)
    out["largest_cell_share_of_pub_unmatched"] = (
        max((c["n_pub_unmatched_forced"] for c in out["per_cell"]), default=0)
        / max(out["counts"]["supported_pub_unmatched_forced"], 1))
    by_stream = defaultdict(list)
    for c in cells:
        by_stream[c["stream"]].append(c)
    out["by_draw_stream"] = {
        s: {"n_cells": len(cs), "n_insertions": sum(len(c["rows"]) for c in cs),
            "n_supported": sum(cnt(SUPPORTED)(c) for c in cs),
            "n_pub_unmatched_forced": sum(forced(c) for c in cs),
            "n_undecidable": sum(cnt(UNDEC)(c) for c in cs)}
        for s, cs in sorted(by_stream.items())}
    return out


def pack(cells_raw, system, tau=TAU, priority=("S", "D", "I"), reverse=False):
    out = []
    for c in cells_raw:
        rows = classify(c, system, tau, priority, reverse)
        out.append({"meeting": c["meeting"], "city": c["city"], "stream": c["stream"],
                    "cell": c["cell"], "rows": rows, "n_pub": len(c["pseq"]),
                    "n_sys": len(c["units"][system]), "n_gold": len(c["gold"])})
    return out


# ------------------------------------------------------------------------- main
def pub_vs_gold(cells_raw, priority=("S", "D", "I"), reverse=False) -> dict:
    """PUB's own gold-unsupported tokens. A SEPARATE estimand, never merged.

    This is fidelity-to-audio for the published text; the numbers above are
    agreement-with-OpenCouncil. They are different quantities and the project
    rule is that they never become one number.
    """
    per = []
    for c in cells_raw:
        gt = [g["tok"] for g in c["gold"]]
        ops = _align(gt, c["pseq"], priority, reverse)
        per.append((c["meeting"], sum(1 for op, _i, _j in ops if op == "I"), len(gt)))
    return {"pub_insertions_vs_gold_per_gold_token": share(per),
            "note": "PUB tokens unsupported by THIS gold, not a bound on PUB's "
                    "true added words: one listener, one pass, and 17 spans the "
                    "human judged as lost speech were never written into the "
                    "reference."}


def overlap_direct(cells_raw) -> dict:
    """The DIRECT test of "PUB loses speech in overlap", which the insertion split
    is not.

    Conditioning on a system having produced an insertion asks a different
    question and, with 0 and 7 insertions in strict overlap, cannot answer this
    one. This instead takes GOLD occurrences as the denominator and asks how often
    PUB fails to match them, inside genuine simultaneous speech and outside it.
    Gold occurrences carry BLOCK-level times, and a block is longer than its
    intersection with another block, so `block_wholly_inside_overlap` is empty by
    construction and is reported as zero rather than quietly dropped. The
    informative contrast is a block that TOUCHES simultaneous speech against one
    that does not.
    """
    buckets = {"block_wholly_inside_overlap": [], "block_touches_overlap": [],
               "block_outside_overlap": []}
    for c in cells_raw:
        gt = [g["tok"] for g in c["gold"]]
        _every, some = pub_match_band(gt, c["pseq"])
        for i, g in enumerate(c["gold"]):
            if contains(c["ov_regions"], g["s"], g["e"]):
                key = "block_wholly_inside_overlap"
            elif meets(c["ov_regions"], g["s"], g["e"]):
                key = "block_touches_overlap"
            else:
                key = "block_outside_overlap"
            buckets[key].append((c["meeting"], int(i not in some), 1))
    return {k: (share(v) if v else {"n": 0}) for k, v in buckets.items()}


def timestamp_diagnostic(cells_raw, tau: float = TAU) -> dict:
    """Is the undecidable gap between two systems about their words or their clocks?

    Word duration and the fraction of ALL region tokens (not only insertions)
    whose dilated window lies wholly inside certain gold coverage. If the two
    systems place words equally well, the undecidable gap is not a clock artefact.
    """
    out = {}
    for s in SYSTEMS:
        durs = [u["end"] - u["start"] for c in cells_raw for u in c["units"][s]]
        durs.sort()
        out[s] = {
            "n_tokens": len(durs),
            "mean_word_duration_s": (sum(durs) / len(durs)) if durs else None,
            "median_word_duration_s": durs[len(durs) // 2] if durs else None,
        }
    return out


def coverage_diagnostic(cells_raw, tau: float = TAU) -> dict:
    """Where each system PUTS its words, insertions or not.

    The undecidable class is a property of the timestamp as much as of the word:
    a system whose word times drift outside the human's annotated blocks earns
    undecidables it did not deserve. This measures that directly, over ALL the
    system's tokens in the region, so the report can say whether the difference
    in undecidable share between two systems is about their words or about their
    clocks.
    """
    out = {}
    for s in SYSTEMS:
        rows = []
        for c in cells_raw:
            n = sum(1 for u in c["units"][s]
                    if contains(c["certain_cov"], u["start"] - tau, u["end"] + tau))
            rows.append((c["meeting"], n, len(c["units"][s])))
        out[s] = share(rows)
    return out


def main():
    man, ans, _cells, sel = load()
    scored = man["scored_cell_ids"]
    res = {
        "protocol": "insertion-fidelity-2026-08-17",
        "answers_sha256": man["answers_sha256"],
        "cells_sha256": man["cells_sha256"],
        "systems_covered": list(SYSTEMS),
        "systems_not_covered": ["W (three-way fusion)", "ElevenLabs Scribe v2",
                                "Soniox stt-async-v5 (paid)"],
        "frozen_before_looking": {
            "primary_region": PRIMARY_REGION, "tau_s": TAU, "tau_grid": list(TAU_GRID),
            "tau_w_s": TAU_W, "tau_w_grid": list(TAU_W_GRID),
            "pub_boundary_rule": "midpoint (the frozen scorer's rule)",
            "primary_support": "occurrence-level injective local matching, "
                               "objective blind to alignment A",
            "design_review": "Codex job 847c449feb004c4abbf9b5243cdcd75c, before "
                             "any number existed",
        },
        "n_cells": len(scored),
        "n_meetings": len({sel[c]["meeting_id"] for c in scored}),
        "regions": {},
    }

    for region in REGIONS:
        raw = [c for c in (build_cell(cid, ans, sel, region) for cid in scored) if c]
        r = {"n_cells_built": len(raw), "pub_vs_gold": pub_vs_gold(raw),
             "word_time_inside_certain_coverage": coverage_diagnostic(raw),
             "word_duration": timestamp_diagnostic(raw),
             "gold_occurrences_pub_fails_to_match": overlap_direct(raw)}
        for s in SYSTEMS:
            r[s] = summarise(pack(raw, s), s)
        res["regions"][region] = r

        if region != PRIMARY_REGION:
            continue

        # ---- sensitivity: tau grid
        r["tau_sensitivity"] = {
            s: {str(t): {k: v for k, v in summarise(pack(raw, s, tau=t), s).items()
                         if k in ("counts", "n_insertions")}
                for t in TAU_GRID}
            for s in SYSTEMS}

        # ---- sensitivity: alignment tie-break envelope (A, B and C together)
        env = []
        for pri in permutations(("S", "D", "I")):
            for rev in (False, True):
                row = {"priority": "".join(pri), "reversed": rev}
                for s in SYSTEMS:
                    d = summarise(pack(raw, s, priority=pri, reverse=rev), s)
                    row[s] = {"n_insertions": d["n_insertions"],
                              "counts": d["counts"],
                              "share_supported": (d.get("share_gold_supported") or {}).get("point"),
                              "share_pub_unmatched": (d.get("share_pub_unmatched_forced") or {}).get("point")}
                env.append(row)
        r["alignment_sensitivity"] = env

        # ---- sensitivity: PUB boundary rule
        bnd = {}
        for rule in PUB_RULES:
            raw2 = [c for c in (build_cell(cid, ans, sel, region, rule) for cid in scored) if c]
            bnd[rule] = {s: {k: v for k, v in summarise(pack(raw2, s), s).items()
                             if k in ("counts", "n_insertions", "n_pub_tokens")}
                         for s in SYSTEMS}
        r["pub_boundary_sensitivity"] = bnd

        # ---- cross-system corroboration: the class the cut F3 family would delete
        cor = {}
        for s, other in (("ADP", "SNX"), ("SNX", "ADP")):
            packed = pack(raw, s)
            for tw in TAU_W_GRID:
                for c, craw in zip(packed, raw):
                    corroborate(c["rows"], craw["units"][other], tw, f"w{tw}")
            d = {}
            for tw in TAU_W_GRID:
                sub = [{**c, "rows": [r for r in c["rows"] if r[f"w{tw}"]]} for c in packed]
                n = sum(len(c["rows"]) for c in sub)
                e = {"n_insertions_corroborated": n}
                if n:
                    for k in (SUPPORTED, UNDEC, UNSUPPORTED):
                        e[f"share_{k}"] = share(_rows(
                            sub, lambda c, kk=k: sum(1 for x in c["rows"] if x["cls"] == kk),
                            lambda c: len(c["rows"])))
                    e["share_pub_unmatched_forced"] = share(_rows(
                        sub,
                        lambda c: sum(1 for x in c["rows"]
                                      if x["pub_unmatched_forced"] is True),
                        lambda c: len(c["rows"])))
                d[str(tw)] = e
            cor[f"{s}_corroborated_by_{other}"] = d
        r["cross_system_corroboration"] = cor
        r["soniox_word_stats"] = dict(sum((Counter(c["snx_stats"]) for c in raw), Counter()))

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")
    p = res["regions"][PRIMARY_REGION]
    for s in SYSTEMS:
        d = p[s]
        print(s, d["n_insertions"], "insertions", d["counts"])
    return res


if __name__ == "__main__":
    main()
