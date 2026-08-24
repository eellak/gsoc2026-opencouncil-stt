#!/usr/bin/env python3
"""Stage 0 of `docs/specs/2026-08-21-fusion-production.md`: the free two-system screen.

Zero GPU, zero API. Everything runs on the already-cached 247-window substrate of
`2026-08-10-corrected-adapter-label-prefix-fix-vs-ju` (the same substrate as
`exp-2026-08-16-composition-over-selection`), with the 6 sealed temporal-holdout
windows of `eval-freeze-2026-08` removed by the same explicit filter.

WHAT THIS IS. The spec's Stage 0: per-pair diagnostics first, then
3 pairs x 2 bases x 3 arms, every fitted quantity learned LEAVE-ONE-CITY-OUT and the
headline computed only from out-of-fold predictions.

WHAT THIS IS NOT, and every deviation is declared in the output JSON under
`deviations`:

  * There are NO per-word timestamps and NO per-word confidences on this substrate.
    The benchmark report carries hypothesis TEXT only. The project's cached Soniox
    word tokens are `stt-rt-v4` and its cached adapter confidences are a separate
    decode pass; joining either to this run's text is forbidden by
    `exp-2026-08-18-conf-substrate` (0 of 133 windows reproduce) and by the caveat on
    `artifact-soniox-rt-tokens-2026-08-16`. So:
      - the anchored alignment of spec 2.2 is NOT exercised; alignment here is plain
        pairwise text DP, which is exactly the drift-prone thing 2.2 wants to replace;
      - R2's 0.30 s clause degrades to a token-count clause;
      - R3 is fitted WITHOUT calibrated confidence, on span shape only;
      - R4 (diarization) is not implemented at all. It never restores alone, so its
        absence removes a flag, not a restore path;
      - arm P2 as written (calibrated cross-vendor confidence comparison) is NOT
        RUNNABLE. What runs under the name P2 is a context-calibrated surrogate that
        uses span shape, not confidence. It is reported as P2* everywhere.

  * R1 is implemented as a SUPPRESSOR, not a restorer. The spec's own Codex
    correction says an echo means the alignment failed locally and the right action
    is realign-and-collapse, never a licence to emit a second copy. "Restore if R1"
    would emit the duplicate the correction forbids, so restore here is
    `not R1 and R2 and R3`.

Writes results_fusion_pairs.json (aggregates only; never transcript text).

Env: SC (cache dir), N_BOOT (10000), WORKERS
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval import bench_data as B                        # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import rates, sdi        # noqa: E402
from eval.controlled_eval.msa import _align2, oracle_select             # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap, wtoks       # noqa: E402

RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
SCR, SNX, ADP = "scribe-v2-clean", "soniox", "oc-runpod-fixed-2026-08-10"
SHORT = {SCR: "scr", SNX: "snx", ADP: "adp"}
PAIRS = [(ADP, SNX), (SCR, SNX), (SCR, ADP)]
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
OUT = Path(__file__).with_name("results_fusion_pairs.json")

R2_GRID = (1, 2, 3)          # candidate minimum span length, fitted out of fold
R3_MIN_SUPPORT = 20          # train spans needed before a bucket may speak
P2_MIN_SUPPORT = 20


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------- ref alignment
def ref_align(hyp: list[str], ref: list[str]):
    """Align one hypothesis against the reference.

    Returns (cols, hyp_match, ref_match, hyp_col, ref_col) where
      hyp_match[i]  True if hyp token i sits in a column with an equal ref token
      ref_match[j]  ditto for reference token j
      hyp_col[i]    index into cols for hyp token i
    """
    cols = _align2(hyp, ref)
    hyp_match = [False] * len(hyp)
    ref_match = [False] * len(ref)
    hyp_col = [-1] * len(hyp)
    ref_col = [-1] * len(ref)
    i = j = 0
    for n, (h, r) in enumerate(cols):
        if h is not None:
            hyp_col[i] = n
        if r is not None:
            ref_col[j] = n
        if h is not None and r is not None and h == r:
            hyp_match[i] = True
            ref_match[j] = True
        if h is not None:
            i += 1
        if r is not None:
            j += 1
    return cols, hyp_match, ref_match, hyp_col, ref_col


def span_keep_drop(cols, hyp_col, i0, i1):
    """Local KEEP-minus-DROP edit cost for hypothesis tokens [i0, i1).

    KEEP = the edit ops the span already pays inside its own hyp-vs-ref alignment.
    DROP = every reference token in that column range becomes a deletion.
    Local approximation: it reads the span's own alignment to the reference and
    ignores how the merged sequence's context would shift the alignment.
    """
    c0, c1 = hyp_col[i0], hyp_col[i1 - 1]
    keep = 0
    nref = 0
    for n in range(c0, c1 + 1):
        h, r = cols[n]
        if r is not None:
            nref += 1
        if h is not None and r is not None:
            keep += 0 if h == r else 1
        elif h is not None:
            keep += 1
        elif r is not None:
            keep += 1
    return keep - nref, keep, nref


# ------------------------------------------------------------------- islands
def build_islands(cols):
    """Split a pairwise alignment into anchors and maximal disagreement islands.

    cols entries are (x, y). An anchor is a column where both sides emit the same
    normalised token. Returns a list of segments in order:
      ("anchor", token)
      ("island", dict)
    """
    segs = []
    run = []
    xi = yi = 0
    run_x0 = run_y0 = 0
    for (x, y) in cols:
        if x is not None and y is not None and x == y:
            if run:
                segs.append(("island", _island(run, run_x0, run_y0)))
                run = []
            segs.append(("anchor", x))
            xi += 1
            yi += 1
            run_x0, run_y0 = xi, yi
        else:
            run.append((x, y))
            if x is not None:
                xi += 1
            if y is not None:
                yi += 1
    if run:
        segs.append(("island", _island(run, run_x0, run_y0)))
    return segs


def _island(run, x0, y0):
    xt = [x for x, _ in run if x is not None]
    yt = [y for _, y in run if y is not None]
    if xt and yt:
        kind = "identity"
    elif xt:
        kind = "occ_x"
    else:
        kind = "occ_y"
    return {"kind": kind, "x": xt, "y": yt,
            "x0": x0, "x1": x0 + len(xt), "y0": y0, "y1": y0 + len(yt)}


# ------------------------------------------------------------------- assembly
def assemble(segs, base_is_x, priority_is_x, gate, identity_pick):
    """Emit the merged token stream for one window under one arm.

    base_is_x       the base system is X (else Y)
    priority_is_x   identity islands default to X's path (else Y's)
    gate            None for P0, else a callable(island) -> True to restore an
                    other-only span
    identity_pick   None, else callable(island) -> True to take X's path
    Returns (tokens, from_island_flags, accounting Counter).
    """
    out, flags = [], []
    acct = Counter()
    for kind, payload in segs:
        if kind == "anchor":
            out.append(payload)
            flags.append(False)
            continue
        isl = payload
        if isl["kind"] == "identity":
            take_x = priority_is_x
            if identity_pick is not None:
                p = identity_pick(isl)
                if p is not None:
                    take_x = p
                    acct["identity_calibrated"] += 1
            toks = isl["x"] if take_x else isl["y"]
            acct["identity_islands"] += 1
            out.extend(toks)
            flags.extend([True] * len(toks))
            continue
        # occupancy
        own = (isl["kind"] == "occ_x") == base_is_x
        if own:
            acct["occ_base_kept"] += 1
            toks = isl["x"] if isl["kind"] == "occ_x" else isl["y"]
            out.extend(toks)
            flags.extend([False] * len(toks))
            continue
        acct["occ_other_candidates"] += 1
        if gate is not None and gate(isl):
            acct["occ_other_restored"] += 1
            toks = isl["x"] if isl["kind"] == "occ_x" else isl["y"]
            out.extend(toks)
            flags.extend([True] * len(toks))
        else:
            acct["occ_other_dropped"] += 1
    return out, flags, acct


# ----------------------------------------------------------------------- main
def main():
    report = B.load_report(RUN_ID)
    providers = B.provider_ids(report)
    for p in (SCR, SNX, ADP):
        assert p in providers, f"{p} missing from {RUN_ID}"
    items = B.common_items(report, providers)
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())["holdout_windows"]}
    before = len(items)
    items = [it for it in items if it["item_id"] not in sealed]
    assert before - len(items) == 6, f"removed {before - len(items)} sealed, expected 6"
    assert len(items) == 247, f"expected 247 windows, got {len(items)}"
    if os.environ.get("LIMIT"):
        items = items[:int(os.environ["LIMIT"])]
        log(f"LIMIT: {len(items)} windows -- SMOKE RUN, not reportable")

    # CLUSTER on (cityId, meetingId), not meetingId alone.
    clusters = [f"{it['city_id']}/{it['meeting_id']}" for it in items]
    cities = [it["city_id"] for it in items]
    log(f"{len(items)} windows, {len(set(clusters))} (city,meeting) clusters, "
        f"{len(set(cities))} cities")

    toks = {it["item_id"]: {p: wtoks(it["hyp"][p]) for p in (SCR, SNX, ADP)}
            for it in items}
    refs = {it["item_id"]: wtoks(it["ref"]) for it in items}

    # ---- per-hypothesis reference alignment (shared by every pair) ----
    RA = {}
    for it in items:
        w = it["item_id"]
        RA[w] = {p: ref_align(toks[w][p], refs[w]) for p in (SCR, SNX, ADP)}

    # ---- single-system baselines, on this scorer ----
    singles = {p: [sdi(it["ref"], it["hyp"][p]) for it in items] for p in (SCR, SNX, ADP)}

    res = {
        "run_id": RUN_ID,
        "n_items": len(items),
        "n_clusters": len(set(clusters)),
        "cluster_key": "(cityId, meetingId)",
        "n_cities": len(set(cities)),
        "n_boot": N_BOOT,
        "scorer": "eval/controlled_eval/scoring.py + exp_fusion_deletions.sdi",
        "alignment": "pairwise text DP (msa._align2); NO timestamps on this substrate",
        "singles": {SHORT[p]: rates(v) for p, v in singles.items()},
        "deviations": [
            "no per-word timestamps and no per-word confidence exist for this run's "
            "hypothesis text; the anchored alignment of spec 2.2 is not exercised",
            "R2's 0.30 s clause degraded to a token-count clause fitted out of fold",
            "R3 fitted without calibrated confidence (span shape only)",
            "R4 (diarization) not implemented; it never restores alone",
            "arm P2 as specified is not runnable (no confidence); P2* is a "
            "context-calibrated surrogate on span shape",
            "R1 implemented as a suppressor (drop on echo), per the spec's own Codex "
            "correction that an echo means realign-and-collapse, not a second copy",
            "KEEP-minus-DROP costs are LOCAL: read off each span's own hypothesis-to-"
            "reference alignment, not off a re-scored merged window",
        ],
        "diagnostics": {},
        "arms": {},
        "contrasts": {},
        "loo": {},
    }

    PAIRDATA = {}
    for (X, Y) in PAIRS:
        pk = f"{SHORT[X]}+{SHORT[Y]}"
        log(f"== pair {pk}")
        per_window = {}
        for it in items:
            w = it["item_id"]
            cols = _align2(toks[w][X], toks[w][Y])
            segs = build_islands(cols)
            per_window[w] = {"cols": cols, "segs": segs}
        PAIRDATA[pk] = {"X": X, "Y": Y, "w": per_window}

        # ------------------------------------------------ diagnostics
        n_ref = 0
        both_ok = only_x = only_y = neither = 0
        by_city = defaultdict(lambda: [0, 0])       # city -> [union_correct, n_ref]
        by_meet = defaultdict(lambda: [0, 0])
        for it in items:
            w = it["item_id"]
            rx = RA[w][X][2]
            ry = RA[w][Y][2]
            for a, b in zip(rx, ry):
                n_ref += 1
                if a and b:
                    both_ok += 1
                elif a:
                    only_x += 1
                elif b:
                    only_y += 1
                else:
                    neither += 1
            u = sum(1 for a, b in zip(rx, ry) if a or b)
            by_city[it["city_id"]][0] += u
            by_city[it["city_id"]][1] += len(rx)
            key = f"{it['city_id']}/{it['meeting_id']}"
            by_meet[key][0] += u
            by_meet[key][1] += len(rx)

        # pairwise alignment-conditional oracle
        orc = []
        for it in items:
            w = it["item_id"]
            sel = oracle_select(per_window[w]["cols"], refs[w])
            orc.append(sdi(it["ref"], " ".join(t for t in sel if t is not None)))

        # identity disagreements: conditional accuracy per producer
        idis = Counter()
        for it in items:
            w = it["item_id"]
            mx, my = RA[w][X][1], RA[w][Y][1]
            xi = yi = 0
            for (a, b) in per_window[w]["cols"]:
                if a is not None and b is not None and a != b:
                    idis["n"] += 1
                    if mx[xi]:
                        idis["x_correct"] += 1
                    if my[yi]:
                        idis["y_correct"] += 1
                    if mx[xi] and not my[yi]:
                        idis["x_only"] += 1
                    elif my[yi] and not mx[xi]:
                        idis["y_only"] += 1
                    elif mx[xi] and my[yi]:
                        idis["both"] += 1
                    else:
                        idis["neither"] += 1
                if a is not None:
                    xi += 1
                if b is not None:
                    yi += 1

        # KEEP-DROP per singleton direction, and the span feature table
        spans = {"x": [], "y": []}
        for it in items:
            w = it["item_id"]
            for kind, payload in per_window[w]["segs"]:
                if kind != "island" or payload["kind"] == "identity":
                    continue
                side = "x" if payload["kind"] == "occ_x" else "y"
                P = X if side == "x" else Y
                cols_r, _, _, hyp_col, _ = RA[w][P]
                i0 = payload["x0"] if side == "x" else payload["y0"]
                i1 = payload["x1"] if side == "x" else payload["y1"]
                d, keep, nref = span_keep_drop(cols_r, hyp_col, i0, i1)
                spans[side].append({
                    "w": w, "city": it["city_id"],
                    "cluster": f"{it['city_id']}/{it['meeting_id']}",
                    "len": i1 - i0, "kd": d, "keep": keep, "nref": nref,
                    "echo": None,
                })

        def kd_summary(rows):
            if not rows:
                return None
            v = [r["kd"] for r in rows]
            return {
                "n_spans": len(rows),
                "n_tokens": sum(r["len"] for r in rows),
                "mean_keep_minus_drop": statistics.fmean(v),
                "total_keep_minus_drop": sum(v),
                "share_keep_better": sum(1 for x in v if x < 0) / len(v),
                "share_tie": sum(1 for x in v if x == 0) / len(v),
                "by_len": {str(L): {
                    "n": sum(1 for r in rows if min(r["len"], 4) == L),
                    "mean_kd": statistics.fmean(
                        [r["kd"] for r in rows if min(r["len"], 4) == L]) if any(
                        min(r["len"], 4) == L for r in rows) else None}
                    for L in (1, 2, 3, 4)},
            }

        cov = {
            "n_ref_tokens": n_ref,
            "union_coverage_correct": (both_ok + only_x + only_y) / n_ref,
            "both_wrong_rate": neither / n_ref,
            "both_correct": both_ok / n_ref,
            "exclusively_correct_x": only_x / n_ref,
            "exclusively_correct_y": only_y / n_ref,
        }
        res["diagnostics"][pk] = {
            "x": SHORT[X], "y": SHORT[Y],
            "union": cov,
            "pairwise_alignment_conditional_oracle": rates(orc),
            "identity_disagreements": {
                "n_columns": idis["n"],
                "x_correct_rate": idis["x_correct"] / idis["n"] if idis["n"] else None,
                "y_correct_rate": idis["y_correct"] / idis["n"] if idis["n"] else None,
                "x_only": idis["x_only"], "y_only": idis["y_only"],
                "both_correct": idis["both"], "neither_correct": idis["neither"],
                "decisive_share": ((idis["x_only"] + idis["y_only"]) / idis["n"]
                                   if idis["n"] else None),
                "x_wins_of_decisive": (idis["x_only"] / (idis["x_only"] + idis["y_only"])
                                       if (idis["x_only"] + idis["y_only"]) else None),
            },
            "keep_minus_drop": {"x_only_spans": kd_summary(spans["x"]),
                                "y_only_spans": kd_summary(spans["y"])},
            "stability_by_city": {c: {"union_coverage": a / b, "n_ref": b}
                                  for c, (a, b) in sorted(by_city.items())},
            "stability_by_meeting": {
                "n_meetings": len(by_meet),
                "union_coverage_p05": _pct([a / b for a, b in by_meet.values()], 0.05),
                "union_coverage_p50": _pct([a / b for a, b in by_meet.values()], 0.50),
                "union_coverage_p95": _pct([a / b for a, b in by_meet.values()], 0.95),
                "worst": min(((a / b, k) for k, (a, b) in by_meet.items()))[1],
            },
        }
        PAIRDATA[pk]["spans"] = spans
        PAIRDATA[pk]["idis"] = idis
        log(f"   union={cov['union_coverage_correct']:.4f} "
            f"both_wrong={cov['both_wrong_rate']:.4f} "
            f"oracle={rates(orc)['wer']:.4f}")

    if os.environ.get("DIAG_ONLY") != "1":
        run_arms(res, items, clusters, cities, toks, refs, RA, PAIRDATA, singles)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    log(f"-> {OUT}")
    return res


# =========================================================== arms (P0 / P1 / P2*)
_REPAIR_CACHE = {}


def echo_flag(span_toks, base_toks, b0, radius=10):
    """R1, as a SUPPRESSOR. True when the base stream already carries this span's
    tokens nearby, i.e. the island is an alignment artefact and restoring it would
    emit a second copy."""
    lo, hi = max(0, b0 - radius), min(len(base_toks), b0 + radius)
    near = set(base_toks[lo:hi])
    return bool(span_toks) and all(t in near for t in span_toks)


def fit_fold(train_idx, items, PD, pk, base_is_x, RA):
    """Everything fitted on the training cities of one fold."""
    X, Y = PD[pk]["X"], PD[pk]["Y"]
    train_w = {items[i]["item_id"] for i in train_idx}

    # --- frozen system priority for identity islands: conditional accuracy on
    #     identity DISAGREEMENTS, restricted to the training cities.
    xo = yo = 0
    for i in train_idx:
        w = items[i]["item_id"]
        mx, my = RA[w][X][1], RA[w][Y][1]
        xi = yi = 0
        for (a, b) in PD[pk]["w"][w]["cols"]:
            if a is not None and b is not None and a != b:
                if mx[xi] and not my[yi]:
                    xo += 1
                elif my[yi] and not mx[xi]:
                    yo += 1
            if a is not None:
                xi += 1
            if b is not None:
                yi += 1
    priority_is_x = xo >= yo

    # --- R3 buckets on the OTHER-only spans of the training cities
    other_side = "y" if base_is_x else "x"
    rows = [r for r in PD[pk]["spans"][other_side] if r["w"] in train_w]
    buckets = defaultdict(list)
    for r in rows:
        buckets[min(r["len"], 4)].append(r["kd"])
    r3 = {k: (statistics.fmean(v), len(v)) for k, v in buckets.items()}

    # --- R2 length threshold, grid-searched on the training spans under the same gate
    best_L, best_cost = R2_GRID[0], None
    for L in R2_GRID:
        cost = 0
        for r in rows:
            if r["echo"]:
                continue
            if r["len"] < L:
                continue
            m, n = r3.get(min(r["len"], 4), (0.0, 0))
            if n >= R3_MIN_SUPPORT and m < 0:
                cost += r["kd"]
        if best_cost is None or cost < best_cost:
            best_L, best_cost = L, cost

    # --- P2* identity buckets: which side's path carried more correct tokens
    p2 = defaultdict(lambda: [0, 0])
    for i in train_idx:
        w = items[i]["item_id"]
        mx, my = RA[w][X][1], RA[w][Y][1]
        for kind, isl in PD[pk]["w"][w]["segs"]:
            if kind != "island" or isl["kind"] != "identity":
                continue
            cx = sum(mx[k] for k in range(isl["x0"], isl["x1"]))
            cy = sum(my[k] for k in range(isl["y0"], isl["y1"]))
            key = (min(isl["x1"] - isl["x0"], 3), min(isl["y1"] - isl["y0"], 3))
            if cx > cy:
                p2[key][0] += 1
            elif cy > cx:
                p2[key][1] += 1
    p2 = {k: tuple(v) for k, v in p2.items()}
    return {"priority_is_x": priority_is_x, "R2_L": best_L, "R3": r3, "P2": p2,
            "train_x_only": xo, "train_y_only": yo}


def run_arms(res, items, clusters, cities, toks, refs, RA, PD, singles):
    from functools import lru_cache
    import serving_stack.name_repair as NR
    # The frozen repair is unchanged; only its pure edit-distance kernel is memoised,
    # because Stage 0 calls it a few thousand times on the same council vocabulary.
    if not getattr(NR.dl, "cache_info", None):
        NR.dl = lru_cache(maxsize=1 << 21)(NR.dl)
    repair = NR.repair
    from eval.controlled_eval.roster_lexicon import (
        admitted_mined, build_meeting_context, load_city_terms, load_rosters)
    import re as _re

    report_items = B.load_report(RUN_ID)["items"]
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())["holdout_windows"]}
    city_terms = load_city_terms()
    rosters = load_rosters()
    mined_by_city, _ = admitted_mined()
    per_city_freq = {c: Counter() for c in city_terms}
    total = Counter()
    for x in report_items:
        if x["itemId"] in sealed:
            continue
        for t in wtoks(x["referenceText"]):
            total[t] += 1
            if x["cityId"] in per_city_freq:
                per_city_freq[x["cityId"]][t] += 1
    CTX = {}
    for it in items:
        c = it["city_id"]
        if c not in city_terms:
            CTX[it["item_id"]] = None
            continue
        ctx, _a = build_meeting_context(c, it["meeting_id"], city_terms[c],
                                        mined_by_city.get(c, []), rosters,
                                        total - per_city_freq.get(c, Counter()))
        CTX[it["item_id"]] = ctx

    def phonetic(tokens, flags, ctx):
        """Frozen phonetic closed-list repair, applied ONLY to island tokens."""
        if ctx is None or not ctx.present or not tokens:
            return tokens, 0
        text = " ".join(tokens)
        if _re.findall(r"\w+", text) != tokens:
            return tokens, 0
        starts, pos = {}, 0
        for j, t in enumerate(tokens):
            starts[pos] = j
            pos += len(t) + 1
        key = (id(ctx), text)
        r = _REPAIR_CACHE.get(key)
        try:
            if r is None:
                r = repair(text, ctx)
                _REPAIR_CACHE[key] = r
        except Exception:
            return tokens, 0
        out = list(tokens)
        n = 0
        for c in r.changes:
            j = starts.get(c["start"])
            if j is None or not flags[j]:
                continue
            new = wtoks(c["replacement"])
            if len(new) != 1:
                continue
            if out[j] != new[0]:
                out[j] = new[0]
                n += 1
        return out, n

    city_list = sorted(set(cities))
    idx_by_city = defaultdict(list)
    for i, c in enumerate(cities):
        idx_by_city[c].append(i)

    res["arms"] = {}
    res["fold_params"] = {}
    res["contrasts"] = {}
    res["loo"] = {}
    res["accounting"] = {}

    for (X, Y) in PAIRS:
        pk = f"{SHORT[X]}+{SHORT[Y]}"
        for base_is_x in (True, False):
            base = SHORT[X] if base_is_x else SHORT[Y]
            other_side = "y" if base_is_x else "x"
            # R1 echo flags for the other-only spans against the base stream
            eflag = {}
            for it in items:
                w = it["item_id"]
                bt = toks[w][X] if base_is_x else toks[w][Y]
                for kind, isl in PD[pk]["w"][w]["segs"]:
                    if kind != "island":
                        continue
                    if isl["kind"] == "identity":
                        continue
                    if (isl["kind"] == "occ_x") == base_is_x:
                        continue
                    span = isl["x"] if isl["kind"] == "occ_x" else isl["y"]
                    b0 = isl["x0"] if base_is_x else isl["y0"]
                    eflag[(w, isl["x0"], isl["y0"])] = echo_flag(span, bt, b0)
            # attach echo to the islands used by the gate
            for it in items:
                w = it["item_id"]
                for kind, isl in PD[pk]["w"][w]["segs"]:
                    if kind != "island" or isl["kind"] == "identity":
                        continue
                    if (isl["kind"] == "occ_x") == base_is_x:
                        continue
                    isl["_echo"] = eflag[(w, isl["x0"], isl["y0"])]
            # rebuild the span table in island order so echo matches one-to-one
            rows = []
            for it in items:
                w = it["item_id"]
                for kind, isl in PD[pk]["w"][w]["segs"]:
                    if kind != "island" or isl["kind"] == "identity":
                        continue
                    if (isl["kind"] == "occ_x") == base_is_x:
                        continue
                    P = X if isl["kind"] == "occ_x" else Y
                    cols_r, _, _, hyp_col, _ = RA[w][P]
                    i0 = isl["x0"] if isl["kind"] == "occ_x" else isl["y0"]
                    i1 = isl["x1"] if isl["kind"] == "occ_x" else isl["y1"]
                    d, keep, nref = span_keep_drop(cols_r, hyp_col, i0, i1)
                    rows.append({"w": w, "city": it["city_id"], "len": i1 - i0,
                                 "kd": d, "echo": isl["_echo"]})
            PD[pk]["spans"][other_side] = rows

            arm_out = {a: {} for a in ("P0", "P1", "P2*")}
            acct = {a: Counter() for a in ("P0", "P1", "P2*")}
            fold_params = {}
            for held in city_list:
                train_idx = [i for i in range(len(items)) if cities[i] != held]
                fp = fit_fold(train_idx, items, PD, pk, base_is_x, RA)
                fold_params[held] = {"priority_is_x": fp["priority_is_x"],
                                     "R2_L": fp["R2_L"],
                                     "R3_buckets": {str(k): [round(v[0], 4), v[1]]
                                                    for k, v in fp["R3"].items()},
                                     "train_decisive_x": fp["train_x_only"],
                                     "train_decisive_y": fp["train_y_only"]}

                def gate(isl, fp=fp):
                    span = isl["x"] if isl["kind"] == "occ_x" else isl["y"]
                    if isl.get("_echo"):
                        return False
                    if len(span) < fp["R2_L"]:
                        return False
                    m, n = fp["R3"].get(min(len(span), 4), (0.0, 0))
                    return n >= R3_MIN_SUPPORT and m < 0

                def pick(isl, fp=fp):
                    key = (min(isl["x1"] - isl["x0"], 3), min(isl["y1"] - isl["y0"], 3))
                    v = fp["P2"].get(key)
                    if not v or sum(v) < P2_MIN_SUPPORT or v[0] == v[1]:
                        return None
                    return v[0] > v[1]

                for i in idx_by_city[held]:
                    it = items[i]
                    w = it["item_id"]
                    segs = PD[pk]["w"][w]["segs"]
                    t0, f0, a0 = assemble(segs, base_is_x, fp["priority_is_x"], None, None)
                    arm_out["P0"][w] = " ".join(t0)
                    acct["P0"] += a0
                    t1, f1, a1 = assemble(segs, base_is_x, fp["priority_is_x"], gate, None)
                    t1r, n1 = phonetic(t1, f1, CTX[w])
                    arm_out["P1"][w] = " ".join(t1r)
                    a1["phonetic_changes"] = n1
                    acct["P1"] += a1
                    t2, f2, a2 = assemble(segs, base_is_x, fp["priority_is_x"], gate, pick)
                    t2r, n2 = phonetic(t2, f2, CTX[w])
                    arm_out["P2*"][w] = " ".join(t2r)
                    a2["phonetic_changes"] = n2
                    acct["P2*"] += a2

            cell = f"{pk}|base={base}"
            res["fold_params"][cell] = fold_params
            best_single = min((SHORT[X], SHORT[Y]),
                              key=lambda s: res["singles"][s]["wer"])
            bs_counts = singles[X if SHORT[X] == best_single else Y]
            base_counts = singles[X if base_is_x else Y]
            for a in ("P0", "P1", "P2*"):
                rows_a = [sdi(it["ref"], arm_out[a][it["item_id"]]) for it in items]
                res["arms"][f"{cell}|{a}"] = rates(rows_a)
                res["accounting"][f"{cell}|{a}"] = dict(acct[a])
                res["contrasts"][f"{cell}|{a} vs best_single({best_single})"] = \
                    _contrast(rows_a, bs_counts, clusters)
                res["contrasts"][f"{cell}|{a} vs base({base})"] = \
                    _contrast(rows_a, base_counts, clusters)
                res["loo"][f"{cell}|{a} vs best_single({best_single})"] = {
                    "window": _loo(rows_a, bs_counts, [it["item_id"] for it in items]),
                    "meeting": _loo(rows_a, bs_counts, clusters),
                    "city": _loo(rows_a, bs_counts, cities),
                }
                log(f"  {cell}|{a:3s} wer={res['arms'][f'{cell}|{a}']['wer']:.4f} "
                    f"del={res['arms'][f'{cell}|{a}']['del_rate']:.4f} "
                    f"ins={res['arms'][f'{cell}|{a}']['ins_rate']:.4f} "
                    f"d_vs_best={res['contrasts'][f'{cell}|{a} vs best_single({best_single})']['wer']['delta']:+.5f}")
    return res


def _contrast(rows_a, rows_b, clusters):
    out = {}
    for metric, idx in (("wer", None), ("sub_rate", 0), ("del_rate", 1), ("ins_rate", 2)):
        ca = [((r[0] + r[1] + r[2]) if idx is None else r[idx], r[3]) for r in rows_a]
        cb = [((r[0] + r[1] + r[2]) if idx is None else r[idx], r[3]) for r in rows_b]
        out[metric] = cluster_bootstrap(ca, cb, clusters, n_boot=N_BOOT)
    return out


def _loo(rows_a, rows_b, keys):
    ca = [(r[0] + r[1] + r[2], r[3]) for r in rows_a]
    cb = [(r[0] + r[1] + r[2], r[3]) for r in rows_b]
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    allidx = list(range(len(ca)))

    def delta(idx):
        den = sum(ca[i][1] for i in idx)
        if not den:
            return None
        return (sum(ca[i][0] for i in idx) - sum(cb[i][0] for i in idx)) / den
    full = delta(allidx)
    ds = []
    for k, idx in groups.items():
        s = set(idx)
        d = delta([i for i in allidx if i not in s])
        if d is not None:
            ds.append((d, str(k)))
    ds.sort()
    if not ds:
        return {"full": full, "n_groups": 0}
    return {"full": full, "min": ds[0][0], "min_group": ds[0][1],
            "max": ds[-1][0], "max_group": ds[-1][1], "n_groups": len(ds),
            "sign_flips": sum(1 for d, _ in ds if (d > 0) != (full > 0)),
            "max_share_of_effect": (abs(full - ds[-1][0]) / abs(full)
                                    if full else None),
            "min_share_of_effect": (abs(full - ds[0][0]) / abs(full)
                                    if full else None)}


def _pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


if __name__ == "__main__":
    main()
