#!/usr/bin/env python3
"""Go/no-go probe: does Soniox per-word confidence predict the errors a human
listener actually found?  (`exp-2026-08-16-soniox-confidence`)

SUBSTRATE
  The frozen gold set of `exp-2026-08-16-gold-set` — 27 scored 15 s cores in 6
  meetings / 6 cities, human-verified word by word. Soniox is re-run through the
  SAME realtime path that produced `hyp/soniox/` (model `stt-rt-v4`, free
  Perplexity temp key), now with `file_transcribe.py --json` keeping per-token
  `confidence` / `start_ms` / `end_ms` / `is_final`.

  The 247 benchmark windows are NOT touched. Those were produced with the paid
  `stt-async-v5`, they are one third of the frozen fusion input W, and re-running
  them on v4 would change the text under every frozen number in this project.

WHAT IS FROZEN, WRITTEN DOWN BEFORE ANY OUTCOME WAS COMPUTED
  * PRIMARY STATISTIC — the equal-weight mean of the six WITHIN-MEETING AUROCs of
    `1 - min-subtoken-confidence` for predicting "this emitted Soniox word is an
    error", ties counted 0.5. Chosen over the pooled AUROC on Codex's advice
    (job 4759402624f643efacc4fb2e9cd63beb): pooled AUROC can look good merely
    because one meeting has both lower confidence and higher WER.
  * GO THRESHOLD — mean within-meeting AUROC >= 0.60. Set before looking.
  * Word-confidence aggregate: `min` over subtokens is PRIMARY, `mean` secondary.
  * Region: `core_envelope` (primary in the gold-set prereg), with the same
    uncertainty time-masking. `core_strict` and `clip` reported as sensitivity.
  * Calibration buckets: equal-width 0.0..1.0 step 0.1 AND sample deciles.
  * Alignment: `eval/gold_set_score.align_ops` (tie-break S > D > I) is primary;
    all 6 op priority orders x forward/reversed are a sensitivity envelope.
  * Nulls: AUROC null is 0.5 (not 0). Average-precision null is the emitted-word
    error prevalence. Bottom-decile lift is measured against that prevalence.
  Everything else — overlap split, S-vs-M, I-vs-M, mean aggregate, subtoken-count
  diagnostics — is SECONDARY and is not allowed to become the conclusion.

WHAT THIS CANNOT DO
  Confidence can only flag a word the system EMITTED. Deletions are invisible to
  it by construction, so the deletion share of all edit operations is reported as
  an explicit ceiling. And `stt-rt-v4` is not `stt-async-v5`: nothing here is
  evidence about the paid model, only motivation to test it.

Writes aggregates only. Transcript text never leaves the cache.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.scoring import wtoks                      # noqa: E402
from eval.gold_set_score import (                                   # noqa: E402
    GOLD, REGIONS, align_ops, gold_view, load, masked_intervals,
    region_blocks, region_span, in_intervals,
)

TOKDIR = GOLD / "hyp" / "soniox-tokens"
OLDDIR = GOLD / "hyp" / "soniox"
OUT = ROOT / "eval" / "results_soniox_confidence.json"

PRIMARY_REGION = "core_envelope"
GO_THRESHOLD = 0.60
EQUAL_WIDTH_EDGES = [i / 10 for i in range(11)]
N_PERM = int(os.environ.get("N_PERM", "2000"))


# --------------------------------------------------------------- word building
def group_words(tokens: list[dict]) -> tuple[list[dict], dict]:
    """Soniox emits SUBWORD tokens. Explode them to runes carrying the token's
    confidence, then cut words on whitespace.

    This is the algorithm the user's production Soniox client already uses
    (`soniox-core`, `internal/soniox/soniox.go:36` `LowConfidenceWords`): finals
    only, rune-level carry, whitespace-delimited words, and the word score is the
    MIN confidence over the word's LEXICAL runes only (punctuation excluded, so a
    trailing comma at 0.10 cannot condemn its word). Production flags a word when
    that min is strictly below 0.5 — a threshold chosen as a UX judgement on
    2026-06-14 and NEVER calibrated against ground truth.

    Three aggregates are carried, and the report says which was preregistered:
      conf_min      min over ALL runes (punctuation included) — PREREGISTERED here
      conf_min_lex  min over LEXICAL runes only — the production definition
      conf_mean     mean over all runes — secondary

    `start_ms`/`end_ms` are optional on the wire (`*int`, omitempty in the Go SDK),
    so words with no usable start time are counted and excluded rather than
    silently placed at 0.
    """
    runes: list[tuple[str, float | None, int | None, int | None]] = []
    for t in tokens:
        if not t.get("is_final", False):
            continue                       # finals only, as in production
        txt = t.get("text", "")
        c = t.get("confidence")
        c = float(c) if c is not None else None
        for ch in txt:
            runes.append((ch, c, t.get("start_ms"), t.get("end_ms")))

    words, cur = [], []
    for r in runes:
        if r[0].isspace():
            if cur:
                words.append(cur)
                cur = []
            continue
        cur.append(r)
    if cur:
        words.append(cur)

    out, no_time, no_conf, no_lex = [], 0, 0, 0
    for w in words:
        confs = [c for _ch, c, _s, _e in w if c is not None]
        lex = [c for ch, c, _s, _e in w if c is not None and (ch.isalpha() or ch.isdigit())]
        if not confs:
            no_conf += 1
            continue
        if not lex:
            no_lex += 1               # production never flags a lexeme-free word
        starts = [s for _ch, _c, s, _e in w if s is not None]
        ends = [e for _ch, _c, _s, e in w if e is not None]
        if not starts:
            no_time += 1
            continue
        out.append({"raw": "".join(ch for ch, _c, _s, _e in w),
                    "start": min(starts) / 1000.0,
                    "end": (max(ends) / 1000.0) if ends else (min(starts) / 1000.0),
                    "conf_min": min(confs),
                    "conf_min_lex": min(lex) if lex else min(confs),
                    "conf_mean": sum(confs) / len(confs),
                    "n_sub": len({(s, e) for _ch, _c, s, e in w}),
                    "n_runes": len(w)})
    return out, {"words_without_timestamp": no_time, "words_without_confidence": no_conf,
                 "words_without_lexical_rune": no_lex}


def word_units(words: list[dict]) -> list[dict]:
    """Map each Soniox word onto the scorer's normalized token space.

    `wtoks` drops punctuation and splits on `\\w+`, so a Soniox word can yield 0
    normalized tokens (pure punctuation — dropped) or, rarely, more than one (a
    word-internal split), in which case every resulting token inherits the same
    word-level confidence. Both counts are reported.
    """
    units, dropped, split = [], 0, 0
    for w in words:
        ts = wtoks(w["raw"])
        if not ts:
            dropped += 1
            continue
        if len(ts) > 1:
            split += 1
        for t in ts:
            u = dict(w)
            u["tok"] = t
            units.append(u)
    return units, dropped, split


# ------------------------------------------------------------------- statistics
def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney AUROC with ties worth 0.5. label 1 = positive (error)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rsum = sum(r for r, y in zip(ranks, labels) if y)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def average_precision(scores: list[float], labels: list[int]) -> float | None:
    """AP by the step-interpolation-free sum over recall increments."""
    if not any(labels):
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    npos = sum(labels)
    ap = 0.0
    for i in order:
        if labels[i]:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1
    return ap / npos


def bucket_table(units: list[dict], key: str, edges: list[float]) -> list[dict]:
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        last = b == edges[-1]
        sel = [u for u in units if (a <= u[key] < b) or (last and u[key] == b)]
        rows.append({"lo": a, "hi": b, "n": len(sel),
                     "n_err": sum(u["err"] for u in sel),
                     "err_rate": (sum(u["err"] for u in sel) / len(sel)) if sel else None,
                     "n_meetings": len({u["meeting"] for u in sel})})
    return rows


def quantile_edges(vals: list[float], q: int = 10) -> list[float]:
    v = sorted(vals)
    if not v:
        return []
    e = [v[0]] + [v[min(len(v) - 1, int(round(i * len(v) / q)))] for i in range(1, q)] + [v[-1]]
    out = []
    for x in e:
        if not out or x > out[-1]:
            out.append(x)
    return out


def per_meeting_auroc(units: list[dict], score_key: str) -> dict:
    by = defaultdict(list)
    for u in units:
        by[u["meeting"]].append(u)
    res = {}
    for m, us in sorted(by.items()):
        res[m] = {"auroc": auroc([1.0 - u[score_key] for u in us], [u["err"] for u in us]),
                  "n": len(us), "n_err": sum(u["err"] for u in us)}
    vals = [r["auroc"] for r in res.values() if r["auroc"] is not None]
    mean = sum(vals) / len(vals) if vals else None
    loo = {}
    for m in res:
        rest = [r["auroc"] for k, r in res.items() if k != m and r["auroc"] is not None]
        loo[m] = sum(rest) / len(rest) if rest else None
    return {"per_meeting": res, "mean_within_meeting_auroc": mean,
            "n_meetings_informative": len(vals), "leave_one_meeting_out_mean": loo}


def meeting_cluster_ci(units: list[dict], score_key: str, n_boot: int = 2000,
                       seed: int = 21) -> dict:
    """Resample the 6 MEETINGS and recompute the equal-weight mean of the
    within-meeting AUROCs. Six clusters: descriptive, NOT a significance test and
    NOT an estimate of performance on an unseen meeting."""
    import numpy as np
    by = defaultdict(list)
    for u in units:
        by[u["meeting"]].append(u)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        ms = [a for k in pick
              for a in [auroc([1.0 - u[score_key] for u in by[keys[k]]],
                              [u["err"] for u in by[keys[k]]])]
              if a is not None]
        if ms:
            vals.append(sum(ms) / len(ms))
    v = np.array(vals)
    return {"ci95_meeting_cluster": [float(np.percentile(v, 2.5)),
                                     float(np.percentile(v, 97.5))],
            "n_clusters": len(keys), "n_boot": len(vals)}


def permutation_null(units: list[dict], score_key: str, n: int = N_PERM, seed: int = 21) -> dict:
    """Shuffle confidences WITHIN meeting (Codex: a global shuffle destroys the
    meeting structure the primary statistic is built on)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for u in units:
        by[u["meeting"]].append(u)
    keys = sorted(by)
    obs = per_meeting_auroc(units, score_key)["mean_within_meeting_auroc"]
    if obs is None:
        return {"observed": None}
    vals = []
    for _ in range(n):
        ms = []
        for m in keys:
            us = by[m]
            sc = np.array([1.0 - u[score_key] for u in us])
            lb = [u["err"] for u in us]
            rng.shuffle(sc)
            a = auroc(list(sc), lb)
            if a is not None:
                ms.append(a)
        if ms:
            vals.append(sum(ms) / len(ms))
    vals = np.array(vals)
    return {"observed": obs, "null_mean": float(vals.mean()),
            "null_p2.5": float(np.percentile(vals, 2.5)),
            "null_p97.5": float(np.percentile(vals, 97.5)),
            "p_value_one_sided": float((vals >= obs).sum() + 1) / (len(vals) + 1),
            "n_perm": len(vals)}


# ------------------------------------------------------------------ reproduction
def reproduction(scored: list[str]) -> dict:
    from eval.controlled_eval.scoring import edist
    rows = []
    for cid in sorted(set(scored)):
        newp, oldp = TOKDIR / f"{cid}.json", OLDDIR / f"{cid}.json"
        if not newp.exists() or not oldp.exists():
            rows.append({"cell": cid, "status": "missing"})
            continue
        new = json.loads(newp.read_text())
        old = json.loads(oldp.read_text())
        nt, ot = wtoks(new.get("text", "")), wtoks(old.get("text", ""))
        rows.append({"cell": cid, "status": "ok",
                     "raw_exact": new.get("text", "") == old.get("text", ""),
                     "norm_exact": nt == ot,
                     "n_old": len(ot), "n_new": len(nt),
                     "wer_old_vs_new": (edist(ot, nt) / len(ot)) if ot else None})
    ok = [r for r in rows if r["status"] == "ok"]
    tot_ref = sum(r["n_old"] for r in ok)
    tot_err = sum((r["wer_old_vs_new"] or 0) * r["n_old"] for r in ok)
    return {"n_cells": len(ok),
            "raw_exact_rate": sum(r["raw_exact"] for r in ok) / len(ok) if ok else None,
            "norm_exact_rate": sum(r["norm_exact"] for r in ok) / len(ok) if ok else None,
            "pooled_wer_old_vs_new": (tot_err / tot_ref) if tot_ref else None,
            "cells": rows}


# ------------------------------------------------------------------------- build
def build_units(scored, ans, sel, region: str, op_priority=("S", "D", "I"),
                reverse=False) -> tuple[list[dict], dict]:
    """Label every emitted Soniox word inside the scored intervals M / S / I."""
    units, ceiling = [], Counter()
    drop_punct = split_words = 0
    for cid in scored:
        row = sel[cid]
        mid = row["meeting_id"]
        p = TOKDIR / f"{cid}.json"
        if not p.exists():
            continue
        seq, _by, _ov, blocks, _kept = gold_view(ans[cid], region)
        span = region_span(blocks, region)
        if span is None or not seq:
            continue
        outside = [b for b in ans[cid]["b"]["blocks"]
                   if b["id"] not in {x["id"] for x in blocks}]
        ivs = masked_intervals(blocks, span, outside)
        ov_blocks = [b for b in blocks if b.get("ov_with") and not b.get("text_unc")]

        words, wstats = group_words(json.loads(p.read_text())["tokens"])
        for k, v in wstats.items():
            ceiling[k] += v
        words = [w for w in words if in_intervals(w["start"], ivs)]
        us, dp, sw = word_units(words)
        drop_punct += dp
        split_words += sw

        hyp = [u["tok"] for u in us]
        ops = _align(seq, hyp, op_priority, reverse)
        lab = {}
        for op, _i, j in ops:
            if op in ("M", "S", "I"):
                lab[j] = op
            ceiling[op] += 1
        for j, u in enumerate(us):
            op = lab.get(j)
            if op is None:
                continue
            u = dict(u)
            u["op"] = op
            u["err"] = int(op != "M")
            u["cell"] = cid
            u["meeting"] = mid
            u["overlap"] = any(b["s"] <= u["start"] < b["e"] for b in ov_blocks)
            units.append(u)
    ceiling = dict(ceiling)
    tot_ops = sum(ceiling.get(k, 0) for k in ("S", "D", "I"))
    ceiling["deletion_share_of_edits"] = (ceiling.get("D", 0) / tot_ops) if tot_ops else None
    ceiling["edits_visible_to_confidence"] = ((ceiling.get("S", 0) + ceiling.get("I", 0))
                                             / tot_ops) if tot_ops else None
    ceiling["punctuation_only_words_dropped"] = drop_punct
    ceiling["words_splitting_into_multiple_tokens"] = split_words
    return units, ceiling


def _align(ref, hyp, priority, reverse):
    if priority == ("S", "D", "I") and not reverse:
        return align_ops(ref, hyp)
    r, h = (ref[::-1], hyp[::-1]) if reverse else (ref, hyp)
    ops = _align_pri(r, h, priority)
    if not reverse:
        return ops
    n, m = len(ref), len(hyp)
    return [(o, (n - 1 - i) if i >= 0 else -1, (m - 1 - j) if j >= 0 else -1)
            for o, i, j in reversed(ops)]


def _align_pri(ref, hyp, priority):
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0], op[i][0] = i, "D"
    for j in range(1, m + 1):
        d[0][j], op[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j], op[i][j] = d[i - 1][j - 1], "M"
                continue
            cost = {"S": d[i - 1][j - 1] + 1, "D": d[i - 1][j] + 1, "I": d[i][j - 1] + 1}
            best = min(cost.values())
            d[i][j] = best
            op[i][j] = next(k for k in priority if cost[k] == best)
    out, i, j = [], n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o in ("M", "S"):
            out.append((o, i - 1, j - 1)); i, j = i - 1, j - 1
        elif o == "D":
            out.append(("D", i - 1, -1)); i -= 1
        else:
            out.append(("I", -1, j - 1)); j -= 1
    out.reverse()
    return out


# -------------------------------------------------------------------------- main
def subset_metrics(units: list[dict], key: str) -> dict:
    s = [1.0 - u[key] for u in units]
    y = [u["err"] for u in units]
    prev = (sum(y) / len(y)) if y else None
    m = {"n": len(units), "n_err": sum(y), "prevalence": prev,
         "pooled_auroc": auroc(s, y), "average_precision": average_precision(s, y),
         "ap_null_prevalence": prev}
    m.update(per_meeting_auroc(units, key))
    return m


def main():
    _man, ans, cells, sel = load()
    man = _man
    scored = man["scored_cell_ids"]
    res = {
        "protocol": "soniox-confidence-probe-2026-08-16",
        "model": "stt-rt-v4 (free realtime path) — NOT the paid stt-async-v5",
        "answers_sha256": man["answers_sha256"],
        "frozen_before_looking": {
            "primary_statistic": "mean of within-meeting AUROC, 1 - conf_min, ties 0.5",
            "go_threshold": GO_THRESHOLD,
            "primary_region": PRIMARY_REGION,
            "auroc_null": 0.5,
            "equal_width_edges": EQUAL_WIDTH_EDGES,
            "inherited_operating_point": "conf_min_lex < 0.5 (soniox-core production, "
                                         "a UX judgement never calibrated until now)",
            "reproduction_ceiling": "soniox-core measured run-to-run word similarity at "
                                    "93.2% on identical input; 100% is NOT the ceiling",
            "silence_trimming": "NOT enabled (soniox-core ?trim=1 loses ~11% of words)",
        },
        "n_cells_scored": len(scored),
        "n_meetings": len({sel[c]["meeting_id"] for c in scored}),
    }
    missing = [c for c in scored if not (TOKDIR / f"{c}.json").exists()]
    assert not missing, f"no token JSON for {len(missing)} scored cells: {missing}"
    res["reproduction"] = reproduction(scored)

    res["regions"] = {}
    for region in REGIONS:
        units, ceiling = build_units(scored, ans, sel, region)
        r = {"edit_op_ceiling": ceiling,
             "conf_min": subset_metrics(units, "conf_min"),
             "conf_min_lex": subset_metrics(units, "conf_min_lex"),
             "conf_mean": subset_metrics(units, "conf_mean")}
        if units:
            # the inherited production operating point: flag when the min over
            # LEXICAL runes is strictly below 0.5 (soniox-core, unchanged since
            # 2026-06-14, never calibrated against ground truth until now).
            for key in ("conf_min", "conf_min_lex"):
                flagged = [u for u in units if u[key] < 0.5]
                tp = sum(u["err"] for u in flagged)
                nerr = sum(u["err"] for u in units)
                r[f"operating_point_0.5_{key}"] = {
                    "n_flagged": len(flagged), "flag_rate": len(flagged) / len(units),
                    "precision": (tp / len(flagged)) if flagged else None,
                    "recall": (tp / nerr) if nerr else None,
                    "prevalence": nerr / len(units),
                    "lift": ((tp / len(flagged)) / (nerr / len(units)))
                            if flagged and nerr else None}
        if region == PRIMARY_REGION and units:
            r["calibration_equal_width_conf_min"] = bucket_table(units, "conf_min",
                                                                 EQUAL_WIDTH_EDGES)
            r["calibration_deciles_conf_min"] = bucket_table(
                units, "conf_min", quantile_edges([u["conf_min"] for u in units]))
            for name, grp in (("overlap", [u for u in units if u["overlap"]]),
                              ("non_overlap", [u for u in units if not u["overlap"]])):
                r[f"split_{name}"] = subset_metrics(grp, "conf_min") if grp else None
            # insertion / substitution specificity — SECONDARY
            ins = [u for u in units if u["op"] in ("I", "M")]
            subs = [u for u in units if u["op"] in ("S", "M")]
            for u in ins:
                u["err"] = int(u["op"] == "I")
            r["insertions_vs_match"] = subset_metrics(ins, "conf_min")
            for u in ins:
                u["err"] = int(u["op"] != "M")
            for u in subs:
                u["err"] = int(u["op"] == "S")
            r["substitutions_vs_match"] = subset_metrics(subs, "conf_min")
            for u in subs:
                u["err"] = int(u["op"] != "M")
            # bottom-decile lift, operational companion
            ordered = sorted(units, key=lambda u: u["conf_min"])
            k = max(1, len(ordered) // 10)
            bot = ordered[:k]
            prev = sum(u["err"] for u in units) / len(units)
            br = sum(u["err"] for u in bot) / len(bot)
            r["bottom_decile"] = {"n": len(bot), "err_rate": br, "prevalence": prev,
                                  "lift": (br / prev) if prev else None}
            # length confound diagnostics — SECONDARY (Codex b)
            r["subtoken_diagnostics"] = {
                "auroc_of_subtoken_count_alone": auroc([float(u["n_sub"]) for u in units],
                                                       [u["err"] for u in units]),
                "by_n_sub": {str(n): {"n": len(g), "err_rate": sum(u["err"] for u in g) / len(g),
                                      "auroc": auroc([1 - u["conf_min"] for u in g],
                                                     [u["err"] for u in g])}
                             for n in sorted({u["n_sub"] for u in units})
                             for g in [[u for u in units if u["n_sub"] == n]]},
            }
            r["permutation_null_conf_min"] = permutation_null(units, "conf_min")
            r["meeting_cluster_ci_conf_min"] = meeting_cluster_ci(units, "conf_min")
            r["calibration_equal_width_conf_min_lex"] = bucket_table(
                units, "conf_min_lex", EQUAL_WIDTH_EDGES)
            # alignment tie-break sensitivity envelope (Codex c)
            env = []
            for pri in permutations(("S", "D", "I")):
                for rev in (False, True):
                    us2, _c = build_units(scored, ans, sel, region, pri, rev)
                    if not us2:
                        continue
                    pm = per_meeting_auroc(us2, "conf_min")
                    env.append({"priority": "".join(pri), "reversed": rev,
                                "mean_within_meeting_auroc": pm["mean_within_meeting_auroc"],
                                "pooled_auroc": auroc([1 - u["conf_min"] for u in us2],
                                                      [u["err"] for u in us2]),
                                "n_S": sum(1 for u in us2 if u["op"] == "S"),
                                "n_I": sum(1 for u in us2 if u["op"] == "I"),
                                "n": len(us2)})
            r["alignment_sensitivity"] = env
        res["regions"][region] = r

    p = res["regions"][PRIMARY_REGION]["conf_min"]["mean_within_meeting_auroc"]
    res["decision"] = {
        "primary_value": p, "threshold": GO_THRESHOLD,
        "go": bool(p is not None and p >= GO_THRESHOLD),
        "note": "stt-rt-v4 is not stt-async-v5. A GO is a hypothesis to test, "
                "not evidence that the paid model's confidence behaves the same.",
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")
    print(json.dumps(res["reproduction"] | {"cells": "..."}, ensure_ascii=False, indent=1))
    print(json.dumps(res["decision"], ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main()
