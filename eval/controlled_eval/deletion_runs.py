#!/usr/bin/env python3
"""Why whole passages disappear: a census of consecutive-deletion runs.

Open question 3 of the final report. The gap-to-scribe screen
(`exp-2026-08-23-gap-to-scribe`, docs/reports/2026-08-23-gap-to-scribe.md) found that
36% of what our adapter deletes vanishes in runs of five or more consecutive
reference words, against 19% for Scribe, and did not find the cause.

This script re-derives that number from the same public substrate, then characterises
the runs and tests the explanations that can be tested without a new decode.

Substrate: the PUBLIC report.json of benchmark run
`2026-08-22-post-june-held-out-test-clean-pack-cont-` (391 held-out post-June windows,
117 meetings, 8 providers). Reading it is free; no GPU, no ASR API call.

Everything here is agreement-with-OpenCouncil: the reference is OpenCouncil's own
published transcript, so a "deletion" is a word in the published text that a system
did not write. It records product compatibility, not fidelity to audio.

Writes eval/results_deletion_runs.json (aggregates and word-level run text only for
counts - no transcript text is emitted).

Env: REPORT (path to report.json). Default is the scratchpad copy.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.scoring import wtoks  # noqa: E402

REPORT = Path(os.environ.get("REPORT", ""))
OUT = ROOT / "eval" / "results_deletion_runs.json"

OURS = "oc-cleanpack-cont-s47-b"
INCUMBENT = "oc-adapter-fixed-restage-2026-08-22"
BASE = "hf-openai-whisper-large-v3"
SCRIBE = "scribe"
SONIOX = "soniox"
GPT4O = "gpt-4o-transcribe"
GLADIA = "gladia-prod"
SYSTEMS = [OURS, INCUMBENT, BASE, SCRIBE, SONIOX, GPT4O, GLADIA]

LONG = 5  # a "run" is long at this many consecutive deleted reference tokens


def trace(ref, hyp, prefer="sub"):
    """Per-reference-token outcome from one Levenshtein backtrace.

    ('=', w) kept, ('S', w, got) substituted, ('D', w) deleted. Returns (outcomes, ins).

    `prefer` chooses the tie-break order when several optimal alignments exist:
      "sub" - take the diagonal (substitution) first, as gap.py did
      "del" - take the deletion edge first
    The two bracket how much the run census depends on an arbitrary choice.
    """
    n, mm = len(ref), len(hyp)
    D = [[0] * (mm + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(mm + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        r = ref[i - 1]
        Di, Dp = D[i], D[i - 1]
        for j in range(1, mm + 1):
            Di[j] = Dp[j - 1] if r == hyp[j - 1] else 1 + min(Dp[j - 1], Dp[j], Di[j - 1])
    i, j = n, mm
    out = [None] * n
    ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and D[i][j] == D[i - 1][j - 1]:
            out[i - 1] = ("=", ref[i - 1])
            i, j = i - 1, j - 1
            continue
        del_ok = i > 0 and D[i][j] == D[i - 1][j] + 1
        sub_ok = i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1
        order = (("D", del_ok), ("S", sub_ok)) if prefer == "del" else (("S", sub_ok), ("D", del_ok))
        for kind, ok in order:
            if not ok:
                continue
            if kind == "S":
                out[i - 1] = ("S", ref[i - 1], hyp[j - 1])
                i, j = i - 1, j - 1
            else:
                out[i - 1] = ("D", ref[i - 1])
                i -= 1
            break
        else:
            ins += 1
            j -= 1
    return out, ins


def runs_of(outcomes):
    """[(start_index, length)] for every maximal block of consecutive 'D' outcomes."""
    out = []
    start = None
    for k, a in enumerate(outcomes):
        if a[0] == "D":
            if start is None:
                start = k
        elif start is not None:
            out.append((start, k - start))
            start = None
    if start is not None:
        out.append((start, len(outcomes) - start))
    return out


def census(runs):
    """runs: list of lengths. Returns the table the report quotes."""
    tot = sum(runs)
    long_tok = sum(r for r in runs if r >= LONG)
    return {
        "runs": len(runs),
        "tokens_deleted": tot,
        "runs_1": sum(1 for r in runs if r == 1),
        "runs_2_4": sum(1 for r in runs if 2 <= r <= 4),
        "runs_5plus": sum(1 for r in runs if r >= LONG),
        "tokens_in_5plus": long_tok,
        "share_in_5plus": long_tok / tot if tot else 0.0,
        "longest": max(runs) if runs else 0,
    }


NUMERIC = re.compile(r"\d")


def main():
    d = json.load(open(REPORT))
    items = d["items"]

    res = {
        "substrate": {
            "report_run": d["manifest"]["id"] if isinstance(d.get("manifest"), dict) and "id" in d.get("manifest", {}) else None,
            "generatedAt": d.get("generatedAt"),
            "items": len(items),
            "systems": SYSTEMS,
            "long_run_threshold": LONG,
            "note": "agreement-with-OpenCouncil reference; deletions are published words a system did not write",
        }
    }

    # ---- pass 1: trace every system on every window, both tie-breaks -------------
    tr = {}          # (itemId, system, prefer) -> outcomes
    ref_toks = {}
    for it in items:
        ref = wtoks(it["referenceText"])
        ref_toks[it["itemId"]] = ref
        for s in SYSTEMS:
            p = it["perProvider"].get(s)
            hyp = wtoks(p["hypothesisText"]) if p and p.get("status") == "ok" else None
            if hyp is None:
                continue
            for pref in ("sub", "del"):
                tr[(it["itemId"], s, pref)] = trace(ref, hyp, pref)[0]

    # ---- 1. the run census, verified, all systems, both tie-breaks ---------------
    tbl = {}
    for s in SYSTEMS:
        tbl[s] = {}
        for pref in ("sub", "del"):
            lens = []
            for it in items:
                o = tr.get((it["itemId"], s, pref))
                if o is None:
                    continue
                lens += [L for _, L in runs_of(o)]
            tbl[s][pref] = census(lens)
    res["run_census"] = tbl

    # ---- 2. domination: who carries the tokens lost in long runs ----------------
    def long_runs_for(s, pref="sub"):
        """[(itemId, cityId, meetingId, start, length, ref_len)] for runs >= LONG."""
        out = []
        for it in items:
            o = tr.get((it["itemId"], s, pref))
            if o is None:
                continue
            for st, L in runs_of(o):
                if L >= LONG:
                    out.append((it["itemId"], it["cityId"], it["meetingId"], st, L, len(o)))
        return out

    lr_ours = long_runs_for(OURS)
    lr_scribe = long_runs_for(SCRIBE)

    def dom(lr, key, n=8):
        c = collections.Counter()
        for r in lr:
            c[{"city": r[1], "meeting": f"{r[1]}/{r[2]}", "window": r[0]}[key]] += r[4]
        tot = sum(c.values())
        return {
            "total_tokens_in_long_runs": tot,
            "top": [{"key": k, "tokens": v, "share": v / tot} for k, v in c.most_common(n)],
            "top5_share": sum(v for _, v in c.most_common(5)) / tot if tot else 0.0,
            "distinct": len(c),
        }

    res["domination"] = {
        "ours": {k: dom(lr_ours, k) for k in ("city", "meeting", "window")},
        "scribe": {k: dom(lr_scribe, k) for k in ("city", "meeting", "window")},
    }

    # ---- 3. where in the window does a long run sit? ----------------------------
    # H: window edges / truncation. A run that reaches the last reference token is a
    # hypothesis that stopped early; one that starts at token 0 is a late start.
    def edge_profile(lr):
        dec = collections.Counter()
        touch_end = touch_start = 0
        tok_end = tok_start = tok_mid = 0
        for _, _, _, st, L, n in lr:
            dec[min(9, int(10 * st / max(1, n)))] += 1
            if st + L >= n:
                touch_end += 1
                tok_end += L
            elif st == 0:
                touch_start += 1
                tok_start += L
            else:
                tok_mid += L
        return {
            "runs": len(lr),
            "start_decile_counts": [dec[i] for i in range(10)],
            "runs_touching_window_end": touch_end,
            "runs_touching_window_start": touch_start,
            "tokens_in_end_runs": tok_end,
            "tokens_in_start_runs": tok_start,
            "tokens_in_interior_runs": tok_mid,
        }

    res["edges"] = {"ours": edge_profile(lr_ours), "scribe": edge_profile(lr_scribe)}

    # ---- 4. truncation / short output -------------------------------------------
    # If long runs are the decoder stopping early, windows carrying them should have a
    # markedly shorter hypothesis than reference.
    def length_split(s):
        with_long, without = [], []
        for it in items:
            o = tr.get((it["itemId"], s, "sub"))
            if o is None:
                continue
            p = it["perProvider"][s]
            ratio = len(wtoks(p["hypothesisText"])) / max(1, len(ref_toks[it["itemId"]]))
            (with_long if any(L >= LONG for _, L in runs_of(o)) else without).append(ratio)
        med = lambda v: sorted(v)[len(v) // 2] if v else None
        return {
            "windows_with_long_run": len(with_long),
            "windows_without": len(without),
            "median_hyp_over_ref_with_long_run": med(with_long),
            "median_hyp_over_ref_without": med(without),
            "mean_with": sum(with_long) / len(with_long) if with_long else None,
            "mean_without": sum(without) / len(without) if without else None,
        }

    res["length_ratio"] = {s: length_split(s) for s in (OURS, SCRIBE, BASE, INCUMBENT)}

    # ---- 5. is the reference at fault? -------------------------------------------
    # For every long run we delete, ask what the OTHER systems did with the same
    # reference tokens. If nobody wrote them, the published transcript is the more
    # likely explanation than seven independent decoders.
    def cross(lr, owner):
        others = [s for s in SYSTEMS if s != owner]
        buckets = collections.Counter()
        per_other_kept = collections.Counter()
        tok_by_bucket = collections.Counter()
        for iid, _, _, st, L, _ in lr:
            kept_by = 0
            for s in others:
                o = tr.get((iid, s, "sub"))
                if o is None:
                    continue
                span = o[st:st + L]
                k = sum(1 for a in span if a[0] == "=")
                if k >= 0.5 * L:
                    kept_by += 1
                    per_other_kept[s] += 1
            b = ("none" if kept_by == 0 else "one" if kept_by == 1 else
                 "few_2_3" if kept_by <= 3 else "most_4plus")
            buckets[b] += 1
            tok_by_bucket[b] += L
        return {
            "runs_by_how_many_other_systems_kept_half_the_span": dict(buckets),
            "tokens_by_bucket": dict(tok_by_bucket),
            "kept_by_system": dict(per_other_kept),
        }

    res["cross_system"] = {"ours": cross(lr_ours, OURS), "scribe": cross(lr_scribe, SCRIBE)}

    # ---- 6. do our long runs overlap the incumbent's and base whisper's? ---------
    # Inherited from whisper, or made by fine-tuning?
    def overlap_with(lr, other):
        both = 0
        for iid, _, _, st, L, _ in lr:
            o = tr.get((iid, other, "sub"))
            if o is None:
                continue
            if sum(1 for a in o[st:st + L] if a[0] == "D") >= 0.5 * L:
                both += 1
        return {"runs": len(lr), "also_mostly_deleted_by_" + other: both,
                "share": both / len(lr) if lr else 0.0}

    res["inheritance"] = {
        "ours_long_runs_vs_base": overlap_with(lr_ours, BASE),
        "ours_long_runs_vs_incumbent": overlap_with(lr_ours, INCUMBENT),
        "base_long_runs_vs_ours": overlap_with(long_runs_for(BASE), OURS),
    }

    # ---- 7. what is in the deleted text? ----------------------------------------
    def content(lr, owner):
        n_tok = 0
        digits = caps = short_words = 0
        wlen = 0
        rep_ahead = 0   # run text repeats text elsewhere in the reference
        for iid, _, _, st, L, _ in lr:
            ref = ref_toks[iid]
            span = ref[st:st + L]
            n_tok += L
            wlen += sum(len(w) for w in span)
            digits += sum(1 for w in span if NUMERIC.search(w))
            short_words += sum(1 for w in span if len(w) <= 3)
            key = " ".join(span[:3])
            rest = " ".join(ref[:st] + ref[st + L:])
            if len(span) >= 3 and key in rest:
                rep_ahead += 1
        # baseline over all reference tokens
        allw = [w for iid in ref_toks for w in ref_toks[iid]]
        return {
            "tokens": n_tok,
            "mean_word_len": wlen / n_tok if n_tok else None,
            "mean_word_len_corpus": sum(len(w) for w in allw) / len(allw),
            "share_with_digit": digits / n_tok if n_tok else None,
            "share_with_digit_corpus": sum(1 for w in allw if NUMERIC.search(w)) / len(allw),
            "share_short_words": short_words / n_tok if n_tok else None,
            "share_short_words_corpus": sum(1 for w in allw if len(w) <= 3) / len(allw),
            "runs_whose_first_3_words_repeat_elsewhere_in_window": rep_ahead,
        }

    res["content"] = {"ours": content(lr_ours, OURS), "scribe": content(lr_scribe, SCRIBE)}

    # ---- 8. proportional time placement, chunk-boundary proxy -------------------
    # Weak by construction: reference tokens have no timestamps, so position is mapped
    # to time by assuming a constant speech rate across the window. Reported as a
    # proxy, not as evidence about chunking.
    chunk = collections.Counter()
    chunk_all = collections.Counter()
    for iid, _, _, st, L, n in lr_ours:
        it = next(x for x in items if x["itemId"] == iid)
        t = it["startSec"] + it["durationSec"] * (st / max(1, n))
        chunk[int(t % 30) // 3] += 1
    for it in items:
        n = len(ref_toks[it["itemId"]])
        for k in range(n):
            t = it["startSec"] + it["durationSec"] * (k / max(1, n))
            chunk_all[int(t % 30) // 3] += 1
    res["chunk_phase_proxy"] = {
        "long_run_starts_by_3s_bin_of_30s_phase": [chunk[i] for i in range(10)],
        "all_reference_tokens_by_bin": [chunk_all[i] for i in range(10)],
        "caveat": "position->time is a constant-rate assumption; no word timestamps in this substrate",
    }

    # ---- 9. bootstrap the 36% over meetings --------------------------------------
    def boot_share(s, n_boot=10000, seed=20260824):
        pm = collections.defaultdict(lambda: [0, 0])
        for it in items:
            o = tr.get((it["itemId"], s, "sub"))
            if o is None:
                continue
            k = f"{it['cityId']}/{it['meetingId']}"
            for _, L in runs_of(o):
                pm[k][1] += L
                if L >= LONG:
                    pm[k][0] += L
        keys = sorted(pm)
        rng = random.Random(seed)
        pt = sum(pm[k][0] for k in keys) / max(1, sum(pm[k][1] for k in keys))
        b = []
        for _ in range(n_boot):
            samp = [rng.choice(keys) for _ in keys]
            num = sum(pm[k][0] for k in samp)
            den = sum(pm[k][1] for k in samp)
            b.append(num / den if den else 0.0)
        b.sort()
        return {"point": pt, "ci95": [b[int(0.025 * n_boot)], b[int(0.975 * n_boot)]],
                "meetings": len(keys)}

    res["bootstrap_share_in_5plus"] = {"ours": boot_share(OURS), "scribe": boot_share(SCRIBE)}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
