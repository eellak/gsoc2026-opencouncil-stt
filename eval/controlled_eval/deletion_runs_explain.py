#!/usr/bin/env python3
"""Test the candidate explanations for the long deletion runs against the runs.

Stage three of the open-question-3 screen. `deletion_runs.py` counts the runs,
`deletion_run_context.py` puts them on the meeting timeline; this file asks each
hypothesis a question with a matched baseline attached, so that "enriched" means
enriched against something.

The matched baseline throughout is: the SAME run lengths, in the SAME windows, at
random start positions, drawn once with a fixed seed. It controls for the fact that
long spans of any kind touch more utterances and look faster than short ones.

Free: public benchmark report + public meeting API. No GPU, no ASR call.

    REPORT=<report.json> python3 -m eval.controlled_eval.deletion_runs_explain
"""
from __future__ import annotations

import collections
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.deletion_run_context import CHUNK, load_utterances, window_tokens  # noqa: E402
from eval.controlled_eval.deletion_runs import (  # noqa: E402
    BASE, INCUMBENT, LONG, OURS, SCRIBE, SYSTEMS, runs_of, trace,
)
from eval.controlled_eval.scoring import wtoks  # noqa: E402

REPORT = Path(os.environ.get("REPORT", ""))
OUT = ROOT / "eval" / "results_deletion_runs_explain.json"
SEED = 20260824
N_BOOT = 4000


def tok_time(toks, sel, k):
    _, ui, ki, n = toks[k]
    s, e = sel[ui][0], sel[ui][1]
    return s + (e - s) * ((ki + 0.5) / max(1, n))


def main():
    d = json.load(open(REPORT))
    items = d["items"]

    ctx = {}
    for it in items:
        u = load_utterances(it["cityId"], it["meetingId"])
        if u is None:
            continue
        t0, t1 = it["startSec"], it["startSec"] + it["durationSec"]
        toks, sel = window_tokens(u, t0, t1)
        if [t[0] for t in toks] == wtoks(it["referenceText"]):
            ctx[it["itemId"]] = (toks, sel, t0, t1)

    tr = {}
    for it in items:
        ref = wtoks(it["referenceText"])
        for s in SYSTEMS:
            p = it["perProvider"].get(s)
            if p and p.get("status") == "ok":
                tr[(it["itemId"], s)] = trace(ref, wtoks(p["hypothesisText"]))[0]

    def long_runs(system, timed_only=True):
        out = []
        for it in items:
            if timed_only and it["itemId"] not in ctx:
                continue
            o = tr.get((it["itemId"], system))
            if o is None:
                continue
            for st, L in runs_of(o):
                if L >= LONG:
                    out.append((it, st, L))
        return out

    rng = random.Random(SEED)

    res = {
        "method": {
            "baseline": "same run lengths, same windows, uniformly random start, seed %d" % SEED,
            "windows_with_timeline": len(ctx),
            "windows_total": len(items),
            "long_run_threshold": LONG,
            "note": "agreement-with-OpenCouncil reference throughout",
        }
    }

    # ---- H1. speech rate: is the dropped audio faster than its neighbourhood? ----
    def rate(toks, sel, st, L):
        a, b = tok_time(toks, sel, st), tok_time(toks, sel, st + L - 1)
        return L / (b - a) if b - a > 0.3 else None

    h1 = {}
    for name, s in (("ours", OURS), ("base", BASE), ("incumbent", INCUMBENT), ("scribe", SCRIBE)):
        obs, bl, win = [], [], []
        for it, st, L in long_runs(s):
            toks, sel, t0, t1 = ctx[it["itemId"]]
            r = rate(toks, sel, st, L)
            if r:
                obs.append(r)
                win.append(len(toks) / it["durationSec"])
            n = len(toks)
            if n > L:
                r2 = rate(toks, sel, rng.randrange(0, n - L), L)
                if r2:
                    bl.append(r2)
        h1[name] = {
            "runs": len(obs),
            "median_tok_per_s_in_run": statistics.median(obs) if obs else None,
            "median_tok_per_s_matched_random_span": statistics.median(bl) if bl else None,
            "median_tok_per_s_whole_window": statistics.median(win) if win else None,
        }
    res["H1_speech_rate"] = h1
    res["H1_note"] = ("rate is measured centre-of-first-word to centre-of-last-word, which "
                      "inflates short spans; only the matched-random-span column is a fair "
                      "comparator, and the whole-window column is not")

    # ---- H2. the reference itself: human-edited text, and what others wrote ------
    def edited_share(toks, sel, st, L):
        return sum(1 for t in toks[st:st + L] if sel[t[1]][4]) / L

    corpus_tok = corpus_ed = 0
    for iid, (toks, sel, _, _) in ctx.items():
        for t in toks:
            corpus_tok += 1
            corpus_ed += 1 if sel[t[1]][4] else 0

    h2 = {"corpus_share_tokens_in_human_edited_utterances": corpus_ed / corpus_tok}
    for name, s in (("ours", OURS), ("base", BASE), ("incumbent", INCUMBENT), ("scribe", SCRIBE)):
        n = e = 0
        bn = be = 0
        for it, st, L in long_runs(s):
            toks, sel, _, _ = ctx[it["itemId"]]
            n += L
            e += edited_share(toks, sel, st, L) * L
            if len(toks) > L:
                st2 = rng.randrange(0, len(toks) - L)
                bn += L
                be += edited_share(toks, sel, st2, L) * L
        h2[name] = {
            "long_run_tokens": n,
            "share_in_human_edited_utterances": e / n if n else None,
            "matched_random_baseline": be / bn if bn else None,
        }
    res["H2_reference_is_human_edited"] = h2

    # ---- H3. joint bucket: reference-side vs genuinely dropped speech -----------
    buck = collections.Counter()
    tokb = collections.Counter()
    for it, st, L in long_runs(OURS):
        toks, sel, _, _ = ctx[it["itemId"]]
        kept = 0
        for s in SYSTEMS:
            if s == OURS:
                continue
            oo = tr.get((it["itemId"], s))
            if oo and sum(1 for a in oo[st:st + L] if a[0] == "=") >= 0.5 * L:
                kept += 1
        who = "others_wrote_it" if kept >= 4 else "nobody_wrote_it" if kept == 0 else "mixed"
        ref = "human_edited" if edited_share(toks, sel, st, L) >= 0.5 else "asr_reference"
        buck[who + "|" + ref] += 1
        tokb[who + "|" + ref] += L
    res["H3_reference_vs_real_speech"] = {
        "runs": dict(buck), "tokens": dict(tokb), "total_tokens": sum(tokb.values()),
        "legend": "others_wrote_it = >=4 of the other 6 systems kept >=half the span; "
                  "human_edited = >=half the span sits in utterances a human edited",
    }

    # ---- H4. chunk phase: last 5 s of a 30 s encoder frame ----------------------
    h4 = {}
    for name, s in (("ours", OURS), ("base", BASE), ("incumbent", INCUMBENT), ("scribe", SCRIBE)):
        per_meeting = collections.defaultdict(lambda: [0, 0])
        exp = collections.Counter()
        E = 0
        by_chunk = collections.Counter()
        for it, st, L in long_runs(s):
            toks, sel, t0, _ = ctx[it["itemId"]]
            t = tok_time(toks, sel, st) - t0
            k = it["cityId"] + "/" + it["meetingId"]
            per_meeting[k][1] += 1
            if int((t % CHUNK) // 5) == 5:
                per_meeting[k][0] += 1
            by_chunk[int(t // CHUNK)] += 1
        for iid, (toks, sel, t0, _) in ctx.items():
            for k2 in range(len(toks)):
                E += 1
                exp[int(((tok_time(toks, sel, k2) - t0) % CHUNK) // 5)] += 1
        keys = sorted(per_meeting)
        num = sum(per_meeting[k][0] for k in keys)
        den = sum(per_meeting[k][1] for k in keys)
        r2 = random.Random(SEED)
        b = []
        for _ in range(N_BOOT):
            samp = [r2.choice(keys) for _ in keys] if keys else []
            a = sum(per_meeting[k][0] for k in samp)
            c = sum(per_meeting[k][1] for k in samp)
            b.append(a / c if c else 0.0)
        b.sort()
        h4[name] = {
            "runs": den,
            "share_starting_in_last_5s_of_chunk": num / den if den else None,
            "ci95": [b[int(0.025 * N_BOOT)], b[int(0.975 * N_BOOT)]] if b else None,
            "expected_from_token_mass": exp[5] / E if E else None,
            "runs_by_chunk_index": [by_chunk[i] for i in range(6)],
        }
    res["H4_chunk_phase"] = h4
    res["H4_caveat"] = ("the 30 s grid is anchored to the start of the decoded window. "
                        "faster-whisper advances its seek to the end of the last emitted "
                        "segment, so only the first frame is guaranteed to sit on that grid; "
                        "this is post-hoc, one of many comparisons run, and is not preregistered")

    # ---- H5. training-granularity: is this new, or inherited from whisper? ------
    def share5(system):
        lens = []
        for it in items:
            o = tr.get((it["itemId"], system))
            if o is None:
                continue
            lens += [L for _, L in runs_of(o)]
        t = sum(lens)
        return {"tokens_deleted": t,
                "tokens_in_5plus": sum(L for L in lens if L >= LONG),
                "share": sum(L for L in lens if L >= LONG) / t if t else None}
    res["H5_inherited_from_base"] = {s: share5(s) for s in SYSTEMS}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
