#!/usr/bin/env python3
"""Fusion on the 391-window held-out post-June set, with the current best adapter.

`exp-2026-08-16-composition-over-selection` established that per-column composition
beats whole-window selection, on a 247-window run whose adapter row was the July
model. This reruns the same arms on windows none of the adapters ever trained on,
with the clean-pack adapter measured on 2026-08-23.

Arms, all of them mechanical - no LLM anywhere in this script:

  single      each system on its own, rescored here so every number on the page
              comes off one scorer
  V           whole-window consensus: keep the hypothesis the other two agree with
              most. A deployable selector.
  W           exact 3-way MSA + hierarchical per-column vote. The output is a text
              none of the three systems produced.
  oracle_win  per window, the lowest-WER hypothesis. The ceiling for any method that
              picks whole windows.
  oracle_msa  per column of the exact 3-way MSA, the best entry. Codex's review of
              this plan is right that this is NOT the unrestricted information
              ceiling: it is the best path through ONE fixed alignment lattice, and
              the MSA that produced that lattice minimises sum-of-pairs edit cost,
              not oracle WER. A different valid alignment could score lower. Read it
              as "how good this substrate gets if every column choice were right",
              which is the question asked, with the lattice named.

Both oracles read the reference. They are ceilings, not systems, and they are
labelled as such everywhere they are printed.

Scoring is this repo's wtoks/sdi over the benchmark's referenceText. It is NOT the
benchmark app's published metric, which trims window-boundary words by cross-provider
consensus - so a single-system WER here will not equal the leaderboard number. Every
arm on this page, single systems included, is rescored the same way; that is what
makes the comparison internally consistent.

Writes results_composition_postjune.json (aggregates only, never transcript text).

Env: REPORT (path to a downloaded report.json), WORKERS, N_BOOT (10000)
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.msa import align3, compose, oracle_columns   # noqa: E402
from eval.controlled_eval.scoring import wtoks                         # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import sdi              # noqa: E402

REPORT = Path(os.environ.get("REPORT", ""))
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
WORKERS = int(os.environ.get("WORKERS", str(min(12, os.cpu_count() or 4))))
OUT = Path(__file__).with_name("results_composition_postjune.json")

OURS_NEW = "oc-cleanpack-cont-s47-b"
OURS_OLD = "oc-adapter-fixed-restage-2026-08-22"
TRIO = ["scribe", "soniox", OURS_NEW]

# Alternative trios, declared here before any of them is scored so the best one
# cannot be picked after the fact and reported as the headline.
ALT_TRIOS = [
    ["scribe", "soniox", OURS_OLD],
    ["scribe", "soniox", "hf-openai-whisper-large-v3"],
    ["scribe", "soniox", "gpt-4o-transcribe"],
    [OURS_NEW, OURS_OLD, "scribe"],
    [OURS_NEW, "hf-openai-whisper-large-v3", "gpt-4o-transcribe"],
]


def log(m):
    print(m, flush=True)


def align_one(payload):
    """Pure alignment + composition for one window, so it can run in a worker."""
    wid, toks, pivot, ref = payload
    lo = min(len(t) for t in toks)
    hi = max(len(t) for t in toks)
    # The band has to cover the largest length difference or the exact aligner cannot
    # reach the corner of the DP cube and silently returns a worse alignment.
    band = max(40, hi - lo + 20)
    cols = align3(*toks, band=band)
    w_tokens, _ = compose(cols, pivot)
    return wid, {"w": w_tokens, "oracle_msa": oracle_columns(cols, ref)}


def wer_of(counts):
    S, D, I, R = counts
    return (S + D + I) / R if R else 0.0


def totals(per_window, keys):
    S = D = I = R = 0
    for k in keys:
        s, d, i, r = per_window[k]
        S += s; D += d; I += i; R += r
    return S, D, I, R


def boot_ci(per_a, per_b, clusters, by_cluster, seed=20260823):
    """Paired meeting-clustered bootstrap on the WER difference a - b."""
    rng = random.Random(seed)
    point = wer_of(totals(per_a, [w for c in clusters for w in by_cluster[c]])) - \
            wer_of(totals(per_b, [w for c in clusters for w in by_cluster[c]]))
    out = []
    for _ in range(N_BOOT):
        sel = [rng.choice(clusters) for _ in clusters]
        wins = [w for c in sel for w in by_cluster[c]]
        out.append(wer_of(totals(per_a, wins)) - wer_of(totals(per_b, wins)))
    out.sort()
    return point, out[int(0.025 * N_BOOT)], out[int(0.975 * N_BOOT)]


def consensus_pick(hyps, trio):
    """Keep the hypothesis whose token set agrees most with the other two.

    Agreement is measured with the same sdi scorer, each candidate scored against
    each other candidate as if it were the reference, so the selector never sees the
    true reference.
    """
    best, best_score = trio[0], None
    for p in trio:
        score = 0
        for q in trio:
            if q == p:
                continue
            s, d, i, r = sdi(hyps[q], hyps[p])   # strings, not token lists
            score += (s + d + i) / max(1, r)
        if best_score is None or score < best_score:
            best, best_score = p, score
    return best


def run_trio(trio, items, refs, reft, texts, tokens, by_cluster, clusters, label):
    pivot_of = {}
    for it in items:
        w = it["id"]
        pick = consensus_pick({p: texts[w][p] for p in trio}, trio)
        pivot_of[w] = trio.index(pick)

    payloads = [(it["id"], [tokens[it["id"]][p] for p in trio],
                 pivot_of[it["id"]], reft[it["id"]]) for it in items]
    log(f"  aligning {len(payloads)} windows on {WORKERS} workers ({label})")
    A = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for n, (wid, r) in enumerate(ex.map(align_one, payloads, chunksize=1), 1):
            A[wid] = r
            if n % 50 == 0:
                log(f"    {n}/{len(payloads)}")

    per = {}
    per["V"] = {it["id"]: sdi(refs[it["id"]], texts[it["id"]][trio[pivot_of[it["id"]]]])
                for it in items}
    per["W"] = {w: sdi(refs[w], " ".join(A[w]["w"])) for w in A}
    per["oracle_msa"] = {w: sdi(refs[w], " ".join(A[w]["oracle_msa"])) for w in A}
    per["oracle_win"] = {}
    for it in items:
        w = it["id"]
        per["oracle_win"][w] = min((sdi(refs[w], texts[w][p]) for p in trio),
                                   key=lambda c: c[0] + c[1] + c[2])
    for p in trio:
        per[f"single:{p}"] = {it["id"]: sdi(refs[it["id"]], texts[it["id"]][p])
                              for it in items}
    artefacts = {"W_text": {w: " ".join(A[w]["w"]) for w in A},
                 "V_pick": {w: trio[pivot_of[w]] for w in pivot_of}}
    return per, artefacts


def main():
    assert REPORT.is_file(), "set REPORT to a downloaded report.json"
    d = json.loads(REPORT.read_text())
    providers = [p["instanceId"] for p in d["manifest"]["providers"]]

    items, tokens, texts, refs, reft = [], {}, {}, {}, {}
    dropped = 0
    for it in d["items"]:
        pp = it.get("perProvider") or {}
        # A window is only usable if every provider we might compare returned text.
        # Scoring one arm on a different subset than the arm it is compared against
        # is the exact defect that produced this project's earlier false results.
        usable = [p for p in providers if p in pp and pp[p].get("status") == "ok"]
        need = set(TRIO) | {OURS_OLD, "hf-openai-whisper-large-v3",
                            "gpt-4o-transcribe", "gladia-prod"}
        if not need.issubset(set(usable)):
            dropped += 1
            continue
        w = it["itemId"]
        items.append({"id": w, "cluster": (it["cityId"], it["meetingId"])})
        texts[w] = {p: pp[p]["hypothesisText"] for p in usable}
        tokens[w] = {p: wtoks(pp[p]["hypothesisText"]) for p in usable}
        refs[w] = it["referenceText"]          # sdi() tokenises this itself
        reft[w] = wtoks(it["referenceText"])   # oracle_columns needs the token list

    if os.environ.get("LIMIT"):     # smoke only; never used for a reported number
        items = items[:int(os.environ["LIMIT"])]
        log(f"LIMIT set: {len(items)} windows - SMOKE RUN, not reportable")

    by_cluster = defaultdict(list)
    for it in items:
        by_cluster[it["cluster"]].append(it["id"])
    clusters = sorted(by_cluster)
    log(f"{len(items)} windows ({dropped} dropped for a missing provider), "
        f"{len(clusters)} (city, meeting) clusters")
    # bench_data sets cluster=meetingId; on the full set that collapses 117 real
    # meetings into 66, because meeting ids repeat across cities. Hence the tuple.
    bare = len({c[1] for c in clusters})
    log(f"clustering on (city, meeting): {len(clusters)} clusters; "
        f"meeting id alone would give {bare}")
    if len(items) >= 100:
        assert len(clusters) > bare, \
            "meeting ids do not repeat across cities - check the clustering key"

    import hashlib, subprocess
    rh = hashlib.sha256(REPORT.read_bytes()).hexdigest()[:16]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    res = {"run": d["manifest"]["id"], "report_sha256_16": rh, "code_commit": commit,
           "boot_seed": 20260823, "windows": len(items),
           "clusters": len(clusters), "trio": TRIO, "arms": {}, "contrasts": {},
           "alt_trios": {}, "oracle_window_pools": {}}

    log(f"main trio: {TRIO}")
    per, art = run_trio(TRIO, items, refs, reft, texts, tokens, by_cluster, clusters, "main")
    allw = [w for c in clusters for w in by_cluster[c]]
    for arm, pw in per.items():
        S, D, I, R = totals(pw, allw)
        res["arms"][arm] = {"wer": (S + D + I) / R, "del": D / R, "ins": I / R,
                            "sub": S / R, "ref_tokens": R}
    # Per-window error counts (never text) so domination and leave-one-meeting-out
    # can be recomputed without redoing the alignment.
    res["per_window"] = {arm: {w: list(pw[w]) for w in allw} for arm, pw in per.items()}
    res["window_cluster"] = {it["id"]: list(it["cluster"]) for it in items}

    # Codex flagged "W vs best single" as test-set selection: whichever system wins
    # on THESE 391 windows was chosen using these references. scribe is the
    # prespecified comparator - it is what the product actually runs.
    COMPARATOR = "single:scribe"
    best_single = min((a for a in res["arms"] if a.startswith("single:")),
                      key=lambda a: res["arms"][a]["wer"])
    res["prespecified_comparator"] = COMPARATOR
    res["best_single_posthoc"] = best_single
    for a, b in [("W", "V"), ("W", COMPARATOR), ("V", COMPARATOR),
                 ("oracle_msa", "W"), ("oracle_win", "W")]:
        pt, lo, hi = boot_ci(per[a], per[b], clusters, by_cluster)
        res["contrasts"][f"{a} vs {b}"] = {"delta": pt, "ci": [lo, hi]}

    # ---- whole-window oracle over every subset of the provider pool, which needs no
    # ---- alignment and answers "would more systems help, at the selection level"
    pool = [p for p in providers if p != "oc-cleanpack-cont-s47-2026-08-22"]
    singles = {p: {w: sdi(refs[w], texts[w][p]) for w in allw} for p in pool
               if all(p in texts[w] for w in allw)}
    for k in range(1, len(singles) + 1):
        best = None
        for sub in combinations(sorted(singles), k):
            pw = {w: min((singles[p][w] for p in sub), key=lambda c: c[0] + c[1] + c[2])
                  for w in allw}
            S, D, I, R = totals(pw, allw)
            cand = ((S + D + I) / R, list(sub))
            if best is None or cand[0] < best[0]:
                best = cand
        res["oracle_window_pools"][k] = {"wer": best[0], "systems": best[1]}
        log(f"  oracle_window k={k}: {best[0]:.4f}  {best[1]}")

    # ---- alternative trios, declared before scoring
    seen = {tuple(sorted(TRIO))}
    for trio in (ALT_TRIOS if not os.environ.get("SKIP_ALT") else []):
        key = tuple(sorted(trio))
        if key in seen or not set(trio).issubset(set(singles)):
            continue
        seen.add(key)
        log(f"alt trio: {trio}")
        p2, _ = run_trio(trio, items, refs, reft, texts, tokens, by_cluster, clusters, "alt")
        res["alt_trios"][" + ".join(trio)] = {
            arm: {"wer": wer_of(totals(pw, allw))}
            for arm, pw in p2.items()}

    # --- reference-leakage test, run before anything is written -----------------
    # V and W must be computable with no reference at all. Feed both the scoring
    # reference and the oracle reference through a shuffle and recompute: the
    # COMPOSED TEXT and the vote's pick must come back identical. Comparing scores
    # would be circular, because scores are computed against the very references
    # being shuffled.
    order = sorted(reft)
    sh_t = dict(zip(order, [reft[k] for k in order][::-1]))
    sh_s = dict(zip(order, [refs[k] for k in order][::-1]))
    _, art2 = run_trio(TRIO, items, sh_s, sh_t, texts, tokens, by_cluster,
                       clusters, "leak-check")
    assert art2["W_text"] == art["W_text"], \
        "W's text changed when the references were shuffled - a reference leaks into it"
    assert art2["V_pick"] == art["V_pick"], \
        "V's pick changed when the references were shuffled - a reference leaks into it"
    log("leakage test passed: W text and V pick are identical under shuffled references")
    res["leakage_test"] = "passed: W text and V pick unchanged under shuffled references"

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    log(f"\nwrote {OUT}")
    log(f"\n{'arm':34s} {'WER':>7s} {'del':>7s} {'ins':>7s} {'sub':>7s}")
    for arm in sorted(res["arms"], key=lambda a: res["arms"][a]["wer"]):
        m = res["arms"][arm]
        log(f"{arm:34s} {m['wer']:7.4f} {m['del']:7.4f} {m['ins']:7.4f} {m['sub']:7.4f}")
    log("")
    for k, v in res["contrasts"].items():
        log(f"{k:34s} {v['delta']:+.4f}  95% CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")


if __name__ == "__main__":
    main()
