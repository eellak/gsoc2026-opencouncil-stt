# -*- coding: utf-8 -*-
"""Live three-system fusion over real audio, scored on one stack.

What this does that no earlier run did: it calls ElevenLabs Scribe v2, Soniox
stt-async-v5 and our own CT2 fine-tune on the SAME audio bytes, in the same
process, and feeds all three into the SAME fusion/fuse.py the frozen 0.11205 was
measured with. Every arm below is decoded live and scored with one normalizer,
so the comparison is inside one stack.

The metric is AGREEMENT-WITH-OPENCOUNCIL: the reference is OpenCouncil's own
published transcript, not a human who listened to this audio. It records product
compatibility. It does not decide which system is closer to the audio, and no
sentence in the output may say that it does.

The sample is drawn from a seed fixed in this file, with a per-city cap, before
any number was seen. Do not re-draw it after reading a result.

Usage:
    .venv-eval/bin/python -m eval.live_fusion.run_live --n 30
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = Path(os.environ.get("FUSION_REPO", "/home/harold/projects/opencouncil/opencouncil-tasks"))
sys.path.insert(0, str(TASKS))
sys.path.insert(0, str(REPO))

from fusion.normalize import sdi, wtoks  # noqa: E402

from eval.live_fusion import providers as P  # noqa: E402

BENCH_JSON = Path.home() / ".cache/oc-public/bench_2026-06-10-oc-benchmark.json"
AUDIO_DIR = Path.home() / ".cache/oc-public/bench_windows"
FUSE = TASKS / "fusion" / "fuse.py"

# Frozen before the first live call. Changing any of these invalidates the draw.
SAMPLE_SEED = "live-fusion-2026-09-07/v1"
MAX_PER_CITY = 4

# Meetings our adapter was trained on. The 260-window benchmark is heavily
# contaminated for OUR arm -- an earlier audit found 148 of 253 windows inside a
# training fold -- and a contaminated `ours` inflates the fusion arm too, because
# fusion reads it as one of three votes. The pool is therefore restricted to
# windows whose (city, meeting) appears in none of the training packs.
TRAINING_PACKS = ["clean-pack-arm-2026-08", "spliced-pack-arm-2026-08",
                  "spliced-pack-arm-jitter-2026-08"]


def trained_meetings():
    """(city, meeting) pairs the adapter was trained on. Key is the pair: meeting
    slugs repeat across cities, and a meeting-only key over-excludes."""
    seen = set()
    for pack in TRAINING_PACKS:
        path = Path.home() / ".cache/oc-public" / pack / "manifest.jsonl"
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                seen.add((row["city_id"], row["meeting_id"]))
    return seen

# The production chunking, mirrored from PRODUCTION_CHUNKING in
# src/lib/fusion/FusionTranscriber.ts.
CHUNKING = {"max_tokens": 120, "anchor_n": 3, "search_radius": 200}

ARMS = {
    "W": {"arm": "W", "guard": False},
    "rules_on": {"arm": "rules", "guard": True},
}


def draw(items, n, cap=MAX_PER_CITY):
    """Deterministic draw: training-disjoint windows, per-city cap, fixed seed.

    The whole ordered list is decided here, before any decoding, and every drawn
    id stays in the run ledger whatever happens to it. Scoring only the windows
    that happened to finish would make the result conditional on completion.
    """
    trained = trained_meetings()
    rng = random.Random(SAMPLE_SEED)
    pool = [it for it in sorted(items, key=lambda it: it["itemId"])
            if (AUDIO_DIR / f"{it['itemId']}.wav").exists()
            and (it["cityId"], it["meetingId"]) not in trained]
    rng.shuffle(pool)
    picked, per_city = [], defaultdict(int)
    for item in pool:
        if per_city[item["cityId"]] >= cap:
            continue
        picked.append(item)
        per_city[item["cityId"]] += 1
        if len(picked) == n:
            break
    return picked, len(pool)


def fuse(payload):
    proc = subprocess.run([sys.executable, str(FUSE)], input=json.dumps(payload),
                          capture_output=True, text=True, cwd=str(TASKS))
    if proc.returncode != 0:
        raise RuntimeError(f"fuse.py exit {proc.returncode}: {proc.stderr[-400:]}")
    return json.loads(proc.stdout)


def fused_text(out):
    return " ".join(t["norm"] for t in out["tokens"] if t.get("norm"))


def score(ref_text, hyp_tokens):
    s, d, i, n = sdi(ref_text, " ".join(hyp_tokens))
    return {"s": s, "d": d, "i": i, "n": n}


def totals(rows):
    t = {"s": 0, "d": 0, "i": 0, "n": 0}
    for r in rows:
        for k in t:
            t[k] += r[k]
    t["wer"] = (t["s"] + t["d"] + t["i"]) / t["n"] if t["n"] else None
    t["deletion_rate"] = t["d"] / t["n"] if t["n"] else None
    return t


def bootstrap(per_window, arm_a, arm_b, clusters, replicates=4000, seed=7):
    """Paired cluster bootstrap of WER(arm_a) - WER(arm_b), resampling meetings."""
    rng = random.Random(seed)
    by_cluster = defaultdict(list)
    for wid, cluster in clusters.items():
        by_cluster[cluster].append(wid)
    names = sorted(by_cluster)
    deltas = []
    for _ in range(replicates):
        pick = [names[rng.randrange(len(names))] for _ in names]
        ta = {"s": 0, "d": 0, "i": 0, "n": 0}
        tb = {"s": 0, "d": 0, "i": 0, "n": 0}
        for cluster in pick:
            for wid in by_cluster[cluster]:
                for k in ta:
                    ta[k] += per_window[wid][arm_a][k]
                    tb[k] += per_window[wid][arm_b][k]
        if ta["n"] and tb["n"]:
            deltas.append((ta["s"] + ta["d"] + ta["i"]) / ta["n"]
                          - (tb["s"] + tb["d"] + tb["i"]) / tb["n"])
    deltas.sort()
    if not deltas:
        return None
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]
    return {"lo": round(lo, 5), "hi": round(hi, 5), "replicates": len(deltas)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cap", type=int, default=MAX_PER_CITY,
                    help="windows per city. Stage A is 4 and stage B is 8, both fixed in "
                         "PREDECLARED.md before any aggregate existed; the shuffle is the "
                         "same, so stage A is a prefix of stage B.")
    ap.add_argument("--ours", choices=["local", "gpu"], default="local",
                    help="local = CPU int8 on this machine, the canonical measurement "
                         "stack. gpu = the RunPod serverless endpoint production calls. "
                         "The two are different stacks and their numbers are never merged.")
    ap.add_argument("--out", default=str(REPO / "eval/results_live_fusion.json"))
    args = ap.parse_args()

    scribe_key = os.environ.get("SCRIBE_API_KEY")
    soniox_key = os.environ.get("SONIOX_API_KEY")
    runpod_key = os.environ.get("RUNPOD_API_KEY")
    if not scribe_key or not soniox_key:
        print("SCRIBE_API_KEY and SONIOX_API_KEY must be set", file=sys.stderr)
        return 2
    if args.ours == "gpu" and not runpod_key:
        print("RUNPOD_API_KEY must be set for --ours gpu", file=sys.stderr)
        return 2

    bench = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
    sample, pool_size = draw(bench["items"], args.n, args.cap)
    print(f"[live] {len(sample)} of {pool_size} training-disjoint windows, "
          f"seed {SAMPLE_SEED}, cap {args.cap}/city", file=sys.stderr)

    per_window, clusters, failures = {}, {}, []
    shas = {}
    pool_size_note = pool_size
    started = time.time()

    for k, item in enumerate(sample, 1):
        wid = item["itemId"]
        wav = AUDIO_DIR / f"{wid}.wav"
        t0 = time.time()
        try:
            sw, s_sha = P.scribe_words(wid, wav, scribe_key)
            nw, n_sha = P.soniox_words(wid, wav, soniox_key)
            if args.ours == "gpu":
                ow, o_sha = P.ours_gpu_words(wid, wav, runpod_key)
            else:
                ow, o_sha = P.ours_words(wid, wav)
        except Exception as exc:                      # noqa: BLE001
            failures.append({"id": wid, "stage": "provider", "error": type(exc).__name__,
                             "detail": str(exc)[:200]})
            print(f"[live] {k}/{len(sample)} {wid} PROVIDER FAIL {type(exc).__name__}",
                  file=sys.stderr)
            continue

        shas = {"scribe": s_sha, "soniox": n_sha, "ours": o_sha}
        # An empty stream is a REAL recognizer outcome, not an error -- but
        # fusing two systems and one empty one is a different system from the one
        # being measured, so the window is recorded and excluded from every arm
        # rather than quietly degenerating the fusion to a two-way vote.
        if not sw or not nw or not ow:
            failures.append({"id": wid, "stage": "empty_stream",
                             "note": "valid empty recognition, excluded from all arms so the "
                                     "arms stay paired",
                             "counts": {"scribe": len(sw), "soniox": len(nw), "ours": len(ow)}})
            print(f"[live] {k}/{len(sample)} {wid} EMPTY STREAM "
                  f"(scribe {len(sw)} soniox {len(nw)} ours {len(ow)})", file=sys.stderr)
            continue

        payload_systems = [
            {"id": "scribe", "params_sha": s_sha, "words": sw},
            {"id": "soniox", "params_sha": n_sha, "words": nw},
            {"id": "ours", "params_sha": o_sha, "words": ow},
        ]

        ref = item["referenceText"]
        row = {
            "scribe": score(ref, wtoks(" ".join(w["raw"] for w in sw))),
            "soniox": score(ref, wtoks(" ".join(w["raw"] for w in nw))),
            "ours": score(ref, wtoks(" ".join(w["raw"] for w in ow))),
        }
        try:
            for name, cfg in ARMS.items():
                payload = {
                    "schema": "oc-fusion-in/1",
                    "audio_sha256": "0" * 64,
                    "systems": payload_systems,
                    "config": {**cfg, "llm": None, "chunking": dict(CHUNKING)},
                }
                out = fuse(payload)
                row[name] = score(ref, fused_text(out).split())
                if name == "rules_on":
                    row["chunks"] = out["config"]["chunking"].get("n_chunks")
                    row["forced_cuts"] = out["config"]["chunking"].get("forced_cuts")
        except Exception as exc:                      # noqa: BLE001
            failures.append({"id": wid, "stage": "fuse", "error": type(exc).__name__,
                             "detail": str(exc)[:200]})
            print(f"[live] {k}/{len(sample)} {wid} FUSE FAIL {type(exc).__name__}",
                  file=sys.stderr)
            continue

        row["words"] = {"scribe": len(sw), "soniox": len(nw), "ours": len(ow)}
        row["duration_sec"] = item["durationSec"]
        per_window[wid] = row
        clusters[wid] = f"{item['cityId']}/{item['meetingId']}"
        print(f"[live] {k}/{len(sample)} {wid} ok in {time.time() - t0:.0f}s "
              f"(scribe {len(sw)} soniox {len(nw)} ours {len(ow)} words)", file=sys.stderr)

        _write(args.out, per_window, clusters, failures, shas, sample, started,
                   pool_size_note, args.ours, args.cap)

    _write(args.out, per_window, clusters, failures, shas, sample, started,
                   pool_size_note, args.ours, args.cap)
    print(f"[live] done: {len(per_window)} scored, {len(failures)} failed, "
          f"{time.time() - started:.0f}s", file=sys.stderr)
    return 0


def compare(per_window, arm_a, arm_b):
    """Everything needed to see whether one window carries the headline.

    A corpus WER difference can be produced entirely by a single window; the
    leave-one-out range says whether removing any one of them reverses it, and
    the win/tie/loss counts say whether the effect is broad or local.
    """
    ids = sorted(per_window)

    def corpus(subset):
        ta = tb = 0
        na = nb = 0
        for wid in subset:
            ra, rb = per_window[wid][arm_a], per_window[wid][arm_b]
            ta += ra["s"] + ra["d"] + ra["i"]; na += ra["n"]
            tb += rb["s"] + rb["d"] + rb["i"]; nb += rb["n"]
        return (ta / na - tb / nb) if na and nb else None

    full = corpus(ids)
    loo = [(wid, corpus([w for w in ids if w != wid])) for wid in ids]
    values = [v for _, v in loo if v is not None]
    wins = ties = losses = 0
    for wid in ids:
        ra, rb = per_window[wid][arm_a], per_window[wid][arm_b]
        ea = ra["s"] + ra["d"] + ra["i"]
        eb = rb["s"] + rb["d"] + rb["i"]
        if ea < eb:
            wins += 1
        elif ea == eb:
            ties += 1
        else:
            losses += 1
    worst = max(loo, key=lambda item: item[1] if item[1] is not None else -1)
    return {
        "corpus_delta_wer": round(full, 5) if full is not None else None,
        "leave_one_out_min": round(min(values), 5) if values else None,
        "leave_one_out_max": round(max(values), 5) if values else None,
        "reverses_on_removing_one_window": bool(values and full is not None
                                                and max(values) * full <= 0),
        "window_that_hurts_the_claim_most": worst[0],
        "windows_better": wins, "windows_tied": ties, "windows_worse": losses,
        "note": ("negative corpus_delta_wer means the first arm has fewer errors; "
                 "if reverses_on_removing_one_window is true the benefit is one window"),
    }


def _write(path, per_window, clusters, failures, shas, sample, started, pool_size=None,
           ours_stack="local", cap=MAX_PER_CITY):
    """Rewritten after every window so a killed run still has its numbers."""
    arms = ["scribe", "soniox", "ours", "W", "rules_on"]
    agg = {a: totals([r[a] for r in per_window.values()]) for a in arms} if per_window else {}

    # One window carrying the headline is a different finding from the same
    # delta spread over the sample, and this project has been burned by not
    # looking. Reported for the contrast that matters: rules_on against Scribe.
    domination = None
    if per_window:
        contrib = {}
        for wid, r in per_window.items():
            contrib[wid] = ((r["scribe"]["s"] + r["scribe"]["d"] + r["scribe"]["i"])
                            - (r["rules_on"]["s"] + r["rules_on"]["d"] + r["rules_on"]["i"]))
        net = sum(contrib.values())
        if net:
            top = max(contrib, key=lambda w: abs(contrib[w]))
            domination = {"net_errors_saved": net, "largest_window": top,
                          "largest_window_share": round(contrib[top] / net, 3)}

    sensitivity = {}
    if len(per_window) >= 2:
        for a, b in (("rules_on", "scribe"), ("rules_on", "W")):
            sensitivity[f"{a}_vs_{b}"] = compare(per_window, a, b)

    out = {
        "experiment": "exp-2026-08-26-fusion-integration",
        "run": "live-fusion-2026-09-07",
        "metric": "agreement-with-OpenCouncil",
        "metric_note": ("The reference is OpenCouncil's own published transcript, not a human "
                        "who listened to this audio. This records product compatibility and "
                        "decides nothing about fidelity to the audio."),
        "sample": {
            "source": "bench_2026-06-10-oc-benchmark.json items with cached window audio",
            "seed": SAMPLE_SEED,
            "pool": ("bench windows with cached audio whose (city, meeting) is in none of "
                     + ", ".join(TRAINING_PACKS)),
            "pool_size": pool_size, "max_per_city": cap,
            "n_drawn": len(sample), "n_scored": len(per_window), "n_failed": len(failures),
            "selected_ids": [it["itemId"] for it in sample],
            "contamination_caveat": ("Disjoint from the 2026-08 training packs only. The "
                                     "adapter continued from an earlier one whose data was "
                                     "not audited here, so this is an upper bound on how "
                                     "clean the `ours` arm is."),
        },
        "stack": {
            "scribe": {"params": P.SCRIBE_PARAMS, "params_sha": shas.get("scribe")},
            "soniox": {"params": P.SONIOX_PARAMS, "params_sha": shas.get("soniox"),
                       "transport": "file upload by file_id, not audio_url (no public bucket "
                                    "credential here); model and decode flags are the frozen ones"},
            "ours_stack": ours_stack,
            "ours_stack_note": ("local = CPU int8 on the mini-PC, the canonical measurement "
                                "stack. gpu = the RunPod serverless endpoint, float16 on CUDA "
                                "at beam 2, which is what production calls. The runbook is "
                                "explicit that the two give different tokens from the same "
                                "weights, so these are two runs, never one."),
            "ours": {"model_dir": str(P.OURS_MODEL_DIR), "model_sha256": P.OURS_MODEL_SHA,
                     "artifact": "artifact-ct2-cleanpack-cont-s47",
                     "runtime": P.OURS_RUNTIME, "decode": P.OURS_DECODE,
                     "params_sha": shas.get("ours"),
                     "deviation": "word_timestamps=True; the frozen CPU config had it False"},
            "fusion": {"repo": str(TASKS), "chunking": CHUNKING,
                       "normalizer": "fusion.normalize.wtoks", "scorer": "fusion.normalize.sdi"},
        },
        "totals": agg,
        "contrasts": {},
        "domination_rules_on_vs_scribe": domination,
        "sensitivity": sensitivity,
        "failures": failures,
        "per_window": per_window,
        "clusters": clusters,
        "elapsed_sec": round(time.time() - started, 1),
    }
    if len(per_window) >= 2:
        for a, b in (("rules_on", "scribe"), ("rules_on", "W"), ("W", "scribe")):
            out["contrasts"][f"{a}_minus_{b}"] = bootstrap(per_window, a, b, clusters)
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
