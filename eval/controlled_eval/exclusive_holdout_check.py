#!/usr/bin/env python3
"""Held-out replication of the hybrid probe, on windows it was never tuned on.

The hybrid rule (assign from the exclusive timeline, fall back to the regular one
when it yields nothing) was written after Phase 2 failed, on the same 25 windows
that produced the failure. That is the weakest thing about it. This script runs the
identical measurement on the **next** 25 windows by turn density, ranks 26 to 50,
which no part of the rule has seen.

PREDICTION, written before the run (git history is the timestamp):
  - drops: hybrid <= regular is arithmetic, not evidence. The hybrid only drops when
    BOTH timelines drop, so it can never exceed the regular count. Expected to hold
    trivially; it is reported for completeness, not as a result.
  - guess branch: the real prediction. Phase 2 saw 190 -> 46, a 76% reduction. If the
    effect is genuine and not an artefact of those particular windows, the held-out
    reduction should land in the same neighbourhood, say 60% or more.
  - speaker changes: expected around 5-10% of utterances, as in Phase 2 (51/718).

What this still cannot answer: whether the changed attributions are CORRECT. That
needs the blinded listening. This only tests whether the mechanical effect replicates.

Usage: python eval/controlled_eval/exclusive_holdout_check.py [--select|--diar|--score]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exclusive_diar_api import SC, api_key, log, run_one  # noqa: E402
from oc_merge_port import find_best_speaker, segments  # noqa: E402

WAV = SC / "winwav"
USED = SC / "exclusive_phase2_windows.json"
WINDOWS = SC / "exclusive_holdout_windows.json"
DIAR = SC / "exclusive_holdout_diar.json"
ASR = SC / "exclusive_holdout_asr.json"
OUT = Path(__file__).resolve().parent / "results_exclusive_holdout.json"
N = 25


def select():
    corpus = json.loads((SC / "precision2_corpus.json").read_text())
    used = {w["window_id"] for w in json.loads(USED.read_text())}
    rows = []
    for wid, v in corpus.items():
        turns, dur = v.get("turns") or [], float(v.get("measured_dur") or 0)
        if not turns or dur <= 0 or wid in used or not (WAV / f"{wid}.wav").exists():
            continue
        rows.append({"window_id": wid, "n_turns": len(turns),
                     "duration_sec": round(dur, 3),
                     "turns_per_min": round(len(turns) / (dur / 60), 4)})
    rows.sort(key=lambda r: (-r["turns_per_min"], r["window_id"]))
    WINDOWS.write_text(json.dumps(rows[:N], ensure_ascii=False, indent=1))
    log(f"{len(rows[:N])} held-out windows -> {WINDOWS}")
    for r in rows[:3]:
        log(f"  {r['window_id']}  {r['turns_per_min']} turns/min")


def diar():
    key = api_key()
    store = json.loads(DIAR.read_text()) if DIAR.exists() else {}
    for i, w in enumerate(json.loads(WINDOWS.read_text()), 1):
        wid = w["window_id"]
        if wid in store and "error" not in store[wid]:
            continue
        r = run_one(WAV / f"{wid}.wav", key, f"excl_ho_{wid}", exclusive=True)
        if r.get("status") != "succeeded":
            store[wid] = {"error": r.get("status", "unknown")}
        else:
            o = r.get("output") or {}
            store[wid] = {"diarization": o.get("diarization", []),
                          "exclusiveDiarization": o.get("exclusiveDiarization")}
        DIAR.write_text(json.dumps(store, ensure_ascii=False))
        log(f"  {i}/{N} {wid}: {store[wid].get('error') or 'ok'}")


def asr():
    from faster_whisper import WhisperModel
    from oc_inference_harness import _words_to_utterances

    store = json.loads(ASR.read_text()) if ASR.exists() else {}
    model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=14)
    for i, w in enumerate(json.loads(WINDOWS.read_text()), 1):
        wid = w["window_id"]
        if wid in store:
            continue
        segs, _ = model.transcribe(str(WAV / f"{wid}.wav"), language="el",
                                   task="transcribe", beam_size=1, temperature=0.0,
                                   condition_on_previous_text=False, vad_filter=False,
                                   word_timestamps=True)
        words = [{"word": x.word, "start": round(x.start, 3), "end": round(x.end, 3),
                  "confidence": 0.0} for s in segs for x in (s.words or [])]
        store[wid] = {"utterances": _words_to_utterances(words)}
        ASR.write_text(json.dumps(store, ensure_ascii=False))
        log(f"  {i}/{N} {wid}: {len(words)} words, {len(store[wid]['utterances'])} utts")


def score():
    diarz, asrz = json.loads(DIAR.read_text()), json.loads(ASR.read_text())
    t = collections.Counter()
    for wid in sorted(set(diarz) & set(asrz)):
        d = diarz[wid]
        if "error" in d or not d.get("exclusiveDiarization"):
            continue
        reg, exc = segments(d["diarization"]), segments(d["exclusiveDiarization"])
        for u in asrz[wid]["utterances"]:
            words = [(float(x["start"]), float(x["end"])) for x in u["words"]]
            a, b = (find_best_speaker(reg, float(u["start"]), float(u["end"]), words),
                    find_best_speaker(exc, float(u["start"]), float(u["end"]), words))
            h = b if b is not None else a
            t["n"] += 1
            t["drop_regular"] += a is None
            t["drop_exclusive"] += b is None
            t["drop_hybrid"] += h is None
            t["guess_regular"] += a is not None and a.branch == "guess"
            t["guess_exclusive"] += b is not None and b.branch == "guess"
            t["guess_hybrid"] += h is not None and h.branch == "guess"
            if h is not None and a is not None:
                t["speaker_changed"] += h.speaker != a.speaker

    red = 1 - t["guess_hybrid"] / t["guess_regular"] if t["guess_regular"] else None
    res = {
        "note": "held-out replication of a post-hoc rule; mechanical effect only, "
                "says nothing about whether the changed attributions are correct",
        "n_windows": len(set(diarz) & set(asrz)), "n_utterances": t["n"],
        "drops": {"regular": t["drop_regular"], "exclusive": t["drop_exclusive"],
                  "hybrid": t["drop_hybrid"]},
        "guess_branch": {"regular": t["guess_regular"],
                         "exclusive": t["guess_exclusive"], "hybrid": t["guess_hybrid"]},
        "guess_reduction_hybrid_vs_regular": round(red, 4) if red is not None else None,
        "speaker_changed_vs_regular": t["speaker_changed"],
        "speaker_changed_frac": round(t["speaker_changed"] / t["n"], 4) if t["n"] else None,
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:] or ["--select", "--diar", "--asr", "--score"]
    if "--select" in a:
        select()
    if "--diar" in a:
        diar()
    if "--asr" in a:
        asr()
    if "--score" in a:
        score()
