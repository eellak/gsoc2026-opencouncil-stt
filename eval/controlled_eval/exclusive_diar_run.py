#!/usr/bin/env python3
"""Phase 1 of the exclusive-diarization prereg: submit the 285 synthetic jobs.

Frozen by `docs/specs/exclusive-diarization-preregistration.md`:

  arm A, exclusive=True    95 jobs   (clean control, defines the local main speaker)
  arm C, exclusive=True    95 jobs   (the proposal)
  arm C, exclusive=False   95 jobs   (paired status-quo comparator + invariance check)

Submission order is randomized with seed 20260807 over the flat triple list, one
media object per job, sequential. Cost is projected from the first ten completed
jobs against the published €0.112/h precision-2 batch rate; over ~$20 projected and
the run falls back to the frozen 25-item comparator subset, then stops.

Resumable: every completed job is written to ~/.cache/oc-overlap/exclusive_phase1.json
immediately and skipped on the next invocation.

Usage: python eval/controlled_eval/exclusive_diar_run.py [--limit N]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import SC, api_key, log, run_one  # noqa: E402

MIX = SC / "mixtures"
MANIFEST = SC / "synth_overlap_manifest.json"
OUT = SC / "exclusive_phase1.json"
SEED = 20260807
EUR_PER_HOUR = 0.112  # pyannote.ai published precision-2 batch rate, 2026-08-07
USD_PER_EUR = 1.10    # documented constant, only used for the $20 stop rule
CAP_USD = 20.0
COMPARATOR_FALLBACK_N = 25


def plan() -> list[tuple[str, str, bool]]:
    man = json.loads(MANIFEST.read_text())
    ids = sorted(it["item_id"] for it in man["items"]
                 if (MIX / f"{it['item_id']}__A.wav").exists()
                 and (MIX / f"{it['item_id']}__C.wav").exists())
    triples = ([(i, "A", True) for i in ids]
               + [(i, "C", True) for i in ids]
               + [(i, "C", False) for i in ids])
    random.Random(SEED).shuffle(triples)
    return triples


def key_of(t: tuple[str, str, bool]) -> str:
    return f"{t[0]}|{t[1]}|{'excl' if t[2] else 'base'}"


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    key = api_key()
    triples = plan()
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    log(f"{len(triples)} planned jobs, {len(store)} already stored")

    comparator_subset = None  # set if the cost rule fires
    done_secs = sum(v.get("quantity") or 0 for v in store.values() if "error" not in v)
    n_priced = sum(1 for v in store.values() if v.get("quantity"))
    submitted = 0

    for t in triples:
        k = key_of(t)
        if k in store and "error" not in store[k]:
            continue
        iid, arm, excl = t
        if comparator_subset is not None and not excl and iid not in comparator_subset:
            store[k] = {"skipped": "cost_rule_comparator_subset"}
            continue
        path = MIX / f"{iid}__{arm}.wav"
        try:
            r = run_one(path, key, f"excl_p1_{k.replace('|', '_')}", exclusive=excl)
        except RuntimeError as e:
            store[k] = {"error": str(e)[:300]}
            log(f"  {k}: {e}")
        else:
            if r.get("status") != "succeeded":
                store[k] = {"error": r.get("status", "unknown")}
                log(f"  {k}: FAILED {r.get('status')}")
            else:
                o = r.get("output") or {}
                store[k] = {
                    "diarization": o.get("diarization", []),
                    "exclusiveDiarization": o.get("exclusiveDiarization"),
                    "quantity": r.get("quantity"),
                    "createdAt": r.get("createdAt"),
                }
                q = r.get("quantity")
                if q:
                    done_secs += q
                    n_priced += 1
        OUT.write_text(json.dumps(store, ensure_ascii=False))
        submitted += 1

        # ---------------------------------------------------- frozen cost rule
        if n_priced == 10 and comparator_subset is None:
            mean_sec = done_secs / n_priced
            proj_usd = (mean_sec * len(triples) / 3600) * EUR_PER_HOUR * USD_PER_EUR
            log(f"cost projection from 10 jobs: mean {mean_sec:.0f}s/job, "
                f"{len(triples)} jobs -> ${proj_usd:.2f}")
            if proj_usd > CAP_USD:
                ids = sorted({x[0] for x in triples})[:COMPARATOR_FALLBACK_N]
                comparator_subset = set(ids)
                log(f"OVER CAP -> comparator restricted to first {COMPARATOR_FALLBACK_N} ids")
                proj2 = (mean_sec * (190 + COMPARATOR_FALLBACK_N) / 3600) * EUR_PER_HOUR * USD_PER_EUR
                if proj2 > CAP_USD:
                    log(f"STILL over cap (${proj2:.2f}) -> stopping per prereg")
                    return

        if limit and submitted >= limit:
            log(f"--limit {limit} reached")
            break

        if submitted % 20 == 0:
            ok = sum(1 for v in store.values() if "error" not in v and "skipped" not in v)
            log(f"  {ok}/{len(triples)} ok, {done_secs / 3600:.2f} audio-hours "
                f"(~€{done_secs / 3600 * EUR_PER_HOUR:.2f})")

    ok = sum(1 for v in store.values() if "diarization" in v)
    err = sum(1 for v in store.values() if "error" in v)
    log(f"\ndone: {ok} ok, {err} failed, {len(triples)} planned")
    log(f"audio: {done_secs / 3600:.2f} h  ~€{done_secs / 3600 * EUR_PER_HOUR:.2f}")
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
