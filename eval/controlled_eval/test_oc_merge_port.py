#!/usr/bin/env python3
"""Parity: the Python port must match the real pinned TypeScript exactly.

Four frozen fixture families (preregistration § Port):

  1. boundary / direct        — words exactly on segment edges, one eligible segment
  2. competing + engineered tie — winner, drift value, deterministic tie behaviour
  3. false coverage / drop    — segment overlaps the envelope but covers no word
  4. randomized differential  — 2000 random cases, seed 20260807

The oracle is `oc_merge_oracle.mts` run under tsx against a clone of
schemalabz/opencouncil-tasks at commit 5ff16a3c…; set OC_TASKS to its path.

Usage: OC_TASKS=/path/to/opencouncil-tasks python eval/controlled_eval/test_oc_merge_port.py
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oc_merge_port import find_best_speaker, segments  # noqa: E402

HERE = Path(__file__).resolve().parent
ORACLE = HERE / "oc_merge_oracle.mts"
OC_TASKS = os.environ.get("OC_TASKS", "")
SEED = 20260807
N_RANDOM = 2000
TOL = 1e-9


def oracle(fixtures: list[dict]) -> list[dict]:
    r = subprocess.run(["npx", "--yes", "tsx", str(ORACLE), OC_TASKS],
                       input=json.dumps(fixtures), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"oracle failed: {r.stderr[-2000:]}")
    return json.loads(r.stdout)


def port(f: dict) -> dict:
    r = find_best_speaker(segments(f["diarization"]), f["start"], f["end"],
                          [(w["start"], w["end"]) for w in f["words"]])
    return {"speaker": None, "drift": None} if r is None else {
        "speaker": r.speaker, "drift": r.drift}


def fixtures_1_boundary() -> list[dict]:
    return [
        {"diarization": [{"start": 0.0, "end": 2.0, "speaker": "S1"}],
         "start": 0.0, "end": 2.0,
         "words": [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]},
        # word touching both edges exactly -> containment is inclusive
        {"diarization": [{"start": 1.0, "end": 3.0, "speaker": "S1"},
                         {"start": 5.0, "end": 6.0, "speaker": "S2"}],
         "start": 1.0, "end": 3.0,
         "words": [{"start": 1.0, "end": 3.0}]},
        # segment starting exactly at utterance end still counts as overlapping
        {"diarization": [{"start": 0.0, "end": 2.0, "speaker": "S1"},
                         {"start": 2.0, "end": 4.0, "speaker": "S2"}],
         "start": 0.0, "end": 2.0,
         "words": [{"start": 0.2, "end": 1.8}]},
    ]


def fixtures_2_competing() -> list[dict]:
    return [
        # two speakers cover the words, asymmetric timestamps
        {"diarization": [{"start": 0.0, "end": 5.0, "speaker": "S1"},
                         {"start": 0.5, "end": 4.0, "speaker": "S2"},
                         {"start": 4.5, "end": 9.0, "speaker": "S2"}],
         "start": 1.0, "end": 3.5,
         "words": [{"start": 1.0, "end": 1.8}, {"start": 2.0, "end": 3.5}]},
        # engineered exact tie: symmetric geometry, both drift the same
        {"diarization": [{"start": 0.0, "end": 1.0, "speaker": "S1"},
                         {"start": 2.0, "end": 3.0, "speaker": "S1"},
                         {"start": 0.0, "end": 1.0, "speaker": "S2"},
                         {"start": 2.0, "end": 3.0, "speaker": "S2"}],
         "start": 0.0, "end": 3.0,
         "words": [{"start": 0.0, "end": 1.0}, {"start": 1.2, "end": 1.8},
                   {"start": 2.0, "end": 3.0}]},
        # three candidates, one clearly best
        {"diarization": [{"start": 0.0, "end": 10.0, "speaker": "S1"},
                         {"start": 0.0, "end": 10.0, "speaker": "S2"},
                         {"start": 0.0, "end": 10.0, "speaker": "S3"},
                         {"start": 3.0, "end": 4.0, "speaker": "S2"}],
         "start": 2.0, "end": 6.0,
         "words": [{"start": 2.0, "end": 3.0}, {"start": 4.0, "end": 6.0}]},
    ]


def fixtures_3_drop() -> list[dict]:
    return [
        # the prereg's case: overlaps the envelope, covers no word -> drop
        {"diarization": [{"start": 0.8, "end": 1.2, "speaker": "S1"},
                         {"start": 9.0, "end": 9.5, "speaker": "S2"}],
         "start": 0.0, "end": 2.0,
         "words": [{"start": 0.0, "end": 0.4}, {"start": 1.6, "end": 2.0}]},
        # nothing near the utterance at all
        {"diarization": [{"start": 20.0, "end": 25.0, "speaker": "S1"}],
         "start": 0.0, "end": 2.0,
         "words": [{"start": 0.0, "end": 2.0}]},
        # exactly one touching segment that covers no word: fast path still fires
        {"diarization": [{"start": 0.8, "end": 1.2, "speaker": "S1"}],
         "start": 0.0, "end": 2.0,
         "words": [{"start": 0.0, "end": 0.4}, {"start": 1.6, "end": 2.0}]},
    ]


def fixtures_4_random(n: int) -> list[dict]:
    rng = random.Random(SEED)
    out = []
    while len(out) < n:
        n_spk = rng.randint(1, 4)
        diar = []
        for s in range(n_spk):
            t = rng.uniform(0, 3)
            for _ in range(rng.randint(1, 6)):
                d = rng.uniform(0.1, 4.0)
                diar.append({"start": round(t, 3), "end": round(t + d, 3),
                             "speaker": f"S{s}"})
                t += d + rng.uniform(-0.3, 3.0)  # negative gap -> overlap
                t = max(t, 0.0)
        rng.shuffle(diar)
        u0 = rng.uniform(0, 12)
        u1 = u0 + rng.uniform(0.3, 6.0)
        words, t = [], u0
        while t < u1:
            d = rng.uniform(0.05, 0.9)
            words.append({"start": round(t, 3), "end": round(min(t + d, u1), 3)})
            t += d + rng.uniform(0, 0.4)
        if not words:
            continue
        out.append({"diarization": diar, "start": round(u0, 3),
                    "end": round(u1, 3), "words": words})
    return out


def check(name: str, fixtures: list[dict]) -> bool:
    exp = oracle(fixtures)
    bad = 0
    for i, (f, e) in enumerate(zip(fixtures, exp)):
        g = port(f)
        same_spk = g["speaker"] == e["speaker"]
        same_drift = (g["drift"] is None and e["drift"] is None) or (
            g["drift"] is not None and e["drift"] is not None
            and abs(g["drift"] - e["drift"]) <= TOL)
        if not (same_spk and same_drift):
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH #{i}: port={g} ts={e}")
                print(f"    fixture={json.dumps(f)}")
    ok = bad == 0
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {len(fixtures) - bad}/{len(fixtures)}")
    return ok


def main():
    if not OC_TASKS or not Path(OC_TASKS, "src/lib/DiarizationManager.ts").exists():
        print("set OC_TASKS to a clone of schemalabz/opencouncil-tasks @ 5ff16a3c")
        sys.exit(2)
    sha = subprocess.run(["git", "-C", OC_TASKS, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"oracle: {OC_TASKS} @ {sha}")
    if sha != "5ff16a3c20968d6a5610d3584322b9a0059ad482":
        print("WARNING: not the pinned commit")

    results = [
        check("1 boundary/direct", fixtures_1_boundary()),
        check("2 competing/tie", fixtures_2_competing()),
        check("3 false coverage/drop", fixtures_3_drop()),
        check(f"4 randomized differential (n={N_RANDOM})", fixtures_4_random(N_RANDOM)),
    ]
    print("\nALL PASS" if all(results) else "\nFAILURES PRESENT")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
