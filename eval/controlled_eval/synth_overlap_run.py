#!/usr/bin/env python3
"""Transcribe the synthetic-overlap mixtures. Runs on the GPU pod, not the mini-PC.

Decoder settings are frozen by `docs/specs/synthetic-overlap-preregistration.md` and are
identical in every arm: greedy, no temperature fallback, no VAD, no carry-over of previous
text. `condition_on_previous_text=True` would let arm A's decode influence nothing but
would make each file's own segments depend on each other in a way that interacts with the
inserted event, which is exactly the mechanism we are trying to measure cleanly.

Processing order is shuffled with a fixed seed so any drift in the runtime cannot line up
with an arm.

Usage (on the pod):
  MIX=/workspace/mixtures MODELS=/workspace/models python synth_overlap_run.py
Env: MIX MODELS OUT_JSON ARMS(comma) RESUME(1)
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

MIX = Path(os.environ.get("MIX", "/workspace/mixtures"))
MODELS = Path(os.environ.get("MODELS", "/workspace/models"))
OUT_JSON = Path(os.environ.get("OUT_JSON", "/workspace/synth_overlap_hyps.json"))
ARMS = [a for a in os.environ.get("ARMS", "A,B,C,D,E,F,G,H,donor").split(",") if a]

# name -> ct2 model dir or hub id. The fine-tune is the same CTranslate2 build the
# benchmark scored as `oc-minipc-finetune`, so the numbers stay comparable.
SYSTEMS = {
    "finetune": str(MODELS / "ct2"),
    "whisper-large-v3": "large-v3",
}
# the ±3 dB gain controls only need the primary system
FINETUNE_ONLY = {"G", "H"}

DECODE = dict(language="el", beam_size=1, temperature=0.0, best_of=1,
              condition_on_previous_text=False, vad_filter=False,
              without_timestamps=False)


def log(*a):
    print(*a, flush=True)


def main():
    manifest = json.loads((MIX / "manifest.json").read_text())
    items = manifest["items"]

    jobs = []
    for sysname in SYSTEMS:
        for it in items:
            for arm in ARMS:
                if arm in FINETUNE_ONLY and sysname != "finetune":
                    continue
                p = MIX / f"{it['item_id']}__{arm}.wav"
                if p.exists():
                    jobs.append((sysname, it["item_id"], arm, p))
    random.Random(20260803).shuffle(jobs)
    log(f"{len(jobs)} transcriptions over {len(SYSTEMS)} systems")

    out = {}
    if OUT_JSON.exists() and os.environ.get("RESUME", "1") == "1":
        out = json.loads(OUT_JSON.read_text())
        log(f"resuming with {len(out)} done")

    from faster_whisper import WhisperModel

    # One model in memory at a time: the cheap GPUs this is sized for have 16-24 GB and
    # two large-v3 in float16 is a needless risk of an OOM halfway through a paid run.
    for sysname, spec in SYSTEMS.items():
        todo = [j for j in jobs if j[0] == sysname and f"{j[0]}|{j[1]}|{j[2]}" not in out]
        if not todo:
            continue
        log(f"loading {sysname} ({spec})")
        model = WhisperModel(spec, device="cuda", compute_type="float16")
        t0 = time.time()
        for i, (_, item_id, arm, path) in enumerate(todo, 1):
            segs, _ = model.transcribe(str(path), **DECODE)
            segs = [(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in segs]
            # Segment times are kept so the analysis can ask whether the damage stays
            # local to the inserted event or spills into the rest of the window.
            out[f"{sysname}|{item_id}|{arm}"] = {
                "text": " ".join(t for _, _, t in segs).strip(), "segments": segs}
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                log(f"  {sysname} {i}/{len(todo)} ({el / i:.1f}s each, "
                    f"eta {(len(todo) - i) * el / i / 60:.0f}min)")
                tmp = OUT_JSON.with_suffix(".part")
                tmp.write_text(json.dumps(out, ensure_ascii=False))
                tmp.replace(OUT_JSON)
        del model
    tmp = OUT_JSON.with_suffix(".part")
    tmp.write_text(json.dumps(out, ensure_ascii=False))
    tmp.replace(OUT_JSON)
    log(f"{len(out)} hypotheses -> {OUT_JSON}")


if __name__ == "__main__":
    main()
