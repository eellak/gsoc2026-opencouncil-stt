#!/usr/bin/env python3
"""Decode the frozen long-form set with one system, keeping the timestamps.

Step 1 of the preflight in the
[preregistration](../../docs/specs/window-shape-preregistration.md). The question is
whether fine-tuning on short cut clips damages long-form behaviour, so the two things worth
keeping are the segment timings and the raw text, not a score.

`condition_on_previous_text=True` is deliberate and preregistered. Production runs it False
because that is the guard against repetition loops, and measuring loop burden with the guard
on would hide the effect in both models. The mechanism in the literature is the model
forgetting to *consume* previous text, which only shows when it is given some.

Resumable per span: this runs for hours, and a pass that cannot resume is a pass that gets
restarted from zero.

  MODEL=large-v3 TAG=base .venv-eval/bin/python -m eval.controlled_eval.longform_decode
  MODEL=/home/harold/oc-asr-serve/ct2 TAG=finetune ...
Env: SET MODEL TAG BEAM
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SET = Path(os.environ.get("SET", Path.home() / "oc-longform")).expanduser()
MODEL = os.environ.get("MODEL", "large-v3")
TAG = os.environ.get("TAG") or sys.exit("TAG is required")
BEAM = int(os.environ.get("BEAM", "5"))


def log(*a):
    print(*a, flush=True)


def main() -> None:
    manifest = json.loads((SET / "manifest.json").read_text())
    dest = SET / f"hyp_{TAG}.json"
    hyps = json.loads(dest.read_text()) if dest.exists() else {}
    log(f"{TAG}: {len(manifest)} spans, {len(hyps)} already done, model {MODEL}")

    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=16)

    for i, row in enumerate(manifest, 1):
        if row["wav"] in hyps:
            continue
        t0 = time.time()
        segs, info = model.transcribe(
            str(SET / "audio" / row["wav"]), language="el", beam_size=BEAM,
            condition_on_previous_text=True,   # preregistered: the sensitive condition
            word_timestamps=True, vad_filter=False)
        out = [{"start": float(s.start), "end": float(s.end), "text": s.text}
               for s in segs]
        hyps[row["wav"]] = {"segments": out,
                            "text": "".join(s["text"] for s in out).strip(),
                            "decode_sec": round(time.time() - t0, 1)}
        dest.write_text(json.dumps(hyps, ensure_ascii=False, indent=1))
        log(f"  {i}/{len(manifest)} {row['wav']} {len(out)} segs "
            f"{time.time()-t0:.0f}s ({(time.time()-t0)/row['dur_sec']:.2f}x RT)")

    log(f"{TAG}: {len(hyps)} spans -> {dest}")


if __name__ == "__main__":
    main()
