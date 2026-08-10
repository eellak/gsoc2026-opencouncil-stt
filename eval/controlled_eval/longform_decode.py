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

import atexit
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

    # One decoder per tag. Two of these ran at once on 2026-08-10 because a supervising
    # loop restarted what it wrongly believed was dead; each held its own in-memory dict
    # and overwrote the other's spans, so finished work silently disappeared and the log
    # printed span 16 before span 15. Unattended runs cannot afford that.
    lock = SET / f"hyp_{TAG}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        other = lock.read_text().strip()
        if other.isdigit() and Path(f"/proc/{other}").exists():
            sys.exit(f"{TAG}: already decoding as pid {other}; refusing to run twice")
        log(f"{TAG}: stale lock from pid {other or '?'}, taking it over")
        lock.unlink()
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    atexit.register(lambda: lock.unlink(missing_ok=True))

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
            # Segment timings only. The timestamp metric matches segment starts and ends to
            # VAD speech islands, so word-level alignment buys nothing and costs an extra
            # pass per segment. It also perturbs the segment boundaries being measured.
            word_timestamps=False, vad_filter=False)
        out = [{"start": float(s.start), "end": float(s.end), "text": s.text}
               for s in segs]
        hyps[row["wav"]] = {"segments": out,
                            "text": "".join(s["text"] for s in out).strip(),
                            "decode_sec": round(time.time() - t0, 1)}
        # Merge against what is on disk rather than trusting the in-memory copy, and
        # rename into place. Belt and braces after the double-decoder incident: the lock
        # should make this impossible, but losing hours of decoding to a clobbered write
        # is worse than one extra read.
        on_disk = json.loads(dest.read_text()) if dest.exists() else {}
        on_disk.update(hyps)
        hyps = on_disk
        tmp = dest.with_suffix(".part")
        tmp.write_text(json.dumps(hyps, ensure_ascii=False, indent=1))
        tmp.replace(dest)
        log(f"  {i}/{len(manifest)} {row['wav']} {len(out)} segs "
            f"{time.time()-t0:.0f}s ({(time.time()-t0)/row['dur_sec']:.2f}x RT)")

    log(f"{TAG}: {len(hyps)} spans -> {dest}")


if __name__ == "__main__":
    main()
