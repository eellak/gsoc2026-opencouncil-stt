#!/usr/bin/env python3
"""Re-decode the 247 benchmark windows with `artifact-adapter-fixed`, asking
faster-whisper for per-word probabilities.

`exp-2026-08-16-adapter-confidence`. The 247-window benchmark decode
(`oc-runpod-fixed-2026-08-10`) ran with `word_timestamps: false`, so no word
probability was ever computed. This pass re-runs the **frozen** decode config of
that run with exactly one field changed - `word_timestamps: true` - and records
every word's probability, plus the segment `avg_logprob` our serving code already
exposes.

FROZEN CONFIG. `notebooks/decode_ablation.CONTROL` is the same dict as the run's
`decode.json`. Nothing here may deviate from it except `word_timestamps`.

SEED. `decode_ablation.seed_for("A", window_id)` - the same per-window seed the
control arm used. Common random numbers: any difference against the cached arm-A
text is attributable to `word_timestamps`, not to a different sampling draw.

STACK. `MODEL_DIR=/home/harold/oc-asr-serve/ct2-fixed` (artifact-ct2-fixed), CPU
int8. This is NOT the stack the cached benchmark text came from (that was the
same server code on a RunPod GPU). The two effects are separated on the 37
windows that the eval freeze and this benchmark share, where a cached local
CPU int8 `word_timestamps=false` decode already exists (`$SC/decode-ablation/
eval-A.json`).

Output: $SC/adapter-confidence/wt-true-247.json - word text and probabilities,
i.e. verbatim council speech. It never enters git.

    SC=~/.cache/oc-public .venv-eval/bin/python eval/adapter_confidence_decode.py
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "decode_ablation", ROOT / "notebooks/decode_ablation.py")
DA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DA)

from eval.controlled_eval import bench_data as B  # noqa: E402

RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
PROVIDER = "oc-runpod-fixed-2026-08-10"
N_WINDOWS = 247
N_SEALED_INSIDE = 6
THREADS = int(os.environ.get("AC_THREADS", "12"))


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def out_dir() -> Path:
    d = sc() / "adapter-confidence"
    d.mkdir(parents=True, exist_ok=True)
    return d


def substrate() -> list[dict]:
    """The 247 windows: the fusion substrate, sealed holdout removed."""
    report = B.load_report(RUN_ID)
    items = B.common_items(report, B.provider_ids(report))
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text()
    )["holdout_windows"]}
    before = len(items)
    items = [it for it in items if it["item_id"] not in sealed]
    assert before - len(items) == N_SEALED_INSIDE, (
        f"expected {N_SEALED_INSIDE} sealed windows removed, got {before - len(items)}")
    assert len(items) == N_WINDOWS, f"expected {N_WINDOWS} windows, got {len(items)}"
    return items


def main() -> None:
    import argparse

    import ctranslate2
    from faster_whisper import WhisperModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--word-timestamps", choices=["true", "false"], default="true")
    args = ap.parse_args()
    wt = args.word_timestamps == "true"

    kw = dict(DA.CONTROL, word_timestamps=wt)
    assert {k: v for k, v in kw.items() if k != "word_timestamps"} == \
           {k: v for k, v in DA.CONTROL.items() if k != "word_timestamps"}

    items = substrate()
    dest = out_dir() / f"wt-{'true' if wt else 'false'}-247.json"
    state = json.loads(dest.read_text()) if dest.exists() else {
        "run_id": RUN_ID, "provider": PROVIDER, "config": kw,
        "model_dir": DA.MODEL_DIR, "device": "cpu", "compute": "int8",
        "threads": THREADS, "windows": {}}
    # PREREGISTERED AMENDMENT, 2026-08-16 20:05 EEST, made before any contrast was
    # computed: this box is shared and the paired pass was measured at ~14 h. The
    # REMAINING queue is randomized with a fixed seed - IDENTICAL in both passes, so
    # the slower pass's completed set is nested inside the faster one's - and decoding
    # stops at a wall-clock deadline fixed in advance. The analysis set is then the
    # substrate-order head already decoded (disclosed, city-clustered) plus a uniform
    # random sample of the remainder. See the prereg spec.
    # The shuffle is applied to the FULL substrate, not to the per-pass todo list:
    # the two passes have different already-decoded sets, so shuffling their todos
    # separately would produce two different orders and destroy the nesting.
    import random
    order = list(items)
    random.Random(21).shuffle(order)
    todo = [it for it in order if it["item_id"] not in state["windows"]]
    print(f"{len(todo)} windows to decode ({len(state['windows'])} done)", flush=True)
    if not todo:
        return

    model = WhisperModel(DA.MODEL_DIR, device="cpu", compute_type="int8",
                         cpu_threads=THREADS)
    diag = DA.Diagnostics()
    fw_log = logging.getLogger("faster_whisper")
    prev = fw_log.level
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)
    try:
        for i, it in enumerate(todo, 1):
            wid = it["item_id"]
            wav = sc() / "bench_windows" / f"{wid}.wav"
            if not wav.exists():
                raise SystemExit(f"missing audio for {wid}")
            diag.reset()
            ctranslate2.set_random_seed(DA.seed_for("A", wid))
            t0 = time.time()
            segments, info = model.transcribe(str(wav), **kw)
            segs = list(segments)
            elapsed = time.time() - t0

            state["windows"][wid] = {
                "text": "".join(s.text for s in segs).strip(),
                "n_segments": len(segs),
                "audio_seconds": round(info.duration, 2),
                "seed": DA.seed_for("A", wid),
                "wall_seconds": round(elapsed, 1),
                "segments": [{
                    "start": round(float(s.start), 3),
                    "end": round(float(s.end), 3),
                    "text": s.text,
                    "avg_logprob": (None if s.avg_logprob is None
                                    else round(float(s.avg_logprob), 6)),
                    "no_speech_prob": (None if s.no_speech_prob is None
                                       else round(float(s.no_speech_prob), 6)),
                    "temperature": (None if s.temperature is None
                                    else float(s.temperature)),
                    "words": [{"w": w.word,
                               "s": round(float(w.start), 3),
                               "e": round(float(w.end), 3),
                               "p": round(float(w.probability), 6)}
                              for w in (s.words or [])],
                } for s in segs],
                **diag.snapshot(),
            }
            state.setdefault("resolved_options", DA.opts_to_dict(info.transcription_options))
            tmp = dest.with_suffix(".part")
            tmp.write_text(json.dumps(state, ensure_ascii=False))
            tmp.replace(dest)
            print(f"  {i}/{len(todo)} {wid} {elapsed:.0f}s segs={len(segs)}", flush=True)
    finally:
        fw_log.removeHandler(diag)
        fw_log.setLevel(prev)
    print(f"-> {dest}", flush=True)


if __name__ == "__main__":
    main()
