#!/usr/bin/env python3
"""Arm C phase-2: timestamped re-decode of the coverage-ambiguous eval windows.

Purpose: the preliminary VAD pass (c_preliminary_vad.py, output
c-preliminary-vad-2026-08-12.json) works on aggregate decoded_seconds, which sums
segment spans including inter-island silence. For a handful of windows the
undecoded tail (audio - decoded) is large enough that, at the window's own speech
rate, it could in principle hold as many words as were deleted. This script
re-decodes ONLY those windows with word_timestamps=True so segment/word spans can
be intersected with the Silero islands recorded in the preliminary JSON.

Decode config mirrors eval-A (the control arm) exactly except word_timestamps.
DO NOT run while another decode job owns the CPU. Not run as part of phase 1.

Usage:
  .venv-eval/bin/python scripts/analysis/c_phase2_timed_decode.py [wid ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WAV_DIR = Path.home() / ".cache/oc-public/bench_windows"
MODEL_DIR = "/home/harold/oc-asr-serve/ct2-fixed"
OUT = ROOT / "data/reports/finetune-research/c-phase2-timed-decode-2026-08-12.json"

# Windows where phase-1 could not settle the coverage question:
#  - the single positive VAD-minus-decoded gap window (oct31_2), plus windows with
#    audio_minus_decoded > 5 s whose generous coverage bound >= observed deletions
#    or nearly so (sep24, may22, aug29, feb27_305602, sep10),
#  - and the three deletion-dominant windows (jan30, jan21, apr7) purely to
#    confirm the within-span mechanism with word-level timings.
DEFAULT_WINDOWS = [
    "win_argos_oct31__2_2025_2353650",
    "win_argos_sep24_2025_371824",
    "win_argos_may22_2026_1650387",
    "win_argos_aug29_2025_2731295",
    "win_orestiada_feb27_2026_305602",
    "win_argos_sep10_2025_573077",
    "win_argos_jan30_2026_204180",
    "win_orestiada_jan21_2026_7536686",
    "win_argos_apr7_2026_960810",
]

# eval-A config, frozen BEFORE seeing any phase-2 number; only word_timestamps
# differs (True here, False in eval-A).
DECODE = dict(
    language="el",
    task="transcribe",
    beam_size=5,
    condition_on_previous_text=False,
    vad_filter=False,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    no_speech_threshold=0.6,
    log_prob_threshold=-1.0,
    compression_ratio_threshold=2.4,
    word_timestamps=True,
)


def main() -> None:
    import torch
    torch.set_num_threads(8)
    from faster_whisper import WhisperModel

    wids = sys.argv[1:] or DEFAULT_WINDOWS
    model = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")

    out = {"model_dir": MODEL_DIR, "device": "cpu", "compute_type": "int8",
           "config": {**DECODE}, "windows": {}}
    for wid in wids:
        wav = WAV_DIR / f"{wid}.wav"
        t0 = time.time()
        segments, info = model.transcribe(str(wav), **DECODE)
        segs = []
        for s in segments:
            segs.append({
                "start": round(s.start, 2), "end": round(s.end, 2),
                "text": s.text,
                "avg_logprob": round(s.avg_logprob, 3),
                "no_speech_prob": round(s.no_speech_prob, 3),
                "compression_ratio": round(s.compression_ratio, 3),
                "temperature": s.temperature,
                "words": [{"start": round(w.start, 2), "end": round(w.end, 2),
                           "word": w.word, "prob": round(w.probability, 3)}
                          for w in (s.words or [])],
            })
        out["windows"][wid] = {
            "duration": round(info.duration, 2),
            "n_segments": len(segs),
            "decoded_seconds": round(sum(s["end"] - s["start"] for s in segs), 2),
            "wall_seconds": round(time.time() - t0, 1),
            "segments": segs,
        }
        print(f"{wid}: {len(segs)} segments, "
              f"{out['windows'][wid]['decoded_seconds']}s decoded, "
              f"{out['windows'][wid]['wall_seconds']}s wall", flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
