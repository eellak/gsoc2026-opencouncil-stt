#!/usr/bin/env python3
"""Phase 2 input prep: transcribe the 25 frozen windows with word timestamps.

One transcription per window, shared by both variants, so the utterance set is
identical by construction and the regular-vs-exclusive comparison is paired.

faster-whisper large-v3, greedy, VAD off, language el, CPU int8 — the decoder
settings frozen in the preregistration. Utterances are built by the vault's own
`_words_to_utterances` (pause >= 1.0 s / sentence-final punctuation / 30 s cap),
imported from `eval/oc_inference_harness.py` so there is one implementation.

Output (~/.cache/oc-overlap/exclusive_phase2_asr.json) contains utterance TEXT and
therefore never enters git.

Usage: ~/.cache/oc-overlap/fwvenv/bin/python eval/controlled_eval/exclusive_phase2_asr.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exclusive_diar_api import SC, log  # noqa: E402
from oc_inference_harness import _words_to_utterances  # noqa: E402

WINDOWS = SC / "exclusive_phase2_windows.json"
WAV = SC / "winwav"
OUT = SC / "exclusive_phase2_asr.json"


def main():
    from faster_whisper import WhisperModel

    windows = json.loads(WINDOWS.read_text())
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    log(f"{len(windows)} windows, {len(store)} already transcribed")

    model = WhisperModel("large-v3", device="cpu", compute_type="int8",
                         cpu_threads=14)

    for i, w in enumerate(windows, 1):
        wid = w["window_id"]
        if wid in store:
            continue
        path = WAV / f"{wid}.wav"
        if not path.exists():
            store[wid] = {"error": "missing_wav"}
            log(f"  {wid}: missing wav")
            continue
        t0 = time.time()
        segs, info = model.transcribe(
            str(path), language="el", task="transcribe", beam_size=1,
            temperature=0.0, condition_on_previous_text=False,
            vad_filter=False, word_timestamps=True)
        words = []
        for s in segs:
            for wd in (s.words or []):
                words.append({"word": wd.word, "start": round(wd.start, 3),
                              "end": round(wd.end, 3), "confidence": 0.0})
        utts = _words_to_utterances(words)
        store[wid] = {"n_words": len(words), "n_utterances": len(utts),
                      "duration": info.duration, "utterances": utts,
                      "elapsed_sec": round(time.time() - t0, 1)}
        OUT.write_text(json.dumps(store, ensure_ascii=False))
        log(f"  {i}/{len(windows)} {wid}: {len(words)} words, {len(utts)} utts, "
            f"{store[wid]['elapsed_sec']}s")

    ok = [v for v in store.values() if "error" not in v]
    log(f"\n{len(ok)}/{len(windows)} transcribed, "
        f"{sum(v['n_utterances'] for v in ok)} utterances total")
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
