#!/usr/bin/env python3
"""Where does the segmentation gap enter: the training, the merge, or the conversion?

The [preflight](../../docs/reports/2026-08-09-longform-preflight.md) found the fine-tune
cutting the same audio into a third as many segments as base. That was read as a
consequence of training on single-utterance clips, and Codex pointed out the reading skips
a step: "one engine, one machine" controls the runtime, not the lineage of the artifact.
The base is Systran's conversion of the released weights; ours went through a LoRA merge and
a CTranslate2 int8 export. A merge fault or a conversion loss would produce the same
symptom with the training format innocent.

So walk the ladder on fixed 30-second chunks, one encoder window each, and count how many
timestamp pairs each rung emits:

  1. HF base                 openai/whisper-large-v3
  2. HF base + live LoRA     the adapter applied by peft, nothing merged
  3. HF merged               the merge that build_model.sh produced
  4. CT2 int8                what actually serves

Reading it: a gap already at rung 2 is learned behaviour and the finding stands. A gap that
appears at 3 is a merge bug. A gap that appears at 4 is the conversion. Nothing here needs
new audio, new listening, or a GPU.

  .venv-eval/bin/python -m eval.controlled_eval.conversion_ladder --chunks 40
Env: SET ADAPTER MERGED CT2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from pathlib import Path

SET = Path(os.environ.get("SET", Path.home() / "oc-longform")).expanduser()
BASE_ID = os.environ.get("BASE_ID", "openai/whisper-large-v3")
ADAPTER = os.environ.get("ADAPTER", "/home/harold/oc-asr-serve/adapter-fixed-2026-08-01")
MERGED = os.environ.get("MERGED", "/home/harold/oc-asr-serve/merged")
CT2 = os.environ.get("CT2", "/home/harold/oc-asr-serve/ct2")
CHUNK_SEC = 30.0
TS = re.compile(r"<\|(\d+\.\d+)\|>")


def log(*a):
    print(*a, flush=True)


def chunks(n: int) -> list[tuple[str, float]]:
    """Fixed (wav, offset) pairs, spread over the meetings rather than over one of them."""
    manifest = json.loads((SET / "manifest.json").read_text())
    per = max(1, n // len(manifest))
    out = []
    for r in manifest:
        for k in range(per):
            # deterministic offsets inside the ten minutes, away from the edges
            out.append((r["wav"], 60.0 + k * 137.0))
            if len(out) >= n:
                return out
    return out


def audio(wav: str, off: float):
    import soundfile as sf
    arr, sr = sf.read(str(SET / "audio" / wav), dtype="float32",
                      start=int(off * 16000), frames=int(CHUNK_SEC * 16000))
    return arr, sr


def hf_rung(name: str, load, items) -> list[dict]:
    import torch
    from transformers import WhisperProcessor
    proc = WhisperProcessor.from_pretrained(BASE_ID, language="greek", task="transcribe")
    model = load()
    model.eval()
    rows = []
    for i, (wav, off) in enumerate(items, 1):
        arr, sr = audio(wav, off)
        feats = proc.feature_extractor(arr, sampling_rate=sr,
                                       return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, language="el", task="transcribe",
                                 return_timestamps=True, num_beams=1, max_new_tokens=440)
        txt = proc.tokenizer.decode(ids[0], decode_with_timestamps=True)
        marks = TS.findall(txt)
        rows.append({"wav": wav, "off": off, "pairs": len(marks) // 2,
                     "words": len(TS.sub(" ", txt).split())})
        if i % 10 == 0:
            log(f"  {name}: {i}/{len(items)}")
    del model
    return rows


def ct2_rung(name: str, path: str, items) -> list[dict]:
    from faster_whisper import WhisperModel
    m = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=16)
    rows = []
    for i, (wav, off) in enumerate(items, 1):
        arr, _ = audio(wav, off)
        segs, _ = m.transcribe(arr, language="el", beam_size=1,
                               condition_on_previous_text=False,
                               word_timestamps=False, vad_filter=False)
        segs = list(segs)
        rows.append({"wav": wav, "off": off, "pairs": len(segs),
                     "words": sum(len(s.text.split()) for s in segs)})
        if i % 10 == 0:
            log(f"  {name}: {i}/{len(items)}")
    del m
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--json", default="eval/controlled_eval/results_conversion_ladder.json")
    a = ap.parse_args()

    items = chunks(a.chunks)
    log(f"{len(items)} chunks of {CHUNK_SEC:.0f}s from "
        f"{len({w for w, _ in items})} meetings")

    import torch
    from transformers import WhisperForConditionalGeneration

    def base():
        return WhisperForConditionalGeneration.from_pretrained(
            BASE_ID, dtype=torch.float32)

    def lora():
        from peft import PeftModel
        return PeftModel.from_pretrained(base(), ADAPTER)

    def merged():
        return WhisperForConditionalGeneration.from_pretrained(MERGED, dtype=torch.float32)

    out = {"chunk_sec": CHUNK_SEC, "n": len(items), "rungs": {}}
    for name, fn in [("1_hf_base", lambda: hf_rung("1_hf_base", base, items)),
                     ("2_hf_lora", lambda: hf_rung("2_hf_lora", lora, items)),
                     ("3_hf_merged", lambda: hf_rung("3_hf_merged", merged, items)),
                     ("4_ct2_int8", lambda: ct2_rung("4_ct2_int8", CT2, items))]:
        log(f"rung {name}")
        rows = fn()
        out["rungs"][name] = {
            "pairs_mean": round(st.mean(r["pairs"] for r in rows), 2),
            "words_mean": round(st.mean(r["words"] for r in rows), 1),
            "by_chunk": rows,
        }
        r = out["rungs"][name]
        log(f"  -> {r['pairs_mean']} timestamp pairs, {r['words_mean']} words per 30s")
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1))

    log("")
    for k, v in out["rungs"].items():
        log(f"{k:14s} {v['pairs_mean']:6.2f} pairs  {v['words_mean']:6.1f} words")


if __name__ == "__main__":
    main()
