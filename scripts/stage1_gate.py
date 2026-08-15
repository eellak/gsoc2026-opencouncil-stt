#!/usr/bin/env python3
"""Stage-1 catastrophic gate — run on the pod BEFORE stage 2 spends GPU time.

Prereg (docs/specs/2026-08-15-external-packs-screens-prereg.md §4): the stage-1
checkpoint must produce non-empty transcriptions on a smoke set drawn from
TRAINING meetings (never the 39 frozen windows). Strengthened per Codex review
(job c12eb5b8) with three degeneracy checks:

  per clip : non-empty output
             no 5-gram repeated >= 3x (looping decoder)
             output length <= 3x reference token length (runaway generation)
  overall  : >= 20% token overlap with the reference on >= min-overlap-clips clips

Clips come from the stage-2 superset clip build (WORK/manifest.json), so the
smoke set is training data by construction. Prints one PASS/FAIL line and exits
0/1 — the launch sequence must stop on 1.

    python stage1_gate.py --adapter /workspace/stage1/adapter --work /workspace/stage2
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path


def toks(s: str) -> list[str]:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", s)


def repeated_5gram(t: list[str]) -> bool:
    if len(t) < 5:
        return False
    c = collections.Counter(tuple(t[i:i + 5]) for i in range(len(t) - 4))
    return c.most_common(1)[0][1] >= 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--work", required=True, help="stage-2 WORK_DIR (manifest.json)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--min-overlap-clips", type=int, default=3)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    a = ap.parse_args()

    import numpy as np
    man = json.loads((Path(a.work) / "manifest.json").read_text())
    pool = man["train"]
    idx = np.random.default_rng(a.seed).choice(len(pool), size=a.n, replace=False)
    clips = [pool[i] for i in sorted(idx.tolist())]

    import soundfile as sf
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(a.model, language="greek", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, a.adapter).to(dev).eval()
    model.generation_config.language, model.generation_config.task = "greek", "transcribe"

    failures, overlap_ok = [], 0
    for c in clips:
        wav, sr = sf.read(c["audio"], dtype="float32")
        feats = proc(wav, sampling_rate=sr, return_tensors="pt").input_features
        with torch.no_grad():
            out = model.generate(feats.to(dev, dtype), max_new_tokens=225)
        hyp = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        ht, rt = toks(hyp), toks(c["text"])
        overlap = len(set(ht) & set(rt)) / len(set(rt)) if rt else 0.0
        overlap_ok += overlap >= 0.20
        name = Path(c["audio"]).stem
        if not hyp:
            failures.append(f"{name}: EMPTY output")
        elif repeated_5gram(ht):
            failures.append(f"{name}: 5-gram repeated >=3x")
        elif rt and len(ht) > 3 * len(rt):
            failures.append(f"{name}: output {len(ht)} toks > 3x ref {len(rt)}")
        print(f"  {name}: {len(ht)} toks, overlap {overlap:.2f} :: {hyp[:80]}",
              flush=True)

    if overlap_ok < a.min_overlap_clips:
        failures.append(f"token overlap >=20% on only {overlap_ok}/{a.n} clips "
                        f"(need {a.min_overlap_clips})")
    if failures:
        print("STAGE1 GATE: FAIL\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)
    print(f"STAGE1 GATE: PASS ({a.n} clips, overlap>=20% on {overlap_ok})", flush=True)


if __name__ == "__main__":
    main()
