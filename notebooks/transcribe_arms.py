"""Transcribe the 32 dev reference windows with each trained arm, on the pod.

Runs on the GPU that just finished training, before the pod is torn down. Doing this
locally on CPU instead costs several hours per adapter for 3.7 minutes of audio; here
it is a couple of minutes each, on hardware that is already paid for and idle between
arms.

Only the 32 `dev` clips are uploaded to the pod. The 16 locked ones stay on the local
machine — they are the final test and nothing that touches a GPU should be able to see
them by accident.

Decoding is frozen and identical for every arm: greedy, Greek, no timestamps. The arms
differ in training data and nothing else, so any decode difference would be a confound.

    transcribe_arms.py --clips /workspace/refclips --out /workspace/hyps \
                       [--adapters /workspace/whisper-run]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL_ID = "openai/whisper-large-v3"


def log(m):
    print(f"[transcribe {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", default="/workspace/whisper-run")
    ap.add_argument("--base", action="store_true",
                    help="also transcribe with the untrained base model")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    clips = sorted(Path(args.clips).glob("*.wav"))
    if not clips:
        raise SystemExit(f"no clips in {args.clips}")
    log(f"{len(clips)} clips")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = WhisperProcessor.from_pretrained(MODEL_ID, language="greek",
                                            task="transcribe")

    # Adapters are only read when COMPLETE exists: a directory alone can be a run that
    # died mid-save, and scoring a half-written adapter would look like a bad arm.
    todo = []
    if args.base:
        todo.append(("base", None))
    for d in sorted(Path(args.adapters).glob("adapter_*")):
        if (d / "COMPLETE").exists():
            todo.append((d.name.replace("adapter_", ""), d))
        else:
            log(f"skip {d.name}: no COMPLETE marker")
    log(f"{len(todo)} arms to transcribe: {[t[0] for t in todo]}")

    for tag, adapter in todo:
        dest = out_dir / f"{tag}.json"
        if dest.exists():
            log(f"skip {tag}: already transcribed")
            continue
        t0 = time.time()
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16).to("cuda")
        if adapter is not None:
            model = PeftModel.from_pretrained(model, str(adapter),
                                              torch_dtype=torch.float16)
        model.eval()
        model.generation_config.language = "greek"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None

        hyps = {}
        for i, p in enumerate(clips, 1):
            y, sr = sf.read(str(p))
            if y.ndim > 1:
                y = y.mean(axis=1)
            feats = proc.feature_extractor(y, sampling_rate=sr,
                                           return_tensors="pt").input_features
            with torch.no_grad():
                ids = model.generate(feats.to("cuda", torch.float16),
                                     num_beams=1, do_sample=False,
                                     max_new_tokens=200)
            hyps[p.name] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if i % 10 == 0:
                log(f"  {tag}: {i}/{len(clips)}")
        dest.write_text(json.dumps(hyps, ensure_ascii=False, indent=1))
        log(f"{tag}: {len(hyps)} clips in {time.time()-t0:.0f}s -> {dest.name}")

        del model
        torch.cuda.empty_cache()

    log(f"done; {len(list(out_dir.glob('*.json')))} hypothesis files in {out_dir}")


if __name__ == "__main__":
    main()
