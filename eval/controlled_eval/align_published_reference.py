"""Force-align the published reference of the two repair windows onto their audio.

The repair page has to show the reviewer WHICH published words fall in the 20 s interval
being listened to; the validation manifest carries no word times, so they are derived
here with the same CTC aligner `anchor_timings.py` already uses. No model transcript is
involved: only the published reference text is aligned, and not one character of it
changes.

Output: `<root>/aligned.json` under ~/.cache, never in git.

    .venv-eval/bin/python -m eval.controlled_eval.align_published_reference
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval.controlled_eval.anchor_timings import token_times
from eval.controlled_eval.build_reference_repair_page import DEFAULT_DENSE, DEFAULT_OUT, WINDOW_IDS

WINDOW_LENGTH = 30
BATCH_SIZE = 8
SR = 16000


def load_session(threads: int):
    import onnxruntime
    from ctc_forced_aligner import MODEL_URL, Tokenizer, ensure_onnx_model
    model_path = os.path.join(os.path.expanduser("~"), "ctc_forced_aligner", "model.onnx")
    ensure_onnx_model(model_path, MODEL_URL)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(model_path, sess_options=options), Tokenizer()


def align_window(session, tokenizer, audio_path: Path, text: str) -> list[list[float]]:
    import numpy as np
    import soundfile as sf
    from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                                    postprocess_results, preprocess_text)
    arr, sr = sf.read(str(audio_path), dtype="float32")
    if sr != SR:
        raise RuntimeError(f"{audio_path} is {sr} Hz, the aligner expects {SR}")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    emissions, stride = generate_emissions(
        session, np.ascontiguousarray(arr, dtype=np.float32),
        window_length=WINDOW_LENGTH, batch_size=BATCH_SIZE)
    tok_star, txt_star = preprocess_text(text, romanize=True, language="ell",
                                         split_size="word", star_frequency="edges")
    segments, scores, blank = get_alignments(emissions, tok_star, tokenizer)
    spans = get_spans(tok_star, segments, blank)
    words = [w for w in postprocess_results(txt_star, spans, stride, scores)
             if w.get("text") != "<star>"]
    times = token_times(words, text)
    if times is None:
        raise RuntimeError(f"aligner output does not reproduce the reference tokens: {audio_path}")
    return [[round(start, 3), round(end, 3)] for start, end in times]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    validation = args.dense_root / "corrected-input/eval/validation"
    manifest = {row["window_id"]: row
                for row in json.loads((validation / "manifest.json").read_text())["eval_windows"]}
    session, tokenizer = load_session(args.threads)
    out = {}
    for window_id in WINDOW_IDS:
        row = manifest[window_id]
        text = " ".join(row["ref_tokens"])
        times = align_window(session, tokenizer, validation / row["audio_local"], text)
        if len(times) != len(row["ref_tokens"]):
            raise RuntimeError(f"{window_id}: {len(times)} times for {len(row['ref_tokens'])} tokens")
        out[window_id] = times
        print(f"[align] {window_id}: {len(times)} tokens", flush=True)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "aligned.json").write_text(json.dumps(out) + "\n")
    print(json.dumps({"aligned": str(args.root / "aligned.json"),
                      "windows": {k: len(v) for k, v in out.items()}}))


if __name__ == "__main__":
    main()
