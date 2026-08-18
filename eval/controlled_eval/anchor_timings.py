#!/usr/bin/env python3
"""Per-token times for the three frozen hypotheses, on one common clock.

Why this stage exists at all is argued in
`docs/specs/2026-08-18-anchored-realignment-prereg.md` §3: on the 247-window fusion
substrate **none of W's three voters carries usable per-word times**. Scribe's are not
in the benchmark report, Soniox's were discarded by the client (and the free model that
has them is a different model with different text), and the adapter was decoded with
`word_timestamps: false` (the re-decode that has them changes the transcript in 101 of
102 paired windows).

So the times are DERIVED: each frozen hypothesis text is force-aligned to the same
window audio with one CTC aligner, and every token gets an interval on one clock. No
character of any hypothesis changes, so W is untouched and the baseline stays the
frozen baseline.

The aligner is the PyPI `ctc-forced-aligner` (deskpai, ONNX, MMS CTC) already used by
`eval/hf_export/build.py`. Emissions are computed ONCE per window and reused for all
three texts, so the three streams are timed against a bit-identical acoustic posterior.

Output: `$SC/anchored-2026-08/timings.json`, verbatim council speech, never in git.

    .venv-eval/bin/python -m eval.controlled_eval.anchor_timings --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import bench_data as B          # noqa: E402
from eval.controlled_eval.scoring import wtoks            # noqa: E402

RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
TRIO = ["scribe-v2-clean", "soniox", "oc-runpod-fixed-2026-08-10"]
CACHE_NAME = "anchored-2026-08"
WINDOW_LENGTH = 30
BATCH_SIZE = 8
SR = 16000


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def cache_dir() -> Path:
    return sc() / CACHE_NAME


def timings_path() -> Path:
    return cache_dir() / "timings.json"


def wav_path(item_id: str) -> Path:
    return sc() / "bench_windows" / f"{item_id}.wav"


def token_times(words: list[dict], text: str):
    """Expand aligner words onto the scorer's token space.

    Returns a list of (start, end) parallel to `wtoks(text)`, or None when the
    expansion does not reproduce `wtoks(text)` exactly. Times travel by OCCURRENCE
    INDEX, never by token string, so a repeated word is never confused with itself.
    """
    toks, times = [], []
    for w in words:
        for t in wtoks(w.get("text", "")):
            toks.append(t)
            times.append((float(w["start"]), float(w["end"])))
    return times if toks == wtoks(text) else None


# --------------------------------------------------------------------- worker
_STATE: dict = {}


def _init(threads: int):
    import onnxruntime
    from ctc_forced_aligner import MODEL_URL, Tokenizer, ensure_onnx_model
    mp = os.path.join(os.path.expanduser("~"), "ctc_forced_aligner", "model.onnx")
    ensure_onnx_model(mp, MODEL_URL)
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    _STATE["sess"] = onnxruntime.InferenceSession(mp, sess_options=so)
    _STATE["tok"] = Tokenizer()


def _one(payload):
    """One window: emissions once, then all three texts against them."""
    import numpy as np
    import soundfile as sf
    from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                                    postprocess_results, preprocess_text)
    item_id, hyps = payload
    p = wav_path(item_id)
    if not p.exists():
        return item_id, {"status": "no_audio"}
    try:
        arr, sr = sf.read(str(p), dtype="float32")
        if sr != SR:
            return item_id, {"status": "bad_sample_rate"}
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        emissions, stride = generate_emissions(
            _STATE["sess"], np.ascontiguousarray(arr, dtype=np.float32),
            window_length=WINDOW_LENGTH, batch_size=BATCH_SIZE)
    except Exception as ex:                                    # noqa: BLE001
        return item_id, {"status": f"emission_failed:{type(ex).__name__}"}

    out = {}
    for name, text in zip(TRIO, hyps):
        try:
            tok_star, txt_star = preprocess_text(
                text, romanize=True, language="ell", split_size="word",
                star_frequency="edges")
            segs, scores, blank = get_alignments(emissions, tok_star, _STATE["tok"])
            spans = get_spans(tok_star, segs, blank)
            words = [w for w in postprocess_results(txt_star, spans, stride, scores)
                     if w.get("text") != "<star>"]
        except Exception as ex:                                # noqa: BLE001
            return item_id, {"status": f"align_failed:{type(ex).__name__}"}
        tt = token_times(words, text)
        if tt is None:
            return item_id, {"status": "token_map_mismatch"}
        out[name] = [[round(s, 3), round(e, 3)] for s, e in tt]
    return item_id, {"status": "ok", "times": out,
                     "audio_seconds": round(len(arr) / SR, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    report = B.load_report(RUN_ID)
    # provider list identical to fusion_lab.load_substrate: the substrate is the
    # windows every provider in the run returned, so the item set matches exactly.
    items = B.common_items(report, B.provider_ids(report))
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())["holdout_windows"]}
    items = [it for it in items if it["item_id"] not in sealed]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} windows", flush=True)

    got: dict = {}
    out_p = timings_path()
    if out_p.exists():
        got = json.loads(out_p.read_text())
    todo = [(it["item_id"], [it["hyp"][p] for p in TRIO])
            for it in items if it["item_id"] not in got]
    print(f"{len(todo)} to align on {args.workers}x{args.threads}", flush=True)

    if todo:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(args.threads,)) as ex:
            for n, (wid, rec) in enumerate(ex.map(_one, todo, chunksize=1), 1):
                got[wid] = rec
                if n % 10 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_text(json.dumps(got, ensure_ascii=False))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(got, ensure_ascii=False))

    ok = sum(1 for v in got.values() if v.get("status") == "ok")
    bad: dict = {}
    for v in got.values():
        if v.get("status") != "ok":
            bad[v.get("status")] = bad.get(v.get("status"), 0) + 1
    print(f"-> {out_p}  ok={ok}/{len(got)}  other={bad}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
