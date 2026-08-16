#!/usr/bin/env python3
"""Call pyannoteAI precision-2 with transcription on 247 benchmark windows.

One `POST /v1/diarize` with `{"exclusive": true, "transcription": true}` returns
`diarization`, `exclusiveDiarization`, `wordLevelTranscription` and
`turnLevelTranscription` in a single job. The STT behind it is Nvidia
Parakeet-TDT-0.6b-v3, a transducer, which is why it is interesting here: it is
monotonic in time and structurally cannot skip audio the way an autoregressive
decoder can.

Sample: the 253 windows common to all 9 providers of the 2026-08-10 benchmark
report, MINUS the sealed holdout windows of the 2026-08 evaluation freeze.

Six of the seven sealed windows turn out to sit inside those 253 — the benchmark
run predates the freeze and covers them, and `exp-2026-08-16-fusion-deletions`
scored on all 253. This run does not: 247 windows are uploaded, the sealed six
are never sent and never scored. Every comparator (single systems, the trio vote)
is therefore recomputed on the same 247 rather than quoted from the 253-window
report, so nothing here is compared across two samples.

Responses land in `$SC/parakeet/<item_id>.json` and stay there. They carry
verbatim council speech, so they are cache, never git.

    SC=~/.cache/oc-public python3 eval/controlled_eval/parakeet_run.py --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
from eval.controlled_eval import bench_data as B                    # noqa: E402
from eval.controlled_eval.exclusive_diar_api import (               # noqa: E402
    api_key, log, run_one)

RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
HOLDOUT = ROOT / "research/eval-freeze-2026-08/manifest.json"


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def sealed_window_ids() -> set[str]:
    man = json.loads(HOLDOUT.read_text())
    return {w["window_id"] for w in man["holdout_windows"]}


def target_items() -> list[dict]:
    report = B.load_report(RUN_ID)
    items = B.common_items(report, B.provider_ids(report))
    sealed = sealed_window_ids()
    out = [it for it in items if it["item_id"] not in sealed]
    dropped = len(items) - len(out)
    if dropped:
        log(f"dropping {dropped} sealed holdout windows: {len(items)} -> {len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: only the first N windows")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--no-flac", dest="flac", action="store_false",
                    help="upload the raw wav instead of a lossless FLAC")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel jobs. Serial runs ~1 window/min (a 4.7 MB "
                         "upload plus polling), i.e. 4 h for the full sample.")
    args = ap.parse_args()

    key = api_key()
    wav_dir = sc() / "bench_windows"
    out_dir = sc() / "parakeet"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = target_items()
    if args.limit:
        items = items[:args.limit]

    missing = [it["item_id"] for it in items
               if not (wav_dir / f"{it['item_id']}.wav").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} windows have no local audio: {missing[:5]}")

    todo = [it for it in items if not (out_dir / f"{it['item_id']}.json").exists()]
    log(f"{len(items)} windows targeted, {len(todo)} still to call")

    failed = []

    def one(wid: str) -> tuple[str, str]:
        src = wav_dir / f"{wid}.wav"
        tmpdir = None
        try:
            if args.flac:
                # FLAC is lossless, so this changes the upload size and nothing the
                # model hears: ~4.5 MB of 16 kHz mono PCM becomes ~2.2 MB. The
                # upload dominates the wall clock, not the 10 s job.
                tmpdir = tempfile.mkdtemp(prefix="pk_")
                enc = Path(tmpdir) / f"{wid}.flac"
                p = subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", str(src),
                     "-c:a", "flac", str(enc)], capture_output=True, text=True,
                    timeout=args.timeout)
                if p.returncode != 0 or not enc.exists():
                    return wid, f"flac encode failed: {p.stderr[:120]}"
                src = enc
            r = run_one(src, key, wid, exclusive=True,
                        transcription=True, timeout=args.timeout)
        except Exception as e:                                   # noqa: BLE001
            return wid, f"error: {e}"
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
        if r.get("status") != "succeeded":
            return wid, str(r.get("status"))
        # write via a temp file so a killed run never leaves half a response behind
        tmp = out_dir / f".{wid}.part"
        tmp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out_dir / f"{wid}.json")
        return wid, "ok"

    ids = [it["item_id"] for it in todo]
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(one, ids))
    else:
        results = [one(w) for w in ids]

    for n, (wid, st) in enumerate(results, 1):
        if st != "ok":
            log(f"  [{n}/{len(ids)}] FAIL {wid}: {st}")
            failed.append(wid)
        else:
            log(f"  [{n}/{len(ids)}] ok {wid}")

    if failed:
        log(f"{len(failed)} failed: {failed}")
        sys.exit(1)
    log("done")


if __name__ == "__main__":
    main()
