#!/usr/bin/env python3
"""Run pyannoteAI precision-2 over all 232 benchmark windows.

Codex reviewed the plan at high effort and rejected the framing this started with. This
is NOT an independent prevalence estimate and the output must never be described as one:
precision-2 sees the same audio, comes from the same model family, and the listening
audit showed both diarizers share the same blind spot — they count miked speakers while
the human counted the room. Two models with one blind spot agreeing about zeros is weak
evidence about recall.

What the pass IS worth doing for:
  * paired measurement-robustness of the bucket association (`precision2_analyze.py`);
  * a disagreement map — the windows where the two detectors differ are where a further
    human audit buys the most;
  * event geometry, as a sensitivity grid for the synthetic experiment's dose, NOT a
    replacement for its preregistered uniform prior.

Raw output, model id, parameters and request metadata are all stored, because anything
conceived after looking at this data is exploratory by construction and has to be
labelled that way later.

Usage:
  SC=~/.cache/oc-overlap python eval/controlled_eval/precision2_corpus.py
Env: PYANNOTE_API_KEY  SC  MODEL  N_ITEMS  CONCURRENCY
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as B  # noqa: E402
import precision2_compare as P  # noqa: E402

ROOT = Path("/home/harold/opencouncil-fine-tuning")
AUDIO = ROOT / "data/asr/audio"
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
OUT = SC / "precision2_corpus.json"
MODEL = os.environ.get("MODEL", "precision-2")
N_ITEMS = int(os.environ.get("N_ITEMS", "0"))
CONC = int(os.environ.get("CONCURRENCY", "4"))

_lock = threading.Lock()


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def slice_wav(src: Path, start: float, dur: float, dst: Path) -> float:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)], check=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(dst)], capture_output=True, text=True,
                       check=True)
    return float(r.stdout.strip())


def one(it, key) -> tuple[str, dict]:
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "w.wav"
        dur = slice_wav(it["_audio"], it["_start"], it["_dur"], w)
        try:
            media = P.upload(w, key, f"corpus/{it['item_id']}")
            body = {"url": media, "model": MODEL, "confidence": True, "exclusive": True}
            r = P.curl(["-X", "POST", f"{P.API}/diarize",
                        "-H", f"Authorization: Bearer {key}",
                        "-H", "Content-Type: application/json", "-d", json.dumps(body)])
            if "jobId" not in r:
                return it["item_id"], {"error": f"no jobId: {r}"}
            job = P.wait_job(r["jobId"], key, timeout=900)
        except RuntimeError as e:
            return it["item_id"], {"error": str(e)[:300]}
    if job.get("status") != "succeeded":
        return it["item_id"], {"error": job.get("status"), "raw": job}
    out = job.get("output", {})
    turns = out.get("diarization", [])
    return it["item_id"], {
        "measured_dur": dur,
        "feat": P.overlap_of(turns, dur),
        "turns": turns,
        "exclusive": out.get("exclusiveDiarization"),
        "model": MODEL,
    }


def main():
    key = P.api_key()
    report = B.load_report()
    providers = B.provider_ids(report)
    items = B.common_items(report, providers)
    by_id = {it["itemId"]: it for it in report["items"]}

    kept = []
    for it in items:
        p = AUDIO / f"{it['city_id']}__{it['meeting_id']}.mp3"
        if not p.exists():
            continue
        raw = by_id[it["item_id"]]
        it["_audio"], it["_start"], it["_dur"] = p, raw["startSec"], raw["durationSec"]
        kept.append(it)
    if N_ITEMS:
        kept = kept[:N_ITEMS]

    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [it for it in kept if it["item_id"] not in done]
    audio_h = sum(it["_dur"] for it in todo) / 3600
    log(f"{len(kept)} windows, {len(done)} cached, {len(todo)} to run "
        f"({audio_h:.2f} h of audio), model={MODEL}, concurrency={CONC}")

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = {ex.submit(one, it, key): it["item_id"] for it in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            iid, res = f.result()
            with _lock:
                done[iid] = res
                tmp = OUT.with_suffix(".part")
                tmp.write_text(json.dumps(done, ensure_ascii=False))
                tmp.replace(OUT)
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                log(f"  {i}/{len(todo)} ({el / i:.1f}s each, "
                    f"eta {(len(todo) - i) * el / i / 60:.0f}min)")

    err = {k: v for k, v in done.items() if "feat" not in v}
    log(f"done: {len(done) - len(err)} ok, {len(err)} failed -> {OUT}")
    for k, v in list(err.items())[:5]:
        log(f"  {k}: {str(v)[:160]}")


if __name__ == "__main__":
    main()
