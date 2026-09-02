"""Diarize whole training meeting recordings for the overlap-free clip census.

Frozen criteria: docs/specs/2026-08-19-overlap-clean-selection.md. One pyannoteAI
`precision-2` job per AUDIO FILE, non-exclusive so overlapping turns survive. Speaker
labels are local to their job and are never compared across files.

Audio and raw job output stay under ~/.cache, never in git.

    .venv-eval/bin/python -m eval.controlled_eval.diarize_training_meetings --per-city 1
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from eval.controlled_eval.exclusive_diar_api import api_key, diarize, upload, wait_job

TRAIN_PARQUET = Path.home() / ".cache/oc-public/train-screens-2026-08/run1-data/train.parquet"
ROOT = Path.home() / ".cache/oc-public/training-diarization-2026-08"
SEED = 20260819


def audio_files(parquet: Path):
    import pandas as pd
    frame = pd.read_parquet(parquet)
    frame["dur"] = frame.end - frame.start
    grouped = frame.groupby(["city_id", "meeting_id", "audio_url"]).agg(
        rows=("dur", "size"), speech_hours=("dur", "sum"), span=("end", "max")).reset_index()
    grouped["speech_hours"] = grouped.speech_hours / 3600
    return grouped


def stratified(grouped, per_city: int) -> list[dict]:
    """One (or more) audio file per city, drawn with a fixed seed. Not outcome-selected."""
    rng = random.Random(SEED)
    picked = []
    for city in sorted(grouped.city_id.unique()):
        rows = grouped[grouped.city_id == city].to_dict("records")
        rows.sort(key=lambda r: r["audio_url"])
        picked.extend(rng.sample(rows, min(per_city, len(rows))))
    return picked


def local_audio(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (hashlib.sha256(url.encode()).hexdigest()[:16] + Path(url).suffix)
    if target.is_file() and target.stat().st_size > 0:
        return target
    part = target.with_suffix(target.suffix + ".part")
    run = subprocess.run(["curl", "-sSL", "-o", str(part), url], capture_output=True, text=True)
    if run.returncode != 0 or not part.is_file() or part.stat().st_size == 0:
        raise RuntimeError(f"download failed for {url}: {run.stderr[:200]}")
    part.replace(target)
    return target


def job_path(root: Path, url: str) -> Path:
    return root / "jobs" / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".json")


def diarize_one(row: dict, key: str, root: Path, timeout: float) -> dict:
    out = job_path(root, row["audio_url"])
    if out.is_file():
        return {"audio_url": row["audio_url"], "status": "cached"}
    audio = local_audio(row["audio_url"], root / "audio")
    media = upload(audio, key, out.stem + audio.suffix)
    job = diarize(media, key, exclusive=False)
    started = time.time()
    result = wait_job(job, key, timeout=timeout)
    if result.get("status") != "succeeded":
        return {"audio_url": row["audio_url"], "status": result.get("status"), "job": job}
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_url": row["audio_url"],
        "city_id": row["city_id"],
        "meeting_id": row["meeting_id"],
        "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        "audio_bytes": audio.stat().st_size,
        "model": "precision-2",
        "exclusive": False,
        "job_id": job,
        "seconds": round(time.time() - started, 1),
        "output": result.get("output", result),
    }
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(out)
    return {"audio_url": row["audio_url"], "status": "succeeded",
            "seconds": payload["seconds"], "job": job}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=TRAIN_PARQUET)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--per-city", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--all", action="store_true", help="every audio file, not a stratified draw")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    grouped = audio_files(args.parquet)
    picked = (sorted(grouped.to_dict("records"), key=lambda r: r["audio_url"])
              if args.all else stratified(grouped, args.per_city))
    if args.limit:
        picked = picked[:args.limit]
    key = api_key()
    results = []

    def run(row):
        try:
            return row, diarize_one(row, key, args.root, args.timeout)
        except Exception as error:                                  # noqa: BLE001
            return row, {"status": f"error:{type(error).__name__}", "detail": str(error)[:200]}

    with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for row, result in pool.map(run, picked):
            results.append(result)
            print(json.dumps({"city": row["city_id"], "meeting": row["meeting_id"],
                              "rows": int(row["rows"]), **{k: v for k, v in result.items()
                                                           if k != "audio_url"}},
                             ensure_ascii=False), flush=True)
    print(json.dumps({"requested": len(picked),
                      "succeeded": sum(r["status"] in ("succeeded", "cached") for r in results)}))


if __name__ == "__main__":
    main()
