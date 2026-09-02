"""Fetch full OpenCouncil transcripts for the training meetings.

The training parquet holds a ~10% sample of each meeting (median gap between consecutive
rows 15.7 s), so no natural same-speaker run reaches a useful pack length. The public API
serves the whole transcript — every utterance with timestamps, plus the speaker segment
it belongs to — which is what the overlap-free packing arm needs.

Transcript text and speaker identity are PII under the same rule as the rest of the
repo: everything lands under ~/.cache, never in git.

    .venv-eval/bin/python -m eval.controlled_eval.fetch_training_transcripts
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

API = "https://opencouncil.gr/api"
TRAIN_PARQUET = Path.home() / ".cache/oc-public/train-screens-2026-08/run1-data/train.parquet"
ROOT = Path.home() / ".cache/oc-public/training-transcripts-2026-08"


def training_meetings(parquet: Path) -> list[dict]:
    import pandas as pd
    frame = pd.read_parquet(parquet)
    frame["dur"] = frame.end - frame.start
    grouped = frame.groupby(["city_id", "meeting_id", "audio_url"]).agg(
        sampled_rows=("dur", "size")).reset_index()
    return grouped.to_dict("records")


def fetch_one(city: str, meeting: str, out: Path, timeout: int = 90) -> dict:
    target = out / f"{city}__{meeting}.json"
    if target.is_file() and target.stat().st_size > 0:
        return {"city": city, "meeting": meeting, "status": "cached"}
    url = f"{API}/cities/{city}/meetings/{meeting}"
    run = subprocess.run(["curl", "-sS", "-m", str(timeout), "-w", "%{http_code}",
                          "-o", str(target.with_suffix(".part")), url],
                         capture_output=True, text=True)
    code = run.stdout.strip()[-3:]
    part = target.with_suffix(".part")
    if run.returncode != 0 or code != "200" or not part.is_file():
        part.unlink(missing_ok=True)
        return {"city": city, "meeting": meeting, "status": f"http_{code or 'error'}"}
    try:
        payload = json.loads(part.read_text())
        segments = payload["transcript"]
    except (json.JSONDecodeError, KeyError, TypeError):
        part.unlink(missing_ok=True)
        return {"city": city, "meeting": meeting, "status": "bad_payload"}
    part.replace(target)
    utterances = sum(len(s.get("utterances", [])) for s in segments)
    return {"city": city, "meeting": meeting, "status": "ok",
            "segments": len(segments), "utterances": utterances}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=TRAIN_PARQUET)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meetings = training_meetings(args.parquet)
    if args.limit:
        meetings = meetings[:args.limit]
    results = []
    for row in meetings:
        result = fetch_one(row["city_id"], row["meeting_id"], args.out)
        results.append(result)
        if result["status"] not in ("ok", "cached"):
            print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["status"] == "ok":
            time.sleep(args.sleep)
    ok = sum(r["status"] in ("ok", "cached") for r in results)
    print(json.dumps({"requested": len(results), "available": ok,
                      "failed": len(results) - ok,
                      "utterances": sum(r.get("utterances", 0) for r in results)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
