"""Cut the frozen-manifest windows whose audio is not in the local bench cache.

`artifact-bench-240-audio` covers 240 of the benchmark's 260 windows: the other 20
were skipped because the whole meeting mp3 had never been downloaded. Eight of those
20 are in the 2026-08 freeze (one eval window, all seven holdout windows), so they
would silently drop out of every arm.

data.opencouncil.gr serves HTTP range requests, so ffmpeg can seek straight to the
window and pull ~2 minutes instead of a multi-hour recording - the same trick
`scripts/fetch_clip.py` uses for dataset rows. The audio URL comes from the meeting
JSON already cached under `$SC/meetings/`.

Audio stays under `$SC`. Never in git.

    SC=~/.cache/oc-public .venv-eval/bin/python -m eval.controlled_eval.cut_freeze_audio
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def audio_url(city: str, meeting: str) -> str:
    p = sc() / "meetings" / f"{city}__{meeting}.json"
    if not p.exists():
        raise SystemExit(f"no cached meeting json for {city}__{meeting}")
    url = json.loads(p.read_text())["meeting"].get("audioUrl")
    if not url:
        raise SystemExit(f"{city}__{meeting}: meeting json carries no audioUrl")
    return url


def main() -> None:
    man = json.loads(MANIFEST.read_text())
    out = sc() / "bench_windows"
    out.mkdir(parents=True, exist_ok=True)

    todo = [r for r in man["eval_windows"] + man["holdout_windows"]
            if not (out / f"{r['window_id']}.wav").exists()]
    print(f"{len(todo)} windows to fetch")

    failed = []
    for r in todo:
        dst = out / f"{r['window_id']}.wav"
        url = audio_url(r["city"], r["meeting_id"])
        # -ss before -i so ffmpeg range-seeks the remote file instead of decoding
        # everything up to the offset.
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-ss", f"{r['start_sec']:.3f}", "-t", f"{r['duration_sec']:.3f}",
               "-i", url, "-ac", "1", "-ar", "16000", str(dst)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        ok = p.returncode == 0 and dst.exists() and dst.stat().st_size > 100_000
        print(f"  {'ok  ' if ok else 'FAIL'} {r['window_id']} "
              f"{dst.stat().st_size if dst.exists() else 0} bytes")
        if not ok:
            failed.append(r["window_id"])
            print(p.stderr.strip()[:400])

    if failed:
        sys.exit(f"{len(failed)} windows could not be cut: {failed}")
    print("all frozen windows have local audio")


if __name__ == "__main__":
    main()
