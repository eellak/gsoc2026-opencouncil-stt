#!/usr/bin/env python3
"""Transcribe the frozen gold-set clips with the two ASR sources reachable
without a GPU: the local corrected adapter (artifact-ct2-fixed, CPU int8) and
Soniox (realtime WS, text only).

Writes JSON per cell under ~/.cache/oc-public/gold-set-2026-08/hyp/<system>/.
Never writes transcript text into the repo.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

CLIPS = Path.home() / "oc-gold-set" / "clips"
OUT = Path.home() / ".cache/oc-public/gold-set-2026-08/hyp"
CELLS = json.loads((Path.home() / ".cache/oc-public/gold-set-2026-08/cells-frozen.json").read_text())
IDS = [c["id"] for c in CELLS["cells"]]
KEY = (Path.home() / "oc-asr-serve" / "api_key.txt").read_text().strip()
SONIOX_DIR = Path("/home/harold/projects/soniox-tools")


def adapter(cid: str) -> dict:
    p = CLIPS / f"{cid}.wav"
    r = subprocess.run(
        ["curl", "-s", "-m", "300", "-X", "POST", "http://localhost:8000/transcribe/upload",
         "-H", f"x-api-key: {KEY}", "-F", f"file=@{p}", "-F", "language=el"],
        capture_output=True, text=True, timeout=320)
    return json.loads(r.stdout)


def soniox(cid: str) -> dict:
    p = CLIPS / f"{cid}.wav"
    r = subprocess.run(
        [str(SONIOX_DIR / ".venv/bin/python"), "file_transcribe.py", str(p), "--lang", "el"],
        capture_output=True, text=True, timeout=300, cwd=str(SONIOX_DIR), errors="replace")
    out = r.stdout
    mark = "===== TRANSCRIPT ====="
    text = out.split(mark, 1)[1].strip() if mark in out else ""
    return {"text": text, "ok": bool(text)}


def main():
    which = sys.argv[1]
    fn = {"adapter": adapter, "soniox": soniox}[which]
    d = OUT / which
    d.mkdir(parents=True, exist_ok=True)
    for cid in IDS:
        f = d / f"{cid}.json"
        if f.exists():
            continue
        t0 = time.time()
        try:
            res = fn(cid)
        except Exception as e:                                   # noqa: BLE001
            print(f"FAIL {cid}: {e}", flush=True)
            continue
        f.write_text(json.dumps(res, ensure_ascii=False))
        print(f"{which} {cid} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
