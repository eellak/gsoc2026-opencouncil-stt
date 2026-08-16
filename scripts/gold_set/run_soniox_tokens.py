#!/usr/bin/env python3
"""Re-transcribe the 30 frozen gold-set clips through the SAME Soniox realtime
path used to build `hyp/soniox/` (model stt-rt-v4, free Perplexity temp key),
this time keeping the per-token `confidence` / `start_ms` / `end_ms` / `is_final`
that `file_transcribe.py --json` now preserves.

The 247 benchmark windows are NOT touched: those were produced with the paid
`stt-async-v5` and are one third of the frozen fusion input W.

Writes raw token JSON per cell under
  ~/.cache/oc-public/gold-set-2026-08/hyp/soniox-tokens/
Never into git.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CLIPS = Path.home() / "oc-gold-set" / "clips"
GOLD = Path.home() / ".cache/oc-public/gold-set-2026-08"
OUT = GOLD / "hyp" / "soniox-tokens"
SONIOX_DIR = Path(os.environ.get("SONIOX_TOOLS_DIR",
                                 Path.home() / "projects" / "soniox-tools"))
PY = Path(os.environ.get("SONIOX_TOOLS_PYTHON", SONIOX_DIR / ".venv/bin/python"))


def transcribe(cid: str) -> dict:
    p = CLIPS / f"{cid}.wav"
    r = subprocess.run(
        [str(PY), "file_transcribe.py", str(p), "--lang", "el", "--json"],
        capture_output=True, text=True, timeout=300, cwd=str(SONIOX_DIR),
        errors="replace")
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"rc={r.returncode} stderr={r.stderr[-400:]!r}")
    return json.loads(r.stdout)


def main() -> None:
    if not PY.exists():
        sys.exit(f"no soniox-tools interpreter at {PY} — set SONIOX_TOOLS_DIR "
                 f"or SONIOX_TOOLS_PYTHON")
    ids = [c["id"] for c in json.loads((GOLD / "cells-frozen.json").read_text())["cells"]]
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for cid in ids:
        f = OUT / f"{cid}.json"
        if f.exists():
            continue
        t0 = time.time()
        last = None
        for attempt in (1, 2):
            try:
                d = transcribe(cid)
                # the temp key expires roughly hourly; resolve_api_key() re-mints
                # per process, so a retry is enough to ride through an expiry.
                f.write_text(json.dumps(d, ensure_ascii=False))
                print(f"ok {cid} {len(d['tokens'])} tok {time.time()-t0:.1f}s", flush=True)
                break
            except Exception as e:                                # noqa: BLE001
                last = e
                print(f"retry {cid} (attempt {attempt}): {e}", flush=True)
                time.sleep(5)
        else:
            failed.append((cid, str(last)))
            print(f"FAIL {cid}: {last}", flush=True)
    print(f"done. {len(list(OUT.glob('*.json')))}/{len(ids)} cells, {len(failed)} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
