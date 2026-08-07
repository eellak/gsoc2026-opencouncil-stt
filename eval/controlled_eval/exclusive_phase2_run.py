#!/usr/bin/env python3
"""Phase 2 diarization: the 25 frozen windows, one `exclusive: true` call each.

One call returns both `diarization` (status quo) and `exclusiveDiarization`
(proposal), which keeps the two variants paired on the same job — the invariance
of the regular timeline under the flag is what Phase 1 measures over 95 items.

Only runs after the Phase 1 gate passes; the script checks the Phase 1 result file
and refuses otherwise.

Usage: python eval/controlled_eval/exclusive_phase2_run.py [--force]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import SC, api_key, log, run_one  # noqa: E402

WINDOWS = SC / "exclusive_phase2_windows.json"
WAV = SC / "winwav"
OUT = SC / "exclusive_phase2_diar.json"
PHASE1 = Path(__file__).resolve().parent / "results_exclusive_phase1.json"


def main():
    if "--force" not in sys.argv:
        if not PHASE1.exists():
            log("Phase 1 result missing — the gate has not been evaluated.")
            sys.exit(2)
        if not json.loads(PHASE1.read_text()).get("gate_passed"):
            log("Phase 1 gate FAILED — the preregistration stops here. "
                "No Phase 2 spend.")
            sys.exit(1)

    key = api_key()
    windows = json.loads(WINDOWS.read_text())
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    log(f"{len(windows)} windows, {len(store)} already done")

    for i, w in enumerate(windows, 1):
        wid = w["window_id"]
        if wid in store and "error" not in store[wid]:
            continue
        path = WAV / f"{wid}.wav"
        if not path.exists():
            store[wid] = {"error": "missing_wav"}
            log(f"  {wid}: missing wav")
            continue
        try:
            r = run_one(path, key, f"excl_p2_{wid}", exclusive=True)
        except RuntimeError as e:
            store[wid] = {"error": str(e)[:300]}
            log(f"  {wid}: {e}")
        else:
            if r.get("status") != "succeeded":
                store[wid] = {"error": r.get("status", "unknown")}
                log(f"  {i}/{len(windows)} {wid}: FAILED {r.get('status')}")
            else:
                o = r.get("output") or {}
                store[wid] = {"diarization": o.get("diarization", []),
                              "exclusiveDiarization": o.get("exclusiveDiarization")}
                log(f"  {i}/{len(windows)} {wid}: "
                    f"{len(store[wid]['diarization'])} -> "
                    f"{len(store[wid]['exclusiveDiarization'] or [])} segments")
        OUT.write_text(json.dumps(store, ensure_ascii=False))

    ok = sum(1 for v in store.values() if "diarization" in v)
    log(f"\n{ok}/{len(windows)} ok -> {OUT}")


if __name__ == "__main__":
    main()
