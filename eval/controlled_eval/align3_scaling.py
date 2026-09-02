#!/usr/bin/env python3
"""Measure align3 wall time and peak memory as the input grows toward a 15-minute segment.

WHY. Every fusion number this project holds was measured on ~143 s benchmark windows,
which are ~310 tokens per system. Production hands the pipeline 15-minute segments,
~2,500 tokens per system, and `align3` has never been run at that size (spec
2026-08-26-fusion-integration, risk 1). If it does not fit, chunking is not an escape
hatch that gets bolted on after a production timeout - it is part of the system that
has to be measured before anything else is built.

INPUT IS REAL. Sequences are built by concatenating consecutive windows from the SAME
meeting in the atlas cache, so the disagreement structure between the three systems is
the real one at every length. Synthetic repetition would understate the band.

Each size runs in its own subprocess under an address-space limit, so an input that
does not fit reports MemoryError instead of taking the machine down with it.

    python3 eval/controlled_eval/align3_scaling.py
    SIZES=400,1000 MEM_GB=8 python3 eval/controlled_eval/align3_scaling.py
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ATLAS = Path(os.environ.get(
    "ATLAS", Path.home() / ".cache/oc-public/fusion_lab/atlas_b85d8064a16e0dd1.json"))
SIZES = [int(x) for x in os.environ.get("SIZES", "400,1000,2500,4000").split(",")]
MEM_GB = float(os.environ.get("MEM_GB", "12"))


def sequences(target: int):
    """Three token lists of ~target length, from consecutive windows of one meeting."""
    d = json.loads(ATLAS.read_text())
    by = defaultdict(list)
    for w in d["windows"]:
        by[(w["city"], w["meeting"])].append(w)
    for key in sorted(by, key=lambda k: -len(by[k])):
        wins = by[key]
        out = [[], [], []]
        for w in wins:
            for k in range(3):
                out[k] += w["hyps"][k]
            if min(len(x) for x in out) >= target:
                return [x[:target] for x in out], key
    return None, None


def child(target: int):
    lim = int(MEM_GB * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    from eval.controlled_eval.msa import align3
    seqs, key = sequences(target)
    if seqs is None:
        print(json.dumps({"n": target, "status": "no_meeting_long_enough"}))
        return
    band = max(40, max(len(s) for s in seqs) - min(len(s) for s in seqs) + 20)
    t0 = time.perf_counter()
    try:
        cols = align3(*seqs, band=band)
    except MemoryError:
        print(json.dumps({"n": target, "status": "MemoryError", "band": band,
                          "sec": round(time.perf_counter() - t0, 2),
                          "peak_rss_mb": round(resource.getrusage(
                              resource.RUSAGE_SELF).ru_maxrss / 1024)}))
        return
    print(json.dumps({
        "n": target, "status": "ok", "band": band, "cols": len(cols),
        "sec": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "meeting": "/".join(key)}))


def main():
    print(f"align3 scaling, address-space limit {MEM_GB} GB per run", flush=True)
    print(f"{'tokens':>7} {'band':>6} {'sec':>9} {'peak RSS':>10}  status", flush=True)
    rows = []
    for n in SIZES:
        p = subprocess.run([sys.executable, __file__, "--child", str(n)],
                           capture_output=True, text=True, cwd=ROOT)
        line = (p.stdout or "").strip().splitlines()
        try:
            r = json.loads(line[-1])
        except Exception:
            r = {"n": n, "status": f"killed rc={p.returncode}",
                 "stderr": (p.stderr or "")[-200:]}
        rows.append(r)
        print(f"{r['n']:>7} {r.get('band', '-'):>6} {str(r.get('sec', '-')):>9} "
              f"{str(r.get('peak_rss_mb', '-')) + ' MB':>10}  {r['status']}", flush=True)
    out = ROOT / "eval/results_align3_scaling.json"
    out.write_text(json.dumps({"mem_gb": MEM_GB, "atlas": ATLAS.name, "rows": rows},
                              ensure_ascii=False, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(int(sys.argv[2]))
    else:
        main()
