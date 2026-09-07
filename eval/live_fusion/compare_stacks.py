# -*- coding: utf-8 -*-
"""Does the serving stack move the fused number?

Two runs of the same windows differ in one thing: `ours` came from CPU int8 on
this machine in one and from the float16 GPU endpoint production calls in the
other. Everything else -- audio, Scribe, Soniox, fusion, scorer -- is the same.

Only the windows BOTH runs scored are compared. Comparing 40 windows against 37
would price the missing three as a stack difference.

Usage: python3 -m eval.live_fusion.compare_stacks a.json b.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ["scribe", "soniox", "ours", "W", "rules_on"]


def wer(rows, arm):
    s = d = i = n = 0
    for row in rows:
        s += row[arm]["s"]; d += row[arm]["d"]; i += row[arm]["i"]; n += row[arm]["n"]
    return (s + d + i) / n if n else None, d / n if n else None


def main(argv):
    if len(argv) != 3:
        print("usage: compare_stacks.py a.json b.json")
        return 2
    a = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    b = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    shared = sorted(set(a["per_window"]) & set(b["per_window"]))
    if not shared:
        print("no shared windows")
        return 1

    rows_a = [a["per_window"][w] for w in shared]
    rows_b = [b["per_window"][w] for w in shared]
    print(f"{len(shared)} windows scored by both "
          f"({a['sample']['n_scored']} and {b['sample']['n_scored']} scored in total)")
    # Runs written before the field existed are the local stack by construction.
    label_a = a["stack"].get("ours_stack", "local")
    label_b = b["stack"].get("ours_stack", "local")
    print(f"{'arm':10s} {label_a:>10s} {label_b:>10s}   delta")
    for arm in ARMS:
        wa, _ = wer(rows_a, arm)
        wb, _ = wer(rows_b, arm)
        print(f"{arm:10s} {wa:10.5f} {wb:10.5f}   {wb - wa:+.5f}")
    print()
    print("deletion rate")
    for arm in ARMS:
        _, da = wer(rows_a, arm)
        _, db = wer(rows_b, arm)
        print(f"{arm:10s} {da:10.5f} {db:10.5f}   {db - da:+.5f}")
    # Scribe and Soniox are the same recordings in both runs, so any difference
    # in those two rows is a bug in this comparison, not a stack effect.
    for arm in ("scribe", "soniox"):
        wa, _ = wer(rows_a, arm)
        wb, _ = wer(rows_b, arm)
        if abs(wa - wb) > 1e-9:
            print(f"WARNING: {arm} differs between the runs ({wa:.5f} vs {wb:.5f}); "
                  "these should be the same cached responses")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
