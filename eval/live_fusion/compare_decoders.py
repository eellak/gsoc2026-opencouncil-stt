# -*- coding: utf-8 -*-
"""Did removing the temperature ladder cost anything?

`temperature=0.0` made the decoder deterministic and roughly ten times faster on
GPU. The ladder it removed existed to retry segments that trip the
log-probability and compression-ratio thresholds, so the open question is
whether what it was rescuing was speech or degenerate repetition.

The comparison only means something against a yardstick: the ladder disagrees
with ITSELF by ~10% self-WER per window, so a difference smaller than that is
not evidence of anything. Pass a second ladder draw as the third argument and
the same statistics are computed between the two draws — that row is the noise
floor, and the ladder-vs-zero row has to clear it.

Scribe and Soniox are read from cache in every condition, so only `ours` differs.
Their rows are printed as a control: if they move, the comparison is broken.

    python3 -m eval.live_fusion.compare_decoders ladder.json zero.json [draw2.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ["scribe", "soniox", "ours", "W", "rules_on"]


def totals(rows, arm):
    s = d = i = n = 0
    for row in rows:
        s += row[arm]["s"]; d += row[arm]["d"]; i += row[arm]["i"]; n += row[arm]["n"]
    return s, d, i, n


def wer(rows, arm):
    s, d, i, n = totals(rows, arm)
    return ((s + d + i) / n, d / n) if n else (None, None)


def win_tie_loss(a, b, ids, arm):
    """Per window: does `a` have fewer errors than `b` on this arm?"""
    w = t = l = 0
    for wid in ids:
        ea = sum(a[wid][arm][k] for k in "sdi")
        eb = sum(b[wid][arm][k] for k in "sdi")
        if ea < eb:
            w += 1
        elif ea == eb:
            t += 1
        else:
            l += 1
    return w, t, l


def report(label, a, b, ids):
    rows_a = [a[w] for w in ids]
    rows_b = [b[w] for w in ids]
    print(f"\n--- {label} ({len(ids)} windows) ---")
    print(f"{'arm':10s} {'A wer':>9s} {'B wer':>9s} {'delta':>9s}   {'A del':>8s} {'B del':>8s}   W/T/L")
    for arm in ARMS:
        wa, da = wer(rows_a, arm)
        wb, db = wer(rows_b, arm)
        w, t, l = win_tie_loss(a, b, ids, arm)
        print(f"{arm:10s} {wa:9.5f} {wb:9.5f} {wb - wa:+9.5f}   {da:8.5f} {db:8.5f}   {w}/{t}/{l}")
    for arm in ("scribe", "soniox"):
        wa, _ = wer(rows_a, arm)
        wb, _ = wer(rows_b, arm)
        if abs(wa - wb) > 1e-9:
            print(f"  CONTROL BROKEN: {arm} moved ({wa:.5f} -> {wb:.5f}); only `ours` should differ")
    # Words emitted is the crude version of "what did the ladder rescue".
    for name, rows in (("A", rows_a), ("B", rows_b)):
        words = sum(r["words"]["ours"] for r in rows if "words" in r)
        if words:
            print(f"  {name} ours words emitted: {words}")


def main(argv):
    if len(argv) not in (3, 4):
        print("usage: compare_decoders.py ladder.json zero.json [ladder_draw2.json]")
        return 2
    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in argv[1:]]
    for path, d in zip(argv[1:], loaded):
        policy = d["stack"].get("ours_temperature_policy", "ladder (pre-flag run)")
        draw = d["stack"].get("ours_draw", 0)
        print(f"{Path(path).name}: policy={policy} draw={draw} "
              f"stack={d['stack'].get('ours_stack', 'local')} "
              f"scored={d['sample']['n_scored']}")

    ladder, zero = loaded[0]["per_window"], loaded[1]["per_window"]
    shared = sorted(set(ladder) & set(zero))
    report("ladder (A) vs temperature=0.0 (B)", ladder, zero, shared)

    if len(loaded) == 3:
        draw2 = loaded[2]["per_window"]
        noise_ids = sorted(set(ladder) & set(draw2))
        report("ladder draw 1 (A) vs ladder draw 2 (B) — THE NOISE FLOOR",
               ladder, draw2, noise_ids)
        print("\nRead the second block first. Any ladder-vs-zero difference smaller than"
              "\nthe draw-to-draw difference is the decoder talking to itself, not a finding.")
    else:
        print("\nNo second ladder draw supplied, so there is no noise floor to read the"
              "\nfirst block against. Do not call any difference above a finding yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
