#!/usr/bin/env python3
"""Timestamp completeness and boundary error, as frozen in the preregistration.

Two separate things, and they are separate on purpose.

**Completeness.** Does the model emit a syntactically valid, monotonic sequence of segment
times at all? This is the failure the literature describes: a model fine-tuned only on
short padded clips forgets that timestamps exist. A model that emits nothing usable fails
here no matter how good its remaining boundaries look.

**Boundary error.** Mean absolute error against speech boundaries from an independent
source. The trap, and the reason the frozen procedure is what it is: scoring only the
boundaries that happened to match rewards a model for emitting almost none. So unmatched
boundaries on either side take a fixed 2-second penalty rather than being dropped.

The independent source here is Silero VAD, and the circularity warning in the
preregistration applies to arm B only, which is built with silence-snapped boundaries. For
the preflight, neither system was trained with any timestamp supervision at all, so nothing
being compared has seen this teacher. Version and parameters are pinned below and any
result quoting this number has to say which they were.

  .venv-eval/bin/python -m eval.controlled_eval.timestamp_quality \
      --set ~/oc-longform --tags base finetune --json results_longform_ts.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Frozen VAD post-processing, from the preregistration.
MERGE_GAP = 0.5        # islands closer than this are one island
MIN_ISLAND = 0.2       # shorter than this is not speech
MATCH_WINDOW = 2.0     # a boundary further than this is unmatched
PENALTY = 2.0          # what an unmatched boundary costs, either side
VAD_THRESHOLD = 0.5


def islands(wav: Path) -> list[tuple[float, float]]:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    model = load_silero_vad()
    ts = get_speech_timestamps(read_audio(str(wav), sampling_rate=16000), model,
                               sampling_rate=16000, threshold=VAD_THRESHOLD,
                               return_seconds=True)
    out: list[list[float]] = []
    for t in ts:
        s, e = float(t["start"]), float(t["end"])
        if out and s - out[-1][1] < MERGE_GAP:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= MIN_ISLAND]


def match(pred: list[float], ref: list[float]) -> tuple[float, int, int]:
    """Greedy one-to-one nearest matching. Returns (error sum, matched, unmatched)."""
    used, err, matched = set(), 0.0, 0
    for p in sorted(pred):
        best, bd = None, MATCH_WINDOW
        for i, r in enumerate(ref):
            if i in used:
                continue
            if abs(p - r) <= bd:
                best, bd = i, abs(p - r)
        if best is None:
            err += PENALTY
        else:
            used.add(best)
            err += bd
            matched += 1
    err += PENALTY * (len(ref) - len(used))          # reference boundaries never covered
    return err, matched, (len(pred) - matched) + (len(ref) - len(used))


def valid(segs: list[dict]) -> bool:
    """Monotonic, non-negative, non-inverted."""
    last = -1.0
    for s in segs:
        a, b = float(s["start"]), float(s["end"])
        if a < last or b < a or a < 0:
            return False
        last = a
    return bool(segs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()

    root = Path(a.set).expanduser()
    manifest = json.loads((root / "manifest.json").read_text())

    cache_path = root / "vad_islands.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for r in manifest:
        if r["wav"] not in cache:
            cache[r["wav"]] = islands(root / "audio" / r["wav"])
            cache_path.write_text(json.dumps(cache))
            print(f"  VAD {r['wav']}: {len(cache[r['wav']])} islands", flush=True)

    out = {"vad": {"threshold": VAD_THRESHOLD, "merge_gap": MERGE_GAP,
                   "min_island": MIN_ISLAND, "match_window": MATCH_WINDOW,
                   "penalty": PENALTY},
           "systems": {}}

    for tag in a.tags:
        p = root / f"hyp_{tag}.json"
        if not p.exists():
            print(f"{tag}: no hypotheses yet, skipped")
            continue
        hyps = json.loads(p.read_text())
        rows = []
        for r in manifest:
            h = hyps.get(r["wav"])
            if not h:
                continue
            segs = h["segments"]
            ref = cache[r["wav"]]
            ref_b = [s for s, _ in ref] + [e for _, e in ref]
            pred_b = [float(s["start"]) for s in segs] + [float(s["end"]) for s in segs]
            err, matched, unmatched = match(pred_b, ref_b)
            rows.append({"wav": r["wav"], "city_id": r["city_id"],
                         "meeting_id": r["meeting_id"], "segments": len(segs),
                         "valid": valid(segs),
                         "mae": round(err / max(1, len(pred_b) + (len(ref_b) - matched)), 3),
                         "matched": matched, "ref_boundaries": len(ref_b),
                         "coverage": round(matched / max(1, len(ref_b)), 3)})
        n = len(rows)
        out["systems"][tag] = {
            "spans": n,
            "valid_share": round(sum(1 for r in rows if r["valid"]) / max(1, n), 4),
            "coverage": round(sum(r["matched"] for r in rows)
                              / max(1, sum(r["ref_boundaries"] for r in rows)), 4),
            "mae": round(sum(r["mae"] for r in rows) / max(1, n), 3),
            "by_span": rows,
        }
        s = out["systems"][tag]
        print(f"{tag}: {s['spans']} spans, valid {s['valid_share']:.1%}, "
              f"coverage {s['coverage']:.1%}, MAE {s['mae']}s")

    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
