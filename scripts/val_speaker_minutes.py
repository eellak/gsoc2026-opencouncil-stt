#!/usr/bin/env python3
"""Per-speaker speaking-minute distribution for the held-out validation cities.

Reads the cached meeting JSON under ui/.cache/meetings/{city}__*.json and sums
utterance durations grouped by speaker (speakerTag.personId, falling back to the
speakerTagId for un-named diarization tags). Used to set, from the data, the
validation speaker-selection thresholds:

  - reliability floor (min minutes/speaker for a stable per-speaker WER)
  - budget + per-speaker cap (so one dominant speaker can't own the val set)

See data/reports/finetune-research/val-split-whole-city-vs-stratified.md.

Usage:
    python3 scripts/val_speaker_minutes.py [city ...]   # default: orestiada argos
"""
import json
import glob
import sys
import collections

CACHE = "ui/.cache/meetings/{city}__*.json"
DEFAULT_CITIES = ("orestiada", "argos")
BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]


def utt_seconds(u):
    try:
        d = float(u["endTimestamp"]) - float(u["startTimestamp"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return d if 0 < d < 3600 else 0.0   # guard against bad/zero/huge spans


def speaker_minutes(city):
    """personId(or TAG:tagId) -> total speaking minutes across cached meetings."""
    secs = collections.defaultdict(float)
    files = glob.glob(CACHE.format(city=city))
    for f in files:
        d = json.load(open(f))
        for seg in (d.get("transcript") or []):
            tag = seg.get("speakerTag") or {}
            pid = tag.get("personId") or f"TAG:{seg.get('speakerTagId')}"
            for u in (seg.get("utterances") or []):
                secs[pid] += utt_seconds(u)
    return {k: v / 60 for k, v in secs.items()}, len(files)


def main(cities):
    for city in cities:
        mins_by_spk, n_meetings = speaker_minutes(city)
        mins = sorted(mins_by_spk.values(), reverse=True)
        named = sum(1 for k in mins_by_spk if not k.startswith("TAG:"))
        print(f"\n===== {city.upper()} =====")
        print(f"meetings: {n_meetings}   speakers: {len(mins)} "
              f"(named {named}, unnamed tags {len(mins) - named})")
        print(f"total speech: {sum(mins):.0f} min")
        print("per-speaker minute distribution:")
        for lo, hi in BUCKETS:
            n = sum(1 for m in mins if lo <= m < hi)
            cum = sum(m for m in mins if m >= lo)
            lab = f"{lo:>2.0f}-{hi:<3.0f}" if hi < 1e8 else f"{lo:>2.0f}+   "
            print(f"  {lab} min : {n:>3} spk | cum speech of spk >= {lo}min: {cum:6.0f} min")
        print("top 15 (min):", [f"{m:.1f}" for m in mins[:15]])


if __name__ == "__main__":
    main(sys.argv[1:] or list(DEFAULT_CITIES))
