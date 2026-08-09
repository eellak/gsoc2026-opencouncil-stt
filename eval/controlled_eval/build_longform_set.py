#!/usr/bin/env python3
"""Freeze the long-form evaluation set for the window-shape preflight.

The [preregistration](../../docs/specs/window-shape-preregistration.md) asks whether
training on short cut clips damages long-form behaviour. Every measurement this project has
made so far runs on 20-second windows, which is exactly the regime where a long-form
collapse cannot appear: a repetition loop needs room to run.

So this picks continuous spans of at least ten minutes, one per meeting, from the
development pool. Meetings, not spans, are the unit: loops are rare, and a bootstrap over
one meeting's worth of clusters says nothing. The 88 final-benchmark meetings are not
touched, and neither are the 16 locked windows.

The span is chosen deterministically and its reference text comes from the published
transcript, which is agreement-with-OpenCouncil and is named that way everywhere it is
used. Nothing here writes council text or audio into the repo.

  SC=~/.cache/oc-public .venv-eval/bin/python -m eval.controlled_eval.build_longform_set
Env: SC OUT N_MEETINGS SPAN_SEC
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
OUT = Path(os.environ.get("OUT", Path.home() / "oc-longform")).expanduser()
N_MEETINGS = int(os.environ.get("N_MEETINGS", "20"))
SPAN_SEC = float(os.environ.get("SPAN_SEC", "600"))
SKIP_HEAD = 300.0        # roll call and procedure, same margin as the reference sampler
SKIP_TAIL = 120.0
MIN_WORDS = 800          # a ten-minute span with less than this is mostly silence


def log(*a):
    print(*a, flush=True)


def h(*parts) -> int:
    return int.from_bytes(
        hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "big")


def utterances(rec):
    out = []
    for seg in rec.get("transcript") or []:
        for u in seg.get("utterances") or []:
            if u.get("startTimestamp") is None:
                continue
            out.append((float(u["startTimestamp"]), float(u["endTimestamp"]),
                        u.get("text") or ""))
    return sorted(out)


def dev_pool() -> set[tuple[str, str]]:
    """The development pool, which is the only thing allowed to be spent on diagnostics.

    reference_pool.csv is the burn registry: the original 48 reference meetings plus the 9
    May meetings the 2026-08-09 dev build consumed. The 88 final-benchmark meetings are
    everything else and are not in it.

    The 16 locked meetings come out too. Policy seals the locked *windows*, not the
    meetings, so ten minutes elsewhere in one of them is legal on a literal reading. It is
    still the wrong thing to do: the policy's own account of leakage is same speakers, same
    room, same recording conditions, and the locked split is the one thing in this project
    that has to stay unseen. Eight of the first twenty picks were locked meetings.
    """
    p = ROOT / "data/public-meetings/reference_pool.csv"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    locked = {(r["city_id"], r["meeting_id"]) for r in rows if r["split"] == "locked"}
    return {(r["city_id"], r["meeting_id"]) for r in rows} - locked


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit(f"{OUT} is not empty; refusing to rebuild a frozen set over itself")
    pool = dev_pool()
    log(f"{len(pool)} meetings in the development pool")

    cands = []
    for p in sorted((SC / "meetings").glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        m = d.get("meeting") or {}
        key = (m.get("cityId"), m.get("id"))
        if key not in pool or d.get("transcriptHiddenForReview") or not m.get("audioUrl"):
            continue
        utts = utterances(d)
        if not utts:
            continue
        end = max(u[1] for u in utts)
        if end - SKIP_HEAD - SKIP_TAIL < SPAN_SEC:
            continue
        # Deterministic span: walk a 60 s grid, keep the densest legal start for this
        # meeting under a fixed hash order, so a rerun picks the same ten minutes.
        starts = []
        t = SKIP_HEAD
        while t + SPAN_SEC <= end - SKIP_TAIL:
            words = sum(len((x[2] or "").split()) for x in utts
                        if x[1] > t and x[0] < t + SPAN_SEC)
            if words >= MIN_WORDS:
                starts.append((t, words))
            t += 60.0
        if not starts:
            continue
        t0, words = sorted(starts, key=lambda s: h("s", key[1], s[0]))[0]
        cands.append({"city_id": key[0], "meeting_id": key[1], "audio_url": m["audioUrl"],
                      "start_sec": t0, "dur_sec": SPAN_SEC, "ref_words": words,
                      "n_utterances": sum(1 for x in utts
                                          if x[1] > t0 and x[0] < t0 + SPAN_SEC)})

    log(f"{len(cands)} meetings with a usable {SPAN_SEC/60:.0f}-minute span")
    if len(cands) < N_MEETINGS:
        log(f"  SHORT: asked for {N_MEETINGS}, the pool yields {len(cands)}")

    # Spread over cities the same way the reference sampler does, so one city cannot
    # dominate a set whose whole point is meeting-level clustering.
    by_city: dict[str, list] = {}
    for c in cands:
        by_city.setdefault(c["city_id"], []).append(c)
    for city in by_city:
        by_city[city].sort(key=lambda c: h("m", c["meeting_id"]))
    picks, used = [], {c: 0 for c in by_city}
    while len(picks) < N_MEETINGS:
        moved = False
        for city in sorted(by_city):
            if len(picks) >= N_MEETINGS or used[city] >= len(by_city[city]):
                continue
            picks.append(by_city[city][used[city]])
            used[city] += 1
            moved = True
        if not moved:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)
    cache = SC / "mp3"
    rows = []
    for i, c in enumerate(picks, 1):
        mp3 = cache / f"{c['city_id']}__{c['meeting_id']}.mp3"
        if not mp3.exists():
            r = subprocess.run(["curl", "-sSfL", "-o", str(mp3), c["audio_url"]],
                               capture_output=True)
            if r.returncode != 0:
                log(f"  download failed: {c['city_id']}/{c['meeting_id']}")
                continue
        wav = f"lf_{i:03d}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{c['start_sec']:.3f}",
                        "-t", f"{c['dur_sec']:.3f}", "-i", str(mp3), "-ac", "1",
                        "-ar", "16000", str(OUT / "audio" / wav)], check=True)
        rows.append({**c, "wav": wav})
        log(f"  cut {i}/{len(picks)} {c['city_id']}/{c['meeting_id']} "
            f"@{c['start_sec']:.0f}s {c['ref_words']}w")

    # The manifest carries identities and times only. Reference text stays out of it and
    # out of the repo; the scorer reads it from the cached meeting JSON at scoring time.
    (OUT / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    log(f"{len(rows)} spans, {len({r['city_id'] for r in rows})} cities, "
        f"{sum(r['dur_sec'] for r in rows)/3600:.1f} h of audio -> {OUT}")


if __name__ == "__main__":
    main()
