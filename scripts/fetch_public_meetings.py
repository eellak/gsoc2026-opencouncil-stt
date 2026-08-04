#!/usr/bin/env python3
"""Fetch OpenCouncil's public meetings from the open API.

Council meetings released after 2026-05-16 are public, and the API at
`https://opencouncil.gr/api` serves them without authentication. That matters because
every evaluation in this project so far has run on meetings that are also in the training
data or in the benchmark, and the [delivery plan](../docs/specs/gsoc-delivery-plan.md)
needs a set of meetings nothing has touched.

Endpoints used (documented at https://opencouncil.gr/docs):
  GET /api/cities/all                              minimal city list with meeting counts
  GET /api/cities/{cityId}/meetings                meeting list, includes audioUrl
  GET /api/cities/{cityId}/meetings/{meetingId}    full record: transcript, people,
                                                   speakerTags, subjects

WHAT GOES WHERE. The full records contain utterance text and speaker names, which is PII
under the same rule that governs the rest of this repo, so they are cached under $SC and
never written into git. What lands in `data/public-meetings/index.csv` is only what is
already public on the website: city, meeting id, date, audio URL and counts.

Usage:
  SC=~/.cache/oc-public python scripts/fetch_public_meetings.py index
  SC=~/.cache/oc-public python scripts/fetch_public_meetings.py fetch [--limit N]
Env: SC  SINCE (default 2026-05-16)  CONCURRENCY
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
API = "https://opencouncil.gr/api"
SINCE = os.environ.get("SINCE", "2026-05-16")
CONC = int(os.environ.get("CONCURRENCY", "6"))
INDEX = ROOT / "data/public-meetings/index.csv"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def get(url: str) -> dict | list:
    """curl, not urllib: same reason as everywhere else in this repo."""
    r = subprocess.run(["curl", "-sSfL", "--retry", "2", url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{url}: {r.stderr[:200]}")
    return json.loads(r.stdout)


def already_local() -> set[tuple[str, str]]:
    """(city, meeting) pairs this project has already trained or evaluated on.

    Two sources, because a meeting can be in either: the cached meeting JSON used to build
    the dataset, and the training manifest itself.
    """
    have = set()
    for f in glob.glob(str(ROOT / "data/asr/meeting_json/*.json")):
        b = Path(f).stem
        if "__" in b:
            have.add(tuple(b.split("__", 1)))
    man = ROOT / "data/asr/train_manifest.csv"
    if man.exists():
        with man.open() as f:
            for r in csv.DictReader(f):
                have.add((r["city_id"], r["meeting_id"]))
    return have


def build_index() -> list[dict]:
    cities = [c for c in get(f"{API}/cities/all") if c["_count"]["councilMeetings"] > 0]
    log(f"{len(cities)} cities with council meetings")
    have = already_local()
    log(f"{len(have)} (city, meeting) pairs already used by this project")

    rows, total = [], 0
    for c in cities:
        try:
            meetings = get(f"{API}/cities/{c['id']}/meetings")
        except RuntimeError as e:
            log(f"  {c['id']}: {e}")
            continue
        for m in meetings:
            total += 1
            date = (m.get("dateTime") or "")[:10]
            rows.append({
                "city_id": c["id"], "meeting_id": m["id"], "date": date,
                "released": bool(m.get("released")),
                "has_audio": bool(m.get("audioUrl")),
                "audio_url": m.get("audioUrl") or "",
                "n_subjects": len(m.get("subjects") or []),
                "seen_by_project": (c["id"], m["id"]) in have,
                "eligible": bool(m.get("released")) and bool(m.get("audioUrl"))
                            and date >= SINCE and (c["id"], m["id"]) not in have,
            })
    log(f"{total} meetings listed, {sum(1 for r in rows if r['eligible'])} eligible "
        f"(released, has audio, on/after {SINCE}, never used here)")
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with INDEX.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["city_id"], r["date"])))
    log(f"index -> {INDEX}")
    return rows


def fetch_details(limit: int = 0):
    if not INDEX.exists():
        raise SystemExit("run `index` first")
    with INDEX.open() as f:
        rows = [r for r in csv.DictReader(f) if r["eligible"] == "True"]
    if limit:
        rows = rows[:limit]
    out = SC / "meetings"
    out.mkdir(parents=True, exist_ok=True)

    def one(r):
        dst = out / f"{r['city_id']}__{r['meeting_id']}.json"
        if dst.exists() and dst.stat().st_size > 500:
            return r, "cached"
        try:
            d = get(f"{API}/cities/{r['city_id']}/meetings/{r['meeting_id']}")
        except RuntimeError as e:
            return r, f"error: {e}"
        tmp = dst.with_suffix(".part")
        tmp.write_text(json.dumps(d, ensure_ascii=False))
        tmp.replace(dst)
        return r, "fetched"

    log(f"fetching {len(rows)} meeting records -> {out}")
    counts = {"fetched": 0, "cached": 0, "error": 0}
    with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
        for i, (r, status) in enumerate(ex.map(one, rows), 1):
            counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
            if status.startswith("error"):
                log(f"  {r['city_id']}/{r['meeting_id']}: {status[:120]}")
            if i % 25 == 0 or i == len(rows):
                log(f"  {i}/{len(rows)}")
    log(f"done: {counts}")

    # a PII-free summary of what came back, so the index says how usable each record is
    summary = []
    for p in sorted(out.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        tr = d.get("transcript") or []
        n_utt = sum(len(s.get("utterances") or []) for s in tr)
        dur = max((float(s.get("endTimestamp") or 0) for s in tr), default=0.0)
        summary.append({
            "city_id": (d.get("meeting") or {}).get("cityId"),
            "meeting_id": (d.get("meeting") or {}).get("id"),
            "n_segments": len(tr), "n_utterances": n_utt,
            "duration_sec": round(dur, 1),
            "n_people": len(d.get("people") or []),
            "n_speaker_tags": len(d.get("speakerTags") or []),
            "hidden_for_review": bool(d.get("transcriptHiddenForReview")),
        })
    dst = ROOT / "data/public-meetings/fetched_summary.csv"
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    log(f"summary -> {dst}  ({len(summary)} records, "
        f"{sum(s['duration_sec'] for s in summary) / 3600:.1f} h of audio)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "index"
    if cmd == "index":
        build_index()
    elif cmd == "fetch":
        n = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
        fetch_details(n)
    else:
        raise SystemExit(__doc__)
