"""Fetch per-meeting taskStatus (esp. `humanReview`) + dateTime + name.

The corrections CSV and speakers.parquet can't tell a *fully human-reviewed*
meeting from a merely task-processed one. The meeting JSON's
`taskStatus.humanReview` flag can: it gates whether a meeting's no-edit
utterances are trustworthy ground truth and whether the Human-Intervention-Rate
metric is well-defined for it.

Output: data/eval/meeting_meta.json  ->  {"<city> <meeting>": {humanReview, ...}}
Resumable: existing entries are kept; only missing meetings are fetched.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data-1779206108158.csv"
EVAL = ROOT / "data" / "eval"
AVAIL = ROOT / "data" / "reports" / "meeting-availability-2026-06-19.json"
BASE = "https://opencouncil.gr/api/cities/{city}/meetings/{meeting}"
OUT = EVAL / "meeting_meta.json"

TASK_FLAGS = ("transcribe", "fixTranscript", "humanReview", "transcriptSent",
              "summarize", "processAgenda")


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(CSV, usecols=["meeting_id", "city_id"]).drop_duplicates()
    priv = set(json.loads(AVAIL.read_text()).get("private_meeting_keys", []))
    meta: dict = json.loads(OUT.read_text()) if OUT.exists() else {}

    todo = sorted({(str(r.city_id), str(r.meeting_id)) for r in pairs.itertuples()})
    ok, fail, skipped = 0, 0, 0
    for i, (city, meeting) in enumerate(todo, 1):
        key = f"{city} {meeting}"
        if key in meta:
            continue
        if key in priv:
            meta[key] = {"city_id": city, "meeting_id": meeting, "private": True}
            skipped += 1
            continue
        url = BASE.format(city=city, meeting=meeting)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "oc-eval/1.0", "Accept": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=60))
            ts = d.get("taskStatus") or {}
            mt = d.get("meeting") or {}
            entry = {"city_id": city, "meeting_id": meeting, "private": False,
                     "name": mt.get("name"), "dateTime": mt.get("dateTime"),
                     "transcriptHiddenForReview": d.get("transcriptHiddenForReview")}
            entry.update({f: ts.get(f) for f in TASK_FLAGS})
            meta[key] = entry
            ok += 1
            if ok % 25 == 0:
                print(f"[{i}/{len(todo)}] ok={ok} fail={fail} skip={skipped}", flush=True)
                OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  FAIL {key}: {type(e).__name__} {str(e)[:50]}", flush=True)
        time.sleep(0.15)

    OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    pub = [m for m in meta.values() if not m.get("private")]
    rev = [m for m in pub if m.get("humanReview")]
    print(f"\nDONE: {ok} ok, {fail} failed, {skipped} private")
    print(f"  public meetings: {len(pub)} | humanReview=true: {len(rev)} | "
          f"humanReview=false: {len(pub) - len(rev)}")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
