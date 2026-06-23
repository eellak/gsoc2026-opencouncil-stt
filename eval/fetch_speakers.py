"""Fetch per-utterance speaker identity + duration from the meeting JSON.

GET https://opencouncil.gr/api/cities/{city}/meetings/{meeting} returns
`transcript[]` segments; each carries `speakerTag.personId` and `utterances[]`
with start/end timestamps and `lastModifiedBy`. We flatten to one row per
utterance so we can build a speaker- and meeting-disjoint split and per-speaker
data floors (neither speaker_id nor duration exist in the corrections CSV).

Public meetings only (the publishable benchmark); private meetings are skipped
and counted. Outputs:
  data/eval/speakers.parquet      — one row per utterance (the split foundation)
  data/eval/speaker_stats.json    — per-person + per-meeting summary
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
OUT_PARQUET = EVAL / "speakers.parquet"
OUT_STATS = EVAL / "speaker_stats.json"


def _fetch(city: str, meeting: str) -> dict:
    url = BASE.format(city=city, meeting=meeting)
    req = urllib.request.Request(
        url, headers={"User-Agent": "oc-eval/1.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _rows(d: dict, city: str, meeting: str):
    people = {p["id"]: p for p in (d.get("people") or [])}
    for seg in d.get("transcript") or []:
        tag = seg.get("speakerTag") or {}
        pid = tag.get("personId")
        person = people.get(pid) or {}
        pname = (person.get("name") or "").strip() or None
        label = tag.get("label")
        for u in seg.get("utterances") or []:
            try:
                dur = float(u.get("endTimestamp", 0)) - float(u.get("startTimestamp", 0))
            except (TypeError, ValueError):
                dur = 0.0
            yield {
                "utterance_id": u.get("id"),
                "meeting_id": meeting,
                "city_id": city,
                "person_id": pid,            # None = unidentified speaker
                "person_name": pname,
                "speaker_label": label,      # SPEAKER_xx fallback identity
                "dur_s": max(0.0, dur),
                "last_modified_by": u.get("lastModifiedBy"),  # None / 'user' / 'task' ...
                "n_chars": len(u.get("text") or ""),
            }


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(CSV, usecols=["meeting_id", "city_id"]).drop_duplicates()
    priv = set(json.loads(AVAIL.read_text()).get("private_meeting_keys", []))

    # Resume: keep rows from meetings already fetched in a prior run, refetch the rest.
    rows, ok, fail, skipped = [], 0, 0, 0
    done_meetings: set[str] = set()
    if OUT_PARQUET.exists():
        prev = pd.read_parquet(OUT_PARQUET)
        done_meetings = set(prev["city_id"].astype(str) + " " + prev["meeting_id"].astype(str))
        rows = prev.to_dict("records")
        print(f"resume: {len(done_meetings)} meetings already fetched, {len(rows)} rows", flush=True)

    todo = [(str(r.city_id), str(r.meeting_id)) for r in pairs.itertuples()]
    for i, (city, meeting) in enumerate(sorted(todo), 1):
        if f"{city} {meeting}" in priv:
            skipped += 1
            continue
        if f"{city} {meeting}" in done_meetings:
            continue
        try:
            d = _fetch(city, meeting)
            n0 = len(rows)
            rows.extend(_rows(d, city, meeting))
            ok += 1
            if ok % 25 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] ok={ok} fail={fail} skip={skipped} rows={len(rows)}", flush=True)
            _ = len(rows) - n0
        except Exception as e:  # noqa: BLE001 — log and continue
            fail += 1
            print(f"  FAIL {city}/{meeting}: {type(e).__name__} {str(e)[:60]}", flush=True)
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)

    # summary
    df["min"] = df.dur_s / 60.0
    by_person = (
        df[df.person_id.notna()]
        .groupby("person_id")
        .agg(
            name=("person_name", "first"),
            minutes=("min", "sum"),
            n_utts=("utterance_id", "size"),
            n_meetings=("meeting_id", "nunique"),
            n_cities=("city_id", "nunique"),
        )
        .sort_values("minutes", ascending=False)
    )
    stats = {
        "meetings_ok": ok,
        "meetings_failed": fail,
        "meetings_skipped_private": skipped,
        "n_utterances": len(df),
        "total_minutes": round(df["min"].sum(), 1),
        "n_persons_identified": int(df.person_id.notna().sum() and by_person.shape[0]),
        "minutes_unidentified": round(df[df.person_id.isna()]["min"].sum(), 1),
        "last_modified_by_counts": df.last_modified_by.fillna("__none__").value_counts().to_dict(),
        "persons_with_ge_10min": int((by_person.minutes >= 10).sum()),
        "persons_with_ge_30min": int((by_person.minutes >= 30).sum()),
        "top20_persons": [
            {"person_id": pid, "name": r["name"], "minutes": round(r.minutes, 1),
             "n_meetings": int(r.n_meetings), "n_cities": int(r.n_cities)}
            for pid, r in by_person.head(20).iterrows()
        ],
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nDONE: {ok} ok, {fail} failed, {skipped} private skipped")
    print(f"  {len(df):,} utterances, {stats['total_minutes']:.0f} min total")
    print(f"  persons: {by_person.shape[0]} identified, "
          f"{stats['persons_with_ge_10min']} with >=10min, "
          f"{stats['persons_with_ge_30min']} with >=30min")
    print(f"  -> {OUT_PARQUET}\n  -> {OUT_STATS}")


if __name__ == "__main__":
    main()
