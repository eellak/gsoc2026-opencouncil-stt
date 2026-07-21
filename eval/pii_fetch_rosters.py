"""Fetch the full per-meeting roster for EVERY meeting in the HF dataset (not
just the eval subset that eval/fetch_rosters.py covers), for the PII scan's
person allowlist.

Writes two files under data/pii/:
  rosters_full.json  -> {"<city>/<meeting>": [name terms...]}  (allowlist input)
  people_roles.json  -> {"<city>/<meeting>": [{"name","name_short","roles"}]}
                        (role metadata so a future run can allowlist ONLY elected
                        officials/staff, addressing Codex review critical #1).

Resumable: existing entries in rosters_full.json are kept; only missing meetings
are fetched. Private/unavailable meetings are skipped and counted.

Usage:
  .venv-eval/bin/python -m eval.pii_fetch_rosters
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from eval.fetch_rosters import _terms_from

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "data" / "hf-dataset" / "public"
OUT = ROOT / "data" / "pii"
ROSTERS_FULL = OUT / "rosters_full.json"
PEOPLE_ROLES = OUT / "people_roles.json"
BASE = "https://opencouncil.gr/api/cities/{city}/meetings/{meeting}"


def _people_roles(meeting_json: dict) -> list[dict]:
    out = []
    for p in meeting_json.get("people") or []:
        roles = []
        for r in p.get("roles") or []:
            # role dicts vary; keep any human-readable type/name field present
            for k in ("type", "name", "role", "title"):
                v = r.get(k) if isinstance(r, dict) else None
                if v:
                    roles.append(str(v))
                    break
        out.append({"name": (p.get("name") or "").strip(),
                    "name_short": (p.get("name_short") or "").strip(),
                    "roles": roles})
    return out


def _dataset_meetings() -> list[tuple[str, str]]:
    parts = [pd.read_parquet(PUB / f"{s}.parquet", columns=["city_id", "meeting_id"])
             for s in ("train", "validation")]
    df = pd.concat(parts, ignore_index=True)
    return sorted({(str(c), str(m)) for c, m in zip(df.city_id, df.meeting_id)})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rosters = json.loads(ROSTERS_FULL.read_text()) if ROSTERS_FULL.exists() else {}
    people = json.loads(PEOPLE_ROLES.read_text()) if PEOPLE_ROLES.exists() else {}
    meetings = _dataset_meetings()
    todo = [(c, m) for c, m in meetings if f"{c}/{m}" not in rosters]
    print(f"{len(meetings)} dataset meetings, {len(rosters)} already fetched, "
          f"{len(todo)} to fetch")

    ok = fail = 0
    for i, (city, meeting) in enumerate(todo, 1):
        key = f"{city}/{meeting}"
        url = BASE.format(city=city, meeting=meeting)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "oc-pii/1.0", "Accept": "application/json"})
            data = json.load(urllib.request.urlopen(req, timeout=30))
            rosters[key] = _terms_from(data)
            people[key] = _people_roles(data)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  FAIL {key:28} {type(e).__name__} {str(e)[:50]}")
        if i % 25 == 0:
            ROSTERS_FULL.write_text(json.dumps(rosters, ensure_ascii=False))
            PEOPLE_ROLES.write_text(json.dumps(people, ensure_ascii=False))
            print(f"  [{i}/{len(todo)}] ok={ok} fail={fail} (checkpoint)")
        time.sleep(0.2)

    ROSTERS_FULL.write_text(json.dumps(rosters, ensure_ascii=False))
    PEOPLE_ROLES.write_text(json.dumps(people, ensure_ascii=False))
    print(f"\n{ok} ok, {fail} failed -> {ROSTERS_FULL} ({len(rosters)} meetings total)")


if __name__ == "__main__":
    main()
