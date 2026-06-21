"""Fetch the real per-meeting roster from the OpenCouncil meeting endpoint.

GET https://opencouncil.gr/api/cities/{cityId}/meetings/{meetingId} returns a
dict with `people` (full roster), `parties`, `subjects`. We build a small,
clean per-meeting candidate set for the deterministic name corrector:
full names + name_short + individual name tokens (len>=4) + party names.

Output: data/improve_loop/rosters.json  ->  {"<city>/<meeting>": [terms...]}
Private/unavailable meetings are skipped and counted.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "data" / "improve_loop" / "rosters.json"
BASE = "https://opencouncil.gr/api/cities/{city}/meetings/{meeting}"


def _terms_from(meeting_json: dict) -> list[str]:
    terms: set[str] = set()
    for p in meeting_json.get("people") or []:
        for key in ("name", "name_short"):
            v = (p.get(key) or "").strip()
            if v:
                terms.add(v)
                for tok in v.replace(".", " ").split():
                    if len(tok) >= 4 and tok[:1].isupper():
                        terms.add(tok)          # first name / surname mentions
    for party in meeting_json.get("parties") or []:
        v = (party.get("name") or "").strip()
        if v:
            terms.add(v)
    return sorted(terms)


def main():
    cdf = pd.read_parquet(EVAL / "chains.parquet")
    split = json.loads((EVAL / "split.json").read_text())
    ev = cdf[cdf.meeting_id.isin(split["eval_meeting_ids"])]
    pairs = sorted({(str(r.city_id), str(r.meeting_id))
                    for r in ev[["city_id", "meeting_id"]].drop_duplicates().itertuples()})

    rosters, ok, fail = {}, 0, 0
    for city, meeting in pairs:
        url = BASE.format(city=city, meeting=meeting)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "oc-eval/1.0",
                                                       "Accept": "application/json"})
            data = json.load(urllib.request.urlopen(req, timeout=30))
            terms = _terms_from(data)
            rosters[f"{city}/{meeting}"] = terms
            ok += 1
            print(f"  ok   {city}/{meeting:18} people={len(data.get('people') or [])} terms={len(terms)}")
        except Exception as e:
            fail += 1
            print(f"  FAIL {city}/{meeting:18} {type(e).__name__} {str(e)[:60]}")
        time.sleep(0.2)

    OUT.write_text(json.dumps(rosters, ensure_ascii=False, indent=2))
    print(f"\n{ok} ok, {fail} failed -> {OUT}")


if __name__ == "__main__":
    main()
