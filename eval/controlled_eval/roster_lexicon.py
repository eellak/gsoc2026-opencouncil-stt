#!/usr/bin/env python3
"""One closed term list per council meeting, for wayfinder #18.

Three sources are merged into a single list per (city, meeting):

  1. the per-meeting ROSTER (`data/pii/rosters_full.json`, external to the
     benchmark) intersected with the city's frozen `person_surname` terms, by the
     verbatim rule of `scripts/serving_stack/name_repair.meeting_roster_terms`;
  2. the rest of the city's hash-frozen term file
     (`research/ds_wer/terms/{city}.json`, v1, 2026-08-12) — the two `place_name`
     entries, which no roster can ever "contain";
  3. an explicitly chosen slice of the 147 error-mined candidates of
     `exp-2026-08-16-error-mined-terms` (`data/glossary/candidates_error_mined.json`,
     sha256-frozen), restricted to each candidate's own `eligible_cities`.

The tier choice in (3) is FROZEN in `docs/specs/2026-08-16-roster-selection-prereg.md`
before any WER number was computed, and is deliberately narrow:

  IN  - all 33 `person` candidates (user instruction: the names go in);
      - the 31 candidates carrying `review_priority` 1 or 2, i.e. the first human
        review sitting of #20.
  OUT - the 52 unreviewed `review` candidates and the 55 `review-backlog`
        singletons (one human correction, no independent witness);
      - anything with a non-empty `ambiguous_wrong_forms`, because at least one of
        its observed wrong forms is itself a legitimate Greek word or acronym.
      - the `injection` tier is empty by construction upstream; nothing to take.

Nothing here writes to `research/ds_wer/terms/`. The admitted slice is assembled at
runtime, so the frozen files stay frozen.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from serving_stack.name_repair import (  # noqa: E402
    RosterContext, meeting_roster_terms, rnorm,
)

TERMS_DIR = ROOT / "research/ds_wer/terms"
ROSTERS = ROOT / "data/pii/rosters_full.json"
MINED = ROOT / "data/glossary/candidates_error_mined.json"

CITIES = ("argos", "athens", "chalandri", "chania", "orestiada", "samothraki",
          "sparta", "vrilissia", "xylokastro", "zografou")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_city_terms() -> dict[str, list[dict]]:
    return {c: json.loads((TERMS_DIR / f"{c}.json").read_text())["terms"]
            for c in CITIES}


def load_rosters() -> dict[str, list[str]]:
    return json.loads(ROSTERS.read_text())


# --------------------------------------------------------------- mined slice
def admitted_mined() -> tuple[dict[str, list[dict]], dict]:
    """The frozen tier choice, as term dicts keyed by city.

    Returns (per_city_terms, accounting)."""
    cands = json.loads(MINED.read_text())["candidates"]
    stats = Counter()
    per_city: dict[str, list[dict]] = {c: [] for c in CITIES}
    admitted = []
    for c in cands:
        person = c["category"] == "person"
        reviewed = c.get("review_priority") in (1, 2)
        if not (person or reviewed):
            stats["out_tier_not_chosen"] += 1
            continue
        if c.get("ambiguous_wrong_forms"):
            stats["out_ambiguous_wrong_forms"] += 1
            stats[f"out_ambiguous_{c['category']}"] += 1
            continue
        aliases = sorted({rnorm(a) for a in c["display_aliases"]} | {c["key"]})
        term = {
            "id": f"mined:{c['key']}",
            "canonical": c["term"],
            "klass": f"mined_{c['category']}",
            "covers": [c["term"]],
            "aliases": aliases,
            "_surface": c["display_aliases"][0],
        }
        cities = [x for x in c["eligible_cities"] if x in per_city]
        if not cities:
            stats["out_no_eligible_benchmark_city"] += 1
            continue
        for city in cities:
            per_city[city].append(term)
        stats["in"] += 1
        stats[f"in_{c['category']}"] += 1
        admitted.append({"term": c["term"], "category": c["category"],
                         "tier": c["tier"], "review_priority": c["review_priority"],
                         "cities": cities, "n_corrections": c["n_corrections"]})
    return per_city, {"counts": dict(stats), "admitted": admitted,
                      "n_candidates": len(cands),
                      "sha256_candidates": sha256(MINED)}


# --------------------------------------------------------- per-meeting context
def build_meeting_context(city: str, meeting_id: str, city_terms: list[dict],
                          mined_terms: list[dict], rosters: dict,
                          seen_freq: Counter) -> tuple[RosterContext, dict]:
    """One closed list for one meeting, as a `RosterContext` the frozen arm-E rule
    can consume unchanged.

    Person surnames still require roster presence (that is the whole point of the
    rule: a surname earns its place because the person is in the room). Toponyms,
    acronyms and organisations cannot be in a roster, so they are admitted at city
    level with `persons_in_meeting = [canonical]` — a single referent, which is
    exactly what the rule's `n_persons == 1` clause asks for.
    """
    ctx = RosterContext(seen_freq=seen_freq)
    persons = [t for t in city_terms if t["klass"] == "person_surname"]
    roster_entries = rosters.get(f"{city}/{meeting_id}", [])
    if roster_entries:
        ctx.present.update(meeting_roster_terms(persons, roster_entries))
    non_person = [t for t in city_terms if t["klass"] != "person_surname"]
    for t in non_person + mined_terms:
        if t["klass"].startswith("mined_person"):
            # a mined person name is admitted only if the roster puts them in the
            # room, on the same principle as (1)
            hit = meeting_roster_terms([t], roster_entries) if roster_entries else {}
            if not hit:
                continue
            ctx.present[t["id"]] = hit[t["id"]]
            continue
        ctx.present[t["id"]] = {"term": t, "persons_in_meeting": [t["canonical"]]}
    for tid, info in ctx.present.items():
        for cov in info["persons_in_meeting"]:
            parts = cov.split()
            if len(parts) >= 2:
                ctx.first_names.add(rnorm(parts[0]))
        for alias in info["term"]["aliases"]:
            an = rnorm(alias)
            ctx.valid_aliases.add(an)
            ctx.alias_surface[(tid, an)] = info["term"].get("_surface", alias)
    return ctx, {"n_terms": len(ctx.present), "has_roster": bool(roster_entries)}
