"""Pool B, stage 3: filter the LLM's phrase proposals with the same pre-registered rules.

`gpt-5.6-luna` proposes; it does not decide. Every proposal from
scripts/glossary_poolb_llm.py has to earn its place the same way an acronym does:

  C0  not already in the frozen per-city lists, and not already one of the 52
      `procedural` terms of research/ds_wer/terms/*.v2.json
  C1  attested (not proposed, *attested*) in >= 3 distinct leak-free meetings across
      >= 3 cities; "universal" needs >= 5 cities and >= 10 city-meetings
  C2  at least one content word rarer than zipf 3.5 in general Greek
  C3  the production phrase scorer stays under partial_ratio 90 against general-Greek
      n-grams
  C4  attested as a token n-gram in the official Πρότυπος Κανονισμός Λειτουργίας
      Δημοτικού Συμβουλίου (ΦΕΚ Β' 109/2025), the same external authority
      scripts/build_ds_wer_terms_v2.py uses for its procedural class

The attestation count is computed over the 98 TRAIN-fold, benchmark-absent meetings, so
a term the model invented or half-remembered dies here.

Output (local): data/glossary/candidates_pool_b_phrases.json

Run:  <venv>/bin/python scripts/glossary_poolb_filter.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.glossary_distill import (CROSS_CITY_MIN_CITIES,  # noqa: E402
                                      MODELREG, PHRASE_REJECT, PHRASE_REVIEW,
                                      UNIVERSAL_MIN_CITIES,
                                      UNIVERSAL_MIN_CITY_MEETINGS,
                                      ZIPF_MAX_PHRASE_CONTENT, _FUNCTION_WORDS,
                                      build_bg_ngrams, frozen_aliases, match_norm,
                                      nfc_lower, zipf_max)
from scripts.glossary_poolb_mine import meeting_text, source_meetings  # noqa: E402

PROPOSALS = ROOT / "data/glossary/poolb_proposals.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"
OUT = ROOT / "data/glossary/candidates_pool_b_phrases.json"


def existing_procedural() -> set[str]:
    out: set[str] = set()
    for p in TERMS_DIR.glob("*.v2.json"):
        d = json.loads(p.read_text())
        for t in d["terms"]:
            if t["klass"] in ("procedural", "org_body_generic"):
                out.add(match_norm(t["canonical"]))
                out.update(match_norm(a) for a in t.get("aliases", []))
    return out


def attestation(terms: list[str]) -> dict[str, tuple[set, set]]:
    """Distinct (city, meeting) and cities each term actually occurs in."""
    keys = {t: " " + match_norm(t) + " " for t in terms}
    meets: dict[str, set] = defaultdict(set)
    cities: dict[str, set] = defaultdict(set)
    for city, meeting in source_meetings():
        blob = " " + match_norm(meeting_text(city, meeting)).replace("\n", " ") + " "
        for t, k in keys.items():
            if k in blob:
                meets[t].add((city, meeting))
                cities[t].add(city)
    return {t: (meets[t], cities[t]) for t in terms}


def main() -> None:
    from rapidfuzz import fuzz
    from wordfreq import zipf_frequency

    prop = json.loads(PROPOSALS.read_text())
    proposed = [r["term"] for r in prop["proposals"]]
    kinds = {r["term"]: r.get("kind") for r in prop["proposals"]}
    why = {r["term"]: r.get("why_not_common") for r in prop["proposals"]}

    proc = existing_procedural()
    frozen = frozen_aliases()
    frozen_all = set().union(*frozen.values()) if frozen else set()
    modelreg = (" " + match_norm(MODELREG.read_text(errors="replace")).replace("\n", " ")
                + " ") if MODELREG.exists() else ""
    bg = build_bg_ngrams()

    att = attestation(proposed)
    rows, rejections = [], defaultdict(int)
    for term in proposed:
        n = match_norm(term)
        rec = {"term": term, "kind": kinds.get(term), "why_not_common": why.get(term)}
        if n in proc:
            rejections["C0 already in v2 procedural"] += 1
            continue
        if n in frozen_all:
            rejections["C0 already frozen per-city"] += 1
            continue
        meets, cities = att[term]
        rec["n_meetings_attested"] = len(meets)
        rec["n_cities_attested"] = len(cities)
        rec["cities"] = sorted(cities)
        if len(cities) >= UNIVERSAL_MIN_CITIES and \
                len(meets) >= UNIVERSAL_MIN_CITY_MEETINGS:
            rec["scope"] = "universal"
        elif len(cities) >= CROSS_CITY_MIN_CITIES and len(meets) >= 3:
            rec["scope"] = "cross_city"
        else:
            rejections["C1 attestation"] += 1
            continue
        content = [w for w in nfc_lower(term).split() if w not in _FUNCTION_WORDS]
        zs = [zipf_max(w, zipf_frequency) for w in content] or [9.0]
        rec["zipf_content_min"] = round(min(zs), 2)
        if min(zs) > ZIPF_MAX_PHRASE_CONTENT:
            rejections["C2 rarity"] += 1
            continue
        best, best_g = 0.0, None
        for g in bg:
            if g == n:
                continue
            s = fuzz.partial_ratio(n, g)
            if s > best:
                best, best_g = s, g
        rec["confusability"] = {
            "scorer": "partial_ratio", "score": round(best, 1), "nearest": best_g,
            "verdict": ("reject" if best >= PHRASE_REJECT
                        else "review" if best >= PHRASE_REVIEW else "clear")}
        if rec["confusability"]["verdict"] == "reject":
            rejections["C3 confusable"] += 1
            continue
        if not (modelreg and f" {n} " in modelreg):
            rejections["C4 no external attestation"] += 1
            continue
        rec["category"] = "legal_procedural"
        rec["category_evidence"] = "modelreg:ΦΕΚ Β' 109/2025"
        rec["metric_eligible"] = True
        rec["injection_eligible"] = rec["confusability"]["verdict"] == "clear"
        rows.append(rec)

    rows.sort(key=lambda r: (-r["n_cities_attested"], -r["n_meetings_attested"],
                             r["term"]))
    OUT.write_text(json.dumps({
        "source": prop["source"], "llm": prop["model"],
        "n_proposed": len(proposed), "n_survivors": len(rows),
        "rejections": dict(rejections), "candidates": rows,
    }, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"n_proposed": len(proposed), "n_survivors": len(rows),
                      "rejections": dict(rejections),
                      "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()},
                     ensure_ascii=False, indent=1))
    for r in rows:
        print(f"  {r['term']:42s} {r['scope']:10s} c={r['n_cities_attested']} "
              f"m={r['n_meetings_attested']:2d} z={r['zipf_content_min']:.2f} "
              f"c3={r['confusability']['verdict']}")


if __name__ == "__main__":
    main()
