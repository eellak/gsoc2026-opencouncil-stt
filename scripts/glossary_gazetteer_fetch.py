"""Fetch an OSM gazetteer (place names, street names, named public facilities) for
every city that appears in the correction database.

`scripts/build_ds_wer_terms_v2.py` already does this, but only for argos and
orestiada, and only for `place` nodes. Issue #20 needs toponym evidence for the other
nine cities, and needs street names: the one unresolved candidate of issue #19,
ΝΕΓΡΗ (athens), is a fragment of the odonym «Φωκίωνος Νέγρη», and odonyms are exactly
the class the model mangles.

Network only, cached under ~/.cache/oc-public/glossary-error/. No GPU, no writes
inside the repo.

Run:  <venv>/bin/python scripts/glossary_gazetteer_fetch.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

CACHE = Path.home() / ".cache/oc-public/glossary-error"
V2_CACHE = Path.home() / ".cache/oc-public/ds-wer-v2"
OVERPASS = "https://overpass-api.de/api/interpreter"

# admin_level=7 municipality boundary relations, resolved via Nominatim 2026-08-16 and
# checked against the returned display_name. argos / orestiada repeat the ids already
# pinned in build_ds_wer_terms_v2.CITY_SRC.
CITY_REL = {
    "argos": 2185766,
    "orestiada": 2345114,
    "athens": 1370736,
    "chania": 2187320,
    "sparta": 2179053,
    "xylokastro": 2186728,
    "zografou": 1392944,
    "vrilissia": 1404550,
    "chalandri": 1391619,
    "samothraki": 5222497,
    "vari-voula-vouliagmeni": 2080695,
}

PLACE_TYPES = ("city", "town", "village", "hamlet", "suburb", "quarter",
               "neighbourhood", "locality")
AMENITIES = ("school", "hospital", "theatre", "library", "college", "university",
             "kindergarten", "townhall", "clinic", "police", "fire_station")


def _query(rel: int) -> str:
    return (
        f"[out:json][timeout:300];rel({rel});map_to_area->.a;("
        f'node["place"~"^({"|".join(PLACE_TYPES)})$"](area.a);'
        f'way["highway"]["name"](area.a);'
        f'node["amenity"~"^({"|".join(AMENITIES)})$"]["name"](area.a);'
        f'way["amenity"~"^({"|".join(AMENITIES)})$"]["name"](area.a);'
        f");out tags;"
    )


def fetch(city: str) -> Path:
    out = CACHE / f"overpass-{city}.json"
    if out.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    q = _query(CITY_REL[city])
    r = subprocess.run(
        ["curl", "-sSf", "--retry", "2", "--max-time", "400", "-X", "POST", OVERPASS,
         "--data-urlencode", f"data={q}"],
        capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise SystemExit(f"{city}: overpass rc={r.returncode} {r.stderr[:300]!r}")
    out.write_bytes(r.stdout)
    return out


def main() -> None:
    manifest = {}
    for city in CITY_REL:
        p = fetch(city)
        b = p.read_bytes()
        els = json.loads(b)["elements"]
        manifest[city] = {
            "osm_rel": CITY_REL[city],
            "n_elements": len(els),
            "sha256": hashlib.sha256(b).hexdigest(),
        }
        print(f"{city:26s} {len(els):7,d} elements", flush=True)
    (CACHE / "manifest.json").write_text(
        json.dumps({"source": "overpass-api.de", "retrieved": "2026-08-16",
                    "place_types": list(PLACE_TYPES), "amenities": list(AMENITIES),
                    "cities": manifest}, ensure_ascii=False, indent=1) + "\n")
    print("manifest:", CACHE / "manifest.json")
    _ = V2_CACHE  # documented sibling cache; nothing is written there


if __name__ == "__main__":
    sys.exit(main())
