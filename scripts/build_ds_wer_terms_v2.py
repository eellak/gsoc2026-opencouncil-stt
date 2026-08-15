#!/usr/bin/env python3
"""Build the **v2** DS-WER term lists for the two validation cities (argos, orestiada).

v1 (`scripts/build_ds_wer_terms.py`, 2026-08-11) covered roster surnames plus the
city/municipality name and nothing else. On the 39 frozen validation windows that is
only **250** reference term occurrences, and `exp-2026-08-12-ds-wer` closed with a 95%
interval that includes zero largely because of it. v2 widens the denominator.

v1 files are **not touched**: the 2026-08-12 numbers stay reproducible. v2 is written
to `research/ds_wer/terms/{city}.v2.json`.

**Every term comes from a named external source, recorded per term in `source`.**
Nothing here reads a model output, a decode, a provider hypothesis or an error
analysis. The only transcript text this script reads is a frequency table over the
*eight seen benchmark cities* used as a common-word stoplist - never argos/orestiada,
and only ever to **reject** a candidate, never to select one.

Sources
-------
S1  OpenCouncil public registry `GET /api/cities/{city}/people`
      surnames (v1 rule, verbatim)                       -> person_surname
      full names, given x surname inflection cross       -> person_full_name
      `roles[].party.name`                               -> org_party
      `roles[].administrativeBody.name`                  -> org_body / org_body_generic
S2  OpenCouncil `GET /api/cities/{city}`                 -> place_city, place_municipality
S3a ELSTAT 2021 census, official municipality / municipal-unit / community table
    (permanent population results)                       -> place_admin_unit
S3b OpenStreetMap Overpass, place nodes inside the municipality's admin_level=7
    relation. Nominatives merge into the matching S3a community term; the rest stay
    as a clearly separated supplementary class           -> place_settlement_osm
S4  Greek council procedural vocabulary. A fixed, city-independent candidate list is
    written into this file, and a candidate is kept **only if it is attested as a
    token n-gram in the official Πρότυπος Κανονισμός Λειτουργίας Δημοτικού Συμβουλίου
    (ΦΕΚ Β' 109/2025)**, with its attestation count recorded -> procedural

Cuts (preregistered here, before any v2 score is computed)
---------------------------------------------------------
`entities`  PRIMARY.  person_surname, person_full_name, org_party, org_body,
                      place_city, place_municipality, place_admin_unit,
                      place_settlement_osm
`procedure` SECONDARY. procedural, org_body_generic
`all`       DESCRIPTIVE union. Explicitly **not** the system-ranking metric: it mixes
            a frequent, easy vocabulary into the denominator and would flatter any
            system that is good at ordinary Greek and bad at names - which is exactly
            our system's profile.
Each cut is aligned and scored independently (`scripts/ds_wer_v2.py`); class scores
therefore need not sum to `all`.

Bias guards
-----------
G1  No hypothesis, decode, or error analysis is an input.
G2  Common-word stoplist, applied to the whole normalized alias (so "Νέα" can be
    rejected while "Νέα Κίος" survives) and to that alias only, never to the other
    aliases of the same term:
      (a) the alias occurs >= COMMON_FREQ times in the 220 benchmark windows of the
          eight cities that are NOT argos/orestiada; or
      (b) it is a single-token place or organisation alias that appears as a
          *lowercase* (i.e. common-noun) headword in the LibreOffice el_GR hunspell
          dictionary.
    (a) alone is underpowered here - the seen-city corpus is only ~67k tokens - which
    is why (b) is enforced rather than merely recorded. Both reasons are written into
    `dropped_aliases`, so every rejection is auditable. G2 does not apply to the
    `procedural` class: frequency is the point there.
G3  Latin-script aliases are dropped and counted.
G4  v1's rules kept verbatim: MIN_ALIAS_LEN, and an alias claimed by more than one
    term is dropped from every term that claims it.
G5  Inflection only by the fixed suffix rules of `build_ds_wer_terms.inflect`.
G6  Standalone given names are **not** terms. A roll call must not charge two errors
    for one misheard person, and given names are easy denominator mass. Full names are
    single atomic multi-token terms instead, and longest-match-leftmost makes a
    full-name mention cost at most one entity error.

    python3 scripts/build_ds_wer_terms_v2.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from build_ds_wer_terms import MIN_ALIAS_LEN, inflect  # noqa: E402

OUT = ROOT / "research/ds_wer/terms"
MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
CACHE = Path.home() / ".cache/oc-public/ds-wer-v2"
API = "https://opencouncil.gr/api"
OVERPASS = "https://overpass-api.de/api/interpreter"
HUNSPELL_URL = ("https://raw.githubusercontent.com/LibreOffice/dictionaries/"
                "master/el_GR/el_GR.dic")
ELSTAT_URL = ("https://www.statistics.gr/documents/20181/17286366/"
              "APOF_APOT_MON_DHM_KOIN.pdf/41ae8e6c-5860-b58e-84f7-b64f9bc53ec4")
MODELREG_URL = ("https://www.ypes.gr/wp-content/uploads/2025/01/"
                "apof2804-20250120-fek-109B-20250121.pdf")
RETRIEVED = "2026-08-15"
VERSION = 2

# admin_level=7 boundary relation of each municipality (OSM), and the exact ELSTAT
# municipality heading each city maps to.
CITY_SRC = {
    "argos": {"osm_rel": 2185766, "elstat": "ΔΗΜΟΣ ΑΡΓΟΥΣ - ΜΥΚΗΝΩΝ"},
    "orestiada": {"osm_rel": 2345114, "elstat": "ΔΗΜΟΣ ΟΡΕΣΤΙΑΔΑΣ"},
}

PLACE_TYPES = ("city", "town", "village", "hamlet", "suburb", "quarter")

# G2(a) threshold. Same corpus and same number as the "common Greek word" rule already
# used by scripts/analysis/name_lexicon_audit.py.
COMMON_FREQ = 5
MAX_NGRAM = 6

GREEK_RE = re.compile(r"^[Ͱ-Ͽἀ-῿\s'’·\-\.]+$")

CUTS = {
    "entities": ("person_surname", "person_full_name", "org_party", "org_body",
                 "place_city", "place_municipality", "place_admin_unit",
                 "place_settlement_osm"),
    "procedure": ("procedural", "org_body_generic"),
}
CUTS["all"] = CUTS["entities"] + CUTS["procedure"]

# G2 applies to these classes only.
NAMELIKE = CUTS["entities"]
# `person_surname` is exempt from G2(a): v1's surname list must survive verbatim
# inside v2 or the two versions stop nesting, and a surname that is frequent in eight
# OTHER councils is a frequent surname, not common vocabulary - which is the only
# thing rule (a) exists to reject. Decided before any v2 score existed; it changes
# exactly two aliases (Ηλία, Παπαδόπουλου).
RULE_A_EXEMPT = ("person_surname",)
# hunspell rule G2(b) is enforced only where common-noun collision is endemic.
HUNSPELL_CLASSES = ("place_city", "place_municipality", "place_admin_unit",
                    "place_settlement_osm", "org_party", "org_body",
                    "org_body_generic")

# Two sources naming the same thing ("Ορεστιάδα" the city and "Δημοτική Κοινότητα
# Ορεστιάδος"; "Δημοτικό Συμβούλιο" the OpenCouncil body and the same phrase in the
# Model Regulation) must become ONE term, not two terms that annihilate each other
# under the collision rule. Terms in the same family that share a normalized alias are
# merged; the surviving class is the first one listed here. Person classes never
# merge - two people sharing a surface form are genuinely different terms and keep
# v1's drop-from-both behaviour.
FAMILY = {"place_city": "place", "place_municipality": "place",
          "place_admin_unit": "place", "place_settlement_osm": "place",
          "org_party": "org", "org_body": "org", "org_body_generic": "org",
          "procedural": "org"}
PRECEDENCE = ("place_city", "place_municipality", "place_admin_unit",
              "place_settlement_osm", "org_party", "org_body", "org_body_generic",
              "procedural")

# ---------------------------------------------------------------- S4 candidates
# City-independent. Written before any v2 count existed and filtered by attestation
# in the official Model Regulation, so it cannot be tuned to argos/orestiada.
PROCEDURAL_CANDIDATES: tuple[str, ...] = (
    "ημερήσια διάταξη", "ημερήσιας διάταξης", "εκτός ημερήσιας διάταξης",
    "απαρτία", "απαρτίας", "πρακτικά", "συνεδρίαση", "συνεδρίασης",
    "ειδική συνεδρίαση", "τακτική συνεδρίαση", "κατεπείγον", "τηλεδιάσκεψη",
    "δια περιφοράς", "πρόσκληση", "εισήγηση", "ψηφοφορία", "φανερή ψηφοφορία",
    "μυστική ψηφοφορία", "πλειοψηφία", "απόλυτη πλειοψηφία", "απόφαση",
    "αναβολή", "πρόταση", "παρών", "διαύγεια", "κανονισμός λειτουργίας",
    "αποχώρηση", "προσέλευση", "παραίτηση", "προεδρείο", "αντιπρόεδρος",
    "γραμματέας", "επικεφαλής", "παράταξη", "δημοτική παράταξη",
    "δήμαρχος", "αντιδήμαρχος", "δημοτικός σύμβουλος", "δημοτικό συμβούλιο",
    "δημοτική επιτροπή", "πρόεδρος", "συμβούλιο", "κοινότητα", "περιφέρεια",
    "απολογισμός", "μελέτη", "πίστωση", "δαπάνη", "παράταση", "θέμα",
    "μέλη", "διοικητικό συμβούλιο", "ανακοίνωση", "ερώτηση",
    "δημοτική αρχή", "δημότες", "υπηρεσία", "διάταξη", "νομοθεσία",
)


# ---------------------------------------------------------------------- fetching

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _cached(name: str, argv: list[str]) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        r = subprocess.run(argv, capture_output=True)
        if r.returncode != 0 or not r.stdout:
            raise SystemExit(f"{name}: rc={r.returncode} {r.stderr[:300]!r}")
        p.write_bytes(r.stdout)
    return p.read_bytes()


def get_json(url: str, name: str) -> tuple[object, str]:
    b = _cached(name, ["curl", "-sSfL", "--retry", "2", url])
    return json.loads(b), sha(b)


def get_pdf_text(url: str, name: str) -> tuple[str, str]:
    b = _cached(name, ["curl", "-sSfL", "--retry", "2", url])
    p = CACHE / name
    t = CACHE / (name + ".txt")
    if not t.exists():
        subprocess.run(["pdftotext", "-layout", str(p), str(t)], check=True)
    return t.read_text(encoding="utf-8"), sha(b)


def overpass(city: str) -> tuple[list[dict], str, str]:
    rel = CITY_SRC[city]["osm_rel"]
    q = (f"[out:json][timeout:120];rel({rel});map_to_area->.a;"
         f'(node["place"~"^({"|".join(PLACE_TYPES)})$"](area.a););out tags meta;')
    b = _cached(f"overpass-{city}.json",
                ["curl", "-sSf", "--retry", "2", "-X", "POST", OVERPASS,
                 "--data-urlencode", f"data={q}"])
    return json.loads(b)["elements"], sha(b), q


def hunspell_common() -> set[str]:
    b = _cached("el_GR.dic", ["curl", "-sSfL", "--retry", "2", HUNSPELL_URL])
    out = set()
    for line in b.decode("iso-8859-7").splitlines()[1:]:
        w = line.split("/")[0].strip()
        if w and w[:1].islower():
            k = norm_key(w)
            if k:
                out.add(k)
    return out


def seen_city_ngrams() -> collections.Counter:
    """n-gram counts over the 220 benchmark windows of the EIGHT non-validation cities.

    Rejection only. argos and orestiada references are never read here.
    """
    from eval.controlled_eval import bench_data as B
    man = json.loads(MANIFEST.read_text())
    rep = B.load_report(man["source_run"])
    c: collections.Counter = collections.Counter()
    n_win = n_tok = 0
    for it in rep["items"]:
        if it["cityId"] in ("argos", "orestiada"):
            continue
        n_win += 1
        toks = ftoks(it["referenceText"])
        n_tok += len(toks)
        for n in range(1, MAX_NGRAM + 1):
            for i in range(len(toks) - n + 1):
                c[" ".join(toks[i:i + n])] += 1
    c["__windows__"] = n_win
    c["__tokens__"] = n_tok
    return c


# ------------------------------------------------------------- ELSTAT admin units

ELSTAT_MUNI = re.compile(r"^\s*(?:[\d.]+\s+)?(ΔΗΜΟΣ\s+.+?)\s*(?:\(|$)")
ELSTAT_UNIT = re.compile(
    r"^\s*(?:[\d.]+\s+)?(ΔΗΜΟΤΙΚΗ ΕΝΟΤΗΤΑ\s+[^\d]+?)\s+[\d.]+\s*$")
ELSTAT_COMM = re.compile(
    r"^\s*(?:[\d.]+\s+)?((?:Δημοτική|Τοπική) Κοινότητα\s+[^\d]+?)\s+[\d.]+\s*$")
ELSTAT_STOP = re.compile(
    r"^\s*(?:[\d.]+\s+)?(ΠΕΡΙΦΕΡΕΙΑ|ΠΕΡΙΦΕΡΕΙΑΚΗ|ΑΠΟΚΕΝΤΡΩΜΕΝΗ|ΣΥΝΟΛΟ|ΕΛΛΑΔΑ)")


def elstat_units(text: str, municipality: str) -> list[dict]:
    """Municipal units and communities listed under one municipality heading."""
    out: list[dict] = []
    inside = False
    for raw in text.splitlines():
        line = raw.rstrip()
        m = ELSTAT_MUNI.match(line)
        if m:
            inside = re.sub(r"\s+", " ", m.group(1)).strip() == municipality
            continue
        if not inside:
            continue
        if ELSTAT_STOP.match(line):
            inside = False
            continue
        for rx, level in ((ELSTAT_UNIT, "municipal_unit"),
                          (ELSTAT_COMM, "community")):
            m = rx.match(line)
            if m:
                out.append({"name": re.sub(r"\s+", " ", m.group(1)).strip(),
                            "level": level})
                break
    return out


# ------------------------------------------------------------------- normalizing

def norm_key(text: str) -> str:
    return " ".join(ftoks(text))


def is_greek(s: str) -> bool:
    return bool(GREEK_RE.match(unicodedata.normalize("NFC", s or "")))


ADMIN_PREFIX = re.compile(
    r"^(ΔΗΜΟΤΙΚΗ ΕΝΟΤΗΤΑ|Δημοτική Ενότητα|Δημοτική Κοινότητα|Τοπική Κοινότητα"
    r"|Κοινότητα)\s+")


def lcp(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ------------------------------------------------------------------ term sources

def raw_terms(city: str, elstat_text: str,
              modelreg: collections.Counter) -> tuple[list[dict], dict]:
    people_url = f"{API}/cities/{city}/people"
    city_url = f"{API}/cities/{city}"
    people, people_sha = get_json(people_url, f"people-{city}.json")
    city_rec, city_sha = get_json(city_url, f"city-{city}.json")
    elements, osm_sha, osm_q = overpass(city)

    raw: list[dict] = []
    src_people = f"opencouncil:{people_url}"

    # --- S1a surnames: v1 rule verbatim, one term per distinct surname -----------
    by_surname: dict[str, dict] = {}
    for p in people:
        name = (p.get("name") or "").strip()
        if not name:
            continue
        surname = name.split()[-1]
        key = norm_key(surname)
        if not key:
            continue
        rec = by_surname.setdefault(key, {
            "id": f"person:{key.replace(' ', '_')}", "canonical": surname,
            "klass": "person_surname", "source": f"{src_people}#name.surname",
            "covers": [], "aliases": inflect(surname)})
        rec["covers"].append(name)
    raw.extend(sorted(by_surname.values(), key=lambda t: t["id"]))

    # --- S1b full names: atomic multi-token terms (G6) ---------------------------
    by_full: dict[str, dict] = {}
    for p in people:
        name = (p.get("name") or "").strip()
        parts = name.split()
        if len(parts) < 2:
            continue
        given, surname = parts[0], parts[-1]
        key = norm_key(name)
        if not key:
            continue
        aliases = []
        for g in inflect(given):
            for s in inflect(surname):
                aliases += [f"{g} {s}", f"{s} {g}"]
        by_full.setdefault(key, {
            "id": f"full:{key.replace(' ', '_')}", "canonical": name,
            "klass": "person_full_name",
            "source": f"{src_people}#name (given x surname, fixed suffix rules)",
            "covers": [name], "aliases": aliases})
    raw.extend(sorted(by_full.values(), key=lambda t: t["id"]))

    # --- S1c parties, S1d administrative bodies ----------------------------------
    for field, klass, prefix in (("party", "org_party", "party"),
                                 ("administrativeBody", "org_body", "body")):
        objs: dict[str, dict] = {}
        for p in people:
            for role in p.get("roles") or []:
                obj = role.get(field)
                if not obj or not (obj.get("name") or "").strip():
                    continue
                k = norm_key(obj["name"])
                if not k:
                    continue
                rec = objs.setdefault(k, {
                    "id": f"{prefix}:{k.replace(' ', '_')}",
                    "canonical": obj["name"].strip(), "klass": klass,
                    "source": f"{src_people}#roles[].{field}.name",
                    "covers": [obj["name"].strip()], "aliases": []})
                for kn in ("name", "name_short"):
                    v = (obj.get(kn) or "").strip()
                    if v and v not in rec["aliases"]:
                        rec["aliases"].append(v)
        raw.extend(sorted(objs.values(), key=lambda t: t["id"]))

    # --- S2 city and municipality name (v1 rule, verbatim) -----------------------
    for label, klass, value in (("city_name", "place_city", city_rec.get("name")),
                                ("municipality_name", "place_municipality",
                                 city_rec.get("name_municipality"))):
        if value:
            raw.append({"id": f"place:{label}", "canonical": value, "klass": klass,
                        "source": f"opencouncil:{city_url}#{label}",
                        "covers": [value], "aliases": [value]})

    # --- S3a ELSTAT municipal units and communities ------------------------------
    units = elstat_units(elstat_text, CITY_SRC[city]["elstat"])
    admin: dict[str, dict] = {}
    for u in units:
        head = ADMIN_PREFIX.sub("", u["name"]).strip()   # genitive, e.g. Δαλαμανάρας
        k = norm_key(head)
        if not k:
            continue
        rec = admin.setdefault(k, {
            "id": f"admin:{k.replace(' ', '_')}", "canonical": head,
            "klass": "place_admin_unit",
            "source": f"elstat2021:{CITY_SRC[city]['elstat']}/{u['level']}",
            "covers": [], "aliases": [head], "_levels": []})
        if u["name"] not in rec["covers"]:
            rec["covers"].append(u["name"])
            rec["aliases"].append(u["name"])
            rec["_levels"].append(u["level"])

    # --- S3b OSM nominatives; merged into the matching ELSTAT community ----------
    osm_used, osm_only, osm_meta = 0, {}, []
    for e in elements:
        name = (e["tags"].get("name") or "").strip()
        if not name:
            continue
        osm_meta.append({"type": e["type"], "id": e["id"],
                         "version": e.get("version"), "timestamp": e.get("timestamp"),
                         "place": e["tags"].get("place"), "name": name,
                         "name_el": e["tags"].get("name:el")})
        if not is_greek(name):
            continue
        k = norm_key(name)
        if not k:
            continue
        # mechanical merge: normalized first-token common prefix >= 5 chars, unique
        cands = [a for a in admin.values()
                 if lcp(k.split()[0], norm_key(a["canonical"]).split()[0]) >= 5]
        if len(cands) == 1:
            if name not in cands[0]["aliases"]:
                cands[0]["aliases"].append(name)
                cands[0]["covers"].append(f"osm:{name}")
            osm_used += 1
        elif k not in admin:
            osm_only.setdefault(k, {
                "id": f"osm:{k.replace(' ', '_')}", "canonical": name,
                "klass": "place_settlement_osm",
                "source": (f"osm:area(rel {CITY_SRC[city]['osm_rel']}) "
                           f"node[place={e['tags'].get('place')}] id={e['id']} "
                           f"v{e.get('version')}"),
                "covers": [name], "aliases": [name]})
    for a in admin.values():
        a.pop("_levels", None)
    raw.extend(sorted(admin.values(), key=lambda t: t["id"]))
    raw.extend(sorted(osm_only.values(), key=lambda t: t["id"]))

    # --- S4 procedural, attested in the official Model Regulation ----------------
    for phrase in PROCEDURAL_CANDIDATES:
        k = norm_key(phrase)
        if not k or modelreg.get(k, 0) == 0:
            continue
        raw.append({"id": f"proc:{k.replace(' ', '_')}", "canonical": phrase,
                    "klass": "procedural",
                    "source": "modelreg:ΦΕΚ Β' 109/2025 Πρότυπος Κανονισμός "
                              "Λειτουργίας Δημοτικού Συμβουλίου",
                    "covers": [phrase], "aliases": [phrase],
                    "modelreg_attestations": modelreg[k]})

    prov = {
        "opencouncil_people": {"url": people_url, "sha256": people_sha,
                               "n_people": len(people)},
        "opencouncil_city": {"url": city_url, "sha256": city_sha},
        "elstat2021": {"url": ELSTAT_URL, "heading": CITY_SRC[city]["elstat"],
                       "n_units": len(units)},
        "osm": {"endpoint": OVERPASS, "query": osm_q, "sha256": osm_sha,
                "relation": CITY_SRC[city]["osm_rel"], "retrieved": RETRIEVED,
                "n_elements": len(elements), "merged_into_elstat": osm_used,
                "osm_only_terms": len(osm_only), "elements": osm_meta},
        "modelreg": {"url": MODELREG_URL},
    }
    return raw, prov


# ---------------------------------------------------------------------- building

def merge_families(raw: list[dict]) -> tuple[list[dict], list[dict]]:
    """Union terms of the same family that share a normalized alias."""
    parent: dict[str, str] = {t["id"]: t["id"] for t in raw}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_alias: dict[tuple[str, str], str] = {}
    for t in raw:
        fam = FAMILY.get(t["klass"])
        if not fam:
            continue
        for a in t["aliases"]:
            k = norm_key(a)
            if not k:
                continue
            other = by_alias.setdefault((fam, k), t["id"])
            ra, rb = find(other), find(t["id"])
            if ra != rb:
                parent[rb] = ra

    groups: dict[str, list[dict]] = {}
    for t in raw:
        groups.setdefault(find(t["id"]), []).append(t)

    out, merges = [], []
    for members in groups.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        members.sort(key=lambda t: (PRECEDENCE.index(t["klass"])
                                    if t["klass"] in PRECEDENCE else 99, t["id"]))
        head = dict(members[0])
        for m in members[1:]:
            for a in m["aliases"]:
                if a not in head["aliases"]:
                    head["aliases"].append(a)
            head["covers"] = list(head["covers"]) + list(m["covers"])
            head["source"] = f"{head['source']} + {m['source']}"
        merges.append({"kept": head["id"], "klass": head["klass"],
                       "absorbed": [m["id"] for m in members[1:]]})
        out.append(head)
    return out, merges


def build_city(city: str, meetings: list[str], ngr: collections.Counter,
               common: set[str], elstat_text: str,
               modelreg: collections.Counter) -> dict:
    raw, prov = raw_terms(city, elstat_text, modelreg)

    # A generic administrative body ("Δημοτικό Συμβούλιο") is procedural vocabulary,
    # not a discriminative named entity. Decided mechanically by G2(a) on the seen
    # cities, so no city-specific judgement is involved.
    for t in raw:
        if (t["klass"] == "org_body"
                and ngr.get(norm_key(t["canonical"]), 0) >= COMMON_FREQ):
            t["klass"] = "org_body_generic"

    raw, merges = merge_families(raw)

    dropped: list[dict] = []

    def keep_alias(t: dict, a: str) -> str | None:
        k = norm_key(a)
        if not k:
            return None
        if not is_greek(a):
            dropped.append({"alias": a, "reason": "not_greek", "term": t["id"]})
            return None
        if len(k.replace(" ", "")) < MIN_ALIAS_LEN:
            dropped.append({"alias": a, "reason": "too_short", "term": t["id"]})
            return None
        if t["klass"] in NAMELIKE:
            n = ngr.get(k, 0)
            if n >= COMMON_FREQ and t["klass"] not in RULE_A_EXEMPT:
                dropped.append({"alias": a, "reason": "common_in_seen_cities",
                                "seen_city_count": n, "term": t["id"]})
                return None
            if t["klass"] in HUNSPELL_CLASSES and " " not in k and k in common:
                dropped.append({"alias": a, "reason": "common_noun_in_el_GR_dict",
                                "seen_city_count": n, "term": t["id"]})
                return None
        return k

    claims: dict[str, set[str]] = {}
    kept: dict[str, list[str]] = {}
    for t in raw:
        ks: list[str] = []
        for a in t["aliases"]:
            k = keep_alias(t, a)
            if k and k not in ks:
                ks.append(k)
        kept[t["id"]] = ks
        for k in ks:
            claims.setdefault(k, set()).add(t["id"])
    collisions = {k for k, ids in claims.items() if len(ids) > 1}

    terms = []
    for t in raw:
        ks = []
        for k in kept[t["id"]]:
            if k in collisions:
                dropped.append({"alias": k, "reason": "collision", "term": t["id"]})
            else:
                ks.append(k)
        if not ks:
            dropped.append({"alias": t["canonical"], "reason": "no_usable_alias",
                            "term": t["id"]})
            continue
        rec = {"id": t["id"], "canonical": t["canonical"], "klass": t["klass"],
               "cut": "procedure" if t["klass"] in CUTS["procedure"] else "entities",
               "source": t["source"], "covers": sorted(set(t["covers"])),
               "aliases": sorted(ks)}
        if "modelreg_attestations" in t:
            rec["modelreg_attestations"] = t["modelreg_attestations"]
        terms.append(rec)

    # G2(a) threshold sensitivity, computed on the seen corpus only.
    sens = {str(thr): sum(1 for t in raw if t["klass"] in NAMELIKE
                          for a in set(t["aliases"])
                          if ngr.get(norm_key(a), 0) >= thr)
            for thr in (3, 5, 10)}

    return {
        "city": city,
        "version": VERSION,
        "supersedes": f"research/ds_wer/terms/{city}.json (v1, 2026-08-11)",
        "retrieved": RETRIEVED,
        "cuts": {k: list(v) for k, v in CUTS.items()},
        "primary_cut": "entities",
        "provenance": prov,
        "rule": {
            "never": "no term is taken from, or filtered by, any model output, decode, "
                     "provider hypothesis or error analysis",
            "common_word_stoplist": {
                "applies_to": list(NAMELIKE),
                "rule_a": f"normalized alias (whole alias, any length) occurs >= "
                          f"{COMMON_FREQ} times in the {ngr.get('__windows__')} "
                          f"benchmark windows ({ngr.get('__tokens__')} tokens) of the "
                          f"eight cities that are NOT argos/orestiada. Rejection only.",
                "rule_a_sensitivity_alias_rejections": sens,
                "rule_b": "single-token place/organisation alias is a lowercase "
                          f"(common-noun) headword of el_GR hunspell ({HUNSPELL_URL})",
                "rule_b_deviation": "Codex advised recording rule_b as a diagnostic "
                                    "only. It is enforced here because the seen-city "
                                    "corpus is ~67k tokens, too small for rule_a to "
                                    "reject Γυμνό/Εξοχή/Λίμνες/Αεροδρόμιο; every "
                                    "rule_b rejection is listed in dropped_aliases.",
            },
            "given_names": "standalone given names are NOT terms; full names are "
                           "atomic multi-token terms instead (G6)",
            "aliases_from": "fixed Greek suffix rules in "
                            "scripts/build_ds_wer_terms.py::inflect for single-token "
                            "person names and their full-name cross product; "
                            "otherwise the source string verbatim",
            "osm_merge_rule": "an OSM nominative merges into the ELSTAT community "
                              "term when the normalized first tokens share a >= 5 "
                              "character prefix and the match is unique; otherwise it "
                              "becomes a separate place_settlement_osm term",
            "alias_normalization": "aliases are stored already normalized by the "
                                   "frozen scorer (eval.controlled_eval.eval_freeze."
                                   "ftoks); the matcher uses the same normalizer",
            "min_alias_chars": MIN_ALIAS_LEN,
            "collision_rule": "an alias claimed by more than one term is dropped from "
                              "every term that claims it",
        },
        "same_entity_merges": merges,
        "meetings_in_scope": meetings,
        "terms": sorted(terms, key=lambda t: t["id"]),
        "dropped_aliases": sorted(
            dropped, key=lambda d: (d["reason"], d["term"], str(d["alias"]))),
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    man = json.loads(MANIFEST.read_text())
    by_city: dict[str, list[str]] = {}
    for r in man["eval_windows"]:
        by_city.setdefault(r["city"], []).append(r["meeting_id"])

    elstat_text, elstat_sha = get_pdf_text(ELSTAT_URL, "elstat2021.pdf")
    modelreg_text, modelreg_sha = get_pdf_text(MODELREG_URL, "modelreg2025.pdf")
    mr_toks = ftoks(re.sub(r"-\n\s*", "", modelreg_text))
    modelreg: collections.Counter = collections.Counter()
    for n in range(1, MAX_NGRAM + 1):
        for i in range(len(mr_toks) - n + 1):
            modelreg[" ".join(mr_toks[i:i + n])] += 1

    ngr = seen_city_ngrams()
    common = hunspell_common()

    OUT.mkdir(parents=True, exist_ok=True)
    for city, meetings in sorted(by_city.items()):
        rec = build_city(city, sorted(set(meetings)), ngr, common,
                         elstat_text, modelreg)
        rec["provenance"]["elstat2021"]["sha256"] = elstat_sha
        rec["provenance"]["modelreg"]["sha256"] = modelreg_sha
        p = OUT / f"{city}.v2.json"
        text = json.dumps(rec, ensure_ascii=False, indent=1) + "\n"
        p.write_text(text)
        by_klass = collections.Counter(t["klass"] for t in rec["terms"])
        print(f"{city}: {len(rec['terms'])} terms, "
              f"{sum(len(t['aliases']) for t in rec['terms'])} aliases, "
              f"{len(rec['dropped_aliases'])} dropped")
        for k, v in sorted(by_klass.items()):
            print(f"    {k:24} {v}")
        print(f"  sha256 {sha(text.encode())} -> {p}")


if __name__ == "__main__":
    main()
