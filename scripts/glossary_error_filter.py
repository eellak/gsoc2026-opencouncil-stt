"""Stage 2 of issue #20: filter the error-mined candidates and tier them.

The confusability machinery is imported from `scripts/glossary_distill.py` rather than
re-implemented, so the thresholds that cut ΑΜΕΑ in #19 (88 / 90 against the same
background) cut the same way here. Nothing in this file relaxes them.

New relative to #19, pre-registered on issue #20 and reviewed by Codex (job
316755b6, effort high) before any number was produced:

  C4   city-self-name filter, category-aware — #19 accepted ΒΡΙΛΗΣΣΙΩΝ, which is only
       the city's own name in capitals; the user rejected it. An exact roster person
       match is exempt (a councillor named Βούλα is not the suburb).
  external toponym evidence for all 11 cities (OSM place nodes, street names, named
       public facilities) instead of the two cities #19 could reach. A match on a
       *component* of an odonym only counts as strong when the unchanged neighbouring
       tokens complete that odonym.
  ranking by BENEFIT — how many times a human corrected *to* this term — not by
       corpus frequency.

Structure follows Codex's correction: one global candidate identity, then per-city
eligibility (C0, C4, evidence), then global C2/C3/C5, then a single tier precedence.
C0 is applied *after* aggregation and per (city, key), so a term that is already in a
city's frozen list is recorded as `existing_term_with_error_evidence` instead of
vanishing — a frozen term the model keeps getting wrong is a finding, not a duplicate.

Outputs (local, gitignored — mined term strings):
  data/glossary/candidates_error_mined.json
  data/glossary/candidates_error_mined.summary.json   (aggregates; safe to quote)

Run:  .venv-eval/bin/python scripts/glossary_error_filter.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from glossary_distill import (  # noqa: E402
    COMMON_ZIPF_FLOOR, MIN_CHARS_INJECTION, MIN_CHARS_METRIC, PHRASE_REJECT,
    PHRASE_REVIEW, WORD_REJECT, WORD_REVIEW, ZIPF_MAX_ENTITY, ZIPF_MAX_ROSTER,
    ZIPF_MAX_PHRASE_CONTENT, _ACRONYM_RE, _FUNCTION_WORDS, build_bg_ngrams,
    build_common, confusability, hunspell_headwords, match_norm, nfc_lower,
    zipf_max,
)

RAW = ROOT / "data/glossary/error_pairs.raw.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"
ROSTERS = ROOT / "data/pii/rosters_full.json"
POOL_A = ROOT / "data/glossary/candidates_pool_a.json"
POOL_B = ROOT / "data/glossary/candidates_pool_b.json"
OFFICIAL = ROOT / "research/glossary/official-sources-2026-08-16.json"
GAZ_CACHE = Path.home() / ".cache/oc-public/glossary-error"
MODELREG = Path.home() / ".cache/oc-public/ds-wer-v2/modelreg2025.pdf.txt"
OUT = ROOT / "data/glossary/candidates_error_mined.json"
OUT_SUM = ROOT / "data/glossary/candidates_error_mined.summary.json"

# ------------------------------------------------------------------ pre-registered
INJ_MIN_CORRECTIONS = 2      # the model got it wrong at least twice ...
INJ_MIN_UTTERANCES = 2       # ... in two different utterances ...
INJ_MIN_MEETINGS = 2         # ... in two different meetings
MIN_GAZ_TOKEN_LEN = 4
REVIEW_BACKLOG_CAP = 60      # internal-only singletons the user could still skim

# C4: a candidate proposed for city X that is X's own name is a false positive of the
# shape rule. Stems are unaccented, lowercase, final sigma folded.
CITY_SELF_STEMS = {
    "athens": ("αθην",),
    "chania": ("χανι", "χανιω"),
    "sparta": ("σπαρτ",),
    "xylokastro": ("ξυλοκαστρ", "ευρωστιν"),
    "zografou": ("ζωγραφ",),
    "vrilissia": ("βριλησσ",),
    "chalandri": ("χαλανδρ",),
    "samothraki": ("σαμοθρακ",),
    "argos": ("αργος", "αργους", "μυκην"),
    "orestiada": ("ορεστιαδ",),
    "vari-voula-vouliagmeni": ("βαρη", "βουλα", "βουλιαγμεν"),
}

_GREEK_RE = re.compile(r"[Α-Ωα-ωΆ-Ώά-ώΪΫϊϋΐΰ]")

LABEL_TO_CATEGORY = {
    "place_name": "toponym",
    "person_name": "person",
    "org_party_name": "org",
    "acronym_abbreviation": "acronym",
    "legal_admin_term": "legal_procedural",
}

# Evidence strength. Only `external` may reach the injection tier; an acronym whose
# expansion is not attested outside the transcripts is `weak_external` per Codex.
STRENGTH = {
    "roster": "external",
    "gazetteer_osm": "external",
    "official_source": "external",
    "modelreg": "external",
    "gazetteer_osm_component": "weak_external",
    "acronym_shape": "weak_external",
    "llm_label+caps": "internal",
    "none": "none",
}


# --------------------------------------------------------------------- evidence
def load_gazetteer(hun: set[str]) -> tuple[dict[str, set[str]], dict[str, dict],
                                           dict[str, dict]]:
    """Per city: full normalised OSM names, and an index of their long components.

    A component that is an ordinary Greek word (hunspell headword) is not indexed:
    «Οδού Νικηταρά» matched the generic word «οδού» inside an unrelated OSM name and
    came out as a toponym with evidence. Codex caught it in the final-set review.
    """
    full: dict[str, set[str]] = defaultdict(set)
    meta: dict[str, dict] = defaultdict(dict)
    comp: dict[str, dict] = defaultdict(dict)
    for p in sorted(GAZ_CACHE.glob("overpass-*.json")):
        city = p.stem[len("overpass-"):]
        for el in json.loads(p.read_text())["elements"]:
            tags = el.get("tags") or {}
            name = tags.get("name")
            if not name or not re.search(r"[Α-Ωα-ωΆ-Ώά-ώ]", name):
                continue
            kind = ("place_osm" if tags.get("place")
                    else "street_osm" if tags.get("highway")
                    else "facility_osm")
            n = match_norm(name)
            full[city].add(n)
            meta[city].setdefault(n, {"kind": kind, "osm_name": name})
            toks = n.split()
            if len(toks) > 1:
                for tok in toks:
                    if len(tok) >= MIN_GAZ_TOKEN_LEN and tok not in hun:
                        comp[city].setdefault(tok, {"kind": kind, "osm_name": name,
                                                    "full": n})
    return full, meta, comp


def load_rosters() -> dict[str, set[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    if not ROSTERS.exists():
        return idx
    for key, names in json.loads(ROSTERS.read_text()).items():
        city = key.split("/")[0]
        for nm in names:
            for tok in match_norm(nm).split():
                if len(tok) >= MIN_GAZ_TOKEN_LEN:
                    idx[city].add(tok)
    return idx


def load_frozen() -> tuple[dict[str, set[str]], set[str], dict[str, set[str]]]:
    per_city: dict[str, set[str]] = defaultdict(set)
    person_toks: dict[str, set[str]] = defaultdict(set)
    for p in sorted(TERMS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        city = d["city"]
        for t in d["terms"]:
            forms = [t["canonical"], *t.get("aliases", [])]
            for a in forms:
                per_city[city].add(match_norm(a))
            if t.get("klass", "").startswith("person"):
                for a in forms:
                    for tok in match_norm(a).split():
                        if len(tok) >= MIN_GAZ_TOKEN_LEN:
                            person_toks[city].add(tok)
    already19: set[str] = set()
    for p in (POOL_A, POOL_B):
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["candidates"]:
            if r.get("tier") in (None, "accepted"):
                already19.add(match_norm(r["term"]))
    return per_city, already19, person_toks


def load_official() -> dict[str, dict]:
    if not OFFICIAL.exists():
        return {}
    return {match_norm(t["term"]): t for t in json.loads(OFFICIAL.read_text())["terms"]}


def gaz_context_hit(tn: str, city: str, left: list[str], right: list[str],
                    full: dict[str, set[str]], comp: dict[str, dict]):
    """Component match, promoted to strong only if the neighbours complete the name.

    «Νέγρη» alone is not a place; «Φωκίωνος Νέγρη» is. Codex asked for exactly this.
    """
    hit = None
    for tok in tn.split():
        if tok in comp.get(city, {}):
            hit = comp[city][tok]
            break
    if hit is None:
        return None, None
    for lc in left or [""]:
        for rc in right or [""]:
            for li in range(len(lc.split()) + 1):
                lpart = " ".join(lc.split()[len(lc.split()) - li:]) if li else ""
                for ri in range(len(rc.split()) + 1):
                    rpart = " ".join(rc.split()[:ri]) if ri else ""
                    cand = " ".join(x for x in (match_norm(lpart), tn,
                                                match_norm(rpart)) if x)
                    if cand != tn and cand in full.get(city, set()):
                        return "gazetteer_osm", {"completed_as": cand, **hit}
    return "gazetteer_osm_component", hit


def merge_lanes(lanes: dict[str, list[dict]]) -> list[dict]:
    """One global candidate identity per normalised key, per Codex's correction.

    The same term is often corrected once inside the near gate and once outside it;
    two records would double the review list and split its benefit count. Counts are
    summed, meetings and cities unioned, and the safety gate later reads the `near`
    lane's own counts so a hard-lane occurrence can never buy an injection.
    """
    merged: dict[str, dict] = {}
    for lane in ("near", "hard", "deletion"):
        for r in lanes[lane]:
            key = match_norm(r["canonical"])
            m = merged.get(key)
            if m is None:
                m = merged[key] = {
                    "key": key, "canonical": r["canonical"], "lane": lane,
                    "lanes": {}, "display_aliases": [], "wrong_forms": {},
                    "labels": defaultdict(int), "cities": defaultdict(int),
                    "meetings": set(), "dates": set(),
                    "n_corrections": 0, "n_utterances": 0,
                    "left_ctx": [], "right_ctx": [],
                    "cap_on_span": 0, "span_seen": 0, "_top": 0}
            for side in ("left_ctx", "right_ctx"):
                for x in r[side]:
                    if x not in m[side]:
                        m[side].append(x)
            m["lanes"][lane] = {"n_corrections": r["n_corrections"],
                                "n_utterances": r["n_utterances"],
                                "n_meetings": r["n_meetings"]}
            if r["n_corrections"] > m["_top"]:
                m["_top"] = r["n_corrections"]
                m["canonical"] = r["canonical"]
            m["n_corrections"] += r["n_corrections"]
            m["n_utterances"] += r["n_utterances"]
            m["cap_on_span"] += r["cap_on_span"]
            m["span_seen"] += r["span_seen"]
            m["meetings"].update(r["source_meetings"])
            m["dates"].add(r["n_dates"])
            for a in r["display_aliases"]:
                if a not in m["display_aliases"]:
                    m["display_aliases"].append(a)
            for w, n in r["wrong_forms"].items():
                m["wrong_forms"][w] = m["wrong_forms"].get(w, 0) + n
            for k, n in r["labels"].items():
                m["labels"][k] += n
            for c, n in r["cities"].items():
                m["cities"][c] += n
    out = []
    for m in merged.values():
        lane = ("near" if "near" in m["lanes"]
                else "hard" if "hard" in m["lanes"] else "deletion")
        out.append({**m, "lane": lane, "labels": dict(m["labels"]),
                    "cities": dict(sorted(m["cities"].items(), key=lambda kv: -kv[1])),
                    "n_meetings": len(m["meetings"]),
                    "n_dates": max(m["dates"]) if m["dates"] else 0,
                    "n_cities": len(m["cities"]),
                    "source_meetings": sorted(m["meetings"]),
                    "wrong_forms": dict(sorted(m["wrong_forms"].items(),
                                               key=lambda kv: -kv[1])[:6])})
    for m in out:
        m.pop("meetings", None)
        m.pop("dates", None)
        m.pop("_top", None)
    return out


def main() -> None:
    from rapidfuzz import fuzz
    from wordfreq import iter_wordlist, zipf_frequency

    raw = json.loads(RAW.read_text())
    common, common_z = build_common(zipf_frequency, iter_wordlist)
    bg = build_bg_ngrams()
    # Sorted, not set: `confusability` scans these in iteration order and keeps
    # the first best hit, so an unordered set makes the frozen sha256 depend on
    # PYTHONHASHSEED. Verdicts never moved; the reported `nearest` did.
    common = sorted(common)
    bg = sorted(bg)
    hun = hunspell_headwords()
    gaz_full, gaz_meta, gaz_comp = load_gazetteer(hun)
    roster = load_rosters()
    frozen, already19, frozen_person = load_frozen()
    official = load_official()
    modelreg_norm = (" " + match_norm(MODELREG.read_text(errors="replace")) + " "
                     if MODELREG.exists() else "")
    print(f"common={len(common):,} bg_ngrams={len(bg):,} hunspell={len(hun):,} "
          f"gaz_cities={len(gaz_full)} roster_cities={len(roster)} "
          f"official={len(official)}", flush=True)

    rows: list[dict] = []
    existing_idx: dict[str, dict] = {}
    c4_audit: list[dict] = []
    reasons: dict[str, int] = defaultdict(int)

    for r in merge_lanes(raw["lanes"]):
        term = r["canonical"]
        tn = r["key"]
        lane = r["lane"]
        rec = {"term": term, "key": tn, "lane": lane,
               "lanes": r["lanes"],
               "display_aliases": r["display_aliases"],
               "n_corrections": r["n_corrections"],
               "n_utterances": r["n_utterances"],
               "n_meetings": r["n_meetings"], "n_dates": r["n_dates"],
               "n_cities": r["n_cities"], "cities": r["cities"],
               "source_meetings": r["source_meetings"],
               "labels": r["labels"], "wrong_forms": r["wrong_forms"],
               "cap_on_span": r["cap_on_span"], "span_seen": r["span_seen"]}

        # C7: Latin-script brand names (Vodafone, OpenCouncil, Hellas Direct).
        # Real terms, but a Greek frequency list cannot judge them and the
        # error is transliteration, not lexical confusion. Recorded, not proposed.
        if not _GREEK_RE.search(term):
            reasons["C7 non-Greek script"] += 1
            continue

        label_cat = None
        if r["labels"]:
            label_cat = LABEL_TO_CATEGORY.get(
                max(r["labels"].items(), key=lambda kv: kv[1])[0])

        # --------------------------------------------- per-city eligibility
        city_recs = {}
        for c in r["cities"]:
            cr = {"c0_frozen": False, "c4": None, "evidence": "none",
                  "detail": None, "category": label_cat or "unknown"}
            roster_hit = any(tok in roster.get(c, set())
                             or tok in frozen_person.get(c, set())
                             for tok in tn.split())
            # Order matters: a great many Athens street names ARE surnames, so the
            # roster and the gazetteer collide. The LLM row label breaks the tie
            # — it is the only witness that saw the sentence.
            if tn in official:
                o = official[tn]
                cr["evidence"], cr["category"] = "official_source", o["category"]
                cr["detail"] = o["source"]
            elif roster_hit and label_cat == "person":
                cr["evidence"], cr["category"] = "roster", "person"
                cr["detail"] = c
            elif tn in gaz_full.get(c, set()):
                cr["evidence"] = "gazetteer_osm"
                cr["category"] = ("org"
                                  if gaz_meta[c][tn]["kind"] == "facility_osm"
                                  else "toponym")
                cr["detail"] = gaz_meta[c][tn]
            elif roster_hit:
                cr["evidence"], cr["category"] = "roster", "person"
                cr["detail"] = c
            else:
                ev, det = gaz_context_hit(tn, c, r["left_ctx"], r["right_ctx"],
                                          gaz_full, gaz_comp)
                if ev:
                    cr["evidence"] = ev
                    # ΕΛΜΕΠΑ is an institution that happens to sit inside the OSM
                    # name «Θέατρο ΕΛΜΕΠΑ»; it is not a place name.
                    cr["category"] = ("org" if det.get("kind") == "facility_osm"
                                      else "toponym")
                    cr["detail"] = det
                elif " " in term and modelreg_norm and f" {tn} " in modelreg_norm:
                    cr["evidence"], cr["category"] = "modelreg", "legal_procedural"
                elif _ACRONYM_RE.match(term) and tn not in hun:
                    cr["evidence"], cr["category"] = "acronym_shape", "acronym"
                elif label_cat in ("toponym", "person", "org") \
                        and r["cap_on_span"] > 0:
                    cr["evidence"] = "llm_label+caps"
            # C0 after aggregation, per (city, key)
            if tn in frozen.get(c, set()) or tn in already19:
                cr["c0_frozen"] = True
            # C4, exempting a candidate that actually resolved to a roster person.
            # The exemption keys on the resolved evidence, not on a bare roster
            # token hit: «Άνω Βριλήσσια» touches a roster token in vrilissia and
            # would otherwise slip past its own city's name filter.
            if cr["evidence"] != "roster":
                for stem in CITY_SELF_STEMS.get(c, ()):
                    if any(w.startswith(stem) for w in tn.split()):
                        cr["c4"] = f"{c}:{stem}"
                        break
            city_recs[c] = cr
        rec["per_city_eligibility"] = city_recs

        eligible = [c for c, cr in city_recs.items()
                    if not cr["c0_frozen"] and not cr["c4"]]
        if not eligible:
            if any(cr["c0_frozen"] for cr in city_recs.values()):
                reasons["C0 already frozen"] += 1
                if r["n_corrections"] >= 2:
                    prev = existing_idx.get(tn)
                    if prev is None:
                        existing_idx[tn] = {
                            "term": term, "lanes": [lane],
                            "n_corrections": r["n_corrections"],
                            "n_meetings": r["n_meetings"],
                            "cities": dict(r["cities"])}
                    else:
                        prev["lanes"].append(lane)
                        prev["n_corrections"] += r["n_corrections"]
                        prev["n_meetings"] = max(prev["n_meetings"],
                                                 r["n_meetings"])
                        for c, n in r["cities"].items():
                            prev["cities"][c] = prev["cities"].get(c, 0) + n
            else:
                reasons["C4 city self-name"] += 1
                c4_audit.append({
                    "term": term, "n_corrections": r["n_corrections"],
                    "cities": r["cities"],
                    "rule": [cr["c4"] for cr in city_recs.values() if cr["c4"]],
                    "evidence": [cr["evidence"] for cr in city_recs.values()]})
            continue
        rec["eligible_cities"] = eligible
        best = max(eligible,
                   key=lambda c: ("external weak_external internal none".split()
                                  .index(STRENGTH[city_recs[c]["evidence"]]) * -1,
                                  r["cities"][c]))
        evidence = city_recs[best]["evidence"]
        category = city_recs[best]["category"]
        rec["category"] = category
        rec["category_evidence"] = evidence
        rec["evidence_detail"] = city_recs[best]["detail"]
        rec["evidence_city"] = best
        strength = STRENGTH[evidence]
        rec["evidence_strength"] = strength

        # ------------------------------------------------------- C2 rarity
        z = zipf_max(term, zipf_frequency)
        rec["zipf"] = round(z, 2)
        if " " in term:
            content = [w for w in nfc_lower(term).split()
                       if w not in _FUNCTION_WORDS]
            zs = [zipf_max(w, zipf_frequency) for w in content] or [9.0]
            rec["zipf_content_min"] = round(min(zs), 2)
            rec["c2"] = "reject" if min(zs) > ZIPF_MAX_PHRASE_CONTENT else "clear"
        else:
            cap = ZIPF_MAX_ROSTER if category == "person" else ZIPF_MAX_ENTITY
            rec["zipf_cap"] = cap
            if z > cap:
                rec["c2"] = "reject"
            elif z == 0.0 and category != "acronym":
                # #19's rule, kept verbatim: zipf 0.0 means "out of wordfreq's
                # list", not "rare", so the term needs a witness before anything may
                # substitute it automatically. I proposed exempting candidates with
                # external (gazetteer) evidence and Codex rejected it as post hoc
                # (job 7397f2a7): OSM shows that a name occurs locally, not that OSM
                # holds the authoritative spelling, nor that the observed mention
                # refers to that entity. The exemption survives only as a labelled
                # sensitivity count, never as a tier.
                rec["c2"] = "needs_validation"
            else:
                rec["c2"] = "clear"
        if rec["c2"] == "reject":
            reasons["C2 rarity"] += 1
            continue

        # -------------------------------------------------- C3 confusability
        c3 = confusability(term, common, common_z, bg, fuzz)
        rec["confusability"] = c3
        rec["c3"] = c3["verdict"]
        if c3["verdict"] == "reject":
            reasons["C3 confusable"] += 1
            continue

        # --------------------------------------------------- C5 length, tier
        nchars = len(tn.replace(" ", ""))
        rec["n_chars"] = nchars
        # A term the human left lowercase is probably not a proper name, even
        # when the gazetteer matches it: «φαναριού» is a traffic light, and OSM
        # also has a street Φαναρίου. Flagged for the reviewer, and it blocks
        # the injection tier; it does not drop the candidate.
        rec["lowercase_in_gold"] = (rec["cap_on_span"] == 0
                                    and category in ("toponym", "person", "org"))
        # Codex's safety point on the metric tier: a wrong->right substitution table
        # is only safe if the WRONG form is never legitimate. «ΜΜΕ», «ΣΟΥ», «ΕΙΔΑ»,
        # «MAP» and «αφημί» all are. Flagged per observed wrong form, so no downstream
        # table can be built from this file without seeing it.
        rec["ambiguous_wrong_forms"] = sorted(
            w for w in rec["wrong_forms"]
            if match_norm(w) in hun
            or zipf_max(w, zipf_frequency) >= COMMON_ZIPF_FLOOR
            or (_ACRONYM_RE.match(w) and len(w) >= MIN_CHARS_METRIC)
            or not _GREEK_RE.search(w))
        rec["metric_eligible"] = nchars >= MIN_CHARS_METRIC and strength != "none"
        rec["injection_eligible"] = (nchars >= MIN_CHARS_INJECTION
                                     and rec["c3"] == "clear"
                                     and rec["c2"] == "clear"
                                     and strength == "external"
                                     and not rec["lowercase_in_gold"]
                                     and lane == "near")
        if strength == "none":
            reasons["C6 no category evidence"] += 1
            continue
        # Codex rejected the zipf==0 exemption as post hoc, so it is not a tier. It
        # is still worth reporting how many candidates it WOULD have promoted, as a
        # labelled sensitivity arm alongside the pre-specified result.
        rec["sensitivity_zipf0_external"] = (
            rec["c2"] == "needs_validation" and strength == "external"
            and nchars >= MIN_CHARS_INJECTION and rec["c3"] == "clear"
            and lane == "near" and not rec["lowercase_in_gold"])
        recurrent = (r["n_corrections"] >= INJ_MIN_CORRECTIONS
                     and r["n_utterances"] >= INJ_MIN_UTTERANCES
                     and r["n_meetings"] >= INJ_MIN_MEETINGS)
        rec["recurrent"] = recurrent
        # The safety gate reads the near lane alone: a hard-lane occurrence is
        # exactly the case we are not confident about, so it must not be able to
        # buy a term its second "independent" correction.
        nl = r["lanes"].get("near", {})
        rec["recurrent_near"] = (nl.get("n_corrections", 0) >= INJ_MIN_CORRECTIONS
                                 and nl.get("n_utterances", 0) >= INJ_MIN_UTTERANCES
                                 and nl.get("n_meetings", 0) >= INJ_MIN_MEETINGS)
        if lane == "deletion":
            # A substitution list cannot re-insert a word the model never emitted.
            rec["tier"] = "metric-only" if (strength == "external" and recurrent) \
                else "review"
            rec["injection_eligible"] = False
        elif rec["injection_eligible"] and rec["recurrent_near"]:
            rec["tier"] = "injection"
        elif strength in ("external", "weak_external") and recurrent:
            rec["tier"] = "metric-only"
        elif strength in ("external", "weak_external"):
            rec["tier"] = "review"
        elif recurrent:
            rec["tier"] = "review"
        else:
            rec["tier"] = "review-backlog"
        rows.append(rec)

    for r in rows:
        if r["tier"] != "review":
            r["review_priority"] = None
        elif r["n_corrections"] >= INJ_MIN_CORRECTIONS:
            r["review_priority"] = 1
        elif (r["category_evidence"] == "gazetteer_osm" and r["lane"] == "near"
              and r["span_seen"] > 0 and r["cap_on_span"] == r["span_seen"]):
            r["review_priority"] = 2
        else:
            r["review_priority"] = 3

    def benefit(r):
        return (-r["n_corrections"], -r["n_meetings"], -r["n_cities"], r["term"])

    rows.sort(key=benefit)
    backlog = [r for r in rows if r["tier"] == "review-backlog"]
    for i, r in enumerate(backlog):
        if i >= REVIEW_BACKLOG_CAP:
            r["tier"] = "review-backlog-overflow"
    existing = sorted(existing_idx.values(),
                      key=lambda r: (-r["n_corrections"], r["term"]))

    crit = {
        "preregistered_on": "https://github.com/eellak/gsoc2026-opencouncil-stt/issues/20",
        "codex_review_job": "316755b6ffc647e89f7cfff7726e6d6c",
        "inherits_from": "scripts/glossary_distill.py (issue #19)",
        **{k: raw["source"][k] for k in raw["source"]},
        "zipf_max_entity": ZIPF_MAX_ENTITY, "zipf_max_roster": ZIPF_MAX_ROSTER,
        "zipf_max_phrase_content": ZIPF_MAX_PHRASE_CONTENT,
        "common_zipf_floor": COMMON_ZIPF_FLOOR,
        "word_reject": WORD_REJECT, "word_review": WORD_REVIEW,
        "phrase_reject": PHRASE_REJECT, "phrase_review": PHRASE_REVIEW,
        "min_chars_metric": MIN_CHARS_METRIC,
        "min_chars_injection": MIN_CHARS_INJECTION,
        "injection_min_corrections": INJ_MIN_CORRECTIONS,
        "injection_min_utterances": INJ_MIN_UTTERANCES,
        "injection_min_meetings": INJ_MIN_MEETINGS,
        "review_backlog_cap": REVIEW_BACKLOG_CAP,
        "city_self_name_filter": {k: list(v) for k, v in CITY_SELF_STEMS.items()},
        "evidence_strength": STRENGTH,
        "background": {"frequency": "wordfreq el", "lexicon": "el_GR.dic",
                       "ngrams_from": ["cv-el", "stoma"],
                       "gazetteer": "overpass-api.de admin_level=7, 11 cities, "
                                    "2026-08-16",
                       "official_sources":
                           "research/glossary/official-sources-2026-08-16.json",
                       "n_common_forms": len(common), "n_bg_ngrams": len(bg),
                       "n_hunspell_headwords": len(hun)},
    }
    OUT.write_text(json.dumps(
        {"criteria": crit, "mining_stats": raw["stats"],
         "n_input": raw["n_candidates"], "n_survivors": len(rows),
         "rejections": dict(reasons),
         "existing_term_with_error_evidence": existing,
         "rejected_c4_city_self_name": c4_audit,
         "candidates": rows}, ensure_ascii=False, indent=1) + "\n")

    def tally(key):
        d = defaultdict(int)
        for r in rows:
            d[r[key]] += 1
        return dict(sorted(d.items()))

    tier_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tier_lane: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    corr_by_tier: dict[str, int] = defaultdict(int)
    for r in rows:
        tier_cat[r["tier"]][r["category"]] += 1
        tier_lane[r["tier"]][r["lane"]] += 1
        corr_by_tier[r["tier"]] += r["n_corrections"]
    cities = defaultdict(int)
    for r in rows:
        for c in r["eligible_cities"]:
            cities[c] += 1
    summary = {
        "criteria": crit,
        "mining_stats": raw["stats"],
        "funnel": {
            "mined_keys": raw["n_candidates"],
            "rejections": dict(sorted(reasons.items())),
            "n_survivors": len(rows),
            "existing_terms_with_error_evidence": len(existing),
        },
        "by_tier": tally("tier"),
        "by_category": tally("category"),
        "by_evidence": tally("category_evidence"),
        "by_lane": tally("lane"),
        "by_tier_category": {k: dict(sorted(v.items()))
                             for k, v in sorted(tier_cat.items())},
        "by_tier_lane": {k: dict(sorted(v.items()))
                         for k, v in sorted(tier_lane.items())},
        "corrections_by_tier": dict(sorted(corr_by_tier.items())),
        "cities_touched": dict(sorted(cities.items())),
        "n_injection_eligible": sum(1 for r in rows if r["injection_eligible"]),
        "review_queue_first_sitting": sum(
            1 for r in rows if r.get("review_priority") in (1, 2)),
        "review_queue_deferred": sum(
            1 for r in rows if r.get("review_priority") == 3),
        "n_with_ambiguous_wrong_form": sum(
            1 for r in rows if r["ambiguous_wrong_forms"]),
        "sensitivity_zipf0_external": sum(
            1 for r in rows if r.get("sensitivity_zipf0_external") and r["recurrent"]),
        "n_metric_eligible": sum(1 for r in rows if r["metric_eligible"]),
        "corrections_covered": sum(r["n_corrections"] for r in rows),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    OUT_SUM.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")

    # A flat sheet for the human pass: one line per candidate, benefit-ordered, with
    # the wrong form the model produced and the reason it is not a common word.
    sheet = ["\t".join(["tier", "n_corrections", "n_meetings", "cities", "term",
                        "category", "evidence", "evidence_detail", "wrong_forms",
                        "zipf", "nearest_common", "fuzz", "lane", "caps",
                        "review_priority", "ambiguous_wrong_forms"])]
    for r in rows:
        det = r["evidence_detail"]
        det = det.get("osm_name", "") if isinstance(det, dict) else (det or "")
        sheet.append("\t".join([
            r["tier"], str(r["n_corrections"]), str(r["n_meetings"]),
            ",".join(r["eligible_cities"]), r["term"], r["category"],
            r["category_evidence"], str(det),
            " | ".join(list(r["wrong_forms"])[:3]), f"{r['zipf']:.2f}",
            str(r["confusability"]["nearest"] or ""),
            str(r["confusability"]["score"]), r["lane"],
            f"{r['cap_on_span']}/{r['span_seen']}",
            str(r.get("review_priority") or ""),
            " | ".join(r["ambiguous_wrong_forms"])]))
    (ROOT / "data/glossary/review_sheet_error_mined.tsv").write_text(
        "\n".join(sheet) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "criteria"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
