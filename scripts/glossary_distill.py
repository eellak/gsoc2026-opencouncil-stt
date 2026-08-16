"""Distil the 2026-06-20 glossary into a short, auditable candidate list.

The criteria are pre-registered on issue #19 (comment 5306477417) and were reviewed by
Codex (job 45db933d, effort high) *before* a single number was produced. They are
constants at the top of this file; changing one after seeing a result invalidates the
pre-registration, so don't.

Two pools:
  A  per-city additions distilled from `data/glossary/glossary.json`
  B  universal / cross-city council terms, mined by scripts/glossary_poolb_mine.py from
     meetings that are in the TRAIN fold *and* absent from the public benchmark

Two eligibility flags per surviving term, because a domain-WER lexicon and a prompt
injection list do not carry the same risk:
  metric_eligible     may enter ds-wer (3-char acronyms allowed)
  injection_eligible  may enter a prompt / hotword list (>=4 chars, clean C3)

Outputs (local, gitignored — they contain mined term strings):
  data/glossary/candidates_pool_a.json
  data/glossary/candidates_pool_b.json
  data/glossary/candidates.summary.json   (aggregates only; safe to quote)

Run:  <venv>/bin/python scripts/glossary_distill.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ pre-registered
ZIPF_MAX_ENTITY = 4.0        # acronym / toponym / institutional / legal single token
ZIPF_MAX_ROSTER = 3.5        # roster proper names
ZIPF_MAX_PHRASE_CONTENT = 3.5   # at least one content word must be this rare
COMMON_ZIPF_FLOOR = 3.5      # background common-surface-form set goes down to here
COMMON_ZIPF_ASSERT = 4.0     # nothing at or above this may be missing from that set

WORD_REJECT = 88             # eval/glossary.py select_glossary_terms word_cutoff
WORD_REVIEW = 85
PHRASE_REJECT = 90           # ... phrase_cutoff
PHRASE_REVIEW = 87

CITY_MIN_MEETINGS = 3
CITY_MIN_DATES = 2
UNIVERSAL_MIN_CITIES = 5
UNIVERSAL_MIN_CITY_MEETINGS = 10
CROSS_CITY_MIN_CITIES = 3

MIN_CHARS_METRIC = 3
MIN_CHARS_INJECTION = 4

GLOSSARY = ROOT / "data/glossary/glossary.json"
POOLB_RAW = ROOT / "data/glossary/poolb_acronyms.raw.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"
HUNSPELL = Path.home() / ".cache/oc-public/ds-wer-v2/el_GR.dic"
MODELREG = Path.home() / ".cache/oc-public/ds-wer-v2/modelreg2025.pdf.txt"
BG_SENTENCES = [
    Path.home() / ".cache/oc-public/cv-el/filtered.jsonl",
    Path.home() / ".cache/oc-public/stoma/filtered.jsonl",
]
OUT_A = ROOT / "data/glossary/candidates_pool_a.json"
OUT_B = ROOT / "data/glossary/candidates_pool_b.json"
OUT_SUM = ROOT / "data/glossary/candidates.summary.json"

_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΆΈΉΊΌΎΏΪΫ"
_ACRONYM_RE = re.compile(rf"^[{_UPPER}]{{2,}}$")
_FUNCTION_WORDS = {
    "ο", "η", "το", "οι", "τα", "του", "της", "των", "τον", "την", "τους", "τις",
    "και", "με", "για", "σε", "στο", "στη", "στην", "στον", "στους", "στα", "στις",
    "απο", "από", "προς", "κατα", "κατά", "ως", "να", "θα", "δεν", "μη", "μην", "ή",
}


# ------------------------------------------------------------------------ helpers
def nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def unaccent(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    return "".join(c for c in d if unicodedata.category(c) != "Mn")


def match_norm(s: str) -> str:
    """The normalisation the production matcher uses (accentless, final sigma folded)."""
    return unaccent(nfc_lower(s)).replace("ς", "σ")


def zipf_max(term: str, zipf) -> float:
    """Highest general-Greek frequency reading of the term.

    Queried both accented and unaccented, because wordfreq keys are accented:
    zipf('κάνει')=5.99 but zipf('κανει')=4.50, and the lower reading would let an
    ordinary word through. Max is the conservative direction.
    """
    a = nfc_lower(term)
    return max(zipf(a, "el"), zipf(unaccent(a), "el"))


def meeting_date_key(meeting_id: str) -> str:
    """`apr30_2_2026` and `apr30_2026` are the same date, different sessions."""
    return re.sub(r"_\d+(?=_\d{4}$)", "", meeting_id)


# ------------------------------------------------------------------- common set
def build_common(zipf, iter_wordlist) -> tuple[set[str], dict[str, float]]:
    """Common Greek surface forms, in matcher normalisation, down to COMMON_ZIPF_FLOOR."""
    forms: dict[str, float] = {}
    for w in iter_wordlist("el"):
        z = zipf(w, "el")
        if z < COMMON_ZIPF_FLOOR:
            break
        k = match_norm(w)
        if len(k) >= 3:
            forms[k] = max(forms.get(k, 0.0), z)
    # Codex asked for the assertion rather than a magic top-N cap.
    assert all(z >= COMMON_ZIPF_FLOOR for z in forms.values())
    return set(forms), forms


def hunspell_headwords() -> set[str]:
    """Lowercase headwords of el_GR, i.e. ordinary Greek words.

    Deviation from the pre-registration, disclosed in the report: the plan put these
    into C3's fuzzy neighbour set. Decoded correctly (iso-8859-7, not utf-8, which
    silently yields 25 entries instead of 828k) this is a full morphology list, not a
    frequency list, so as a fuzzy corpus it would reject nearly everything and would
    cost hours. It is used the way scripts/build_ds_wer_terms_v2.py rule_b uses it:
    an exact membership test for "this is an ordinary Greek word".
    """
    out: set[str] = set()
    if not HUNSPELL.exists():
        return out
    for line in HUNSPELL.read_bytes().decode("iso-8859-7").splitlines()[1:]:
        head = line.split("/")[0].strip()
        if head and head == nfc_lower(head) and len(head) >= 3:
            out.add(match_norm(head))
    return out


def build_bg_ngrams(max_n: int = 4) -> set[str]:
    """Word n-grams of general (non-council) Greek, for the phrase confusability test."""
    out: set[str] = set()
    for p in BG_SENTENCES:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    ref = json.loads(line).get("ref") or ""
                except Exception:
                    continue
                toks = match_norm(ref).split()
                for n in range(2, max_n + 1):
                    for i in range(len(toks) - n + 1):
                        out.add(" ".join(toks[i:i + n]))
    return out


# ------------------------------------------------------------------------ C3
def confusability(term: str, common: set[str], common_z: dict[str, float],
                  bg_ngrams: set[str], fuzz) -> dict:
    """Run the production scorer against common background forms.

    Returns the worst (highest-scoring) collision and the resulting verdict.

    The identity match is excluded. This is a bug fix made after the first run; every
    such fix is listed in the report. wordfreq's Greek list contains
    lowercase spellings of common acronyms (κεδε, κτελ, ασεπ, αμεα, ενφια, κεπ, επαλ),
    so the term scored 100 against *itself* and every one of them self-rejected. The
    pre-registered criterion, and Codex's wording of it, is a collision with a common
    **non-target** form; how frequent the term itself is in general Greek is C2's job,
    not C3's.
    """
    t = match_norm(term)
    if " " in term:
        best, best_w = 0.0, None
        for g in bg_ngrams:
            if g == t:
                continue
            s = fuzz.partial_ratio(t, g)
            if s > best:
                best, best_w = s, g
        verdict = ("reject" if best >= PHRASE_REJECT
                   else "review" if best >= PHRASE_REVIEW else "clear")
        return {"scorer": "partial_ratio", "score": round(best, 1),
                "nearest": best_w, "nearest_zipf": None, "verdict": verdict}
    best, best_w = 0.0, None
    for w in common:
        if w == t or abs(len(w) - len(t)) > 3:
            continue
        s = fuzz.ratio(t, w)
        if s > best:
            best, best_w = s, w
    verdict = ("reject" if best >= WORD_REJECT
               else "review" if best >= WORD_REVIEW else "clear")
    return {"scorer": "ratio", "score": round(best, 1), "nearest": best_w,
            "nearest_zipf": common_z.get(best_w) if best_w else None,
            "verdict": verdict}


# --------------------------------------------------------------------- C0 already
def frozen_aliases() -> dict[str, set[str]]:
    """Everything already covered by the hash-frozen per-city lists."""
    out: dict[str, set[str]] = defaultdict(set)
    for p in sorted(TERMS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        city = d["city"]
        for t in d["terms"]:
            for a in t.get("aliases", []):
                out[city].add(match_norm(a))
            out[city].add(match_norm(t["canonical"]))
    return out


# -------------------------------------------------------------------- stats
def term_stats():
    """Per-term recurrence over the reconstructed TRAIN fold (the glossary's own fold)."""
    import pandas as pd

    from eval.glossary import _extract_terms
    from eval.splits import split_by_meeting

    split = json.loads((ROOT / "data/eval/split.json").read_text())
    chains = pd.read_parquet(ROOT / "data/eval/chains.parquet").to_dict("records")
    train, _ = split_by_meeting(chains, eval_meetings=set(split["eval_meeting_ids"]))
    city_meet: dict[tuple, set] = defaultdict(set)
    for c in train:
        city, meet = c["city_id"], c["meeting_id"]
        for t in _extract_terms(c.get("gold_final", "") or ""):
            city_meet[(city, t)].add(meet)
    return city_meet


def gazetteer() -> set[str]:
    """Place names with an external authority behind them.

    Only argos and orestiada have a built gazetteer (ELSTAT 2021 + OSM, via
    build_ds_wer_terms_v2). The other nine cities cannot earn toponym evidence in this
    cycle; that is a scope gap, recorded rather than papered over.
    """
    out: set[str] = set()
    for p in TERMS_DIR.glob("*.v2.json"):
        d = json.loads(p.read_text())
        for t in d["terms"]:
            if t["klass"].startswith("place_"):
                out.add(match_norm(t["canonical"]))
                out.update(match_norm(a) for a in t.get("aliases", []))
    return out


def category_evidence(term: str, modelreg_norm: str, gaz: set[str],
                      hunspell: set[str]) -> tuple[str, str]:
    """C4: capitalisation inside a transcript is not evidence of anything.

    «Κάνει» and «Πράγμα» got into the 2026-06 global list purely because they start
    sentences. A candidate therefore has to point at a source outside the transcripts.
    Returns (category, evidence_kind); evidence_kind "none" means unverified.
    """
    n = match_norm(term)
    if _ACRONYM_RE.match(term):
        # An agenda item typed in caps is not an acronym. ΑΝΑΨΥΚΤΗΡΙΟ, ΚΑΦΕΤΕΡΙΑ and
        # ΠΑΡΑΣΚΕΥΑΣΤΗΡΙΟ all reached the accepted tier on the first pass this way.
        if n in hunspell:
            return "shouted_common_word", "none"
        return "acronym", "acronym_shape"        # meaning still owed by stage 2
    if modelreg_norm and " " in term and f" {n} " in modelreg_norm:
        # Phrases only. A single common token occurring somewhere in a regulation is
        # not institutional evidence: «Ανω» and «Άκη» passed that way on the first run.
        return "legal_procedural", "modelreg"
    if n in gaz:
        return "toponym", "gazetteer"
    return ("phrase" if " " in term else "proper_noun"), "none"


# ----------------------------------------------------------------------- main
def main() -> None:
    from rapidfuzz import fuzz
    from wordfreq import iter_wordlist, zipf_frequency

    common, common_z = build_common(zipf_frequency, iter_wordlist)
    bg = build_bg_ngrams()
    frozen = frozen_aliases()
    gaz = gazetteer()
    hun = hunspell_headwords()
    modelreg_norm = (" " + match_norm(MODELREG.read_text(errors="replace")) + " "
                     if MODELREG.exists() else "")
    print(f"common surface forms: {len(common):,}  background n-grams: {len(bg):,}  "
          f"hunspell headwords: {len(hun):,}")

    gloss = json.loads(GLOSSARY.read_text())
    stats = term_stats()

    # ------------------------------------------------------------------ pool A
    rows_a = []
    reasons = defaultdict(int)
    for city, terms in sorted(gloss["per_city"].items()):
        for term in terms:
            rec = {"city": city, "term": term}
            meets = stats.get((city, term), set())
            dates = {meeting_date_key(m) for m in meets}
            rec["n_meetings"] = len(meets)
            rec["n_dates"] = len(dates)
            if len(meets) < CITY_MIN_MEETINGS or len(dates) < CITY_MIN_DATES:
                reasons["C1 recurrence"] += 1
                continue
            if match_norm(term) in frozen.get(city, set()):
                reasons["C0 already frozen"] += 1
                continue
            kind, evid = category_evidence(term, modelreg_norm, gaz, hun)
            rec["category"] = kind
            rec["category_evidence"] = evid
            z = zipf_max(term, zipf_frequency)
            rec["zipf"] = round(z, 2)
            if " " in term:
                content = [w for w in nfc_lower(term).split()
                           if w not in _FUNCTION_WORDS]
                zs = [zipf_max(w, zipf_frequency) for w in content] or [9.0]
                if min(zs) > ZIPF_MAX_PHRASE_CONTENT:
                    reasons["C2 rarity"] += 1
                    continue
                rec["zipf_content_min"] = round(min(zs), 2)
            else:
                cap = ZIPF_MAX_ROSTER if kind == "proper_noun" else ZIPF_MAX_ENTITY
                rec["zipf_cap"] = cap
                if z > cap:
                    reasons["C2 rarity"] += 1
                    continue
                if z == 0.0 and kind != "acronym":
                    rec["needs_validation"] = "zipf 0.0 means out-of-list, not rare"
            c3 = confusability(term, common, common_z, bg, fuzz)
            rec["confusability"] = c3
            if c3["verdict"] == "reject":
                reasons["C3 confusable"] += 1
                continue
            n = len(match_norm(term).replace(" ", ""))
            rec["tier"] = "accepted" if evid != "none" else "unverified"
            rec["metric_eligible"] = n >= MIN_CHARS_METRIC and evid != "none"
            rec["injection_eligible"] = (n >= MIN_CHARS_INJECTION
                                         and c3["verdict"] == "clear"
                                         and evid != "none"
                                         and "needs_validation" not in rec)
            rows_a.append(rec)

    # ------------------------------------------------------------------ pool B
    raw = json.loads(POOLB_RAW.read_text())
    rows_b = []
    audit_b = []          # why a frequently-seen acronym did not make it
    reasons_b = defaultdict(int)

    def drop(term, why, **kw):
        if kw.get("n_meetings", 0) >= 5:
            audit_b.append({"term": term, "rejected_by": why, **kw})

    for r in raw["acronyms"]:
        term = r["acronym"]
        rec = {"term": term, "category": "acronym",
               "n_meetings": r["n_meetings"], "n_cities": r["n_cities"],
               "n_occurrences": r["n_occurrences"], "cities": r["cities"],
               "surface_variants": r["surface_variants"]}
        if r["n_cities"] >= UNIVERSAL_MIN_CITIES and \
                r["n_meetings"] >= UNIVERSAL_MIN_CITY_MEETINGS:
            rec["scope"] = "universal"
        elif r["n_cities"] >= CROSS_CITY_MIN_CITIES:
            rec["scope"] = "cross_city"
        else:
            reasons_b["C1 recurrence (single/two-city)"] += 1
            drop(term, "C1", n_meetings=r["n_meetings"], n_cities=r["n_cities"])
            continue
        z = zipf_max(term, zipf_frequency)
        rec["zipf"] = round(z, 2)
        if z > ZIPF_MAX_ENTITY:
            reasons_b["C2 rarity"] += 1
            drop(term, "C2", n_meetings=r["n_meetings"], n_cities=r["n_cities"],
                 zipf=round(z, 2))
            continue
        c3 = confusability(term, common, common_z, bg, fuzz)
        rec["confusability"] = c3
        if c3["verdict"] == "reject":
            reasons_b["C3 confusable"] += 1
            drop(term, "C3", n_meetings=r["n_meetings"], n_cities=r["n_cities"],
                 zipf=round(z, 2), nearest=c3["nearest"], score=c3["score"])
            continue
        n = len(term)
        rec["metric_eligible"] = n >= MIN_CHARS_METRIC
        rec["injection_eligible"] = n >= MIN_CHARS_INJECTION and c3["verdict"] == "clear"
        rows_b.append(rec)

    rows_b.sort(key=lambda r: (-r["n_cities"], -r["n_meetings"], r["term"]))

    # ----------------------------------------------------------------- write
    crit = {
        "preregistered_on": "https://github.com/eellak/gsoc2026-opencouncil-stt/"
                            "issues/19#issuecomment-5306477417",
        "codex_review_job": "45db933dfd254c1eb1d323b225ee0deb",
        "zipf_max_entity": ZIPF_MAX_ENTITY, "zipf_max_roster": ZIPF_MAX_ROSTER,
        "zipf_max_phrase_content": ZIPF_MAX_PHRASE_CONTENT,
        "common_zipf_floor": COMMON_ZIPF_FLOOR,
        "word_reject": WORD_REJECT, "word_review": WORD_REVIEW,
        "phrase_reject": PHRASE_REJECT, "phrase_review": PHRASE_REVIEW,
        "city_min_meetings": CITY_MIN_MEETINGS, "city_min_dates": CITY_MIN_DATES,
        "universal_min_cities": UNIVERSAL_MIN_CITIES,
        "universal_min_city_meetings": UNIVERSAL_MIN_CITY_MEETINGS,
        "min_chars_metric": MIN_CHARS_METRIC,
        "min_chars_injection": MIN_CHARS_INJECTION,
        "background": {"frequency": "wordfreq el", "lexicon": HUNSPELL.name,
                       "ngrams_from": [p.parent.name for p in BG_SENTENCES],
                       "n_common_forms": len(common), "n_bg_ngrams": len(bg),
                       "n_hunspell_headwords": len(hun)},
    }
    a = {"criteria": crit,
         "source": {"artifact": "data/glossary/glossary.json",
                    "manifest": "research/glossary/glossary-2026-06-20.manifest.json"},
         "n_input": sum(len(v) for v in gloss["per_city"].values()),
         "n_survivors": len(rows_a),
         "rejections": dict(reasons), "candidates": rows_a}
    b = {"criteria": crit, "source": raw["source"],
         "n_input": raw["n_distinct_acronyms"], "n_survivors": len(rows_b),
         "rejections": dict(reasons_b), "candidates": rows_b,
         "rejection_audit_frequent": sorted(
             audit_b, key=lambda r: (-r["n_meetings"], r["term"]))}
    OUT_A.write_text(json.dumps(a, ensure_ascii=False, indent=1) + "\n")
    OUT_B.write_text(json.dumps(b, ensure_ascii=False, indent=1) + "\n")

    per_city_acc = defaultdict(int)
    per_city_unv = defaultdict(int)
    for r in rows_a:
        (per_city_acc if r["tier"] == "accepted" else per_city_unv)[r["city"]] += 1
    by_evidence = defaultdict(int)
    for r in rows_a:
        by_evidence[r["category_evidence"]] += 1
    summary = {
        "criteria": crit,
        "pool_a": {"n_input": a["n_input"],
                   "n_passed_c1_c3": len(rows_a),
                   "n_accepted": sum(1 for r in rows_a if r["tier"] == "accepted"),
                   "n_unverified_c4": sum(1 for r in rows_a
                                          if r["tier"] == "unverified"),
                   "accepted_per_city": dict(sorted(per_city_acc.items())),
                   "unverified_per_city": dict(sorted(per_city_unv.items())),
                   "by_category_evidence": dict(by_evidence),
                   "rejections": dict(reasons),
                   "n_injection_eligible": sum(1 for r in rows_a
                                               if r["injection_eligible"]),
                   "sha256": hashlib.sha256(OUT_A.read_bytes()).hexdigest()},
        "pool_b": {"n_input": b["n_input"], "n_survivors": len(rows_b),
                   "rejections": dict(reasons_b),
                   "n_universal": sum(1 for r in rows_b
                                      if r["scope"] == "universal"),
                   "n_cross_city": sum(1 for r in rows_b
                                       if r["scope"] == "cross_city"),
                   "n_injection_eligible": sum(1 for r in rows_b
                                               if r["injection_eligible"]),
                   "sha256": hashlib.sha256(OUT_B.read_bytes()).hexdigest()},
    }
    OUT_SUM.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "criteria"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
