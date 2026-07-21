"""PII scan over the HF dataset text — find utterances that mention NON-elected
people (private citizens / third parties) or leak structured personal data, so
they can be dropped (text + audio together) before any public release.

Legal context: the DPO blocked publication; even so, removing third-party PII is
a floor the mentor asked for. This is HARM-REDUCTION, not legal anonymisation —
the linked/clipped audio still carries the voice. This pass is a high-recall
CANDIDATE GENERATOR for HUMAN REVIEW (or an LLM adjudication stage), not a final
drop decision: NER precision on uncased Greek council text is poor (pronouns,
roles, place/saint names get mislabelled PERSON) and residual false negatives
remain (bare surname shared with a councillor, special-category PII).

Design:
  - Allowlist = per-(city, meeting) list of KNOWN-PERSON records, each a token
    set from one full name. Person-level (not flat-token) so a private
    "Γιώργος Παπαδόπουλος" is NOT cleared just because a different councillor
    "Μαρία Παπαδοπούλου" shares the surname (Codex review, critical #2/#3).
    A name is per-meeting: a councillor in city A is a private person in city B.
    Base = speakers.parquet person_name (every speaker — NB may include invited
    non-elected participants; documented limitation, Codex critical #1);
    extended by rosters.json full people where available.
  - A PERSON span is KNOWN if some one person record covers all its meaningful
    tokens (fuzzy, Greek-inflection tolerant). Else -> flag the utterance.
  - Titles/roles/pronouns/common nouns are stripped (fuzzy) so GLiNER's non-name
    PERSON spans don't flood the report.
  - Structured PII (AFM/AMKA/phone/IBAN/email/plate): strong distinctive patterns
    fire directly; weak digit-runs are context-gated (keyword within a symmetric
    window) to avoid flooding on council numbers (law refs, budgets).
  - We scan BOTH `text` and `before_text`: the audio says whichever, so a name in
    either field is a real exposure; the report keeps both + which field hit.

Stages (checkpointed, run in order):
  allowlist  build data/pii/allowlist.json
  scan       NER + regex per row -> data/pii/scan.jsonl (resumable)
  report     aggregate counts + data/pii/flagged.csv (+ optional gated dataset)

Usage:
  .venv-eval/bin/python -m eval.pii_scan allowlist
  .venv-eval/bin/python -m eval.pii_scan scan [--ner gliner|greekbert] [--limit N]
  .venv-eval/bin/python -m eval.pii_scan report [--write-gated]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rapidfuzz import fuzz, process               # noqa: E402

from eval.scoring import greek_normalize          # noqa: E402

SPEAKERS = ROOT / "data" / "eval" / "speakers.parquet"
ROSTERS = ROOT / "data" / "improve_loop" / "rosters.json"
ROSTERS_FULL = ROOT / "data" / "pii" / "rosters_full.json"   # all dataset meetings
PUB = ROOT / "data" / "hf-dataset" / "public"
OUT = ROOT / "data" / "pii"
ALLOWLIST = OUT / "allowlist.json"
SCAN_JSONL = OUT / "scan.jsonl"

GREEKBERT_MODEL = "amichailidis/bert-base-greek-uncased-v1-finetuned-ner"  # ORG-only
GLINER_MODEL = "urchade/gliner_multi-v2.1"
NER_THRESH = 0.5
MIN_TOK = 4          # min token length to treat as a matchable name token
FUZZ = 88            # rapidfuzz ratio: inflected match (Παπαδήμα~Παπαδήμας ~94)
                     # vs different surname (~<60). Greek inflects name endings.
# Bump when allowlist/classification logic changes so old scan lines are ignored.
SCAN_VERSION = "3"      # v3: full per-meeting rosters merged into the allowlist

# titles / roles / honorifics / pronouns / generic nouns that GLiNER sweeps into
# a PERSON span ("ο κύριος Παπαδόπουλος", "δήμαρχο", "εσείς", "ο εργολάβος").
# Matched FUZZILY so inflected forms (δήμαρχο/δημάρχου, προέδρο) also strip.
NAME_STOPWORDS = [greek_normalize(w) for w in (
    # honorifics
    "κυριος κυρια κυριε κ "
    # council roles
    "προεδρος αντιπροεδρος δημαρχος αντιδημαρχος συμβουλος συναδελφος γραμματεας "
    "εισηγητης γενικος εντεταλμενος προεδρειο διοικηση παραταξη αντιπολιτευση "
    # generic person-ish nouns
    "δημοτης δημοτισσα δημοτικος κατοικος πολιτης υπαλληλος διευθυντης "
    "προϊσταμενος υπουργος υφυπουργος περιφερειαρχης βουλευτης εργολαβος αναδοχος "
    "μελετητης μηχανικος δικηγορος γονεας παιδι προσωπικο υπηρεσια ανθρωπος "
    # pronouns / determiners GLiNER mislabels
    "εσεις εμεις εμενα εσενα αυτος αυτη αυτο καποιος καποια καθενας οποιος κανεις "
    "εκεινος αλλος ενας καθε τιποτα παρων"
).split()]

# very common Greek given names: a span that is ONLY such a first name cannot be
# tied to an individual -> reported (first_name_only), never auto-dropped.
COMMON_FIRST = [greek_normalize(w) for w in (
    "γιωργος γιαννης κωστας δημητρης νικος βασιλης μιχαλης χρηστος θανασης "
    "μαρια ελενη κατερινα σοφια αννα δεσποινα γεωργιος ιωαννης κωνσταντινος "
    "δημητριος νικολαος βασιλειος αθανασιος παναγιωτης αποστολος σπυρος στελιος "
    "ανδρεας αντωνης μπαμπης δημητρα γιαννα αλεκα εμμανουηλ σπυριδων"
).split()]


def _fuzzy_in(tok: str, choices) -> bool:
    """True if tok matches any choice within Greek-inflection tolerance."""
    if not choices:
        return False
    return process.extractOne(tok, choices, scorer=fuzz.ratio,
                              score_cutoff=FUZZ) is not None


def _is_stop(tok: str) -> bool:
    return _fuzzy_in(tok, NAME_STOPWORDS)


# ---------- structured PII ----------
# Strong distinctive patterns fire directly. Weak digit runs (AFM/AMKA/generic)
# are context-gated with a SYMMETRIC window (Codex review): ASR word order varies
# ("το ΑΦΜ του είναι 123..." AND "123... είναι το ΑΦΜ").
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_IBAN = re.compile(r"\bGR\d{2}(?:[ ]?\d){23}\b", re.I)      # GR + 25 digits
# NB: license-plate detection was removed — a "3 letters + 4 digits" pattern
# matches council speech everywhere ("του 2025", dates, protocol numbers) and
# produced ~99% false positives (Codex review). Plates are rare in ASR anyway.
_MOBILE = re.compile(r"\b(?:\+?30[ ]?)?69\d[ ]?\d{3}[ ]?\d{4}\b")   # GR mobile
_PII_KEYWORDS = re.compile(
    r"(αφμ|α\.?φ\.?μ|αμκα|α\.?μ\.?κ\.?α|ταυτοτητ|διαβατηρι|"
    r"αριθμο[ςσ]?\s+μητρωου|τηλεφων|κινητο|iban|λογαριασμο)", re.I)
_DIGIT_RUN = re.compile(r"\d[\d\.\-/ ]{6,}\d")


def structured_pii(text: str) -> list[str]:
    """Tags of structured identifiers found (strong direct, weak context-gated)."""
    hits = []
    if _EMAIL.search(text):
        hits.append("email")
    if _IBAN.search(text):
        hits.append("iban")
    if _MOBILE.search(text):
        hits.append("mobile")
    norm = greek_normalize(text)
    for km in _PII_KEYWORDS.finditer(norm):
        window = norm[max(0, km.start() - 40):km.end() + 40]   # symmetric
        if _DIGIT_RUN.search(window):
            hits.append(f"id:{km.group(1)[:6]}")
    return sorted(set(hits))


# ---------- allowlist (person-level) ----------

def name_tokens(name: str) -> list[str]:
    """Normalised name tokens (len>=MIN_TOK), titles/roles/pronouns stripped."""
    return sorted({t for t in greek_normalize(name).split()
                   if len(t) >= MIN_TOK and not _is_stop(t)})


def build_allowlist() -> dict[str, list[list[str]]]:
    """Per 'city/meeting' -> list of KNOWN-PERSON records (each a token list).

    Person-level so cross-person token bleed cannot clear a private name.
    Base: every speaker's person_name (always available; may include invited
    non-elected participants — documented limitation). Extended: rosters.json
    people terms with >=2 name tokens (bare single tokens are dropped to avoid
    reintroducing token bleed; party names are not persons)."""
    import pandas as pd
    sp = pd.read_parquet(SPEAKERS, columns=["city_id", "meeting_id", "person_name"])
    people: dict[str, set[tuple]] = collections.defaultdict(set)
    speaker_keys: set[str] = set()
    for c, m, n in zip(sp.city_id, sp.meeting_id, sp.person_name):
        key = f"{c}/{m}"
        speaker_keys.add(key)
        if isinstance(n, str) and n.strip():
            toks = tuple(name_tokens(n))
            if toks:
                people[key].add(toks)
    roster_keys: set[str] = set()
    for src in (ROSTERS, ROSTERS_FULL):            # improve_loop + full-coverage
        if not src.exists():
            continue
        ro = json.loads(src.read_text())
        for key, terms in ro.items():
            roster_keys.add(key)
            for t in terms:
                toks = tuple(name_tokens(t))
                if len(toks) >= 2:                 # full-name records only
                    people[key].add(toks)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {
        "_meta": {"scan_version": SCAN_VERSION,
                  "full_roster_keys": sorted(roster_keys)},
        "people": {k: sorted(list(t) for t in v) for k, v in people.items()},
    }
    ALLOWLIST.write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print(f"allowlist: {len(people)} meetings ({len(speaker_keys)} from speakers, "
          f"{len(roster_keys)} with full roster); -> {ALLOWLIST}")
    return out


# ---------- NER ----------

def make_ner(name: str):
    """Return fn(list[str]) -> list[list[str]] of PERSON span texts per input."""
    if name == "greekbert":                        # NB: this model is ORG-only
        from transformers import pipeline
        ner = pipeline("token-classification", model=GREEKBERT_MODEL,
                       aggregation_strategy="simple")

        def persons(texts):
            out = ner(list(texts), batch_size=16)
            if texts and isinstance(out, list) and out and isinstance(out[0], dict):
                out = [out]
            return [[e["word"] for e in ents
                     if e.get("entity_group", "").upper().startswith("PER")]
                    for ents in out]
        return persons
    if name == "gliner":
        from gliner import GLiNER
        model = GLiNER.from_pretrained(GLINER_MODEL)

        def persons(texts):
            out = model.batch_predict_entities(
                list(texts), ["person"], threshold=NER_THRESH)
            return [[e["text"] for e in ents] for ents in out]
        return persons
    raise ValueError(f"unknown NER {name}")


def classify_persons(spans: list[str], people: list[list[str]]) -> dict:
    """Person-level: a span is KNOWN only if ONE person record covers all its
    meaningful tokens. Spans whose tokens are all common given names are
    first_name_only (ambiguous, not flagged); everything else is unknown."""
    unknown, first_only = [], []
    for s in spans:
        toks = [t for t in name_tokens(s)]         # stopwords already stripped
        if not toks:
            continue
        if any(all(_fuzzy_in(t, person) for t in toks) for person in people):
            continue                               # one known person covers it
        if all(_fuzzy_in(t, COMMON_FIRST) for t in toks):
            first_only.append(s)                   # only common given name(s)
        else:
            unknown.append(s)
    return {"unknown": sorted(set(unknown)),
            "first_name_only": sorted(set(first_only))}


# ---------- scan ----------

def _load_rows():
    import pandas as pd
    parts = []
    for split in ("train", "validation"):
        df = pd.read_parquet(PUB / f"{split}.parquet")
        df["split"] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _row_key(r) -> str:
    return f"{r['split']}|{r['city_id']}|{r['meeting_id']}|{r['utterance_id']}"


def stage_scan(args) -> None:
    al = json.loads(ALLOWLIST.read_text())
    people_by_key = al["people"]
    full_roster = set(al.get("_meta", {}).get("full_roster_keys", []))
    df = _load_rows()

    done = set()
    if SCAN_JSONL.exists():
        for l in SCAN_JSONL.open():
            l = l.strip()
            if l:
                try:
                    d = json.loads(l)
                    if d.get("scan_version") == SCAN_VERSION:
                        done.add(d["row_key"])
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f"resume: {len(done)} already scanned (version {SCAN_VERSION})")

    ner = make_ner(args.ner)
    df["_rk"] = df.apply(_row_key, axis=1)
    todo = df[~df["_rk"].isin(done)]
    if args.limit:
        todo = todo.iloc[:args.limit]
    print(f"scanning {len(todo)} rows with NER={args.ner}")

    OUT.mkdir(parents=True, exist_ok=True)
    recs = todo.to_dict("records")
    B = 64
    n_flag = 0
    with SCAN_JSONL.open("a") as out:
        for i in range(0, len(recs), B):
            batch = recs[i:i + B]
            texts = [str(r.get("text") or "") for r in batch]
            befores = [str(r.get("before_text") or "") for r in batch]
            per_t = ner(texts)
            per_b = ner(befores)
            for r, pt, pb, tx, bx in zip(batch, per_t, per_b, texts, befores):
                key = f"{r['city_id']}/{r['meeting_id']}"
                people = people_by_key.get(key, [])
                cls_t = classify_persons(list(pt), people)
                cls_b = classify_persons(list(pb), people)
                unknown = sorted(set(cls_t["unknown"]) | set(cls_b["unknown"]))
                first = sorted(set(cls_t["first_name_only"]) |
                               set(cls_b["first_name_only"]))
                struct_t, struct_b = structured_pii(tx), structured_pii(bx)
                struct = sorted(set(struct_t) | set(struct_b))
                flag = bool(unknown or struct)
                if flag:
                    n_flag += 1
                out.write(json.dumps({
                    "scan_version": SCAN_VERSION, "row_key": _row_key(r),
                    "utterance_id": r["utterance_id"], "city_id": r["city_id"],
                    "meeting_id": r["meeting_id"], "source": r.get("source"),
                    "split": r["split"], "has_full_roster": key in full_roster,
                    "has_allowlist": bool(people),
                    "unknown": unknown, "first_name_only": first,
                    "struct_pii": struct,
                    "hit_field": ("text" if (cls_t["unknown"] or struct_t) else "")
                                 + ("before" if (cls_b["unknown"] or struct_b)
                                    else ""),
                    "flag": flag, "text": tx, "before_text": bx,
                }, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{min(i + B, len(recs))}/{len(recs)}] flagged so far: "
                  f"{n_flag}", flush=True)
    print(f"-> {SCAN_JSONL} (+{n_flag} flagged this run)")


# ---------- report ----------

def stage_report(args) -> None:
    rows = [json.loads(l) for l in SCAN_JSONL.open() if l.strip()
            ] if SCAN_JSONL.exists() else []
    rows = [r for r in rows if r.get("scan_version") == SCAN_VERSION]
    n = len(rows)
    if n == 0:
        print("no scan rows for current version — run `scan` first")
        return
    flagged = [r for r in rows if r["flag"]]
    by_reason = collections.Counter()
    for r in flagged:
        if r["unknown"]:
            by_reason["unknown_person"] += 1
        if r["struct_pii"]:
            by_reason["structured_pii"] += 1
    no_roster = sum(1 for r in rows if not r.get("has_full_roster"))
    no_allow = sum(1 for r in rows if not r.get("has_allowlist"))
    by_city = collections.Counter(r["city_id"] for r in flagged)
    by_source = collections.Counter(r.get("source") for r in flagged)
    by_split = collections.Counter(r["split"] for r in flagged)
    first_only_rows = sum(1 for r in rows if r["first_name_only"] and not r["flag"])

    md = [
        "# PII scan report", "",
        f"Scanned **{n}** utterances. Flagged **{len(flagged)}** "
        f"({100 * len(flagged) / n:.1f}%) as candidates to drop.", "",
        "Flag = an UNKNOWN person (no single roster/speaker record covers the "
        "name) or structured PII. This is a **high-recall candidate generator**, "
        "not a final decision: NER precision on uncased Greek council text is low "
        "(place/saint names, first-name-only mentions, national public figures "
        "still slip in) — **review (or LLM-adjudicate) before dropping**. It does "
        "not anonymise the audio.", "",
        "## Flag reasons (rows may hit both)", "",
        f"- unknown person mentioned: **{by_reason['unknown_person']}**",
        f"- structured PII (id/phone/iban/email/plate): "
        f"**{by_reason['structured_pii']}**", "",
        f"Also: {first_only_rows} rows mention only a common first name "
        f"(ambiguous, NOT flagged). Coverage: {no_roster} rows without a FULL "
        f"roster (speaker-only allowlist, higher false-positive rate); "
        f"{no_allow} rows with no allowlist at all.", "",
        "## Flagged by split", "",
        *[f"- {k}: {v}" for k, v in by_split.most_common()], "",
        "## Flagged by source", "",
        *[f"- {k}: {v}" for k, v in by_source.most_common()], "",
        "## Flagged by city", "",
        *[f"- {k}: {v}" for k, v in by_city.most_common()], "",
    ]
    (OUT / "report.md").write_text("\n".join(md) + "\n")

    import csv
    with (OUT / "flagged.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utterance_id", "city_id", "meeting_id", "source", "split",
                    "has_full_roster", "reason", "hit_field", "unknown",
                    "struct_pii", "text", "before_text"])
        for r in flagged:
            reason = "+".join(x for x, on in
                              (("unknown_person", r["unknown"]),
                               ("structured_pii", r["struct_pii"])) if on)
            w.writerow([r["utterance_id"], r["city_id"], r["meeting_id"],
                        r.get("source"), r["split"], r.get("has_full_roster"),
                        reason, r.get("hit_field", ""), "; ".join(r["unknown"]),
                        "; ".join(r["struct_pii"]), r["text"], r["before_text"]])
    print(f"-> {OUT/'report.md'} , {OUT/'flagged.csv'} "
          f"({len(flagged)}/{n} flagged, {100*len(flagged)/n:.1f}%)")

    if args.write_gated:
        import pandas as pd
        drop = {r["utterance_id"] for r in flagged}
        gdir = ROOT / "data" / "hf-dataset" / "public-pii-gated"
        gdir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation"):
            part = pd.read_parquet(PUB / f"{split}.parquet")
            kept = part[~part.utterance_id.isin(drop)].reset_index(drop=True)
            kept.to_parquet(gdir / f"{split}.parquet", index=False)
            print(f"  gated {split}: {len(part)} -> {len(kept)} "
                  f"(-{len(part) - len(kept)})")
        print(f"-> {gdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("allowlist")
    ps = sub.add_parser("scan")
    ps.add_argument("--ner", choices=["greekbert", "gliner"], default="gliner")
    ps.add_argument("--limit", type=int, default=0)
    pr = sub.add_parser("report")
    pr.add_argument("--write-gated", action="store_true")
    args = ap.parse_args()
    if args.stage == "allowlist":
        build_allowlist()
    elif args.stage == "scan":
        stage_scan(args)
    elif args.stage == "report":
        stage_report(args)


if __name__ == "__main__":
    main()
