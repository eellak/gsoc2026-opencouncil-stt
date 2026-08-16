"""Stage 1 of issue #20: mine name/toponym candidates out of the human corrections.

Issue #19 mined the 2026-06 glossary — i.e. what *exists* in the transcripts — and all
47 survivors were acronyms, because "acronym shape" was the only category evidence a
transcript can give. This script uses a different source: `data/asr/export.jsonl`,
the human correction database, where the wrong form the model produced sits next to
the right answer a human typed.

It does NOT re-classify the corrections. Two existing classifications are reused:

  * the per-utterance LLM labels already stored in `error_categories`
    (person_name 226, place_name 151, acronym_abbreviation 124, org_party_name 63,
    legal_admin_term 43) — these select which rows are worth looking at;
  * `eval/controlled_eval/exp_edit_taxonomy.py`'s normalisation and op alignment, so
    the op counts stay comparable with results_edit_taxonomy.json.

Opcode direction: SequenceMatcher transforms `initial_before_text` (what the ASR
produced) into `final_after_text` (what the human approved). So `insert` means the
ASR **deleted** a word, and `replace` means it produced a wrong form.

Three lanes, per Codex review 316755b6:
  near      replace ops where the wrong form is a near-corruption of the right one
            (the 1-2 character errors exp-2026-08-11-error-analysis measured)
  hard      replace ops outside that gate — kept for the audit, never auto-applied
  deletion  insert ops, i.e. words the ASR dropped — a closed substitution list
            cannot fix these, so they are watch-listed and ranked separately

Normalisation policy: the candidate key is NFC, accent-stripped, lowercased `\\w+`
tokens joined by single spaces. Final sigma is **kept** here (so the key stays
comparable with exp_edit_taxonomy); the filter stage folds it, matching the
production matcher.

Output (local, gitignored — it contains mined term strings and correction context):
  data/glossary/error_pairs.raw.json

Run:  .venv-eval/bin/python scripts/glossary_error_mine.py
"""
from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "data/asr/export.jsonl"
OUT = ROOT / "data/glossary/error_pairs.raw.json"

# ------------------------------------------------------------------ pre-registered
ENTITY_LABELS = ("person_name", "place_name", "org_party_name",
                 "acronym_abbreviation", "legal_admin_term")
MAX_TOKENS_PER_SIDE = 2
MAX_CHAR_DIST_ABS = 4
CHAR_DIST_FRAC = 0.34
NEAR_D2_MIN_LEN = 5      # two edits inside a 4-char acronym is not a near-corruption
CTX_TOKENS = 3

# Address forms swallowed into a two-token op ("κύριε Κανλουπούλου" -> "κυρία
# Κανελλοπούλου") would key the candidate as «κυρία Κανελλοπούλου» and inject that
# whole string. They are trimmed off both sides of the op before keying.
# Roll-call and voting markers glue themselves onto the name in a two-token op
# ("Κουράσης, ναι", "Λιόλιος παρών", "Μπραϊκούδη, αιτιολογήστε"), and "Δήμου
# Σαμοθράκης" is the municipality, not a term. Codex flagged all of these in the
# final-set review; they are trimmed for the same reason as the address forms.
_TRIM = {"κυριε", "κυρια", "κυριο", "κυριοσ", "κυριου", "κ", "κα", "κος",
         "δημαρχε", "προεδρε", "προεδροσ", "συναδελφε", "ο", "η", "το", "του",
         "τησ", "των", "τον", "την", "στο", "στη", "στην", "στον", "στα", "στισ",
         "και", "με", "για", "σε", "απο", "στουσ",
         "ναι", "οχι", "υπερ", "κατα", "παρων", "παρουσα", "απων", "απουσα",
         "λευκο", "αποχη", "αιτιολογηστε", "ψηφιζω",
         "δημοσ", "δημου", "δημο", "δημε", "δημαρχοσ", "αντιδημαρχοσ",
         "τμημα", "τμηματοσ", "υπηρεσια", "υπηρεσιασ"}

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENT_END = re.compile(r"[.!?;·:»\"\)\]]\s*$|^\s*$|[-–—]\s*$")


def strip_accents(s: str) -> str:
    d = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in d if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Normalised tokens plus their spans in the NFC source text.

    Spans, not whitespace joins: the surface form must be recovered from the original
    string so that punctuation-bearing display forms («ΛΟΑΤΚΙ+») survive.
    """
    t = unicodedata.normalize("NFC", text or "")
    spans = [m.span() for m in _WORD_RE.finditer(t)]
    toks = [strip_accents(t[a:b]).lower() for a, b in spans]
    return toks, spans


def cedist(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


def near_lane(dist: int, after_nospace: str) -> bool:
    """Codex's closeness gate: d==1, or d==2 on a term of >=5 chars, or <=34%."""
    if dist < 1 or dist > MAX_CHAR_DIST_ABS:
        return False
    if dist == 1:
        return True
    if dist == 2 and len(after_nospace) >= NEAR_D2_MIN_LEN:
        return True
    return dist / max(1, len(after_nospace)) <= CHAR_DIST_FRAC


def trim_span(toks: list[str], a: int, b: int) -> tuple[int, int]:
    """Drop address forms and function words from the edges of an op span."""
    while b - a > 1 and toks[a].replace("ς", "σ") in _TRIM:
        a += 1
    while b - a > 1 and toks[b - 1].replace("ς", "σ") in _TRIM:
        b -= 1
    return a, b


def labels_of(row: dict) -> list[str]:
    v = row.get("error_categories")
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            return []
    return v if isinstance(v, list) else []


def display_form(text: str, spans: list[tuple[int, int]], j1: int, j2: int) -> str:
    """Raw slice covering the op span, plus a trailing '+' if the source has one."""
    t = unicodedata.normalize("NFC", text or "")
    a, b = spans[j1][0], spans[j2 - 1][1]
    if b < len(t) and t[b] == "+":
        b += 1
    return t[a:b]


def _new(key: str) -> dict:
    return {"key": key, "surfaces": Counter(), "display": Counter(),
            "wrong_forms": Counter(), "n_corrections": 0, "utterances": set(),
            "meetings": set(), "dates": set(), "cities": Counter(),
            "city_meetings": defaultdict(set), "city_utterances": defaultdict(set),
            "labels": Counter(), "cap_on_span": 0, "span_seen": 0,
            "sentence_initial": 0, "char_dists": Counter(),
            "left_ctx": Counter(), "right_ctx": Counter()}


def main() -> None:
    lanes: dict[str, dict[str, dict]] = {"near": {}, "hard": {}, "deletion": {}}
    stats = Counter()

    for line in open(EXPORT, encoding="utf-8"):
        d = json.loads(line)
        if d.get("include_status") != "include":
            continue
        stats["rows_included"] += 1
        labs = [x for x in labels_of(d) if x in ENTITY_LABELS]
        if not labs:
            continue
        stats["rows_entity_labelled"] += 1
        bef_txt, aft_txt = d.get("initial_before_text", ""), d.get("final_after_text", "")
        if not aft_txt:
            continue
        bt, bspans = tokenize(bef_txt)
        at, aspans = tokenize(aft_txt)
        aft_nfc = unicodedata.normalize("NFC", aft_txt)
        city, meet = d.get("city_id"), d.get("meeting_id")
        date = (d.get("meeting_date") or "")[:10]
        mkey = f"{city}/{meet}"
        uid = d.get("utterance_id")

        for tag, i1, i2, j1, j2 in SequenceMatcher(None, bt, at,
                                                   autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            j1, j2 = trim_span(at, j1, j2)
            i1, i2 = trim_span(bt, i1, i2)
            b_side, a_side = bt[i1:i2], at[j1:j2]
            if b_side == a_side:
                stats["drop_trimmed_to_equal"] += 1
                continue
            if not a_side:
                stats["ops_delete_from_human"] += 1
                continue
            aj = " ".join(a_side)
            a_nospace = "".join(a_side)
            if not b_side:                       # ASR dropped these words
                stats["ops_asr_deletion"] += 1
                if len(a_side) > MAX_TOKENS_PER_SIDE:
                    continue
                lane, dist, bj = "deletion", None, ""
            else:
                stats["ops_replace"] += 1
                if len(a_side) > MAX_TOKENS_PER_SIDE or len(b_side) > MAX_TOKENS_PER_SIDE:
                    stats["drop_too_many_tokens"] += 1
                    continue
                bj = " ".join(b_side)
                dist = cedist("".join(b_side), a_nospace)
                if dist == 0:
                    stats["drop_formatting_only"] += 1
                    continue
                lane = "near" if near_lane(dist, a_nospace) else "hard"
            stats[f"ops_kept_{lane}"] += 1

            rec = lanes[lane].setdefault(aj, _new(aj))
            rec["n_corrections"] += 1
            rec["surfaces"][aft_nfc[aspans[j1][0]:aspans[j2 - 1][1]]] += 1
            rec["display"][display_form(aft_txt, aspans, j1, j2)] += 1
            if bj:
                rec["wrong_forms"][
                    unicodedata.normalize("NFC", bef_txt)[bspans[i1][0]:bspans[i2 - 1][1]]
                ] += 1
            rec["utterances"].add(uid)
            rec["meetings"].add(mkey)
            rec["dates"].add(date)
            rec["cities"][city] += 1
            rec["city_meetings"][city].add(mkey)
            rec["city_utterances"][city].add(uid)
            if dist is not None:
                rec["char_dists"][dist] += 1
            for x in labs:
                rec["labels"][x] += 1
            rec["left_ctx"][" ".join(at[max(0, j1 - CTX_TOKENS):j1])] += 1
            rec["right_ctx"][" ".join(at[j2:j2 + CTX_TOKENS])] += 1
            # Internal witness: capitalisation ON the corrected span, in the
            # human-approved text, away from a sentence boundary. Codex's point is
            # that the LLM label may itself be reading the capitalisation, so this is
            # one correlated witness, not two — it can only reach the review tier.
            rec["span_seen"] += 1
            head = aft_nfc[aspans[j1][0]:aspans[j1][1]]
            sent_initial = j1 == 0 or bool(
                _SENT_END.search(aft_nfc[:aspans[j1][0]][-4:]))
            if sent_initial:
                rec["sentence_initial"] += 1
            elif head[:1].isupper():
                rec["cap_on_span"] += 1

    def dump(recs: dict[str, dict]) -> list[dict]:
        rows = []
        for rec in recs.values():
            rows.append({
                "key": rec["key"],
                "canonical": rec["surfaces"].most_common(1)[0][0],
                "display_aliases": [k for k, _ in rec["display"].most_common(4)],
                "wrong_forms": dict(rec["wrong_forms"].most_common(6)),
                "n_corrections": rec["n_corrections"],
                "n_utterances": len(rec["utterances"]),
                "n_meetings": len(rec["meetings"]),
                "n_dates": len(rec["dates"]),
                "n_cities": len(rec["cities"]),
                "cities": dict(rec["cities"].most_common()),
                "per_city": {c: {"n_meetings": len(rec["city_meetings"][c]),
                                 "n_utterances": len(rec["city_utterances"][c]),
                                 "n_corrections": rec["cities"][c]}
                             for c in rec["cities"]},
                "source_meetings": sorted(rec["meetings"]),
                "labels": dict(rec["labels"].most_common()),
                "cap_on_span": rec["cap_on_span"],
                "span_seen": rec["span_seen"],
                "sentence_initial": rec["sentence_initial"],
                "char_dists": {str(k): v for k, v in sorted(rec["char_dists"].items())},
                "left_ctx": [k for k, _ in rec["left_ctx"].most_common(4) if k],
                "right_ctx": [k for k, _ in rec["right_ctx"].most_common(4) if k],
            })
        rows.sort(key=lambda r: (-r["n_corrections"], -r["n_meetings"], r["key"]))
        return rows

    out = {
        "source": {"file": "data/asr/export.jsonl",
                   "entity_labels": list(ENTITY_LABELS),
                   "max_tokens_per_side": MAX_TOKENS_PER_SIDE,
                   "max_char_dist_abs": MAX_CHAR_DIST_ABS,
                   "char_dist_frac": CHAR_DIST_FRAC,
                   "near_d2_min_len": NEAR_D2_MIN_LEN,
                   "opcode_direction": "before_text -> after_text; insert == ASR deletion"},
        "stats": dict(sorted(stats.items())),
        "lanes": {lane: dump(recs) for lane, recs in lanes.items()},
    }
    out["n_candidates"] = {k: len(v) for k, v in out["lanes"].items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"stats": out["stats"], "n_candidates": out["n_candidates"]},
                     indent=1))


if __name__ == "__main__":
    main()
