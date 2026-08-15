#!/usr/bin/env python3
"""Deterministic per-meeting roster -> faster-whisper `hotwords` string (arm B).

Spec: docs/specs/2026-08-12-serving-stack-plan.md, arm B, with BOTH frozen
amendments ("Τροποποίηση 2026-08-12" and "Τροποποίηση 2η", adopted before any
decode). Frozen rules now:

- Candidates are SURNAMES ONLY, one per person. Full names are dropped from the
  primary arm entirely (2nd amendment). Roster entries with an initial-with-dot
  ("Β. Μαυρίδης"), commas, or party names never become entries themselves - a
  dotted entry still counts toward its person's canonical surname.
- Person/surname source: the DS-WER term files
  `research/ds_wer/terms/{city}.json` (klass person_surname; `canonical` =
  surname, `covers` = full names). A person is in the meeting when their
  canonical surname appears as a word of a roster entry or one of their covered
  full names appears verbatim in the roster (both sides under the audit
  normalization, as in name_lexicon_audit.meeting_roster_terms).
- FALLBACK, for cities WITHOUT a term file only (documented imprecision): parse
  rosters_full.json directly. Comma entries and entries of >2 words are skipped
  as party-like; "X. Surname" and "First Last" contribute the surname; a
  single-word entry counts as a surname unless it appears as the first word of
  some two-word entry (evidence it is a first name). Two-word party names
  ("Δημοτική Επαναφορά") cannot be told apart from persons here - the term-file
  path is authoritative for the eval cities.
- ORDER (2nd amendment): NOT alphabetical. Surnames rank by
  `SHA-256(FROZEN_SALT || meeting || surname_key)` ascending, where
  FROZEN_SALT = "oc-hotwords-2026-08-12", `meeting` is the qualified roster key
  "city/meeting_id" (the repo's meeting identity; avoids cross-city collisions),
  and `surname_key` is the frozen NFC+casefold normalization of the canonical
  surname (so byte-level Unicode form cannot reorder). Alphabetical order and
  two-pass relevance selection are explicitly rejected by the spec. The hash
  makes each meeting's excluded tail a different pseudo-random subset instead of
  a correlated alphabetical blind spot.
- Dedup AFTER frozen Unicode normalization: NFC + casefold. First occurrence in
  the deterministic order keeps its surface form.
- Budget (default 160; the exploratory secondary mode runs the identical policy
  at a different budget, e.g. 200): GREEDY inclusion of WHOLE surnames in hash
  order - each surname is included iff the joined string still fits the budget,
  otherwise it is dropped (recorded, logged) and the walk continues. A surname
  is never truncated. Token counts use the ACTUAL model tokenizer exactly the
  way faster-whisper will consume the string:
  `tokenizer.encode(" " + hotwords.strip(), add_special_tokens=False)` (see
  faster_whisper/tokenizer.py::Tokenizer.encode and transcribe.py::get_prompt).
  Budgets stay < 223 (faster-whisper's own silent cut at max_length//2 - 1), so
  the upstream truncation can never fire.
- No document-frequency filtering, ever, in this module.
- Uncovered meeting (no roster key, or empty roster) -> None: the caller must
  make the window an exact no-op (do not pass the kwarg at all).

Tokenizer: pass any object exposing `encode(text, add_special_tokens=False)` whose
result has `.ids` (e.g. `tokenizers.Tokenizer.from_file(<ct2 dir>/tokenizer.json)`).
The ct2 model dir ships `tokenizer.json`, so the tokenizer loads WITHOUT the model.
Tests stub it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROSTERS_PATH = ROOT / "data/pii/rosters_full.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"
HOTWORD_TOKEN_BUDGET = 160
FROZEN_SALT = "oc-hotwords-2026-08-12"

log = logging.getLogger("roster_hotwords")

_ROSTERS: dict | None = None
_TERMS_CACHE: dict[str, list[dict] | None] = {}


def _load_rosters() -> dict:
    global _ROSTERS
    if _ROSTERS is None:
        _ROSTERS = json.loads(ROSTERS_PATH.read_text())
    return _ROSTERS


def _load_person_terms(city: str) -> list[dict] | None:
    """person_surname terms for a city, or None when no term file exists."""
    if city not in _TERMS_CACHE:
        p = TERMS_DIR / f"{city}.json"
        if not p.exists():
            _TERMS_CACHE[city] = None
        else:
            data = json.loads(p.read_text())
            _TERMS_CACHE[city] = [t for t in data["terms"]
                                  if t["klass"] == "person_surname"]
    return _TERMS_CACHE[city]


def norm_key(s: str) -> str:
    """The frozen dedup/hash normalization: NFC + casefold."""
    return unicodedata.normalize("NFC", s).casefold()


def rnorm(s: str) -> str:
    """The audit's matching normalization (name_lexicon_audit.py): NFD strip
    combining marks, NFC, lowercase, fold final sigma."""
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s).lower().replace("ς", "σ")


def hash_rank(meeting_key: str, surname: str) -> str:
    """The 2nd amendment's frozen rank: SHA-256(salt || meeting || surname_key)."""
    payload = (FROZEN_SALT + meeting_key + norm_key(surname)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _has_initial(entry: str) -> bool:
    return any(re.fullmatch(r"\w\.", w) for w in entry.split())


def _surnames_from_terms(person_terms: list[dict],
                         entries: list[str]) -> list[str]:
    """Canonical surnames of the persons of THIS meeting, term-file path."""
    roster_norm = [rnorm(e) for e in entries]
    roster_words: set[str] = set()
    for e in roster_norm:
        roster_words.update(re.findall(r"\w+", e))
    surnames = []
    for t in person_terms:
        in_roster = (rnorm(t["canonical"]) in roster_words
                     or any(rnorm(c) in roster_norm for c in t.get("covers", [])))
        if in_roster:
            surnames.append(t["canonical"])
    return surnames


def _surnames_from_roster(entries: list[str]) -> list[str]:
    """Fallback parse for cities without a term file. See module docstring for
    its documented imprecision (two-word party names look like persons)."""
    surnames, singles = [], []
    first_words: set[str] = set()
    for e in entries:
        if "," in e:
            continue                       # party-like / list entry
        words = e.split()
        if len(words) > 2:
            continue                       # party-like
        if len(words) == 2:
            surnames.append(words[-1])     # "Β. Μαυρίδης" / "Βασίλης Μαυρίδης"
            if not _has_initial(e):
                first_words.add(norm_key(words[0]))
        else:
            singles.append(e)
    # a single-word entry is a surname unless it is evidently a first name
    surnames.extend(e for e in singles if norm_key(e) not in first_words)
    return surnames


def ordered_candidates(city: str, meeting: str, entries: list[str],
                       terms: list[dict] | None = None) -> tuple[list[str], dict]:
    """Deterministic candidate order: unique surnames ranked by the frozen
    per-meeting hash (2nd amendment). Returns (ordered, meta).

    `terms`: explicit person_surname term list, or None to load the city's term
    file (fallback roster parsing only when no file exists).
    """
    cleaned = [e.strip() for e in entries if e and e.strip()]
    if terms is None:
        terms = _load_person_terms(city)
    if terms is not None:
        surnames = _surnames_from_terms(terms, cleaned)
        source = "term_file"
    else:
        surnames = _surnames_from_roster(cleaned)
        source = "roster_fallback"

    meeting_key = f"{city}/{meeting}"
    seen: set[str] = set()
    unique: list[str] = []
    dupes = 0
    for s in surnames:
        k = norm_key(s)
        if k in seen:
            dupes += 1
            continue
        seen.add(k)
        unique.append(s)
    # hash rank; norm_key as a total-order tiebreak (hash collisions only)
    out = sorted(unique, key=lambda s: (hash_rank(meeting_key, s), norm_key(s)))
    return out, {"source": source, "n_surnames": len(out),
                 "n_duplicates": dupes, "order": "sha256-salted",
                 "salt": FROZEN_SALT}


def count_tokens(tokenizer, hotwords: str) -> int:
    """Token count exactly as faster-whisper will encode the hotwords string."""
    return len(tokenizer.encode(" " + hotwords.strip(),
                                add_special_tokens=False).ids)


def build_hotwords_detail(city: str, meeting: str, tokenizer,
                          rosters: dict | None = None,
                          terms: list[dict] | None = None,
                          budget: int = HOTWORD_TOKEN_BUDGET) -> dict:
    """Full construction record. `hotwords` is None for uncovered meetings.

    Never returns a string over `budget` tokens, never truncates a surname, and
    never drops silently: every skipped surname is in `dropped` and logged.
    """
    assert budget < 223, "budget must stay under faster-whisper's silent 223 cut"
    rosters = _load_rosters() if rosters is None else rosters
    key = f"{city}/{meeting}"
    entries = rosters.get(key) or []
    detail = {"meeting": key, "n_roster_entries": len(entries), "budget": budget,
              "hotwords": None, "kept": [], "dropped": [], "n_duplicates": 0,
              "n_surnames": 0, "source": None, "order": None, "tokens": 0}
    if not entries:
        detail["reason"] = "uncovered"
        return detail

    ordered, meta = ordered_candidates(city, meeting, entries, terms)
    detail.update(meta)
    if not ordered:
        detail["reason"] = "no_persons_matched"
        log.warning("hotwords[%s]: roster has %d entries but no person matched "
                    "(%s)", key, len(entries), meta["source"])
        return detail

    # greedy: include each WHOLE surname in hash order iff it still fits
    kept: list[str] = []
    dropped: list[str] = []
    for s in ordered:
        candidate = ", ".join(kept + [s])
        if count_tokens(tokenizer, candidate) <= budget:
            kept.append(s)
        else:
            dropped.append(s)

    detail["kept"], detail["dropped"] = kept, dropped
    if dropped:
        log.warning("hotwords[%s]: budget %d tokens - dropped %d/%d surnames: %s",
                    key, budget, len(dropped), len(ordered), ", ".join(dropped))
    if not kept:
        detail["reason"] = "nothing_fits_budget"
        return detail

    hw = ", ".join(kept)
    tokens = count_tokens(tokenizer, hw)
    assert tokens <= budget, (
        f"hotwords[{key}]: {tokens} tokens > budget {budget} - construction bug")
    detail["hotwords"], detail["tokens"] = hw, tokens
    return detail


def build_hotwords(city: str, meeting: str, tokenizer,
                   rosters: dict | None = None,
                   budget: int = HOTWORD_TOKEN_BUDGET) -> str | None:
    """Roster -> hotwords string, or None when the meeting has no usable roster."""
    return build_hotwords_detail(city, meeting, tokenizer, rosters,
                                 budget=budget)["hotwords"]
