"""Glossary mining from the training split only.

Spec: docs/specs/fix-task-eval-harness.md §3, runbook Step 3.

Mine, from the corrected (gold) text of TRAIN chains:
  - all-caps Greek acronym tokens (ΚΕΔΕ, ΔΕΥΑ, ΕΕΤΑ ...)
  - capitalised multi-token names / single proper nouns
  - frequent before->after entity substitutions (the corrected entity form)

Group global vs per-city by cross-meeting frequency. A term must recur across
meetings to survive (single-meeting entities are already covered by the live
agenda injection — see the 2026-06-20 glossary-granularity decision), which also
keeps held-out, eval-only terms out of the glossary.
"""
from __future__ import annotations

import re
from collections import defaultdict

_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΆΈΉΊΌΎΏΪΫ"
_LOWER = "αβγδεζηθικλμνξοπρστυφωχψωάέήίόύώϊϋΐΰς"
_ACRONYM_RE = re.compile(rf"^[{_UPPER}]{{2,}}$")
_PROPER_TOKEN_RE = re.compile(rf"^[{_UPPER}][{_LOWER}{_UPPER}]+$")
_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$", flags=re.UNICODE)

# recurrence thresholds
KEEP_MIN_MEETINGS = 2   # a term must recur across >=2 meetings to be kept at all
GLOBAL_MIN_CITIES = 2   # kept terms spanning >=2 cities are global; else per-city

# stop-words that are capitalised at sentence start but are not entities
_STOP = {
    "Το", "Τα", "Της", "Των", "Τον", "Την", "Τους", "Στο", "Στη", "Στην",
    "Στον", "Στους", "Στα", "Σύμφωνα", "Και", "Με", "Για", "Από", "Είναι",
    "Ναι", "Όχι", "Κύριε", "Κυρία", "Πρόεδρε", "Επίσης", "Επειδή", "Όταν",
}


def _clean(token: str) -> str:
    return _STRIP_RE.sub("", token)


def _extract_terms(text: str) -> set[str]:
    """Candidate glossary terms from one corrected line."""
    if not text:
        return set()
    raw = text.split()
    tokens = [_clean(t) for t in raw]
    terms: set[str] = set()

    # acronyms and standalone proper nouns
    for tok in tokens:
        if not tok:
            continue
        if _ACRONYM_RE.match(tok):
            terms.add(tok)
        elif _PROPER_TOKEN_RE.match(tok) and tok not in _STOP:
            terms.add(tok)

    # capitalised multi-token phrases (e.g. «Νέα Δημοκρατία»)
    i = 0
    while i < len(tokens):
        if tokens[i] and _PROPER_TOKEN_RE.match(tokens[i]) and tokens[i] not in _STOP:
            j = i + 1
            while j < len(tokens) and tokens[j] and _PROPER_TOKEN_RE.match(tokens[j]):
                j += 1
            if j - i >= 2:
                terms.add(" ".join(tokens[i:j]))
            i = j
        else:
            i += 1
    return terms


def mine_glossary(train_chains) -> dict:
    """Return {"global": [...], "per_city": {city_id: [...]}}.

    Mined from train chains' gold_final text. Recurrence is measured in distinct
    meetings (globally and within a city).
    """
    # term -> set of meetings / set of cities; (city, term) -> set of meetings
    term_meetings: dict[str, set] = defaultdict(set)
    term_cities: dict[str, set] = defaultdict(set)
    city_term_meetings: dict[tuple, set] = defaultdict(set)

    for c in train_chains:
        meeting = c["meeting_id"]
        city = c["city_id"]
        for term in _extract_terms(c.get("gold_final", "")):
            term_meetings[term].add(meeting)
            term_cities[term].add(city)
            city_term_meetings[(city, term)].add(meeting)

    # keep only terms that recur across >=KEEP_MIN_MEETINGS meetings
    kept = {t for t, mset in term_meetings.items() if len(mset) >= KEEP_MIN_MEETINGS}

    global_terms = sorted(t for t in kept if len(term_cities[t]) >= GLOBAL_MIN_CITIES)
    global_set = set(global_terms)

    # per-city: kept, single-city terms recurring within that city
    per_city: dict[str, list[str]] = defaultdict(list)
    for (city, term), mset in city_term_meetings.items():
        if term in global_set or term not in kept:
            continue
        if len(mset) >= KEEP_MIN_MEETINGS:
            per_city[city].append(term)
    per_city = {city: sorted(terms) for city, terms in per_city.items()}

    return {"global": global_terms, "per_city": per_city}


def prepare_retrieval_pool(gloss: dict, city_id: str) -> dict:
    """Pre-split the global + this-city terms into single words vs phrases,
    with their normalised forms, for fast per-utterance retrieval."""
    from eval.scoring import greek_normalize

    terms = list(
        dict.fromkeys(gloss.get("global", []) + gloss.get("per_city", {}).get(city_id, []))
    )
    words = [(t, greek_normalize(t)) for t in terms if " " not in t]
    phrases = [(t, greek_normalize(t)) for t in terms if " " in t]
    return {"words": words, "phrases": phrases}


# minimum length for word-level fuzzy matching — short tokens («ένα», «όλο»)
# produce distractor noise, so only substantive words/entities are matched.
_MIN_WORD_LEN = 4


def select_glossary_terms(pool: dict, utterance: str, max_terms: int = 20,
                          word_cutoff: int = 88, phrase_cutoff: int = 90) -> list[str]:
    """Retrieve glossary terms relevant to one utterance (fuzzy, from input only).

    Single-word terms are matched against utterance tokens (close-but-not-exact,
    the misspelling case); phrases via partial-ratio over the whole line. This
    mirrors the prompt's roster/name matching and bounds the injected block.
    """
    from rapidfuzz import fuzz

    from eval.scoring import greek_normalize

    u = greek_normalize(utterance)
    if not u:
        return []
    utoks = [t for t in u.split() if len(t) >= _MIN_WORD_LEN]
    utok_set = set(u.split())
    scored: dict[str, float] = {}
    for term, tn in pool["words"]:
        if not tn or len(tn) < _MIN_WORD_LEN or tn in utok_set:
            continue  # skip empties, short noise, and already-correct tokens
        best = max((fuzz.ratio(tn, tok) for tok in utoks), default=0)
        if best >= word_cutoff:
            scored[term] = max(scored.get(term, 0), best)
    for term, tn in pool["phrases"]:
        if not tn:
            continue
        sc = fuzz.partial_ratio(tn, u)
        if sc >= phrase_cutoff:
            scored[term] = max(scored.get(term, 0), sc)
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:max_terms]
    return [t for t, _ in ranked]


def build_glossary_block(gloss: dict, city_id: str) -> str:
    """Render the glossary block injected into the augmented user prompt:
    global terms + this city's terms. Empty string if nothing to inject."""
    global_terms = gloss.get("global", [])
    city_terms = gloss.get("per_city", {}).get(city_id, [])
    if not global_terms and not city_terms:
        return ""
    lines = ["Known terms (correct spelling — use when a word matches one phonetically):"]
    if global_terms:
        lines.append("General: " + ", ".join(global_terms))
    if city_terms:
        lines.append("This municipality: " + ", ".join(city_terms))
    return "\n".join(lines) + "\n"


def render_terms_block(terms: list[str]) -> str:
    """Render a retrieved-terms block for injection into the augmented prompt."""
    if not terms:
        return ""
    return (
        "Known correct spellings for names/terms in this municipality "
        "(use one only if a word in the text is clearly the same word misspelled):\n"
        + ", ".join(terms)
        + "\n"
    )
