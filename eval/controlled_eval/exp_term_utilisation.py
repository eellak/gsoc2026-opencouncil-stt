#!/usr/bin/env python3
"""Descriptive utilisation audit for terms already in the frozen lists.

The audit deliberately measures the current arm-E behavior; it does not change the
rule, add an arm, tune a threshold, or score the sealed temporal holdout.  It reads
the 26 records retained by exp-2026-08-16-error-mined-terms and emits counts only.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval import bench_data as B  # noqa: E402
from eval.controlled_eval.fusion_lab import load_substrate  # noqa: E402
from eval.controlled_eval.exp_roster_selection import (  # noqa: E402
    TRIO,
    restricted_repair,
)
from eval.controlled_eval.roster_lexicon import (  # noqa: E402
    CITIES,
    build_meeting_context,
    load_rosters,
)
from eval.controlled_eval.scoring import wtoks  # noqa: E402
from serving_stack.name_repair import (  # noqa: E402
    RosterContext,
    rnorm,
    select,
)


RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
TERMS_DIR = ROOT / "research/ds_wer/terms"
MINED = ROOT / "data/glossary/candidates_error_mined.json"
RAW_PAIRS = ROOT / "data/glossary/error_pairs.raw.json"
OUT = ROOT / "eval/controlled_eval/results_term_utilisation.json"

CAUSE_MISSING_ROSTER = "missing-roster"
CAUSE_PHONETIC_DISTANCE = "phonetic-distance"
CAUSE_PROTECTED_AGREEMENT = "protected-agreement"
CAUSE_SURFACE_ABSENT = "surface-absent"

OUTCOME_CORRECTED = "fired-and-corrected"
OUTCOME_WORSE = "fired-and-worse"
OUTCOME_NO_FIRE = "did-not-fire"


def _one_token_norm(value: str) -> str | None:
    """Return the arm-E token normalisation for a single surface form."""
    words = re.findall(r"\w+", value or "")
    if len(words) != 1:
        return None
    return rnorm(words[0])


def load_frozen_city_terms() -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Load the current city lists, preferring a v2 file when it exists."""
    terms: dict[str, list[dict]] = {}
    used: dict[str, str] = {}
    for city in CITIES:
        v1 = TERMS_DIR / f"{city}.json"
        v2 = TERMS_DIR / f"{city}.v2.json"
        path = v2 if v2.exists() else v1
        terms[city] = json.loads(path.read_text())["terms"]
        used[city] = path.name
    return terms, used


def _human_surface_forms(existing: dict, raw_rows: list[dict]) -> set[str]:
    """Collect the corrected target surfaces, never the ASR wrong forms."""
    surfaces = {existing["term"]}
    for row in raw_rows:
        if rnorm(row.get("canonical", "")) != rnorm(existing["term"]):
            continue
        surfaces.update(row.get("display_aliases", []))
    return {n for s in surfaces if (n := _one_token_norm(s)) is not None}


def load_existing_terms() -> list[dict]:
    """Return the fixed 26-term diagnostic population with stable term ids."""
    mined = json.loads(MINED.read_text())
    existing = mined["existing_term_with_error_evidence"]
    if len(existing) != 26:
        raise AssertionError(f"expected 26 existing high-correction terms, got {len(existing)}")

    raw = json.loads(RAW_PAIRS.read_text())
    raw_rows = [row for lane in raw["lanes"].values() for row in lane]
    out = []
    for row in existing:
        term_id = f"existing:{row['term_id'] if 'term_id' in row else row['term']}"
        out.append({
            "term_id": term_id,
            "cities": sorted(row["cities"]),
            "corrected_surface_norms": _human_surface_forms(row, raw_rows),
            "n_corrections": row["n_corrections"],
        })
    return out


def _list_surface_norms(term: dict) -> set[str]:
    surfaces = {term.get("canonical", ""), *term.get("aliases", [])}
    return {
        normalized
        for surface in surfaces
        if (normalized := _one_token_norm(surface)) is not None
    }


def build_term_lookup(city_terms: dict[str, list[dict]]) -> dict[tuple[str, str], list[str]]:
    """Map corrected surfaces to frozen-list term ids."""
    list_lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    for city, terms in city_terms.items():
        for term in terms:
            for surface in _list_surface_norms(term):
                list_lookup[(city, surface)].append(term["id"])

    return dict(list_lookup)


def coverage(
    existing: list[dict],
    list_lookup: dict[tuple[str, str], list[str]],
) -> dict:
    """Compare exact corrected surfaces with exact one-token list surfaces."""
    term_city_total = sum(len(item["cities"]) for item in existing)
    term_city_present = 0
    terms_present = 0
    surface_total = 0
    surface_present = 0
    by_term: dict[str, dict] = {}

    for item in existing:
        item_pairs_present = 0
        item_surface_total = len(item["corrected_surface_norms"]) * len(item["cities"])
        item_surface_present = 0
        for city in item["cities"]:
            matching = [
                surface
                for surface in item["corrected_surface_norms"]
                if list_lookup.get((city, surface))
            ]
            if matching:
                term_city_present += 1
                item_pairs_present += 1
            item_surface_present += len(matching)
        surface_total += item_surface_total
        surface_present += item_surface_present
        if item_pairs_present:
            terms_present += 1
        by_term[item["term_id"]] = {
            "term_city_pairs": {"count": item_pairs_present, "denominator": len(item["cities"])},
            "surface_forms_present": {
                "count": item_surface_present,
                "denominator": item_surface_total,
            },
            "human_corrected_surface_forms": {
                "count": len(item["corrected_surface_norms"]),
                "denominator": len(item["corrected_surface_norms"]),
            },
        }

    return {
        "terms_present": {"count": terms_present, "denominator": len(existing)},
        "term_city_pairs_present": {"count": term_city_present, "denominator": term_city_total},
        "surface_forms_present": {"count": surface_present, "denominator": surface_total},
        "human_corrected_surface_forms": {
            "count": surface_total,
            "denominator": surface_total,
        },
        "by_term_id": by_term,
    }


def classify_near_miss(
    reference_token: str,
    hypothesis_token: str,
    context: RosterContext | None,
    protected: set[str],
    *,
    term_id: str | None = None,
    requires_roster: bool = False,
    has_roster: bool = True,
    hypothesis_text: str | None = None,
    token_start: int | None = None,
    token_end: int | None = None,
) -> dict:
    """Classify one reference/hypothesis token pair at the diagnostic seam.

    The actual repair is always delegated to ``restricted_repair``.  The extra
    cause labels are diagnostics around that frozen call; they do not alter it.
    """
    text = hypothesis_text if hypothesis_text is not None else hypothesis_token
    ref_norm = rnorm(reference_token)
    hyp_norm = rnorm(hypothesis_token)

    if context is None or not context.present or (requires_roster and not has_roster):
        return {"outcome": OUTCOME_NO_FIRE, "cause": CAUSE_MISSING_ROSTER, "replacement": None}

    if term_id is None or term_id not in context.present:
        return {"outcome": OUTCOME_NO_FIRE, "cause": CAUSE_SURFACE_ABSENT, "replacement": None}

    listed = {
        normalized
        for surface in (
            context.present[term_id]["term"].get("canonical", ""),
            *context.present[term_id]["term"].get("aliases", []),
        )
        if (normalized := _one_token_norm(surface)) is not None
    }
    if ref_norm not in listed:
        return {"outcome": OUTCOME_NO_FIRE, "cause": CAUSE_SURFACE_ABSENT, "replacement": None}

    if hyp_norm in protected:
        return {
            "outcome": OUTCOME_NO_FIRE,
            "cause": CAUSE_PROTECTED_AGREEMENT,
            "replacement": None,
        }

    repaired, changes = restricted_repair(text, context, protected)
    change = None
    for candidate in changes:
        if token_start is not None and token_end is not None:
            if candidate["start"] == token_start and candidate["end"] == token_end:
                change = candidate
                break
        elif rnorm(candidate["original"]) == hyp_norm:
            change = candidate
            break

    if change is not None:
        replacement = change["replacement"]
        outcome = OUTCOME_CORRECTED if rnorm(replacement) == ref_norm else OUTCOME_WORSE
        return {"outcome": outcome, "cause": None, "replacement": replacement}

    decision = select(hyp_norm, context, text)["decision"]
    cause = CAUSE_PHONETIC_DISTANCE if decision == "no_candidate" else f"frozen-guard:{decision}"
    return {"outcome": OUTCOME_NO_FIRE, "cause": cause, "replacement": None}


def _one_to_one_replacements(reference: list[str], hypothesis: list[str]) -> dict[int, int]:
    """Return reference-index -> hypothesis-index single-token substitutions."""
    a = [rnorm(token) for token in reference]
    b = [rnorm(token) for token in hypothesis]
    pairs: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
            pairs[i1] = j1
    return pairs


def _token_spans(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text = " ".join(tokens)
    return text, [(m.start(), m.end()) for m in re.finditer(r"\w+", text)]


def _empty_term_stats() -> dict:
    return {
        "near_miss_occurrences": 0,
        OUTCOME_CORRECTED: 0,
        OUTCOME_WORSE: 0,
        OUTCOME_NO_FIRE: 0,
        "causes": Counter(),
    }


def _count_pair(stats: dict, result: dict) -> None:
    stats["near_miss_occurrences"] += 1
    stats[result["outcome"]] += 1
    if result["outcome"] == OUTCOME_NO_FIRE:
        stats["causes"][result["cause"]] += 1


def _seen_frequencies(report: dict) -> dict[str, Counter]:
    per_city = {city: Counter() for city in CITIES}
    total = Counter()
    for item in report["items"]:
        tokens = wtoks(item["referenceText"])
        total.update(rnorm(token) for token in tokens)
        if item["cityId"] in per_city:
            per_city[item["cityId"]].update(rnorm(token) for token in tokens)
    return {city: total - per_city[city] for city in CITIES}


def _make_context_factory(city_terms: dict[str, list[dict]], report: dict):
    rosters = load_rosters()
    frequencies = _seen_frequencies(report)
    cache: dict[tuple[str, str], tuple[RosterContext, bool]] = {}

    def get(city: str, meeting: str) -> tuple[RosterContext, bool]:
        key = (city, meeting)
        if key not in cache:
            roster = rosters.get(f"{city}/{meeting}", [])
            ctx, _ = build_meeting_context(
                city,
                meeting,
                city_terms[city],
                [],
                rosters,
                frequencies.get(city, Counter()),
            )
            cache[key] = (ctx, bool(roster))
        return cache[key]

    return get


def run_diagnostic() -> dict:
    # Keep the benchmark substrate on the documented local cache by default.  This
    # diagnostic must never turn a cache miss into a network fetch.
    os.environ.setdefault("SC", str(Path.home() / ".cache/oc-public"))
    city_terms, used_files = load_frozen_city_terms()
    existing = load_existing_terms()
    list_lookup = build_term_lookup(city_terms)
    cov = coverage(existing, list_lookup)

    substrate = load_substrate()
    report = B.load_report(RUN_ID)
    get_context = _make_context_factory(city_terms, report)

    targets_by_city: dict[str, list[dict]] = defaultdict(list)
    for item in existing:
        for city in item["cities"]:
            list_ids = []
            for surface in item["corrected_surface_norms"]:
                list_ids.extend(list_lookup.get((city, surface), []))
            list_ids = sorted(set(list_ids))
            target = {
                "term_id": item["term_id"],
                "corrected_surface_norms": item["corrected_surface_norms"],
                "list_term_ids": list_ids,
                "requires_roster": any(
                    term_id.startswith("person:")
                    for term_id in list_ids
                ),
            }
            targets_by_city[city].append(target)

    all_stats = _empty_term_stats()
    by_term = {item["term_id"]: _empty_term_stats() for item in existing}

    for window in substrate.windows:
        context, has_roster = get_context(window.city, window.meeting)
        protected = set.intersection(*[
            {rnorm(token) for token in hypothesis}
            for hypothesis in window.hyps
        ])
        pair_maps = [_one_to_one_replacements(window.ref, hypothesis) for hypothesis in window.hyps]

        for ref_index, reference_token in enumerate(window.ref):
            ref_norm = rnorm(reference_token)
            targets = [
                target
                for target in targets_by_city.get(window.city, [])
                if ref_norm in target["corrected_surface_norms"]
            ]
            if not targets:
                continue
            for target in targets:
                for system_index, hypothesis in enumerate(window.hyps):
                    hypothesis_index = pair_maps[system_index].get(ref_index)
                    if hypothesis_index is None:
                        continue
                    hypothesis_token = hypothesis[hypothesis_index]
                    if rnorm(hypothesis_token) == ref_norm:
                        continue
                    text, spans = _token_spans(hypothesis)
                    for_term_id = target["list_term_ids"][0] if target["list_term_ids"] else None
                    result = classify_near_miss(
                        reference_token,
                        hypothesis_token,
                        context,
                        protected,
                        term_id=for_term_id,
                        requires_roster=target["requires_roster"],
                        has_roster=has_roster,
                        hypothesis_text=text,
                        token_start=spans[hypothesis_index][0],
                        token_end=spans[hypothesis_index][1],
                    )
                    _count_pair(all_stats, result)
                    _count_pair(by_term[target["term_id"]], result)

    def serialise_stats(stats: dict, denominator: int) -> dict:
        return {
            "near_miss_occurrences": {"count": stats["near_miss_occurrences"], "denominator": denominator},
            OUTCOME_CORRECTED: {"count": stats[OUTCOME_CORRECTED], "denominator": denominator},
            OUTCOME_WORSE: {"count": stats[OUTCOME_WORSE], "denominator": denominator},
            OUTCOME_NO_FIRE: {"count": stats[OUTCOME_NO_FIRE], "denominator": denominator},
            "did_not_fire_causes": {
                cause: {"count": count, "denominator": stats[OUTCOME_NO_FIRE]}
                for cause, count in sorted(stats["causes"].items())
            },
        }

    result = {
        "experiment": "exp-2026-08-17-term-utilisation",
        "source_experiment": "exp-2026-08-16-error-mined-terms",
        "run_id": RUN_ID,
        "substrate": {
            "windows": len(substrate.windows),
            "meetings": len({window.meeting for window in substrate.windows}),
            "cities": len({window.city for window in substrate.windows}),
            "trio": list(TRIO),
            "sealed_holdout_read": 0,
        },
        "term_lists": used_files,
        "population": {
            "terms": len(existing),
            "human_corrections": sum(item["n_corrections"] for item in existing),
        },
        "coverage": cov,
        "utilisation": serialise_stats(all_stats, all_stats["near_miss_occurrences"]),
        "by_term_id": {
            term_id: serialise_stats(stats, stats["near_miss_occurrences"])
            for term_id, stats in sorted(by_term.items())
        },
    }
    return result


def _summary(result: dict) -> str:
    cov = result["coverage"]
    util = result["utilisation"]
    causes = util["did_not_fire_causes"]
    cause_line = " ".join(
        f"{cause}={values['count']}/{values['denominator']}"
        for cause, values in sorted(causes.items())
    ) or "none=0/0"
    return (
        f"coverage terms={cov['terms_present']['count']}/{cov['terms_present']['denominator']} "
        f"term-city={cov['term_city_pairs_present']['count']}/{cov['term_city_pairs_present']['denominator']} "
        f"surfaces={cov['surface_forms_present']['count']}/{cov['surface_forms_present']['denominator']} "
        f"human-corrected={cov['human_corrected_surface_forms']['count']}/"
        f"{cov['human_corrected_surface_forms']['denominator']}\n"
        f"utilisation near-miss={util['near_miss_occurrences']['count']} "
        f"corrected={util[OUTCOME_CORRECTED]['count']}/{util['near_miss_occurrences']['count']} "
        f"worse={util[OUTCOME_WORSE]['count']}/{util['near_miss_occurrences']['count']} "
        f"did-not-fire={util[OUTCOME_NO_FIRE]['count']}/{util['near_miss_occurrences']['count']}\n"
        f"causes {cause_line}"
    )


def main() -> None:
    result = run_diagnostic()
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
    print("term lists: " + ", ".join(f"{city}={name}" for city, name in result["term_lists"].items()))
    print(_summary(result))
    print(f"results: {OUT}")


if __name__ == "__main__":
    main()
