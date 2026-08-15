#!/usr/bin/env python3
"""Step 0 feasibility audit for the name-repair lexicon (analysis only, no production code).

Spec: docs/specs/2026-08-11-name-repair-plan.md (Βήμα 0/1) plus Codex additions in
docs/specs/2026-08-12-serving-stack-plan.md section A.

Inputs (all local, no network):
  - control hyps : ~/.cache/oc-public/decode-ablation/eval-A.json
  - references   : data/reports/finetune-research/2026-08-10-corrected-adapter-report-full.json
  - window list  : research/eval-freeze-2026-08/manifest.json (39 eval windows ONLY)
  - term files   : research/ds_wer/terms/{argos,orestiada}.json (person_surname terms)
  - rosters      : data/pii/rosters_full.json ("city/meeting" keys)

Output: data/reports/finetune-research/name-lexicon-audit-2026-08-12.json
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from ds_wer import TermList, tag, align, MATCH, SUB, DEL, INS  # noqa: E402

CACHE = Path.home() / ".cache/oc-public/decode-ablation/eval-A.json"
REPORT = ROOT / "data/reports/finetune-research/2026-08-10-corrected-adapter-report-full.json"
MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"
ROSTERS = ROOT / "data/pii/rosters_full.json"
OUT = ROOT / "data/reports/finetune-research/name-lexicon-audit-2026-08-12.json"

WORDS_DENOM = 11903  # per task instruction; manifest says 11911 eval_ref_tokens
COMMON_FREQ_THRESHOLD = 5  # token count in seen-city references => "common Greek word"


# ---------- normalization (plan Βήμα 1: NFC, lowercase, strip tonos, ς→σ) ----------

def rnorm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFC", s).lower().replace("ς", "σ")
    return s


# ---------- Damerau-Levenshtein (OSA variant) ----------

def dl(a: str, b: str, cap: int = 10) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[n][m]


def rule_allows(token_len: int, dist: int) -> bool:
    """Conservative length rule from the plan (length = hypothesis token length)."""
    if token_len < 6:
        return False
    if 6 <= token_len <= 9:
        return dist == 1
    return dist <= 2 and dist / token_len <= 0.20


# ---------- data loading ----------

def load_all():
    manifest = json.loads(MANIFEST.read_text())
    eval_windows = manifest["eval_windows"]
    hyps = json.loads(CACHE.read_text())["windows"]
    report = json.loads(REPORT.read_text())
    refs = {it["itemId"]: it["referenceText"] for it in report["items"]}

    terms = {}
    for city in ("argos", "orestiada"):
        data = json.loads((TERMS_DIR / f"{city}.json").read_text())
        person_terms = [t for t in data["terms"] if t["klass"] == "person_surname"]
        terms[city] = {"list": TermList(person_terms), "raw": person_terms}

    rosters = json.loads(ROSTERS.read_text())

    # common-word list: reference texts of SEEN cities (everything except the two
    # training-disjoint eval cities), never the 39 windows themselves
    seen_freq = Counter()
    for it in report["items"]:
        if it["cityId"] in ("argos", "orestiada"):
            continue
        for tok in ftoks(it["referenceText"]):
            seen_freq[rnorm(tok)] += 1
    return eval_windows, hyps, refs, terms, rosters, seen_freq


def meeting_roster_terms(city_terms: list[dict], roster_entries: list[str]):
    """Term ids present in this meeting's roster.

    A term is 'present' when its canonical surname (rnorm) appears as a word in any
    roster entry, or any of its covered full names appears verbatim (rnorm)."""
    roster_norm = [rnorm(e) for e in roster_entries]
    roster_words = set()
    for e in roster_norm:
        roster_words.update(re.findall(r"\w+", e))
    present = {}
    for t in city_terms:
        surname = rnorm(t["canonical"])
        covered_present = []
        for cov in t.get("covers", []):
            cn = rnorm(cov)
            if any(cn == e or cn in roster_norm for e in roster_norm):
                covered_present.append(cov)
        if surname in roster_words or covered_present:
            present[t["id"]] = {
                "term": t,
                "persons_in_meeting": covered_present or t.get("covers", []),
            }
    return present


def first_names_of(terms_present) -> set[str]:
    names = set()
    for info in terms_present.values():
        for cov in info["persons_in_meeting"]:
            parts = cov.split()
            if len(parts) >= 2:
                names.add(rnorm(parts[0]))
    return names


# ---------- distance to a term over its aliases ----------

def term_distance(token_norm: str, term: dict):
    """(min_dist, best_alias). Aliases in the term files are stored ftoks-normalized;
    rnorm folds their final sigma the same way as the token."""
    best = (99, None)
    for alias in term["aliases"]:
        a = rnorm(alias)
        d0 = dl(token_norm, a)
        if d0 < best[0]:
            best = (d0, a)
    return best


# ---------- independent signal for distance-2 (from raw hypothesis text) ----------

def has_signal(raw_hyp: str, token_norm: str, first_names: set[str]) -> dict:
    """Capital off sentence-start, adjacent 'κ.'/κύριος/κυρία, or adjacent first name."""
    words = re.findall(r"[\w.]+", raw_hyp)
    sigs = {"capital_mid": False, "kyrios": False, "first_name": False}
    for i, w in enumerate(words):
        if rnorm(re.sub(r"\.$", "", w)) != token_norm:
            continue
        if w[:1].isupper():
            prev = words[i - 1] if i > 0 else ""
            if i > 0 and not re.search(r"[.;!?]$", prev):
                sigs["capital_mid"] = True
        for j in (i - 1, i + 1):
            if 0 <= j < len(words):
                nb = rnorm(words[j])
                if nb in ("κ.", "κ", "κυριοσ", "κυρια", "κυριε"):
                    sigs["kyrios"] = True
                if nb in first_names:
                    sigs["first_name"] = True
    sigs["any"] = any(sigs.values())
    return sigs


# ---------- the selection rule ----------

def select(token_norm: str, terms_present: dict, seen_freq: Counter,
           raw_hyp: str, first_names: set[str], valid_aliases: set[str]):
    """Apply the full conservative rule to one hypothesis token.

    Returns dict with decision + reasons. decision in
    {fire, abstain_tie, abstain_margin, abstain_multi_person, abstain_common,
     abstain_already_valid, abstain_no_signal, no_candidate}."""
    out = {"token": token_norm, "decision": "no_candidate", "candidates": []}
    if token_norm in valid_aliases:
        out["decision"] = "abstain_already_valid"
        return out
    scored = []
    for tid, info in terms_present.items():
        d0, alias = term_distance(token_norm, info["term"])
        scored.append((d0, tid, alias, info))
    if not scored:
        return out
    scored.sort(key=lambda x: (x[0], x[1]))
    best_d, best_tid, best_alias, best_info = scored[0]
    if not rule_allows(len(token_norm), best_d):
        return out
    out["candidates"] = [(t, d) for d, t, _, _ in scored[:3]]
    ties = [s for s in scored if s[0] == best_d]
    if len(ties) > 1:
        out["decision"] = "abstain_tie"
        return out
    second_d = scored[1][0] if len(scored) > 1 else 99
    if second_d < best_d + 2:
        out["decision"] = "abstain_margin"
        return out
    if token_norm in seen_freq and seen_freq[token_norm] >= COMMON_FREQ_THRESHOLD:
        out["decision"] = "abstain_common"
        out["common_freq"] = seen_freq[token_norm]
        return out
    persons = best_info["persons_in_meeting"]
    out["n_persons"] = len(persons)
    if len(persons) != 1:
        out["decision"] = "abstain_multi_person"
        out["selected"] = {"term": best_tid, "alias": best_alias, "dist": best_d}
        return out
    if best_d == 2:
        sigs = has_signal(raw_hyp, token_norm, first_names)
        out["signals"] = sigs
        if not sigs["any"]:
            out["decision"] = "abstain_no_signal"
            out["selected"] = {"term": best_tid, "alias": best_alias, "dist": best_d}
            return out
    out["decision"] = "fire"
    out["selected"] = {"term": best_tid, "alias": best_alias, "dist": best_d,
                       "person": persons[0]}
    return out


# ---------- main ----------

def main():
    eval_windows, hyps, refs, terms, rosters, seen_freq = load_all()

    mention_rows = []       # every gold person-surname mention in the 39 refs
    fp_rows = []            # rule firings over ALL hyp tokens
    per_window_stats = []
    missing_roster_map = []

    for w in eval_windows:
        wid, city, meeting = w["window_id"], w["city"], w["meeting_id"]
        ref_text, hyp_text = refs[wid], hyps[wid]["text"]
        tl: TermList = terms[city]["list"]
        raw_terms = terms[city]["raw"]
        term_by_id = {t["id"]: t for t in raw_terms}

        roster_key = f"{city}/{meeting}"
        roster_entries = rosters.get(roster_key, [])
        if not roster_entries:
            missing_roster_map.append(roster_key)
        present = meeting_roster_terms(raw_terms, roster_entries)
        fnames = first_names_of(present)
        valid_aliases = set()
        for info in present.values():
            valid_aliases.update(rnorm(a) for a in info["term"]["aliases"])

        rt = ftoks(ref_text)
        ht = ftoks(hyp_text)
        ref_tag = tag(rt, tl)
        hyp_tag = tag(ht, tl)
        ops = align([s for _, s in ref_tag], [s for _, s in hyp_tag])

        n_names = n_match = n_sub = n_del = 0
        for op, i, j in ops:
            ref_is_term = i is not None and ref_tag[i][0]
            if not ref_is_term:
                continue
            tid = ref_tag[i][1]
            term = term_by_id[tid]
            # surface form of the mention in the reference (single-token surnames)
            # tag() collapses; recover surface by re-matching position
            row = {
                "window": wid, "city": city, "meeting": meeting, "term": tid,
                "canonical": term["canonical"], "outcome": None,
                "in_meeting_roster": tid in present,
            }
            n_names += 1
            if op == MATCH:
                n_match += 1
                row["outcome"] = "correct"
            elif op == DEL:
                n_del += 1
                row["outcome"] = "deleted"
            elif op == SUB:
                n_sub += 1
                hyp_sym = hyp_tag[j]
                if hyp_sym[0]:
                    row["outcome"] = "sub_other_roster_name"
                    row["hyp_trace"] = hyp_sym[1]
                else:
                    row["outcome"] = "sub_token"
                    row["hyp_trace"] = hyp_sym[1]
            mention_rows.append(row)
        per_window_stats.append({"window": wid, "meeting": f"{city}/{meeting}",
                                 "N": n_names, "match": n_match, "sub": n_sub,
                                 "del": n_del})

        # ---- surface index: which ref token does each raw hyp token align to ----
        raw_ops = align(rt, ht)
        hyp_aligned_ref = {}   # hyp idx -> (op, ref token or None)
        for op, i, j in raw_ops:
            if j is not None:
                hyp_aligned_ref[j] = (op, rt[i] if i is not None else None)

        # ---- annotate sub mentions with distances (curve) ----
        # map collapsed sub back onto raw ref surface: find ref token whose rnorm
        # matches an alias of the term; use alignment on raw tokens instead:
        # for each mention row above with sub_token trace, the trace token itself is
        # the hyp symbol (single token). Ref surface form: search rt for a token that
        # is an alias of tid — take the occurrence order.

        # ---- false-positive surface: run rule over every hyp token ----
        for jdx, tok in enumerate(ht):
            tn = rnorm(tok)
            res = select(tn, present, seen_freq, hyp_text, fnames, valid_aliases)
            if res["decision"] in ("no_candidate",):
                continue
            op, ref_tok = hyp_aligned_ref.get(jdx, (INS, None))
            fired = res["decision"] == "fire"
            row = {"window": wid, "meeting": f"{city}/{meeting}", "hyp_token": tok,
                   "norm": tn, "decision": res["decision"],
                   "align_op": op, "ref_token": ref_tok}
            if fired:
                repl = res["selected"]["alias"]
                row["selected_term"] = res["selected"]["term"]
                row["replacement"] = repl
                row["dist"] = res["selected"]["dist"]
                if op == MATCH:
                    row["effect"] = "correct_to_wrong"          # token was right
                elif ref_tok is not None and rnorm(ref_tok) == repl:
                    row["effect"] = "wrong_to_correct"
                elif ref_tok is not None:
                    # was the ref token a name mention at all?
                    ref_is_name = tuple([rnorm(ref_tok)]) and any(
                        rnorm(ref_tok) in {rnorm(a) for a in t["aliases"]}
                        for t in raw_terms)
                    row["effect"] = ("wrong_to_other_wrong_name_site"
                                     if ref_is_name else
                                     "wrong_to_other_wrong_nonname")
                else:
                    row["effect"] = "fired_on_insertion"
            fp_rows.append(row)

    # ---------- per-mention repair curve on the missed (substituted) mentions ----------
    missed = [r for r in mention_rows if r["outcome"] in
              ("deleted", "sub_token", "sub_other_roster_name")]
    subs = [r for r in mention_rows if r["outcome"] == "sub_token"]

    # ref surface forms per (window, term) in order, to score exact-form repair
    ref_surfaces = defaultdict(list)
    for w in eval_windows:
        wid, city = w["window_id"], w["city"]
        tl = terms[city]["list"]
        rt = ftoks(refs[wid])
        for is_term, sym in tag(rt, tl):
            pass
        # recover surfaces: walk tokens and re-tag keeping surface
        raw_terms = terms[city]["raw"]
        alias_to_tid = {}
        for t in raw_terms:
            for a in t["aliases"]:
                alias_to_tid[rnorm(a)] = t["id"]
        for tok in rt:
            tid = alias_to_tid.get(rnorm(tok))
            if tid:
                ref_surfaces[(wid, tid)].append(rnorm(tok))

    surface_cursor = Counter()
    curve = {"oracle_d1": 0, "d2_conservative": 0, "with_inflection": 0}
    sel_counts = Counter()
    correct_person = correct_form = 0
    oracle_repairable_ids = []

    for idx, r in enumerate(subs):
        wid, tid, city = r["window"], r["term"], r["city"]
        meeting = r["meeting"]
        trace = r["hyp_trace"]
        if r["outcome"] == "sub_other_roster_name":
            # trace is a term id, not a token: hyp already a valid roster name
            r["repair"] = {"eligible": False, "why": "hyp_is_valid_roster_name"}
            continue
        tn = rnorm(trace)
        key = (wid, tid)
        surfaces = ref_surfaces.get(key, [])
        cur = surface_cursor[key]
        gold_form = surfaces[cur] if cur < len(surfaces) else (
            surfaces[0] if surfaces else rnorm(r["canonical"]))
        surface_cursor[key] += 1

        d_gold = dl(tn, gold_form)
        term = next(t for t in terms[city]["raw"] if t["id"] == tid)
        d_inf, best_alias = term_distance(tn, term)

        rep = {"trace": trace, "gold_form": gold_form, "d_gold": d_gold,
               "d_inflected": d_inf, "best_alias": best_alias,
               "len_trace": len(tn)}
        r["repair"] = rep

        rep["oracle_d1"] = d_gold <= 1
        rep["d2_conservative"] = rule_allows(len(tn), d_gold)
        rep["with_inflection"] = rule_allows(len(tn), d_inf)
        for k in ("oracle_d1", "d2_conservative", "with_inflection"):
            if rep[k]:
                curve[k] += 1
        if rep["oracle_d1"]:
            oracle_repairable_ids.append(idx)

        # full selection with meeting roster distractors
        w = next(x for x in eval_windows if x["window_id"] == wid)
        roster_key = f"{city}/{w['meeting_id']}"
        roster_entries = rosters.get(roster_key, [])
        present = meeting_roster_terms(terms[city]["raw"], roster_entries)
        fnames = first_names_of(present)
        valid_aliases = set()
        for info in present.values():
            valid_aliases.update(rnorm(a) for a in info["term"]["aliases"])
        res = select(tn, present, seen_freq, hyps[wid]["text"], fnames, valid_aliases)
        rep["selection"] = {"decision": res["decision"],
                            "selected": res.get("selected"),
                            "candidates": res.get("candidates", [])}
        sel_counts[res["decision"]] += 1
        if res["decision"] == "fire":
            sel = res["selected"]
            rep["picked_right_person"] = sel["term"] == tid
            rep["picked_right_form"] = sel["alias"] == gold_form
            if rep["picked_right_person"]:
                correct_person += 1
                if rep["picked_right_form"]:
                    correct_form += 1

    # among ORACLE-repairable, how many fire AND pick the right person
    oracle_fire_right = sum(
        1 for i in oracle_repairable_ids
        if subs[i]["repair"]["selection"]["decision"] == "fire"
        and subs[i]["repair"].get("picked_right_person"))
    oracle_fire = sum(
        1 for i in oracle_repairable_ids
        if subs[i]["repair"]["selection"]["decision"] == "fire")

    # ---------- aggregates ----------
    n_mentions = len(mention_rows)
    n_missed = len(missed)
    n_del = sum(1 for r in mention_rows if r["outcome"] == "deleted")
    n_sub_name = sum(1 for r in mention_rows if r["outcome"] == "sub_other_roster_name")

    fires = [r for r in fp_rows if r["decision"] == "fire"]
    fx = Counter(r.get("effect") for r in fires)
    abst = Counter(r["decision"] for r in fp_rows if r["decision"] != "fire")

    by_meeting = defaultdict(Counter)
    for r in missed:
        by_meeting[f"{r['city']}/{r['meeting']}"]["missed"] += 1
    for r in subs:
        rep = r.get("repair", {})
        if rep.get("selection", {}).get("decision") == "fire" and rep.get("picked_right_person"):
            by_meeting[f"{r['city']}/{r['meeting']}"]["selectable_right"] += 1
    by_person = defaultdict(Counter)
    for r in missed:
        by_person[r["canonical"]]["missed"] += 1
        by_person[r["canonical"]][r["outcome"]] += 1

    delta_errors = (-fx.get("wrong_to_correct", 0)
                    + fx.get("correct_to_wrong", 0))
    wer_effect = delta_errors / WORDS_DENOM

    n_selectable = sel_counts.get("fire", 0)
    gate_15 = n_selectable / n_missed if n_missed else 0
    gate_80 = (oracle_fire_right / oracle_fire) if oracle_fire else None

    result = {
        "generated": "2026-08-12",
        "windows": len(eval_windows),
        "denominator_words": WORDS_DENOM,
        "manifest_ref_tokens": 11911,
        "gold_mentions": {
            "total": n_mentions,
            "correct": sum(1 for r in mention_rows if r["outcome"] == "correct"),
            "missed_total": n_missed,
            "deleted_unaddressable": n_del,
            "sub_with_token_trace": len(subs),
            "sub_to_other_roster_name": n_sub_name,
        },
        "curve": {
            "oracle_distance_1": curve["oracle_d1"],
            "distance_2_conservative_rule_vs_gold_form": curve["d2_conservative"],
            "with_inflection_handling": curve["with_inflection"],
        },
        "selection_on_missed_subs": dict(sel_counts),
        "selection_correct_person": correct_person,
        "selection_correct_exact_form": correct_form,
        "oracle_d1_that_fire": oracle_fire,
        "oracle_d1_fire_right_person": oracle_fire_right,
        "gate": {
            "selectable_share_of_missed": gate_15,
            "gate_15pct_pass": gate_15 >= 0.15,
            "oracle_right_person_share": gate_80,
            "gate_80pct_pass": (gate_80 is not None and gate_80 >= 0.80),
        },
        "false_positive_surface": {
            "total_hyp_tokens_considered": "all ftoks tokens of 39 hyps",
            "firings_total": len(fires),
            "firings_by_effect": dict(fx),
            "abstentions_by_reason": dict(abst),
        },
        "wer_effect_estimate": {
            "delta_errors": delta_errors,
            "wrong_to_correct": fx.get("wrong_to_correct", 0),
            "correct_to_wrong": fx.get("correct_to_wrong", 0),
            "neutral_or_other": {k: v for k, v in fx.items()
                                 if k not in ("wrong_to_correct", "correct_to_wrong")},
            "wer_points": wer_effect,
        },
        "clustering": {
            "by_meeting": {k: dict(v) for k, v in sorted(by_meeting.items())},
            "by_person": {k: dict(v) for k, v in sorted(by_person.items())},
        },
        "per_window": per_window_stats,
        "missing_roster_keys": missing_roster_map,
        "mentions": mention_rows,
        "rule_firings": fires,
        "abstention_rows": [r for r in fp_rows if r["decision"] != "fire"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    # console summary
    for k in ("gold_mentions", "curve", "selection_on_missed_subs", "gate",
              "wer_effect_estimate"):
        print(k, json.dumps(result[k], ensure_ascii=False))
    print("firings_by_effect", dict(fx))
    print("abstentions", dict(abst))
    print("oracle_fire", oracle_fire, "oracle_fire_right", oracle_fire_right)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
