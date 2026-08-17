#!/usr/bin/env python3
"""Arm E (frozen phonetic roster repair, no LLM) on top of W.

Preregistered in `docs/specs/2026-08-17-name-repair-on-w-prereg.md`, frozen before
any WER number on W was computed. Closes `exp-2026-08-11-name-repair`.

E has one measured positive, on V (the whole-window vote): -0.00083
[-0.00119,-0.00049] with both rate gates unchanged to the digit, because the repair
only ever moves substitutions. `exp-2026-08-16-composition-over-selection` displaced V
with W the same day. This script asks whether the number transfers.

THE RE-SPECIFICATION. E's firing rule was "act only where the three systems
disagree", implemented on V by masking tokens whose normalised string appears in ALL
THREE hypotheses anywhere - a SET rule, which is well defined only because V is one
system's whole window. W is assembled column by column from an exact three-way MSA and
its output is a text none of the three systems produced, so the rule is re-specified
POSITIONALLY: a W token inherits the column it was voted from and is protected iff
that column's class is `agree` ([x, x, x]). See the prereg for the justification and
for the declared secondary arm that also protects `exact_2_of_3`.

Nothing in `name_repair.py` or `msa.py` is touched. The cached MSA alignment is keyed
on sha256(msa.py); the key is re-verified at the end of the run.

TWO STRATA, never merged: pooled over all 247 windows (the shipping number) and
conditional on windows whose meeting has a roster (the mechanism number).

Writes results_name_repair_w.json (aggregates + the repair's own change list; never
transcript text).

Env: SC (cache dir), N_BOOT (10000)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval import bench_data as B                        # noqa: E402
from eval.controlled_eval import fusion_lab as FL                       # noqa: E402
from eval.controlled_eval.column_classes import column_class            # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import rates             # noqa: E402
from eval.controlled_eval.roster_lexicon import (                       # noqa: E402
    admitted_mined, build_meeting_context, load_city_terms, load_rosters, sha256,
    TERMS_DIR,
)
from eval.controlled_eval.scoring import cluster_bootstrap, head2head, wtoks  # noqa: E402
from serving_stack.name_repair import repair, rnorm                     # noqa: E402

N_BOOT = int(os.environ.get("N_BOOT", "10000"))
OUT = Path(__file__).with_name("results_name_repair_w.json")

# the three protection policies of the prereg
POLICIES = {
    "W+E": ("agree",),
    "W+E-strict": ("agree", "exact_2_of_3"),
    "W+E-unrestricted": (),
}


def log(m):
    print(m, flush=True)


# ------------------------------------------------------- column <-> token map
def token_columns(w) -> list[int]:
    """Column index behind each W token.

    `compose` votes columns in order and appends only the non-epsilon results, so the
    j-th emitted token is the j-th column whose decision carries a token. Recovering
    this by membership would mis-attribute repeated words, which is exactly what
    `oracle_select`'s docstring warns about."""
    idx = [d["col"] for d in w.decisions if d["token"] is not None]
    assert len(idx) == len(w.w_tokens), \
        f"{w.item_id}: {len(idx)} voted columns vs {len(w.w_tokens)} tokens"
    return idx


class RepairIdea(FL.Idea):
    """Arm E on W's token stream, under one protection policy.

    Fits no parameter: leave-one-city-out is vacuous by construction and
    `fusion_lab.evaluate` says so in `fold_note`."""
    fitted = False

    def __init__(self, name, ctxs, protect_classes):
        self.name = name
        self.ctxs = ctxs
        self.protect = set(protect_classes)
        self.changes: list[dict] = []
        self.abstained_protected: list[dict] = []

    def apply(self, w, params):
        ctx = self.ctxs[w.item_id]
        if not ctx.present or not w.w_tokens:
            return list(w.w_tokens)
        cols = token_columns(w)
        protected = {j for j, c in enumerate(cols)
                     if column_class(w.cols[c]) in self.protect}
        text = " ".join(w.w_tokens)
        # Codex job 73670922: the regex-index == token-index mapping holds only if the
        # join round-trips. One empty or non-\w vote would shift every later index.
        assert re.findall(r"\w+", text) == w.w_tokens, \
            f"{w.item_id}: W token stream does not round-trip through the join"
        # offset -> token index; tokens are pure \w by construction (scoring.wtoks)
        starts, pos = {}, 0
        for j, t in enumerate(w.w_tokens):
            starts[pos] = j
            pos += len(t) + 1
        res = repair(text, ctx)
        out = list(w.w_tokens)
        for c in res.changes:
            j = starts.get(c["start"])
            assert j is not None, f"{w.item_id}: change at {c['start']} off a token boundary"
            assert out[j] == c["original"], f"{w.item_id}: token mismatch at {j}"
            rec = {"window": w.item_id, "city": w.city, "meeting": w.meeting,
                   "col_class": column_class(w.cols[cols[j]]),
                   "original": c["original"], "replacement": c["replacement"],
                   "term": c["term"], "dist": c["dist"]}
            if j in protected:
                self.abstained_protected.append(rec)
                continue
            repl = wtoks(c["replacement"])
            assert len(repl) == 1, \
                f"{w.item_id}: replacement {c['replacement']!r} is not one token"
            out[j] = repl[0]
            self.changes.append(rec)
        return out


# ------------------------------------------------------------------ strata
def stratum(detail, keep, clusters_all, n_boot):
    """Clustered contrast on a subset of windows, computed from `evaluate`'s detail."""
    idx = [i for i, k in enumerate(keep) if k]
    if not idx:
        return None
    ra = [detail["rows_arm"][i] for i in idx]
    rw = [detail["rows_W"][i] for i in idx]
    cl = [clusters_all[i] for i in idx]
    ca = [(sum(r[:3]), r[3]) for r in ra]
    cw = [(sum(r[:3]), r[3]) for r in rw]
    a, b = rates(ra), rates(rw)
    out = {
        "n_windows": len(idx), "n_meetings": len(set(cl)),
        "ref_tokens": sum(r[3] for r in ra),
        "arm": a, "W": b,
        "wer": cluster_bootstrap(ca, cw, cl, n_boot=n_boot),
        "head2head_wer": head2head(ca, cw),
    }
    for metric, k in (("sub_rate", 0), ("del_rate", 1), ("ins_rate", 2)):
        out[metric] = cluster_bootstrap([(r[k], r[3]) for r in ra],
                                        [(r[k], r[3]) for r in rw], cl, n_boot=n_boot)
    out["gates"] = {
        "del_rate_gate": {"W": b["del_rate"], "arm": a["del_rate"],
                          "pass": a["del_rate"] <= b["del_rate"]},
        "ins_rate_gate": {"W": b["ins_rate"], "arm": a["ins_rate"],
                          "pass": a["ins_rate"] <= b["ins_rate"]},
        # Codex job 73670922: the primary test is DIRECTIONAL. A CI entirely above
        # zero establishes harm and must never pass as "excludes zero".
        "directional_ci_upper_below_zero": out["wer"]["ci95"][1] < 0,
        "wer_ci_excludes_zero": out["wer"]["excludes_zero"],
        "wer_improves": a["wer"] < b["wer"],
    }
    return out


def domination(detail):
    """Share of the total gain carried by the single most influential window AND by
    the single most influential meeting.

    Codex job de7a5729: the bootstrap unit here is the MEETING, so a window-level
    domination check is the weaker of the two and both are reported. The units are
    net reference-edit operations removed, not "errors" — this is
    agreement-with-OpenCouncil."""
    gains = [(sum(detail["rows_W"][i][:3]) - sum(detail["rows_arm"][i][:3]), i)
             for i in range(len(detail["item_ids"]))]
    tot = sum(g for g, _ in gains)
    by_meeting: dict = {}
    for g, i in gains:
        m = detail["meetings"][i]
        by_meeting[m] = by_meeting.get(m, 0) + g
    gains.sort(reverse=True)
    top = gains[0]
    mtop = max(by_meeting.items(), key=lambda kv: kv[1])
    return {"net_reference_edits_removed": tot,
            "largest_window": detail["item_ids"][top[1]],
            "largest_window_edits_removed": top[0],
            "largest_window_share": (top[0] / tot) if tot else None,
            "largest_meeting": mtop[0],
            "largest_meeting_edits_removed": mtop[1],
            "largest_meeting_share": (mtop[1] / tot) if tot else None,
            "max_abs_meeting_contribution": max(abs(v) for v in by_meeting.values()),
            "meetings_improved": sum(1 for v in by_meeting.values() if v > 0),
            "meetings_worsened": sum(1 for v in by_meeting.values() if v < 0),
            "windows_improved": sum(1 for g, _ in gains if g > 0),
            "windows_worsened": sum(1 for g, _ in gains if g < 0),
            "windows_tied": sum(1 for g, _ in gains if g == 0)}


def di_invariance(detail):
    """Per-window deletion/insertion invariance.

    Codex job de7a5729: pooled rate equality can conceal offsetting per-window
    changes, because a substitution can move the optimal Levenshtein alignment even
    when the token count is preserved. This checks the gate below the pooled level."""
    dd = [a[1] - b[1] for a, b in zip(detail["rows_arm"], detail["rows_W"])]
    di = [a[2] - b[2] for a, b in zip(detail["rows_arm"], detail["rows_W"])]
    return {"windows_with_deletion_change": sum(1 for d in dd if d),
            "windows_with_insertion_change": sum(1 for d in di if d),
            "max_abs_window_deletion_change": max(abs(d) for d in dd) if dd else 0,
            "max_abs_window_insertion_change": max(abs(d) for d in di) if di else 0}


# ----------------------------------------------------------------------- main
def main():
    key_before = FL._cache_path().name
    sub = FL.load_substrate()
    log(json.dumps(sub.meta, indent=1))

    # ---- lexicon, exactly as exp_roster_selection.py builds it ----
    report = B.load_report(FL.RUN_ID)
    city_terms = load_city_terms()
    rosters = load_rosters()
    mined_by_city, mined_acct = admitted_mined()

    per_city_freq = {c: Counter() for c in city_terms}
    total = Counter()
    for it in report["items"]:
        for tok in wtoks(it["referenceText"]):
            t = rnorm(tok)
            total[t] += 1
            if it["cityId"] in per_city_freq:
                per_city_freq[it["cityId"]][t] += 1
    loo_freq = {c: total - per_city_freq[c] for c in per_city_freq}

    ctxs, lex_acct = {}, {}
    for w in sub.windows:
        ctx, acct = build_meeting_context(
            w.city, w.meeting, city_terms[w.city], mined_by_city.get(w.city, []),
            rosters, loo_freq.get(w.city, Counter()))
        ctxs[w.item_id] = ctx
        lex_acct[w.item_id] = acct

    has_roster = [bool(rosters.get(f"{w.city}/{w.meeting}")) for w in sub.windows]
    has_person_term = [bool(lex_acct[w.item_id]["has_roster"]
                            and any(t.startswith("person:")
                                    for t in ctxs[w.item_id].present))
                       for w in sub.windows]
    log(f"roster-covered windows: {sum(has_roster)}/{len(sub.windows)}; "
        f"with >=1 roster person term: {sum(has_person_term)}")

    # ---- column census on this substrate, outcome-blind ----
    col_census = Counter()
    tok_census = Counter()
    for w in sub.windows:
        for c in w.cols:
            col_census[column_class(c)] += 1
        for j, ci in enumerate(token_columns(w)):
            tok_census[column_class(w.cols[ci])] += 1

    clusters = [w.meeting for w in sub.windows]
    res = {
        "prereg": "docs/specs/2026-08-17-name-repair-on-w-prereg.md",
        "substrate": sub.meta,
        "n_boot": N_BOOT,
        "scorer": "eval/controlled_eval/scoring.py (not the benchmark app's)",
        "align_cache_key_before": key_before,
        "lexicon": {
            "terms_files": "research/ds_wer/terms/{city}.json (v1, frozen 2026-08-12)",
            "terms_dir_sha256": {c: sha256(TERMS_DIR / f"{c}.json") for c in city_terms},
            "mined": mined_acct["counts"],
            "sha256_candidates": mined_acct["sha256_candidates"],
            "terms_per_meeting": {
                "min": min(a["n_terms"] for a in lex_acct.values()),
                "median": sorted(a["n_terms"] for a in lex_acct.values())[len(sub.windows) // 2],
                "max": max(a["n_terms"] for a in lex_acct.values())},
        },
        "roster_coverage": {
            "definition": "non-empty data/pii/rosters_full.json entry for {city}/{meeting}",
            "windows_with_roster": sum(has_roster),
            "windows": len(sub.windows),
            "meetings_with_roster": len({w.meeting for w, h in zip(sub.windows, has_roster) if h}),
            "meetings": len(set(clusters)),
            "windows_with_roster_person_term": sum(has_person_term),
            "per_city": {c: {"windows": sum(1 for w in sub.windows if w.city == c),
                             "with_roster": sum(1 for w, h in zip(sub.windows, has_roster)
                                                if h and w.city == c)}
                         for c in sorted({w.city for w in sub.windows})},
        },
        "column_census": {"columns": dict(col_census), "w_token_columns": dict(tok_census)},
        "arms": {},
    }

    details: dict = {}
    for name, protect in POLICIES.items():
        idea = RepairIdea(name, ctxs, protect)
        r = FL.evaluate(idea, sub, fold="city", n_boot=N_BOOT, return_detail=True)
        detail = r.pop("detail")
        r["protected_classes"] = list(protect)
        r["firings"] = {
            "n_changes": len(idea.changes),
            "windows_changed": len({c["window"] for c in idea.changes}),
            "meetings_changed": len({c["meeting"] for c in idea.changes}),
            "by_distance": dict(Counter(c["dist"] for c in idea.changes)),
            "by_column_class": dict(Counter(c["col_class"] for c in idea.changes)),
            "blocked_by_protection": len(idea.abstained_protected),
            "blocked_by_column_class":
                dict(Counter(c["col_class"] for c in idea.abstained_protected)),
            "detail": idea.changes,
        }
        r["domination"] = domination(detail)
        r["di_invariance"] = di_invariance(detail)
        details[name] = detail
        r["strata"] = {
            "pooled_all_windows": stratum(detail, [True] * len(sub.windows),
                                          clusters, N_BOOT),
            "roster_conditional": stratum(detail, has_roster, clusters, N_BOOT),
            "no_roster": stratum(detail, [not h for h in has_roster], clusters, N_BOOT),
            "fired_windows_only": stratum(
                detail, [w.item_id in {c["window"] for c in idea.changes}
                         for w in sub.windows], clusters, N_BOOT),
        }
        res["arms"][name] = r
        log(FL.summary_line(r))

    # ---- direct arm-to-arm contrasts (Codex job de7a5729) ----
    # A significant primary arm beside a non-significant sensitivity does NOT show the
    # two differ. These are the paired contrasts that do. Exploratory: they share the
    # 95% level with the primary endpoint and carry no multiplicity correction.
    def arm_contrast(a, b):
        ra = [(sum(r[:3]), r[3]) for r in details[a]["rows_arm"]]
        rb = [(sum(r[:3]), r[3]) for r in details[b]["rows_arm"]]
        return cluster_bootstrap(ra, rb, clusters, n_boot=N_BOOT)

    res["arm_vs_arm"] = {
        "note": "exploratory, no multiplicity correction; positive delta = first arm worse",
        "W+E-unrestricted vs W+E": arm_contrast("W+E-unrestricted", "W+E"),
        "W+E-strict vs W+E": arm_contrast("W+E-strict", "W+E"),
    }

    res["align_cache_key_after"] = FL._cache_path().name
    res["align_cache_unchanged"] = res["align_cache_key_after"] == key_before
    res["msa_sha256"] = hashlib.sha256(
        (ROOT / "eval/controlled_eval/msa.py").read_bytes()).hexdigest()

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")
    for name, r in res["arms"].items():
        p = r["strata"]["pooled_all_windows"]
        c = r["strata"]["roster_conditional"]
        log(f"  {name:20s} pooled dWER {p['wer']['delta']:+.5f} {p['wer']['ci95']} | "
            f"roster-cond {c['wer']['delta']:+.5f} {c['wer']['ci95']} | "
            f"fires {r['firings']['n_changes']}")


if __name__ == "__main__":
    main()
