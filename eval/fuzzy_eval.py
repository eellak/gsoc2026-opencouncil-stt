"""Standalone evaluation of the deterministic fuzzy name corrector (no LLM).

Answers: does fixing names/toponyms in CODE recover named-entity errors WITHOUT
collateral damage to other categories? Runs three configurations (Codex review):
  v1  name-like single tokens only       (most conservative)
  v2  name-like single tokens + phrases
  v3  unfiltered single + phrases         (quantifies the danger of an untyped list)

For each: per-category exact-rate (= entity recall on named_entity), intervention
rate, collateral (rows made worse on NON-entity categories), and precision
(intervened rows that moved closer to gold). Pure code -> no quota.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from eval.fuzzy_correct import fuzzy_correct
from eval.scoring import greek_normalize, score_pair

ROOT = Path(__file__).resolve().parent.parent
LOOP = ROOT / "data" / "improve_loop"
GLOSS = ROOT / "data" / "glossary" / "glossary.json"

CONFIGS = [
    ("v0_strict", dict(name_like_only=True, phrases=True, max_dist_cap=1, min_len=6, margin=2)),
    ("v1_single_namelike", dict(name_like_only=True, phrases=False)),
    ("v2_single+phrase", dict(name_like_only=True, phrases=True)),
    ("v3_unfiltered", dict(name_like_only=False, phrases=True)),
]


def _terms_for(gloss, city):
    return list(gloss.get("global", [])) + gloss.get("per_city", {}).get(str(city), [])


def evaluate(rows, gloss, cfg) -> dict:
    by_cat = defaultdict(lambda: {"n": 0, "exact": 0, "intervened": 0,
                                  "worse": 0, "better": 0})
    interventions = 0
    correct_interventions = 0
    for r in rows:
        inp, gold, cat = r["input_raw"], r["gold_final"], r["category"]
        terms = _terms_for(gloss, r["city_id"])
        out = fuzzy_correct(inp, terms, **cfg)

        changed = greek_normalize(out) != greek_normalize(inp)
        s_in = score_pair(inp, inp, gold)
        s_out = score_pair(inp, out, gold)

        c = by_cat[cat]
        c["n"] += 1
        if s_out["normalized_exact"]:
            c["exact"] += 1
        if changed:
            c["intervened"] += 1
            interventions += 1
            if s_out["wer"] < s_in["wer"]:
                c["better"] += 1
                correct_interventions += 1
            elif s_out["wer"] > s_in["wer"]:
                c["worse"] += 1
    n = sum(c["n"] for c in by_cat.values())
    exact = sum(c["exact"] for c in by_cat.values())
    non_ent_worse = sum(c["worse"] for cat, c in by_cat.items() if cat != "named_entity")
    non_ent_n = sum(c["n"] for cat, c in by_cat.items() if cat != "named_entity")
    return {
        "config": cfg,
        "n": n,
        "hir": round(1 - exact / n, 4),
        "interventions": interventions,
        "precision": round(correct_interventions / interventions, 4) if interventions else None,
        "named_entity_recall": round(by_cat["named_entity"]["exact"] / by_cat["named_entity"]["n"], 4)
        if by_cat["named_entity"]["n"] else None,
        "collateral_worse_per100_nonentity": round(100 * non_ent_worse / non_ent_n, 2) if non_ent_n else None,
        "per_cat": {cat: {"n": c["n"],
                          "exact_rate": round(c["exact"] / c["n"], 3),
                          "intervened": c["intervened"],
                          "better": c["better"], "worse": c["worse"]}
                    for cat, c in sorted(by_cat.items())},
    }


def main():
    rows = [json.loads(l) for l in (LOOP / "test.jsonl").read_text().splitlines() if l.strip()]
    gloss = json.loads(GLOSS.read_text())
    # baseline = input left unchanged
    base_exact = sum(1 for r in rows if score_pair(r["input_raw"], r["input_raw"], r["gold_final"])["normalized_exact"])
    print(f"test rows: {len(rows)}  | baseline (no-op) HIR = {1 - base_exact/len(rows):.4f}\n")
    out = {"test_n": len(rows), "baseline_hir": round(1 - base_exact / len(rows), 4), "configs": {}}
    for name, cfg in CONFIGS:
        res = evaluate(rows, gloss, cfg)
        out["configs"][name] = res
        print(f"### {name}")
        print(f"  HIR {res['hir']}  | named_entity recall {res['named_entity_recall']} "
              f"| precision {res['precision']} | interventions {res['interventions']} "
              f"| collateral worse/100 non-entity {res['collateral_worse_per100_nonentity']}")
        for cat, c in res["per_cat"].items():
            print(f"    {cat:22} exact {c['exact_rate']:.3f}  intervened {c['intervened']:3}  "
                  f"(+{c['better']}/-{c['worse']})")
        print()
    (LOOP / "fuzzy_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"saved -> {LOOP/'fuzzy_eval.json'}")


if __name__ == "__main__":
    main()
