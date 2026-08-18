#!/usr/bin/env python3
"""Runner for F1, the LLM majority arbiter. EXPLORATORY_CONTAMINATED_NOT_CONFIRMATORY.

Frozen preregistration: `docs/specs/2026-08-17-llm-f1-arbiter-prereg.md`.
Failed audit that makes every number here exploratory: `exp-2026-08-17-confirmation-audit`.

THREE STAGES, and the firewall between them is the point.

  pilot     picks the production batch size from pass-level invalid rate alone, on 120
            deterministic reference-blind questions at batch sizes {6,12,24,48}, plus
            an A/A replicate to estimate the stochastic disagreement floor.
  infer     builds the questions, asks both candidate orders, seals and hashes the
            question set and the decision set. NEVER loads a reference.
  analyze   loads the sealed decisions and only now touches the reference, to score.

No confirmation batch is frozen and no confirmation is spent: this module does not
import `autoresearch`.

    SC=~/.cache/oc-public .venv-eval/bin/python -m eval.controlled_eval.run_llm_arbiter pilot
    ...                                                                        infer
    ...                                                                        analyze
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval import exp_llm_arbiter as A                  # noqa: E402
from eval.controlled_eval.exp_llm_arbiter import (                     # noqa: E402
    FUNCTION_WORDS, bucket_of, is_numeric, surface_suffix_neighbor,
)
from eval.controlled_eval.exp_fusion_deletions import rates, sdi       # noqa: E402
from eval.controlled_eval.exp_majority_taxonomy import (               # noqa: E402
    load_meeting_contexts, load_term_lexicons,
)
from eval.controlled_eval.fusion_lab import (                          # noqa: E402
    _cache_path, evaluate, load_substrate, log, sc,
)
from scripts.ds_wer import TermList, aggregate, ds_wer                 # noqa: E402

STAMP = "EXPLORATORY_CONTAMINATED_NOT_CONFIRMATORY"
AUDIT = "exp-2026-08-17-confirmation-audit"
SEARCH_CITIES = ("athens", "chalandri", "chania", "orestiada", "vrilissia", "zografou")
CONFIRM_CITIES = ("argos", "samothraki", "sparta", "xylokastro")

PILOT_N = 120
PILOT_BATCHES = (6, 12, 24, 48)
INVALID_CEILING = 0.02
Z95, Z80 = 1.6448536269514722, 0.8416212335729143
DELTA0 = -0.0010                    # hypothetical future sealed-evaluation margin

TERMS_DIR = ROOT / "research/ds_wer/terms"
OUT = Path(__file__).with_name("results_llm_arbiter.json")


def store() -> Path:
    d = A.cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ stage: build
def questions_and_hash():
    """The sealed question set. Built without ever reading a reference."""
    sub = load_substrate()
    assert _cache_path().name == A.ALIGN_CACHE_EXPECTED, (
        f"MSA alignment cache key moved: {_cache_path().name} != "
        f"{A.ALIGN_CACHE_EXPECTED}. msa.py must not be edited.")
    ctxs = load_meeting_contexts(sub)
    qs = A.build_questions(sub, ctxs)
    seal = {"n_questions": len(qs),
            "sha256": A.sha256_json([[q["id"], q["majority"], q["minority"],
                                      q["context"], q["terms"], q["flip"]]
                                     for q in qs])}
    return sub, qs, seal


# ------------------------------------------------------------------ stage: pilot
def stage_pilot():
    sub, qs, seal = questions_and_hash()
    log(f"{len(qs)} eligible questions, seal {seal['sha256'][:16]}")
    sample = A.pilot_sample(qs, PILOT_N)
    out = {"stamp": STAMP, "failed_audit": AUDIT, "n": PILOT_N,
           "seal": seal, "conditions": {}}
    caches_by_bs = {}
    for bs in PILOT_BATCHES:
        cond, caches = {}, {}
        b = {q["id"]: bb for bb in A.plan_batches(sample, bs) for q in bb}
        for p in (1, 2):
            caches[p], acct = A.run_pass(sample, p, bs, store() / f"pilot_b{bs}.jsonl")
            answers = [caches[p].get(A.cache_key(q, p, b[q["id"]])) for q in sample]
            labs = [a.get("pick") if isinstance(a, dict) else None for a in answers]
            cond[f"pass{p}"] = {
                "invalid": sum(1 for a in answers if a is None),
                "invalid_rate": sum(1 for a in answers if a is None) / len(sample),
                "abstain": sum(1 for l in labs if l == A.ABSTAIN),
                **{k: v for k, v in acct.items() if k != "jobs"}}
        caches_by_bs[bs] = caches
        r = A.resolve(sample, caches, bs)
        cond["outcomes"] = dict(Counter(v["outcome"] for v in r.values()))
        cond["pass_level_invalid_rate"] = (
            (cond["pass1"]["invalid"] + cond["pass2"]["invalid"]) / (2 * len(sample)))
        cond["wall_s"] = cond["pass1"]["wall_s"] + cond["pass2"]["wall_s"]
        out["conditions"][str(bs)] = cond
        log(f"batch {bs}: pass-level invalid {cond['pass_level_invalid_rate']:.4f} "
            f"wall {cond['wall_s']:.0f}s outcomes {cond['outcomes']}")

    ok = [bs for bs in PILOT_BATCHES
          if out["conditions"][str(bs)]["pass_level_invalid_rate"] <= INVALID_CEILING]
    if ok:
        chosen, why = max(ok), f"largest batch with invalid rate <= {INVALID_CEILING}"
    else:
        best = min(out["conditions"][str(b)]["pass_level_invalid_rate"]
                   for b in PILOT_BATCHES)
        chosen = max(b for b in PILOT_BATCHES
                     if out["conditions"][str(b)]["pass_level_invalid_rate"] == best)
        why = "no batch met the ceiling; lowest invalid rate, ties to the larger batch"
    out["chosen_batch_size"], out["selection_reason"] = chosen, why
    log(f"CHOSEN batch size {chosen} ({why})")

    # A/A replicate at the chosen batch size: same candidate order asked twice.
    cache3, acct3 = A.run_pass(sample, 3, chosen, store() / f"pilot_aa_b{chosen}.jsonl")
    r_aa = A.resolve(sample, {1: caches_by_bs[chosen][1], 3: cache3}, chosen,
                     passes=(1, 3))
    out["aa_replicate"] = {
        "outcomes": dict(Counter(v["outcome"] for v in r_aa.values())),
        "stochastic_disagreement_rate": sum(
            1 for v in r_aa.values() if v["outcome"] == "order_disagree") / len(sample),
        "wall_s": acct3["wall_s"]}
    log(f"A/A stochastic disagreement {out['aa_replicate']['stochastic_disagreement_rate']:.4f}")
    (store() / "pilot.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ------------------------------------------------------------------ stage: infer
def stage_infer(batch_size: int):
    sub, qs, seal = questions_and_hash()
    log(f"{len(qs)} eligible questions, seal {seal['sha256'][:16]}, batch {batch_size}")
    caches, accts = {}, {}
    t0 = time.time()
    for p in (1, 2):
        caches[p], accts[p] = A.run_pass(
            qs, p, batch_size, store() / f"answers_p{p}_b{batch_size}.jsonl")
    dec = A.resolve(qs, caches, batch_size)
    sealed = {
        "stamp": STAMP, "failed_audit": AUDIT, "batch_size": batch_size,
        "seal": seal,
        "accounting": {str(p): {k: v for k, v in accts[p].items() if k != "jobs"}
                       for p in accts},
        "n_jobs": sum(len(accts[p]["jobs"]) for p in accts),
        "wall_s": time.time() - t0,
        "decisions": {qid: {"outcome": v["outcome"], "token": v.get("token"),
                             "conf": v.get("conf")}
                      for qid, v in dec.items()},
    }
    sealed["decision_sha256"] = A.sha256_json(
        sorted((k, v["outcome"]) for k, v in sealed["decisions"].items()))
    p = store() / f"sealed_decisions_b{batch_size}.json"
    p.write_text(json.dumps(sealed, ensure_ascii=False))
    log(f"sealed {len(dec)} decisions -> {p} ({sealed['decision_sha256'][:16]})")
    log(dict(Counter(v["outcome"] for v in dec.values())))
    return sealed


# ------------------------------------------------------------------ stage: analyze
def power_block(detail) -> dict:
    """Meeting-clustered ratio-estimator planning arithmetic. NOT a test."""
    n_m, d_m = defaultdict(int), defaultdict(int)
    for i, mtg in enumerate(detail["meetings"]):
        n_m[mtg] += detail["rows_W"][i][3]
        d_m[mtg] += (sum(detail["rows_arm"][i][:3]) - sum(detail["rows_W"][i][:3]))
    ms = sorted(n_m)
    M, N = len(ms), sum(n_m.values())
    D = sum(d_m.values())
    delta = D / N
    rss = sum((d_m[m] - delta * n_m[m]) ** 2 for m in ms)
    se = math.sqrt((M / (M - 1)) * rss / (N ** 2))
    gap = DELTA0 - delta
    z = Z95 + Z80
    if delta < DELTA0:
        k_req = math.ceil(M * ((z * se) / gap) ** 2)
        tok_req = math.ceil(k_req * (N / M))
    else:
        k_req = tok_req = None
    grid = {}
    for assumed in (-0.0015, -0.0020, -0.0030, -0.0040, -0.0050, -0.0075, -0.0100):
        g = DELTA0 - assumed
        grid[f"{assumed:+.4f}"] = {
            "meetings_required": math.ceil(M * ((z * se) / g) ** 2),
            "ref_tokens_required": math.ceil(
                math.ceil(M * ((z * se) / g) ** 2) * (N / M))}
    return {
        "note": ("planning arithmetic for a HYPOTHETICAL future sealed evaluation. "
                 "Not a test. No p-value, no significance claim, no gate result."),
        "sign_convention": "negative delta = F1 better than W",
        "M_meetings": M, "N_ref_tokens": N, "net_edits_D": D,
        "delta_hat": delta, "residual_sum_squares": rss, "SE": se,
        "margin_delta0": DELTA0, "margin_gap_delta0_minus_delta": gap,
        "z_alpha_one_sided_0.05": Z95, "z_power_0.80": Z80,
        "meetings_required_at_observed_effect": k_req,
        "ref_tokens_required_at_observed_effect": tok_req,
        "mean_tokens_per_meeting": N / M,
        "mde_at_available_mass_wer": DELTA0 - z * se,
        "mde_at_available_mass_edits": (DELTA0 - z * se) * N,
        "rounding": "ceil on all cluster and token requirements",
        "sensitivity_grid_assumed_true_effect": grid,
        "per_meeting": {m: {"n_m": n_m[m], "d_m": d_m[m]} for m in ms},
    }


def stage_analyze(batch_size: int):
    sealed = json.loads(
        (store() / f"sealed_decisions_b{batch_size}.json").read_text())
    sub, qs, seal = questions_and_hash()
    assert seal["sha256"] == sealed["seal"]["sha256"], "question set moved after sealing"
    byq = {q["id"]: q for q in qs}
    dec = sealed["decisions"]
    assert set(dec) == set(byq), "sealed decisions do not cover the question set"

    counts = Counter(v["outcome"] for v in dec.values())
    n = len(qs)
    overrides: dict[str, dict[int, str]] = defaultdict(dict)
    for qid, v in dec.items():
        if v["outcome"] == "override":
            q = byq[qid]
            overrides[q["item_id"]][q["col"]] = q["minority"]

    idea = A.F1Arbiter(dict(overrides))
    res = evaluate(idea, sub, fold="city", return_detail=True)
    assert idea.applied == counts["override"], (
        f"override_decisions {counts['override']} != overrides_applied {idea.applied}")

    detail = res.pop("detail")
    out_tokens, w_tokens = detail["out_tokens"], detail["w_tokens"]

    # ---- named-entity error rate (DS-WER). LEAKY TERM LISTS: the lists were mined
    # from material overlapping this benchmark and the source reports call them
    # "optimistic and leaky". Every number below carries that caveat.
    terms = {p.stem: TermList.load(p) for p in sorted(TERMS_DIR.glob("*.json"))
             if ".v2" not in p.name}
    ne = {}
    for arm, toks in (("F1", out_tokens), ("W", w_tokens)):
        rows = [ds_wer(" ".join(w.ref), " ".join(toks[w.item_id]), terms[w.city])
                for w in sub.windows if w.city in terms]
        ne[arm] = aggregate(rows)
    ne["cities_covered"] = sorted(set(w.city for w in sub.windows) & set(terms))
    ne["caveat"] = ("term lists mined from material overlapping this benchmark; the "
                    "source reports call them optimistic and leaky "
                    "(docs/reports/2026-08-17-majority-error-taxonomy.md:229)")

    # ---- descriptive buckets over the eligible set and the overrides
    _, per_city_terms, _ = load_term_lexicons()
    buckets = {"eligible": Counter(), "override": Counter(), "confirm": Counter(),
               "abstain_explicit": Counter(), "order_disagree": Counter(),
               "invalid": Counter()}
    for qid, q in byq.items():
        b = bucket_of(q["majority"], q["minority"], per_city_terms.get(q["city"], set()))
        buckets["eligible"][b] += 1
        buckets[dec[qid]["outcome"]][b] += 1

    # ---- domination and heterogeneity
    per_meeting = defaultdict(int)
    for i, mtg in enumerate(detail["meetings"]):
        per_meeting[mtg] += abs(sum(detail["rows_arm"][i][:3])
                                - sum(detail["rows_W"][i][:3]))
    tot = sum(per_meeting.values())
    dom = {"total_abs_edit_movement": tot,
           "largest_meeting_share": (max(per_meeting.values()) / tot) if tot else None,
           "n_meetings_moved": sum(1 for v in per_meeting.values() if v)}
    per_window = [abs(sum(detail["rows_arm"][i][:3]) - sum(detail["rows_W"][i][:3]))
                  for i in range(len(detail["item_ids"]))]
    dom["largest_window_share"] = (max(per_window) / tot) if tot else None

    # ---- partition split, DESCRIPTIVE ONLY, not a search/confirm inference
    def part(cities):
        idx = [i for i, c in enumerate(detail["cities"]) if c in cities]
        den = sum(detail["rows_W"][i][3] for i in idx)
        a = sum(sum(detail["rows_arm"][i][:3]) for i in idx)
        b = sum(sum(detail["rows_W"][i][:3]) for i in idx)
        return {"n_windows": len(idx), "ref_tokens": den, "wer_F1": a / den,
                "wer_W": b / den, "delta": (a - b) / den}

    result = {
        "stamp": STAMP,
        "failed_audit": AUDIT,
        "standing": ("EXPLORATORY. No confirmation batch frozen, no confirmation "
                     "spent, budget stays 5 of 5. Every interval below is DESCRIPTIVE "
                     "- never confirmatory, never gate-valid, never "
                     "multiplicity-controlled. F1 did not and could not pass a ship "
                     "gate."),
        "prereg": "docs/specs/2026-08-17-llm-f1-arbiter-prereg.md",
        "batch_size": batch_size,
        "seal": seal, "decision_sha256": sealed["decision_sha256"],
        "align_cache": _cache_path().name,
        "model": {"model": A.LLM_MODEL, "effort": A.LLM_EFFORT,
                  "prompt_version": A.PROMPT_VERSION, "seed": A.SEED,
                  "n_jobs": sealed["n_jobs"], "wall_s": sealed["wall_s"],
                  "accounting": sealed["accounting"]},
        "eligibility": {
            "n_eligible_columns": n,
            "rule": "column_class == exact_2_of_3, reference-blind by signature",
            "split_merge_stratum": sum(1 for q in qs if q["split_merge"]),
        },
        "outcomes": dict(counts),
        "rates": {
            "explicit_abstention_rate": counts["abstain_explicit"] / n,
            "order_instability_rate": counts["order_disagree"] / n,
            "invalid_rate": counts["invalid"] / n,
            "override_rate": counts["override"] / n,
            "confirm_rate": counts["confirm"] / n,
            "operational_non_decision_rate": (
                counts["abstain_explicit"] + counts["order_disagree"]
                + counts["invalid"]) / n,
            "denominator": n,
            "note": ("only `confirm` and `override` are valid model decisions. An "
                     "`invalid` or an `order_disagree` leaves W unchanged but is NOT "
                     "evidence of knowing abstention."),
        },
        "application_invariant": {
            "override_decisions": counts["override"],
            "overrides_applied": idea.applied,
            "mapping_failures": idea.mapping_failures,
            "collisions": idea.collisions,
            "ok": idea.applied == counts["override"] and not idea.collisions,
        },
        "primary_wer": res,
        "named_entity_error_rate": ne,
        "buckets": {k: dict(v) for k, v in buckets.items()},
        "bucket_precedence": ["name_entity", "numeric", "morphology", "function_word",
                              "other_content"],
        "domination": dom,
        "partition_descriptive_only": {
            "note": ("reported for heterogeneity description ONLY. This is not a "
                     "search/confirm inference and the confirm side is not a held-out "
                     "test here; no confirmation is spent."),
            "search": part(set(SEARCH_CITIES)),
            "confirm": part(set(CONFIRM_CITIES)),
        },
        "planning": power_block(detail),
        "floor_arithmetic": {
            "margin": DELTA0,
            "net_edits_for_margin": abs(DELTA0) * sum(len(w.ref) for w in sub.windows),
            "named_entity_perfect_play_edits": 41,
            "roster_funnel_cap_edits": 28,
            "function_word_edits": 276,
            "note": ("a name-targeted result cannot clear the margin even played "
                     "perfectly; names are descriptive only. A function-word gain is "
                     "largely OpenCouncil house orthographic style normalisation and "
                     "may never be headlined as ASR improvement."),
        },
        "scope": ("One fixed trio, one benchmark, one realization. Seventh "
                  "reference-conditioned pass over the same 247 windows. Per-seed WER "
                  "spread on training is 2.1 points, larger than the effect sought. "
                  "Nothing here establishes anything about ASR systems generally."),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    log(f"wrote {OUT}")
    log(json.dumps({k: result[k] for k in ("outcomes", "rates")},
                   ensure_ascii=False, indent=1))
    a, b = res["out_of_fold"], res["baseline_W"]
    log(f"WER W {b['wer']:.5f} -> F1 {a['wer']:.5f} "
        f"({res['vs_W']['wer']['delta']:+.5f} {res['vs_W']['wer']['ci95']} DESCRIPTIVE)")
    return result


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    bs = int(os.environ.get("BATCH", "0"))
    if stage == "pilot":
        stage_pilot()
    elif stage == "infer":
        stage_infer(bs or int(json.loads(
            (store() / "pilot.json").read_text())["chosen_batch_size"]))
    elif stage == "analyze":
        stage_analyze(bs or int(json.loads(
            (store() / "pilot.json").read_text())["chosen_batch_size"]))
    else:
        raise SystemExit("stage must be pilot | infer | analyze")
