#!/usr/bin/env python3
"""Arm E shadow evaluation: the four safety tests of the name-repair plan
(Βήμα 2) plus ONE frozen WER pass, on the 39 frozen eval windows.

Order of operations, per prereg (docs/specs/2026-08-12-serving-stack-plan.md,
arm E): the rule and its amendment guard (scripts/serving_stack/name_repair.py)
were frozen before this script computed any WER number. No tuning after.

  a. invariance on clean : repair over the 39 REFERENCE texts, count correct->wrong
  b. transition          : repair over control hyps (eval-A), classify every change
  c. temptation          : deliberately wrong roster (next meeting, same city,
                           cyclic in sorted order), count fires and identity swaps
  d. empirical corruption: corrupt correctly-transcribed roster names with the
                           confusion ops observed in the audit's wrong->correct
                           firings, report restore rate

WER: control vs control+repair with the repo's frozen machinery — ftoks
tokenizer, exp_same_stack.sdi alignment, scoring.cluster_bootstrap clustered by
meeting — exactly as notebooks/decode_ablation.py scores (score step only; no
decode here).

Output: data/reports/finetune-research/name-repair-eval-2026-08-12.json

Run: .venv-eval/bin/python scripts/serving_stack/eval_name_repair.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap, wtoks  # noqa: E402
from ds_wer import align, MATCH, INS  # noqa: E402
from serving_stack.name_repair import (  # noqa: E402
    build_context, dl, repair, rnorm, select,
)

CACHE = Path.home() / ".cache/oc-public/decode-ablation/eval-A.json"
REPORT = ROOT / "data/reports/finetune-research/2026-08-10-corrected-adapter-report-full.json"
MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"
ROSTERS = ROOT / "data/pii/rosters_full.json"
AUDIT = ROOT / "data/reports/finetune-research/name-lexicon-audit-2026-08-12.json"
OUT = ROOT / "data/reports/finetune-research/name-repair-eval-2026-08-12.json"


def load_all():
    manifest = json.loads(MANIFEST.read_text())
    eval_windows = manifest["eval_windows"]
    hyps = json.loads(CACHE.read_text())["windows"]
    report = json.loads(REPORT.read_text())
    refs = {it["itemId"]: it["referenceText"] for it in report["items"]}
    terms = {}
    for city in ("argos", "orestiada"):
        data = json.loads((TERMS_DIR / f"{city}.json").read_text())
        terms[city] = [t for t in data["terms"] if t["klass"] == "person_surname"]
    rosters = json.loads(ROSTERS.read_text())
    # common-word list, verbatim from the Step-0 audit: seen-city references only
    seen_freq = Counter()
    for it in report["items"]:
        if it["cityId"] in ("argos", "orestiada"):
            continue
        for tok in ftoks(it["referenceText"]):
            seen_freq[rnorm(tok)] += 1
    return eval_windows, hyps, refs, terms, rosters, seen_freq


def contexts(eval_windows, terms, rosters, seen_freq, roster_override=None):
    """One RosterContext per window. roster_override: window_id -> roster key."""
    ctxs = {}
    for w in eval_windows:
        wid, city = w["window_id"], w["city"]
        key = (roster_override or {}).get(wid, f"{city}/{w['meeting_id']}")
        ctxs[wid] = build_context(terms[city], rosters.get(key, []), seen_freq)
    return ctxs


def classify_changes(ref_text, hyp_text, ctx):
    """Walk ftoks(hyp) with the frozen scorer tokenization, align to ftoks(ref),
    classify every rule firing. Mirrors the Step-0 audit's alignment method."""
    rt, ht = ftoks(ref_text), ftoks(hyp_text)
    raw_ops = align(rt, ht)
    hyp_aligned_ref = {}
    for op, i, j in raw_ops:
        if j is not None:
            hyp_aligned_ref[j] = (op, rt[i] if i is not None else None)
    rows = []
    for jdx, tok in enumerate(ht):
        tn = rnorm(tok)
        res = select(tn, ctx, hyp_text)
        if res["decision"] != "fire":
            continue
        sel = res["selected"]
        surface = wtoks(ctx.alias_surface[(sel["term"], sel["alias"])])[0]
        op, ref_tok = hyp_aligned_ref.get(jdx, (INS, None))
        row = {"hyp_token": tok, "replacement_wtok": surface, "term": sel["term"],
               "person": sel["person"], "dist": sel["dist"], "align_op": op,
               "ref_token": ref_tok}
        if op == MATCH:
            row["effect"] = "correct_to_wrong"
        elif ref_tok is None:
            row["effect"] = "fired_on_insertion"
        elif surface == ref_tok:
            row["effect"] = "wrong_to_correct"
        else:
            d_before = dl(tn, rnorm(ref_tok))
            d_after = dl(rnorm(surface), rnorm(ref_tok))
            if d_after < d_before:
                row["effect"] = "wrong_to_less_wrong"
            elif d_after > d_before:
                row["effect"] = "wrong_to_worse"
            else:
                row["effect"] = "wrong_to_other_wrong"
        rows.append(row)
    return rows


# ------------------------------------------------------------------ safety a

def test_a_invariance(eval_windows, refs, ctxs):
    fires = []
    for w in eval_windows:
        wid = w["window_id"]
        res = repair(refs[wid], ctxs[wid])
        for ch in res.changes:
            fires.append({"window": wid, **{k: ch[k] for k in
                          ("original", "replacement", "term", "dist")}})
    return {"windows": len(eval_windows), "correct_to_wrong": len(fires),
            "fires": fires, "pass": len(fires) == 0}


# ------------------------------------------------------------------ safety b

def test_b_transition(eval_windows, refs, hyps, ctxs):
    rows = []
    for w in eval_windows:
        wid = w["window_id"]
        ch = classify_changes(refs[wid], hyps[wid]["text"], ctxs[wid])
        for r in ch:
            r["window"] = wid
        rows.extend(ch)
        # consistency: token-walk fires == text-level repair changes
        n_text = len(repair(hyps[wid]["text"], ctxs[wid]).changes)
        n_tok = sum(1 for r in ch
                    if r["replacement_wtok"] != wtoks(r["hyp_token"])[0])
        assert n_text == n_tok, (wid, n_text, n_tok)
    by_effect = Counter(r["effect"] for r in rows)
    return {"fires_total": len(rows), "by_effect": dict(by_effect), "rows": rows}


# ------------------------------------------------------------------ safety c

def test_c_temptation(eval_windows, refs, hyps, terms, rosters, seen_freq):
    """Deliberately wrong roster, two variants:
      - same_city_next_meeting: next roster key of the same city (sorted,
        cyclic). NOTE: rosters of the same municipality overlap almost fully
        across meetings, so this variant mostly probes roster-date robustness.
      - other_city: first roster key of the OTHER eval city — a genuinely
        wrong roster; expected ~0 fires and 0 identity swaps."""
    by_city = defaultdict(list)
    for k in sorted(rosters):
        by_city[k.split("/")[0]].append(k)
    other = {"argos": "orestiada", "orestiada": "argos"}

    def run(override, label):
        ctxs = contexts(eval_windows, terms, rosters, seen_freq, override)
        rows = []
        for w in eval_windows:
            wid = w["window_id"]
            ch = classify_changes(refs[wid], hyps[wid]["text"], ctxs[wid])
            for r in ch:
                r["window"] = wid
                r["wrong_roster"] = override[wid]
            rows.extend(ch)
        identity_swaps = [r for r in rows if r["effect"] == "correct_to_wrong"]
        return {"variant": label, "roster_map": override,
                "fires_total": len(rows),
                "by_effect": dict(Counter(r["effect"] for r in rows)),
                "identity_swaps_on_correct": len(identity_swaps), "rows": rows}

    ov_same, ov_cross = {}, {}
    for w in eval_windows:
        wid, city = w["window_id"], w["city"]
        own = f"{city}/{w['meeting_id']}"
        keys = by_city[city]
        ov_same[wid] = (keys[(keys.index(own) + 1) % len(keys)]
                        if own in keys else keys[0])
        ov_cross[wid] = by_city[other[city]][0]
    return {"same_city_next_meeting": run(ov_same, "same_city_next_meeting"),
            "other_city": run(ov_cross, "other_city")}


# ------------------------------------------------------------------ safety d

def observed_ops(audit):
    """Character-level ops from the audit's wrong->correct firings (d=1 only:
    unambiguous single edit between norm token and replacement)."""
    subs, dels, inss = Counter(), Counter(), Counter()
    for r in audit["rule_firings"]:
        if r.get("effect") != "wrong_to_correct" or r.get("dist") != 1:
            continue
        a, b = r["norm"], r["replacement"]     # hyp -> correct
        # find the single edit turning b (correct) into a (observed error)
        if len(a) == len(b):
            diff = [(x, y) for x, y in zip(b, a) if x != y]
            if len(diff) == 1:
                subs[f"{diff[0][0]}->{diff[0][1]}"] += 1  # correct -> observed char
        elif len(a) == len(b) - 1:             # ASR dropped a char
            for i in range(len(b)):
                if b[:i] + b[i + 1:] == a:
                    dels[b[i]] += 1
                    break
        elif len(a) == len(b) + 1:             # ASR inserted a char
            for i in range(len(a)):
                if a[:i] + a[i + 1:] == b:
                    inss[a[i]] += 1
                    break
    return {"sub": dict(subs), "del": dict(dels), "ins": dict(inss)}


def corruptions(token_norm, ops):
    """Deterministic corrupted variants of a correct token, one per applicable
    observed op (sub at first applicable position; drop of a doubled consonant
    or observed-deleted char; no insertions applied — they need a position
    convention the audit does not pin down)."""
    out = []
    for key in sorted(ops["sub"]):
        corr, obs = key.split("->")
        i = token_norm.find(corr)
        if i > 0:  # keep first char stable: corrupting it changes capitalization site
            out.append((f"sub:{corr}->{obs}",
                        token_norm[:i] + obs + token_norm[i + 1:]))
    for ch in sorted(ops["del"]):
        i = token_norm.find(ch + ch)           # double-consonant drop
        if i >= 0:
            out.append((f"drop_double:{ch}", token_norm[:i] + token_norm[i + 1:]))
        else:
            i = token_norm.find(ch)
            if i > 0:
                out.append((f"drop:{ch}", token_norm[:i] + token_norm[i + 1:]))
    # dedupe corrupted forms, keep first label
    seen, uniq = set(), []
    for lab, c in out:
        if c != token_norm and c not in seen:
            seen.add(c)
            uniq.append((lab, c))
    return uniq


def test_d_corruption(eval_windows, refs, hyps, ctxs, ops):
    trials = []
    for w in eval_windows:
        wid = w["window_id"]
        ctx = ctxs[wid]
        if not ctx.present:
            continue
        ref_text, hyp_text = refs[wid], hyps[wid]["text"]
        rt, ht = ftoks(ref_text), ftoks(hyp_text)
        aligned = {}
        for op, i, j in align(rt, ht):
            if j is not None:
                aligned[j] = op
        raw_matches = list(re.finditer(r"\w+", hyp_text))
        # map raw token index -> ftoks index (fillers stripped)
        fidx, mapping = 0, {}
        for k, m in enumerate(raw_matches):
            if ftoks(m.group(0)):
                mapping[k] = fidx
                fidx += 1
        assert fidx == len(ht), (wid, fidx, len(ht))
        for k, m in enumerate(raw_matches):
            j = mapping.get(k)
            if j is None or aligned.get(j) != MATCH:
                continue
            tn = rnorm(m.group(0))
            if tn not in ctx.valid_aliases or len(tn) < 6:
                continue  # only budget-eligible correct roster names
            for lab, corrupt in corruptions(tn, ops):
                mutated = hyp_text[:m.start()] + corrupt + hyp_text[m.end():]
                res = repair(mutated, ctx)
                restored = any(c["start"] == m.start()
                               and rnorm(c["replacement"]) == tn
                               for c in res.changes)
                wrong_person = any(c["start"] == m.start()
                                   and rnorm(c["replacement"]) != tn
                                   for c in res.changes)
                trials.append({"window": wid, "token": m.group(0), "op": lab,
                               "corrupted": corrupt, "restored": restored,
                               "repaired_to_other": wrong_person})
    n = len(trials)
    restored = sum(t["restored"] for t in trials)
    other = sum(t["repaired_to_other"] for t in trials)
    by_op = defaultdict(lambda: [0, 0])
    for t in trials:
        fam = t["op"].split(":")[0]
        by_op[fam][0] += t["restored"]
        by_op[fam][1] += 1
    return {"trials": n, "restored": restored,
            "restore_rate": restored / n if n else None,
            "repaired_to_other_person": other,
            "by_op_family": {k: {"restored": v[0], "trials": v[1],
                                 "rate": v[0] / v[1]} for k, v in by_op.items()},
            "rows": trials}


# ------------------------------------------------------------------ WER pass

def wer_eval(eval_windows, refs, hyps, ctxs):
    per_ctrl, per_rep, blocks, wids = {}, {}, [], []
    changed = []
    for w in eval_windows:
        wid = w["window_id"]
        ref = ftoks(refs[wid])
        hyp_text = hyps[wid]["text"]
        rep = repair(hyp_text, ctxs[wid])
        per_ctrl[wid] = (*sdi(ref, ftoks(hyp_text)), len(ref))
        per_rep[wid] = (*sdi(ref, ftoks(rep.text)), len(ref))
        blocks.append(f"{w['city']}/{w['meeting_id']}")
        wids.append(wid)
        if rep.changes:
            changed.append({"window": wid, "n_changes": len(rep.changes),
                            "changes": rep.changes})

    def totals(per):
        s = sum(v[0] for v in per.values())
        d = sum(v[1] for v in per.values())
        i = sum(v[2] for v in per.values())
        n = sum(v[3] for v in per.values())
        return {"sub": s, "del": d, "ins": i, "ref_words": n,
                "wer": (s + d + i) / n}

    tc, tr = totals(per_ctrl), totals(per_rep)
    counts_c = [(sum(per_ctrl[w][:3]), per_ctrl[w][3]) for w in wids]
    counts_r = [(sum(per_rep[w][:3]), per_rep[w][3]) for w in wids]
    boot = cluster_bootstrap(counts_r, counts_c, blocks)  # delta = repair - control

    # leave-one-window-out sign stability of the pooled delta
    loo = []
    for skip in wids:
        keep = [w for w in wids if w != skip]
        ec = sum(sum(per_ctrl[w][:3]) for w in keep)
        er = sum(sum(per_rep[w][:3]) for w in keep)
        n = sum(per_ctrl[w][3] for w in keep)
        loo.append((er - ec) / n)
    signs = {"neg": sum(1 for d in loo if d < 0),
             "zero": sum(1 for d in loo if d == 0),
             "pos": sum(1 for d in loo if d > 0)}

    return {
        "control": tc, "repaired": tr,
        "delta": {"wer": tr["wer"] - tc["wer"], "sub": tr["sub"] - tc["sub"],
                  "del": tr["del"] - tc["del"], "ins": tr["ins"] - tc["ins"]},
        "bootstrap_repair_minus_control": boot,
        "leave_one_out": {"deltas_min": min(loo), "deltas_max": max(loo),
                          "sign_counts": signs,
                          "all_negative": signs["pos"] == 0 and signs["zero"] == 0},
        "windows_changed": len(changed), "changed_windows": changed,
    }


def main():
    eval_windows, hyps, refs, terms, rosters, seen_freq = load_all()
    ctxs = contexts(eval_windows, terms, rosters, seen_freq)
    audit = json.loads(AUDIT.read_text())

    result = {"generated": "2026-08-12",
              "spec": "docs/specs/2026-08-11-name-repair-plan.md Βήμα 1-2 + "
                      "arm E prereg + frozen suffix-family amendment",
              "control": str(CACHE), "windows": len(eval_windows)}

    result["safety_a_invariance_on_clean"] = test_a_invariance(eval_windows, refs, ctxs)
    result["safety_b_transition_on_errors"] = test_b_transition(
        eval_windows, refs, hyps, ctxs)
    result["safety_c_temptation_wrong_roster"] = test_c_temptation(
        eval_windows, refs, hyps, terms, rosters, seen_freq)
    ops = observed_ops(audit)
    result["safety_d_empirical_corruption"] = {
        "observed_ops_from_audit": ops,
        **test_d_corruption(eval_windows, refs, hyps, ctxs, ops)}

    # amendment accounting: what did the suffix-family guard block?
    guard_blocked = []
    for w in eval_windows:
        wid = w["window_id"]
        for tok in ftoks(hyps[wid]["text"]):
            res = select(rnorm(tok), ctxs[wid], hyps[wid]["text"])
            if res["decision"] == "abstain_suffix_family":
                guard_blocked.append({"window": wid, "token": tok,
                                      "would_have_been": res["selected"]})
    result["amendment_guard_blocked"] = guard_blocked

    # -------- the ONE WER pass, computed last, reported as-is --------
    result["wer"] = wer_eval(eval_windows, refs, hyps, ctxs)

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for k in ("safety_a_invariance_on_clean", "safety_b_transition_on_errors"):
        r = dict(result[k])
        r.pop("rows", None)
        r.pop("fires", None)
        print(k, json.dumps(r, ensure_ascii=False))
    for variant, r in result["safety_c_temptation_wrong_roster"].items():
        r = dict(r)
        r.pop("rows", None)
        r.pop("roster_map", None)
        print("safety_c", variant, json.dumps(r, ensure_ascii=False))
    d = dict(result["safety_d_empirical_corruption"])
    d.pop("rows", None)
    print("safety_d", json.dumps(d, ensure_ascii=False))
    print("guard_blocked", json.dumps(guard_blocked, ensure_ascii=False))
    wr = dict(result["wer"])
    wr.pop("changed_windows", None)
    print("wer", json.dumps(wr, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
