#!/usr/bin/env python3
"""Roster-grounded selection: does a closed term list inside the fusion selector
close the distance to the oracle? (wayfinder #18)

Four arms, declared on the issue before any number was computed:

  V     the 3-system consensus vote of `exp-2026-08-16-fusion-deletions`
  V+E   the frozen phonetic closed-list repair applied to V's chosen text, NO LLM
  V+L   an LLM selector that sees the 3 candidates AND the term list, allowed only
        to pick one candidate and to swap single words for closed-list terms
  V+L+E both

Plus one ABLATION, added on Codex review (job ada1cc4a) and not a gated arm:

  V+Lsel  the LLM's CHOICE only, its replacements discarded. Without it,
          `V+L+E - V+E` cannot say whether any gain came from the LLM's
          replacements or merely from its choice of hypothesis.

Preregistered smallest meaningful improvement for the LLM over the free control
(Codex review, frozen before any number): the LLM is justified only if
WER(V+E) - WER(V+L+E) is at least 10% of the gap V+E still leaves to the trio
oracle, AND the paired bootstrap CI on that contrast excludes zero, AND both rate
gates pass. Anything less is reported as "the free control already has it".

Hard limits, from the issue, not relaxed here:
  - no free rewriting: the output is a SELECTION among the three hypotheses plus,
    optionally, single-word replacements by a closed-list term;
  - the mechanism acts ONLY on tokens the three systems disagree about; a token
    that appears identically in all three hypotheses is untouchable (this binds E
    as well as L, which is stricter than arm E was in the serving-stack ladder —
    the unrestricted variant is reported as a labelled sensitivity, not as an arm);
  - no decode-time biasing; every arm is post-hoc text.

Gates, frozen: reject on any rise in `ins_rate`; reject on any rise in `del_rate`;
leave-one-window-out on the headline delta; paired clustered bootstrap by meeting,
10,000 replicates; report the share of the oracle gap recovered.

The 6 sealed temporal-holdout windows of `eval-freeze-2026-08` are filtered out
explicitly, exactly as `exp_fusion_deletions.py` does.

Writes results_roster_selection.json (aggregates only, never transcript text).

Env: SC (cache dir), N_BOOT (10000), LLM (1 to call the model), LLM_BATCH (4)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval import bench_data as B                        # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import rates, sdi        # noqa: E402
from eval.controlled_eval.roster_lexicon import (                       # noqa: E402
    admitted_mined, build_meeting_context, load_city_terms, load_rosters, sha256,
    TERMS_DIR,
)
from eval.controlled_eval.scoring import cluster_bootstrap, head2head, wtoks  # noqa: E402
from serving_stack.name_repair import (                                 # noqa: E402
    RosterContext, repair, rnorm, rule_allows, dl,
)

RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
TRIO = ["scribe-v2-clean", "soniox", "oc-runpod-fixed-2026-08-10"]
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
OUT = Path(__file__).with_name("results_roster_selection.json")

CLIENT = "/home/harold/codex-bridge/codex_client.py"
LLM_MODEL = "gpt-5.6-luna"
LLM_CACHE = Path.home() / ".cache/oc-public/roster-selection/llm_cache.jsonl"


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ agreement
def agreed_tokens(item, providers) -> set[str]:
    """Normalised tokens that appear in EVERY hypothesis. These are the safe text
    the issue forbids touching.

    Codex (job ada1cc4a) asked for agreement defined on ALIGNED occurrences rather
    than on "the string occurs somewhere in each transcript". The set rule is kept
    because it errs in the safe direction: it protects strictly more tokens than an
    alignment would, so no arm can act anywhere an alignment-based rule would have
    forbidden. Declared, not silently resolved."""
    sets = [set(rnorm(t) for t in wtoks(item["hyp"][p])) for p in providers]
    out = sets[0]
    for s in sets[1:]:
        out = out & s
    return out


def restricted_repair(text: str, ctx: RosterContext, protected: set[str]):
    """Arm E, but blind to tokens all three systems agree on.

    Implemented by masking: a protected token is temporarily an unknown word to the
    rule. Cheapest faithful way to do it without touching the frozen rule itself."""
    if not ctx.present:
        return text, []
    res = repair(text, ctx)
    if not res.changes:
        return text, []
    keep = [c for c in res.changes if rnorm(c["original"]) not in protected]
    if len(keep) == len(res.changes):
        return res.text, res.changes
    # re-apply only the surviving changes on the original text
    pieces, last = [], 0
    for c in keep:
        pieces.append(text[last:c["start"]])
        pieces.append(c["replacement"])
        last = c["end"]
    pieces.append(text[last:])
    return "".join(pieces), keep


# ------------------------------------------------------------------- LLM arm
PROMPT = """Είσαι επιμελητής μεταγραφών ελληνικών δημοτικών συμβουλίων.

Για κάθε στοιχείο σου δίνω ΤΡΕΙΣ μεταγραφές (A, B, C) του ΙΔΙΟΥ δίλεπτου ήχου, από
τρία ανεξάρτητα συστήματα ASR, και μια ΚΛΕΙΣΤΗ ΛΙΣΤΑ όρων (επώνυμα παρόντων,
τοπωνύμια, ακρωνύμια) που αφορούν αυτή τη συνεδρίαση.

Κάνε δύο πράγματα:
1. Διάλεξε ΜΙΑ από τις A, B, C — τη συνολικά ακριβέστερη. ΜΗΝ προτιμήσεις μια
   μεταγραφή επειδή είναι μεγαλύτερη. Και οι λέξεις που λείπουν και οι λέξεις που
   περισσεύουν χωρίς να ειπώθηκαν είναι εξίσου λάθη.
2. Προαιρετικά, πρότεινε αντικαταστάσεις ΜΙΑΣ ΛΕΞΗΣ: αν μια λέξη της επιλεγμένης
   μεταγραφής είναι παραμορφωμένη μορφή ενός όρου της λίστας, δώσε
   {"from": "<η λέξη ακριβώς όπως εμφανίζεται>", "to": "<ο όρος ακριβώς όπως στη λίστα>"}.

ΑΠΑΡΑΒΑΤΟΙ ΚΑΝΟΝΕΣ:
- ΔΕΝ ξαναγράφεις τίποτα. Δεν προσθέτεις, δεν αφαιρείς, δεν αναδιατάσσεις λέξεις.
- Η ΜΟΝΗ επιτρεπτή αλλαγή είναι μία λέξη -> ένας όρος της λίστας, γραμμένος ακριβώς
  όπως στη λίστα. Οτιδήποτε άλλο θα απορριφθεί μηχανικά.
- Η λίστα δίνει ΕΠΙΤΡΕΠΤΕΣ ΓΡΑΦΕΣ. Το ότι ένας όρος είναι στη λίστα ΔΕΝ σημαίνει
  ότι ειπώθηκε σε αυτό το απόσπασμα.
- ΜΗΝ αγγίξεις λέξη που εμφανίζεται ΙΔΙΑ και στις τρεις μεταγραφές.
- Το "from" πρέπει να εμφανίζεται ΑΚΡΙΒΩΣ ΜΙΑ ΦΟΡΑ στην επιλεγμένη μεταγραφή. Αν
  εμφανίζεται περισσότερες, μην την προτείνεις.
- Σε αμφιβολία, μην προτείνεις τίποτα. Καμία αντικατάσταση είναι αποδεκτή απάντηση.
- Μην αλλάξεις στίξη ή πεζά/κεφαλαία.

Επίστρεψε ΜΟΝΟ ένα JSON array, ένα αντικείμενο ανά στοιχείο:
{"id": <το id>, "pick": "A"|"B"|"C", "replacements": [{"from": "...", "to": "..."}]}

ΣΤΟΙΧΕΙΑ:
"""


def parse_json_array(text: str):
    m = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    blob = m[-1] if m else None
    if blob is None:
        i, j = text.find("["), text.rfind("]")
        if i < 0 or j < 0:
            return []
        blob = text[i:j + 1]
    try:
        return json.loads(blob)
    except Exception:
        return []


def call_llm(batch: list[dict], timeout_wait: int = 900):
    payload = json.dumps(batch, ensure_ascii=False)
    p = subprocess.run(
        [sys.executable, CLIENT, "enqueue", "exec",
         "-c", f"model={LLM_MODEL}", "-c", "model_reasoning_effort=low",
         PROMPT + payload],
        capture_output=True, text=True, timeout=180)
    try:
        job = json.loads(p.stdout)["job_id"]
    except Exception:
        raise RuntimeError(f"enqueue failed: {p.stdout[-300:]} {p.stderr[-300:]}")
    w = subprocess.run([sys.executable, CLIENT, "wait", job],
                       capture_output=True, text=True, timeout=timeout_wait + 300)
    res = json.loads(w.stdout)
    if res.get("status") != "completed":
        raise RuntimeError(f"job {job}: {res.get('status')}")
    return parse_json_array(res.get("output") or "")


def llm_pass(items, ctxs, visible, batch_size=4):
    """One cached LLM decision per window. Returns wid -> raw decision."""
    LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if LLM_CACHE.exists():
        for line in LLM_CACHE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["id"]] = r
    todo = [it for it in items if it["item_id"] not in cache]
    log(f"LLM: {len(cache)} cached, {len(todo)} to call")
    chunks = [todo[k:k + batch_size] for k in range(0, len(todo), batch_size)]

    def build(chunk):
        out = []
        for it in chunk:
            terms = sorted({info["term"]["canonical"]
                            for info in ctxs[it["item_id"]].present.values()}
                           | set(visible.get(it["item_id"], [])))
            out.append({"id": it["item_id"],
                        "A": it["hyp"][TRIO[0]], "B": it["hyp"][TRIO[1]],
                        "C": it["hyp"][TRIO[2]], "terms": terms})
        return out

    def run(chunk):
        try:
            return chunk, call_llm(build(chunk))
        except Exception as e:  # a failed batch is a no-op, recorded, never retried
            return chunk, e

    lock = threading.Lock()
    done = 0
    with open(LLM_CACHE, "a") as f, ThreadPoolExecutor(max_workers=3) as pool:
        for chunk, got in pool.map(run, chunks):
            if isinstance(got, Exception):
                log(f"  batch failed: {got}")
                got = []
            by_id = {g.get("id"): g for g in got if isinstance(g, dict)}
            with lock:
                for it in chunk:
                    r = by_id.get(it["item_id"]) or {"id": it["item_id"],
                                                     "pick": None,
                                                     "replacements": [],
                                                     "missing": True}
                    r["id"] = it["item_id"]
                    cache[it["item_id"]] = r
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                done += len(chunk)
                log(f"  {done}/{len(todo)}")
    return cache


def apply_llm(item, decision, ctx, protected, fallback_pick):
    """Mechanically enforce every limit on the LLM's answer.

    The model's text output is never trusted: only its CHOICE among the three
    inputs, and replacements that survive the same closed-list + distance budget
    arm E uses."""
    acct = Counter()
    letter = (decision or {}).get("pick")
    idx = {"A": 0, "B": 1, "C": 2}.get(letter if isinstance(letter, str) else "")
    if idx is None:
        acct["pick_invalid"] += 1
        pick = fallback_pick
    else:
        pick = TRIO[idx]
        acct["pick_ok"] += 1
        if pick != fallback_pick:
            acct["pick_differs_from_vote"] += 1
    text = item["hyp"][pick]
    # what an accepted replacement is written as: the SAME stored surface form arm E
    # would have written, so L and E cannot disagree on spelling for the same term
    surfaces = {}
    for tid, info in ctx.present.items():
        for alias in info["term"]["aliases"]:
            an = rnorm(alias)
            surfaces[an] = ctx.alias_surface.get((tid, an), alias)
        surfaces.setdefault(rnorm(info["term"]["canonical"]),
                            info["term"]["canonical"])
    applied = []
    seen_src = set()
    for rep in (decision or {}).get("replacements") or []:
        if not isinstance(rep, dict):
            acct["rep_malformed"] += 1
            continue
        src, dst = str(rep.get("from", "")), str(rep.get("to", ""))
        sn, dn = rnorm(src), rnorm(dst)
        if not sn or not dn or len(wtoks(src)) != 1 or len(wtoks(dst)) != 1:
            acct["rep_not_single_token"] += 1
            continue
        if sn in seen_src:
            acct["rep_duplicate_source"] += 1
            continue
        seen_src.add(sn)
        if dn not in surfaces:
            acct["rep_target_off_list"] += 1
            continue
        if sn in protected:
            acct["rep_on_agreed_token"] += 1
            continue
        occ = len(re.findall(rf"(?<!\w){re.escape(src)}(?!\w)", text))
        if occ == 0:
            acct["rep_source_absent"] += 1
            continue
        if occ > 1:
            # "replace exactly one word" is ambiguous when the word repeats
            acct["rep_source_ambiguous"] += 1
            continue
        d = dl(sn, dn)
        if d == 0:
            acct["rep_noop"] += 1
            continue
        if not rule_allows(len(sn), d):
            acct["rep_over_budget"] += 1
            continue
        text, n = re.subn(rf"(?<!\w){re.escape(src)}(?!\w)", surfaces[dn], text)
        acct["rep_applied"] += n
        applied.append({"from": src, "to": surfaces[dn], "dist": d})
    return pick, text, applied, acct


# ----------------------------------------------------------------------- main
def main():
    report = B.load_report(RUN_ID)
    providers = B.provider_ids(report)
    items = B.common_items(report, providers)
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())["holdout_windows"]}
    before = len(items)
    items = [it for it in items if it["item_id"] not in sealed]
    log(f"{RUN_ID}: {before} common -> {len(items)} after removing "
        f"{before - len(items)} sealed holdout windows")

    clusters = [it["cluster"] for it in items]

    # -------- lexicon --------
    city_terms = load_city_terms()
    rosters = load_rosters()
    mined_by_city, mined_acct = admitted_mined()
    log(f"mined slice: {mined_acct['counts']}")

    # common-word table, leave-one-city-out over benchmark references
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
    for it in items:
        city = it["city_id"]
        ctx, acct = build_meeting_context(
            city, it["meeting_id"], city_terms[city], mined_by_city.get(city, []),
            rosters, loo_freq.get(city, Counter()))
        ctxs[it["item_id"]] = ctx
        lex_acct[it["item_id"]] = acct
    log(f"windows with a roster: {sum(a['has_roster'] for a in lex_acct.values())}"
        f"/{len(items)}; median terms/meeting "
        f"{sorted(a['n_terms'] for a in lex_acct.values())[len(items) // 2]}")

    agreed = {it["item_id"]: agreed_tokens(it, TRIO) for it in items}

    # -------- arms --------
    per_arm: dict[str, list] = {}
    texts: dict[str, dict[str, str]] = {}

    for p in providers:
        per_arm[f"single:{p}"] = [sdi(it["ref"], it["hyp"][p]) for it in items]

    v_pick = {it["item_id"]: B.consensus_pick(it, TRIO) for it in items}
    texts["V"] = {it["item_id"]: it["hyp"][v_pick[it["item_id"]]] for it in items}

    e_changes, e_unrestricted_changes = [], 0
    texts["V+E"] = {}
    for it in items:
        wid = it["item_id"]
        t, ch = restricted_repair(texts["V"][wid], ctxs[wid], agreed[wid])
        texts["V+E"][wid] = t
        for c in ch:
            e_changes.append({"window": wid, "original": c["original"],
                              "replacement": c["replacement"], "term": c["term"],
                              "dist": c["dist"]})
        e_unrestricted_changes += len(repair(texts["V"][wid], ctxs[wid]).changes)

    llm_acct = Counter()
    llm_applied = []
    if os.environ.get("LLM") == "1":
        decisions = llm_pass(
            items, ctxs,
            {w: a["visible_only"] for w, a in lex_acct.items()},
            batch_size=int(os.environ.get("LLM_BATCH", "4")))
        texts["V+Lsel"], texts["V+L"], texts["V+L+E"] = {}, {}, {}
        for it in items:
            wid = it["item_id"]
            pick, t, applied, acct = apply_llm(
                it, decisions.get(wid), ctxs[wid], agreed[wid], v_pick[wid])
            llm_acct.update(acct)
            for a in applied:
                llm_applied.append({"window": wid, **a})
            texts["V+Lsel"][wid] = it["hyp"][pick]
            texts["V+L"][wid] = t
            t2, ch2 = restricted_repair(t, ctxs[wid], agreed[wid])
            texts["V+L+E"][wid] = t2
            llm_acct["e_on_top_changes"] += len(ch2)

    for arm, tx in texts.items():
        per_arm[arm] = [sdi(it["ref"], tx[it["item_id"]]) for it in items]

    for name, pool in (("oracle_trio", TRIO), ("oracle_all", providers)):
        per_arm[name] = [min((sdi(it["ref"], it["hyp"][p]) for p in pool),
                             key=lambda c: c[0] + c[1] + c[2]) for it in items]

    # -------- contrasts, gates --------
    def counts(arm, idx=None):
        if idx is None:
            return [(r[0] + r[1] + r[2], r[3]) for r in per_arm[arm]]
        return [(r[idx], r[3]) for r in per_arm[arm]]

    def contrast(a, b):
        out = {}
        for metric, idx in (("wer", None), ("sub_rate", 0), ("del_rate", 1),
                            ("ins_rate", 2)):
            out[metric] = cluster_bootstrap(counts(a, idx), counts(b, idx),
                                            clusters, n_boot=N_BOOT)
        out["head2head_wer"] = head2head(counts(a), counts(b))
        return out

    def loo(a, b):
        """Leave-one-window-out on the pooled WER delta."""
        ca, cb = counts(a), counts(b)
        full = (sum(x[0] for x in ca) - sum(x[0] for x in cb)) / sum(x[1] for x in ca)
        ds = []
        for k in range(len(ca)):
            e = (sum(x[0] for i, x in enumerate(ca) if i != k)
                 - sum(x[0] for i, x in enumerate(cb) if i != k))
            n = sum(x[1] for i, x in enumerate(ca) if i != k)
            ds.append(e / n)
        worst = max(range(len(ds)), key=lambda k: abs(ds[k] - full))
        return {"delta": full, "min": min(ds), "max": max(ds),
                "sign_flips": sum(1 for d in ds if (d > 0) != (full > 0)),
                "max_influence_window": items[worst]["item_id"],
                "delta_without_it": ds[worst]}

    arms = [a for a in ("V", "V+E", "V+Lsel", "V+L", "V+L+E") if a in per_arm]
    res = {
        "run_id": RUN_ID, "n_items": len(items),
        "n_meetings": len(set(clusters)),
        "trio": TRIO, "n_boot": N_BOOT,
        "scorer": "eval/controlled_eval/scoring.py (not the benchmark app's)",
        "lexicon": {
            "terms_dir_sha256": {c: sha256(TERMS_DIR / f"{c}.json")
                                 for c in city_terms},
            "rosters_windows": sum(a["has_roster"] for a in lex_acct.values()),
            "mined": mined_acct,
            "terms_per_meeting": {
                "min": min(a["n_terms"] for a in lex_acct.values()),
                "median": sorted(a["n_terms"] for a in lex_acct.values())[len(items) // 2],
                "max": max(a["n_terms"] for a in lex_acct.values())},
        },
        "arms": {k: rates(v) for k, v in per_arm.items()},
        "E": {"changes": len(e_changes), "windows_changed":
              len({c["window"] for c in e_changes}),
              "changes_unrestricted": e_unrestricted_changes,
              "detail": e_changes},
        "L": {"model": LLM_MODEL, "reasoning_effort": "low",
              "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
              "batch_size": int(os.environ.get("LLM_BATCH", "4")),
              "accounting": dict(llm_acct), "applied": llm_applied},
        "contrasts": {}, "loo": {}, "gates": {}, "oracle_recovery": {},
        "by_mining_disjoint": {},
    }

    v = res["arms"]["V"]
    o_trio = res["arms"]["oracle_trio"]["wer"]
    o_all = res["arms"]["oracle_all"]["wer"]
    for a in arms:
        if a == "V":
            continue
        res["contrasts"][f"{a} vs V"] = contrast(a, "V")
        res["loo"][f"{a} vs V"] = loo(a, "V")
        m = res["arms"][a]
        res["gates"][a] = {
            "ins_rate_gate": {"V": v["ins_rate"], "arm": m["ins_rate"],
                              "pass": m["ins_rate"] <= v["ins_rate"]},
            "del_rate_gate": {"V": v["del_rate"], "arm": m["del_rate"],
                              "pass": m["del_rate"] <= v["del_rate"]},
            "wer_ci_excludes_zero":
                res["contrasts"][f"{a} vs V"]["wer"]["excludes_zero"],
            "loo_sign_stable": res["loo"][f"{a} vs V"]["sign_flips"] == 0,
        }
        res["oracle_recovery"][a] = {
            "vs_oracle_trio": ((v["wer"] - m["wer"]) / (v["wer"] - o_trio)
                               if v["wer"] != o_trio else None),
            "vs_oracle_all": ((v["wer"] - m["wer"]) / (v["wer"] - o_all)
                              if v["wer"] != o_all else None),
        }

    # ---- the preregistered LLM-justification test, against the free control ----
    if "V+L+E" in per_arm:
        e = res["arms"]["V+E"]
        gap_left = e["wer"] - o_trio
        c = contrast("V+E", "V+L+E")   # positive delta = V+E worse = LLM helps
        res["contrasts"]["V+E vs V+L+E"] = c
        res["loo"]["V+E vs V+L+E"] = loo("V+E", "V+L+E")
        thr = 0.10 * gap_left
        res["llm_justification"] = {
            "wer_V+E": e["wer"], "wer_V+L+E": res["arms"]["V+L+E"]["wer"],
            "improvement": e["wer"] - res["arms"]["V+L+E"]["wer"],
            "oracle_gap_left_after_E": gap_left,
            "threshold_10pct_of_gap": thr,
            "meets_threshold": (e["wer"] - res["arms"]["V+L+E"]["wer"]) >= thr,
            "ci_excludes_zero": c["wer"]["excludes_zero"],
            "ci95": c["wer"]["ci95"],
        }
        # selection vs replacement, the ablation Codex asked for
        res["contrasts"]["V+Lsel vs V"] = contrast("V+Lsel", "V")
        res["contrasts"]["V+L vs V+Lsel"] = contrast("V+L", "V+Lsel")

    # ---- mechanism + falsification checks (Codex job ee20cd08) ----
    if "V+Lsel" in per_arm:
        sel = {it["item_id"]: TRIO[{"A": 0, "B": 1, "C": 2}[decisions[it["item_id"]]["pick"]]]
               if (decisions.get(it["item_id"]) or {}).get("pick") in ("A", "B", "C")
               else v_pick[it["item_id"]] for it in items}
        oracle_best = {}
        for it in items:
            oracle_best[it["item_id"]] = min(
                TRIO, key=lambda p: sum(sdi(it["ref"], it["hyp"][p])[:3]))
        trans: dict = {}
        n_short = n_dev = 0
        for i, it in enumerate(items):
            wid = it["item_id"]
            v, l = v_pick[wid], sel[wid]
            if v == l:
                continue
            n_dev += 1
            sv = per_arm["V"][i]
            sl = per_arm["V+Lsel"][i]
            k = f"{v} -> {l}"
            a = trans.setdefault(k, {"n": 0, "d_err": 0, "d_del": 0, "d_sub": 0,
                                     "d_ins": 0, "d_words": 0})
            a["n"] += 1
            a["d_err"] += sum(sl[:3]) - sum(sv[:3])
            a["d_sub"] += sl[0] - sv[0]
            a["d_del"] += sl[1] - sv[1]
            a["d_ins"] += sl[2] - sv[2]
            dw = len(wtoks(it["hyp"][l])) - len(wtoks(it["hyp"][v]))
            a["d_words"] += dw
            if dw < 0:
                n_short += 1

        def loo_by(keyf):
            """Leave-one-<group>-out on the V+Lsel - V pooled WER delta."""
            groups: dict = {}
            for i, it in enumerate(items):
                groups.setdefault(keyf(it), []).append(i)
            err = lambda arm, idx: sum(sum(per_arm[arm][i][:3]) for i in idx)
            ref = lambda idx: sum(per_arm["V"][i][3] for i in idx)
            allidx = list(range(len(items)))
            full = (err("V+Lsel", allidx) - err("V", allidx)) / ref(allidx)
            ds = []
            for k, idx in groups.items():
                keep = [i for i in allidx if i not in set(idx)]
                ds.append(((err("V+Lsel", keep) - err("V", keep)) / ref(keep), k))
            ds.sort()
            return {"full": full, "min": ds[0][0], "min_group": ds[0][1],
                    "max": ds[-1][0], "max_group": ds[-1][1],
                    "sign_flips": sum(1 for d, _ in ds if (d > 0) != (full > 0))}

        dmg = sorted((sum(per_arm["V+Lsel"][i][:3]) - sum(per_arm["V"][i][:3]), i)
                     for i in range(len(items)))
        pos = sum(c for c, _ in dmg if c > 0)
        res["mechanism"] = {
            "vote_picks": dict(Counter(v_pick.values())),
            "llm_picks": dict(Counter(sel.values())),
            "n_deviations": n_dev,
            "llm_picked_shorter_in": n_short,
            "transitions": trans,
            "vote_equals_oracle_best": sum(1 for w in sel if v_pick[w] == oracle_best[w]),
            "llm_equals_oracle_best": sum(1 for w in sel if sel[w] == oracle_best[w]),
            "loo_meeting": loo_by(lambda it: it["cluster"]),
            "loo_city": loo_by(lambda it: it["city_id"]),
            "largest_window_share_of_damage": dmg[-1][0] / pos if pos else None,
            "largest_window": items[dmg[-1][1]]["item_id"],
            "llm_replacements_proposed": sum(
                len((decisions.get(it["item_id"]) or {}).get("replacements") or [])
                for it in items),
        }

    # ---- windows disjoint from the mining fold, reported separately ----
    fold = json.loads((ROOT / "data/glossary/glossary.build-manifest.json").read_text())
    mined_pairs = {(p["city_id"], p["meeting_id"])
                   for p in fold["source_meetings"]["pairs"]}
    if mined_pairs:
        dis = [i for i, it in enumerate(items)
               if (it["city_id"], it["meeting_id"]) not in mined_pairs
               and not it["in_training"]]
        res["by_mining_disjoint"] = {
            "n_windows": len(dis),
            "arms": {a: rates([per_arm[a][i] for i in dis]) for a in arms},
            "note": "exploratory, underpowered; disjoint from BOTH the glossary "
                    "mining fold and the fine-tune training manifest",
        }

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")
    for k, val in res["arms"].items():
        log(f"  {k:34s} wer={val['wer']:.4f}  del={val['del_rate']:.4f}  "
            f"ins={val['ins_rate']:.4f}  sub={val['sub_rate']:.4f}")
    for k, val in res["contrasts"].items():
        log(f"  {k}: dWER {val['wer']['delta']:+.5f} {val['wer']['ci95']}")


if __name__ == "__main__":
    main()
