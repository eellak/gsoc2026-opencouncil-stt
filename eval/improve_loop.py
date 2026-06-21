"""Autoresearch-style improvement loop for the fix-task prompt.

Mirrors karpathy/autoresearch: one editable artifact (here, the **system
prompt**), a single metric (**net fix** = edit_application - overcorrection on a
dev subset), and an LLM-researcher that reads the results and proposes the next
mutation. Keep the best, iterate.

Flow:
  1. build-data : carve disjoint dev/test subsets from the held-out sample,
     balanced over the substantive categories (formatting cats excluded — their
     fix is rule-based, not prompt-driven).
  2. run        : evaluate seed variants, then R rounds of
     (LLM proposes a new system prompt -> evaluate on dev -> keep best).
     Checkpointed leaderboard; resumable by prompt hash.
  3. validate   : re-score the best variant vs baseline on the held-out TEST set.

Cost: evaluation uses a cheap backend (--eval-backend, default haiku); proposals
use a stronger one (--propose-backend, default claude sonnet). Each variant costs
len(dev) fix-calls (one arm — the variant's own prompt).

Usage:
  python -m eval.improve_loop build-data [--dev-per-cat 10] [--test-per-cat 10]
  python -m eval.improve_loop run   [--rounds 4] [--eval-backend haiku] [--workers 4]
  python -m eval.improve_loop validate [--eval-backend claude]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.backends import generate
from eval.fix_call import parse_numbered
from eval.glossary import prepare_retrieval_pool, render_terms_block, select_glossary_terms
from eval.prompts import SYSTEM_PROMPT as BASELINE_PROMPT
from eval.prompts import build_user_prompt
from eval.scoring import score_pair

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
LOOP = ROOT / "data" / "improve_loop"
VARIANTS = LOOP / "variants"
LEADERBOARD = LOOP / "leaderboard.jsonl"
LOG = ROOT / "eval" / "loop.log"

# substantive categories where prompt wording actually moves the needle
DEV_CATS = ["named_entity", "acronym_abbreviation", "homophone", "number_date",
            "morph_grammar", "word_boundary", "other_lexical"]

_lock = threading.Lock()
_pools: dict[str, dict] = {}
_glossary: dict = {}


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with _lock:
        LOG.open("a", encoding="utf-8").write(line + "\n")


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _get_pool(city: str) -> dict:
    with _lock:
        if city not in _pools:
            _pools[city] = prepare_retrieval_pool(_glossary, city)
        return _pools[city]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_data(dev_per_cat: int, test_per_cat: int) -> None:
    import pandas as pd

    from eval.categorize import categorize

    LOOP.mkdir(parents=True, exist_ok=True)
    cdf = pd.read_parquet(EVAL / "chains.parquet")
    split = json.loads((EVAL / "split.json").read_text())
    ev = cdf[cdf["meeting_id"].isin(split["eval_meeting_ids"])].copy()
    ev = ev[ev.chain_type == "pure_correction"].copy()
    ev["category"] = [categorize(b, a) for b, a in zip(ev.input_raw, ev.gold_final)]
    ev = ev[ev.category.isin(DEV_CATS)]

    dev_rows, test_rows = [], []
    for cat, grp in ev.groupby("category"):
        grp = grp.sample(frac=1.0, random_state=21).reset_index(drop=True)
        dev = grp.iloc[:dev_per_cat]
        test = grp.iloc[dev_per_cat:dev_per_cat + test_per_cat]
        dev_rows.append(dev)
        test_rows.append(test)

    cols = ["utterance_id", "city_id", "meeting_id", "input_raw", "gold_final", "category"]
    for name, parts in [("dev", dev_rows), ("test", test_rows)]:
        df = pd.concat(parts)[cols]
        path = LOOP / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps({c: r[c] for c in cols}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(df)} rows -> {path}")


def _load_rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (LOOP / f"{name}.jsonl").read_text().splitlines() if l.strip()]


def strat_split(rows: list[dict], frac_propose: float = 0.6) -> tuple[list[dict], list[dict]]:
    """Split rows into (propose, select), stratified by category.

    Deterministic and order-independent: within each category, rows are ordered
    by a hash of their utterance_id, then the first `frac_propose` go to the
    propose half. This guarantees the proposer is never scored on a row it was
    shown — the core anti-overfit safeguard (the select half stays blind to the
    proposer)."""
    from collections import defaultdict

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    propose, select = [], []
    for cat in sorted(by_cat):
        rs = sorted(by_cat[cat], key=lambda r: _hash(str(r["utterance_id"])))
        k = round(len(rs) * frac_propose)
        propose.extend(rs[:k])
        select.extend(rs[k:])
    return propose, select


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _score_row(variant: dict, row: dict, backend: str, model: str | None) -> dict:
    city, inp, gold = row["city_id"], row["input_raw"], row["gold_final"]
    block = ""
    if variant.get("use_glossary"):
        terms = select_glossary_terms(_get_pool(city), inp)
        block = render_terms_block(terms)
    up = build_user_prompt(city, [inp], glossary_block=block)
    raw = generate(variant["system_prompt"], up, backend=backend, model=model)
    parsed = parse_numbered(raw, 1)
    out = parsed[0] if parsed else raw.strip()
    s = score_pair(inp, out, gold)
    s["category"] = row["category"]
    s["net"] = s["edit_application"] - s["overcorrection"]
    s["input_raw"] = inp
    s["gold_final"] = gold
    s["output"] = out
    return s


def evaluate_variant(variant: dict, rows: list[dict], backend: str, model: str | None,
                     workers: int) -> dict:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_score_row, variant, r, backend, model) for r in rows]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                _log(f"  row error: {str(e)[:100]}")
    if not results:
        return {"n": 0, "mean_net": -1.0, "fails": []}

    def mean(key):
        xs = [r[key] for r in results]
        return sum(xs) / len(xs)

    per_cat = {}
    for cat in DEV_CATS:
        cr = [r for r in results if r["category"] == cat]
        if cr:
            per_cat[cat] = round(sum(r["net"] for r in cr) / len(cr), 3)
    # failure examples captured during evaluation (no extra calls)
    fails = [{"category": r["category"], "input_raw": r["input_raw"],
              "gold_final": r["gold_final"], "output": r["output"]}
             for r in results if not r["normalized_exact"]][:8]
    # HIR = human intervention rate: fraction still needing a human edit
    hir = 1.0 - mean("normalized_exact")
    return {
        "n": len(results),
        "hir": round(hir, 4),                     # PRIMARY product metric (lower better)
        "mean_wer": round(mean("wer"), 4),        # secondary error-volume metric
        "mean_net": round(mean("net"), 4),        # search surrogate / diagnostic
        "mean_edit": round(mean("edit_application"), 4),
        "mean_over": round(mean("overcorrection"), 4),
        "mean_exact": round(mean("normalized_exact"), 4),
        "per_cat": per_cat,
        "fails": fails,
    }


# --------------------------------------------------------------------------- #
# seed variants (grounded in the baseline A/B findings)
# --------------------------------------------------------------------------- #
def seed_variants() -> list[dict]:
    base = BASELINE_PROMPT
    glossary_note = (
        "\n\nYou may also be given a list of known correct spellings of local "
        "names/terms. Use it ONLY to fix a word that is clearly the same name "
        "misspelled — never force an unrelated word to match the list."
    )
    anti_over = (
        "\n\nIMPORTANT: change as few words as possible. If a word is already a "
        "valid Greek word and plausibly what the speaker said, leave it. Do not "
        "rephrase, add, or remove words. Most lines need 0–1 changes."
    )
    acronym_rule = (
        "\n\n5. ACRONYMS & legal references — write known acronyms in capitals "
        "(ΔΕΥΑ, ΚΕΔΕ, ΦΕΚ), and legal references in standard form (Ν. 4412/2016, "
        "άρθρο 75 παρ. 2). Only when you are confident of the expansion."
    )
    return [
        {"id": "v0_baseline", "parent": None, "description": "verbatim task-v2, no glossary",
         "system_prompt": base, "use_glossary": False},
        {"id": "v1_glossary", "parent": "v0_baseline",
         "description": "baseline + selective glossary block",
         "system_prompt": base + glossary_note, "use_glossary": True},
        {"id": "v2_antiovercorrect", "parent": "v0_baseline",
         "description": "baseline + strong anti-overcorrection guardrail",
         "system_prompt": base + anti_over, "use_glossary": False},
        {"id": "v3_acronym", "parent": "v0_baseline",
         "description": "baseline + explicit acronym/legal rule",
         "system_prompt": base + acronym_rule, "use_glossary": False},
        {"id": "v4_combo", "parent": "v0_baseline",
         "description": "baseline + glossary + anti-overcorrection",
         "system_prompt": base + glossary_note + anti_over, "use_glossary": True},
    ]


# --------------------------------------------------------------------------- #
# LLM researcher: propose the next system prompt
# --------------------------------------------------------------------------- #
PROPOSER_SYSTEM = (
    "You are a prompt engineer optimising a system prompt that corrects Greek "
    "city-council ASR transcripts. You are given the current best prompt, its "
    "scores (net = fraction of correct edits applied minus overcorrection harm; "
    "higher is better), per-category net scores, and failure examples. Propose a "
    "REVISED system prompt that should raise net. Keep the numbered-line "
    "input/output contract intact. Output ONLY the new system prompt, wrapped "
    "exactly as <PROMPT>...</PROMPT> with no other text."
)


# Diversity foci for the frozen-pool proposer: each candidate is nudged toward a
# distinct failure mode so the pool spans the space instead of clustering.
PROPOSE_FOCI = [
    "Focus on named entities, parties, and organisation/legal acronyms.",
    "Focus on homophone and ο/ω-η/ι-αι/ε sound-confusion spelling errors.",
    "Focus on morphology, agreement, and verb voice/person — without over-rewriting.",
    "Focus on numbers, dates, ordinals, and house-style digit conventions.",
    "Focus on being maximally conservative: fix only near-certain errors, never paraphrase.",
    "Focus on word boundaries (joined/split words) and punctuation/accents.",
    "Focus on the output contract: exact line count, no truncation completion, no commentary.",
    "Focus on balancing recall (catching real errors) against overcorrection harm.",
]


def propose_variant(best: dict, best_entry: dict, fails: list[dict],
                    backend: str, model: str | None, idx: int,
                    focus: str | None = None) -> dict | None:
    ex_txt = "\n".join(
        f"- [{f['category']}] input: {f['input_raw'][:90]!r} | gold: {f['gold_final'][:90]!r} | model: {f['output'][:90]!r}"
        for f in fails[:6]
    )
    focus_line = f"\nFor this revision: {focus}\n" if focus else ""
    user = (
        f"Current best net = {best_entry['mean_net']} "
        f"(edit {best_entry['mean_edit']}, overcorrection {best_entry['mean_over']}).\n"
        f"Per-category net: {json.dumps(best_entry.get('per_cat', {}), ensure_ascii=False)}\n\n"
        f"Failure examples (model output != gold):\n{ex_txt}\n"
        f"{focus_line}\n"
        f"Current prompt:\n<PROMPT>\n{best['system_prompt']}\n</PROMPT>\n\n"
        "Return the improved prompt."
    )
    raw = generate(PROPOSER_SYSTEM, user, backend=backend, model=model, timeout=240)
    if "<PROMPT>" in raw and "</PROMPT>" in raw:
        new_prompt = raw.split("<PROMPT>", 1)[1].split("</PROMPT>", 1)[0].strip()
    else:
        new_prompt = raw.strip()
    if len(new_prompt) < 200:  # implausible / refusal
        return None
    return {
        "id": f"r{idx}_{_hash(new_prompt)}",
        "parent": best["id"],
        "description": f"LLM-proposed round {idx}",
        "system_prompt": new_prompt,
        "use_glossary": best.get("use_glossary", False),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _save_variant(v: dict) -> None:
    VARIANTS.mkdir(parents=True, exist_ok=True)
    (VARIANTS / f"{v['id']}.txt").write_text(v["system_prompt"], encoding="utf-8")


def _done_hashes() -> set[str]:
    done = set()
    if LEADERBOARD.exists():
        for l in LEADERBOARD.read_text().splitlines():
            if l.strip():
                try:
                    done.add(json.loads(l)["prompt_hash"])
                except Exception:
                    pass
    return done


def _append_leaderboard(entry: dict) -> None:
    with _lock:
        LEADERBOARD.open("a", encoding="utf-8").write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_loop(rounds, eval_backend, eval_model, propose_backend, propose_model, workers,
             candidates=1, min_improve=0.02) -> None:
    global _glossary
    _glossary = json.loads((ROOT / "data" / "glossary" / "glossary.json").read_text())
    LOOP.mkdir(parents=True, exist_ok=True)
    dev = _load_rows("dev")
    _log(f"loop start: dev={len(dev)} rows, eval_backend={eval_backend}, rounds={rounds}")

    done = _done_hashes()
    variants = {}
    fails_by_id: dict[str, list] = {}
    best_id, best_hir = None, 2.0  # HIR is the objective — lower is better

    # load existing leaderboard for resume / best tracking
    if LEADERBOARD.exists():
        for l in LEADERBOARD.read_text().splitlines():
            if not l.strip():
                continue
            e = json.loads(l)
            if e.get("hir", 2.0) < best_hir:
                best_hir, best_id = e["hir"], e["id"]

    def run_one(v):
        nonlocal best_id, best_hir
        h = _hash(v["system_prompt"] + str(v["use_glossary"]))
        if h in done:
            _log(f"  skip {v['id']} (already evaluated)")
            return
        _save_variant(v)
        m = evaluate_variant(v, dev, eval_backend, eval_model, workers)
        fails_by_id[v["id"]] = m.pop("fails", [])
        entry = {"id": v["id"], "parent": v["parent"], "description": v["description"],
                 "use_glossary": v["use_glossary"], "prompt_hash": h,
                 "backend": eval_backend, **m}
        _append_leaderboard(entry)
        done.add(h)
        variants[v["id"]] = v
        flag = ""
        if m["hir"] < best_hir:
            best_hir, best_id = m["hir"], v["id"]
            flag = "  <-- NEW BEST"
        _log(f"  {v['id']}: HIR={m['hir']} wer={m['mean_wer']} net={m['mean_net']} "
             f"exact={m['mean_exact']}{flag}")

    # phase 1: seeds
    _log("== seeds ==")
    seeds = seed_variants()
    for v in seeds:
        variants[v["id"]] = v
        run_one(v)

    # phase 2: LLM-proposed rounds (predeclared budget + stopping rule)
    stalls = 0
    for rnd in range(1, rounds + 1):
        prev_best = best_hir
        _log(f"== round {rnd}/{rounds} (propose from best={best_id} HIR={best_hir}) ==")
        best = variants.get(best_id)
        if best is None:  # best from a prior session; reload prompt
            p = VARIANTS / f"{best_id}.txt"
            best = ({"id": best_id, "system_prompt": p.read_text(), "use_glossary": False}
                    if p.exists() else seeds[0])
        best_entry = _best_entry()
        fails = fails_by_id.get(best_id, [])
        # generate `candidates` proposals this round, evaluate each, keep the best
        produced = 0
        for c in range(candidates):
            try:
                v = propose_variant(best, best_entry, fails, propose_backend,
                                    propose_model, rnd * 10 + c)
            except Exception as e:
                _log(f"  propose failed: {str(e)[:120]}")
                continue
            if v is None:
                continue
            run_one(v)
            produced += 1
        if produced == 0:
            _log("  no usable proposals; stopping")
            break
        # stopping rule: stop after consecutive rounds without >= min_improve HIR gain
        if prev_best - best_hir < min_improve:
            stalls += 1
            _log(f"  round gain {prev_best - best_hir:+.4f} < {min_improve}; stall {stalls}")
            if stalls >= 2:
                _log("  stopping: 2 consecutive rounds below min improvement")
                break
        else:
            stalls = 0

    _log(f"loop done. best={best_id} HIR={best_hir}")
    print(f"\nBEST: {best_id}  HIR={best_hir}")
    print(f"prompt saved at {VARIANTS / (str(best_id) + '.txt')}")


def run_pool(candidates, eval_backend, eval_model, propose_backend, propose_model,
             workers, propose_frac=0.6) -> None:
    """Anti-overfit search (Codex-reviewed): no iterative feedback.

    1. Split dev into propose/select (stratified, deterministic).
    2. The proposer sees failures ONLY from the propose half.
    3. Generate a FROZEN pool of candidates up front (seeds + `candidates`
       LLM proposals across diverse foci), then rank the whole pool ONCE on the
       select half — which the proposer never saw. No round feeds the next, so
       the select set is not adaptively reused (Dwork et al. holdout reuse).
    4. Winner = lowest select-half HIR, ties toward the simpler/incumbent prompt.
    Every attempted prompt is logged so the effective search size is auditable.
    """
    global _glossary
    _glossary = json.loads((ROOT / "data" / "glossary" / "glossary.json").read_text())
    LOOP.mkdir(parents=True, exist_ok=True)
    dev = _load_rows("dev")
    propose_rows, select_rows = strat_split(dev, propose_frac)
    _log(f"pool start: dev={len(dev)} (propose={len(propose_rows)}, select={len(select_rows)}), "
         f"budget={candidates} LLM candidates, eval_backend={eval_backend}/{eval_model}")

    done = _done_hashes()
    seeds = seed_variants()
    base_v = seeds[0]

    # baseline failures on the PROPOSE half — the only signal the proposer sees
    _log("== baseline pass on propose half (for proposer signal) ==")
    base_propose = evaluate_variant(base_v, propose_rows, eval_backend, eval_model, workers)
    base_entry = {"mean_net": base_propose["mean_net"], "mean_edit": base_propose["mean_edit"],
                  "mean_over": base_propose["mean_over"], "per_cat": base_propose["per_cat"]}
    fails = base_propose["fails"]

    # build the frozen candidate pool
    pool = list(seeds)
    _log(f"== proposing {candidates} candidates (blind to select half) ==")
    for i in range(candidates):
        try:
            v = propose_variant(base_v, base_entry, fails, propose_backend, propose_model,
                                idx=i, focus=PROPOSE_FOCI[i % len(PROPOSE_FOCI)])
        except Exception as e:
            _log(f"  propose {i} failed: {str(e)[:120]}")
            continue
        if v is not None:
            pool.append(v)
    _log(f"frozen pool size = {len(pool)} ({len(seeds)} seeds + {len(pool) - len(seeds)} proposed)")

    # rank the ENTIRE frozen pool once, on the select half
    _log("== ranking frozen pool on select half ==")
    for v in pool:
        h = _hash(v["system_prompt"] + str(v["use_glossary"]))
        if h in done:
            _log(f"  skip {v['id']} (already scored)")
            continue
        _save_variant(v)
        m = evaluate_variant(v, select_rows, eval_backend, eval_model, workers)
        m.pop("fails", None)
        entry = {"id": v["id"], "parent": v["parent"], "description": v["description"],
                 "use_glossary": v["use_glossary"], "prompt_hash": h,
                 "backend": eval_backend, "split": "select",
                 "prompt_len": len(v["system_prompt"]), **m}
        _append_leaderboard(entry)
        done.add(h)
        _log(f"  {v['id']}: HIR={m['hir']} wer={m['mean_wer']} over={m['mean_over']} "
             f"(len={len(v['system_prompt'])})")

    be = _best_entry()
    # paired CI of the winner vs baseline on the select half (same rows)
    best_prompt = (VARIANTS / f"{be['id']}.txt").read_text()
    best_v = {"id": be["id"], "system_prompt": best_prompt,
              "use_glossary": be.get("use_glossary", False)}
    base_sel = _per_row(base_v, select_rows, eval_backend, eval_model, workers)
    best_sel = _per_row(best_v, select_rows, eval_backend, eval_model, workers)
    uids = [u for u in base_sel if u in best_sel]
    pairs = [(base_sel[u]["needs_edit"], best_sel[u]["needs_edit"]) for u in uids]
    n = len(uids)
    hir_base = sum(1 for b, _ in pairs if b) / n if n else 0.0
    hir_best = sum(1 for _, w in pairs if w) / n if n else 0.0
    ci = _bootstrap_ci(pairs, lambda ps: (sum(b for b, _ in ps) - sum(w for _, w in ps)) / len(ps))
    _log(f"pool done. WINNER={be['id']} select-HIR baseline {hir_base:.3f} -> {hir_best:.3f} "
         f"({(hir_base-hir_best)*100:+.1f}pp), 95% CI {ci}")
    print(f"\nWINNER: {be['id']}  select-HIR {hir_base:.3f} -> {hir_best:.3f}  CI95 {ci}")
    print(f"Run: python -m eval.improve_loop validate --eval-backend {eval_backend} "
          f"--eval-model {eval_model} --workers {workers}")


def _entry_rank(e: dict) -> tuple:
    """Deterministic ordering key: lowest HIR wins; ties break toward the
    shorter (simpler) prompt, then v0_baseline, then id — favouring the
    incumbent over a marginally-equal challenger (winner's-curse guard)."""
    return (e.get("hir", 2.0), e.get("prompt_len", 10 ** 9),
            0 if e.get("id") == "v0_baseline" else 1, str(e.get("id")))


def _best_entry() -> dict:
    entries = [json.loads(l) for l in LEADERBOARD.read_text().splitlines() if l.strip()]
    return min(entries, key=_entry_rank) if entries else {"hir": 2.0}


def _per_row(variant, rows, backend, model, workers) -> dict:
    """Return {uid: {needs_edit: bool, wer: float}} for paired stats."""
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_score_row, variant, r, backend, model): r for r in rows}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                s = fut.result()
                out[r["utterance_id"]] = {
                    "needs_edit": not s["normalized_exact"], "wer": s["wer"]}
            except Exception as e:
                _log(f"  val row error: {str(e)[:90]}")
    return out


def _bootstrap_ci(pairs, stat, n=2000, seed_terms=None):
    """Paired bootstrap 95% CI for a per-row statistic over (base,best) pairs.

    Deterministic resampling via an LCG seeded from the data length (no RNG —
    keeps runs reproducible without Math.random-style nondeterminism)."""
    m = len(pairs)
    if m == 0:
        return (0.0, 0.0)
    state = (m * 2654435761) & 0xFFFFFFFF
    samples = []
    for _ in range(n):
        idx = []
        for _ in range(m):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            idx.append(state % m)
        samples.append(stat([pairs[i] for i in idx]))
    samples.sort()
    lo = samples[int(0.025 * n)]
    hi = samples[int(0.975 * n)]
    return (round(lo, 4), round(hi, 4))


def validate(eval_backend, eval_model, workers) -> None:
    global _glossary
    _glossary = json.loads((ROOT / "data" / "glossary" / "glossary.json").read_text())
    test = _load_rows("test")
    be = _best_entry()
    best_prompt = (VARIANTS / f"{be['id']}.txt").read_text()
    best_v = {"id": be["id"], "system_prompt": best_prompt,
              "use_glossary": be.get("use_glossary", False)}
    base_v = seed_variants()[0]
    _log(f"validate on test={len(test)} (USED ONCE): baseline vs {be['id']} "
         f"(backend={eval_backend}/{eval_model})")

    base = _per_row(base_v, test, eval_backend, eval_model, workers)
    best = _per_row(best_v, test, eval_backend, eval_model, workers)
    uids = [u for u in base if u in best]

    pairs = [(base[u]["needs_edit"], best[u]["needs_edit"]) for u in uids]
    wer_pairs = [(base[u]["wer"], best[u]["wer"]) for u in uids]
    n = len(uids)
    hir_base = sum(1 for b, _ in pairs if b) / n
    hir_best = sum(1 for _, w in pairs if w) / n
    improved = sum(1 for b, w in pairs if b and not w)   # base needed edit, best clean
    regressed = sum(1 for b, w in pairs if not b and w)  # base clean, best needs edit
    unchanged = n - improved - regressed
    rel = (hir_base - hir_best) / hir_base if hir_base else 0.0
    ci = _bootstrap_ci(pairs, lambda ps: (sum(b for b, _ in ps) - sum(w for _, w in ps)) / len(ps))
    # McNemar exact (two-sided) on discordant pairs
    b01, b10 = regressed, improved
    mcnemar_p = _mcnemar_exact(b10, b01)
    wer_base = sum(b for b, _ in wer_pairs) / n
    wer_best = sum(w for _, w in wer_pairs) / n

    out = {
        "test_n": n,
        "selection_backend": eval_backend, "selection_model": eval_model,
        "best_id": be["id"], "best_use_glossary": be.get("use_glossary", False),
        "HIR": {"baseline": round(hir_base, 4), "best": round(hir_best, 4),
                 "abs_reduction_pp": round((hir_base - hir_best) * 100, 2),
                 "relative_error_reduction": round(rel, 4),
                 "ci95_abs_diff": ci},
        "WER": {"baseline": round(wer_base, 4), "best": round(wer_best, 4),
                 "abs_reduction": round(wer_base - wer_best, 4)},
        "paired": {"improved": improved, "regressed": regressed, "unchanged": unchanged,
                    "mcnemar_p": mcnemar_p},
    }
    (LOOP / "validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    _log(f"validation: HIR {hir_base:.3f} -> {hir_best:.3f} "
         f"({(hir_base-hir_best)*100:+.1f}pp, rel {rel:+.1%}); "
         f"improved {improved} regressed {regressed}; McNemar p={mcnemar_p}")


def _mcnemar_exact(b10: int, b01: int) -> float:
    """Two-sided exact McNemar p-value via the binomial(n, 0.5) tail."""
    n = b10 + b01
    if n == 0:
        return 1.0
    from math import comb
    k = min(b10, b01)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    bd = sub.add_parser("build-data")
    bd.add_argument("--dev-per-cat", type=int, default=10)
    bd.add_argument("--test-per-cat", type=int, default=20)
    rn = sub.add_parser("run")
    rn.add_argument("--rounds", type=int, default=4)
    rn.add_argument("--candidates", type=int, default=1, help="proposals per round")
    rn.add_argument("--min-improve", type=float, default=0.02, help="min HIR gain/round")
    rn.add_argument("--eval-backend", default="claude")
    rn.add_argument("--eval-model", default="haiku")
    rn.add_argument("--propose-backend", default="claude")
    rn.add_argument("--propose-model", default="sonnet")
    rn.add_argument("--workers", type=int, default=4)
    rp = sub.add_parser("run-pool", help="anti-overfit frozen-pool search (no iterative feedback)")
    rp.add_argument("--candidates", type=int, default=8, help="LLM proposals in the frozen pool")
    rp.add_argument("--propose-frac", type=float, default=0.6, help="fraction of dev shown to proposer")
    rp.add_argument("--eval-backend", default="codex")
    rp.add_argument("--eval-model", default="gpt-5.5")
    rp.add_argument("--propose-backend", default="codex")
    rp.add_argument("--propose-model", default="gpt-5.5")
    rp.add_argument("--workers", type=int, default=4)
    va = sub.add_parser("validate")
    va.add_argument("--eval-backend", default="claude")
    va.add_argument("--eval-model", default="sonnet")
    va.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.cmd == "build-data":
        build_data(args.dev_per_cat, args.test_per_cat)
    elif args.cmd == "run":
        run_loop(args.rounds, args.eval_backend, args.eval_model,
                 args.propose_backend, args.propose_model, args.workers,
                 candidates=args.candidates, min_improve=args.min_improve)
    elif args.cmd == "run-pool":
        run_pool(args.candidates, args.eval_backend, args.eval_model,
                 args.propose_backend, args.propose_model, args.workers,
                 propose_frac=args.propose_frac)
    elif args.cmd == "validate":
        validate(args.eval_backend, args.eval_model, args.workers)


if __name__ == "__main__":
    main()
