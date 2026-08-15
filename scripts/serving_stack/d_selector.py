#!/usr/bin/env python3
"""Arm D of the serving-stack plan: the PREREG N-BEST SELECTOR.

Spec: docs/specs/2026-08-12-serving-stack-plan.md, arm D, "Prereg selector
2026-08-12" - frozen after the oracle screen (ceiling 0.0071, 85 errors) and
before any selector scoring. Implemented EXACTLY:

  argmax_h [(A(h) - A(top1))/s_A + lambda * (L(h) - L(top1))/s_L]

over DISTINCT hypotheses per decode chunk (distinct after the frozen
normalization `ftoks`; each distinct group keeps its best original beam rank
and that hypothesis' raw text and score). A(h) is the CT2 score, which CT2
already normalizes per token (length_penalty=1: score = cum_logprob/len), so
lambda=0 reproduces beam8-top1 exactly - asserted on all 39 windows before
anything else. L(h) is the per-word LM logprob of the ftoks token sequence
with BOS/EOS (total log10 prob / (n_words + 1), the EOS counted as a scored
word). lambda comes from the frozen grid {0, 0.25, 0.5, 1, 2, 4}; ties go to
the smaller lambda and then to the lower beam rank; an alternative is chosen
only when its combined score is STRICTLY positive. No other feature.

Cross-fitting: leave-one-meeting-out over the 31 meetings (fold identity =
`meeting_id`, the frozen resampling block of the eval freeze - note the block
convention pools the argos and orestiada meetings that share the id
`apr7_2026`, which is why 32 (city, meeting) pairs are 31 blocks). Per fold:
s_A and s_L are the RMS of the (candidate - top1) differences over every
candidate of every non-skipped chunk of the 30 training meetings; lambda
minimizes the pooled window-level S+D+I of those training meetings (frozen
`sdi` on the pipeline-join reconstruction, same scoring as the baseline).
The fitted (s_A, s_L, lambda) apply ONCE to the held-out meeting. Only the
concatenated out-of-fold selections are reported.

Gate (frozen): recovery >= 25% of the oracle ceiling AND the one-sided 95%
upper bound on delta-WER (selector - beam8-top1) from a meeting-clustered
paired bootstrap (10000 resamples, seed 7, NO refitting inside replicates)
< +0.0010.

LM: word 4-gram over SEEN-CITY text only - the referenceText of the 220
seen-city items of the 2026-08-10 full benchmark report (argos and orestiada
rows are excluded and asserted absent; those are the only cities in the 39
eval windows, so the LM never sees eval material). Tokenization = the frozen
`ftoks`, so LM tokens match scored tokens. A frozen manifest (files, sha256,
row counts) is written BEFORE any selector scoring.

DEVIATION (documented prominently, per the fallback rule): KenLM *training*
is unavailable on this box - no lmplz on PATH, no cmake and no boost headers
to build one, and the kenlm 0.3.0 wheel that `pip install kenlm` produces is
query-only. NLTK is likewise absent and may not be installed (network is
restricted to the kenlm install). The 4-gram is therefore estimated by the
pure-python INTERPOLATED KNESER-NEY estimator below (continuation counts,
one Ney discount D_n = n1/(n1+2*n2) per order) - NOT lmplz's modified
Kneser-Ney with three per-order discounts. The model is written as a
standard ARPA file and QUERIED THROUGH KENLM (kenlm.Model), so L(h) is a
per-word KenLM logprob as preregistered; only the estimator deviates.

    SC=~/.cache/oc-public .venv-eval/bin/python scripts/serving_stack/d_selector.py build-lm
    SC=~/.cache/oc-public .venv-eval/bin/python scripts/serving_stack/d_selector.py score

Outputs: $SC/serving-stack/d-selector-lm-corpus.txt and d-selector-lm.arpa
(transcript-derived - never in the repo);
data/reports/finetune-research/d-selector-lm-manifest-2026-08-12.json (frozen
before scoring) and d-selector-eval-2026-08-12.json (no transcript text).
ONE scoring pass - no tuning after the out-of-fold number is seen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from scripts.serving_stack import nbest_arm  # noqa: E402

DA = nbest_arm.DA

ARM = "D-selector"
ORDER = 4
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
N_BOOT, BOOT_SEED = 10000, 7
GATE_RECOVERY = 0.25
GATE_UPPER = 0.0010

REPORT_FULL = ROOT / ("data/reports/finetune-research/"
                      "2026-08-10-corrected-adapter-report-full.json")
EVAL_CITIES = ("argos", "orestiada")   # unseen; FORBIDDEN in the LM corpus
LM_MANIFEST = ROOT / ("data/reports/finetune-research/"
                      "d-selector-lm-manifest-2026-08-12.json")
RESULTS_REPO = ROOT / ("data/reports/finetune-research/"
                       "d-selector-eval-2026-08-12.json")


def log(m):
    print(m, flush=True)


def out_dir() -> Path:
    d = DA.sc() / "serving-stack"
    d.mkdir(parents=True, exist_ok=True)
    return d


def corpus_path() -> Path:
    return out_dir() / "d-selector-lm-corpus.txt"


def arpa_path() -> Path:
    return out_dir() / "d-selector-lm.arpa"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------- interpolated KN -> ARPA
def _ney_discount(counts: dict) -> float:
    """D = n1 / (n1 + 2*n2) from the count-of-counts; clamped, tiny-corpus safe."""
    cc = Counter(counts.values())
    n1, n2 = cc.get(1, 0), cc.get(2, 0)
    if n1 + 2 * n2 == 0:
        return 0.5
    return min(max(n1 / (n1 + 2 * n2), 1e-4), 1 - 1e-4)


def train_kn_arpa(sentences: list[list[str]], order: int = ORDER) -> str:
    """Interpolated Kneser-Ney n-gram as a deterministic ARPA string.

    Adjusted counts: raw at the highest order and for n-grams starting with
    <s> (which have no left extension); continuation counts elsewhere.
    Probabilities stored are the fully interpolated values; backoff weight of
    a context h is D_n * types(h) / total(h) - the standard interpolated-KN-
    in-ARPA layout that KenLM queries natively. <unk> carries the uniform
    share of the unigram interpolation mass, so it is never -inf.
    """
    assert sentences, "empty LM corpus"
    raw = [defaultdict(int) for _ in range(order + 1)]      # raw[n][gram]
    for toks in sentences:
        seq = ["<s>"] + list(toks) + ["</s>"]
        for n in range(1, order + 1):
            for i in range(len(seq) - n + 1):
                raw[n][tuple(seq[i:i + n])] += 1

    adj = [None] * (order + 1)
    adj[order] = dict(raw[order])
    for n in range(order - 1, 0, -1):
        cont = defaultdict(int)
        for g in raw[n + 1]:
            cont[g[1:]] += 1
        a = {}
        for g, c in raw[n].items():
            a[g] = c if g[0] == "<s>" else cont.get(g, 0)
        adj[n] = {g: c for g, c in a.items() if c > 0}

    disc = [None] * (order + 1)
    for n in range(1, order + 1):
        disc[n] = _ney_discount(adj[n])

    vocab = sorted(w for (w,) in adj[1] if w != "<s>")
    v_open = len(vocab) + 1                                 # + <unk>
    d1 = disc[1]
    total1 = sum(c for g, c in adj[1].items() if g != ("<s>",))
    types1 = sum(1 for g in adj[1] if g != ("<s>",))
    lam1 = d1 * types1 / total1
    prob = [None, {}]                                       # prob[n][gram]
    for w in vocab:
        c = adj[1][(w,)]
        prob[1][(w,)] = max(c - d1, 0.0) / total1 + lam1 / v_open
    prob[1][("<unk>",)] = lam1 / v_open
    prob[1][("<s>",)] = 10.0 ** -99                         # never predicted

    bows = [None, {}]                                       # bow of context, per n-1
    for n in range(2, order + 1):
        by_ctx = defaultdict(dict)
        for g, c in adj[n].items():
            by_ctx[g[:-1]][g[-1]] = c
        dn = disc[n]
        pn, bown = {}, {}
        for ctx, ws in by_ctx.items():
            tot = sum(ws.values())
            bow = dn * len(ws) / tot
            bown[ctx] = bow
            for w, c in ws.items():
                lower = prob[n - 1][(ctx[1:] + (w,)) if n > 2 else (w,)]
                pn[ctx + (w,)] = max(c - dn, 0.0) / tot + bow * lower
        prob.append(pn)
        bows.append(bown)

    def l10(p: float) -> float:
        return max(math.log10(p), -99.0) if p > 0 else -99.0

    lines = ["\\data\\"]
    entries = [None] + [sorted(prob[n]) for n in range(1, order + 1)]
    for n in range(1, order + 1):
        lines.append(f"ngram {n}={len(entries[n])}")
    for n in range(1, order + 1):
        lines.append("")
        lines.append(f"\\{n}-grams:")
        ctx_bows = bows[n + 1] if n < order else {}
        for g in entries[n]:
            head = f"{l10(prob[n][g]):.7f}\t{' '.join(g)}"
            if g in ctx_bows:
                head += f"\t{l10(ctx_bows[g]):.7f}"
            lines.append(head)
    lines += ["", "\\end\\", ""]
    return "\n".join(lines)


def load_lm(path: Path):
    """KenLM model handle. A missing/unreadable LM is a HARD error, never silent."""
    if not path.exists():
        raise FileNotFoundError(
            f"KenLM model missing: {path} - run `d_selector.py build-lm` first; "
            f"the selector must never run without its preregistered LM")
    import kenlm
    return kenlm.Model(str(path))


def make_lm_scorer(model):
    """text -> per-word KenLM log10 prob with BOS/EOS, cached by text."""
    cache: dict[str, float] = {}

    def score(text: str) -> float:
        if text not in cache:
            toks = ftoks(text)
            total = model.score(" ".join(toks), bos=True, eos=True)
            cache[text] = total / (len(toks) + 1)           # EOS is a scored word
        return cache[text]

    score.cache = cache
    return score


# ------------------------------------------------------------- selector core
def dedup_candidates(hyps: list[dict]) -> list[dict]:
    """Distinct hypotheses after frozen normalization, best beam rank kept.

    hyps are in original beam order (rank = index). Each distinct ftoks key
    keeps its lowest-rank member's rank, raw text and CT2 score. Candidates
    come back in rank order, so candidates[0] is (the group of) top1.
    """
    seen: dict[tuple, int] = {}
    out: list[dict] = []
    for rank, h in enumerate(hyps):
        key = tuple(ftoks(h["text"]))
        if key in seen:
            continue
        seen[key] = rank
        out.append({"rank": rank, "text": h["text"], "A": h["score"]})
    return out


def select_chunk(cands: list[dict], lam: float, s_a: float, s_l: float,
                 lm_score) -> dict:
    """The prereg rule for one chunk. Returns the chosen candidate.

    argmax over alternatives of (dA/s_A + lambda*dL/s_L); ties to the lower
    beam rank (iteration is in rank order and only a STRICTLY greater score
    displaces the incumbent); the alternative is taken only if its combined
    score is strictly positive, else top1 stays.
    """
    top = cands[0]
    best, best_g = None, 0.0
    for c in cands[1:]:
        g = (c["A"] - top["A"]) / s_a
        if lam:
            g += lam * (lm_score(c["text"]) - lm_score(top["text"])) / s_l
        if g > best_g:                       # strict: positivity + rank ties
            best, best_g = c, g
    return best if best is not None else top


def window_candidates(rec: dict) -> list[list[dict]]:
    """Dedup'd candidate lists for the non-skipped chunks of one window."""
    return [dedup_candidates(c["hyps"])
            for c in rec["chunks"] if not c["skipped"]]


def select_window(chunk_cands: list[list[dict]], lam: float, s_a: float,
                  s_l: float, lm_score) -> tuple[str, list[int]]:
    """(reconstructed text under the pipeline join, chosen beam rank per chunk)."""
    picks = [select_chunk(cands, lam, s_a, s_l, lm_score)
             for cands in chunk_cands]
    text = "".join(p["text"] for p in picks).strip()
    return text, [p["rank"] for p in picks]


def rms(vals: list[float]) -> float:
    if not vals:
        return 1.0
    v = math.sqrt(sum(x * x for x in vals) / len(vals))
    return v if v > 0 else 1.0


def fold_scales(train_cands: list[list[list[dict]]], lm_score
                ) -> tuple[float, float]:
    """(s_A, s_L): RMS of candidate-minus-top1 differences over the fold."""
    da, dl = [], []
    for window in train_cands:
        for cands in window:
            top = cands[0]
            for c in cands[1:]:
                da.append(c["A"] - top["A"])
                dl.append(lm_score(c["text"]) - lm_score(top["text"]))
    return rms(da), rms(dl)


def pick_lambda(totals: dict[float, int]) -> float:
    """Smallest-lambda argmin over the frozen grid (strict improvement only)."""
    best_lam, best = None, None
    for lam in GRID:
        t = totals[lam]
        if best is None or t < best:
            best_lam, best = lam, t
    return best_lam


# --------------------------------------------------------------------- build-lm
def lm_rows() -> list[dict]:
    rep = json.loads(REPORT_FULL.read_text())
    rows = [it for it in rep["items"] if it["cityId"] not in EVAL_CITIES]
    assert all(it["cityId"] not in EVAL_CITIES for it in rows)
    assert rows, "no seen-city rows in the full report"
    return rows


def build_lm() -> None:
    rows = lm_rows()
    sents = [ftoks(it["referenceText"]) for it in rows]
    n_words = sum(len(s) for s in sents)
    corpus_path().write_text(
        "\n".join(" ".join(s) for s in sents) + "\n", encoding="utf-8")
    log(f"LM corpus: {len(rows)} seen-city rows, {n_words} ftoks words "
        f"-> {corpus_path()}")

    arpa = train_kn_arpa(sents, ORDER)
    arpa_path().write_text(arpa, encoding="utf-8")
    counts = {ln.split("=")[0].split()[1]: int(ln.split("=")[1])
              for ln in arpa.splitlines()
              if ln.startswith("ngram ")}
    log(f"ARPA {ORDER}-gram: {counts} -> {arpa_path()}")

    model = load_lm(arpa_path())            # smoke: kenlm loads and scores it
    probe = model.score("κύριε πρόεδρε", bos=True, eos=True)
    assert math.isfinite(probe), probe
    log(f"kenlm loads the ARPA; score('κύριε πρόεδρε') = {probe:.3f}")

    per_city = Counter(it["cityId"] for it in rows)
    manifest = {
        "arm": ARM,
        "frozen_at": "2026-08-12",
        "purpose": ("frozen LM-corpus manifest for the arm D prereg selector; "
                    "written BEFORE any selector scoring"),
        "corpus_rule": ("referenceText of every seen-city item of the "
                        "2026-08-10 full benchmark report; argos and "
                        "orestiada (the only eval/unseen cities) excluded "
                        "and asserted absent"),
        "excluded_cities": list(EVAL_CITIES),
        "tokenizer": "eval.controlled_eval.eval_freeze.ftoks (frozen scorer)",
        "rows": len(rows),
        "rows_per_city": dict(sorted(per_city.items())),
        "words": n_words,
        "files": [
            {"path": str(REPORT_FULL.relative_to(ROOT)),
             "sha256": sha256(REPORT_FULL), "rows": len(rows),
             "role": "source (220 of its 260 items; 40 eval-city excluded)"},
            {"path": str(corpus_path()), "sha256": sha256(corpus_path()),
             "rows": len(rows), "role": "normalized corpus, one item per line"},
            {"path": str(arpa_path()), "sha256": sha256(arpa_path()),
             "rows": sum(counts.values()), "role": "ARPA 4-gram",
             "ngram_counts": counts},
        ],
        "estimator": {
            "requested": "KenLM lmplz modified Kneser-Ney 4-gram",
            "used": ("pure-python interpolated Kneser-Ney (continuation "
                     "counts, one Ney discount per order), ARPA output, "
                     "queried through kenlm.Model"),
            "deviation": ("DEVIATION: lmplz unavailable (no binary, no "
                          "cmake/boost to build it, pip kenlm wheel is "
                          "query-only, nltk absent); estimation is plain "
                          "interpolated KN, not modified KN"),
        },
    }
    LM_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    log(f"frozen LM manifest -> {LM_MANIFEST}")


# ----------------------------------------------------------------------- score
def one_sided_bootstrap(counts_sel: dict, counts_base: dict, wids: list[str],
                        blocks: list[str]) -> dict:
    """Meeting-clustered paired bootstrap on pooled delta-WER; frozen counts,
    no refitting inside replicates. One-sided 95% upper bound = P95."""
    import numpy as np
    a = np.array([(sum(counts_sel[w][:3]), counts_sel[w][3]) for w in wids],
                 dtype=float)
    b = np.array([(sum(counts_base[w][:3]), counts_base[w][3]) for w in wids],
                 dtype=float)
    groups = defaultdict(list)
    for i, m in enumerate(blocks):
        groups[m].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(BOOT_SEED)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        den = a[idx, 1].sum()
        diffs[i] = (a[idx, 0].sum() - b[idx, 0].sum()) / den if den else np.nan
    point = (a[:, 0].sum() - b[:, 0].sum()) / a[:, 1].sum()
    return {"delta": float(point),
            "upper95_one_sided": float(np.nanpercentile(diffs, 95)),
            "ci95_two_sided": [float(x) for x in
                               np.nanpercentile(diffs, [2.5, 97.5])],
            "n_resamples": N_BOOT, "seed": BOOT_SEED,
            "n_clusters": len(keys)}


def score() -> dict:
    if not LM_MANIFEST.exists():
        raise SystemExit(f"frozen LM manifest missing ({LM_MANIFEST}) - "
                         f"run build-lm before any selector scoring")
    state_p = out_dir() / "eval-D-nbest.json"
    if not state_p.exists():
        raise SystemExit(f"N-best state missing: {state_p}")
    state = json.loads(state_p.read_text())
    oracle_res = json.loads((out_dir() / "results-D-eval.json").read_text())

    rows = DA.rows("eval")
    wids = [r["window_id"] for r in rows]
    meeting = {r["window_id"]: r["meeting_id"] for r in rows}
    for wid in wids:
        if wid not in state["windows"]:
            raise SystemExit(f"incomplete N-best state: {wid} missing")

    rep = json.loads(REPORT_FULL.read_text())
    refs_raw = {it["itemId"]: it["referenceText"] for it in rep["items"]}
    refs = {w: ftoks(refs_raw[w]) for w in wids}

    model = load_lm(arpa_path())            # hard error if absent
    lm_score = make_lm_scorer(model)

    cands = {w: window_candidates(state["windows"][w]) for w in wids}
    base = {}
    for w in wids:
        base[w] = (*sdi(refs[w], ftoks(state["windows"][w]["text"])),
                   len(refs[w]))

    # -- lambda=0 must reproduce beam8-top1 EXACTLY, before anything else --
    for w in wids:
        text0, ranks0 = select_window(cands[w], 0.0, 1.0, 1.0, lm_score)
        assert text0 == state["windows"][w]["text"], \
            f"{w}: lambda=0 selection differs from the pipeline top-1"
        assert all(r == 0 for r in ranks0), f"{w}: lambda=0 chose rank != 0"
    log("lambda=0 identity: PASS on all 39 windows")

    # baseline must equal the frozen oracle-screen beam8-top1 exactly
    t_base = DA.rates(base)
    frozen = oracle_res["beam8_top1"]
    for k in ("sub", "del", "ins", "ref_tokens"):
        assert t_base[k] == frozen[k], (k, t_base[k], frozen[k])

    # per-(window, chosen-ranks) sdi cache: selection changes rarely across
    # lambdas/folds and sdi is the expensive step
    sdi_cache: dict[tuple, tuple] = {}

    def window_counts(w: str, lam: float, s_a: float, s_l: float
                      ) -> tuple[tuple, list[int]]:
        text, ranks = select_window(cands[w], lam, s_a, s_l, lm_score)
        key = (w, tuple(ranks))
        if key not in sdi_cache:
            sdi_cache[key] = (*sdi(refs[w], ftoks(text)), len(refs[w]))
        return sdi_cache[key], ranks

    folds = sorted(set(meeting.values()))
    assert len(folds) == 31, f"{len(folds)} folds, prereg says 31"

    sel = {}
    fold_info = {}
    chosen_ranks: dict[str, list[int]] = {}
    for m in folds:
        train = [w for w in wids if meeting[w] != m]
        held = [w for w in wids if meeting[w] == m]
        s_a, s_l = fold_scales([cands[w] for w in train], lm_score)
        totals = {}
        for lam in GRID:
            totals[lam] = sum(sum(window_counts(w, lam, s_a, s_l)[0][:3])
                              for w in train)
        lam_star = pick_lambda(totals)
        for w in held:
            sel[w], chosen_ranks[w] = window_counts(w, lam_star, s_a, s_l)
        fold_info[m] = {"lambda": lam_star, "s_A": round(s_a, 6),
                        "s_L": round(s_l, 6),
                        "train_totals": {str(k): v for k, v in totals.items()},
                        "held_windows": held}

    t_sel = DA.rates(sel)
    t_oracle = oracle_res["oracle_8"]
    ceiling = t_base["wer"] - t_oracle["wer"]
    delta = t_sel["wer"] - t_base["wer"]
    err_base = t_base["sub"] + t_base["del"] + t_base["ins"]
    err_sel = t_sel["sub"] + t_sel["del"] + t_sel["ins"]
    err_ceiling = oracle_res["ceiling"]["errors_recovered_8"]
    recovery = (-delta / ceiling) if ceiling else None

    # -- overrides: chunks where the selector displaced top1, and whether the
    #    window got better or worse when exactly that chunk is reverted --
    overrides = []
    for w in wids:
        ranks = chosen_ranks[w]
        if all(r == 0 for r in ranks):
            continue
        live = cands[w]
        err_sel_w = sum(sel[w][:3])
        for j, r in enumerate(ranks):
            if r == 0:
                continue
            reverted = list(ranks)
            reverted[j] = 0
            by_rank = [{c["rank"]: c for c in cc} for cc in live]
            text = "".join(by_rank[j2][rr]["text"]
                           for j2, rr in enumerate(reverted)).strip()
            err_rev = sum(sdi(refs[w], ftoks(text)))
            overrides.append({
                "window": w, "chunk": j, "rank": r,
                "delta_errors": err_sel_w - err_rev,   # <0: override helped
            })
    n_win = sum(1 for o in overrides if o["delta_errors"] < 0)
    n_loss = sum(1 for o in overrides if o["delta_errors"] > 0)
    n_tie = sum(1 for o in overrides if o["delta_errors"] == 0)

    blocks = [meeting[w] for w in wids]
    boot = one_sided_bootstrap(sel, base, wids, blocks)

    # leave-one-window-out sign stability of the pooled delta
    lowo = []
    for w in wids:
        keep = [x for x in wids if x != w]
        n = sum(base[x][3] for x in keep)
        d = (sum(sum(sel[x][:3]) for x in keep)
             - sum(sum(base[x][:3]) for x in keep)) / n
        lowo.append(d)
    lowo_sign = {"n_negative": sum(1 for d in lowo if d < 0),
                 "n_zero": sum(1 for d in lowo if d == 0),
                 "n_positive": sum(1 for d in lowo if d > 0),
                 "min": min(lowo), "max": max(lowo)}

    lam_dist = Counter(v["lambda"] for v in fold_info.values())
    gate_recovery = recovery is not None and recovery >= GATE_RECOVERY
    gate_upper = boot["upper95_one_sided"] < GATE_UPPER
    passed = bool(gate_recovery and gate_upper)

    per_window = {w: {"n_ref": base[w][3],
                      "base_sdi": list(base[w][:3]),
                      "selector_sdi": list(sel[w][:3]),
                      "chosen_ranks": chosen_ranks[w],
                      "meeting": meeting[w]} for w in wids}

    return {
        "arm": ARM, "set": "eval",
        "prereg": "docs/specs/2026-08-12-serving-stack-plan.md arm D, "
                  "'Prereg selector 2026-08-12'",
        "lm_manifest": str(LM_MANIFEST.relative_to(ROOT)),
        "lm_deviation": ("estimator is pure-python interpolated KN (lmplz "
                         "unavailable); queried through kenlm - see manifest"),
        "config": {"grid": list(GRID), "order": ORDER,
                   "A": "CT2 per-token score (as recorded)",
                   "L": "per-word kenlm log10 prob, BOS/EOS, ftoks tokens",
                   "scales": "fold-level RMS of (candidate - top1) diffs",
                   "cross_fitting": "leave-one-meeting-out, 31 folds, "
                                    "objective = pooled train S+D+I",
                   "tie_break": "smaller lambda, then lower beam rank",
                   "positivity": "alternative only on strictly positive score"},
        "n_windows": len(wids), "n_folds": len(folds),
        "beam8_top1": t_base,
        "selector": t_sel,
        "oracle_8": {k: t_oracle[k] for k in ("wer", "sub", "del", "ins")},
        "delta_wer_selector_vs_top1": delta,
        "errors": {"base": err_base, "selector": err_sel,
                   "recovered": err_base - err_sel,
                   "ceiling_errors": err_ceiling},
        "ceiling_wer": ceiling,
        "recovery_fraction_of_ceiling": recovery,
        "bootstrap": boot,
        "lowo_sign_stability": lowo_sign,
        "per_fold_lambda": {m: fold_info[m]["lambda"] for m in folds},
        "lambda_distribution": {str(k): v for k, v in sorted(lam_dist.items())},
        "fold_detail": fold_info,
        "overrides": {"n_chunks_overridden": len(overrides),
                      "wins": n_win, "losses": n_loss, "ties": n_tie,
                      "detail": overrides},
        "per_window": per_window,
        "gate": {"recovery_threshold": GATE_RECOVERY,
                 "recovery": recovery,
                 "recovery_passed": bool(gate_recovery),
                 "upper_bound_threshold": GATE_UPPER,
                 "upper_bound": boot["upper95_one_sided"],
                 "upper_bound_passed": bool(gate_upper),
                 "passed": passed},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-lm")
    sub.add_parser("score")
    a = ap.parse_args()

    if a.cmd == "build-lm":
        build_lm()
        return

    res = score()
    RESULTS_REPO.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for name in ("beam8_top1", "selector"):
        t = res[name]
        log(f"{name:>10}  WER {t['wer']:.4f}  S {t['sub']} D {t['del']} "
            f"I {t['ins']}")
    o = res["oracle_8"]
    log(f"{'oracle_8':>10}  WER {o['wer']:.4f}  S {o['sub']} D {o['del']} "
        f"I {o['ins']}")
    log(f"delta {res['delta_wer_selector_vs_top1']:+.4f}  recovery "
        f"{res['recovery_fraction_of_ceiling']:.1%} of ceiling "
        f"{res['ceiling_wer']:.4f}")
    log(f"bootstrap upper (one-sided 95%) {res['bootstrap']['upper95_one_sided']:+.4f}")
    log(f"lambda per fold: {res['lambda_distribution']}")
    ov = res["overrides"]
    log(f"overrides: {ov['n_chunks_overridden']} chunks "
        f"(win {ov['wins']} / loss {ov['losses']} / tie {ov['ties']})")
    log(f"gate: {'PASS' if res['gate']['passed'] else 'FAIL'} "
        f"(recovery {'PASS' if res['gate']['recovery_passed'] else 'FAIL'}, "
        f"upper bound {'PASS' if res['gate']['upper_bound_passed'] else 'FAIL'})")
    log(f"-> {RESULTS_REPO}")


if __name__ == "__main__":
    main()
