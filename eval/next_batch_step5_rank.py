"""Next-batch Step 5 — free (text-only) interestingness ranking + diverse select.

Runbook Steps 5-6. Codex-designed formula (2026-07-01). The free score only
*prioritizes* which edits are worth paying to verify; it is NOT a trust signal
(faithfulness = cer_soniox, computed later on the top slice). Principle: the free
score finds interesting edits; audio decides whether they are true.

Pipeline:
  1. exclude already-curated/built utterances
  2. hard prefilter degenerate candidates (logged)
  3. base interestingness score from free features
  4. deterministic edit-signature per candidate
  5. lazy-greedy diverse selection of the top-K (multi-level signatures,
     diminishing returns 1/sqrt(1+count)) -> ranked shortlist for LLM/Soniox vetting

Outputs:
  data/next-batch/ranked.parquet         all survivors with base_score + signatures
  data/next-batch/shortlist.parquet      top-K greedy-diverse, in selection order
  data/next-batch/step5_summary.md
"""
from __future__ import annotations

import argparse
import difflib
import heapq
import json
import math
from pathlib import Path

import pandas as pd

from eval.scoring import greek_normalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "next-batch"

CATEGORY_WEIGHT = {
    "named_entity": 1.35, "homophone": 1.30, "word_boundary": 1.25,
    "other_lexical": 1.15, "insertion_deletion": 1.00, "morph_grammar": 0.55,
    "number_date": 0.35, "acronym_abbreviation": 0.30,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _chain_score(n: int) -> float:
    if n == 1:
        return 1.00
    if n == 2:
        return 0.95
    if n == 3:
        return 0.80
    if n <= 5:
        return 0.55
    return 0.25


def _wdiff_score(w: float) -> float:
    if w <= 1:
        return 1.10
    if w <= 2:
        return 1.00
    if w <= 3:
        return 0.75
    if w <= 5:
        return 0.45
    return 0.15


def _duration_score(d: float) -> float:
    if 1.5 <= d <= 12:
        return 1.00
    if 12 < d <= 25:
        return 0.85
    if 0.8 <= d < 1.5:
        return 0.60
    if d > 25:
        return 0.55
    return 0.25


def _rewrite_guard(cer_proxy: float, char_diff: int, norm_word_diff: float) -> float:
    if norm_word_diff >= 8:
        return 0.0
    if cer_proxy >= 0.55:
        return 0.15
    if char_diff <= 2:
        return 0.25
    return 1.0


def edit_signature(input_raw: str, gold: str, category: str) -> tuple[str, str, str]:
    """(sig1 category, sig2 category:op, sig3 category:tokA->tokB) from the
    normalised token diff. Deterministic, diff-of-before->after (not embeddings)."""
    tb = greek_normalize(input_raw).split()
    ta = greek_normalize(gold).split()
    ops = difflib.SequenceMatcher(a=tb, b=ta, autojunk=False).get_opcodes()
    kinds = {op[0] for op in ops if op[0] != "equal"}
    if kinds == {"replace"}:
        op = "replace"
    elif kinds == {"insert"}:
        op = "insert"
    elif kinds == {"delete"}:
        op = "delete"
    elif kinds:
        op = "mixed"
    else:
        op = "equal"
    tok = ""
    for kind, i1, i2, j1, j2 in ops:
        if kind == "replace":
            tok = f"{' '.join(tb[i1:i2])[:24]}->{' '.join(ta[j1:j2])[:24]}"
            break
        if kind == "delete" and not tok:
            tok = f"-{' '.join(tb[i1:i2])[:24]}"
        if kind == "insert" and not tok:
            tok = f"+{' '.join(ta[j1:j2])[:24]}"
    return category, f"{category}:{op}", f"{category}:{tok}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=15000, help="greedy-diverse shortlist size")
    args = ap.parse_args()

    df = pd.read_parquet(OUT_DIR / "candidates.parquet")
    curated = set(json.loads((OUT_DIR / "curated_ids.json").read_text()))
    log: list[str] = ["# Step 5 — free interestingness ranking + diverse select\n"]

    def drop(label, before, after):
        log.append(f"- {label}: {before:,} -> {after:,}  (dropped {before-after:,})")

    n0 = len(df)
    df = df[~df["utterance_id"].isin(curated)].copy()
    drop("exclude already-curated/built", n0, len(df))

    # normalised lengths for the cer proxy
    ln_in = df["input_raw"].map(lambda s: len(greek_normalize(s)))
    ln_gold = df["gold_final"].map(lambda s: len(greek_normalize(s)))
    df["cer_proxy"] = df["char_diff"] / pd.concat([ln_in, ln_gold], axis=1).max(axis=1).clip(lower=1)

    # ---- hard prefilter (Codex) ----
    def keep_mask(d):
        return (
            d["dur_valid"] & (d["duration"] >= 0.8) & (d["duration"] <= 30)
            & (d["norm_word_diff"] < 8) & (d["n_edits"] <= 8)
            & (d["cer_proxy"] >= 0.05) & (d["cer_proxy"] <= 0.70)
        )
    n = len(df)
    df = df[keep_mask(df)].copy()
    drop("hard prefilter (dur/word-diff/chain/cer_proxy bounds)", n, len(df))

    # ---- base interestingness ----
    df["magnitude_score"] = ((df["cer_proxy"] - 0.08) / 0.22).map(_clamp01)
    df["base_score"] = (
        df["category"].map(CATEGORY_WEIGHT).fillna(0.5)
        * df["magnitude_score"].clip(lower=0.05)  # keep small edits alive, just low
        * df["n_edits"].map(_chain_score)
        * df["norm_word_diff"].map(_wdiff_score)
        * df["duration"].map(_duration_score)
        * df["ebclass"].map({"user_only": 1.00, "task_then_user": 0.90, "task_plus_user": 0.90}).fillna(0.9)
        * [_rewrite_guard(cp, cd, wd) for cp, cd, wd in
           zip(df["cer_proxy"], df["char_diff"], df["norm_word_diff"])]
    )

    sigs = [edit_signature(b, g, c) for b, g, c in
            zip(df["input_raw"], df["gold_final"], df["category"])]
    df["sig1"], df["sig2"], df["sig3"] = (
        [s[0] for s in sigs], [s[1] for s in sigs], [s[2] for s in sigs])

    df = df.sort_values("base_score", ascending=False).reset_index(drop=True)
    df.to_parquet(OUT_DIR / "ranked.parquet", index=False)

    # ---- lazy-greedy diverse selection ----
    rows = df.to_dict("records")
    c1: dict = {}
    c2: dict = {}
    c3: dict = {}
    ccity: dict = {}
    cmeet: dict = {}

    def div_weight(r) -> float:
        return (
            1 / math.sqrt(1 + c1.get(r["sig1"], 0))
            * 1 / math.sqrt(1 + c2.get(r["sig2"], 0))
            * 1 / math.sqrt(1 + c3.get(r["sig3"], 0))
            * 1 / math.sqrt(1 + 0.25 * ccity.get(r["city_id"], 0))
            * 1 / math.sqrt(1 + 0.10 * cmeet.get(r["meeting_id"], 0))
        )

    # max-heap of (-current_estimate, idx); lazy re-evaluation on pop
    heap = [(-rows[i]["base_score"], i) for i in range(len(rows))]
    heapq.heapify(heap)
    picked: list[int] = []
    K = min(args.topk, len(rows))
    while heap and len(picked) < K:
        neg_est, i = heapq.heappop(heap)
        r = rows[i]
        true_score = r["base_score"] * div_weight(r)
        # if stale (heap top could beat it), reinsert with refreshed score
        if heap and true_score < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-true_score, i))
            continue
        picked.append(i)
        c1[r["sig1"]] = c1.get(r["sig1"], 0) + 1
        c2[r["sig2"]] = c2.get(r["sig2"], 0) + 1
        c3[r["sig3"]] = c3.get(r["sig3"], 0) + 1
        ccity[r["city_id"]] = ccity.get(r["city_id"], 0) + 1
        cmeet[r["meeting_id"]] = cmeet.get(r["meeting_id"], 0) + 1

    short = df.iloc[picked].copy()
    short["select_rank"] = range(1, len(short) + 1)
    short.to_parquet(OUT_DIR / "shortlist.parquet", index=False)

    log.append(f"\n**Survivors after prefilter: {len(df):,}** -> greedy-diverse shortlist: {len(short):,}\n")
    log.append("## Shortlist category mix\n")
    for k, v in short["category"].value_counts().items():
        log.append(f"- {k}: {v:,} ({v/len(short)*100:.1f}%)")
    log.append("\n## Shortlist duration buckets\n")
    b = pd.cut(short["duration"], [0, 1.5, 3, 5, 10, 15, 25, 30],
               labels=["<1.5", "1.5-3", "3-5", "5-10", "10-15", "15-25", "25-30"])
    for k, v in b.value_counts().sort_index().items():
        log.append(f"- {k}s: {v:,}")
    log.append(f"\n## Coverage\n- distinct cities: {short['city_id'].nunique()}"
               f"\n- distinct meetings: {short['meeting_id'].nunique()}"
               f"\n- distinct sig3 (specific edits): {short['sig3'].nunique():,}")
    log.append(f"\n- base_score range: {short['base_score'].min():.3f}..{short['base_score'].max():.3f}")
    log.append("\n## ebclass\n")
    for k, v in short["ebclass"].value_counts().items():
        log.append(f"- {k}: {v:,}")

    (OUT_DIR / "step5_summary.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"\nwrote ranked.parquet ({len(df):,}), shortlist.parquet ({len(short):,}), step5_summary.md")


if __name__ == "__main__":
    main()
