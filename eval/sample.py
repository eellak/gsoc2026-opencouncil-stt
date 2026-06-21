"""Build the stratified held-out eval sample (runbook Step 5 / spec §7).

Per category, take up to PER_CAT held-out pure_correction rows; categories with
fewer use all (uncertainty reported downstream). Drops no_change (nothing to
fix). Writes data/eval/sample.jsonl with one row per utterance chain.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eval.categorize import categorize

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
PER_CAT = 100
SEED = 13
DROP = {"no_change"}


def _ebclass(r) -> str:
    if r.has_task and r.has_user:
        return "task_then_user" if r.task_then_user else "task_plus_user"
    if r.has_task:
        return "task_only"
    return "user_only"


def main() -> None:
    cdf = pd.read_parquet(EVAL / "chains.parquet")
    split = json.loads((EVAL / "split.json").read_text())
    ev = cdf[cdf["meeting_id"].isin(split["eval_meeting_ids"])].copy()
    pc = ev[ev.chain_type == "pure_correction"].copy()
    pc["category"] = [categorize(b, a) for b, a in zip(pc.input_raw, pc.gold_final)]
    pc["ebclass"] = pc.apply(_ebclass, axis=1)
    pc = pc[~pc.category.isin(DROP)]

    parts = []
    for cat, grp in pc.groupby("category"):
        n = min(PER_CAT, len(grp))
        parts.append(grp.sample(n=n, random_state=SEED))
    sample = pd.concat(parts).reset_index(drop=True)

    cols = ["utterance_id", "city_id", "meeting_id", "input_raw", "gold_final",
            "category", "ebclass", "chain_type", "has_task", "has_user",
            "task_then_user"]
    out = EVAL / "sample.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for _, r in sample[cols].iterrows():
            f.write(json.dumps({c: _json_safe(r[c]) for c in cols}, ensure_ascii=False) + "\n")

    print(f"wrote {len(sample)} rows -> {out}")
    print(sample.category.value_counts().to_string())


def _json_safe(v):
    import numpy as np
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


if __name__ == "__main__":
    main()
