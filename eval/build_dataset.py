"""Build the eval dataset from the corrections CSV.

Runbook Steps 1-3 on real data:
  - reconstruct + classify chains
  - split by meeting_id (train/eval) and by city_id (zero-shot-city)
  - mine the glossary from the TRAIN meeting split only

Outputs:
  data/eval/chains.parquet     — all chains with metadata
  data/eval/split.json         — eval meeting_ids and eval city_ids
  data/glossary/glossary.json  — {global, per_city}
  data/eval/dataset_stats.json — summary counts
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from eval.chains import reconstruct_chains
from eval.glossary import mine_glossary
from eval.splits import assert_no_leakage, split_by_city, split_by_meeting

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data-1779206108158.csv"
OUT_EVAL = ROOT / "data" / "eval"
OUT_GLOSS = ROOT / "data" / "glossary"

USECOLS = [
    "utterance_id", "edit_timestamp", "before_text", "after_text",
    "edited_by", "meeting_id", "city_id",
]


def main() -> None:
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    OUT_GLOSS.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] reading {CSV.name} ...", flush=True)
    df = pd.read_csv(CSV, usecols=USECOLS, dtype=str)
    df["edit_timestamp"] = pd.to_datetime(df["edit_timestamp"], errors="coerce")
    df["before_text"] = df["before_text"].fillna("")
    df["after_text"] = df["after_text"].fillna("")
    print(f"[{time.time()-t0:6.1f}s] {len(df):,} rows, "
          f"{df['utterance_id'].nunique():,} utterances", flush=True)

    print(f"[{time.time()-t0:6.1f}s] reconstructing chains ...", flush=True)
    chains = reconstruct_chains(df)
    print(f"[{time.time()-t0:6.1f}s] {len(chains):,} chains", flush=True)

    cdf = pd.DataFrame(chains)
    cdf["edited_by_seq"] = cdf["edited_by_seq"].apply(lambda s: "|".join(s))
    cdf.to_parquet(OUT_EVAL / "chains.parquet", index=False)

    # splits (deterministic hash-based eval folds)
    train_m, eval_m = split_by_meeting(chains, eval_frac=0.2)
    assert_no_leakage(train_m, eval_m, key="meeting_id")
    train_c, eval_c = split_by_city(chains, eval_frac=0.15)
    assert_no_leakage(train_c, eval_c, key="city_id")

    split = {
        "eval_meeting_ids": sorted({c["meeting_id"] for c in eval_m}),
        "eval_city_ids": sorted({c["city_id"] for c in eval_c}),
        "n_train_meeting_chains": len(train_m),
        "n_eval_meeting_chains": len(eval_m),
        "n_train_city_chains": len(train_c),
        "n_eval_city_chains": len(eval_c),
    }
    (OUT_EVAL / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2))

    print(f"[{time.time()-t0:6.1f}s] mining glossary from train meetings ...", flush=True)
    gloss = mine_glossary(train_m)
    (OUT_GLOSS / "glossary.json").write_text(
        json.dumps(gloss, ensure_ascii=False, indent=2)
    )

    # summary stats
    def counts(col):
        return cdf[col].value_counts().to_dict()

    stats = {
        "n_rows": int(len(df)),
        "n_chains": int(len(cdf)),
        "chain_type": counts("chain_type"),
        "n_has_task": int(cdf["has_task"].sum()),
        "n_has_user": int(cdf["has_user"].sum()),
        "n_task_then_user": int(cdf["task_then_user"].sum()),
        "links_ok_rate": float(cdf["links_ok"].mean()),
        "multi_edit_chains": int((cdf["n_edits"] > 1).sum()),
        "n_cities": int(cdf["city_id"].nunique()),
        "n_meetings": int(cdf["meeting_id"].nunique()),
        "glossary_global_terms": len(gloss["global"]),
        "glossary_cities_with_terms": len(gloss["per_city"]),
        "glossary_per_city_total_terms": sum(len(v) for v in gloss["per_city"].values()),
        "split": split,
    }
    (OUT_EVAL / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    print(f"[{time.time()-t0:6.1f}s] DONE", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
