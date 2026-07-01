"""Next-batch Step 2a — draw the stratified calibration sample.

Runbook Step 2. Picks ~N candidates whose Soniox re-transcription will calibrate
the faithfulness thresholds. Stratifies so the audit spans easy->hard AND forces
coverage of the strata that actually drive the thresholds (per plan review):
short clips, long clips, every error category, and both human edit-classes.

Deterministic (SEED). Writes data/next-batch/calib/sample.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval.scoring import greek_normalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "next-batch"
CALIB = OUT_DIR / "calib"
SEED = 13


def _short_special(gold: str) -> bool:
    n = greek_normalize(gold)
    return len(n) < 20 or len(n.split()) < 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320, help="target sample size")
    ap.add_argument("--min-dur", type=float, default=0.8,
                    help="drop clips shorter than this (s) — CER unstable, ASR noisy")
    ap.add_argument("--max-dur", type=float, default=30.0)
    args = ap.parse_args()

    CALIB.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(OUT_DIR / "candidates.parquet")

    # valid, extractable durations only for the audit
    df = df[df["dur_valid"] & (df["duration"] >= args.min_dur)
            & (df["duration"] <= args.max_dur)].copy()

    df["short_special"] = [_short_special(g) for g in df["gold_final"]]
    df["n_edits_bucket"] = pd.cut(df["n_edits"], [0, 1, 2, 1e9],
                                  labels=["1", "2", "3+"]).astype(str)
    # word-diff quartiles (normalised — robust to formatting, per review)
    df["wdiff_q"] = pd.qcut(df["norm_word_diff"].rank(method="first"), 4,
                            labels=["q1", "q2", "q3", "q4"]).astype(str)

    picked: dict[str, pd.Series] = {}  # utterance_id -> row

    def take(pool: pd.DataFrame, k: int) -> None:
        pool = pool[~pool["utterance_id"].isin(picked)]
        if len(pool) == 0 or k <= 0:
            return
        s = pool.sample(n=min(k, len(pool)), random_state=SEED)
        for _, r in s.iterrows():
            picked[r["utterance_id"]] = r

    # primary grid: n_edits_bucket x wdiff_q (12 cells) ~70% of budget
    grid_budget = int(args.n * 0.70)
    cells = df.groupby(["n_edits_bucket", "wdiff_q"], observed=True)
    per_cell = max(1, grid_budget // max(1, cells.ngroups))
    for _, g in cells:
        take(g, per_cell)

    # forced coverage strata
    take(df[df["short_special"]], max(30, int(args.n * 0.12)))         # short-special path
    take(df[df["duration"] >= 10], max(25, int(args.n * 0.10)))         # long clips (span risk)
    take(df[df["duration"] >= 20], 15)                                  # near-30s tail
    for cat, g in df.groupby("category"):                               # every category >= 12
        take(g, 12)
    for eb, g in df.groupby("ebclass"):                                 # edit-class coverage
        take(g, 15)

    # top up to N with a random remainder
    take(df, args.n - len(picked))

    sample = pd.DataFrame(list(picked.values())).reset_index(drop=True)
    sample = sample.head(args.n) if len(sample) > args.n else sample
    out = CALIB / "sample.parquet"
    sample.to_parquet(out, index=False)

    print(f"sampled {len(sample)} / target {args.n} -> {out}")
    print("\nby n_edits_bucket:\n" + sample["n_edits_bucket"].value_counts().to_string())
    print("\nby wdiff_q:\n" + sample["wdiff_q"].value_counts().to_string())
    print("\nby category:\n" + sample["category"].value_counts().to_string())
    print("\nby ebclass:\n" + sample["ebclass"].value_counts().to_string())
    print(f"\nshort_special: {int(sample['short_special'].sum())}")
    print("duration describe:\n" + sample["duration"].describe().to_string())


if __name__ == "__main__":
    main()
