"""Next-batch Step 1 — build the candidate pool from the corrections CSV.

Runbook: docs/runbooks/next-batch-selection-onbox-brief.md, Step 1.

Start from the precomputed chains (one row per utterance_id), keep only chains a
human touched, drop degenerate/normalisation-only edits, drop excluded + held-out
meetings, join the audio span, and compute selection features. Every drop is
logged so nothing is capped silently.

Output:
  data/next-batch/candidates.parquet
  data/next-batch/step1_summary.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

from eval.categorize import categorize
from eval.exclusions import load_excluded_keys
from eval.scoring import greek_normalize, wer

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
CSV = ROOT / "data-1779206108158.csv"
OUT_DIR = ROOT / "data" / "next-batch"

# categories that are normalisation-only (greek_normalize-equal before/after)
_NORM_EQUAL_CATS = {"no_change", "accent_tonos", "final_sigma", "punctuation_capitalization"}


def _ebclass(has_task: bool, has_user: bool, task_then_user: bool) -> str:
    if has_task and has_user:
        return "task_then_user" if task_then_user else "task_plus_user"
    if has_task:
        return "task_only"
    return "user_only"


def _load_audio_spans() -> pd.DataFrame:
    """One row per utterance_id with its audio span. utterance_start/end/audio_url
    are constant across a utterance's edits, so the first row is representative."""
    df = pd.read_csv(
        CSV,
        usecols=["utterance_id", "city_id", "meeting_id", "edit_timestamp",
                 "utterance_start", "utterance_end", "audio_url"],
        dtype={"utterance_id": str, "city_id": str, "meeting_id": str, "audio_url": str},
    )
    df["edit_timestamp"] = pd.to_datetime(df["edit_timestamp"], errors="coerce")
    df["utterance_start"] = pd.to_numeric(df["utterance_start"], errors="coerce")
    df["utterance_end"] = pd.to_numeric(df["utterance_end"], errors="coerce")
    df = df.sort_values(["utterance_id", "edit_timestamp"], kind="stable")
    spans = df.groupby("utterance_id", sort=False).first().reset_index()
    return spans[["utterance_id", "utterance_start", "utterance_end", "audio_url"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def drop(label: str, before: int, after: int) -> None:
        log.append(f"- {label}: {before:,} -> {after:,}  (dropped {before - after:,})")

    chains = pd.read_parquet(EVAL / "chains.parquet")
    # join safety: utterance_id must be globally unique in chains (one row each)
    assert chains["utterance_id"].is_unique, "utterance_id not unique in chains.parquet"
    chains["city_id"] = chains["city_id"].astype(str)
    chains["meeting_id"] = chains["meeting_id"].astype(str)
    n0 = len(chains)
    log.append(f"# Step 1 — candidate pool\n\nStart: {n0:,} chains (one per utterance_id)\n")

    # (a) keep only chains a human touched
    df = chains[chains["has_user"]].copy()
    drop("(a) keep has_user==True (drop LLM-only chains)", n0, len(df))

    # link integrity: before==prev-after must hold or input_raw/gold_final are unreliable
    n = len(df)
    df = df[df["links_ok"]].copy()
    drop("(a2) drop broken edit chains (links_ok==False)", n, len(df))

    # (b) degenerate / normalisation-only drops. Categorise first so we can report
    # exactly which orthographic buckets we removed (audit visibility, per review).
    df["category"] = [categorize(b, a) for b, a in zip(df["input_raw"], df["gold_final"])]
    df["_norm_in"] = df["input_raw"].map(greek_normalize)
    df["_norm_gold"] = df["gold_final"].map(greek_normalize)

    norm_equal = df[df["_norm_in"] == df["_norm_gold"]]
    norm_equal_breakdown = norm_equal["category"].value_counts().to_dict()
    n = len(df)
    df = df[df["_norm_in"] != df["_norm_gold"]].copy()
    drop("(b1) drop normalisation-only edits (greek_normalize-equal)", n, len(df))
    log.append("    normalisation-only breakdown (dropped): "
               + ", ".join(f"{k}={v:,}" for k, v in sorted(norm_equal_breakdown.items())))

    n = len(df)
    df = df[df["_norm_gold"] != ""].copy()  # empty corrected text -> nothing to learn
    drop("(b2) drop empty normalised gold_final", n, len(df))

    # post-drop invariants the review asked us to assert
    assert not df["category"].isin(_NORM_EQUAL_CATS).any(), \
        "normalisation-only category survived the normalize-equal drop"

    # (c) meeting-level exclusions
    excl = load_excluded_keys()  # set of (city_id, meeting_id), strings
    n = len(df)
    df = df[~df.apply(lambda r: (r["city_id"], r["meeting_id"]) in excl, axis=1)].copy()
    drop(f"(c1) drop unreviewed-meeting denylist ({len(excl)} meetings)", n, len(df))

    # held-out temporal test set must never enter a training batch (leakage)
    split = json.loads((EVAL / "split.json").read_text())
    eval_mids = set(split.get("eval_meeting_ids", []))
    n = len(df)
    df = df[~df["meeting_id"].isin(eval_mids)].copy()
    drop(f"(c2) drop held-out eval meetings ({len(eval_mids)} meetings)", n, len(df))

    # (d) features
    df["ebclass"] = [
        _ebclass(t, u, tu)
        for t, u, tu in zip(df["has_task"], df["has_user"], df["task_then_user"])
    ]
    assert (df["ebclass"] != "task_only").all(), "task_only survived has_user filter"

    df["char_diff"] = [
        Levenshtein.distance(str(b), str(a))
        for b, a in zip(df["input_raw"], df["gold_final"])
    ]
    df["norm_word_diff"] = [wer(b, a) for b, a in zip(df["input_raw"], df["gold_final"])]

    spans = _load_audio_spans()
    assert spans["utterance_id"].is_unique
    df = df.merge(spans, on="utterance_id", how="left")
    n_no_audio = int(df["audio_url"].isna().sum())
    df["duration"] = df["utterance_end"] - df["utterance_start"]

    # duration validity (kept as a flag, not dropped here — sampling/extraction decide)
    bad_dur = (~(df["duration"] > 0)) | df["duration"].isna()
    df["dur_valid"] = ~bad_dur

    df = df.drop(columns=["_norm_in", "_norm_gold"])

    out_pq = OUT_DIR / "candidates.parquet"
    df.to_parquet(out_pq, index=False)

    # ---- summary ----
    log.append(f"\n**Final candidate pool: {len(df):,} utterances** -> `{out_pq.relative_to(ROOT)}`\n")
    log.append(f"- rows missing audio span after join: {n_no_audio:,}")
    log.append(f"- rows with invalid/zero/NaN duration: {int(bad_dur.sum()):,}")
    log.append("\n## Category distribution (kept)\n")
    for k, v in df["category"].value_counts().items():
        log.append(f"- {k}: {v:,}")
    log.append("\n## ebclass distribution (kept)\n")
    for k, v in df["ebclass"].value_counts().items():
        log.append(f"- {k}: {v:,}")
    log.append("\n## chain length (n_edits) distribution\n")
    for k, v in df["n_edits"].value_counts().sort_index().items():
        log.append(f"- {k} edit(s): {v:,}")
    log.append("\n## duration buckets (valid only, seconds)\n")
    dv = df.loc[df["dur_valid"], "duration"]
    buckets = pd.cut(dv, [0, 1.5, 3, 5, 10, 15, 20, 30, 1e9],
                     labels=["<1.5", "1.5-3", "3-5", "5-10", "10-15", "15-20", "20-30", ">30"])
    for k, v in buckets.value_counts().sort_index().items():
        log.append(f"- {k}s: {v:,}")
    log.append("\n## char_diff (raw) describe\n```\n" + df["char_diff"].describe().to_string() + "\n```")
    log.append("\n## norm_word_diff describe\n```\n" + df["norm_word_diff"].describe().to_string() + "\n```")

    (OUT_DIR / "step1_summary.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"\nwrote {out_pq} ({len(df):,} rows) and step1_summary.md")


if __name__ == "__main__":
    main()
