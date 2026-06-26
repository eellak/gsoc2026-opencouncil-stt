"""Pure helpers for the Whisper hyperparameter sweep.

No torch / transformers imports here on purpose: this module must import and run
on a laptop so the leaderboard logic can be unit-tested without a GPU. The Kaggle
notebook inlines a copy of these functions (Kaggle kernels are self-contained).
"""
import random


def make_grid(lrs, ranks, seed):
    """Full Cartesian product of learning rates x LoRA ranks.

    alpha is fixed at 2 * rank. config_id is stable and human-readable.
    """
    grid = []
    for lr in lrs:
        for rank in ranks:
            grid.append({
                "config_id": f"lr{lr:g}_r{rank}",
                "lr": lr,
                "rank": rank,
                "alpha": 2 * rank,
                "seed": seed,
            })
    return grid


def subsample(records, n, seed):
    """Deterministic subsample of at most n records, preserving original order.

    Returns the full list (new list) when n is None or >= len(records).
    """
    records = list(records)
    if n is None or len(records) <= n:
        return records
    rnd = random.Random(seed)
    idx = sorted(rnd.sample(range(len(records)), n))
    return [records[i] for i in idx]


def build_leaderboard(rows, baseline):
    """Return a new list of rows sorted by val_corr_wer_norm ascending, each with
    reg_delta = val_reg_wer - baseline['val_reg_wer'] (positive = regression).

    Input rows are not mutated.
    """
    out = []
    for r in rows:
        rr = dict(r)
        rr["reg_delta"] = round(r["val_reg_wer"] - baseline["val_reg_wer"], 3)
        out.append(rr)
    out.sort(key=lambda x: x["val_corr_wer_norm"])
    return out


def pick_best(sorted_rows, max_reg_delta=1.0):
    """Lowest val_corr_wer_norm row whose reg_delta <= max_reg_delta.

    Returns None if every row regresses val_reg beyond the threshold.
    """
    for r in sorted_rows:
        if r["reg_delta"] <= max_reg_delta:
            return r
    return None


_COLS = ["config_id", "lr", "rank", "alpha", "epoch",
         "val_corr_wer_norm", "val_reg_wer", "reg_delta", "val_corr_cer",
         "train_loss", "wall_s"]


def render_markdown(sorted_rows, best):
    """Render the leaderboard as a Markdown table plus a best-pick line."""
    header = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    lines = [header, sep]
    for r in sorted_rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in _COLS) + " |")
    best_line = (f"**Best (regression-guarded):** {best['config_id']} "
                 f"(epoch {best['epoch']}, val_corr_wer_norm {best['val_corr_wer_norm']}, "
                 f"reg_delta {best['reg_delta']})"
                 if best else
                 "**Best (regression-guarded):** none — every config regressed val_reg")
    return "\n".join(lines) + "\n\n" + best_line + "\n"
