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
