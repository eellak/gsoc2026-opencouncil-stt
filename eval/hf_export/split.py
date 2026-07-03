"""Seeded speaker split: held-out cities + whole speakers to ~20% of hours.

Recipe (mentor sync 2026-06-23 + val-split report 2026-06-24, spec
docs/specs/hf-dataset-export.md): validation = all rows of the held-out
cities, then whole seeded speakers from the train cities (>= floor_s speech
within the dataset) until validation reaches target_frac of total hours.
Speaker-disjoint by construction; null-speaker rows never go to validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VAL_CITIES = frozenset({"orestiada", "argos"})
DEFAULT_SEED = 20260703


class SplitError(RuntimeError):
    """Validation share landed outside the acceptable window — human call."""


@dataclass
class SplitResult:
    splits: list[str]              # per input row: "train" | "validation"
    val_speakers: set[str]         # train-city speakers moved to validation
    val_share: float               # validation hours / total hours
    skipped_speakers: list[str] = field(default_factory=list)  # would overshoot
    speaker_durations: dict[str, float] = field(default_factory=dict)


def assign_splits(rows: list[dict], *, val_cities: frozenset = VAL_CITIES,
                  seed: int = DEFAULT_SEED, target_frac: float = 0.20,
                  floor_s: float = 180.0,
                  window: tuple[float, float] = (0.18, 0.22)) -> SplitResult:
    total = sum(r["duration_s"] for r in rows)
    if total <= 0:
        raise SplitError("empty dataset")

    val_s = sum(r["duration_s"] for r in rows if r["city_id"] in val_cities)

    spk_dur: dict[str, float] = {}
    for r in rows:
        spk = r.get("speaker_id")
        if r["city_id"] in val_cities or not spk:
            continue
        spk_dur[spk] = spk_dur.get(spk, 0.0) + r["duration_s"]

    eligible = sorted(s for s, d in spk_dur.items() if d >= floor_s)
    rng = np.random.default_rng(seed)
    order = [eligible[i] for i in rng.permutation(len(eligible))]

    chosen: set[str] = set()
    skipped: list[str] = []
    cur = val_s
    for spk in order:
        if cur / total >= target_frac:
            break
        if (cur + spk_dur[spk]) / total > window[1]:
            skipped.append(spk)
            continue
        chosen.add(spk)
        cur += spk_dur[spk]

    share = cur / total
    if not (window[0] <= share <= window[1]):
        raise SplitError(
            f"validation share {share:.1%} outside window "
            f"[{window[0]:.0%}, {window[1]:.0%}] — stop for a human decision "
            f"(held-out cities alone: {val_s / total:.1%})")

    splits = [
        "validation" if (r["city_id"] in val_cities
                         or (r.get("speaker_id") or "") in chosen)
        else "train"
        for r in rows
    ]
    return SplitResult(splits=splits, val_speakers=chosen, val_share=share,
                       skipped_speakers=skipped, speaker_durations=spk_dur)
