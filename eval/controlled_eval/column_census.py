#!/usr/bin/env python3
"""How many columns could arms C and H even touch? Counted before either is built.

The lesson of wayfinder #18 is that an arbiter can be neutral simply because it was
handed 2.6% of the decisions. So this runs first: classify every column of the frozen
3-way alignment by `column_classes.py`, count the classes, and price each class by how
often W's vote differs from the alignment-conditional column oracle inside it. If the
eligible classes are a rounding error, the arms are not worth building and this script
is the answer.

PRICING is DESCRIPTIVE ONLY: "W differs from the column oracle" reads the reference
text. No threshold may be fitted on it. It exists to answer one question - is there
recoverable mass inside the eligible classes, or is the residual somewhere an arm
restricted to these columns can never reach?

Writes results_column_census.json (counts only, never transcript text).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.column_classes import (                     # noqa: E402
    column_class, column_flags, eligibility, split_merge_columns,
)
from eval.controlled_eval.fusion_lab import load_substrate, log       # noqa: E402
from eval.controlled_eval.msa import oracle_select                    # noqa: E402

OUT = Path(__file__).with_name("results_column_census.json")


def main():
    sub = load_substrate()
    log(json.dumps(sub.meta, indent=1))

    n_class = Counter()
    n_flag = Counter()                 # (class, flag) -> n
    w_wrong = Counter()                # class -> columns where W != oracle choice
    w_wrong_flag = Counter()
    elig = Counter()                   # arm -> eligible columns
    elig_wrong = Counter()             # arm -> eligible columns where W != oracle
    elig_loose = Counter()
    sm_total = 0
    sm_blocked = Counter()
    per_city = {}
    windows_with = Counter()

    for w in sub.windows:
        orc_choice = oracle_select(w.cols, w.ref)
        sm = split_merge_columns(w.cols)
        sm_total += len(sm)
        e_strict = eligibility(w.cols, loose=False)
        e_loose = eligibility(w.cols, loose=True)
        seen = set()
        for n, col in enumerate(w.cols):
            klass = column_class(col)
            n_class[klass] += 1
            seen.add(klass)
            per_city.setdefault(w.city, Counter())[klass] += 1
            f = column_flags(col)
            wrong = w.decisions[n]["token"] != orc_choice[n]
            if wrong:
                w_wrong[klass] += 1
            if klass in ("unresolved_two", "unresolved_three"):
                for key in ("strict_homophone", "loose_homophone",
                            "partial_homophone"):
                    if f[key]:
                        n_flag[(klass, key)] += 1
                        if wrong:
                            w_wrong_flag[(klass, key)] += 1
                n_flag[(klass, f"chardist_{min(f['max_char_dist'], 3)}")] += 1
                if n in sm:
                    sm_blocked[klass] += 1
            arm = e_strict.get(n)
            if arm:
                elig[arm] += 1
                elig[f"{arm}:{klass}"] += 1
                if wrong:
                    elig_wrong[arm] += 1
                    elig_wrong[f"{arm}:{klass}"] += 1
            if e_loose.get(n) == "H":
                elig_loose["H_loose"] += 1
                if wrong:
                    elig_loose["H_loose_wrong"] += 1
        for c in seen:
            windows_with[c] += 1

    total = sum(n_class.values())
    res = {
        "substrate": sub.meta,
        "n_columns": total,
        "classes": {k: {"n": v, "share": v / total,
                        "windows_containing": windows_with[k],
                        "w_differs_from_column_oracle": w_wrong[k],
                        "w_differs_share_of_class": w_wrong[k] / v if v else None}
                    for k, v in n_class.most_common()},
        "flags_within_unresolved": {f"{c}:{f}": v for (c, f), v in sorted(n_flag.items())},
        "flags_w_differs": {f"{c}:{f}": v for (c, f), v in sorted(w_wrong_flag.items())},
        "split_merge": {"columns_flagged": sm_total,
                        "share_of_all": sm_total / total,
                        "unresolved_columns_blocked": dict(sm_blocked)},
        "eligible": {k: v for k, v in sorted(elig.items())},
        "eligible_w_differs_from_oracle": {k: v for k, v in sorted(elig_wrong.items())},
        "eligible_loose_variant": dict(elig_loose),
        "per_city_class_shares": {city: {k: v / sum(cnt.values())
                                         for k, v in cnt.items()}
                                  for city, cnt in per_city.items()},
        "note": "w_differs_from_column_oracle READS THE REFERENCE and is descriptive "
                "only. Eligibility is the frozen rule of column_classes.eligibility.",
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")
    log(f"{total} columns")
    for k, v in n_class.most_common():
        log(f"  {k:18s} {v:7d}  {v / total:6.2%}   W!=oracle {w_wrong[k]:6d}")
    log(f"  split_merge flagged {sm_total} ({sm_total / total:.2%})")
    for k, v in sorted(elig.items()):
        log(f"  eligible {k:22s} {v:6d}  ({v / total:.2%})  "
            f"W!=oracle {elig_wrong.get(k, 0)}")
    log(f"  eligible loose-H {dict(elig_loose)}")


if __name__ == "__main__":
    main()
