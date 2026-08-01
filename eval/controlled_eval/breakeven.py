#!/usr/bin/env python3
"""Is a blanket LLM post-editor net positive on a real meeting?

`exp_postedit_gate.py` measures both sides of the trade:
  gain  — WER recovered on utterances that DO need correction
  cost  — WER introduced on utterances that do NOT

Whether running it over everything helps therefore depends on one number nobody had
looked up: what fraction of utterances in a real meeting actually need correction.

That number is already in the repo. `data/reports/meeting-edit-fraction/distribution.tsv`
records, for 327 cached meeting JSONs, how many utterances a human edited
(`lastModifiedBy` set) out of the total. It was built for a different purpose — the
trust cutoff that produced the unreviewed-meetings denylist — but it is the same
quantity.

Run: python eval/controlled_eval/breakeven.py
"""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "data/reports/meeting-edit-fraction/distribution.tsv"
DENY = ROOT / "data/exclusions/unreviewed_meetings.json"
RESULTS = Path(__file__).with_name("results_postedit_gate.json")

# Experiment A (2026-07-29): 8.8% of corrections are pure formatting, which the WER
# normalizer erases. Those utterances were "edited" but needed no WER-relevant fix.
FORMATTING_ONLY = 0.088


def main():
    r = json.loads(RESULTS.read_text())
    gain = r["A_scribe"]["wer_source"] - r["A_scribe"]["wer_gated"]
    costs = {"held-out sample": r["B_clean_held"]["wer_gated"],
             "training-city sample": r["B_clean_train"]["wer_gated"]}

    deny = {(m["city_id"], m["meeting_id"])
            for m in json.loads(DENY.read_text())["meetings"]}
    with open(DIST) as f:
        rows = [x for x in csv.DictReader(f, delimiter="\t")
                if (x["city"], x["meeting"]) not in deny]
    total = sum(int(x["n_utt"]) for x in rows)
    edited = sum(int(x["n_user"]) for x in rows)
    micro = edited / total
    fracs = sorted(float(x["frac_user"]) for x in rows)
    adj = micro * (1 - FORMATTING_ONLY)

    print(f"meetings (after denylist): {len(rows)}   utterances: {total:,}")
    print(f"human-edited: {edited:,} = {micro:.1%} micro")
    print(f"per-meeting: p10 {fracs[len(fracs)//10]:.1%}  median "
          f"{statistics.median(fracs):.1%}  p90 {fracs[9*len(fracs)//10]:.1%}")
    print(f"WER-relevant (excluding formatting-only edits): {adj:.1%}\n")
    print(f"gain on utterances needing correction: {gain:.4f} / reference word")
    for name, cost in costs.items():
        be = cost / (gain + cost)
        below = sum(1 for f in fracs if f * (1 - FORMATTING_ONLY) < be)
        print(f"  cost {cost:.4f} ({name}) -> break-even {be:.1%} | "
              f"margin {adj - be:+.1%} | {below}/{len(rows)} meetings "
              f"({below/len(rows):.0%}) fall below it")


if __name__ == "__main__":
    main()
