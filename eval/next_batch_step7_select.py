"""Next-batch Step 7 — assemble the final fine-tune edit list.

Runbook Step 7. Combines the free-ranked diverse shortlist with the Sonnet
text-plausibility verdicts, keeps the genuine acoustic corrections, and runs a
final greedy-diverse selection with soft category ceilings (Codex bands) to a
target size. Emits the list + stats. Faithfulness (Soniox cer) is NOT yet applied
here — that is the paid gold-upgrade (prep in step5_soniox_bulk); the exact
faithfulness cut stays Angelos's gate.

Outputs:
  data/next-batch/selected_edits.jsonl   the final list (rich rows)
  data/next-batch/step7_summary.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "next-batch"

# Codex soft category ceilings (share of the final set)
CEIL = {
    "named_entity": 0.40, "other_lexical": 0.35, "homophone": 0.25,
    "word_boundary": 0.25, "insertion_deletion": 0.20, "morph_grammar": 0.08,
    "number_date": 0.05, "acronym_abbreviation": 0.05,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=8000)
    ap.add_argument("--include-unsure", action="store_true",
                    help="also allow verdict=unsure with acoustic=true (secondary tier)")
    args = ap.parse_args()

    short = pd.read_parquet(OUT_DIR / "shortlist.parquet")
    judged = pd.read_json(OUT_DIR / "judged.jsonl", lines=True)
    df = short.merge(judged[["utterance_id", "verdict", "acoustic", "why"]],
                     on="utterance_id", how="inner")

    # drop audio-confirmed bad labels found by the faithfulness verification
    bad_path = OUT_DIR / "verified_bad_ids.json"
    n_bad = 0
    if bad_path.exists():
        bad = set(json.loads(bad_path.read_text()))
        n_bad = int(df["utterance_id"].isin(bad).sum())
        df = df[~df["utterance_id"].isin(bad)].copy()

    log = ["# Step 7 — final fine-tune edit list\n"]
    log.append(f"- shortlist: {len(short):,}")
    log.append(f"- judged so far: {len(judged):,}")
    log.append(f"- shortlist ∩ judged: {len(df):,} (dropped {n_bad} audio-confirmed bad)")
    vc = df["verdict"].value_counts().to_dict()
    log.append(f"- verdicts: {vc}")

    # eligible = kept genuine corrections (acoustic preferred); optional unsure tier
    keep = df[(df["verdict"] == "keep")].copy()
    if args.include_unsure:
        unsure = df[(df["verdict"] == "unsure") & (df["acoustic"])].copy()
        keep = pd.concat([keep, unsure])
    log.append(f"- eligible (keep{'+unsure-acoustic' if args.include_unsure else ''}): {len(keep):,}")
    log.append(f"  of which acoustic=true: {int(keep['acoustic'].sum()):,}")

    # rank eligible: acoustic first, then base_score
    keep = keep.sort_values(["acoustic", "base_score"], ascending=[False, False]).reset_index(drop=True)

    # greedy-diverse with soft category ceilings
    rows = keep.to_dict("records")
    c1 = c2 = None
    cnt_sig1: dict = {}
    cnt_sig2: dict = {}
    cnt_sig3: dict = {}
    cnt_cat: dict = {}
    cnt_city: dict = {}
    cnt_meet: dict = {}
    N = min(args.target, len(rows))
    picked = []

    def score(r):
        cat = r["category"]
        cat_share = cnt_cat.get(cat, 0) / max(1, len(picked))
        # soft ceiling: steep penalty once a category exceeds its band
        ceil = CEIL.get(cat, 0.15)
        over = max(0.0, cat_share - ceil)
        cat_pen = 1.0 / (1.0 + 8.0 * over)
        acoustic_boost = 1.15 if r["acoustic"] else 1.0
        return (
            r["base_score"] * acoustic_boost * cat_pen
            / math.sqrt(1 + cnt_sig1.get(r["sig1"], 0))
            / math.sqrt(1 + cnt_sig2.get(r["sig2"], 0))
            / math.sqrt(1 + cnt_sig3.get(r["sig3"], 0))
            / math.sqrt(1 + 0.25 * cnt_city.get(r["city_id"], 0))
            / math.sqrt(1 + 0.10 * cnt_meet.get(r["meeting_id"], 0))
        )

    remaining = list(range(len(rows)))
    while remaining and len(picked) < N:
        best_i = max(remaining, key=lambda i: score(rows[i]))
        remaining.remove(best_i)
        r = rows[best_i]
        picked.append(best_i)
        for d, k in ((cnt_sig1, "sig1"), (cnt_sig2, "sig2"), (cnt_sig3, "sig3"),
                     (cnt_cat, "category"), (cnt_city, "city_id"), (cnt_meet, "meeting_id")):
            d[r[k]] = d.get(r[k], 0) + 1

    final = keep.iloc[picked].copy()
    final["final_rank"] = range(1, len(final) + 1)

    cols = ["utterance_id", "city_id", "meeting_id", "audio_url",
            "utterance_start", "utterance_end", "duration",
            "input_raw", "gold_final", "category", "ebclass", "n_edits",
            "char_diff", "norm_word_diff", "cer_proxy", "base_score",
            "acoustic", "why", "sig2", "sig3", "final_rank"]
    cols = [c for c in cols if c in final.columns]
    out = OUT_DIR / "selected_edits.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for _, r in final[cols].iterrows():
            f.write(json.dumps({c: (r[c].item() if hasattr(r[c], "item") else r[c])
                                for c in cols}, ensure_ascii=False) + "\n")

    hrs = final["duration"].sum() / 3600
    log.append(f"\n**Final selected: {len(final):,} edits -> `{out.relative_to(ROOT)}`**")
    log.append(f"- raw correction-span audio: {hrs:.1f} h "
               f"(concatenation to 15-30s speaker-turn segments expands this toward the ~30h target)")
    log.append(f"- distinct meetings: {final['meeting_id'].nunique()}, "
               f"cities: {final['city_id'].nunique()}, distinct sig3: {final['sig3'].nunique():,}")
    log.append("\n## Category mix (final)\n")
    for k, v in final["category"].value_counts().items():
        log.append(f"- {k}: {v:,} ({v/len(final)*100:.1f}%)")
    log.append(f"\n- acoustic=true: {int(final['acoustic'].sum()):,} "
               f"({final['acoustic'].mean()*100:.1f}%)")
    log.append("\n## Duration buckets\n")
    b = pd.cut(final["duration"], [0, 1.5, 3, 5, 10, 15, 25, 30],
               labels=["<1.5", "1.5-3", "3-5", "5-10", "10-15", "15-25", "25-30"])
    for k, v in b.value_counts().sort_index().items():
        log.append(f"- {k}s: {v:,}")
    log.append("\n## ebclass\n")
    for k, v in final["ebclass"].value_counts().items():
        log.append(f"- {k}: {v:,}")

    (OUT_DIR / "step7_summary.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"\nwrote {out} ({len(final):,}) and step7_summary.md")


if __name__ == "__main__":
    main()
