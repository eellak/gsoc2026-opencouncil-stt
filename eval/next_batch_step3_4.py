"""Next-batch Steps 3 & 4 — faithfulness metric, bucketing, and the human gate.

Runbook Steps 3-4. For every calibration item with a Soniox transcript:
  cer_before = CER(input_raw, gold_final)   # how much the human changed the STT
  cer_soniox = CER(soniox_text, gold_final) # how far a fresh ASR is from the label
Both under greek_normalize. Emit a BUCKET (not a keep/drop boolean), plus aux
signals (word-rate / duration guards, short-utterance path). The STARTING CER
gates below are Codex's first guesses — Step 4 is a HUMAN GATE: Angelos hand-audits
calib_audit.csv and the distribution plot, then the audited cut wins.

Outputs under data/next-batch/:
  calib/calib_scored.parquet    every item with both CERs, aux, bucket
  calib/calib_audit.csv         human-readable audit sheet (sorted for review)
  calib/cer_distributions.png   cer_before vs cer_soniox scatter + histograms
  step3_4_summary.md            bucket counts + what each gate means
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eval.scoring import cer, greek_normalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "next-batch"
CALIB = OUT_DIR / "calib"

# ---- STARTING gates (Codex guesses; the human audit overrides these) ----------
G = dict(
    backbone_before=0.03, backbone_soniox=0.10,
    drop_before=0.08, drop_soniox=0.18, drop_ratio=2.5, drop_floor=0.04,
    keep_before=0.12, keep_soniox=0.15,
    audit_before=0.15, audit_soniox=0.15,
    rate_lo=1.0, rate_hi=4.5, clip_lo=1.5, clip_hi=30.0,
    short_chars=20, short_words=5,
)


def _short_special(gold: str) -> bool:
    n = greek_normalize(gold)
    return len(n) < G["short_chars"] or len(n.split()) < G["short_words"]


def assign_bucket(cb: float, cs: float, short: bool) -> str:
    """Priority-ordered bucketing per runbook Step 3. SHORT-SPECIAL is routed out
    first (plain CER is unstable on <5 words / <20 chars)."""
    if short:
        return "SHORT-SPECIAL"
    # no-edit backbone is essentially absent from a human-corrected pool (cb>0), but
    # keep the rule so the same scorer works on the no-edit backbone pool later.
    if cb <= G["backbone_before"] and cs <= G["backbone_soniox"]:
        return "BACKBONE"
    if (cb <= G["drop_before"] and cs >= G["drop_soniox"]
            and cs >= G["drop_ratio"] * max(cb, G["drop_floor"])):
        return "DROP"
    if cb >= G["keep_before"] and cs <= G["keep_soniox"]:
        return "KEEP"
    if cb >= G["audit_before"] and cs >= G["audit_soniox"]:
        return "AUDIT"
    return "REVIEW"  # falls between the starting gates — exactly what the human sets


def main() -> None:
    man = pd.read_csv(CALIB / "transcribe_manifest.csv")
    cand = pd.read_parquet(OUT_DIR / "candidates.parquet")[
        ["utterance_id", "input_raw", "gold_final", "category", "ebclass",
         "n_edits", "char_diff", "norm_word_diff"]
    ]
    df = man.merge(cand, on="utterance_id", how="left")

    n_total = len(df)
    ok = df[df["asr_ok"] == True].copy()  # noqa: E712
    n_fail = n_total - len(ok)

    ok["cer_before"] = [cer(i, g) for i, g in zip(ok["input_raw"], ok["gold_final"])]
    ok["cer_soniox"] = [cer(s, g) for s, g in zip(ok["soniox_text"].fillna(""), ok["gold_final"])]
    ok["short_special"] = [_short_special(g) for g in ok["gold_final"]]

    norm_words = ok["gold_final"].map(lambda g: len(greek_normalize(g).split()))
    ok["word_rate"] = norm_words / ok["duration"].replace(0, float("nan"))
    ok["rate_flag"] = (ok["word_rate"] < G["rate_lo"]) | (ok["word_rate"] > G["rate_hi"])
    ok["clip_flag"] = (ok["duration"] < G["clip_lo"]) | (ok["duration"] > G["clip_hi"])
    ok["guard_flag"] = ok["rate_flag"] | ok["clip_flag"]

    ok["bucket"] = [
        assign_bucket(cb, cs, sh)
        for cb, cs, sh in zip(ok["cer_before"], ok["cer_soniox"], ok["short_special"])
    ]

    ok.to_parquet(CALIB / "calib_scored.parquet", index=False)

    # ---- audit sheet: ordered so the human reviews the decisive cases first ----
    audit_cols = ["utterance_id", "city_id", "meeting_id", "category", "ebclass",
                  "n_edits", "duration", "word_rate", "guard_flag", "short_special",
                  "cer_before", "cer_soniox", "bucket", "clip_path",
                  "input_raw", "soniox_text", "gold_final"]
    audit = ok[audit_cols].copy()
    audit = audit.round({"duration": 2, "word_rate": 2, "cer_before": 3, "cer_soniox": 3})
    # decisive-first: DROP and AUDIT and REVIEW before KEEP; within, by cer_soniox desc
    order = {"DROP": 0, "AUDIT": 1, "REVIEW": 2, "SHORT-SPECIAL": 3, "KEEP": 4, "BACKBONE": 5}
    audit["_o"] = audit["bucket"].map(order)
    audit = audit.sort_values(["_o", "cer_soniox"], ascending=[True, False]).drop(columns="_o")
    audit.to_csv(CALIB / "calib_audit.csv", index=False)

    # ---- distribution plot ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"DROP": "#d62728", "AUDIT": "#ff7f0e", "REVIEW": "#7f7f7f",
              "KEEP": "#2ca02c", "SHORT-SPECIAL": "#1f77b4", "BACKBONE": "#9467bd"}
    for b, g in ok.groupby("bucket"):
        ax[0].scatter(g["cer_before"], g["cer_soniox"], s=18, alpha=0.6,
                      label=f"{b} ({len(g)})", c=colors.get(b, "#000"))
    ax[0].axhline(G["keep_soniox"], ls="--", c="green", lw=0.8)
    ax[0].axvline(G["keep_before"], ls="--", c="green", lw=0.8)
    ax[0].axhline(G["drop_soniox"], ls="--", c="red", lw=0.8)
    ax[0].set_xlabel("cer_before (human change vs STT)")
    ax[0].set_ylabel("cer_soniox (fresh ASR vs label)")
    ax[0].set_title("Faithfulness scatter (starting gates dashed)")
    ax[0].legend(fontsize=7)
    ax[1].hist(ok["cer_soniox"], bins=40, color="#ff7f0e")
    ax[1].axvline(G["keep_soniox"], ls="--", c="green"); ax[1].axvline(G["drop_soniox"], ls="--", c="red")
    ax[1].set_title("cer_soniox distribution"); ax[1].set_xlabel("cer_soniox")
    ax[2].hist(ok["cer_before"], bins=40, color="#2ca02c")
    ax[2].axvline(G["keep_before"], ls="--", c="green")
    ax[2].set_title("cer_before distribution"); ax[2].set_xlabel("cer_before")
    fig.tight_layout()
    fig.savefig(CALIB / "cer_distributions.png", dpi=110)

    # ---- summary ----
    lines = ["# Steps 3-4 — faithfulness buckets + human gate\n"]
    lines.append(f"Calibration items: {n_total} ({len(ok)} transcribed, {n_fail} failed/excluded)\n")
    lines.append("## Bucket counts (STARTING gates — NOT final)\n")
    for b, c in ok["bucket"].value_counts().items():
        lines.append(f"- {b}: {c}")
    lines.append(f"\n- guard-flagged (rate/clip): {int(ok['guard_flag'].sum())}")
    lines.append(f"- short-special: {int(ok['short_special'].sum())}")
    lines.append("\n## CER summary\n```")
    lines.append("cer_before:\n" + ok["cer_before"].describe().to_string())
    lines.append("\ncer_soniox:\n" + ok["cer_soniox"].describe().to_string())
    lines.append("```")
    lines.append("\n## Starting gates (Codex guesses — Angelos overrides at the gate)\n```")
    for k, v in G.items():
        lines.append(f"{k} = {v}")
    lines.append("```")
    lines.append(
        "\n## HUMAN GATE\n\n"
        "These buckets use the **starting** CER gates. Hand-audit "
        "`calib/calib_audit.csv` (DROP/AUDIT/REVIEW rows first), listen to the "
        "clips under `calib/clips/`, and read `calib/cer_distributions.png`. Pick "
        "the cut where good vs bad separate, then record the chosen thresholds in "
        "the runbook and `docs/decisions/data.md`. Do NOT scale to full bulk "
        "(Step 5) until the thresholds are locked.\n"
    )
    (OUT_DIR / "step3_4_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote calib_scored.parquet, calib_audit.csv, cer_distributions.png, step3_4_summary.md")


if __name__ == "__main__":
    main()
