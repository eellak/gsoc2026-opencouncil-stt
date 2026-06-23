"""Per-meeting human-edit FRACTION distribution → trust cutoff.

Replaces relying on the `taskStatus.humanReview` flag alone (mentor 2026-06-23:
the flag is unreliable — old/2025 corrected meetings often lack it). We compute,
from the cached meeting JSON, per meeting:
  - n_utt           total utterances
  - n_user          utterances whose final edit was a human (lastModifiedBy=='user')
  - n_task          fix-task was the last editor
  - frac_user       n_user / n_utt   <-- the human-intervention fraction (HIR-like)
and the humanReview flag, so we can pick a fraction cutoff and see how well the
flag agrees with it.

Pure local: reads ui/.cache/meetings/*.json. No network, no quota.

Output: data/reports/meeting-edit-fraction/{distribution.tsv,summary.md}
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "ui" / ".cache" / "meetings"
OUT = ROOT / "data" / "reports" / "meeting-edit-fraction"


def meetings():
    for f in sorted(glob.glob(str(CACHE / "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print("skip", os.path.basename(f), e)
            continue
        ts = d.get("taskStatus") or {}
        m = d.get("meeting") or {}
        city = (d.get("city") or {}).get("id") or m.get("cityId")
        mid = m.get("id") or os.path.basename(f).split("__")[-1].replace(".json", "")
        n_utt = n_user = n_task = n_none = 0
        n_seg = n_seg_user = 0
        for seg in d.get("transcript") or []:
            utts = seg.get("utterances") or []
            if not utts:
                continue
            n_seg += 1
            seg_has_user = False
            for u in utts:
                n_utt += 1
                lm = u.get("lastModifiedBy")
                if lm == "user":
                    n_user += 1
                    seg_has_user = True
                elif lm == "task":
                    n_task += 1
                else:
                    n_none += 1
            if seg_has_user:
                n_seg_user += 1
        if n_utt == 0:
            continue
        yield {
            "city": city,
            "meeting": mid,
            "humanReview": bool(ts.get("humanReview")),
            "n_utt": n_utt,
            "n_user": n_user,
            "n_task": n_task,
            "n_none": n_none,
            "frac_user": n_user / n_utt,
            "n_seg": n_seg,
            "frac_seg_user": (n_seg_user / n_seg) if n_seg else 0.0,
        }


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    rows = sorted(meetings(), key=lambda r: r["frac_user"])
    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "distribution.tsv", "w", encoding="utf-8") as fh:
        cols = ["city", "meeting", "humanReview", "n_utt", "n_user", "n_task",
                "n_none", "frac_user", "n_seg", "frac_seg_user"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    n = len(rows)
    hr_true = [r for r in rows if r["humanReview"]]
    hr_false = [r for r in rows if not r["humanReview"]]

    # histogram of frac_user in 5-pt bins
    bins = {}
    for r in rows:
        b = int(r["frac_user"] * 100 // 5) * 5
        bins[b] = bins.get(b, 0) + 1

    # candidate cutoffs: how many meetings fall below, and flag agreement
    def below(thr, pool):
        return [r for r in pool if r["frac_user"] < thr]

    lines = []
    lines.append("# Per-meeting human-edit fraction — distribution & trust cutoff\n")
    lines.append(f"_Computed from {n} cached public meeting JSONs "
                 f"(`ui/.cache/meetings/`). frac_user = user-edited utterances / "
                 f"total utterances._\n")
    lines.append("## Headline\n")
    tot_utt = sum(r["n_utt"] for r in rows)
    tot_user = sum(r["n_user"] for r in rows)
    lines.append(f"- meetings: **{n}**  ·  humanReview=true: **{len(hr_true)}**  ·  "
                 f"false: **{len(hr_false)}**")
    lines.append(f"- micro human-edit fraction (all utts): "
                 f"**{pct(tot_user / tot_utt)}** ({tot_user:,}/{tot_utt:,})")
    fr = [r["frac_user"] for r in rows]
    fr_sorted = sorted(fr)
    med = fr_sorted[n // 2]
    p10 = fr_sorted[int(n * 0.1)]
    p90 = fr_sorted[int(n * 0.9)]
    lines.append(f"- per-meeting frac_user: median **{pct(med)}**, "
                 f"p10 {pct(p10)}, p90 {pct(p90)}\n")

    lines.append("## Histogram (frac_user, 5-pt bins)\n")
    lines.append("| bin | meetings |")
    lines.append("|---|---|")
    for b in sorted(bins):
        bar = "█" * bins[b]
        lines.append(f"| {b}-{b+5}% | {bins[b]:>3} {bar} |")
    lines.append("")

    lines.append("## humanReview flag vs fraction (is the flag reliable?)\n")
    for label, pool in (("humanReview=TRUE", hr_true), ("humanReview=FALSE", hr_false)):
        if not pool:
            continue
        f2 = sorted(r["frac_user"] for r in pool)
        lines.append(f"- **{label}** ({len(pool)}): frac_user "
                     f"min {pct(f2[0])}, median {pct(f2[len(f2)//2])}, "
                     f"max {pct(f2[-1])}")
    # disagreement: humanReview=false but high fraction (looks reviewed) and
    # humanReview=true but very low fraction (suspicious)
    false_high = [r for r in hr_false if r["frac_user"] >= 0.15]
    true_low = [r for r in hr_true if r["frac_user"] < 0.05]
    lines.append(f"- humanReview=FALSE yet frac_user ≥15% (flag likely wrong, "
                 f"looks reviewed): **{len(false_high)}**")
    lines.append(f"- humanReview=TRUE yet frac_user <5% (suspicious / barely "
                 f"touched): **{len(true_low)}**\n")

    lines.append("## Candidate cutoffs (drop meetings below the fraction)\n")
    lines.append("| cutoff | meetings kept | dropped | kept that are humanReview=false |")
    lines.append("|---|---|---|---|")
    for thr in (0.03, 0.05, 0.08, 0.10, 0.15):
        kept = [r for r in rows if r["frac_user"] >= thr]
        kept_false = [r for r in kept if not r["humanReview"]]
        lines.append(f"| ≥{int(thr*100)}% | {len(kept)} | {n-len(kept)} | "
                     f"{len(kept_false)} |")
    lines.append("")

    lines.append("## Lowest-fraction meetings (cut candidates)\n")
    lines.append("| city | meeting | humanReview | n_utt | n_user | frac_user |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows[:15]:
        lines.append(f"| {r['city']} | {r['meeting']} | {r['humanReview']} | "
                     f"{r['n_utt']} | {r['n_user']} | {pct(r['frac_user'])} |")
    lines.append("")

    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'summary.md'} and distribution.tsv")


if __name__ == "__main__":
    main()
