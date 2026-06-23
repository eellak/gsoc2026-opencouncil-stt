"""Render the per-meeting human-edit fraction distribution as a shareable PNG.

Reads data/reports/meeting-edit-fraction/distribution.tsv (from
eval/meeting_edit_fraction.py) and draws a stacked histogram by humanReview flag,
with the proposed cutoffs. Output PNG in the same folder.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "reports" / "meeting-edit-fraction"

rows = list(csv.DictReader(open(D / "distribution.tsv", encoding="utf-8"), delimiter="\t"))
true_f = [float(r["frac_user"]) * 100 for r in rows if r["humanReview"] == "True"]
false_f = [float(r["frac_user"]) * 100 for r in rows if r["humanReview"] == "False"]
n = len(rows)

bins = list(range(0, 105, 5))
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=140)

c_true, c_false = "#2e7d32", "#e65100"
ax.hist([true_f, false_f], bins=bins, stacked=True,
        color=[c_true, c_false], edgecolor="white", linewidth=0.6)

# proposed cutoffs
ax.axvline(5, color="#b71c1c", linestyle="--", linewidth=1.8)
ax.axvline(15, color="#1565c0", linestyle="--", linewidth=1.8)
ax.text(5, ax.get_ylim()[1] * 0.96, "  κόψε junk <5%", color="#b71c1c",
        fontsize=10, va="top", ha="left", fontweight="bold")
ax.text(15, ax.get_ylim()[1] * 0.88, "  εμπιστεύσου backbone ≥15%", color="#1565c0",
        fontsize=10, va="top", ha="left", fontweight="bold")

ax.set_xlabel("Ποσοστό utterances που διόρθωσε άνθρωπος ανά meeting  (frac_user, %)",
              fontsize=11)
ax.set_ylabel("Αριθμός meetings", fontsize=11)
ax.set_title("Κατανομή ανθρώπινων διορθώσεων ανά meeting — "
             f"{n} public meetings\n"
             "Ο flag humanReview χάνει 95 reviewed meetings (frac ≥15% αλλά flag=false)",
             fontsize=12.5, fontweight="bold")
ax.set_xticks(range(0, 105, 10))
ax.grid(axis="y", alpha=0.25)

leg = [mpatches.Patch(color=c_true, label=f"humanReview = true ({len(true_f)})"),
       mpatches.Patch(color=c_false, label=f"humanReview = false ({len(false_f)})")]
ax.legend(handles=leg, fontsize=10, loc="upper right", framealpha=0.95)

# footnote with the headline finding
fig.text(0.012, 0.012,
         "Πηγή: ui/.cache/meetings (eval/meeting_edit_fraction.py).  "
         "humanReview=true: 0 false positives (min 9.6%).  "
         "humanReview=false: 95 meetings με ≥15% → στην ουσία reviewed, λείπει το flag.",
         fontsize=8, color="#555")

fig.tight_layout(rect=(0, 0.03, 1, 1))
out = D / "edit_fraction_distribution.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
