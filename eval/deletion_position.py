"""Where inside a window do the deletions fall?

Codex's discriminating test for the "premature EOS" story: if training on
edge-clipped ~3 s segments taught the model to stop early, its deletions should
pile up at the END of each decoded window. The preregistered bar, set before
looking: the last 15% of the reference must carry at least 2x enrichment.

Descriptive. Two arms on the frozen 39-window harness, same CPU stack, same
normalizer as every other number in this project:
  control  artifact-adapter-fixed      (~/.cache/oc-public/decode-ablation/eval-A.json)
  RUN2     RUN2 stage-2, seed 101      (~/.cache/oc-public/train-screens-2026-08/
                                        run2-eval-stage2/decode.json)

Deletion positions come from a standard edit-distance backtrace, so a deletion is
attributed to the reference index it actually falls on rather than to a block.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402

import decode_ablation as DA  # noqa: E402

SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
ARMS = {
    "control": SC / "decode-ablation/eval-A.json",
    "RUN2": SC / "train-screens-2026-08/run2-eval-stage2/decode.json",
}
LAST_FRACTION = 0.15          # frozen before looking
ENRICHMENT_BAR = 2.0          # frozen before looking
N_BINS = 10


def deletion_indices(ref: list[str], hyp: list[str]) -> list[int]:
    """Reference indices consumed by a DELETE in a minimum edit-distance path.

    Ties are broken toward substitution, then deletion, then insertion — one fixed
    order, so the answer does not drift between arms.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)

    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append(i - 1)
            i -= 1
            continue
        j -= 1
    return out


def main() -> None:
    frozen = [r for r in DA.rows("eval")]
    refs = {r["window_id"]: ftoks(DA.reference_text(r["window_id"])) for r in frozen}

    report = {"last_fraction": LAST_FRACTION, "enrichment_bar": ENRICHMENT_BAR,
              "n_windows": len(frozen), "arms": {}}

    for arm, path in ARMS.items():
        if not path.exists():
            report["arms"][arm] = {"error": f"decode cache missing: {path}"}
            continue
        state = json.loads(path.read_text())["windows"]
        bins = [0] * N_BINS
        tail = ref_tail = total_del = total_ref = 0
        per_window = []
        for r in frozen:
            wid = r["window_id"]
            if wid not in state:
                continue
            ref = refs[wid]
            if not ref:
                continue
            hyp = ftoks(state[wid]["text"])
            idx = deletion_indices(ref, hyp)
            total_del += len(idx)
            total_ref += len(ref)
            cut = (1.0 - LAST_FRACTION) * len(ref)
            in_tail = sum(1 for i in idx if i >= cut)
            tail += in_tail
            ref_tail += sum(1 for i in range(len(ref)) if i >= cut)
            for i in idx:
                bins[min(N_BINS - 1, int(N_BINS * i / len(ref)))] += 1
            per_window.append({"window_id": wid, "ref": len(ref),
                               "del": len(idx), "del_tail": in_tail})

        # enrichment = share of deletions in the tail / share of reference in the tail
        share_del = tail / total_del if total_del else 0.0
        share_ref = ref_tail / total_ref if total_ref else 0.0
        enr = share_del / share_ref if share_ref else 0.0
        report["arms"][arm] = {
            "windows_scored": len(per_window),
            "ref_tokens": total_ref, "deletions": total_del,
            "deletion_rate": total_del / total_ref if total_ref else 0.0,
            "deletions_in_tail": tail,
            "share_of_deletions_in_tail": share_del,
            "share_of_reference_in_tail": share_ref,
            "tail_enrichment": enr,
            "verdict": ("premature-EOS story SUPPORTED" if enr >= ENRICHMENT_BAR
                        else "premature-EOS story NOT SUPPORTED"),
            "decile_counts": bins,
        }

    a, b = report["arms"].get("control", {}), report["arms"].get("RUN2", {})
    if "deletions" in a and "deletions" in b:
        report["extra_deletions_RUN2_minus_control"] = b["deletions"] - a["deletions"]

    out = SC / "deletion-position-2026-08"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, ensure_ascii=False))
    for arm, v in report["arms"].items():
        print(f"\n=== {arm} ===")
        print(json.dumps(v, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
