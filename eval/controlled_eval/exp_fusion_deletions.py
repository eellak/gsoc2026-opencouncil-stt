#!/usr/bin/env python3
"""Does the consensus vote buy its WER win by deleting, and does the fixed adapter vote better?

`exp-2026-08-02-asr-fusion` settled that combining systems beats the best single one by
1.45 points. It measured WER only. Two things it left open matter now:

  1. The vote keeps the hypothesis most similar to the other two. A short hypothesis that
     swallowed speech can win such a vote. If the vote lowers WER while raising the
     deletion rate it is the same trap RUN 1 fell into, and the win is partly an illusion.
  2. Its trio carried `oc-minipc-finetune`, the adapter trained through the label-prefix
     bug. The 2026-08-10 report scores the broken AND the corrected adapter on the same
     windows against the same references, so the third voter can be swapped in a paired
     design with nothing else moving.

Design frozen on wayfinder issue #16 before this ran. Deletion rate is reported beside
WER, not under it.

Writes results_fusion_deletions.json (aggregates only, no transcript text).

Env: SC (cache dir) N_BOOT (10000)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
from eval.controlled_eval import bench_data as B                      # noqa: E402
from eval.controlled_eval.scoring import (cluster_bootstrap, head2head,  # noqa: E402
                                          wtoks)

RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
BASE = "scribe-v2-clean"
SECOND = "soniox"
OLD_THIRD = "oc-minipc-finetune"
NEW_THIRD = "oc-runpod-fixed-2026-08-10"
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
OUT = Path(__file__).with_name("results_fusion_deletions.json")


def log(m):
    print(m, flush=True)


def sdi(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """Levenshtein with a backtrace, returning (S, D, I, N_ref).

    scoring.py's edist gives the distance only; the whole point here is the split.
    """
    a, b = wtoks(ref), wtoks(hyp)
    n, m = len(a), len(b)
    # d[i][j] = cost, op[i][j] = one of 'M','S','D','I'
    d = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0], op[i][0] = i, "D"
    for j in range(1, m + 1):
        d[0][j], op[0][j] = j, "I"
    for i in range(1, n + 1):
        ai = a[i - 1]
        di, dim1 = d[i], d[i - 1]
        opi = op[i]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                di[j], opi[j] = dim1[j - 1], "M"
                continue
            sub, dele, ins = dim1[j - 1] + 1, dim1[j] + 1, di[j - 1] + 1
            best = min(sub, dele, ins)
            di[j] = best
            opi[j] = "S" if best == sub else ("D" if best == dele else "I")
    s = dd = ii = 0
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "M":
            i, j = i - 1, j - 1
        elif o == "S":
            s += 1
            i, j = i - 1, j - 1
        elif o == "D":
            dd += 1
            i -= 1
        else:
            ii += 1
            j -= 1
    return s, dd, ii, n


def rates(rows: list[tuple[int, int, int, int]]) -> dict:
    S = sum(r[0] for r in rows)
    D = sum(r[1] for r in rows)
    I = sum(r[2] for r in rows)
    N = sum(r[3] for r in rows)
    return {"S": S, "D": D, "I": I, "ref_tokens": N,
            "wer": (S + D + I) / N, "del_rate": D / N,
            "ins_rate": I / N, "sub_rate": S / N}


def main():
    report = B.load_report(RUN_ID)
    providers = B.provider_ids(report)
    for p in (BASE, SECOND, OLD_THIRD, NEW_THIRD):
        if p not in providers:
            raise SystemExit(f"provider {p} not in {RUN_ID}: {providers}")
    items = B.common_items(report, providers)
    log(f"{RUN_ID}: {len(items)} items common to all {len(providers)} providers")

    # The 2026-08 evaluation freeze drew its 7 SEALED temporal holdout windows from this
    # same benchmark run. Scoring on them would break the freeze, so they are removed
    # before anything is computed. Discovered 2026-08-16 while resolving wayfinder #17.
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())["holdout_windows"]}
    before = len(items)
    items = [it for it in items if it["item_id"] not in sealed]
    log(f"sealed holdout windows removed: {before - len(items)} -> {len(items)} items")

    clusters = [it["cluster"] for it in items]
    disjoint = [not it["in_training"] for it in items]
    log(f"training-disjoint windows: {sum(disjoint)}/{len(items)}")

    trio_old = [BASE, SECOND, OLD_THIRD]
    trio_new = [BASE, SECOND, NEW_THIRD]

    # per item, per arm -> (S,D,I,N)
    per_arm: dict[str, list] = {}
    picks: dict[str, dict] = {"B_old": {}, "B_new": {}}

    for p in providers:
        per_arm[f"single:{p}"] = [sdi(it["ref"], it["hyp"][p]) for it in items]

    for name, trio in (("B_old", trio_old), ("B_new", trio_new)):
        rows = []
        for it in items:
            pick = B.consensus_pick(it, trio)
            picks[name][pick] = picks[name].get(pick, 0) + 1
            rows.append(sdi(it["ref"], it["hyp"][pick]))
        per_arm[name] = rows

    for name, pool in (("oracle_old", trio_old), ("oracle_new", trio_new),
                       ("oracle_all", providers)):
        rows = []
        for it in items:
            cand = [sdi(it["ref"], it["hyp"][p])
                    for p in pool]
            rows.append(min(cand, key=lambda c: c[0] + c[1] + c[2]))
        per_arm[name] = rows

    res = {
        "run_id": RUN_ID, "n_items": len(items),
        "n_meetings": len(set(clusters)),
        "n_training_disjoint": sum(disjoint),
        "scorer": "eval/controlled_eval/scoring.py (not the benchmark app's)",
        "trio_old": trio_old, "trio_new": trio_new,
        "consensus_picks": picks,
        "arms": {k: rates(v) for k, v in per_arm.items()},
        "contrasts": {},
        "by_contamination": {},
        "influence": {},
    }

    def contrast(a: str, b: str) -> dict:
        """a minus b, paired and clustered by meeting, on every metric."""
        out = {}
        for metric, idx in (("wer", None), ("del_rate", 1), ("ins_rate", 2), ("sub_rate", 0)):
            if metric == "wer":
                ca = [(r[0] + r[1] + r[2], r[3]) for r in per_arm[a]]
                cb = [(r[0] + r[1] + r[2], r[3]) for r in per_arm[b]]
            else:
                ca = [(r[idx], r[3]) for r in per_arm[a]]
                cb = [(r[idx], r[3]) for r in per_arm[b]]
            out[metric] = cluster_bootstrap(ca, cb, clusters, n_boot=N_BOOT)
        out["head2head_wer"] = head2head(
            [(r[0] + r[1] + r[2], r[3]) for r in per_arm[a]],
            [(r[0] + r[1] + r[2], r[3]) for r in per_arm[b]])
        return out

    for a, b in (("B_old", f"single:{BASE}"), ("B_new", f"single:{BASE}"),
                 ("B_new", "B_old")):
        res["contrasts"][f"{a} vs {b}"] = contrast(a, b)

    # contamination split, descriptive
    for name in ("B_old", "B_new", f"single:{BASE}", f"single:{NEW_THIRD}"):
        rows = per_arm[name]
        res["by_contamination"][name] = {
            "disjoint": rates([r for r, d in zip(rows, disjoint) if d]),
            "contaminated": rates([r for r, d in zip(rows, disjoint) if not d]),
        }

    # single-item domination: which window moves B_new - single:BASE most
    base_rows = per_arm[f"single:{BASE}"]
    new_rows = per_arm["B_new"]
    full = rates(new_rows)["del_rate"] - rates(base_rows)["del_rate"]
    worst, shift = None, 0.0
    for k in range(len(items)):
        keep = [i for i in range(len(items)) if i != k]
        d = (rates([new_rows[i] for i in keep])["del_rate"]
             - rates([base_rows[i] for i in keep])["del_rate"])
        if abs(d - full) > abs(shift):
            worst, shift = items[k]["item_id"], d - full
    res["influence"]["del_rate_B_new_vs_base"] = {
        "delta": full, "max_shift_window": worst, "delta_without": full + shift}

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")

    for k, v in res["arms"].items():
        log(f"  {k:34s} wer={v['wer']:.4f}  del={v['del_rate']:.4f}  "
            f"ins={v['ins_rate']:.4f}  sub={v['sub_rate']:.4f}")


if __name__ == "__main__":
    main()
