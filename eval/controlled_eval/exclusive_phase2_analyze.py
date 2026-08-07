#!/usr/bin/env python3
"""Phase 2 gate: the five frozen conditions, evaluated once the answers exist.

Refuses to run before the human answers file exists, and reads the blinding key only
at that point — so no aggregate over the adjudication can be looked at early.

Spec: docs/specs/exclusive-diarization-preregistration.md § Phase 2 gate.
Output: eval/controlled_eval/results_exclusive_phase2.json (aggregates only).

Usage: python eval/controlled_eval/exclusive_phase2_analyze.py [--parity-passed N]
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import SC, log  # noqa: E402

REPLAY = SC / "exclusive_phase2_replay.json"
FREEZE = SC / "exclusive_freeze.json"
AUDIT = Path.home() / "oc-exclusive-audit"
ANSWERS = AUDIT / "answers.json"
KEY = AUDIT.parent / "oc-exclusive-audit-KEY_DO_NOT_OPEN_UNTIL_DONE.json"
OUT = Path(__file__).resolve().parent / "results_exclusive_phase2.json"
N_BOOT = 10000
BOOT_SEED = 7


def cluster_boot(values_by_cluster, stat, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resample whole clusters; return the sorted bootstrap distribution of `stat`."""
    keys = list(values_by_cluster)
    if not keys:
        return []
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        picked = [values_by_cluster[keys[rng.randrange(len(keys))]] for _ in keys]
        v = stat([x for g in picked for x in g])
        if v is not None:
            out.append(v)
    out.sort()
    return out


def main():
    parity = None
    if "--parity-passed" in sys.argv:
        parity = int(sys.argv[sys.argv.index("--parity-passed") + 1])

    if not ANSWERS.exists():
        log(f"answers file not found: {ANSWERS}\n"
            "The gate cannot be evaluated before the human listens. Nothing computed.")
        sys.exit(2)

    rep = json.loads(REPLAY.read_text())
    gate_cfg = json.loads(FREEZE.read_text())["phase2"]["gate"]
    answers = json.loads(ANSWERS.read_text())
    key = {r["id"]: r for r in json.loads(KEY.read_text())}

    # ---------------------------------------------------- attribution (gate 1,2,4)
    tally = collections.Counter()
    wins_by_window = collections.defaultdict(list)
    for aid, ans in answers.items():
        k = key.get(aid)
        if not k:
            continue
        tally[ans] += 1
        if ans in ("a", "b"):
            chosen = k["A"] if ans == "a" else k["B"]
            wins_by_window[k["window_id"]].append(1 if chosen == "exclusive" else 0)

    determinate = sum(len(v) for v in wins_by_window.values())
    wins = sum(sum(v) for v in wins_by_window.values())
    prop = wins / determinate if determinate else 0.0
    boot = cluster_boot(wins_by_window,
                        lambda xs: (sum(xs) / len(xs)) if xs else None)
    lower95 = round(boot[int(0.05 * len(boot))], 4) if boot else None
    n_answered = sum(tally.values())

    # ---------------------------------------------------- drops (gate 3)
    per_w = {p["window_id"]: p for p in rep["per_window"]}
    net_by_window = {
        wid: [(p.get("regression", 0) - p.get("recovery", 0), p["n_utterances"])]
        for wid, p in per_w.items()}

    def per100(pairs):
        n = sum(x[1] for x in pairs)
        return 100 * sum(x[0] for x in pairs) / n if n else None

    dboot = cluster_boot(net_by_window, per100)
    upper95 = round(dboot[int(0.95 * len(dboot))], 4) if dboot else None
    net_point = per100([v[0] for v in net_by_window.values()])

    g = {
        "powered_min_determinate": determinate >= gate_cfg["min_determinate"],
        "attribution_prop_ge_2_3": prop >= gate_cfg["attribution_prop"],
        "attribution_lower_bound_gt_0.5": (lower95 is not None
                                           and lower95 > gate_cfg["attribution_lower_bound_above"]),
        "drop_noninferiority": (upper95 is not None
                                and upper95 < gate_cfg["drop_noninferiority_upper_per_100"]),
        "neither_rate_ok": (tally["neither"] / n_answered <= gate_cfg["max_neither_frac"]
                            if n_answered else False),
        "cant_tell_rate_ok": (tally["unsure"] / n_answered <= gate_cfg["max_cant_tell_frac"]
                              if n_answered else False),
        "port_parity": (parity == gate_cfg["port_parity_tests"]) if parity is not None else None,
    }

    res = {
        "n_windows": rep["n_windows"],
        "n_utterances": rep["n_utterances"],
        "paired_counts": rep["paired"],
        "adjudication": {
            "n_items_built": rep["n_adjudicated"],
            "n_answered": n_answered,
            "answer_tally": dict(tally),
            "n_determinate": determinate,
            "exclusive_wins": wins,
            "exclusive_proportion": round(prop, 4),
            "cluster_ci_lower_95_one_sided": lower95,
            "n_window_clusters": len(wins_by_window),
        },
        "drops": {
            "recovery": rep["paired"].get("recovery", 0),
            "regression": rep["paired"].get("regression", 0),
            "net_regressions_per_100_utt": (round(net_point, 4)
                                            if net_point is not None else None),
            "cluster_ci_upper_95_one_sided": upper95,
        },
        "guess_branch": {
            "regular": rep["paired"].get("guess_reg_True", 0),
            "exclusive": rep["paired"].get("guess_exc_True", 0),
        },
        "gate": g,
        "gate_passed": all(v is True for v in g.values()),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(json.dumps(res, ensure_ascii=False, indent=1))
    log("\nPHASE 2 GATE: " + ("PASS" if res["gate_passed"] else "FAIL"))
    if g["port_parity"] is None:
        log("(pass --parity-passed N with the test suite result to complete the gate)")


if __name__ == "__main__":
    main()
