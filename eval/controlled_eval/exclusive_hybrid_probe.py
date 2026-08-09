#!/usr/bin/env python3
"""POST-HOC, NOT PREREGISTERED: what if production read both timelines?

The frozen experiment asked one question — swap the timeline, yes or no — and the
answer was no: exclusive mode resolves overlap by deleting a segment, and
`findBestSpeakerForUtterance` drops any utterance no speaker fully covers, so
deleted timeline becomes deleted transcript.

This script measures the obvious repair on the data already collected, at zero
additional cost: assign from the exclusive timeline, and when that yields nothing,
fall back to the regular one. It is exploratory. It was written after the Phase 2
gate had already failed, it has no preregistered gate, and no number it produces may
be reported as a test result. It exists to say whether a follow-up experiment is
worth designing.

Usage: python eval/controlled_eval/exclusive_hybrid_probe.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import SC, log  # noqa: E402
from oc_merge_port import find_best_speaker, segments  # noqa: E402

ASR = SC / "exclusive_phase2_asr.json"
DIAR = SC / "exclusive_phase2_diar.json"
OUT = Path(__file__).resolve().parent / "results_exclusive_hybrid_probe.json"


def assign(segs, u):
    words = [(float(w["start"]), float(w["end"])) for w in u["words"]]
    return find_best_speaker(segs, float(u["start"]), float(u["end"]), words)


def main():
    asr = json.loads(ASR.read_text())
    diar = json.loads(DIAR.read_text())

    tot = collections.Counter()
    per_window = {}
    for wid in sorted(set(asr) & set(diar)):
        a, d = asr[wid], diar[wid]
        if "error" in a or "error" in d or not d.get("exclusiveDiarization"):
            continue
        reg = segments(d["diarization"])
        exc = segments(d["exclusiveDiarization"])
        w = collections.Counter()
        for u in a["utterances"]:
            r, e = assign(reg, u), assign(exc, u)
            h = e if e is not None else r          # the repair, one line
            w["n"] += 1
            w["drop_regular"] += r is None
            w["drop_exclusive"] += e is None
            w["drop_hybrid"] += h is None
            w["guess_regular"] += r is not None and r.branch == "guess"
            w["guess_exclusive"] += e is not None and e.branch == "guess"
            w["guess_hybrid"] += h is not None and h.branch == "guess"
            w["fellback"] += e is None and r is not None
            if h is not None and r is not None:
                w["hybrid_differs_from_regular"] += h.speaker != r.speaker
        tot.update(w)
        per_window[wid] = dict(w)

    n = tot["n"]

    def per100(x):
        return round(100 * x / n, 3) if n else None

    # cluster bootstrap over windows: net drops of hybrid vs regular, per 100 utts
    keys = list(per_window)
    rng = random.Random(7)
    boot = []
    for _ in range(10000):
        picked = [per_window[keys[rng.randrange(len(keys))]] for _ in keys]
        nn = sum(p["n"] for p in picked)
        if nn:
            boot.append(100 * sum(p["drop_hybrid"] - p["drop_regular"] for p in picked) / nn)
    boot.sort()

    res = {
        "note": "POST-HOC exploratory, not preregistered, no gate applies",
        "n_utterances": n,
        "n_windows": len(per_window),
        "drops": {"regular": tot["drop_regular"], "exclusive": tot["drop_exclusive"],
                  "hybrid": tot["drop_hybrid"]},
        "guess_branch": {"regular": tot["guess_regular"],
                         "exclusive": tot["guess_exclusive"],
                         "hybrid": tot["guess_hybrid"]},
        "hybrid_fellback_to_regular": tot["fellback"],
        "hybrid_speaker_differs_from_regular": tot["hybrid_differs_from_regular"],
        "net_drops_hybrid_minus_regular_per_100": per100(
            tot["drop_hybrid"] - tot["drop_regular"]),
        "net_drops_ci95_upper_one_sided": round(boot[int(0.95 * len(boot))], 3) if boot else None,
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(json.dumps(res, ensure_ascii=False, indent=1))
    log(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
