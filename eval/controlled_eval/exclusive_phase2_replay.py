#!/usr/bin/env python3
"""Phase 2 replay: production's own assignment over both timelines, same utterances.

For each of the 25 frozen windows the ported `findBestSpeakerForUtterance`
(`oc_merge_port.py`, parity-tested against the pinned TypeScript) runs twice — once
over `diarization` (status quo) and once over `exclusiveDiarization` (proposal) —
on the identical Whisper utterance set, so every outcome is paired.

Emits the frozen paired counts and the adjudication item list. No gate is evaluated
here and no aggregate over human answers is computed; that is
`exclusive_phase2_analyze.py`, which refuses to run before the answers exist.

Output carries utterance text (for the audit page) and stays out of git.

Usage: python eval/controlled_eval/exclusive_phase2_replay.py
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import SC, log  # noqa: E402
from oc_merge_port import replay, segments  # noqa: E402

ASR = SC / "exclusive_phase2_asr.json"
DIAR = SC / "exclusive_phase2_diar.json"
OUT = SC / "exclusive_phase2_replay.json"
SEED = 20260807
MAX_ADJUDICATED = 50


def main():
    asr = json.loads(ASR.read_text())
    diar = json.loads(DIAR.read_text())

    per_window, items, pairs = [], [], collections.Counter()
    for wid in sorted(set(asr) & set(diar)):
        a, d = asr[wid], diar[wid]
        if "error" in a or "error" in d or not d.get("exclusiveDiarization"):
            log(f"  {wid}: skipped ({a.get('error') or d.get('error') or 'no exclusive key'})")
            continue
        reg, exc = segments(d["diarization"]), segments(d["exclusiveDiarization"])
        if not {s.speaker for s in exc} <= {s.speaker for s in reg}:
            log(f"  {wid}: exclusive labels not a subset of regular — skipped")
            continue

        utts = a["utterances"]
        r_reg, r_exc = replay(reg, utts), replay(exc, utts)
        w = collections.Counter()
        for i, (u, x, y) in enumerate(zip(utts, r_reg, r_exc)):
            drop_r, drop_e = x["branch"] == "drop", y["branch"] == "drop"
            if drop_r and not drop_e:
                cell = "recovery"
            elif drop_e and not drop_r:
                cell = "regression"
            elif drop_r and drop_e:
                cell = "both_drop"
            else:
                cell = "both_keep"
            w[cell] += 1
            w[f"guess_reg_{x['branch'] == 'guess'}"] += 1
            w[f"guess_exc_{y['branch'] == 'guess'}"] += 1
            differs = (not drop_r and not drop_e and x["speaker"] != y["speaker"])
            if differs:
                w["speaker_differs"] += 1
                items.append({
                    "window_id": wid, "utt_index": i,
                    "start": float(u["start"]), "end": float(u["end"]),
                    "text": u["text"],
                    "speaker_regular": x["speaker"], "speaker_exclusive": y["speaker"],
                    "branch_regular": x["branch"], "branch_exclusive": y["branch"],
                })
        pairs.update(w)
        per_window.append({"window_id": wid, "n_utterances": len(utts), **dict(w)})
        log(f"  {wid}: {len(utts)} utts  recov {w['recovery']}  regr {w['regression']}  "
            f"differ {w['speaker_differs']}  guess {w['guess_reg_True']}->{w['guess_exc_True']}")

    # --- adjudication sample: differing-speaker utterances only (see spec amendment)
    items.sort(key=lambda x: (x["window_id"], x["start"]))
    sampled = items
    if len(items) > MAX_ADJUDICATED:
        sampled = sorted(random.Random(SEED).sample(items, MAX_ADJUDICATED),
                         key=lambda x: (x["window_id"], x["start"]))

    res = {
        "n_windows": len(per_window),
        "n_utterances": sum(p["n_utterances"] for p in per_window),
        "paired": {k: v for k, v in sorted(pairs.items())},
        "per_window": per_window,
        "n_differing": len(items),
        "n_adjudicated": len(sampled),
        "adjudication_items": sampled,
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    log(f"\nwindows {res['n_windows']}, utterances {res['n_utterances']}")
    log(f"paired drops: recovery {pairs['recovery']}  regression {pairs['regression']}  "
        f"both_drop {pairs['both_drop']}  both_keep {pairs['both_keep']}")
    log(f"guess branch: regular {pairs['guess_reg_True']} -> exclusive {pairs['guess_exc_True']}")
    log(f"speaker differs: {len(items)}, adjudicating {len(sampled)}")
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
