#!/usr/bin/env python3
"""Measure the densities that drive the human-time estimate, and recompute it.

There is no measured verification rate in this repo and the agent cannot
listen, so this is NOT a measured human rate. It is a per-action model whose
inputs - words per core, blocks per core, overlap seconds, how far the prefill
is from the published transcript - ARE measured on the cells that were actually
drawn. The tool records real per-pass time as the human works, which replaces
this estimate after the first half hour.

Writes research/gold-set-2026-08/pilot.json (aggregates only, no text).
"""
import csv
import json
import os
import re
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
GS = SC / "gold-set"

# seconds per word of correction, by how contested the cell is
SEC_PER_WORD = {"P": 4.0, "I": 3.6, "H": 3.0, "M": 2.3, "R": 2.5}
PASS_A = 47.0            # 35 s clip + judgement
PASS_C = {"P": 60, "I": 55, "H": 45, "M": 40, "R": 45}


def toks(s):
    return [w for w in re.split(r"\s+", re.sub(r"[^\w\sͰ-Ͽἀ-῿]", " ", s.lower())) if w]


def main():
    with open(ROOT / "research/gold-set-2026-08/selection.csv", newline="", encoding="utf-8") as fh:
        sel = [r for r in csv.DictReader(fh) if r["required27"] == "True"]
    rows, missing = [], 0
    for r in sel:
        pf = GS / "prefill" / f"{r['cell_id']}.json"
        if not pf.exists():
            missing += 1
            continue
        o = json.loads(pf.read_text())["output"]
        lead = float(r["core_start"]) - float(r["clip_start"])
        core_a, core_b = lead, lead + (float(r["core_end"]) - float(r["core_start"]))
        words = [w for w in (o.get("wordLevelTranscription") or [])
                 if core_a <= w["start"] < core_b]
        turns = [t for t in (o.get("turnLevelTranscription") or [])
                 if t["end"] > core_a and t["start"] < core_b]
        rows.append({
            "cell_id": r["cell_id"], "stream": r["draw_stream"],
            "n_words": len(words), "n_blocks": len(turns),
            "n_prefill_speakers": len({w["speaker"] for w in words}),
            "pyannote_overlap_sec": float(r["pyannote_overlap_sec"]),
        })

    by = {}
    for x in rows:
        by.setdefault(x["stream"], []).append(x)

    total, per_stream = 0.0, {}
    for st, xs in sorted(by.items()):
        w = median(x["n_words"] for x in xs)
        sec = PASS_A + w * SEC_PER_WORD[st] + PASS_C[st]
        per_stream[st] = {"n_cells": len(xs), "median_words_in_core": w,
                          "median_blocks_in_core": median(x["n_blocks"] for x in xs),
                          "sec_per_cell": round(sec), "minutes": round(len(xs) * sec / 60, 1)}
        total += len(xs) * sec

    warm, blind = 10.0, 10.0
    out = {
        "note": "model, not a measured human rate; densities are measured",
        "n_cells": len(rows), "missing_prefill": missing,
        "assumptions": {"pass_a_sec": PASS_A, "sec_per_word": SEC_PER_WORD, "pass_c_sec": PASS_C},
        "per_stream": per_stream,
        "median_words_per_core": median(x["n_words"] for x in rows) if rows else None,
        "median_blocks_per_core": median(x["n_blocks"] for x in rows) if rows else None,
        "cells_minutes": round(total / 60, 1),
        "warmup_minutes": warm, "blind_extra_minutes": blind,
        "total_minutes": round(total / 60 + warm + blind, 1),
        "range_minutes": [round((total / 60 + warm + blind) * 0.82),
                          round((total / 60 + warm + blind) * 1.24)],
        "scored_core_minutes": round(len(rows) * 15 / 60, 2),
        "audio_played_minutes": round(len(rows) * 35 / 60, 2),
    }
    out["human_min_per_scored_audio_min"] = round(out["total_minutes"] / out["scored_core_minutes"], 1)
    (ROOT / "research/gold-set-2026-08/pilot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
