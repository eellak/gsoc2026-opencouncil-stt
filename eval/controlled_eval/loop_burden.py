#!/usr/bin/env python3
"""Loop burden: excess repeated words per ten minutes of input audio.

Frozen in the [preregistration](../../docs/specs/window-shape-preregistration.md) before
any output existed, because thresholds picked after looking at transcripts are not
thresholds.

Two design choices carry it.

**The denominator is audio, not output words.** Per output word, a model improves its score
by writing less, which is the exact failure this metric is supposed to catch.

**High specificity, not general hallucination detection.** No detector that sees only the
output can know whether a real speaker repeated a phrase, and council speech is full of
genuine formulas. The thresholds are set so ordinary repetition does not fire, and the
false-positive check is to run this unchanged over the published human-edited transcripts.
There is no whitelist of Greek phrases and there will not be one: a whitelist built after
seeing the output is how a detector gets tuned into agreement with whatever it was tuned on.

Tandem repeats only, meaning immediately adjacent copies. Positions are unioned so a nested
pattern is not counted twice.

  .venv-eval/bin/python -m eval.controlled_eval.loop_burden --set ~/oc-longform \
      --tags base finetune --json results_longform_loops.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.controlled_eval.scoring import wtoks  # noqa: E402

# (min block words, max block words, minimum consecutive copies) — frozen
RULES = [(1, 4, 6), (5, 11, 3), (12, 10**9, 2)]


def excess_positions(words: list[str]) -> set[int]:
    """Indices belonging to a copy after the first, over every rule, unioned."""
    n = len(words)
    marked: set[int] = set()
    for lo, hi, min_copies in RULES:
        for size in range(lo, min(hi, n // min_copies) + 1):
            i = 0
            while i + size * min_copies <= n:
                block = words[i:i + size]
                copies = 1
                j = i + size
                while j + size <= n and words[j:j + size] == block:
                    copies += 1
                    j += size
                if copies >= min_copies:
                    # everything after the first copy is excess
                    marked.update(range(i + size, j))
                    i = j
                else:
                    i += 1
    return marked


def burden(text: str, audio_sec: float) -> dict:
    w = wtoks(text)
    ex = excess_positions(w)
    return {"words": len(w), "excess": len(ex),
            "per_10min": round(len(ex) / (audio_sec / 600.0), 3) if audio_sec else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()

    root = Path(a.set).expanduser()
    manifest = {r["wav"]: r for r in json.loads((root / "manifest.json").read_text())}
    out = {"rules": RULES, "systems": {}}

    for tag in a.tags:
        p = root / f"hyp_{tag}.json"
        if not p.exists():
            print(f"{tag}: no hypotheses yet, skipped")
            continue
        hyps = json.loads(p.read_text())
        rows = []
        for wav, h in sorted(hyps.items()):
            m = manifest[wav]
            b = burden(h["text"], float(m["dur_sec"]))
            rows.append({"wav": wav, "city_id": m["city_id"],
                         "meeting_id": m["meeting_id"], **b})
        tot_words = sum(r["words"] for r in rows)
        tot_excess = sum(r["excess"] for r in rows)
        tot_sec = sum(float(manifest[r["wav"]]["dur_sec"]) for r in rows)
        out["systems"][tag] = {
            "spans": len(rows), "words": tot_words, "excess": tot_excess,
            "per_10min": round(tot_excess / (tot_sec / 600.0), 3),
            "spans_with_any_loop": sum(1 for r in rows if r["excess"]),
            "worst": sorted(rows, key=lambda r: -r["per_10min"])[:5],
            "by_span": rows,
        }
        s = out["systems"][tag]
        print(f"{tag}: {s['spans']} spans, {s['words']} words, "
              f"{s['excess']} excess -> {s['per_10min']}/10min, "
              f"{s['spans_with_any_loop']} spans with any loop")

    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
