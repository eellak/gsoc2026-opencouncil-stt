#!/usr/bin/env python3
"""Stage 6: assemble the verification package the human opens.

Output (default ~/oc-gold-set, OUTSIDE the repo - it holds council audio and
transcript text, which never enter git):

    index.html  app.js  app.css     copied from scripts/gold_set/page/
    clips/<cell_id>.wav             35 s, 16 kHz mono
    cells.json                      prefill + speech spans + published text
    answers.json                    written by audit_server.py as you work

The stream a cell was drawn from (P/I/H/M/R) is deliberately NOT in cells.json
and the cells are shuffled, so the annotator cannot tell which cells were
picked because a tool thought there was overlap there. The key stays in the
repo at research/gold-set-2026-08/selection.csv.

Run:
    python3 scripts/gold_set/build_page.py
    AUDIT_DIR=~/oc-gold-set PORT=8790 python3 eval/controlled_eval/audit_server.py
    # then open http://localhost:8790/
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
GS = SC / "gold-set"
PAGE = Path(__file__).resolve().parent / "page"
PREFILL_SOURCE = "pyannoteAI precision-2 + faster-whisper-large-v3-turbo, 2026-08-16"


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1] + 0.05:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([round(s, 2), round(e, 2)])
    return out


def published(city, mid, a, b):
    """Utterances of the published OpenCouncil transcript inside [a,b], clip-relative.
    Used only in the pass-3 omission audit, never as the thing being corrected."""
    d = json.loads((SC / "meetings" / f"{city}__{mid}.json").read_text())["transcript"]
    out = []
    for seg in d:
        for u in seg["utterances"]:
            if u["endTimestamp"] > a and u["startTimestamp"] < b:
                out.append({"s": round(u["startTimestamp"] - a, 2),
                            "e": round(u["endTimestamp"] - a, 2),
                            "text": u["text"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "oc-gold-set"))
    ap.add_argument("--tier", default="required27",
                    choices=["required27", "required", "all"])
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    for f in ("index.html", "app.js", "app.css"):
        shutil.copy2(PAGE / f, out / f)

    with open(ROOT / "research/gold-set-2026-08/selection.csv", newline="", encoding="utf-8") as fh:
        sel = list(csv.DictReader(fh))
    warm = [r for r in sel if r["tier"] == "stretch"][:3]
    warm_ids = {r["cell_id"] for r in warm}
    if args.tier == "required27":
        sel = [r for r in sel if r["required27"] == "True"]
    elif args.tier == "required":
        sel = [r for r in sel if r["tier"] == "required"]
    sel = warm + [r for r in sel if r["cell_id"] not in warm_ids]

    cells, missing = [], []
    for r in sel:
        cid = r["cell_id"]
        pf = GS / "prefill" / f"{cid}.json"
        wav = GS / "clips" / f"{cid}.wav"
        if not pf.exists() or not wav.exists():
            missing.append(cid)
            continue
        o = json.loads(pf.read_text())["output"]
        lead = round(float(r["core_start"]) - float(r["clip_start"]), 3)
        core = round(float(r["core_end"]) - float(r["core_start"]), 3)
        dur = round(float(r["clip_end"]) - float(r["clip_start"]), 3)
        shutil.copy2(wav, out / "clips" / f"{cid}.wav")
        cells.append({
            "id": cid, "tier": "warmup" if cid in warm_ids else r["tier"], "calib": r["calibration"] == "True",
            "warmup": cid in warm_ids, "clip": f"clips/{cid}.wav",
            "lead": lead, "core": core, "dur": dur,
            "prefill_source": PREFILL_SOURCE,
            "turns": [{"s": round(t["start"], 3), "e": round(t["end"], 3),
                       "spk": t["speaker"], "text": t["text"]}
                      for t in (o.get("turnLevelTranscription") or [])],
            "words": [{"s": round(w["start"], 3), "e": round(w["end"], 3),
                       "t": w["text"], "spk": w["speaker"]}
                      for w in (o.get("wordLevelTranscription") or [])],
            "speech": merge([[t["start"], t["end"]] for t in o.get("diarization", [])]),
            "alt": published(r["city_id"], r["meeting_id"],
                             float(r["clip_start"]), float(r["clip_end"])),
        })

    # Warm-up cells are practice: they come first, in order, and are excluded from
    # every metric. One of them carries a deliberately deleted phrase, so the
    # annotator meets an omission the prefill does not show before the gold cells
    # start. The key to that probe lives outside the served directory.
    rng = random.Random(args.seed)
    warm_cells = [c for c in cells if c["warmup"]]
    gold = [c for c in cells if not c["warmup"]]
    rng.shuffle(gold)
    probe = None
    for c in warm_cells:
        cand = [t for t in c["turns"] if len(t["text"].split()) >= 8]
        if cand and probe is None:
            t = cand[len(cand) // 2]
            w = t["text"].split()
            probe = {"cell": c["id"], "removed": " ".join(w[2:5]), "turn_start": t["s"]}
            t["text"] = " ".join(w[:2] + w[5:])
            break
    cells = warm_cells + gold
    if probe:
        (out.parent / (out.name + "-PROBE-KEY.json")).write_text(
            json.dumps(probe, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "cells.json").write_text(json.dumps(
        {"protocol": "gold-set-2026-08-16", "prefill_source": PREFILL_SOURCE,
         "n": len(cells), "cells": cells}, ensure_ascii=False), encoding="utf-8")
    if not (out / "answers.json").exists():
        (out / "answers.json").write_text("{}", encoding="utf-8")

    print(f"{len(cells)} cells -> {out}")
    if missing:
        print(f"MISSING prefill/clip for {len(missing)}: {missing[:5]}")
    print(f"scored core: {sum(c['core'] for c in cells)/60:.1f} min · "
          f"audio played: {sum(c['dur'] for c in cells)/60:.1f} min")
    print(f"\nAUDIT_DIR={out} PORT=8790 python3 eval/controlled_eval/audit_server.py")


if __name__ == "__main__":
    main()
