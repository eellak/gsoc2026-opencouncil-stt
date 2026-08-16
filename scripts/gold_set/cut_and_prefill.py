#!/usr/bin/env python3
"""Stage 5: cut the 35 s clips and fetch the prefill the human will CORRECT.

Clips are pulled with ffmpeg HTTP range-seeking straight from the public audio
URL (no whole-meeting download) into $SC/gold-set/clips/<cell_id>.wav, 16 kHz
mono, exactly [clip_start, clip_end] with NO extra padding, so clip-relative
times are exact: core = [LEAD, LEAD+CORE].

The prefill is one pyannoteAI precision-2 job per clip with
transcription=true and transcriptionConfig.model=faster-whisper-large-v3-turbo,
which returns word-level {start,end,text,speaker}. It is stored IMMUTABLY at
$SC/gold-set/prefill/<cell_id>.json and never edited: the human's corrections
are a separate file, so "the human deleted this word" stays distinguishable
from "the prefill never had it".

Cost: ~0.53 audio hours for 54 clips => about EUR 0.15 total.

Usage: python3 scripts/gold_set/cut_and_prefill.py [--cut-only] [--limit N]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.fetch_clip import fetch_clip  # noqa: E402
from eval.controlled_eval.exclusive_diar_api import api_key, run_one  # noqa: E402

SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
GS = SC / "gold-set"
STT = "faster-whisper-large-v3-turbo"


def audio_url(city, mid):
    with open(SC / "meetings" / f"{city}__{mid}.json", encoding="utf-8") as fh:
        return json.load(fh)["meeting"]["audioUrl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(ROOT / "research/gold-set-2026-08/selection.csv"))
    ap.add_argument("--cut-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.selection, newline="", encoding="utf-8") as fh:
        sel = list(csv.DictReader(fh))
    if args.limit:
        sel = sel[:args.limit]
    (GS / "clips").mkdir(parents=True, exist_ok=True)
    (GS / "prefill").mkdir(parents=True, exist_ok=True)

    urls = {}
    key = None if args.cut_only else api_key()
    hashes = {}
    for r in sel:
        cid = r["cell_id"]
        wav = GS / "clips" / f"{cid}.wav"
        if not wav.exists():
            k = (r["city_id"], r["meeting_id"])
            urls.setdefault(k, audio_url(*k))
            fetch_clip(urls[k], float(r["clip_start"]), float(r["clip_end"]),
                       str(wav), pad=0.0)
            print(f"  cut {cid}")
        hashes[cid] = hashlib.sha256(wav.read_bytes()).hexdigest()
        if args.cut_only:
            continue
        pf = GS / "prefill" / f"{cid}.json"
        if pf.exists():
            continue
        res = run_one(wav, key, cid, exclusive=True, transcription=True, stt=STT,
                      timeout=300)
        if res.get("status") != "succeeded":
            print(f"  PREFILL FAILED {cid}: {res.get('status')}")
            continue
        pf.write_text(json.dumps(res), encoding="utf-8")
        nw = len(res["output"].get("wordLevelTranscription") or [])
        print(f"  prefill {cid}: {nw} words")

    (ROOT / "research/gold-set-2026-08/clip_hashes.json").write_text(
        json.dumps({"sha256": hashes, "n": len(hashes),
                    "note": "sha256 of the 16 kHz mono wav for each gold cell; "
                            "audio itself never enters git"}, indent=1), encoding="utf-8")
    print(f"clips: {len(hashes)}  hashes -> research/gold-set-2026-08/clip_hashes.json")


if __name__ == "__main__":
    main()
