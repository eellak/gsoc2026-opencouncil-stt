#!/usr/bin/env python3
"""Cut and splice every pack into a flac, and write the two arms' targets beside it.

Deliberately local. The pod bills by the second and this is pure CPU: decoding 172 meeting
mp3s and writing 3,877 short files costs nothing here and would cost real money there.

Output is one directory that can be shipped whole:
  packs/audio/pk_XXXXXX.flac    16 kHz mono, exactly the manifest duration
  packs/manifest.jsonl          {pack_id, audio, text_p, text_pn, n_utt, dur_sec, ...}

`text_p` carries native Whisper timestamps with the leading space that the round-trip check
proved is load-bearing; `text_pn` is the same words with no timestamps at all. Same audio,
same order, one difference.

Resumable per pack, because this runs for hours unattended and a pass that cannot resume is
a pass that gets restarted from zero.

  .venv-eval/bin/python -m eval.hf_export.assemble_packs
Env: PACKS OUT AUDIO_DIR
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
PACKS = Path(os.environ.get("PACKS", ROOT / "data/hf-dataset/packs/packs.json"))
OUT = Path(os.environ.get("OUT", ROOT / "data/hf-dataset/packs"))
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", ROOT / "data/asr/audio"))
SR = 16000


def log(*a):
    print(*a, flush=True)


def target_p(pack) -> str:
    # The space after each opening timestamp is Whisper's native convention. Without it the
    # last word of a segment fuses with the first word of the next once timestamps are
    # stripped, which cost a word per join in 13 of 60 packs before it was there.
    return "".join(f"<|{i['t_start']:.2f}|> {i['text']}<|{i['t_end']:.2f}|>"
                   for i in pack["items"])


def target_pn(pack) -> str:
    return " ".join(i["text"] for i in pack["items"])


def main() -> None:
    import librosa
    import numpy as np
    import soundfile as sf

    packs = json.loads(PACKS.read_text())
    adir = OUT / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "manifest.jsonl"
    done = set()
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["pack_id"])
    log(f"{len(packs)} packs, {len(done)} already assembled")

    # One meeting decoded at a time, packs grouped so each mp3 is opened exactly once.
    packs.sort(key=lambda p: (p["city_id"], p["meeting_id"], p["pack_id"]))
    cur_key, full = None, None
    written, failed = 0, []
    with man_path.open("a") as mf:
        for n, p in enumerate(packs, 1):
            if p["pack_id"] in done:
                continue
            key = f"{p['city_id']}__{p['meeting_id']}"
            if key != cur_key:
                full = None
                for ext in (".mp3", ".wav", ".m4a"):
                    src = AUDIO_DIR / f"{key}{ext}"
                    if src.exists():
                        full = librosa.load(str(src), sr=SR, mono=True)[0]
                        break
                cur_key = key
                log(f"  [{n}/{len(packs)}] {key}: "
                    f"{'loaded' if full is not None else 'NO AUDIO'}")
            if full is None:
                failed.append(p["pack_id"])
                continue
            parts = []
            for m, i in enumerate(p["items"]):
                if m:
                    parts.append(np.zeros(int(round(i["gap_fill"] * SR)), dtype="float32"))
                a, b = int(round(i["src_start"] * SR)), int(round(i["src_end"] * SR))
                if b > len(full):
                    parts = []
                    break
                parts.append(full[a:b])
            if not parts:
                failed.append(p["pack_id"])
                continue
            wav = np.concatenate(parts)
            if not np.isfinite(wav).all() or float(np.abs(wav).max()) == 0.0:
                failed.append(p["pack_id"])
                continue
            dest = adir / f"{p['pack_id']}.flac"
            sf.write(dest, wav, SR)
            mf.write(json.dumps({
                "pack_id": p["pack_id"], "audio": str(dest),
                "text_p": target_p(p), "text_pn": target_pn(p),
                "city_id": p["city_id"], "meeting_id": p["meeting_id"],
                "n_utt": p["n_utt"], "dur_sec": round(len(wav) / SR, 3),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            written += 1

    log(f"\nwrote {written} packs, {len(failed)} failed")
    if failed:
        log(f"  failed: {failed[:10]}")
    if len(failed) > 0.01 * len(packs):
        sys.exit(f"{len(failed)} failures is over 1% of {len(packs)}; not shippable")


if __name__ == "__main__":
    main()
