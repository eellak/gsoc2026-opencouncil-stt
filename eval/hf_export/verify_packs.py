#!/usr/bin/env python3
"""Assemble sample packs for real and check the tokens the model would actually be trained on.

The accounting gates in `build_packs.py` run on the manifest, and a manifest can be perfect
while the tensor handed to the optimizer carries a duplicated prefix, a silent truncation,
or the wrong timestamp mode. Codex was explicit about this and it is the difference between
a dataset that is correct and one that merely adds up.

So this cuts the audio, builds the targets for both arms, and asserts on the token ids:

  P   <|startoftranscript|><|el|><|transcribe|> <|0.00|> text <|2.34|> <|2.74|> text ... <|endoftext|>
  Pn  <|startoftranscript|><|el|><|transcribe|><|notimestamps|> text text ... <|endoftext|>

The round-trip check is the one that catches the quiet failures: decode the ids, strip the
specials, normalise, and compare against the source texts joined in order. Missing spaces,
fused punctuation and truncation all show up there and nowhere else.

  .venv-eval/bin/python -m eval.hf_export.verify_packs --n 120
Env: PACKS AUDIO_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
PACKS = Path(os.environ.get("PACKS", ROOT / "data/hf-dataset/packs/packs.json"))
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", ROOT / "data/asr/audio"))
SR = 16000
MAX_LABEL = 448
BASE_ID = "openai/whisper-large-v3"


def log(*a):
    print(*a, flush=True)


def target_text(pack, timestamps: bool) -> str:
    if not timestamps:
        return " ".join(i["text"] for i in pack["items"])
    # The leading space is Whisper's native convention and it is not cosmetic: without it
    # the last word of one segment fuses with the first word of the next once the timestamp
    # tokens are stripped, so the round-trip loses a word per join. Thirteen of sixty packs
    # failed exactly that way before this space existed.
    out = []
    for i in pack["items"]:
        out.append(f"<|{i['t_start']:.2f}|> {i['text']}<|{i['t_end']:.2f}|>")
    return "".join(out)


def assemble(pack, cache: dict):
    """Cut each utterance from its meeting audio and splice with silence between."""
    import numpy as np
    import soundfile as sf
    key = f"{pack['city_id']}__{pack['meeting_id']}"
    if key not in cache:
        cache.clear()          # one meeting in memory at a time; these are ~90 min files
        for ext in (".mp3", ".wav", ".m4a"):
            p = AUDIO_DIR / f"{key}{ext}"
            if p.exists():
                import librosa
                cache[key] = librosa.load(str(p), sr=SR, mono=True)[0]
                break
        else:
            return None
    full = cache[key]
    parts = []
    for n, i in enumerate(pack["items"]):
        if n:
            parts.append(np.zeros(int(round(i["gap_fill"] * SR)), dtype="float32"))
        a, b = int(round(i["src_start"] * SR)), int(round(i["src_end"] * SR))
        if b > len(full):
            return None
        parts.append(full[a:b])
    return np.concatenate(parts) if parts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()

    packs = json.loads(PACKS.read_text())
    rnd = random.Random(20260810)
    sample = rnd.sample(packs, min(a.n, len(packs)))
    # group by meeting so the audio cache is not thrashed
    sample.sort(key=lambda p: (p["city_id"], p["meeting_id"]))
    log(f"{len(sample)} packs sampled from {len(packs)}")

    from transformers import WhisperProcessor
    proc = WhisperProcessor.from_pretrained(BASE_ID, language="greek", task="transcribe")
    tok = proc.tokenizer
    from eval.controlled_eval.scoring import wtoks

    ts_begin = tok.convert_tokens_to_ids("<|0.00|>")
    no_ts = tok.convert_tokens_to_ids("<|notimestamps|>")
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    log(f"token ids: sot={sot} notimestamps={no_ts} first_timestamp={ts_begin} eot={eot}")

    bad, cache, n_audio = [], {}, 0
    lens = {"P": [], "Pn": []}
    for p in sample:
        # ---- targets
        tok.predict_timestamps = True
        ids_p = tok("<|startoftranscript|><|el|><|transcribe|>"
                    + target_text(p, True), add_special_tokens=False).input_ids + [eot]
        tok.predict_timestamps = False
        ids_n = tok("<|startoftranscript|><|el|><|transcribe|><|notimestamps|>"
                    + target_text(p, False), add_special_tokens=False).input_ids + [eot]
        lens["P"].append(len(ids_p)); lens["Pn"].append(len(ids_n))

        tstamps = [i for i in ids_p if i >= ts_begin]
        if len(tstamps) != 2 * p["n_utt"]:
            bad.append(f"{p['pack_id']}: {len(tstamps)} timestamp tokens for "
                       f"{p['n_utt']} utterances, expected {2*p['n_utt']}")
        if no_ts in ids_p:
            bad.append(f"{p['pack_id']}: P carries <|notimestamps|>")
        if no_ts not in ids_n:
            bad.append(f"{p['pack_id']}: Pn is missing <|notimestamps|>")
        if any(i >= ts_begin for i in ids_n):
            bad.append(f"{p['pack_id']}: Pn carries timestamp tokens")
        if ids_p.count(sot) != 1 or ids_n.count(sot) != 1:
            bad.append(f"{p['pack_id']}: duplicated or missing <|startoftranscript|>")
        if len(ids_p) > MAX_LABEL or len(ids_n) > MAX_LABEL:
            bad.append(f"{p['pack_id']}: label {max(len(ids_p), len(ids_n))} > {MAX_LABEL}")
        if tstamps != sorted(tstamps):
            bad.append(f"{p['pack_id']}: timestamp tokens not monotonic")

        # ---- round trip
        want = wtoks(" ".join(i["text"] for i in p["items"]))
        got = wtoks(tok.decode(ids_p, skip_special_tokens=True))
        if want != got:
            bad.append(f"{p['pack_id']}: round-trip differs ({len(want)} vs {len(got)} words)")

        # ---- audio
        wav = assemble(p, cache)
        if wav is None:
            bad.append(f"{p['pack_id']}: audio could not be assembled")
            continue
        n_audio += 1
        dur = len(wav) / SR
        if dur > 30.0 + 1e-3:
            bad.append(f"{p['pack_id']}: assembled audio {dur:.2f}s over the window")
        if abs(dur - p["dur_sec"]) > 0.05:
            bad.append(f"{p['pack_id']}: audio {dur:.2f}s != manifest {p['dur_sec']:.2f}s")
        import numpy as np
        if not np.isfinite(wav).all() or float(np.abs(wav).max()) == 0.0:
            bad.append(f"{p['pack_id']}: audio is silent or non-finite")

    log(f"audio assembled for {n_audio}/{len(sample)} packs")
    for k, v in lens.items():
        log(f"  {k}: label tokens mean {sum(v)/len(v):.0f}, max {max(v)} (limit {MAX_LABEL})")

    ex = sample[0]
    tok.predict_timestamps = True
    log("\nπαράδειγμα P:\n  " + target_text(ex, True)[:220])
    log("παράδειγμα Pn:\n  " + target_text(ex, False)[:220])

    if bad:
        log(f"\nTENSOR GATES FAILED ({len(bad)}):")
        for b in bad[:15]:
            log("  " + b)
        sys.exit(1)
    log("\nall tensor-level gates passed")


if __name__ == "__main__":
    main()
