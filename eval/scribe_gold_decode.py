#!/usr/bin/env python3
"""Transcribe the 30 frozen gold cells with Scribe v2 and keep the raw responses.

`exp-2026-09-09-scribe-confidence-fidelity` needs a Scribe row on the gold set,
which has never had one. This writes it once; the probe never calls the API.

Request parameters are taken from `eval/live_fusion/providers.SCRIBE_PARAMS` and
not restated, so the words scored here come from the same configuration the fusion
runs in production. Raw responses are stored whole -- `logprob` is the measured
quantity and a processed word list would throw away the provenance.

Audio and transcript text stay under ~/.cache/oc-public; nothing here goes in git.

    ELEVENLABS_API_KEY=... .venv-eval/bin/python eval/scribe_gold_decode.py [clean|verbatim]

Default is `clean`, the mode production calls. `verbatim` reproduces the 2026-09-09
row and writes to the directory that already holds it; existing files are skipped,
so neither mode can overwrite the other's evidence.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.live_fusion.providers import _scribe_form                 # noqa: E402

GOLD = Path.home() / ".cache/oc-public/gold-set-2026-08"
CLIPS = Path.home() / ".cache/oc-public/gold-set"
URL = "https://api.elevenlabs.io/v1/speech-to-text"

# One directory per request shape, because the two modes are different systems and
# a mixed directory is unreadable after the fact. `hyp/scribe` holds the 2026-09-09
# decode, which is VERBATIM: this script used to retype the form fields and omitted
# no_verbatim, the same defect as eval/live_fusion/providers.py. Those files stay --
# exp-2026-09-09-scribe-confidence-fidelity is computed on them and its AUROC is a
# statement about the verbatim arm. The corrected request writes to hyp/scribe-nv.
OUT_BY_MODE = {"clean": GOLD / "hyp" / "scribe-nv", "verbatim": GOLD / "hyp" / "scribe"}


def key() -> str:
    k = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("SCRIBE_API_KEY")
    env = ROOT / ".env"
    if not k and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            n, _, v = line.partition("=")
            if n.strip() in ("ELEVENLABS_API_KEY", "SCRIBE_API_KEY") and v.strip():
                return v.strip().strip("'\"")
    if not k:
        raise SystemExit("no ELEVENLABS_API_KEY")
    return k


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "clean"
    if mode not in OUT_BY_MODE:
        raise SystemExit(f"mode must be one of {sorted(OUT_BY_MODE)}")
    api = key()
    out_dir = OUT_BY_MODE[mode]
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = json.loads((GOLD / "cells-frozen.json").read_text())["cells"]
    # Derived from SCRIBE_PARAMS, never retyped: retyping is what produced the
    # verbatim row that hyp/scribe still holds.
    data = _scribe_form()
    if mode == "verbatim":
        data.pop("no_verbatim", None)
    done = failed = skipped = 0
    for c in cells:
        dst = out_dir / f"{c['id']}.json"
        if dst.exists():
            skipped += 1
            continue
        wav = CLIPS / c["clip"]
        if not wav.is_file():
            print(f"MISSING AUDIO {c['id']}", file=sys.stderr)
            failed += 1
            continue
        for attempt in range(3):
            try:
                # Reopened per attempt: an exhausted handle uploads zero bytes.
                with open(wav, "rb") as fh:
                    r = requests.post(URL, headers={"xi-api-key": api}, data=data,
                                      files={"file": (wav.name, fh, "audio/wav")},
                                      timeout=600)
                r.raise_for_status()
                break
            except Exception as e:                       # noqa: BLE001
                if attempt == 2:
                    print(f"FAILED {c['id']}: {e}", file=sys.stderr)
                    r = None
                    break
                time.sleep(2 * (attempt + 1))
        if r is None:
            failed += 1
            continue
        body = r.json()
        nw = sum(1 for w in body.get("words", []) if w.get("type") == "word")
        nlp = sum(1 for w in body.get("words", [])
                  if w.get("type") == "word" and w.get("logprob") is not None)
        dst.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        done += 1
        print(f"{c['id']}  words={nw}  with_logprob={nlp}", flush=True)
    print(json.dumps({"decoded": done, "skipped_existing": skipped,
                      "failed": failed, "mode": mode, "out": str(out_dir)}))


if __name__ == "__main__":
    main()
