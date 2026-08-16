#!/usr/bin/env python3
"""WeSep target-speaker extraction for `exp-2026-08-16-tse-overlap`, stage 1.

Runs in /home/harold/wesep-build/.venv, NOT in .venv-eval — WeSep needs its own
torch and wespeaker, and .venv-eval has 351 passing tests depending on it.

    cd /home/harold/wesep-build
    PYTHONPATH=/home/harold/wesep-build/wesep:/home/harold/wesep-build/shim \
      .venv/bin/python /home/harold/opencouncil-fine-tuning/scripts/tse/run_extract.py

Reads the frozen manifest written by `eval/tse_overlap.py build`, writes one wav
per extraction arm beside the inputs. Nothing here chooses anything: the arms,
the enrollments and the levels are all fixed by the manifest.

The CLI (`wesep ...`) is deliberately not used: on master it raises
AttributeError on `args.normalize_output` before writing its output, and
`--bsrnn` exits 1 on a hub key that does not exist. We call the Python API.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from wesep.cli.extractor import load_model_local

# BSRNN's stacked bidirectional LSTMs default to one thread here; the decode side
# runs concurrently, so the split is deliberate rather than 'all cores'.
torch.set_num_threads(int(os.environ.get("TSE_TORCH_THREADS", "8")))

WORK = Path.home() / ".cache/oc-public/tse-2026-08"
AUDIO = WORK / "stage1"
AUDIO2 = WORK / "stage2"
MODEL = Path.home() / ".wesep/english"
SR = 16_000


def rd(p: Path) -> torch.Tensor:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR, f"{p} is {sr} Hz"
    return torch.from_numpy(np.ascontiguousarray(x[:, :1].T))


def main() -> int:
    man = json.loads((WORK / "manifest_stage1.json").read_text())
    sirs = [f"{s:g}" for s in man["sirs_db"]]
    m = load_model_local(str(MODEL))
    m.set_device("cpu")
    m.set_resample_rate(SR)
    m.set_vad(False)          # enrollment-only VAD; an extra failure mode, no benefit

    # (arm, mixture stem template, enrollment file suffix)
    jobs = [("TSE_CLEAN", "{i}.CLEAN", "enroll")]
    for sir in sirs:
        jobs += [(f"TSE.{sir}", "{i}.MIX." + sir, "enroll"),
                 (f"TSE_WRONG.{sir}", "{i}.MIX." + sir, "enroll_wrong"),
                 (f"TSE_ABSENT.{sir}", "{i}.MIX." + sir, "enroll_absent")]

    t0, n, fails = time.time(), 0, []
    for it in man["items"]:
        i = it["item"]
        for arm, mixtpl, enr in jobs:
            out = AUDIO / f"{i}.{arm}.wav"
            if out.exists():
                continue
            mix = AUDIO / (mixtpl.format(i=i) + ".wav")
            enrf = AUDIO / f"{i}.{enr}.wav"
            try:
                y = m.extract_speech_from_pcm(rd(mix), SR, rd(enrf), SR)
            except Exception as e:                                  # noqa: BLE001
                fails.append((i, arm, repr(e)))
                continue
            if y is None:
                # a real, nameable failure mode — recorded, not retried
                fails.append((i, arm, "returned None"))
                continue
            sf.write(str(out), y[0].numpy(), SR)
            n += 1
        if int(i[1:]) % 5 == 0:
            print(f"{i}  {n} written  {time.time()-t0:.0f}s", flush=True)
    print(f"done: {n} extractions in {time.time()-t0:.0f}s, {len(fails)} failures")
    (WORK / "extract_failures.json").write_text(json.dumps(fails, ensure_ascii=False))
    for f in fails[:20]:
        print("  FAIL", f)
    return 0


def main_stage2() -> int:
    """Extract only the enrolled speakers in the frozen real-overlap cases."""
    man = json.loads((WORK / "manifest_stage2.json").read_text())
    m = load_model_local(str(MODEL))
    m.set_device("cpu")
    m.set_resample_rate(SR)
    m.set_vad(False)

    t0, n, fails = time.time(), 0, []
    for case in man["cases"]:
        gid = case["case"]
        mix = AUDIO2 / f"{gid}.BASELINE.wav"
        enrolled = [s for s in case["speakers"] if s["enrollable"]]
        for speaker in enrolled:
            spk = speaker["spk"]
            out = AUDIO2 / f"{gid}.TSE.{spk}.wav"
            if out.exists():
                continue
            enrf = AUDIO2 / f"{gid}.{spk}.enroll.wav"
            try:
                y = m.extract_speech_from_pcm(rd(mix), SR, rd(enrf), SR)
            except Exception as e:                                  # noqa: BLE001
                fails.append({"case": gid, "arm": f"TSE.{spk}",
                              "error": repr(e)})
                continue
            if y is None:
                fails.append({"case": gid, "arm": f"TSE.{spk}",
                              "error": "returned None"})
                continue
            sf.write(str(out), y[0].numpy(), SR)
            n += 1

        # Define wrong enrollment only when another participating speaker is
        # also enrollable; otherwise the case is coverage-limited.
        for target in enrolled:
            others = [s for s in enrolled if s["spk"] != target["spk"]]
            if not others:
                continue
            wrong = others[0]["spk"]
            out = AUDIO2 / f"{gid}.TSE_WRONG.{target['spk']}.wav"
            if out.exists():
                continue
            enrf = AUDIO2 / f"{gid}.{wrong}.enroll.wav"
            try:
                y = m.extract_speech_from_pcm(rd(mix), SR, rd(enrf), SR)
            except Exception as e:                                  # noqa: BLE001
                fails.append({"case": gid,
                              "arm": f"TSE_WRONG.{target['spk']}",
                              "error": repr(e)})
                continue
            if y is None:
                fails.append({"case": gid,
                              "arm": f"TSE_WRONG.{target['spk']}",
                              "error": "returned None"})
                continue
            sf.write(str(out), y[0].numpy(), SR)
            n += 1

    (WORK / "extract_stage2_failures.json").write_text(
        json.dumps(fails, ensure_ascii=False, indent=1))
    print(f"stage2 done: {n} extractions in {time.time() - t0:.0f}s, "
          f"{len(fails)} failures")
    for failure in fails[:20]:
        print("  FAIL", failure)
    return 0


if __name__ == "__main__":
    sys.exit(main_stage2() if len(sys.argv) > 1 and sys.argv[1] == "stage2" else main())
