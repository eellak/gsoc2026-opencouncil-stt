"""Score the correction-only adapter against artifact-ct2-fixed on the frozen windows.

`exp-2026-08-13-correction-only`. The decision rule, the arms and the honesty
labels are fixed in
[`docs/specs/2026-08-13-correction-only-preregistration.md`](../docs/specs/2026-08-13-correction-only-preregistration.md),
committed before the pod existed.

Both arms are decoded under the **control decode configuration** (arm A of the
decode ablation), on this machine, in this environment, with the same per-window
seeds - regardless of what the ablation concluded. Everything except the model
directory is imported from `notebooks/decode_ablation.py` so the two experiments
cannot drift apart.

The control arm reuses `eval-A.json` from the ablation. That file *is* a fresh
decode of `artifact-ct2-fixed` on this machine, in this environment, under this
exact config, produced today - the prereg's "decoded fresh" clause exists to forbid
reusing the GPU benchmark run, not to forbid reusing an identical local computation.

    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/correction_only_score.py decode
    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/correction_only_score.py score
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap  # noqa: E402

import decode_ablation as DA  # noqa: E402

ARM_MODEL = "/home/harold/oc-correction-only/ct2"
ARM_LABEL = "correction-only (artifact-adapter-correction-only)"
CTRL_LABEL = "control (artifact-ct2-fixed)"
DEST = Path.home() / ".cache/oc-public/correction-only/decode.json"


def log(m):
    print(m, flush=True)


def decode() -> None:
    import ctranslate2
    from faster_whisper import WhisperModel

    rows = DA.rows("eval")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(DEST.read_text()) if DEST.exists() else {
        "model": ARM_MODEL, "config": DA.CONTROL, "windows": {}}
    todo = [r for r in rows if r["window_id"] not in state["windows"]]
    log(f"{len(todo)} windows to decode ({len(state['windows'])} already done)")
    if not todo:
        return

    model = WhisperModel(ARM_MODEL, device=DA.DEVICE, compute_type=DA.COMPUTE,
                         cpu_threads=DA.THREADS)
    diag = DA.Diagnostics()
    fw_log = logging.getLogger("faster_whisper")
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)
    try:
        for i, r in enumerate(todo, 1):
            wav = DA.sc() / "bench_windows" / f"{r['window_id']}.wav"
            diag.reset()
            # Same seed as the control arm used for this window: the arms differ by
            # weights, not by the draw.
            ctranslate2.set_random_seed(DA.seed_for("A", r["window_id"]))
            t0 = time.time()
            segs = list(model.transcribe(str(wav), **DA.CONTROL)[0])
            state["windows"][r["window_id"]] = {
                "text": "".join(s.text for s in segs).strip(),
                "n_segments": len(segs),
                "wall_seconds": round(time.time() - t0, 1),
                **diag.snapshot()}
            DEST.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            log(f"  {i}/{len(todo)} {r['window_id']} {state['windows'][r['window_id']]['wall_seconds']}s")
    finally:
        fw_log.removeHandler(diag)
    log(f"-> {DEST}")


def counts(texts: dict[str, str]) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    for r in DA.rows("eval"):
        wid = r["window_id"]
        if wid not in texts:
            raise SystemExit(f"incomplete: {wid} missing. No complete-case subsets.")
        ref = ftoks(DA.reference_text(wid))
        out[wid] = (*sdi(ref, ftoks(texts[wid])), len(ref))
    return out


def score() -> None:
    arm_state = json.loads(DEST.read_text())
    ctrl_state = json.loads((DA.out_dir() / "eval-A.json").read_text())
    arm = counts({k: v["text"] for k, v in arm_state["windows"].items()})
    ctrl = counts({k: v["text"] for k, v in ctrl_state["windows"].items()})

    rows = DA.rows("eval")
    wids = [r["window_id"] for r in rows]
    blocks = [r["meeting_id"] for r in rows]

    res = {"experiment": "exp-2026-08-13-correction-only",
           "arm_model": ARM_MODEL,
           "decode_config": DA.CONTROL,
           "n_windows": len(wids), "n_meetings": len(set(blocks)),
           "bootstrap": {"replicates": DA.BOOTSTRAP_REPLICATES,
                         "seed": DA.BOOTSTRAP_SEED, "blocks": "meeting_id"},
           "totals": {CTRL_LABEL: DA.rates(ctrl), ARM_LABEL: DA.rates(arm)},
           "vs_control": {}}

    for metric, pick in DA.PICK.items():
        ca = [(pick(arm[w]), arm[w][3]) for w in wids]
        cb = [(pick(ctrl[w]), ctrl[w][3]) for w in wids]
        ci = cluster_bootstrap(ca, cb, blocks, n_boot=DA.BOOTSTRAP_REPLICATES,
                               seed=DA.BOOTSTRAP_SEED)
        res["vs_control"][metric] = {"delta": ci["delta"], "ci95": ci["ci95"],
                                     "excludes_zero": ci["excludes_zero"]}
    res["influence"] = DA.influence(arm, ctrl, wids, DA.PICK["wer"])
    res["head2head"] = {
        "arm_better": sum(1 for w in wids if sum(arm[w][:3]) < sum(ctrl[w][:3])),
        "tie": sum(1 for w in wids if sum(arm[w][:3]) == sum(ctrl[w][:3])),
        "control_better": sum(1 for w in wids if sum(arm[w][:3]) > sum(ctrl[w][:3]))}
    res["label"] = ("SUGGESTIVE. One seed per arm against a measured 2.1-point "
                    "per-seed spread (exp-2026-08-08-mixture-ratio). This does not "
                    "crown a winner in either direction.")

    out = DEST.parent / "results.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for label, t in res["totals"].items():
        log(f"{label:48s} WER {t['wer']:.4f}  del {t['del_rate']:.4f}  "
            f"ins {t['ins_rate']:.4f}  S{t['sub']} D{t['del']} I{t['ins']}")
    for m, v in res["vs_control"].items():
        log(f"  {m:9s} {v['delta']:+.5f} [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}]"
            f"{'  excludes zero' if v['excludes_zero'] else ''}")
    log(f"  head2head {res['head2head']}")
    log(f"  influence {json.dumps(res['influence'])}")
    log(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("decode", "score"))
    a = ap.parse_args()
    decode() if a.cmd == "decode" else score()
