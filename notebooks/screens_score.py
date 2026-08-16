"""Score a training-screen adapter against `artifact-ct2-fixed` on the frozen windows.

`exp-2026-08-14-external-packs` (RUN1 in-domain deletion mix, RUN2 external-packs
stage-1 -> stage-2). Same path that scored `artifact-adapter-correction-only`:
everything except the model directory is imported from `notebooks/decode_ablation.py`,
so the screens cannot drift away from the control decode configuration. The control
arm is `eval-A.json` from the decode ablation — a decode of `artifact-ct2-fixed` on
this machine, in this environment, under this exact config.

The decision tree these numbers feed is frozen in
[`docs/specs/2026-08-16-screens-handoff.md`](../docs/specs/2026-08-16-screens-handoff.md)
and must not be re-derived from the results.

    ARM=run2-stage2 SC=~/.cache/oc-public .venv-eval/bin/python notebooks/screens_score.py decode
    ARM=run2-stage2 SC=~/.cache/oc-public .venv-eval/bin/python notebooks/screens_score.py score
    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/screens_score.py pair run2-stage2 run1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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

CTRL_LABEL = "control (artifact-ct2-fixed)"
PICK = dict(DA.PICK, sub_rate=lambda t: t[0])

ARMS = {
    "run1": {
        "model": "/home/harold/oc-run1-screen/ct2",
        "label": "run1-screen (in-domain targeted-deletion mix, seed 101)",
        "dir": "run1-eval",
    },
    "run2-stage2": {
        "model": "/home/harold/oc-run2-stage2/ct2",
        "label": "run2-screen stage2 (external packs stage-1 -> in-domain stage-2, seed 101)",
        "dir": "run2-eval-stage2",
    },
    "run2-stage1": {
        "model": "/home/harold/oc-run2-stage1/ct2",
        "label": "run2-screen stage1 (external packs only, seed 101) - EXTRA, intermediate",
        "dir": "run2-eval-stage1",
    },
}


def arm(name: str | None = None) -> dict:
    name = name or os.environ.get("ARM", "")
    if name not in ARMS:
        raise SystemExit(f"set ARM to one of {sorted(ARMS)} (got {name!r})")
    a = dict(ARMS[name])
    a["dest"] = DA.sc() / "train-screens-2026-08" / a["dir"] / "decode.json"
    a["name"] = name
    return a


def log(m):
    print(m, flush=True)


def check_state(state: dict, a: dict, dest: Path) -> None:
    """Refuse to extend or score hypotheses that came from other weights or config.

    Both halves matter: a stale `decode.json` under the right filename would
    silently score the wrong adapter, and a decode produced under a different
    decode configuration is not comparable to the control arm.
    """
    if state.get("model") != a["model"]:
        raise SystemExit(f"{dest} holds a decode of {state.get('model')}, "
                         f"not {a['model']}")
    if state.get("config") != DA.CONTROL:
        raise SystemExit(f"{dest} was decoded under a different config than the "
                         f"frozen control configuration")


def decode() -> None:
    import ctranslate2
    from faster_whisper import WhisperModel

    a = arm()
    dest = a["dest"]
    rows = DA.rows("eval")
    dest.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(dest.read_text()) if dest.exists() else {
        "model": a["model"], "config": DA.CONTROL, "windows": {}}
    check_state(state, a, dest)
    todo = [r for r in rows if r["window_id"] not in state["windows"]]
    log(f"{a['name']}: {len(todo)} windows to decode ({len(state['windows'])} done)")
    if not todo:
        return

    model = WhisperModel(a["model"], device=DA.DEVICE, compute_type=DA.COMPUTE,
                         cpu_threads=DA.THREADS)
    diag = DA.Diagnostics()
    fw_log = logging.getLogger("faster_whisper")
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)
    try:
        for i, r in enumerate(todo, 1):
            wav = DA.sc() / "bench_windows" / f"{r['window_id']}.wav"
            diag.reset()
            # Common random numbers: the same per-window seed the control used.
            ctranslate2.set_random_seed(DA.seed_for("A", r["window_id"]))
            t0 = time.time()
            segs = list(model.transcribe(str(wav), **DA.CONTROL)[0])
            state["windows"][r["window_id"]] = {
                "text": "".join(s.text for s in segs).strip(),
                "n_segments": len(segs),
                "wall_seconds": round(time.time() - t0, 1),
                **diag.snapshot()}
            dest.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            log(f"  {i}/{len(todo)} {r['window_id']} "
                f"{state['windows'][r['window_id']]['wall_seconds']}s")
    finally:
        fw_log.removeHandler(diag)
    log(f"-> {dest}")


def counts(texts: dict[str, str]) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    for r in DA.rows("eval"):
        wid = r["window_id"]
        if wid not in texts:
            raise SystemExit(f"incomplete: {wid} missing. No complete-case subsets.")
        ref = ftoks(DA.reference_text(wid))
        out[wid] = (*sdi(ref, ftoks(texts[wid])), len(ref))
    return out


def load_counts(path: Path, a: dict | None = None) -> dict[str, tuple[int, int, int, int]]:
    state = json.loads(path.read_text())
    if a is not None:
        check_state(state, a, path)
    return counts({k: v["text"] for k, v in state["windows"].items()})


SCREEN_LABEL = ("SCREEN. One seed (101) against a measured 2.1-point per-seed spread "
                "(exp-2026-08-08-mixture-ratio). Single-seed screen results decide "
                "nothing by themselves.")


def compare(a_counts, a_label, b_counts, b_label, extra: dict) -> dict:
    rows = DA.rows("eval")
    wids = [r["window_id"] for r in rows]
    blocks = [r["meeting_id"] for r in rows]
    res = {"experiment": "exp-2026-08-14-external-packs",
           "decode_config": DA.CONTROL,
           "n_windows": len(wids), "n_meetings": len(set(blocks)),
           "bootstrap": {"replicates": DA.BOOTSTRAP_REPLICATES,
                         "seed": DA.BOOTSTRAP_SEED, "blocks": "meeting_id"},
           **extra,
           "totals": {b_label: DA.rates(b_counts), a_label: DA.rates(a_counts)},
           "delta": {}}
    for metric, pick in PICK.items():
        ca = [(pick(a_counts[w]), a_counts[w][3]) for w in wids]
        cb = [(pick(b_counts[w]), b_counts[w][3]) for w in wids]
        ci = cluster_bootstrap(ca, cb, blocks, n_boot=DA.BOOTSTRAP_REPLICATES,
                               seed=DA.BOOTSTRAP_SEED)
        res["delta"][metric] = {"delta": ci["delta"], "ci95": ci["ci95"],
                                "excludes_zero": ci["excludes_zero"]}
    res["influence"] = {m: DA.influence(a_counts, b_counts, wids, PICK[m])
                        for m in ("wer", "del_rate", "ins_rate", "sub_rate")}
    res["head2head"] = {
        "arm_better": sum(1 for w in wids if sum(a_counts[w][:3]) < sum(b_counts[w][:3])),
        "tie": sum(1 for w in wids if sum(a_counts[w][:3]) == sum(b_counts[w][:3])),
        "other_better": sum(1 for w in wids if sum(a_counts[w][:3]) > sum(b_counts[w][:3]))}
    res["label"] = SCREEN_LABEL
    return res


def report(res: dict, out: Path) -> None:
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for label, t in res["totals"].items():
        log(f"{label:70s} WER {t['wer']:.4f}  del {t['del_rate']:.4f}  "
            f"ins {t['ins_rate']:.4f}  sub {t['sub_rate']:.4f}  "
            f"S{t['sub']} D{t['del']} I{t['ins']}")
    for m, v in res["delta"].items():
        log(f"  {m:9s} {v['delta']:+.5f} [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}]"
            f"{'  excludes zero' if v['excludes_zero'] else ''}")
    log(f"  head2head {res['head2head']}")
    log(f"  influence {json.dumps(res['influence'])}")
    log(f"-> {out}")


def score() -> None:
    a = arm()
    arm_counts = load_counts(a["dest"], a)
    ctrl_counts = load_counts(DA.out_dir() / "eval-A.json")
    res = compare(arm_counts, a["label"], ctrl_counts, CTRL_LABEL,
                  {"comparison": f"{a['name']} vs control",
                   "arm_model": a["model"],
                   "ct2_sha256": Path(a["model"]).parent.joinpath("ct2.sha256").read_text()
                   if Path(a["model"]).parent.joinpath("ct2.sha256").exists() else None})
    report(res, a["dest"].parent / "results.json")


def pair(name_a: str, name_b: str) -> None:
    """Paired per-window comparison of two screen arms (no control involved)."""
    a, b = arm(name_a), arm(name_b)
    res = compare(load_counts(a["dest"], a), a["label"], load_counts(b["dest"], b), b["label"],
                  {"comparison": f"{a['name']} vs {b['name']}",
                   "arm_model": a["model"], "other_model": b["model"]})
    out = DA.sc() / "train-screens-2026-08" / f"pair-{a['name']}-vs-{b['name']}.json"
    report(res, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("decode", "score", "pair"))
    ap.add_argument("names", nargs="*")
    ns = ap.parse_args()
    if ns.cmd == "decode":
        decode()
    elif ns.cmd == "score":
        score()
    else:
        if len(ns.names) != 2:
            raise SystemExit("pair needs two arm names")
        pair(*ns.names)
