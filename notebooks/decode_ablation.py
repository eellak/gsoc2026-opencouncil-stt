"""Decode the frozen evaluation windows under six decode configurations.

`exp-2026-08-12-decode-ablation`. Arms, options, estimands and gates are fixed in
[`docs/specs/2026-08-12-decode-ablation-prereg.md`](../docs/specs/2026-08-12-decode-ablation-prereg.md)
and were committed before this ran. Nothing here may be tuned against a result.

Reuses, rather than re-implements:

- `eval.controlled_eval.scoring.wtoks` and `.cluster_bootstrap` — the frozen
  normalizer and the meeting-clustered paired bootstrap;
- `eval.controlled_eval.exp_same_stack.sdi` — the S/D/I alignment, ties toward
  substitution, which is the code behind `artifact-bench-hyps-fw-finetune`;
- `eval.controlled_eval.eval_freeze.ftoks` / `FILLER_RE` — the frozen filler strip;
- the window audio in `$SC/bench_windows` (`artifact-bench-240-audio`, extended by
  `eval.controlled_eval.cut_freeze_audio`).

Hypotheses and per-window scores stay under `$SC`. Only aggregates go in the repo.

    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/decode_ablation.py smoke
    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/decode_ablation.py decode --arm A
    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/decode_ablation.py score
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap  # noqa: E402

MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
MODEL_DIR = "/home/harold/oc-asr-serve/ct2-fixed"
DEVICE, COMPUTE, THREADS = "cpu", "int8", 16
BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED = 4000, 7

# The control: the server's own call, with every threshold faster-whisper would have
# defaulted to written out explicitly. See the prereg for the two deviations
# (beam is the code default 5, not the mini-PC env's 2; word_timestamps False).
CONTROL = {
    "language": "el",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "word_timestamps": False,
    "vad_filter": False,
    "task": "transcribe",
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
}

ARMS = {
    "A": {},                                        # control
    "B": {"no_speech_threshold": 0.8},              # anti-deletion, exploratory
    "C": {"no_speech_threshold": None},             # anti-deletion, PRIMARY
    "D": {"temperature": [0.0]},                    # anti-insertion, PRIMARY
    "E": {"log_prob_threshold": None},              # compound, exploratory
    "F": {"compression_ratio_threshold": None},     # exploratory
}
PRIMARY = {"C": "del_rate", "D": "ins_rate"}
FAMILY = {"B": "del_rate", "C": "del_rate", "D": "ins_rate", "E": "ins_rate",
          "F": "ins_rate"}


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def out_dir() -> Path:
    d = sc() / "decode-ablation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(m):
    print(m, flush=True)


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def rows(which: str) -> list[dict]:
    key = {"eval": "eval_windows", "holdout": "holdout_windows"}[which]
    return manifest()[key]


def seed_for(arm: str, window_id: str) -> int:
    """Stable per-WINDOW seed, deliberately independent of the arm.

    Positive fallback temperatures sample, so without a seed the arms are not
    reproducible. The handoff plan asked for a seed derived from (arm, window_id) -
    that reproduces, but it also makes every arm draw a *different* random sequence,
    so an arm whose knob changed nothing still came out with different text. Measured
    before scoring: with the no-speech gate never firing on these 39 windows, arm B
    should have been byte-identical to the control and differed on 4 windows - the
    exact 4 that reached a sampling temperature.

    Seeding on the window alone is common random numbers: the standard paired-design
    variance reduction. An inert knob now produces identical output, and any
    difference is attributable to the knob rather than to the draw.
    """
    _ = arm  # deliberately unused; see above
    h = hashlib.sha256(window_id.encode()).digest()
    return int.from_bytes(h[:4], "big")


class Diagnostics(logging.Handler):
    """faster-whisper reports every threshold decision on its debug logger.

    A skipped segment produces no Segment object, so the skip count cannot be
    recovered from the output - only from here. Without it a null result cannot
    distinguish "the knob did nothing" from "the condition never arose".
    """

    NO_SPEECH = re.compile(r"No speech threshold is met \(([\d.]+) > ([\d.]+)\)")
    COMPRESSION = re.compile(r"Compression ratio threshold is not met with temperature ([\d.]+)")
    LOGPROB = re.compile(r"Log probability threshold is not met with temperature ([\d.]+)")

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.reset()

    def reset(self):
        self.no_speech_skips = []
        self.compression_trips = []
        self.logprob_trips = []

    def emit(self, record):
        msg = record.getMessage()
        if (m := self.NO_SPEECH.search(msg)):
            self.no_speech_skips.append(float(m.group(1)))
        elif (m := self.COMPRESSION.search(msg)):
            self.compression_trips.append(float(m.group(1)))
        elif (m := self.LOGPROB.search(msg)):
            self.logprob_trips.append(float(m.group(1)))

    def snapshot(self) -> dict:
        return {"no_speech_skips": len(self.no_speech_skips),
                "no_speech_probs": [round(x, 4) for x in self.no_speech_skips],
                "compression_trips": len(self.compression_trips),
                "logprob_trips": len(self.logprob_trips)}


def opts_to_dict(opts) -> dict:
    if dataclasses.is_dataclass(opts):
        d = dataclasses.asdict(opts)
    elif hasattr(opts, "_asdict"):
        d = dict(opts._asdict())
    else:
        d = {k: getattr(opts, k) for k in dir(opts) if not k.startswith("_")}
    # suppress_tokens is ~100 ints of noise; keep its length, not its contents.
    if isinstance(d.get("suppress_tokens"), (list, tuple)):
        d["suppress_tokens"] = f"<{len(d['suppress_tokens'])} tokens>"
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool,
                                                          list, tuple, type(None)))}


# ------------------------------------------------------------------------ decode
def decode(arm: str, which: str, limit: int | None, model=None) -> Path:
    import ctranslate2
    from faster_whisper import WhisperModel

    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    kw = dict(CONTROL, **ARMS[arm])

    dest = out_dir() / f"{which}-{arm}.json"
    state = json.loads(dest.read_text()) if dest.exists() else {"arm": arm, "set": which,
                                                                "config": kw, "windows": {}}
    todo = [r for r in rows(which)[:limit] if r["window_id"] not in state["windows"]]
    log(f"arm {arm} [{which}]: {len(todo)} windows to decode "
        f"({len(state['windows'])} already done)")
    if not todo:
        return dest

    if model is None:
        model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE,
                             cpu_threads=THREADS)

    diag = Diagnostics()
    fw_log = logging.getLogger("faster_whisper")
    prev_level = fw_log.level
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)
    try:
        for i, r in enumerate(todo, 1):
            wav = sc() / "bench_windows" / f"{r['window_id']}.wav"
            if not wav.exists():
                raise SystemExit(f"missing audio for {r['window_id']} - the manifest "
                                 f"says it should be there")
            diag.reset()
            ctranslate2.set_random_seed(seed_for(arm, r["window_id"]))
            t0 = time.time()
            segments, info = model.transcribe(str(wav), **kw)
            segs = list(segments)          # consume fully before recording anything
            elapsed = time.time() - t0

            state["windows"][r["window_id"]] = {
                "text": "".join(s.text for s in segs).strip(),
                "n_segments": len(segs),
                "decoded_seconds": round(sum(s.end - s.start for s in segs), 2),
                "audio_seconds": round(info.duration, 2),
                "temperatures_used": sorted({s.temperature for s in segs
                                             if s.temperature is not None}),
                "seed": seed_for(arm, r["window_id"]),
                "wall_seconds": round(elapsed, 1),
                **diag.snapshot(),
            }
            state.setdefault("resolved_options", opts_to_dict(info.transcription_options))
            # Written every window: this runs for hours and a pass that cannot be
            # resumed is a pass that gets restarted from zero.
            dest.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            log(f"  {arm} {i}/{len(todo)} {r['window_id']} {elapsed:.0f}s "
                f"segs={len(segs)} skips={diag.snapshot()['no_speech_skips']}")
    finally:
        fw_log.removeHandler(diag)
        fw_log.setLevel(prev_level)
    log(f"arm {arm} [{which}] -> {dest}")
    return dest


def smoke(which: str) -> None:
    """One short window through every arm, to prove diagnostics land before scaling."""
    import ctranslate2
    from faster_whisper import WhisperModel

    r = min(rows(which), key=lambda x: x["duration_sec"])
    log(f"smoke window: {r['window_id']} ({r['duration_sec']}s, "
        f"{r['ref_tokens']} ref tokens)")
    wav = sc() / "bench_windows" / f"{r['window_id']}.wav"
    model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE,
                         cpu_threads=THREADS)
    diag = Diagnostics()
    fw_log = logging.getLogger("faster_whisper")
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)

    ref = ftoks(reference_text(r["window_id"]))
    for arm in ARMS:
        kw = dict(CONTROL, **ARMS[arm])
        diag.reset()
        ctranslate2.set_random_seed(seed_for(arm, r["window_id"]))
        t0 = time.time()
        segs = list(model.transcribe(str(wav), **kw)[0])
        hyp = ftoks("".join(s.text for s in segs))
        s, d, ins = sdi(ref, hyp)
        log(f"  {arm}: {time.time()-t0:5.0f}s  segs={len(segs):2d}  "
            f"WER={(s+d+ins)/len(ref):.4f} (S{s} D{d} I{ins})  "
            f"{diag.snapshot()}")
    fw_log.removeHandler(diag)


# ------------------------------------------------------------------------- score
_REFS: dict[str, str] | None = None


def reference_text(window_id: str) -> str:
    """Reference text from the cached benchmark report. Never written to the repo."""
    global _REFS
    if _REFS is None:
        from eval.controlled_eval import bench_data as B
        rep = B.load_report(manifest()["source_run"])
        _REFS = {it["itemId"]: it["referenceText"] for it in rep["items"]}
    return _REFS[window_id]


def per_window(which: str, arm: str) -> dict[str, tuple[int, int, int, int]]:
    p = out_dir() / f"{which}-{arm}.json"
    if not p.exists():
        raise SystemExit(f"arm {arm} [{which}] not decoded: {p} missing")
    state = json.loads(p.read_text())
    out = {}
    for r in rows(which):
        wid = r["window_id"]
        if wid not in state["windows"]:
            raise SystemExit(f"arm {arm} [{which}] is incomplete: {wid} missing. "
                             f"No complete-case subsets.")
        ref = ftoks(reference_text(wid))
        hyp = ftoks(state["windows"][wid]["text"])
        out[wid] = (*sdi(ref, hyp), len(ref))
    return out


def rates(scores: dict) -> dict:
    v = list(scores.values())
    s, d, i, n = (sum(x[k] for x in v) for k in range(4))
    return {"sub": s, "del": d, "ins": i, "ref_tokens": n,
            "wer": (s + d + i) / n, "del_rate": d / n, "ins_rate": i / n,
            "sub_rate": s / n}


def paired_ci(a: dict, b: dict, wids: list[str], blocks: list[str], pick) -> dict:
    """CI on rate(a) - rate(b) for one error class, resampling meetings."""
    ca = [(pick(a[w]), a[w][3]) for w in wids]
    cb = [(pick(b[w]), b[w][3]) for w in wids]
    return cluster_bootstrap(ca, cb, blocks, n_boot=BOOTSTRAP_REPLICATES,
                             seed=BOOTSTRAP_SEED)


PICK = {"wer": lambda t: t[0] + t[1] + t[2], "del_rate": lambda t: t[1],
        "ins_rate": lambda t: t[2]}


def influence(a: dict, b: dict, wids: list[str], pick) -> dict:
    """Leave-one-window-out on the pooled delta. Does one window carry the sign?"""
    def delta(keep):
        na = sum(a[w][3] for w in keep)
        return (sum(pick(a[w]) for w in keep) - sum(pick(b[w]) for w in keep)) / na
    full = delta(wids)
    worst_id, worst = None, full
    reversed_by = []
    for w in wids:
        d = delta([x for x in wids if x != w])
        if (full < 0) != (d < 0):
            reversed_by.append(w)
        if abs(d - full) > abs(worst - full):
            worst_id, worst = w, d
    return {"delta": full, "max_shift_window": worst_id, "delta_without": worst,
            "sign_reversed_by": reversed_by}


def score(which: str) -> dict:
    wids = [r["window_id"] for r in rows(which)]
    blocks = [r["meeting_id"] for r in rows(which)]
    scores = {arm: per_window(which, arm) for arm in ARMS}
    ctrl = scores["A"]

    result = {"set": which, "n_windows": len(wids), "n_meetings": len(set(blocks)),
              "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                            "blocks": "meeting_id"},
              "arms": {}}
    for arm in ARMS:
        st = json.loads((out_dir() / f"{which}-{arm}.json").read_text())
        row = {"config_change": ARMS[arm], "totals": rates(scores[arm]),
               "diagnostics": {
                   k: sum(w[k] for w in st["windows"].values())
                   for k in ("no_speech_skips", "compression_trips", "logprob_trips",
                             "n_segments")},
               "wall_seconds": round(sum(w["wall_seconds"]
                                         for w in st["windows"].values()), 1)}
        if arm != "A":
            row["vs_control"] = {}
            for metric, pick in PICK.items():
                ci = paired_ci(scores[arm], ctrl, wids, blocks, pick)
                row["vs_control"][metric] = {"delta": ci["delta"], "ci95": ci["ci95"]}
            row["influence"] = influence(scores[arm], ctrl, wids,
                                         PICK[FAMILY.get(arm, "wer")])
            row["gate"] = gate(arm, row["vs_control"])
        result["arms"][arm] = row
    return result


def gate(arm: str, vs: dict) -> dict:
    """The preregistered gate. Upper bound of the 95% interval, margin 0.0."""
    fam = FAMILY[arm]
    target_hi = vs[fam]["ci95"][1]
    passed = target_hi < 0 and vs["wer"]["ci95"][1] <= 0
    if fam == "ins_rate":
        # The anti-insertion family must also not buy its gain with deletions.
        # The anti-deletion family is gated on deletion + WER only, per the prereg.
        passed = passed and vs["del_rate"]["ci95"][1] <= 0
    return {"family": fam, "target_upper_bound": target_hi,
            "wer_upper_bound": vs["wer"]["ci95"][1],
            "del_upper_bound": vs["del_rate"]["ci95"][1],
            "passed": bool(passed),
            "primary": arm in PRIMARY,
            "note": ("ship candidate" if passed and arm in PRIMARY else
                     "exploratory - cannot ship alone" if passed else "did not pass")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "decode", "score"):
        p = sub.add_parser(name)
        p.add_argument("--set", dest="which", default="eval",
                       choices=("eval", "holdout"))
        if name == "decode":
            p.add_argument("--arm", required=True)
            p.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if a.cmd == "smoke":
        smoke(a.which)
    elif a.cmd == "decode":
        for arm in (list(ARMS) if a.arm == "all" else a.arm.split(",")):
            decode(arm, a.which, a.limit)
    else:
        res = score(a.which)
        dest = out_dir() / f"results-{a.which}.json"
        dest.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        for arm, row in res["arms"].items():
            t = row["totals"]
            line = (f"{arm}  WER {t['wer']:.4f}  del {t['del_rate']:.4f}  "
                    f"ins {t['ins_rate']:.4f}  skips {row['diagnostics']['no_speech_skips']:3d}")
            if "gate" in row:
                g = row["gate"]
                line += f"  | {g['note']}"
            log(line)
        log(f"-> {dest}")


if __name__ == "__main__":
    main()
