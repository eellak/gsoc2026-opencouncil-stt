"""Calibrate full-recipe training variation on the 39 frozen validation windows.

The three A adapters are independent runs of the current corrected recipe at seeds
13, 29 and 47. Hypothesis text stays below ``~/.cache/oc-public``; only aggregate
S/D/I counts and variation summaries may be written inside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import decode_ablation as DA  # noqa: E402
from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402

ARMS = {
    "a_s13": {
        "seed": 13,
        "adapter": Path("/home/harold/oc-mix-adapters/adapter_A_s13"),
    },
    "a_s29": {
        "seed": 29,
        "adapter": Path("/home/harold/oc-mix-adapters/adapter_A_s29"),
    },
    "a_s47": {
        "seed": 47,
        "adapter": Path("/home/harold/oc-mix-adapters/adapter_A_s47"),
    },
}
for name, arm in ARMS.items():
    arm["model"] = Path.home() / ".cache/oc-public/seed-calibration-2026-08" / name / "ct2"


def cache_dir() -> Path:
    path = Path.home() / ".cache/oc-public/seed-calibration-2026-08"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode(name: str) -> None:
    import ctranslate2
    from faster_whisper import WhisperModel

    arm = ARMS[name]
    model_path = arm["model"]
    if not model_path.joinpath("model.bin").exists():
        raise SystemExit(f"missing CT2 model: {model_path}")
    dest = cache_dir() / f"decode-{name}.json"
    model_hash = sha256(model_path / "model.bin")
    state = json.loads(dest.read_text()) if dest.exists() else {
        "arm": name,
        "training_seed": arm["seed"],
        "adapter_sha256": sha256(arm["adapter"] / "adapter_model.safetensors"),
        "ct2_model_sha256": model_hash,
        "config": DA.CONTROL,
        "windows": {},
    }
    expected = {
        "arm": name,
        "training_seed": arm["seed"],
        "adapter_sha256": sha256(arm["adapter"] / "adapter_model.safetensors"),
        "ct2_model_sha256": model_hash,
        "config": DA.CONTROL,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise SystemExit(f"{dest}: stale {key}: {state.get(key)!r} != {value!r}")

    rows = DA.rows("eval")
    todo = [row for row in rows if row["window_id"] not in state["windows"]]
    print(f"{name}: {len(todo)} windows to decode ({len(state['windows'])} done)", flush=True)
    if not todo:
        return

    model = WhisperModel(str(model_path), device=DA.DEVICE, compute_type=DA.COMPUTE,
                         cpu_threads=DA.THREADS)
    diag = DA.Diagnostics()
    fw_log = logging.getLogger("faster_whisper")
    previous_level = fw_log.level
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)
    try:
        for idx, row in enumerate(todo, 1):
            wid = row["window_id"]
            wav = DA.sc() / "bench_windows" / f"{wid}.wav"
            if not wav.exists():
                raise SystemExit(f"missing frozen audio: {wav}")
            diag.reset()
            ctranslate2.set_random_seed(DA.seed_for("A", wid))
            started = time.time()
            segments = list(model.transcribe(str(wav), **DA.CONTROL)[0])
            state["windows"][wid] = {
                "text": "".join(segment.text for segment in segments).strip(),
                "wall_seconds": round(time.time() - started, 1),
                "n_segments": len(segments),
                **diag.snapshot(),
            }
            dest.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            print(f"  {idx}/{len(todo)} {wid} {state['windows'][wid]['wall_seconds']}s",
                  flush=True)
    finally:
        fw_log.removeHandler(diag)
        fw_log.setLevel(previous_level)


def counts_for_texts(texts: dict[str, str]) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    for row in DA.rows("eval"):
        wid = row["window_id"]
        if wid not in texts:
            raise SystemExit(f"incomplete decode: {wid} missing")
        reference = ftoks(DA.reference_text(wid))
        hypothesis = ftoks(texts[wid])
        out[wid] = (*sdi(reference, hypothesis), len(reference))
    return out


def load_arm_counts(name: str) -> dict[str, tuple[int, int, int, int]]:
    state = json.loads((cache_dir() / f"decode-{name}.json").read_text())
    return counts_for_texts({wid: row["text"] for wid, row in state["windows"].items()})


def load_fixed_counts() -> dict[str, tuple[int, int, int, int]]:
    state = json.loads((DA.out_dir() / "eval-A.json").read_text())
    return counts_for_texts({wid: row["text"] for wid, row in state["windows"].items()})


def rates(counts: dict[str, tuple[int, int, int, int]]) -> dict[str, float | int]:
    rows = list(counts.values())
    sub, delete, insert, n_ref = (sum(row[i] for row in rows) for i in range(4))
    return {
        "sub": sub,
        "del": delete,
        "ins": insert,
        "ref_tokens": n_ref,
        "wer": (sub + delete + insert) / n_ref,
        "sub_rate": sub / n_ref,
        "del_rate": delete / n_ref,
        "ins_rate": insert / n_ref,
        "recall_error_rate": (sub + delete) / n_ref,
        "hyp_ref_ratio": (n_ref - delete + insert) / n_ref,
    }


def normal_approx_seed_count(sigma_points: float, effect_points: float = 1.0) -> int:
    """Paired-seed count for 80% power and a two-sided 5% normal test.

    This is deliberately labelled an approximation: three calibration runs cannot
    support a precise power claim, and raw control SD is not paired-effect SD.
    """
    return math.ceil(((1.959964 + 0.841621) * sigma_points / effect_points) ** 2)


def range_influence(counts: dict[str, dict[str, tuple[int, int, int, int]]],
                    totals: dict[str, dict]) -> dict:
    high = max(totals, key=lambda name: totals[name]["wer"])
    low = min(totals, key=lambda name: totals[name]["wer"])
    net = (totals[high]["sub"] + totals[high]["del"] + totals[high]["ins"]
           - totals[low]["sub"] - totals[low]["del"] - totals[low]["ins"])
    contributions = []
    for row in DA.rows("eval"):
        wid = row["window_id"]
        a = counts[high][wid]
        b = counts[low][wid]
        contribution = sum(a[:3]) - sum(b[:3])
        contributions.append((abs(contribution), contribution, wid))
    _, top, wid = max(contributions)
    return {
        "high": high,
        "low": low,
        "net_error_difference": net,
        "largest_window": wid,
        "largest_window_error_difference": top,
        "largest_window_share_of_net": abs(top / net) if net else None,
    }


def score(output: Path) -> None:
    counts = {name: load_arm_counts(name) for name in ARMS}
    totals = {name: rates(value) for name, value in counts.items()}
    fixed_counts = load_fixed_counts()
    fixed = rates(fixed_counts)
    wer_points = [totals[name]["wer"] * 100 for name in ARMS]
    sigma = statistics.stdev(wer_points)
    same_seed_delta = (totals["a_s13"]["wer"] - fixed["wer"]) * 100
    result = {
        "experiment": "exp-2026-08-18-training-measurement-rule",
        "set": "eval-freeze-2026-08 validation",
        "n_windows": len(DA.rows("eval")),
        "n_meetings": len({row["meeting_id"] for row in DA.rows("eval")}),
        "decode_config": DA.CONTROL,
        "controls": totals,
        "fixed_same_recipe_seed13": fixed,
        "calibration": {
            "wer_points": {name: totals[name]["wer"] * 100 for name in ARMS},
            "range_points": max(wer_points) - min(wer_points),
            "sample_sd_points": sigma,
            "same_seed_repeat_delta_points_a_s13_minus_fixed": same_seed_delta,
            "naive_control_only_normal_approx_for_1pt_80pct_power_two_sided_5pct":
                normal_approx_seed_count(sigma),
            "confirmation_paired_seeds_per_arm": None,
            "screen_paired_seeds_per_arm": 3,
            "warning": (
                "n=3 raw control outcomes. The SD is imprecise and is not the SD of paired "
                "treatment effects. The naive normal approximation is reported only to show "
                "why control-only variance cannot size confirmation; estimate paired-effect "
                "variance from the 300-step control/treatment pilot."
            ),
        },
        "range_influence": range_influence(counts, totals),
        "same_seed_repeat_caveat": (
            "artifact-adapter-fixed and A_s13 have matching run_meta fields but different "
            "weight hashes. The delta is descriptive run nondeterminism; identical batch order "
            "and hardware provenance were not recorded."
        ),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    decode_parser = sub.add_parser("decode")
    decode_parser.add_argument("arm", choices=sorted(ARMS))
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--output", type=Path,
                              default=ROOT / "eval/results_seed_calibration.json")
    args = parser.parse_args()
    if args.command == "decode":
        decode(args.arm)
    else:
        score(args.output)


if __name__ == "__main__":
    main()
