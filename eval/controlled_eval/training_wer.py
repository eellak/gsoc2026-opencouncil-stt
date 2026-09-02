"""Measure training-set WER on a frozen, outcome-blind CPU sample.

The sample is a deterministic simple random sample of 300 rows from the exact
training parquet used by ``artifact-adapter-fixed``.  Transcript text, audio clips
and hypotheses stay below ``~/.cache/oc-public``.  Only aggregates are written to
the repository.

    .venv-eval/bin/python -m eval.controlled_eval.training_wer prepare
    .venv-eval/bin/python -m eval.controlled_eval.training_wer decode fixed
    .venv-eval/bin/python -m eval.controlled_eval.training_wer decode base
    .venv-eval/bin/python -m eval.controlled_eval.training_wer score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import decode_ablation as DA  # noqa: E402
from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap  # noqa: E402

TRAIN_PARQUET = ROOT / "data/hf-dataset/public/train.parquet"
TRAIN_PARQUET_SHA256 = "f37d3532b4b60389172e479326a025a3ccb8197f07b23ad441218a7a7585c1e8"
SAMPLE_SIZE = 300
SAMPLE_SALT = "training-wer-v1"
PAD_SECONDS = 0.2

ARMS = {
    "base": {
        "artifact": "artifact-ct2-base-large-v3",
        "model": Path.home() / (
            ".cache/huggingface/hub/models--Systran--faster-whisper-large-v3/"
            "snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
        ),
        "model_sha256_16": "69f74147e3334731",
    },
    "fixed": {
        "artifact": "artifact-ct2-fixed",
        "model": Path("/home/harold/oc-asr-serve/ct2-fixed"),
        "model_sha256_16": "8a1a3b257d0c1bdb",
    },
    "july": {
        "artifact": "artifact-ct2-july-broken",
        "model": Path("/home/harold/oc-asr-serve/ct2"),
        "model_sha256_16": "dfffde80906f2cd5",
    },
}


def work_dir() -> Path:
    path = Path.home() / ".cache/oc-public/training-wer-2026-08-n300"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def row_rank(utterance_id: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SALT}|{utterance_id}".encode()).hexdigest()


def safe_key(utterance_id: str) -> str:
    return hashlib.sha256(utterance_id.encode()).hexdigest()[:16]


def choose_sample(rows: list[dict], n: int = SAMPLE_SIZE) -> list[dict]:
    """Outcome-blind simple random sample implemented as a stable hash ordering."""
    if len({row["utterance_id"] for row in rows}) != len(rows):
        raise ValueError("utterance_id must be unique")
    if len(rows) < n:
        raise ValueError(f"need at least {n} rows, got {len(rows)}")
    return sorted(rows, key=lambda row: row_rank(row["utterance_id"]))[:n]


def usable_span(row: dict) -> tuple[float, float]:
    def good(value) -> bool:
        return value is not None and not (isinstance(value, float) and math.isnan(value))

    start = row.get("start_adj") if good(row.get("start_adj")) else row["start"]
    end = row.get("end_adj") if good(row.get("end_adj")) else row["end"]
    return max(0.0, float(start) - PAD_SECONDS), float(end) + PAD_SECONDS


def sample_fingerprint(rows: list[dict]) -> str:
    payload = [
        {
            "utterance_id": row["utterance_id"],
            "city_id": row["city_id"],
            "meeting_id": row["meeting_id"],
            "span": [round(value, 6) for value in usable_span(row)],
        }
        for row in rows
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_parquet_rows() -> list[dict]:
    import pandas as pd

    actual = sha256(TRAIN_PARQUET)
    if actual != TRAIN_PARQUET_SHA256:
        raise SystemExit(f"training parquet hash mismatch: {actual}")
    return pd.read_parquet(TRAIN_PARQUET).to_dict("records")


def _cut_one(row: dict) -> dict:
    key = safe_key(row["utterance_id"])
    clip = work_dir() / "clips" / f"{key}.wav"
    if not clip.exists():
        clip.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / "data/asr/audio" / f"{row['city_id']}__{row['meeting_id']}.mp3"
        if not source.exists():
            raise FileNotFoundError(source)
        start, end = usable_span(row)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.6f}", "-to",
             f"{end:.6f}", "-i", str(source), "-ac", "1", "-ar", "16000", str(clip)],
            check=True,
        )
    return {
        "key": key,
        "utterance_id": row["utterance_id"],
        "city_id": row["city_id"],
        "meeting_id": row["meeting_id"],
        "source": row["source"],
        "duration_s": float(row["duration_s"]),
        "reference": row["text"],
        "clip": str(clip),
    }


def prepare() -> None:
    dest = work_dir() / "sample.json"
    if dest.exists():
        existing = json.loads(dest.read_text())
        if len(existing.get("rows", [])) == SAMPLE_SIZE:
            print(f"sample already frozen: {dest}")
            return
    population = load_parquet_rows()
    selected = choose_sample(population)
    fingerprint = sample_fingerprint(selected)
    if dest.exists():
        frozen = json.loads(dest.read_text())
        if frozen.get("selection", {}).get("sample_sha256") != fingerprint:
            raise SystemExit("incomplete frozen sample has the wrong fingerprint")
    else:
        # The sample identity is persisted before any model is loaded or WER computed.
        frozen = {
            "frozen_unix": time.time(),
            "selection": {
                "method": "lowest SHA-256 ranks of salt + utterance_id",
                "salt": SAMPLE_SALT,
                "sample_size": SAMPLE_SIZE,
                "population_rows": len(population),
                "train_parquet_sha256": TRAIN_PARQUET_SHA256,
                "sample_sha256": fingerprint,
            },
            "rows": [],
        }
        dest.write_text(json.dumps(frozen, ensure_ascii=False, indent=1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        frozen["rows"] = list(pool.map(_cut_one, selected))
    if len(frozen["rows"]) != SAMPLE_SIZE:
        raise SystemExit("sample cut is incomplete")
    dest.write_text(json.dumps(frozen, ensure_ascii=False, indent=1))
    print(f"froze {SAMPLE_SIZE} rows as {fingerprint} at {dest}")


def load_sample() -> dict:
    path = work_dir() / "sample.json"
    if not path.exists():
        raise SystemExit("run prepare before decode")
    state = json.loads(path.read_text())
    rows = state.get("rows", [])
    if len(rows) != SAMPLE_SIZE:
        raise SystemExit(f"incomplete frozen sample: {len(rows)}/{SAMPLE_SIZE}")
    expected = sample_fingerprint(choose_sample(load_parquet_rows()))
    if state["selection"]["sample_sha256"] != expected:
        raise SystemExit("frozen sample fingerprint does not match the selection rule")
    return state


def decode(arm_name: str) -> None:
    import ctranslate2
    from faster_whisper import WhisperModel

    sample = load_sample()
    arm = ARMS[arm_name]
    model_hash = sha256(arm["model"] / "model.bin")
    if not model_hash.startswith(arm["model_sha256_16"]):
        raise SystemExit(f"{arm_name} model hash mismatch: {model_hash}")
    dest = work_dir() / f"decode-{arm_name}.json"
    state = json.loads(dest.read_text()) if dest.exists() else {
        "arm": arm_name,
        "artifact": arm["artifact"],
        "model_sha256": model_hash,
        "sample_sha256": sample["selection"]["sample_sha256"],
        "config": DA.CONTROL,
        "rows": {},
    }
    expected = {
        "artifact": arm["artifact"],
        "model_sha256": model_hash,
        "sample_sha256": sample["selection"]["sample_sha256"],
        "config": DA.CONTROL,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise SystemExit(f"stale decode {key}: {state.get(key)!r} != {value!r}")
    todo = [row for row in sample["rows"] if row["key"] not in state["rows"]]
    print(f"{arm_name}: {len(todo)} rows to decode ({len(state['rows'])} done)", flush=True)
    if not todo:
        return
    model = WhisperModel(str(arm["model"]), device=DA.DEVICE, compute_type=DA.COMPUTE,
                         cpu_threads=DA.THREADS)
    for index, row in enumerate(todo, 1):
        ctranslate2.set_random_seed(DA.seed_for("training-wer", row["key"]))
        started = time.time()
        segments = list(model.transcribe(row["clip"], **DA.CONTROL)[0])
        state["rows"][row["key"]] = {
            # Preserve segment boundaries so scoring cannot invent fused words.
            "segments": [segment.text for segment in segments],
            "wall_seconds": round(time.time() - started, 3),
        }
        dest.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        if index % 25 == 0 or index == len(todo):
            print(f"  {index}/{len(todo)}", flush=True)


def arm_counts(sample: dict, arm_name: str) -> dict[str, tuple[int, int, int, int]]:
    state = json.loads((work_dir() / f"decode-{arm_name}.json").read_text())
    if state["sample_sha256"] != sample["selection"]["sample_sha256"]:
        raise SystemExit(f"{arm_name}: sample mismatch")
    out = {}
    for row in sample["rows"]:
        decoded = state["rows"].get(row["key"])
        if decoded is None:
            raise SystemExit(f"{arm_name}: incomplete decode")
        reference = ftoks(row["reference"])
        hypothesis = [token for segment in decoded["segments"] for token in ftoks(segment)]
        out[row["key"]] = (*sdi(reference, hypothesis), len(reference))
    return out


def decode_wall_seconds(arm_name: str) -> float:
    state = json.loads((work_dir() / f"decode-{arm_name}.json").read_text())
    return sum(row["wall_seconds"] for row in state["rows"].values())


def rates(counts: dict[str, tuple[int, int, int, int]]) -> dict[str, float | int]:
    values = list(counts.values())
    sub, delete, insert, n_ref = (sum(row[i] for row in values) for i in range(4))
    return {
        "sub": sub,
        "del": delete,
        "ins": insert,
        "ref_tokens": n_ref,
        "wer": (sub + delete + insert) / n_ref,
        "sub_rate": sub / n_ref,
        "del_rate": delete / n_ref,
        "ins_rate": insert / n_ref,
    }


def dominance(counts: dict[str, tuple[int, int, int, int]], sample: dict) -> dict:
    by_key = {row["key"]: row for row in sample["rows"]}
    total = sum(sum(value[:3]) for value in counts.values())
    top_key, top_count = max(
        ((key, sum(value[:3])) for key, value in counts.items()), key=lambda item: item[1]
    )
    meetings: dict[tuple[str, str], int] = {}
    for key, value in counts.items():
        row = by_key[key]
        block = (row["city_id"], row["meeting_id"])
        meetings[block] = meetings.get(block, 0) + sum(value[:3])
    top_meeting, top_meeting_count = max(meetings.items(), key=lambda item: item[1])
    return {
        "largest_row_key": top_key,
        "largest_row_errors": top_count,
        "largest_row_share": top_count / total if total else None,
        "largest_meeting_key": hashlib.sha256("|".join(top_meeting).encode()).hexdigest()[:16],
        "largest_meeting_errors": top_meeting_count,
        "largest_meeting_share": top_meeting_count / total if total else None,
    }


def contrast_dominance(a: dict[str, tuple[int, int, int, int]],
                       b: dict[str, tuple[int, int, int, int]]) -> dict:
    contributions = {
        key: sum(a[key][:3]) - sum(b[key][:3])
        for key in a
    }
    net = sum(contributions.values())
    top_key, top_value = max(contributions.items(), key=lambda item: abs(item[1]))
    loo = [net - value for value in contributions.values()]
    return {
        "net_error_difference": net,
        "largest_row_key": top_key,
        "largest_row_error_difference": top_value,
        "largest_row_share_of_net": abs(top_value / net) if net else None,
        "leave_one_out_sign_reversal": any((value > 0) != (net > 0) for value in loo)
        if net else None,
    }


def score(output: Path) -> None:
    sample = load_sample()
    if not (work_dir() / "decode-fixed.json").exists():
        raise SystemExit("fixed arm is required")
    available = [name for name in ARMS if (work_dir() / f"decode-{name}.json").exists()]
    counts = {name: arm_counts(sample, name) for name in available}
    totals = {name: rates(value) for name, value in counts.items()}
    rows = sample["rows"]
    by_key = {row["key"]: row for row in rows}
    keys = [row["key"] for row in rows]
    clusters = [(row["city_id"], row["meeting_id"]) for row in rows]
    levels = {}
    for name in available:
        observed = [(sum(counts[name][key][:3]), counts[name][key][3]) for key in keys]
        zero = [(0, counts[name][key][3]) for key in keys]
        levels[name] = cluster_bootstrap(observed, zero, clusters, n_boot=4000, seed=7)
    paired = None
    if "july" in available:
        paired = cluster_bootstrap(
            [(sum(counts["july"][key][:3]), counts["july"][key][3]) for key in keys],
            [(sum(counts["fixed"][key][:3]), counts["fixed"][key][3]) for key in keys],
            clusters,
            n_boot=4000,
            seed=7,
        )
    result = {
        "experiment": "training-wer-2026-08-18",
        "metric": "agreement with the labels used for training; not fidelity to audio",
        "sample": {
            **sample["selection"],
            "runtime_pilot_note": (
                "An initial n=1000 freeze was stopped manually after 25 unscored fixed-arm "
                "decodes projected 80-100 minutes per arm. No hypotheses or WER were inspected. "
                "This n=300 prefix was then frozen in a new cache before any outcome was scored."
            ),
            "n_meetings": len({(row["city_id"], row["meeting_id"]) for row in rows}),
            "n_cities": len({row["city_id"] for row in rows}),
            "audio_seconds": sum(row["duration_s"] for row in rows),
            "source_rows": {
                source: sum(row["source"] == source for row in rows)
                for source in sorted({row["source"] for row in rows})
            },
        },
        "decode": {
            "device": DA.DEVICE,
            "compute_type": DA.COMPUTE,
            "cpu_threads": DA.THREADS,
            "config": DA.CONTROL,
            "normalizer": "eval.controlled_eval.eval_freeze.ftoks",
            "segment_join": "tokenize each segment, then concatenate token lists",
        },
        "validation_references": {
          "directly_comparable_primary": {
            "source": "~/.cache/oc-public/served-config-2026-08/results.json arm R",
            "set": "eval-freeze-2026-08 validation",
            "n_windows": 39,
            "n_meetings": 31,
            "ref_tokens": 11911,
            "artifact": "artifact-ct2-fixed",
            "sub": 919,
            "del": 683,
            "ins": 242,
            "wer": 0.15481487700444968,
            "sub_rate": 0.07715557048106791,
            "del_rate": 0.05734195281672404,
            "ins_rate": 0.02031735370665771,
            "why_primary": (
                "Same artifact, CONTROL options, CPU int8/16-thread stack, ftoks, "
                "per-segment token concatenation and DA.seed_for item-key policy as "
                "this training sample. Different items necessarily have different seeds."
            ),
          },
          "historical_not_directly_comparable": {
            "artifact_fixed_legacy_fused_join_wer": 0.1589287213500126,
            "artifact_fixed_legacy_fused_join_note": (
                "Same artifact/config/normalizer, but the old decode joined raw segment "
                "strings and can fuse words; do not use it for a numeric train-validation gap."
            ),
            "full_recipe_control_seed_wers": [
                0.1555704810679204,
                0.15582234908907733,
                0.15842498530769877,
            ],
            "full_recipe_control_note": (
                "A_s13/A_s29/A_s47 are distinct training artifacts, not artifact-ct2-fixed. "
                "They calibrate training-run variation and are reported as a range only."
            ),
          },
        },
        "arms": {
            name: {
                "artifact": ARMS[name]["artifact"],
                "decode_wall_seconds": decode_wall_seconds(name),
                **totals[name],
                "wer_ci95_meeting_clustered": levels[name]["ci95"],
                "by_label_source": {
                    source: rates({
                        key: value for key, value in counts[name].items()
                        if by_key[key]["source"] == source
                    })
                    for source in sorted({row["source"] for row in rows})
                },
                "dominance": dominance(counts[name], sample),
            }
            for name in available
        },
        "fixed_train_minus_validation": {
            "wer_delta": totals["fixed"]["wer"] - 0.15481487700444968,
            "status": "descriptive, not a pure generalisation-gap estimate",
            "caveat": (
                "The decode/scoring stack is matched, but training observations are isolated "
                "0.3-30 s utterances with 0.2 s pad while validation observations are long "
                "frozen windows. In-sample status and acoustic unit shape are confounded."
            ),
        },
        "optional_arms_not_run": sorted(set(ARMS) - set(available)),
        "interpretation_guard": (
            "Training WER measures agreement with the labels the model saw. It diagnoses "
            "fit/memorisation but cannot establish label quality or model selection."
        ),
    }
    if paired is not None:
        result["july_minus_fixed"] = {
            "wer_delta": totals["july"]["wer"] - totals["fixed"]["wer"],
            "wer_delta_ci95_meeting_clustered": paired["ci95"],
            "sub_rate_delta": totals["july"]["sub_rate"] - totals["fixed"]["sub_rate"],
            "del_rate_delta": totals["july"]["del_rate"] - totals["fixed"]["del_rate"],
            "ins_rate_delta": totals["july"]["ins_rate"] - totals["fixed"]["ins_rate"],
            "dominance": contrast_dominance(counts["july"], counts["fixed"]),
            "guard": (
                "Forensic comparison only. The July artifact remains KNOWN_BROKEN; "
                "the contrast does not isolate the label-prefix fix or training seed."
            ),
        }
    if "base" in available:
        base_paired = cluster_bootstrap(
            [(sum(counts["base"][key][:3]), counts["base"][key][3]) for key in keys],
            [(sum(counts["fixed"][key][:3]), counts["fixed"][key][3]) for key in keys],
            clusters,
            n_boot=4000,
            seed=7,
        )
        result["base_minus_fixed"] = {
            "wer_delta": totals["base"]["wer"] - totals["fixed"]["wer"],
            "wer_delta_ci95_meeting_clustered": base_paired["ci95"],
            "sub_rate_delta": totals["base"]["sub_rate"] - totals["fixed"]["sub_rate"],
            "del_rate_delta": totals["base"]["del_rate"] - totals["fixed"]["del_rate"],
            "ins_rate_delta": totals["base"]["ins_rate"] - totals["fixed"]["ins_rate"],
            "by_label_source": {
                source: {
                    metric: result["arms"]["base"]["by_label_source"][source][metric]
                    - result["arms"]["fixed"]["by_label_source"][source][metric]
                    for metric in ("wer", "sub_rate", "del_rate", "ins_rate")
                }
                for source in sorted({row["source"] for row in rows})
            },
            "dominance": contrast_dominance(counts["base"], counts["fixed"]),
            "interpretation": (
                "Positive values mean the base model is worse. This paired frozen-sample "
                "contrast measures what adaptation learned on its own labels; it does not "
                "establish label fidelity or validation improvement."
            ),
        }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    decode_parser = sub.add_parser("decode")
    decode_parser.add_argument("arm", choices=sorted(ARMS))
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--output", type=Path,
                              default=ROOT / "eval/results_training_wer.json")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "decode":
        decode(args.arm)
    else:
        score(args.output)


if __name__ == "__main__":
    main()
