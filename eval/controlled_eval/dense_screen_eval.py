"""Convert, decode and score all dense-screen adapters on one frozen GPU stack."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from eval.controlled_eval.eval_freeze import ftoks
from eval.controlled_eval.exp_same_stack import sdi
from eval.controlled_eval.scoring import cluster_bootstrap

MODEL_ID = "openai/whisper-large-v3"
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
TAGS = [f"{arm}_s{seed}" for seed in (13, 29, 47) for arm in ("A", "B")]


def log(message: str) -> None:
    print(f"[dense-eval] {message}", flush=True)


def load_rows(path: Path, key: str, audio_field: str) -> list[dict]:
    state = json.loads(path.read_text())
    rows = state[key]
    for row in rows:
        audio = Path(row[audio_field])
        if not audio.is_absolute():
            row[audio_field] = str(path.parent / audio)
        if not Path(row[audio_field]).is_file():
            raise FileNotFoundError(row[audio_field])
    return rows


def build_ct2(adapter: Path, work: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperProcessor

    merged, ct2 = work / "merged", work / "ct2"
    if ct2.is_dir() and (ct2 / "model.bin").is_file():
        return ct2
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.generation_config.language = "greek"
    model.generation_config.task = "transcribe"
    model.save_pretrained(merged)
    WhisperProcessor.from_pretrained(MODEL_ID, language="greek", task="transcribe").save_pretrained(merged)
    WhisperFeatureExtractor.from_pretrained(MODEL_ID).save_pretrained(merged)
    del model, base
    gc.collect()
    subprocess.run([
        "ct2-transformers-converter", "--model", str(merged), "--output_dir", str(ct2),
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--quantization", "float16",
    ], check=True)
    shutil.rmtree(merged)
    return ct2


def decode_rows(model, rows: list[dict], audio_field: str, id_field: str,
                destination: Path, tag: str) -> None:
    import ctranslate2

    state = json.loads(destination.read_text()) if destination.exists() else {}
    for index, row in enumerate(rows, 1):
        key = row[id_field]
        if key in state:
            continue
        seed = int(hashlib.sha256(f"dense-screen|{key}".encode()).hexdigest()[:8], 16)
        ctranslate2.set_random_seed(seed)
        segments = list(model.transcribe(row[audio_field], **CONTROL)[0])
        state[key] = {"segments": [segment.text for segment in segments]}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(state, ensure_ascii=False))
        if index % 25 == 0 or index == len(rows):
            log(f"{tag} {destination.parent.name}: {index}/{len(rows)}")


def decode(args) -> None:
    from faster_whisper import WhisperModel

    validation = load_rows(args.validation, "eval_windows", "audio_local")
    training = load_rows(args.training, "rows", "clip")
    for tag in TAGS:
        run = args.runs / tag
        adapter = run / "adapter"
        if not (run / "COMPLETE").is_file() or not adapter.is_dir():
            raise SystemExit(f"incomplete training run: {tag}")
        validation_out = args.out / "validation" / f"{tag}.json"
        training_out = args.out / "training" / f"{tag}.json"
        if validation_out.exists() and training_out.exists():
            if len(json.loads(validation_out.read_text())) == len(validation) and len(
                    json.loads(training_out.read_text())) == len(training):
                log(f"skip complete decode {tag}")
                continue
        ct2_work = args.ct2 / tag
        ct2 = build_ct2(adapter, ct2_work)
        model = WhisperModel(str(ct2), device="cuda", compute_type="float16")
        decode_rows(model, validation, "audio_local", "window_id", validation_out, tag)
        decode_rows(model, training, "clip", "key", training_out, tag)
        del model
        shutil.rmtree(ct2_work)


def counts(rows: list[dict], hypotheses: dict, id_field: str, ref_field: str) -> dict:
    output = {}
    for row in rows:
        key = row[id_field]
        reference = row[ref_field] if isinstance(row[ref_field], list) else ftoks(row[ref_field])
        hypothesis = [token for segment in hypotheses[key]["segments"] for token in ftoks(segment)]
        output[key] = (*sdi(reference, hypothesis), len(reference))
    return output


def totals(values: dict) -> dict:
    sums = [sum(row[index] for row in values.values()) for index in range(4)]
    s, d, i, n = sums
    return {"sub": s, "del": d, "ins": i, "ref_tokens": n,
            "wer": (s+d+i)/n, "sub_rate": s/n, "del_rate": d/n, "ins_rate": i/n}


def score(args) -> None:
    validation = load_rows(args.validation, "eval_windows", "audio_local")
    training = load_rows(args.training, "rows", "clip")
    val_counts, train_counts = {}, {}
    for tag in TAGS:
        val_counts[tag] = counts(validation, json.loads(
            (args.out / "validation" / f"{tag}.json").read_text()), "window_id", "ref_tokens")
        train_counts[tag] = counts(training, json.loads(
            (args.out / "training" / f"{tag}.json").read_text()), "key", "reference")

    pairs = []
    for seed in (13, 29, 47):
        a, b = f"A_s{seed}", f"B_s{seed}"
        keys = [row["window_id"] for row in validation]
        clusters = [(row["city"], row["meeting_id"]) for row in validation]
        ci = cluster_bootstrap(
            [(sum(val_counts[b][key][:3]), val_counts[b][key][3]) for key in keys],
            [(sum(val_counts[a][key][:3]), val_counts[a][key][3]) for key in keys],
            clusters, n_boot=4000, seed=7)
        ta, tb = totals(val_counts[a]), totals(val_counts[b])
        contributions = {
            key: sum(val_counts[b][key][:3]) - sum(val_counts[a][key][:3]) for key in keys}
        net = sum(contributions.values())
        loo = [net-value for value in contributions.values()]
        largest = max((abs(value) for value in contributions.values()), default=0)
        pairs.append({
            "seed": seed, "A": ta, "B": tb,
            "delta_wer": tb["wer"]-ta["wer"],
            "delta_del_rate": tb["del_rate"]-ta["del_rate"],
            "delta_ins_rate": tb["ins_rate"]-ta["ins_rate"],
            "delta_wer_ci95_meeting_clustered": ci["ci95"],
            "leave_one_window_out_sign_reversal": any(
                (value > 0) != (net > 0) for value in loo) if net else None,
            "largest_window_share_of_net": largest/abs(net) if net else None,
        })
    mean = lambda field: sum(pair[field] for pair in pairs) / len(pairs)
    promotion = {
        "mean_delta_wer_lt_zero": mean("delta_wer") < 0,
        "at_least_two_of_three_delta_wer_lt_zero": sum(
            pair["delta_wer"] < 0 for pair in pairs) >= 2,
        "mean_delta_del_rate_le_zero": mean("delta_del_rate") <= 0,
        "mean_delta_ins_rate_lt_0.0005": mean("delta_ins_rate") < 0.0005,
        "no_loo_sign_reversal": all(not pair["leave_one_window_out_sign_reversal"] for pair in pairs),
        "no_window_over_25pct": all(
            pair["largest_window_share_of_net"] is not None
            and pair["largest_window_share_of_net"] <= 0.25 for pair in pairs),
    }
    training_wer = {}
    training_by_key = {row["key"]: row for row in training}
    for tag, values in train_counts.items():
        training_wer[tag] = {
            "overall": totals(values),
            "by_label_source": {
                source: totals({key: value for key, value in values.items()
                                if training_by_key[key]["source"] == source})
                for source in ("correction", "no_edit")
            },
        }
    result = {
        "experiment": "dense-screen-300step-2026-08-19",
        "stack": {"device": "cuda", "compute_type": "float16", "config": CONTROL},
        "validation_pairs": pairs,
        "mean_deltas": {field: mean(field) for field in
                        ("delta_wer", "delta_del_rate", "delta_ins_rate")},
        "promotion_gates": promotion,
        "status": "SCREEN - ADVANCE" if all(promotion.values()) else "SCREEN - STOP",
        "training_wer": training_wer,
        "guard": "Paired A/B inference is comparable within this one GPU float16 stack; levels are not directly comparable to historical CPU int8 validation levels.",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"status": result["status"], "mean_deltas": result["mean_deltas"],
                      "promotion_gates": promotion}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("decode", "score", "all"))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ct2", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("decode", "all"):
        decode(args)
    if args.command in ("score", "all"):
        score(args)


if __name__ == "__main__":
    main()
