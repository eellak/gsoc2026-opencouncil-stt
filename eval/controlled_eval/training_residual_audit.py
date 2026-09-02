"""Aggregate residual correction errors and freeze a model-blind listening queue."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from eval.controlled_eval.training_wer import arm_counts, load_sample, rates, work_dir

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "eval/results_training_residual_audit.json"
REVIEW_ROOT = Path.home() / ".cache/oc-public/training-residual-audit-2026-08"
MIN_ROWS = 10
MIN_TOKENS = 50


def duration_bin(value: float) -> str:
    if value < 2:
        return "<2"
    if value < 4:
        return "2-4"
    if value < 8:
        return "4-8"
    return ">=8"


def adjustment_bin(row: dict) -> str:
    values = (row.get("start"), row.get("end"), row.get("start_adj"), row.get("end_adj"))
    if any(value is None or (isinstance(value, float) and math.isnan(value)) for value in values):
        return "none"
    amount = abs(float(row["start_adj"]) - float(row["start"])) + abs(
        float(row["end_adj"]) - float(row["end"])
    )
    if amount == 0:
        return "none"
    if amount <= 0.25:
        return "<=0.25"
    if amount <= 1:
        return "0.25-1"
    return ">1"


def categories(value) -> list[str]:
    if value is None:
        return ["uncategorized"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value] or ["uncategorized"]


def stable_rank(key: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()


def summarize(keys: list[str], fixed: dict, base: dict, total_errors: int) -> dict:
    fixed_part = {key: fixed[key] for key in keys}
    base_part = {key: base[key] for key in keys}
    fixed_rates = rates(fixed_part)
    base_rates = rates(base_part)
    errors = fixed_rates["sub"] + fixed_rates["del"] + fixed_rates["ins"]
    return {
        "n_rows": len(keys),
        **fixed_rates,
        "residual_error_share": errors / total_errors if total_errors else 0.0,
        "base_minus_fixed_wer": base_rates["wer"] - fixed_rates["wer"],
        "ranking_eligible": len(keys) >= MIN_ROWS and fixed_rates["ref_tokens"] >= MIN_TOKENS,
    }


def main() -> None:
    sample = load_sample()
    selected = [row for row in sample["rows"] if row["source"] == "correction"]
    sample_by_id = {row["utterance_id"]: row for row in selected}
    frame = pd.read_parquet(ROOT / "data/hf-dataset/public/train.parquet")
    metadata = {
        row["utterance_id"]: row
        for row in frame[frame["utterance_id"].isin(sample_by_id)].to_dict("records")
    }
    if set(metadata) != set(sample_by_id):
        raise SystemExit("frozen sample/parquet join mismatch")

    fixed_all = arm_counts(sample, "fixed")
    base_all = arm_counts(sample, "base")
    fixed = {row["key"]: fixed_all[row["key"]] for row in selected}
    base = {row["key"]: base_all[row["key"]] for row in selected}
    id_to_key = {row["utterance_id"]: row["key"] for row in selected}
    total_errors = sum(sum(value[:3]) for value in fixed.values())

    groups: dict[str, dict[str, list[str]]] = {
        name: defaultdict(list)
        for name in ("duration", "boundary_status", "adjustment", "overlap", "category")
    }
    for utterance_id, row in metadata.items():
        key = id_to_key[utterance_id]
        groups["duration"][duration_bin(float(row["duration_s"]))].append(key)
        groups["boundary_status"][str(row["boundary_status"])].append(key)
        groups["adjustment"][adjustment_bin(row)].append(key)
        groups["overlap"][str(bool(row["has_overlap"])).lower()].append(key)
        for category in categories(row["error_categories"]):
            groups["category"][category].append(key)

    strata = {
        dimension: {
            label: summarize(keys, fixed, base, total_errors)
            for label, keys in sorted(labels.items())
        }
        for dimension, labels in groups.items()
    }
    candidates = []
    for dimension, labels in strata.items():
        eligible = [(values["wer"], label) for label, values in labels.items()
                    if values["ranking_eligible"]]
        if eligible:
            wer, label = max(eligible)
            candidates.append((wer, dimension, label))
    top = sorted(candidates, reverse=True)[:3]

    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    queue, hidden, used = [], [], set()
    by_key = {row["key"]: row for row in selected}
    for _, dimension, label in top:
        keys = groups[dimension][label]
        errorful = [key for key in keys if sum(fixed[key][:3]) > 0]
        clean = [key for key in keys if sum(fixed[key][:3]) == 0]
        for kind, pool, limit in (("errorful", errorful, 8), ("zero", clean, 4)):
            added = 0
            for key in sorted(pool, key=lambda item: stable_rank(item, f"{dimension}|{label}|{kind}")):
                if key in used:
                    continue
                used.add(key)
                source = by_key[key]
                review_id = hashlib.sha256(f"residual-review|{key}".encode()).hexdigest()[:16]
                queue.append({
                    "review_id": review_id,
                    "audio": source["clip"],
                    "training_label": source["reference"],
                    "label_faithful": None,
                    "boundaries_usable": None,
                    "notes": "",
                })
                hidden.append({
                    "review_id": review_id,
                    "key": key,
                    "dimension": dimension,
                    "stratum": label,
                    "selection_lane": kind,
                    "fixed_sdi": list(fixed[key][:3]),
                })
                added += 1
                if added == limit:
                    break
    (REVIEW_ROOT / "review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue))
    (REVIEW_ROOT / "hidden-key.json").write_text(json.dumps(hidden, indent=1) + "\n")

    result = {
        "experiment": "training-residual-audit-2026-08-19",
        "metric": "agreement with correction labels; not fidelity to audio",
        "sample_sha256": sample["selection"]["sample_sha256"],
        "population": summarize(list(fixed), fixed, base, total_errors),
        "strata": strata,
        "listening_queue": {
            "n_rows": len(queue),
            "selected_strata": [
                {"dimension": dimension, "label": label, "wer": wer}
                for wer, dimension, label in top
            ],
            "review_file": "~/.cache/oc-public/training-residual-audit-2026-08/review.jsonl",
            "blind_to": ["hypothesis", "WER", "model identity", "source stratum"],
            "status": "frozen, not reviewed",
        },
        "guard": "The aggregate identifies where label agreement is low; only blind listening can distinguish label/boundary defects from genuine model residuals.",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "correction_rows": len(selected),
        "fixed_wer": result["population"]["wer"],
        "queue_rows": len(queue),
        "selected_strata": result["listening_queue"]["selected_strata"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
