"""Score the frozen blind listening audit without emitting transcript content."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_ROOT = Path.home() / ".cache/oc-public/training-listening-audit-2026-08"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results_training_listening_audit.json"


def count(values) -> dict[str, int]:
    return dict(sorted(Counter(value if value is not None else "missing" for value in values).items()))


def score(answers: dict, hidden: list[dict]) -> dict:
    training = [row for row in hidden if row["kind"] == "training_clip"]
    windows = [row for row in hidden if row["kind"] == "validation_window"]
    training_rows = []
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in training:
        answer = answers.get(row["review_id"], {})
        item = {
            "label_faithful": answer.get("label_faithful"),
            "boundaries_usable": answer.get("boundaries_usable"),
        }
        training_rows.append(item)
        by_stratum[f"{row['dimension']}:{row['stratum']}"] .append(item)

    def training_summary(rows: list[dict]) -> dict:
        defect = sum(row["label_faithful"] == "no" or row["boundaries_usable"] == "no"
                     for row in rows)
        complete = sum(row["label_faithful"] is not None and
                       row["boundaries_usable"] is not None for row in rows)
        return {
            "n_rows": len(rows),
            "n_complete": complete,
            "label_faithful": count(row["label_faithful"] for row in rows),
            "boundaries_usable": count(row["boundaries_usable"] for row in rows),
            "n_routed_to_protected_correction": defect,
        }

    window_values = [answers.get(row["review_id"], {}).get("reference_completeness")
                     for row in windows]
    complete = (all(row["label_faithful"] is not None and
                    row["boundaries_usable"] is not None for row in training_rows)
                and all(value is not None for value in window_values))
    if not complete:
        mechanism = "unresolved until every required answer is complete"
    elif any(value == "unsure" for value in window_values):
        mechanism = "unresolved because at least one validation window is unsure"
    elif any(value in {"minor_omission", "material_omission"} for value in window_values):
        mechanism = "re-reference omitted speech before model insertion adjudication"
    else:
        mechanism = "extra content is genuine insertion evidence on these two windows only"

    return {
        "experiment": "training-listening-audit-2026-08-19",
        "status": "complete" if complete else "incomplete",
        "training": {
            "overall": training_summary(training_rows),
            "by_selected_stratum": {
                key: training_summary(rows) for key, rows in sorted(by_stratum.items())
            },
        },
        "validation_reference_completeness": count(window_values),
        "frozen_interpretation": mechanism,
        "guards": [
            "Outcome-enriched audit; observed fractions do not estimate dataset prevalence.",
            "No entire stratum is filtered from these verdicts.",
            "No verdict can retroactively promote the dense SCREEN - STOP arm.",
            "Aggregate contains no transcript, audio, free-text note, or window id.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=DEFAULT_ROOT / "served/answers.json")
    parser.add_argument("--key", type=Path, default=DEFAULT_ROOT / "hidden-key.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = score(json.loads(args.answers.read_text()), json.loads(args.key.read_text()))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "status": result["status"],
        "training_complete": result["training"]["overall"]["n_complete"],
        "training_total": result["training"]["overall"]["n_rows"],
        "validation": result["validation_reference_completeness"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
