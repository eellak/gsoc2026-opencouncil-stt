"""Score the two-window reference repair without emitting any transcript content.

Frozen protocol: docs/specs/2026-08-19-dense-reference-repair.md. Deletions are published
tokens the repair removed; insertions are tokens it added. The published/repaired pair is
copied into an immutable private artifact only once both windows finalize.

    .venv-eval/bin/python -m eval.controlled_eval.score_reference_repair
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.controlled_eval.eval_freeze import ftoks
from eval.controlled_eval.exp_same_stack import sdi

DEFAULT_ROOT = Path.home() / ".cache/oc-public/dense-reference-repair-2026-08"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results_dense_reference_repair.json"
UNCERTAIN = "[?]"


def window_status(answer: dict, published: list[str], fingerprint: str) -> tuple[str, str]:
    """complete / needs_adjudication / incomplete, plus the frozen reason."""
    answer = answer or {}
    text = (answer.get("repaired_text") or "")
    if not answer.get("finalized"):
        return "incomplete", "not finalized"
    if not answer.get("listened_to_full_audio"):
        return "incomplete", "full-audio pass not confirmed"
    if answer.get("source_fingerprint") != fingerprint:
        return "incomplete", "answer was written against a different source package"
    if not isinstance(answer.get("t"), (int, float)) or isinstance(answer.get("t"), bool):
        return "incomplete", "no usable client timestamp"
    repaired = ftoks(text)
    if not repaired:
        return "incomplete", "empty repaired reference"
    if repaired == published:
        return "needs_adjudication", ("repair is unchanged from the published reference "
                                      "despite the audit's material_omission verdict")
    return "complete", "finalized"


def score(answers: dict, sources: dict) -> dict:
    fingerprint = sources["source_fingerprint"]
    rows = []
    for window in sources["windows"]:
        published = ftoks(" ".join(window["published_tokens"]))
        answer = answers.get(window["review_id"]) or {}
        text = answer.get("repaired_text") or ""
        status, reason = window_status(answer, published, fingerprint)
        row = {
            "window_id": window["window_id"],
            "status": status,
            "reason": reason,
            "published_tokens": len(published),
            "uncertain_spans": text.count(UNCERTAIN),
        }
        if status in {"complete", "needs_adjudication"}:
            repaired = ftoks(answer["repaired_text"])
            s, d, i = sdi(published, repaired)
            row.update({
                "repaired_tokens": len(repaired),
                "substitutions": s,
                "deletions": d,
                "insertions": i,
                "net_token_delta": len(repaired) - len(published),
                "repaired_sha256": hashlib.sha256(
                    " ".join(repaired).encode()).hexdigest(),
            })
            if len(repaired) != len(published) - d + i:
                raise RuntimeError(f"alignment arithmetic broke on {window['window_id']}")
        rows.append(row)

    status = ("complete" if all(row["status"] == "complete" for row in rows)
              else "needs_adjudication" if any(row["status"] == "needs_adjudication" for row in rows)
              else "incomplete")
    return {
        "experiment": "dense-reference-repair-2026-08-19",
        "spec": "docs/specs/2026-08-19-dense-reference-repair.md",
        "status": status,
        "source_fingerprint": fingerprint,
        "windows": rows,
        "guards": [
            "Reference-assisted repair, not independently transcribed gold: the published words were shown, time-placed, while listening.",
            "Unintelligible spans are marked [?] and are not tokens; overlapped speech is transcribed only where separately intelligible.",
            "Two outcome-selected windows: a fidelity diagnostic, not validation WER and not the strict validation set.",
            "A 39-window WER mixing these repaired references with the 37 published ones is forbidden.",
            "No number here revises the dense screen's SCREEN - STOP decision.",
            "Rescoring any model against these references needs its own preregistration.",
        ],
    }


def freeze_private(answers: dict, sources: dict, root: Path) -> Path | None:
    """Copy both references into an immutable private artifact, once and only once."""
    path = root / "repaired-references.json"
    payload = {
        "source_fingerprint": sources["source_fingerprint"],
        "windows": [{
            "window_id": window["window_id"],
            "published_tokens": window["published_tokens"],
            "audio_faithful_tokens": ftoks(answers[window["review_id"]]["repaired_text"]),
        } for window in sources["windows"]],
    }
    if path.is_file():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"refusing to overwrite a different finalized artifact: {path}")
        return path
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    sources = json.loads((args.root / "sources.json").read_text())
    answers_path = args.root / "served/answers.json"
    answers = json.loads(answers_path.read_text()) if answers_path.is_file() else {}
    result = score(answers, sources)
    frozen = freeze_private(answers, sources, args.root) if result["status"] == "complete" else None
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "status": result["status"],
        "windows": {row["window_id"]: row["status"] for row in result["windows"]},
        "private_artifact": str(frozen) if frozen else None,
        "out": str(args.out),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
