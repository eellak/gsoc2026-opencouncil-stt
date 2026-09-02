"""Offline census for the frozen clean-data filter ladder.

Reads transcript-bearing caches but writes aggregate counts only. No transcript or
audio path is included in the result. The frozen rules are documented in
docs/specs/2026-08-18-clean-data-filter-ladder.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache/oc-public"
DEFAULT_OUT = ROOT / "eval/results_training_data_filter_census.json"

PACKS = {
    "hparl2-v2": CACHE / "training-sets/hparl2-v2/train.jsonl",
    "stoma-v2": CACHE / "training-sets/stoma-v2/train.jsonl",
    "cv-v2": CACHE / "training-sets/cv-v2/train.jsonl",
    "eurospeech-v2": CACHE / "training-sets/eurospeech-v2/train.jsonl",
}
SECOND_WITNESS = {
    "eurospeech-v2": CACHE / "eurospeech-el/filtered.jsonl",
}

MIN_ALIGN_L2 = 0.95
MAX_SECOND_WER_L3 = 0.05
MIN_DURATION_S = 1.5
MAX_DURATION_S = 30.0
MIN_TOKENS = 3
FAST_TOKENS_PER_S = 3.0
HARD_ALIGN_CEILING = 0.85

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
CAP_GREEK_RE = re.compile(r"^[Α-ΩΆΈΉΊΌΎΏΪΫ]")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn").lower()
    return " ".join(TOKEN_RE.findall(text))


def tokens(text: str) -> list[str]:
    return norm(text).split()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_known_name_phrases(root: Path = ROOT) -> set[str]:
    """Load existing council-specific names without emitting them.

    Full roster phrases are deliberately narrower than the glossary, whose global
    and per-city lanes also contain ordinary procedural vocabulary.
    """
    phrases: set[str] = set()
    for rel in ("data/improve_loop/rosters.json", "data/pii/rosters_full.json"):
        data = json.loads((root / rel).read_text())
        for vals in data.values():
            phrases.update(norm(v) for v in vals if norm(v))
    return {p for p in phrases if len(p) >= 3}


def contains_phrase(text_norm: str, phrases: set[str]) -> bool:
    padded = f" {text_norm} "
    return any(f" {p} " in padded for p in phrases)


def diagnostics(row: dict, known_names: set[str]) -> dict[str, bool]:
    text = row.get("text") or ""
    ts = tokens(text)
    dur = float(row.get("dur") or 0.0)
    align = row.get("align")
    raw = TOKEN_RE.findall(text)
    return {
        "known_roster_name": contains_phrase(" ".join(ts), known_names),
        "capitalized_noninitial": any(CAP_GREEK_RE.match(t) for t in raw[1:]),
        "fast_speech": bool(dur > 0 and len(ts) / dur >= FAST_TOKENS_PER_S),
        "hard_example": bool(
            row.get("edge_clipped")
            or (isinstance(align, (int, float)) and 0.50 <= align < HARD_ALIGN_CEILING)
        ),
        "edge_clipped": bool(row.get("edge_clipped")),
    }


def classify(row: dict, second_wer: float | None) -> dict[str, bool | None]:
    """Return nested level membership; None means metadata unavailable."""
    dur = float(row.get("dur") or 0.0)
    align = row.get("align")
    l1 = bool(
        isinstance(align, (int, float))
        and math.isfinite(float(align))
        and (row.get("text_asr") or "").strip()
    )
    l2 = bool(
        l1
        and float(align) >= MIN_ALIGN_L2
        and MIN_DURATION_S <= dur <= MAX_DURATION_S
        and len(tokens(row.get("text") or "")) >= MIN_TOKENS
        and not row.get("edge_clipped")
    )
    if not l2:
        l3 = False
    elif second_wer is None:
        l3 = None
    else:
        l3 = bool(math.isfinite(second_wer) and second_wer <= MAX_SECOND_WER_L3)
    return {"L0": True, "L1": l1, "L2": l2, "L3": l3}


def load_second_wers(path: Path) -> dict[str, float]:
    out = {}
    for row in read_jsonl(path):
        value = row.get("ds_wer")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[row["row_id"]] = float(value)
    return out


def _blank() -> dict:
    return {"rows": 0, "seconds": 0.0,
            "diagnostics": defaultdict(lambda: {"rows": 0, "seconds": 0.0})}


def _round_hours(seconds: float) -> float:
    return round(seconds / 3600.0, 3)


def census(pack_paths: dict[str, Path], second_paths: dict[str, Path],
           known_names: set[str]) -> dict:
    levels = {lvl: defaultdict(_blank) for lvl in ("L0", "L1", "L2", "L3")}
    unmeasurable = {lvl: defaultdict(lambda: {"rows": 0, "seconds": 0.0})
                    for lvl in ("L0", "L1", "L2", "L3")}
    input_meta = {}

    for source, path in pack_paths.items():
        second = load_second_wers(second_paths[source]) if source in second_paths else {}
        input_meta[source] = {
            "train_jsonl_sha256": sha256(path),
            "second_witness": "whisper_turbo_ds_wer" if source in second_paths else None,
            "second_witness_rows": len(second),
        }
        for row in read_jsonl(path):
            dur = float(row.get("dur") or 0.0)
            diag = diagnostics(row, known_names)
            second_wer = second.get(row["id"]) if source in second_paths else None
            membership = classify(row, second_wer)
            for level, keep in membership.items():
                if keep is None:
                    unmeasurable[level][source]["rows"] += 1
                    unmeasurable[level][source]["seconds"] += dur
                    continue
                if not keep:
                    continue
                cell = levels[level][source]
                cell["rows"] += 1
                cell["seconds"] += dur
                for name, flag in diag.items():
                    if flag:
                        cell["diagnostics"][name]["rows"] += 1
                        cell["diagnostics"][name]["seconds"] += dur

    output_levels = {}
    for level, source_cells in levels.items():
        by_source = {}
        for source in pack_paths:
            cell = source_cells[source]
            base = levels["L0"][source]
            u = unmeasurable[level][source]
            complete = u["rows"] == 0
            ds = {}
            names = set(base["diagnostics"]) | set(cell["diagnostics"])
            for name in sorted(names):
                b = base["diagnostics"][name]
                v = cell["diagnostics"][name]
                ds[name] = {
                    "rows": v["rows"],
                    "hours": _round_hours(v["seconds"]),
                    "row_retention_vs_L0": round(v["rows"] / b["rows"], 4)
                    if complete and b["rows"] else None,
                    "hour_retention_vs_L0": round(v["seconds"] / b["seconds"], 4)
                    if complete and b["seconds"] else None,
                }
            by_source[source] = {
                "rows": cell["rows"],
                "hours": _round_hours(cell["seconds"]),
                "row_retention_vs_L0": round(cell["rows"] / base["rows"], 4)
                if complete and base["rows"] else None,
                "hour_retention_vs_L0": round(cell["seconds"] / base["seconds"], 4)
                if complete and base["seconds"] else None,
                "unmeasurable_rows": u["rows"],
                "unmeasurable_hours": _round_hours(u["seconds"]),
                "diagnostics": ds,
            }
        output_levels[level] = {"by_source": by_source}

    equal_hours = {}
    for source in pack_paths:
        l2 = levels["L2"][source]
        equal_hours[source] = {
            "comparison": "L0_random_downsample_vs_L2",
            "target_seconds_per_arm": round(l2["seconds"], 3),
            "target_hours_per_arm": _round_hours(l2["seconds"]),
            "sampling_seeds": [13, 29, 47],
            "match": ["source", "audio_seconds", "optimizer_updates"],
            "do_not_match": ["diagnostic_categories"],
        }

    return {
        "schema_version": 1,
        "frozen_spec": "docs/specs/2026-08-18-clean-data-filter-ladder.md",
        "inputs": input_meta,
        "thresholds": {
            "l2_min_align": MIN_ALIGN_L2,
            "l3_max_second_wer": MAX_SECOND_WER_L3,
            "duration_s": [MIN_DURATION_S, MAX_DURATION_S],
            "min_normalized_tokens": MIN_TOKENS,
            "fast_tokens_per_s": FAST_TOKENS_PER_S,
            "hard_align_interval": [0.50, HARD_ALIGN_CEILING],
        },
        "known_roster_name_phrases": len(known_names),
        "levels": output_levels,
        "equal_hours_design": equal_hours,
        "privacy": "aggregate_only_no_transcript_or_audio_paths",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    result = census(PACKS, SECOND_WITNESS, load_known_name_phrases())
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote aggregate census to {args.out}")


if __name__ == "__main__":
    main()
