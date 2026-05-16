#!/usr/bin/env python3
"""Clean OpenCouncil correction-pair CSV exports.

The source export is useful but not fully analysis-ready: some rows have
missing text/metadata and a small number have broken CSV alignment. This script
keeps only rows that are safe to use for taxonomy, evaluation, and audio
segment extraction, then writes a reject report for manual inspection.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "corrections-sample (1).csv"
CLEAN_CSV = ROOT / "data" / "clean" / "corrections_clean.csv"
REJECTS_CSV = ROOT / "data" / "reports" / "corrections_rejected.csv"
SUMMARY_JSON = ROOT / "data" / "reports" / "corrections_summary.json"
SUMMARY_MD = ROOT / "data" / "reports" / "data_quality.md"
SAMPLE_REVIEW_MD = ROOT / "data" / "reports" / "sample_review.md"


FIELDS = [
    "edit_id",
    "edit_timestamp",
    "edit_updated_at",
    "before_text",
    "after_text",
    "edited_by",
    "utterance_start",
    "utterance_end",
    "duration_seconds",
    "audio_url",
    "youtube_url",
    "meeting_name",
    "meeting_date",
    "before_word_count",
    "after_word_count",
    "heuristic_route",
    "heuristic_error_family",
    "text_similarity",
]


def normalize_text(value: str | None) -> str:
    value = unicodedata.normalize("NFC", value or "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_float(value: str | None) -> float | None:
    try:
        return float(normalize_text(value))
    except ValueError:
        return None


def word_count(value: str) -> int:
    return len(re.findall(r"\w+", value, flags=re.UNICODE))


def strip_punctuation(value: str) -> str:
    return re.sub(r"[^\w\s]", "", value, flags=re.UNICODE)


def text_similarity(before: str, after: str) -> float:
    return SequenceMatcher(None, before.lower(), after.lower()).ratio()


def classify_pair(before: str, after: str) -> tuple[str, str, float]:
    compact_before = normalize_text(before)
    compact_after = normalize_text(after)
    lower_before = compact_before.lower()
    lower_after = compact_after.lower()
    no_punct_before = normalize_text(strip_punctuation(compact_before)).lower()
    no_punct_after = normalize_text(strip_punctuation(compact_after)).lower()
    similarity = text_similarity(compact_before, compact_after)

    if no_punct_before == no_punct_after and compact_before != compact_after:
        if lower_before == lower_after:
            return "rule_based", "punctuation_or_spacing", similarity
        return "llm_post_correction", "capitalization_or_punctuation", similarity

    if lower_before == lower_after and compact_before != compact_after:
        return "rule_based", "capitalization", similarity

    if "?" in compact_before and ";" in compact_after:
        return "rule_based", "greek_question_mark", similarity

    before_words = word_count(compact_before)
    after_words = word_count(compact_after)
    word_delta = abs(before_words - after_words)

    if similarity >= 0.82 and word_delta <= 1:
        return "asr_finetune", "morphological_or_phonetic", similarity

    if similarity >= 0.62 and word_delta <= 2:
        return "asr_finetune", "likely_phonetic_confusion", similarity

    if before_words <= 2 or after_words <= 2 or word_delta >= 4:
        return "review", "missing_hallucinated_or_realigned_speech", similarity

    return "llm_post_correction", "semantic_or_grammar_context", similarity


def parse_date(value: str) -> str:
    value = normalize_text(value)
    if not value:
        return ""

    # Keep stable ISO-like timestamps as-is. Browser-style Date strings in the
    # export are left untouched because timezone abbreviations vary by platform.
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).isoformat(sep=" ")
        except ValueError:
            pass
    return value


@dataclass
class RowResult:
    row: dict[str, str]
    reasons: list[str]


def clean_row(raw: dict[str, str | None]) -> RowResult:
    row = {field: normalize_text(raw.get(field)) for field in raw.keys()}
    reasons: list[str] = []

    before = normalize_text(row.get("before_text"))
    after = normalize_text(row.get("after_text"))
    edited_by = normalize_text(row.get("edited_by")).lower()
    start = parse_float(row.get("utterance_start"))
    end = parse_float(row.get("utterance_end"))
    audio_url = normalize_text(row.get("audio_url"))

    if not normalize_text(row.get("edit_id")):
        reasons.append("missing_edit_id")
    if not before:
        reasons.append("missing_before_text")
    if not after:
        reasons.append("missing_after_text")
    if before and after and before == after:
        reasons.append("no_text_change")
    if edited_by not in {"task", "user"}:
        reasons.append("invalid_edited_by")
    if start is None or end is None:
        reasons.append("invalid_timestamps")
    elif end <= start:
        reasons.append("non_positive_duration")
    if not audio_url.startswith(("http://", "https://")):
        reasons.append("invalid_audio_url")

    duration = "" if start is None or end is None else f"{end - start:.3f}"

    cleaned = {
        "edit_id": normalize_text(row.get("edit_id")),
        "edit_timestamp": parse_date(row.get("edit_timestamp", "")),
        "edit_updated_at": parse_date(row.get("edit_updated_at", "")),
        "before_text": before,
        "after_text": after,
        "edited_by": edited_by,
        "utterance_start": "" if start is None else f"{start:.3f}",
        "utterance_end": "" if end is None else f"{end:.3f}",
        "duration_seconds": duration,
        "audio_url": audio_url,
        "youtube_url": normalize_text(row.get("youtube_url")),
        "meeting_name": normalize_text(row.get("meeting_name")),
        "meeting_date": parse_date(row.get("meeting_date", "")),
        "before_word_count": str(word_count(before)),
        "after_word_count": str(word_count(after)),
    }
    route, family, similarity = classify_pair(before, after)
    cleaned["heuristic_route"] = route
    cleaned["heuristic_error_family"] = family
    cleaned["text_similarity"] = f"{similarity:.4f}"

    return RowResult(cleaned, reasons)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        parsed_rows = list(csv.DictReader(handle))

    clean_rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, str]] = []
    reject_reasons: Counter[str] = Counter()

    for index, raw in enumerate(parsed_rows, start=2):
        result = clean_row(raw)
        if result.reasons:
            reject_reasons.update(result.reasons)
            rejected_rows.append(
                {
                    "source_line_or_record": str(index),
                    "reject_reasons": ";".join(result.reasons),
                    **{field: normalize_text(raw.get(field)) for field in raw.keys()},
                }
            )
            continue
        clean_rows.append(result.row)

    durations = [float(row["duration_seconds"]) for row in clean_rows]
    edited_by_counts = Counter(row["edited_by"] for row in clean_rows)
    route_counts = Counter(row["heuristic_route"] for row in clean_rows)
    family_counts = Counter(row["heuristic_error_family"] for row in clean_rows)
    audio_counts = Counter(row["audio_url"] for row in clean_rows)
    meeting_counts = Counter(row["meeting_name"] for row in clean_rows)

    summary = {
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_records_parsed": len(parsed_rows),
        "clean_records": len(clean_rows),
        "rejected_records": len(rejected_rows),
        "reject_reasons": dict(reject_reasons.most_common()),
        "edited_by_counts": dict(edited_by_counts.most_common()),
        "heuristic_route_counts": dict(route_counts.most_common()),
        "heuristic_error_family_counts": dict(family_counts.most_common()),
        "unique_audio_urls": len(audio_counts),
        "unique_meeting_names": len(meeting_counts),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "median": median(durations) if durations else None,
            "mean": mean(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
        "top_audio_urls": audio_counts.most_common(10),
        "top_meetings": meeting_counts.most_common(10),
    }

    write_csv(CLEAN_CSV, clean_rows, FIELDS)
    write_csv(
        REJECTS_CSV,
        rejected_rows,
        ["source_line_or_record", "reject_reasons", *list(parsed_rows[0].keys())],
    )
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    SUMMARY_MD.write_text(render_markdown(summary), encoding="utf-8")
    SAMPLE_REVIEW_MD.write_text(render_sample_review(clean_rows), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def render_markdown(summary: dict) -> str:
    reject_lines = "\n".join(
        f"- `{reason}`: {count}" for reason, count in summary["reject_reasons"].items()
    )
    editor_lines = "\n".join(
        f"- `{editor}`: {count}" for editor, count in summary["edited_by_counts"].items()
    )
    route_lines = "\n".join(
        f"- `{route}`: {count}"
        for route, count in summary["heuristic_route_counts"].items()
    )
    family_lines = "\n".join(
        f"- `{family}`: {count}"
        for family, count in summary["heuristic_error_family_counts"].items()
    )
    audio_lines = "\n".join(
        f"- {count} rows: `{url}`" for url, count in summary["top_audio_urls"]
    )
    meeting_lines = "\n".join(
        f"- {count} rows: {name or '(missing)'}" for name, count in summary["top_meetings"]
    )
    durations = summary["duration_seconds"]

    return f"""# Correction CSV Data Quality

Source: [`{summary["source_file"]}`](../../{summary["source_file"]})

Outputs:
- Clean CSV: [`data/clean/corrections_clean.csv`](../clean/corrections_clean.csv)
- Rejected rows: [`data/reports/corrections_rejected.csv`](corrections_rejected.csv)
- Machine-readable summary: [`data/reports/corrections_summary.json`](corrections_summary.json)

## Record Counts

- Parsed source records: {summary["source_records_parsed"]}
- Clean analysis-ready records: {summary["clean_records"]}
- Rejected records: {summary["rejected_records"]}

## Rejection Reasons

{reject_lines or "- None"}

Rows can have more than one rejection reason. The most important class is `invalid_edited_by`, which usually indicates broken CSV alignment rather than a real editor value.

## Editor Split

{editor_lines or "- None"}

Use `task` as the current LLM/post-correction stage and `user` as human review intervention. This split is central for HIR and for comparing what the LLM already fixes against what humans still correct.

## Initial Heuristic Routing

{route_lines or "- None"}

This is a rough bootstrap label, not ground truth. Use it to prioritize manual review and mentor discussion:

- `asr_finetune`: likely useful for Whisper/STT fine-tuning.
- `llm_post_correction`: likely useful for prompt examples, grammar/context correction, or dynamic vocabulary.
- `rule_based`: likely cheap normalization before either model.
- `review`: likely needs audio/listening or alignment checks before use.

Error-family counts:

{family_lines or "- None"}

## Audio Coverage

- Unique audio URLs: {summary["unique_audio_urls"]}
- Unique meeting names: {summary["unique_meeting_names"]}
- Duration min/median/mean/max: {durations["min"]:.3f}s / {durations["median"]:.3f}s / {durations["mean"]:.3f}s / {durations["max"]:.3f}s

Top audio files:

{audio_lines or "- None"}

Top meeting labels:

{meeting_lines or "- None"}
"""


def render_sample_review(rows: list[dict[str, str]]) -> str:
    by_route: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_route.setdefault(row["heuristic_route"], []).append(row)

    sections = [
        "# Sample Review Queue",
        "",
        "Use this small queue to validate the bootstrap taxonomy before trusting the labels at scale.",
        "",
        "Source: [`data/clean/corrections_clean.csv`](../clean/corrections_clean.csv)",
        "",
    ]

    for route in ["asr_finetune", "llm_post_correction", "rule_based", "review"]:
        sections.extend([f"## `{route}`", ""])
        sample = by_route.get(route, [])[:12]
        if not sample:
            sections.extend(["No rows.", ""])
            continue
        sections.extend(
            [
                "| edit_id | edited_by | family | before | after |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in sample:
            before = row["before_text"].replace("|", "\\|")
            after = row["after_text"].replace("|", "\\|")
            sections.append(
                f"| `{row['edit_id']}` | `{row['edited_by']}` | `{row['heuristic_error_family']}` | {before} | {after} |"
            )
        sections.append("")

    return "\n".join(sections)


if __name__ == "__main__":
    main()
