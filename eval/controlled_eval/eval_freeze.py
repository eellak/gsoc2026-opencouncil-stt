"""Freeze the evaluation set for the endgame workstreams (Task 0 of the handoff plan).

Writes `research/eval-freeze-2026-08/manifest.json`: the 39 training-disjoint
validation windows every later number is computed on, the 7 temporal-test windows
that are touched exactly once, and the exact software/artifact identity of the box
that will produce those numbers.

The 39 are `argos + orestiada` (the only two benchmark cities never seen in
training) minus the single window whose meeting is dated on or after 2026-06-01.
That last clause is the temporal-test boundary from
`exp-2026-08-10-benchmark-fixed-adapter`; across the whole 260-window benchmark it
selects 7 windows, but only one of them is in argos/orestiada. 40 - 1 = 39.

No transcript text is written here - only IDs, counts and hashes.

    .venv-eval/bin/python -m eval.controlled_eval.eval_freeze
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.controlled_eval import bench_data as B  # noqa: E402
from eval.controlled_eval.scoring import wtoks  # noqa: E402

ROOT = Path("/home/harold/opencouncil-fine-tuning")
OUT = ROOT / "research/eval-freeze-2026-08/manifest.json"

# The benchmark run whose report carries the corrected adapter's hypotheses.
RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"

CLEAN_CITIES = ("argos", "orestiada")
TEMPORAL_TEST_FROM = (2026, 6)  # meetings on/after 2026-06-01 are TEST

WINDOWS_DIR = Path.home() / ".cache/oc-public/bench_windows"
CT2_MODEL = Path("/home/harold/oc-asr-serve/ct2-fixed")
ADAPTER = Path("/home/harold/oc-asr-serve/adapter-fixed-2026-08-01")

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

# Frozen filler set, approximating the benchmark's `wer-nofillers`. The benchmark
# computes that metric server-side and publishes no importable normalizer, so this
# is our own frozen stand-in on top of eval/controlled_eval/scoring.py: strip
# repeated-vowel / hesitation tokens only, and only forms of length >= 2, so that
# the Greek article "ο" and single-letter list markers survive. Derived from the
# metric description in the benchmark report plus the reference texts; never from a
# provider hypothesis. Do not change it without re-running everything that quotes a
# number produced with it.
FILLER_RE = re.compile(r"^(ε{2,}|μ{2,}|α{2,}|ο{3,}|χμ+|εμ+|μχ+|χ{2,})$")


def ftoks(text: str) -> list[str]:
    """The frozen scoring tokenizer: scoring.wtoks, minus hesitation fillers."""
    return [t for t in wtoks(text) if not FILLER_RE.match(t)]


def meeting_ym(meeting_id: str) -> tuple[int, int]:
    m = re.match(r"([a-z]{3})[a-z]*(\d{1,2})_.*?(\d{4})", meeting_id)
    if not m:
        raise SystemExit(f"cannot parse a date out of meeting id {meeting_id!r}")
    return int(m.group(3)), MONTHS[m.group(1)]


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def pkg_version(mod: str) -> str:
    import importlib
    return getattr(importlib.import_module(mod), "__version__", "unknown")


def window_row(it: dict) -> dict:
    toks = ftoks(it["referenceText"])
    wav = WINDOWS_DIR / f"{it['itemId']}.wav"
    return {
        "window_id": it["itemId"],
        "city": it["cityId"],
        "meeting_id": it["meetingId"],          # the resampling block
        "start_sec": round(it["startSec"], 3),
        "duration_sec": round(it["durationSec"], 3),
        "ref_tokens": len(toks),                # frozen normalizer, fillers stripped
        "ref_tokens_raw": len(wtoks(it["referenceText"])),
        "audio_local": wav.exists(),
    }


def main() -> None:
    report = B.load_report(RUN_ID)
    items = [it for it in report["items"] if it["cityId"] in CLEAN_CITIES]
    if len(items) != 40:
        raise SystemExit(f"expected 40 argos+orestiada windows, got {len(items)}")

    late_all = [it["itemId"] for it in report["items"]
                if meeting_ym(it["meetingId"]) >= TEMPORAL_TEST_FROM]
    if len(late_all) != 7:
        raise SystemExit(f"expected 7 temporal-test windows benchmark-wide, got {len(late_all)}")

    clean = [it for it in items if meeting_ym(it["meetingId"]) < TEMPORAL_TEST_FROM]
    if len(clean) != 39:
        raise SystemExit(f"expected 39 clean windows, got {len(clean)}")

    holdout = [it for it in report["items"] if it["itemId"] in set(late_all)]

    eval_rows = [window_row(it) for it in sorted(clean, key=lambda x: x["itemId"])]
    hold_rows = [window_row(it) for it in sorted(holdout, key=lambda x: x["itemId"])]

    code_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()

    manifest = {
        "name": "eval-freeze-2026-08",
        "frozen_at": "2026-08-11",
        "purpose": ("The evaluation set for exp-2026-08-12-decode-ablation, "
                    "exp-2026-08-12-ds-wer and exp-2026-08-13-correction-only. "
                    "Frozen before any arm was decoded."),
        "source_run": RUN_ID,
        "selection_rule": (
            "cityId in {argos, orestiada} (the only training-disjoint benchmark "
            "cities), minus every window whose meeting is dated on or after "
            "2026-06-01 (the temporal TEST boundary). 40 - 1 = 39."),
        "resampling_block": "meeting_id",
        "normalizer": {
            "module": "eval.controlled_eval.scoring",
            "function": "wtoks",
            "filler_strip_regex": FILLER_RE.pattern,
            "note": ("The benchmark's own wer-nofillers normalizer runs server-side "
                     "and is not importable. This is the project's frozen scorer "
                     "plus an explicit filler set - a stand-in, not the benchmark's "
                     "implementation. Numbers are comparable across our arms, not "
                     "byte-identical to the benchmark leaderboard."),
        },
        "environment": {
            "python": sys.version.split()[0],
            "faster_whisper": pkg_version("faster_whisper"),
            "ctranslate2": pkg_version("ctranslate2"),
            "numpy": pkg_version("numpy"),
            "venv": ".venv-eval",
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": 16,
            "host": "minipc",
            "note": ("One device for every arm of every comparison. CPU int8 and "
                     "CUDA int8 give different tokens from the same artifact "
                     "(exp-2026-08-10-benchmark-fixed-adapter caveat 4), so no "
                     "number here may be compared against a GPU-produced one."),
        },
        "artifacts": {
            "artifact-ct2-fixed": {
                "path": str(CT2_MODEL),
                "file": "model.bin",
                "sha256_16": sha16(CT2_MODEL / "model.bin"),
                "ledger_sha256_16": "8a1a3b257d0c1bdb",
            },
            "artifact-adapter-fixed": {
                "path": str(ADAPTER),
                "file": "adapter_model.safetensors",
                "sha256_16": sha16(ADAPTER / "adapter_model.safetensors"),
                "ledger_sha256_16": "ea8f03230846888f",
            },
        },
        "code_sha": code_sha,
        "eval_windows": eval_rows,
        "holdout_windows": hold_rows,
        "totals": {
            "eval_windows": len(eval_rows),
            "eval_ref_tokens": sum(r["ref_tokens"] for r in eval_rows),
            "eval_meetings": len({r["meeting_id"] for r in eval_rows}),
            "eval_audio_missing": [r["window_id"] for r in eval_rows
                                   if not r["audio_local"]],
            "holdout_windows": len(hold_rows),
            "holdout_ref_tokens": sum(r["ref_tokens"] for r in hold_rows),
            "holdout_meetings": len({r["meeting_id"] for r in hold_rows}),
            "holdout_audio_missing": [r["window_id"] for r in hold_rows
                                      if not r["audio_local"]],
        },
        "holdout_caveat": (
            "6 of the 7 holdout windows are in cities that ARE in the training set "
            "(sparta, vrilissia, xylokastro, zografou). They are a temporal holdout, "
            "not a training-disjoint one. That is acceptable for a within-model "
            "decode-config comparison, where both arms share the same weights, and "
            "is NOT acceptable for any model-vs-model claim."),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")

    for k, v in manifest["artifacts"].items():
        if v["sha256_16"] != v["ledger_sha256_16"]:
            raise SystemExit(f"{k}: on-disk hash {v['sha256_16']} != ledger "
                             f"{v['ledger_sha256_16']} - stop and reconcile")
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
