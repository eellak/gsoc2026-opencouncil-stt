#!/usr/bin/env python3
"""Write the freeze record for the exclusive-diarization experiment.

Runs once, BEFORE the first Phase 1 submission. Records every input that could
otherwise move after results are seen: the pinned opencouncil-tasks commit, the
sha256 of each script and of the spec, the manifest hash, the Phase 2 window
selection (computed here so it cannot be re-derived later), model strings, seeds
and payload shapes. The report quotes this file's own hash.

Refuses to overwrite an existing record.

Usage: python eval/controlled_eval/exclusive_freeze.py [--oc-tasks PATH]
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exclusive_diar_api import SC, log  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = SC / "exclusive_freeze.json"
WINDOWS_OUT = SC / "exclusive_phase2_windows.json"
N_WINDOWS = 25
PINNED_SHA = "5ff16a3c20968d6a5610d3584322b9a0059ad482"

SCRIPTS = [
    "eval/controlled_eval/exclusive_diar_api.py",
    "eval/controlled_eval/exclusive_diar_probe.py",
    "eval/controlled_eval/exclusive_diar_run.py",
    "eval/controlled_eval/exclusive_diar_analyze.py",
    "eval/controlled_eval/oc_merge_port.py",
    "eval/controlled_eval/build_exclusive_audit.py",
    "eval/controlled_eval/exclusive_freeze.py",
    "eval/controlled_eval/scoring.py",
    "eval/oc_inference_harness.py",
    "docs/specs/exclusive-diarization-preregistration.md",
]


def sha(p: Path) -> str | None:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def select_phase2_windows() -> list[dict]:
    """Top-25 by speaker-turn density, ties by item id ascending. Frozen here."""
    corpus = json.loads((SC / "precision2_corpus.json").read_text())
    rows = []
    for wid, v in corpus.items():
        turns = v.get("turns") or []
        dur = float(v.get("measured_dur") or 0)
        if not turns or dur <= 0:
            continue
        rows.append({"window_id": wid,
                     "n_turns": len(turns),
                     "duration_sec": round(dur, 3),
                     "turns_per_min": round(len(turns) / (dur / 60), 4),
                     "n_speakers": (v.get("feat") or {}).get("n_speakers")})
    rows.sort(key=lambda r: (-r["turns_per_min"], r["window_id"]))
    return rows[:N_WINDOWS]


def main():
    if OUT.exists():
        log(f"freeze record already exists: {OUT}\nsha256 = {sha(OUT)}")
        return

    windows = select_phase2_windows()
    WINDOWS_OUT.write_text(json.dumps(windows, ensure_ascii=False, indent=1))
    log(f"phase 2 windows -> {WINDOWS_OUT}")
    for r in windows[:5]:
        log(f"  {r['window_id']}  {r['turns_per_min']} turns/min  "
            f"{r['n_turns']} turns / {r['duration_sec']}s")

    rec = {
        "frozen_at": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                    capture_output=True, text=True).stdout.strip(),
        "spec": "docs/specs/exclusive-diarization-preregistration.md",
        "opencouncil_tasks_commit": PINNED_SHA,
        "vault_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "scripts_sha256": {s: sha(REPO / s) for s in SCRIPTS},
        "inputs_sha256": {
            "synth_overlap_manifest.json": sha(SC / "synth_overlap_manifest.json"),
            "precision2_corpus.json": sha(SC / "precision2_corpus.json"),
            "exclusive_phase2_windows.json": sha(WINDOWS_OUT),
        },
        "api": {
            "base": "https://api.pyannote.ai/v1",
            "endpoint": "POST /diarize",
            "payload_baseline": {"url": "media://...", "model": "precision-2"},
            "payload_proposal": {"url": "media://...", "model": "precision-2",
                                 "exclusive": True},
            "poll": "GET /v1/jobs/{id}, 300 s timeout, one retry",
            "rate_eur_per_hour": 0.112,
        },
        "phase1": {
            "arms": ["A/exclusive", "C/exclusive", "C/baseline"],
            "n_items": 95,
            "n_jobs": 285,
            "submission_seed": 20260807,
            "tiling_sec": 5.0,
            "gate": {"absorption_rate": 0.80,
                     "absorption_must_beat_regular": True,
                     "fragmentation_median": 1.2,
                     "fragmentation_frac_above_1_2": 0.10,
                     "merge_sim_guess_plus_drop": "exclusive <= regular"},
        },
        "phase2": {
            "n_windows": N_WINDOWS,
            "window_ids": [w["window_id"] for w in windows],
            "asr": "faster-whisper large-v3, word timestamps, greedy, lang=el, VAD off",
            "utterance_builder": "eval/oc_inference_harness.py::_words_to_utterances",
            "adjudication_seed": 20260807,
            "max_adjudicated": 50,
            "gate": {"min_determinate": 40,
                     "attribution_prop": 0.6667,
                     "attribution_lower_bound_above": 0.5,
                     "drop_noninferiority_upper_per_100": 0.5,
                     "max_neither_frac": 0.20,
                     "max_cant_tell_frac": 0.40,
                     "port_parity_tests": 4},
        },
    }
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    log(f"\n-> {OUT}\nsha256 = {sha(OUT)}")


if __name__ == "__main__":
    main()
