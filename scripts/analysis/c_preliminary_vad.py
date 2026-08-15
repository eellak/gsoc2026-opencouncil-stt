#!/usr/bin/env python3
"""Arm C preliminary feasibility: are our deletions inside VAD-positive audio the
decoder already covered (shifted-window merge cannot help) or inside audio spans
the decode pass never covered (shifted-window merge could recover)?

Spec: docs/specs/2026-08-12-serving-stack-plan.md, arm C pre-check. Analysis only.

Inputs (all local):
  - 39 eval window WAVs : ~/.cache/oc-public/bench_windows/<wid>.wav
  - control hypotheses  : ~/.cache/oc-public/decode-ablation/eval-A.json
                          (aggregate only: text, n_segments, decoded_seconds,
                           audio_seconds — no per-segment timestamps)
  - references          : data/reports/finetune-research/2026-08-10-corrected-adapter-report-full.json
  - window list         : research/eval-freeze-2026-08/manifest.json eval_windows
  - frozen tokenization : eval.controlled_eval.eval_freeze.ftoks
  - S/D/I alignment     : eval.controlled_eval.exp_same_stack.sdi

Output: data/reports/finetune-research/c-preliminary-vad-2026-08-12.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402

WAV_DIR = Path.home() / ".cache/oc-public/bench_windows"
CACHE = Path.home() / ".cache/oc-public/decode-ablation/eval-A.json"
REPORT = ROOT / "data/reports/finetune-research/2026-08-10-corrected-adapter-report-full.json"
MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
OUT = ROOT / "data/reports/finetune-research/c-preliminary-vad-2026-08-12.json"

# Silero VAD defaults — recorded explicitly so the config is auditable.
VAD_CONFIG = {
    "package": "silero_vad",
    "model": "silero_vad (bundled ONNX=False torch jit, load_silero_vad())",
    "sampling_rate": 16000,
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": "inf",
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
    "torch_num_threads": 8,
}

MATERIAL_GAP_ABS_S = 2.0
MATERIAL_GAP_REL = 0.05  # of VAD speech


def load_wav_16k_mono(path: Path) -> torch.Tensor:
    """WAVs in bench_windows are 16 kHz mono s16le; silero's read_audio needs
    torchcodec which is not installed, so load with stdlib wave instead."""
    import wave
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 \
            and w.getsampwidth() == 2, f"unexpected wav format: {path}"
        raw = w.readframes(w.getnframes())
    x = torch.frombuffer(bytearray(raw), dtype=torch.int16)
    return x.to(torch.float32) / 32768.0


def main() -> None:
    from silero_vad import load_silero_vad, get_speech_timestamps
    import silero_vad as sv

    VAD_CONFIG["silero_vad_version"] = getattr(sv, "__version__", "unknown")

    manifest = json.loads(MANIFEST.read_text())
    eval_windows = manifest["eval_windows"]  # 39; holdout untouched
    hyps = json.loads(CACHE.read_text())["windows"]
    refs = {it["itemId"]: it["referenceText"]
            for it in json.loads(REPORT.read_text())["items"]}

    model = load_silero_vad()

    rows = []
    for w in eval_windows:
        wid = w["window_id"]
        wav_path = WAV_DIR / f"{wid}.wav"
        audio = load_wav_16k_mono(wav_path)
        islands = get_speech_timestamps(
            audio, model,
            threshold=VAD_CONFIG["threshold"],
            sampling_rate=VAD_CONFIG["sampling_rate"],
            min_speech_duration_ms=VAD_CONFIG["min_speech_duration_ms"],
            min_silence_duration_ms=VAD_CONFIG["min_silence_duration_ms"],
            speech_pad_ms=VAD_CONFIG["speech_pad_ms"],
            return_seconds=True,
        )
        vad_speech = round(sum(i["end"] - i["start"] for i in islands), 2)

        h = hyps[wid]
        ref_t = ftoks(refs[wid])
        hyp_t = ftoks(h["text"])
        s, d, ins = sdi(ref_t, hyp_t)

        decoded = h["decoded_seconds"]
        audio_s = h["audio_seconds"]
        gap = round(vad_speech - decoded, 2)
        uncovered = max(gap, 0.0)
        rel_gap = round(gap / vad_speech, 4) if vad_speech > 0 else None
        rows.append({
            "window_id": wid,
            "city": w["city"],
            "vad_speech_s": vad_speech,
            "n_islands": len(islands),
            "islands": [[round(i["start"], 2), round(i["end"], 2)] for i in islands],
            "decoded_s": decoded,
            "audio_s": audio_s,
            "audio_minus_decoded_s": round(audio_s - decoded, 2),
            "gap_vad_minus_decoded_s": gap,
            "rel_gap": rel_gap,
            "ref_tokens": len(ref_t),
            "hyp_tokens": len(hyp_t),
            "sub": s, "del": d, "ins": ins,
            "dels_per_uncovered_s": round(d / uncovered, 2) if uncovered > 0 else None,
            "material_gap_abs": gap > MATERIAL_GAP_ABS_S,
            "material_gap_rel": (rel_gap is not None and rel_gap > MATERIAL_GAP_REL),
        })
        # Upper bound on deletions attributable to coverage: even if every
        # undecoded second (audio - decoded) were pure speech at this window's
        # own speech rate, at most this many reference words could live there.
        rate = len(ref_t) / vad_speech if vad_speech > 0 else 0.0
        bound = round(max(audio_s - decoded, 0.0) * rate, 1)
        rows[-1]["speech_rate_tok_per_s"] = round(rate, 2)
        rows[-1]["max_coverage_recoverable_dels"] = bound
        rows[-1]["coverage_bound_covers_dels"] = bound >= d

    rows_by_gap = sorted(rows, key=lambda r: r["gap_vad_minus_decoded_s"], reverse=True)
    rows_by_del = sorted(rows, key=lambda r: r["del"], reverse=True)

    total_del = sum(r["del"] for r in rows)
    del_in_abs = sum(r["del"] for r in rows if r["material_gap_abs"])
    del_in_rel = sum(r["del"] for r in rows if r["material_gap_rel"])

    result = {
        "generated": "2026-08-12",
        "question": ("are deletions located in VAD-positive audio the control decode "
                     "did not cover (recoverable by shifted-window pass) or inside "
                     "covered spans (not recoverable by arm C)?"),
        "vad_config": VAD_CONFIG,
        "material_gap_definition": {
            "abs": f"> {MATERIAL_GAP_ABS_S} s (vad_speech - decoded_seconds)",
            "rel": f"> {MATERIAL_GAP_REL:.0%} of vad_speech",
        },
        "aggregate": {
            "windows": len(rows),
            "total_deletions": total_del,
            "deletions_in_material_abs_gap_windows": del_in_abs,
            "fraction_abs": round(del_in_abs / total_del, 4) if total_del else None,
            "deletions_in_material_rel_gap_windows": del_in_rel,
            "fraction_rel": round(del_in_rel / total_del, 4) if total_del else None,
            "windows_material_abs": sum(r["material_gap_abs"] for r in rows),
            "windows_material_rel": sum(r["material_gap_rel"] for r in rows),
            "coverage_upper_bound_dels": round(
                sum(min(r["del"], r["max_coverage_recoverable_dels"])
                    for r in rows), 1),
            "coverage_upper_bound_fraction": round(
                sum(min(r["del"], r["max_coverage_recoverable_dels"])
                    for r in rows) / total_del, 4) if total_del else None,
        },
        "decoder_telemetry_note": (
            "eval-A per-window telemetry shows temperatures_used=[0.0], "
            "compression_trips=0, logprob_trips=0, no_speech_skips=0 on every "
            "high-deletion window: deletions were produced by confident "
            "first-pass beam decoding, not by fallback resampling or "
            "no-speech gating."),
        "deletion_count_note": (
            "per-window deletions here use eval-A control hyps + frozen ftoks + "
            "exp_same_stack.sdi; counts quoted elsewhere from other arms/reports "
            "(e.g. sep10=61, orestiada_dec11=34) do not match this stack "
            "(here sep10=33, orestiada_dec11=15) and were not used."),
        "phase2": {
            "script": "scripts/analysis/c_phase2_timed_decode.py",
            "status": "written, NOT run",
            "windows": [
                "win_argos_oct31__2_2025_2353650",
                "win_argos_sep24_2025_371824",
                "win_argos_may22_2026_1650387",
                "win_argos_aug29_2025_2731295",
                "win_orestiada_feb27_2026_305602",
                "win_argos_sep10_2025_573077",
                "win_argos_jan30_2026_204180",
                "win_orestiada_jan21_2026_7536686",
                "win_argos_apr7_2026_960810",
            ],
        },
        "caveat": ("decoded_seconds is an aggregate span sum, not per-segment "
                   "coverage; a window can show decoded_s ~ audio_s while still "
                   "having interior unattended spans. Aggregate parity is evidence "
                   "against coverage gaps, not proof of within-span attention."),
        "ranked_by_gap": [r["window_id"] for r in rows_by_gap[:10]],
        "ranked_by_deletions": [
            {"window_id": r["window_id"], "del": r["del"],
             "gap_s": r["gap_vad_minus_decoded_s"]} for r in rows_by_del[:10]],
        "per_window": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))

    print(f"{'window':44s} {'vad_s':>7} {'dec_s':>7} {'aud_s':>7} {'gap':>6} "
          f"{'del':>4} {'sub':>4} {'ins':>4}")
    for r in rows_by_del:
        print(f"{r['window_id']:44s} {r['vad_speech_s']:7.1f} {r['decoded_s']:7.1f} "
              f"{r['audio_s']:7.1f} {r['gap_vad_minus_decoded_s']:6.1f} "
              f"{r['del']:4d} {r['sub']:4d} {r['ins']:4d}")
    print("\naggregate:", json.dumps(result["aggregate"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
