"""Re-measure RUN2 stage-2 against the current adapter on one CPU stack.

The decode and scoring substrates are deliberately the existing frozen ones:
``notebooks.decode_ablation.CONTROL`` for decoding, ``eval_freeze.ftoks`` for
normalisation, ``exp_same_stack.sdi`` for the S/D/I alignment, and
``scoring.cluster_bootstrap`` for meeting-clustered paired intervals.

Hypotheses remain in ``$SC``.  The repository output is aggregate-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap  # noqa: E402

import decode_ablation as DA  # noqa: E402


MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
LEDGER = ROOT / "research/ledger.json"
RESULTS = ROOT / "eval/results_run2_local.json"
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
WINDOWS = SC / "bench_windows"

CONTROL_ADAPTER = Path("/home/harold/oc-asr-serve/adapter-fixed-2026-08-01")
CONTROL_CT2 = Path("/home/harold/oc-asr-serve/ct2-fixed")
RUN2_ADAPTER = SC / "train-screens-2026-08/run2-artifacts/stage2-adapter"
RUN2_CT2 = Path.home() / "oc-run2-stage2/ct2"
RUN2_ATTESTATION = SC / "train-screens-2026-08/run2-artifacts/pod-sha256.txt"
BUILD_MODEL = Path("/home/harold/oc-asr-serve/build_model.sh")

CONTROL_CACHE = SC / "decode-ablation/eval-A.json"
RUN2_CACHE = SC / "train-screens-2026-08/run2-eval-stage2/decode.json"

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
THREADS = 6
CACHE_DECODE_THREADS = 16
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 7

CONTROL_ADAPTER_ARTIFACT = "artifact-adapter-fixed"
RUN2_ARTIFACT = "run2-stage2-adapter-seed101"
CONTROL_ADAPTER_SHA256 = (
    "ea8f03230846888fa7e4c341813efea324cf5596e689da2d48b1de365eb0a5a6"
)
# The ledger has no artifact record for RUN2's stage-2 adapter.  This is the
# full hash from the preserved pod attestation, and the missing ledger record is
# surfaced in the result instead of being silently treated as a ledger check.
RUN2_ADAPTER_SHA256 = (
    "730121a0161aef781f6dee55903cf8f8cf4676d0574b8ec5b4244a8768547fcd"
)
CONTROL_CT2_SHA256 = (
    "8a1a3b257d0c1bdb71877f36db902a46c14697ff587766b91d6c47973f8fb85b"
)
RUN2_CT2_SHA256 = (
    "444de4e963742227654a39bb3eabac45541279177ce59b69ff4669d7787e0cac"
)

SEED_CONFOUND_CAVEAT = (
    "RUN2 differs from the control in BOTH the data mixture AND the random seed - "
    "101 against 13. The measured per-seed WER spread in this project is 2.1 "
    "points, larger than any difference this comparison can resolve. Therefore "
    "no difference found here may be attributed to the mixture. This is a hard "
    "limit, not a footnote."
)
DELETION_CAVEAT = (
    "RUN2 previously raised deletion rate (0.0756 against control 0.0600, with "
    "a CI excluding zero); deletions are watched explicitly because a lower WER "
    "achieved by omitting hard passages is worse."
)


def decode_config() -> dict:
    """Return the already-frozen project decode dictionary, before scoring."""
    return dict(DA.CONTROL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_adapter_hash(path: Path, expected: str) -> str:
    """Verify a LoRA adapter file before any hypothesis is scored."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"sha256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def verify_ct2_hash(path: Path, expected: str) -> str:
    model_bin = path / "model.bin"
    actual = sha256_file(model_bin)
    if actual != expected:
        raise ValueError(
            f"CT2 model.bin sha256 mismatch for {model_bin}: "
            f"expected {expected}, got {actual}"
        )
    return actual


def ledger_artifact(artifact_id: str) -> dict | None:
    ledger = json.loads(LEDGER.read_text())
    return next((a for a in ledger["artifacts"] if a["id"] == artifact_id), None)


def verify_control_against_ledger(path: Path) -> tuple[str, str]:
    artifact = ledger_artifact(CONTROL_ADAPTER_ARTIFACT)
    if artifact is None:
        raise ValueError(f"ledger has no record for {CONTROL_ADAPTER_ARTIFACT}")
    expected = artifact.get("hash", {}).get("value_full")
    if not expected:
        raise ValueError(
            f"ledger record for {CONTROL_ADAPTER_ARTIFACT} has no full sha256"
        )
    actual = verify_adapter_hash(path / "adapter_model.safetensors", expected)
    if actual != CONTROL_ADAPTER_SHA256:
        raise ValueError(
            "embedded control hash disagrees with the ledger; stop and reconcile"
        )
    return actual, "research/ledger.json"


def verify_run2_attestation(path: Path) -> tuple[str, str]:
    actual = verify_adapter_hash(path / "adapter_model.safetensors", RUN2_ADAPTER_SHA256)
    if not RUN2_ATTESTATION.exists():
        raise ValueError(f"missing RUN2 adapter attestation: {RUN2_ATTESTATION}")
    attested = {
        line.split()[0]
        for line in RUN2_ATTESTATION.read_text().splitlines()
        if len(line.split()) >= 2
    }
    if actual not in attested:
        raise ValueError("RUN2 adapter hash is absent from its preserved attestation")
    return actual, str(RUN2_ATTESTATION)


def rows() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text())
    selected = manifest["eval_windows"]
    if len(selected) != 39:
        raise ValueError(f"frozen validation set has {len(selected)} windows, not 39")
    if len({r["meeting_id"] for r in selected}) != 31:
        raise ValueError("frozen validation set does not have the expected 31 meetings")
    return selected


def validate_decode_state(state: dict, expected_model: str | None, path: Path) -> None:
    if state.get("config") != decode_config():
        raise ValueError(f"{path} was decoded with a different frozen config")
    if expected_model is not None and state.get("model") != expected_model:
        raise ValueError(
            f"{path} holds a decode of {state.get('model')}, not {expected_model}"
        )
    actual = set(state.get("windows", {}))
    expected = {r["window_id"] for r in rows()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{path} is not a complete 39-window decode; "
                         f"missing={missing[:3]} extra={extra[:3]}")


def load_decode_cache(path: Path, expected_model: str | None) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    state = json.loads(path.read_text())
    validate_decode_state(state, expected_model, path)
    return state


def _new_decode_state(model: Path) -> dict:
    return {
        "model": str(model),
        "config": decode_config(),
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "cpu_threads": THREADS,
        "windows": {},
    }


def decode_arm(model: Path, destination: Path) -> dict:
    """Decode one arm, resumably, with the frozen config and <=6 CPU threads."""
    import ctranslate2
    from faster_whisper import WhisperModel

    state = json.loads(destination.read_text()) if destination.exists() else _new_decode_state(model)
    validate_decode_state_partial = state.get("config") == decode_config()
    if not validate_decode_state_partial:
        raise ValueError(f"{destination} was decoded with a different frozen config")
    if state.get("model") not in (None, str(model)):
        raise ValueError(f"{destination} belongs to another model")
    state["model"] = str(model)
    state["device"] = DEVICE
    state["compute_type"] = COMPUTE_TYPE
    state["cpu_threads"] = THREADS
    state.setdefault("windows", {})
    destination.parent.mkdir(parents=True, exist_ok=True)

    todo = [r for r in rows() if r["window_id"] not in state["windows"]]
    if not todo:
        validate_decode_state(state, str(model), destination)
        return state

    asr = WhisperModel(str(model), device=DEVICE, compute_type=COMPUTE_TYPE,
                       cpu_threads=THREADS)
    for row in todo:
        wid = row["window_id"]
        wav = WINDOWS / f"{wid}.wav"
        if not wav.exists():
            raise FileNotFoundError(wav)
        ctranslate2.set_random_seed(DA.seed_for("A", wid))
        started = time.monotonic()
        segments, info = asr.transcribe(str(wav), **decode_config())
        segments = list(segments)
        state["windows"][wid] = {
            "text": "".join(segment.text for segment in segments).strip(),
            "n_segments": len(segments),
            "decoded_seconds": round(sum(s.end - s.start for s in segments), 2),
            "audio_seconds": round(info.duration, 2),
            "wall_seconds": round(time.monotonic() - started, 1),
            "seed": DA.seed_for("A", wid),
        }
        destination.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    validate_decode_state(state, str(model), destination)
    return state


def score_counts(refs: dict[str, list[str]], hyps: dict[str, list[str]]) -> dict[str, tuple[int, int, int, int]]:
    """Score through the frozen S/D/I scorer; do not reimplement alignment here."""
    if set(refs) != set(hyps):
        raise ValueError("reference and hypothesis windows are not paired")
    out = {}
    for wid in refs:
        s, d, ins = sdi(refs[wid], hyps[wid])
        out[wid] = (s, d, ins, len(refs[wid]))
    return out


def score_state(state: dict) -> dict[str, tuple[int, int, int, int]]:
    frozen_rows = rows()
    refs = {r["window_id"]: ftoks(DA.reference_text(r["window_id"])) for r in frozen_rows}
    hyps = {r["window_id"]: ftoks(state["windows"][r["window_id"]]["text"])
            for r in frozen_rows}
    return score_counts(refs, hyps)


def rates(scores: dict[str, tuple[int, int, int, int]]) -> dict:
    sub = sum(v[0] for v in scores.values())
    deletion = sum(v[1] for v in scores.values())
    insertion = sum(v[2] for v in scores.values())
    ref_tokens = sum(v[3] for v in scores.values())
    return {
        "substitutions": sub,
        "deletions": deletion,
        "insertions": insertion,
        "ref_tokens": ref_tokens,
        "wer": (sub + deletion + insertion) / ref_tokens,
        "deletion_rate": deletion / ref_tokens,
        "insertion_rate": insertion / ref_tokens,
        "substitution_rate": sub / ref_tokens,
    }


PICKS: dict[str, Callable[[tuple[int, int, int, int]], int]] = {
    "wer": lambda t: t[0] + t[1] + t[2],
    "deletion_rate": lambda t: t[1],
    "insertion_rate": lambda t: t[2],
    "substitution_rate": lambda t: t[0],
}


def paired_comparison(left: dict, right: dict, wids: list[str], blocks: list[str]) -> dict:
    """Return left-minus-right CIs, refusing a cross-config paired comparison."""
    if left.get("config") != right.get("config"):
        raise ValueError("paired comparison refuses different decode configs")
    left_scores = left["scores"]
    right_scores = right["scores"]
    if set(wids) != set(left_scores) or set(wids) != set(right_scores):
        raise ValueError("paired comparison requires complete common windows")
    result = {}
    for metric, pick in PICKS.items():
        left_values = [(pick(left_scores[w]), left_scores[w][3]) for w in wids]
        right_values = [(pick(right_scores[w]), right_scores[w][3]) for w in wids]
        result[metric] = cluster_bootstrap(
            left_values,
            right_values,
            blocks,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
    return result


def single_item_domination(left_scores: dict, right_scores: dict, wids: list[str]) -> dict:
    """Leave one window out before quoting any pooled delta."""
    result = {}
    for metric, pick in PICKS.items():
        full_n = sum(left_scores[w][3] for w in wids)
        full_effect = sum(pick(left_scores[w]) - pick(right_scores[w]) for w in wids)
        per_window = {
            w: pick(left_scores[w]) - pick(right_scores[w]) for w in wids
        }
        if full_effect:
            largest = max(per_window, key=lambda w: abs(per_window[w] / full_effect))
            share = per_window[largest] / full_effect
        else:
            largest = max(per_window, key=lambda w: abs(per_window[w]))
            share = None
        loo = {}
        for w in wids:
            keep = [x for x in wids if x != w]
            denominator = sum(left_scores[x][3] for x in keep)
            effect = sum(pick(left_scores[x]) - pick(right_scores[x]) for x in keep)
            loo[w] = effect / denominator
        full_delta = full_effect / full_n
        largest_loo = max(loo, key=lambda w: abs(loo[w] - full_delta))
        result[metric] = {
            "full_delta": full_delta,
            "largest_net_error_share": share,
            "largest_net_error_share_window": largest,
            "delta_without_largest_shift_window": loo[largest_loo],
            "largest_shift_window": largest_loo,
            "sign_reversed_by": [w for w in wids if (full_delta < 0) != (loo[w] < 0)],
        }
    return result


def _metadata(state: dict, path: Path, source: str, decode_threads: int) -> dict:
    return {
        "path": str(path),
        "model": state.get("model"),
        "config": state["config"],
        "windows": len(state["windows"]),
        "source": source,
        "decode_threads": decode_threads,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


def load_run2_meta() -> dict:
    meta = json.loads((RUN2_ADAPTER / "run_meta.json").read_text())
    expected = {
        "seed": 101,
        "n_train": 42204,
        "max_steps": 10552,
        "epochs": 2,
        "lora_r": 32,
        "label_semantics": "decoder_start_v2",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"RUN2 metadata mismatch for {key}: {meta.get(key)!r}")
    return {key: meta[key] for key in expected}


def build_if_needed() -> dict:
    """Use the verified existing build; expose the approved build command if absent."""
    if RUN2_CT2.exists() and (RUN2_CT2 / "model.bin").exists():
        return {"status": "reused_verified_existing", "path": str(RUN2_CT2)}
    raise RuntimeError(
        "RUN2 CT2 build is absent. The approved build command is "
        f"ADAPTER={RUN2_ADAPTER} BASE=openai/whisper-large-v3 {BUILD_MODEL}; "
        "this run cannot write outside the workspace sandbox."
    )


def assemble_result(control_state: dict, run2_state: dict, control_scores: dict,
                    run2_scores: dict, artifact_info: dict, run2_meta: dict,
                    wall_seconds: float, cache_source: str) -> dict:
    frozen_rows = rows()
    wids = [r["window_id"] for r in frozen_rows]
    blocks = [r["meeting_id"] for r in frozen_rows]
    left = {"config": run2_state["config"], "scores": run2_scores}
    right = {"config": control_state["config"], "scores": control_scores}
    paired = paired_comparison(left, right, wids, blocks)
    domination = single_item_domination(run2_scores, control_scores, wids)
    return {
        "experiment": "exp-2026-08-17-run2-local",
        "seed_confound_caveat": SEED_CONFOUND_CAVEAT,
        "summary": (
            SEED_CONFOUND_CAVEAT + " Local CPU re-measurement: RUN2 stage-2 "
            "versus artifact-adapter-fixed, with deletions reported first-class."
        ),
        "scope": {
            "windows": len(wids),
            "meetings": len(set(blocks)),
            "holdout_windows_touched": 0,
            "locked_evaluation_windows_touched": 0,
            "metric": "agreement-with-OpenCouncil using the frozen scorer",
        },
        "artifacts": artifact_info,
        "run2_training_identity": run2_meta,
        "decode_config": decode_config(),
        "threads_used": {
            "current_process_max": THREADS,
            "decoded_cache_threads": CACHE_DECODE_THREADS,
            "decodes_concurrent": False,
        },
        "stack": {
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "threads_used": THREADS,
            "sequential": True,
            "cache_source": cache_source,
            "cached_decode_threads": CACHE_DECODE_THREADS,
            "note": (
                "Complete CPU caches were reused. Their producing screen report "
                "records CPU int8 with 16 threads for both arms; this process used "
                "at most 6 threads and did not run concurrent decoding."
            ),
        },
        "per_arm": {
            "control_artifact-adapter-fixed": {
                "rates": rates(control_scores),
                "known_cached_stack_wer": 0.1589,
            },
            "run2-stage2": {"rates": rates(run2_scores)},
        },
        "paired_difference": {
            "direction": "run2-stage2 minus control artifact-adapter-fixed",
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "clusters": "meeting_id",
            },
            "metrics": paired,
        },
        "single_item_domination": {
            "performed": True,
            "method": "leave-one-window-out and net-error contribution",
            "metrics": domination,
            "warning": (
                "A delta is not quoted as a model improvement when its sign or "
                "magnitude is supplied by one window."
            ),
        },
        "deletion_watch": {
            "previous_screen_caveat": DELETION_CAVEAT,
            "run2_rate": rates(run2_scores)["deletion_rate"],
            "control_rate": rates(control_scores)["deletion_rate"],
        },
        "known_stack_comparison": {
            "control_local_wer": rates(control_scores)["wer"],
            "control_known_cached_stack_wer": 0.1589,
            "known_cached_gpu_vs_local_symmetric_edit_rate": 0.0926,
            "note": (
                "0.0926 is the previously measured symmetric normalized edit rate "
                "for cached GPU versus local CPU int8 at wt=False, not a WER delta."
            ),
        },
        "wall_clock": {
            "command_seconds": round(wall_seconds, 3),
            "cached_decode_seconds": {
                "control": round(sum(v.get("wall_seconds", 0) for v in control_state["windows"].values()), 1),
                "run2_stage2": round(sum(v.get("wall_seconds", 0) for v in run2_state["windows"].values()), 1),
            },
        },
        "blockers": [
            "The ledger has no artifact record for RUN2 stage-2; its full adapter "
            "hash was verified against the preserved pod attestation instead."
        ],
    }


def run(reuse_cache: bool = True) -> dict:
    started = time.monotonic()
    # Freeze and verify identities before producing or reading any metric.
    config = decode_config()
    if config != DA.CONTROL:
        raise ValueError("frozen decode config drifted from notebooks.decode_ablation")
    control_adapter_hash, control_hash_source = verify_control_against_ledger(CONTROL_ADAPTER)
    run2_adapter_hash, run2_hash_source = verify_run2_attestation(RUN2_ADAPTER)
    control_ct2_hash = verify_ct2_hash(CONTROL_CT2, CONTROL_CT2_SHA256)
    build_info = build_if_needed()
    run2_ct2_hash = verify_ct2_hash(RUN2_CT2, RUN2_CT2_SHA256)
    run2_meta = load_run2_meta()

    if reuse_cache:
        control_state = load_decode_cache(CONTROL_CACHE, None)
        run2_state = load_decode_cache(RUN2_CACHE, str(RUN2_CT2))
        cache_source = "complete verified local CPU decode caches"
    else:
        # Sequential calls are intentional: do not starve another CPU decode.
        control_state = decode_arm(CONTROL_CT2, CONTROL_CACHE)
        run2_state = decode_arm(RUN2_CT2, RUN2_CACHE)
        cache_source = "fresh sequential local CPU decode with 6 threads"

    control_scores = score_state(control_state)
    run2_scores = score_state(run2_state)
    artifact_info = {
        "control": {
            "artifact_id": CONTROL_ADAPTER_ARTIFACT,
            "adapter_path": str(CONTROL_ADAPTER),
            "adapter_model_sha256": control_adapter_hash,
            "adapter_hash_verified_against": control_hash_source,
            "ct2_path": str(CONTROL_CT2),
            "ct2_model_bin_sha256": control_ct2_hash,
            "ct2_hash_verified_against": "research/ledger.json",
        },
        "run2_stage2": {
            "artifact_id": RUN2_ARTIFACT,
            "adapter_path": str(RUN2_ADAPTER),
            "adapter_model_sha256": run2_adapter_hash,
            "adapter_hash_verified_against": run2_hash_source,
            "ct2_path": str(RUN2_CT2),
            "ct2_model_bin_sha256": run2_ct2_hash,
            "ct2_hash_verified_against": "docs/reports/2026-08-16-screens-eval.md",
            "build": build_info,
        },
    }
    result = assemble_result(
        control_state,
        run2_state,
        control_scores,
        run2_scores,
        artifact_info,
        run2_meta,
        time.monotonic() - started,
        cache_source,
    )
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print_compact(result)
    return result


def print_compact(result: dict) -> None:
    print(result["summary"])
    for arm, data in result["per_arm"].items():
        r = data["rates"]
        print(f"{arm}: WER {r['wer']:.4f} del {r['deletion_rate']:.4f} "
              f"ins {r['insertion_rate']:.4f} sub {r['substitution_rate']:.4f}")
    for metric, value in result["paired_difference"]["metrics"].items():
        print(f"delta {metric}: {value['delta']:+.5f} "
              f"[{value['ci95'][0]:+.5f}, {value['ci95'][1]:+.5f}]")
    print(f"threads: current max {THREADS}; cached decode {CACHE_DECODE_THREADS}; "
          f"wall {result['wall_clock']['command_seconds']:.1f}s")
    print(f"wrote {RESULTS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "score"))
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="fresh sequential CPU decode; default reuses complete verified caches",
    )
    args = parser.parse_args()
    run(reuse_cache=not args.fresh)


if __name__ == "__main__":
    main()
