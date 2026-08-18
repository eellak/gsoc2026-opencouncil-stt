#!/usr/bin/env python3
"""Build and score the confidence-bearing fusion substrate.

The three stages deliberately keep verbatim benchmark and decoder material under
``~/.cache/oc-public``.  Nothing written by this module belongs in the repository.
The adapter is decoded with the existing ``RW`` arm semantics: frozen CONTROL plus
``word_timestamps=True`` and no other change.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval.controlled_eval import bench_data as B                    # noqa: E402
from eval.controlled_eval import fusion_lab as F                    # noqa: E402
from eval.controlled_eval.column_classes import column_class       # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import rates, sdi    # noqa: E402
from eval.controlled_eval.msa import align3, compose                 # noqa: E402
from eval.controlled_eval.scoring import wtoks                        # noqa: E402

import decode_ablation as DA                                          # noqa: E402
import served_config_and_july as SERVED                              # noqa: E402


RUN_ID = F.RUN_ID
TRIO = tuple(F.TRIO)
ADAPTER_PROVIDER = "oc-runpod-fixed-2026-08-10"
BASELINE_W = 0.10046
FROZEN_EXACT_2_OF_3_COLUMNS = 6645
EXPECTED_ALIGNMENT_CACHE = "align_65b1c4d64618a429.json"
SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 7


def cache_dir() -> Path:
    """The only directory into which this stage writes output."""
    return Path.home() / ".cache" / "oc-public" / "conf-substrate-2026-08"


def decode_path() -> Path:
    return cache_dir() / "decode-rw.json"


def build_path() -> Path:
    return cache_dir() / "substrate-rw.json"


def measure_path() -> Path:
    return cache_dir() / "measure-rw.json"


def benchmark_report_path() -> Path:
    return B.cache_dir() / f"bench_{RUN_ID}.json"


def model_path() -> Path:
    return Path(SERVED.ARMS["RW"][0])


def resolved_config() -> dict:
    """Return exactly CONTROL with the one approved RW override."""
    got = copy.deepcopy(SERVED.config_for("RW"))
    expected = copy.deepcopy(DA.CONTROL)
    expected["word_timestamps"] = True
    if got != expected:
        raise SystemExit(
            "RW config is not frozen CONTROL plus word_timestamps=True: "
            f"{got!r} != {expected!r}"
        )
    return got


def _code_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _seed_scheme() -> str:
    return "sha256(window_id)[:4] interpreted as big-endian uint32"


def _current_identity() -> dict:
    """Verify the model before constructing any decode state."""
    digest = SERVED.verify_model("RW")
    environment = SERVED.environment()
    return {
        "model": str(model_path()),
        "model_sha256_16": digest,
        "config": resolved_config(),
        "environment": environment,
        "thread_count": DA.THREADS,
        "seed": _seed_scheme(),
        "code_sha": _code_sha(),
    }


def validate_resume(state: dict, expected: dict) -> None:
    """Refuse to mix decode passes with different provenance."""
    for key in ("model", "model_sha256_16", "config", "environment"):
        if state.get(key) != expected.get(key):
            raise SystemExit(
                f"{decode_path()} was written under a different {key}; "
                "delete it rather than extending it"
            )
    if not isinstance(state.get("windows"), dict):
        raise SystemExit(f"{decode_path()} has no valid windows map")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON cache {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"JSON cache {path} is not an object")
    return value


def _write_json(path: Path, value: dict) -> None:
    """Atomically publish each incremental cache update."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    tmp.replace(path)


def _sealed_ids() -> set[str]:
    manifest = json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text()
    )
    return {row["window_id"] for row in manifest["holdout_windows"]}


def substrate_items(limit: int | None = None) -> list[dict]:
    """Load the exact fusion_lab window set, including its sealed-window guard."""
    report = B.load_report(RUN_ID)
    providers = B.provider_ids(report)
    missing = [provider for provider in TRIO if provider not in providers]
    if missing:
        raise SystemExit(f"benchmark report is missing providers: {missing}")
    # The window set must be the one fusion_lab.load_substrate builds, or the new
    # baseline is not measured on the same windows as W. That means conditioning on
    # EVERY provider in the report, not just the trio: restricting to the trio admits
    # windows the other providers are missing, which yields 7 sealed windows instead
    # of 6 and a substrate W never scored.
    items = B.common_items(report, B.provider_ids(report))
    sealed = _sealed_ids()
    before = len(items)
    items = [item for item in items if item["item_id"] not in sealed]
    removed = before - len(items)
    if removed != F.N_SEALED_INSIDE:
        raise SystemExit(
            f"expected {F.N_SEALED_INSIDE} sealed windows inside this run, removed {removed}"
        )
    if len(items) != F.N_WINDOWS:
        raise SystemExit(f"expected {F.N_WINDOWS} windows, got {len(items)}")
    if limit is not None:
        if limit < 1:
            raise SystemExit("limit must be positive")
        items = items[:limit]
    return items


def _window_seed(window_id: str) -> int:
    # This is the common-random-number convention used by decode_ablation and the
    # RW arm.  It intentionally does not vary by arm.
    return DA.seed_for("A", window_id)


def _value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _float_or_none(value):
    return None if value is None else float(value)


def _word_payload(word) -> dict:
    raw = _value(word, "word", _value(word, "w", ""))
    start = _value(word, "start", _value(word, "s"))
    end = _value(word, "end", _value(word, "e"))
    probability = _value(word, "probability", _value(word, "p"))
    if raw is None or start is None or end is None or probability is None:
        raise SystemExit("word timestamps must contain word, start, end and probability")
    probability = float(probability)
    if not 0.0 < probability <= 1.0:
        raise SystemExit(f"word probability outside (0, 1]: {probability!r}")
    return {
        "w": str(raw),
        "s": float(start),
        "e": float(end),
        "p": probability,
    }


def _segment_payload(segment) -> dict:
    raw_words = _value(segment, "words")
    if raw_words is None:
        raise SystemExit("RW returned a segment without word timestamps")
    return {
        "start": float(_value(segment, "start")),
        "end": float(_value(segment, "end")),
        "text": str(_value(segment, "text", "")),
        "avg_logprob": _float_or_none(_value(segment, "avg_logprob")),
        "no_speech_prob": _float_or_none(_value(segment, "no_speech_prob")),
        "temperature": _float_or_none(_value(segment, "temperature")),
        "words": [_word_payload(word) for word in raw_words],
    }


def _resolved_options(info) -> dict | None:
    options = _value(info, "transcription_options")
    if options is None:
        return None
    return DA.opts_to_dict(options)


def _check_resolved_options(options: dict | None) -> None:
    if not options:
        return
    expected = resolved_config()
    aliases = {"temperature": "temperatures"}
    for key, value in expected.items():
        resolved_key = aliases.get(key, key)
        if resolved_key in options:
            got = options[resolved_key]
            if got != value:
                raise SystemExit(
                    f"RW resolved {resolved_key}={got!r}, expected {value!r}"
                )


def decode(limit: int | None = None) -> Path:
    """Incrementally decode the substrate windows with per-word probabilities."""
    identity = _current_identity()
    dest = decode_path()
    if dest.exists():
        state = _read_json(dest)
        validate_resume(state, identity)
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "exp-2026-08-18-conf-substrate",
            **identity,
            "windows": {},
        }
        _write_json(dest, state)

    items = substrate_items(limit)
    todo = [item for item in items if item["item_id"] not in state["windows"]]
    print(
        f"RW confidence decode: {len(todo)} to decode, "
        f"{len(state['windows'])} already present",
        flush=True,
    )
    if not todo:
        return dest

    import ctranslate2
    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(model_path()), device=DA.DEVICE, compute_type=DA.COMPUTE,
        cpu_threads=DA.THREADS,
    )
    logger = logging.getLogger("faster_whisper")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        for number, item in enumerate(todo, 1):
            window_id = item["item_id"]
            wav = DA.sc() / "bench_windows" / f"{window_id}.wav"
            if not wav.exists():
                raise SystemExit(f"missing audio for {window_id}: {wav}")
            seed = _window_seed(window_id)
            ctranslate2.set_random_seed(seed)
            started = time.time()
            segments, info = model.transcribe(str(wav), **identity["config"])
            segments = list(segments)
            stored_segments = [_segment_payload(segment) for segment in segments]
            options = _resolved_options(info)
            _check_resolved_options(options)
            state["windows"][window_id] = {
                "text": " ".join(
                    segment["text"].strip()
                    for segment in stored_segments
                    if segment["text"].strip()
                ).strip(),
                "segments": stored_segments,
                "n_segments": len(stored_segments),
                "audio_seconds": _float_or_none(_value(info, "duration")),
                "temperatures_used": sorted({
                    segment["temperature"] for segment in stored_segments
                    if segment["temperature"] is not None
                }),
                "seed": seed,
                "resolved_options": options,
                "wall_seconds": round(time.time() - started, 1),
            }
            _write_json(dest, state)
            print(
                f"  {number}/{len(todo)} {window_id} "
                f"{state['windows'][window_id]['wall_seconds']:.0f}s "
                f"segments={len(stored_segments)}",
                flush=True,
            )
    finally:
        logger.setLevel(old_level)
    print(f"RW confidence decode -> {dest}", flush=True)
    return dest


def word_token_confidences(words: list[dict]) -> tuple[list[str], list[float]]:
    """Expand timestamped words into frozen-normalizer tokens and probabilities."""
    tokens: list[str] = []
    probabilities: list[float] = []
    for word in words:
        raw = str(word["w"])
        subtokens = wtoks(raw)
        p = float(word["p"])
        if not 0.0 < p <= 1.0:
            raise ValueError(f"word probability outside (0, 1]: {p!r}")
        tokens.extend(subtokens)
        probabilities.extend([p] * len(subtokens))
    return tokens, probabilities


def decoded_tokens_and_confidence(window: dict) -> tuple[list[str], list[float]]:
    words = [word for segment in window["segments"] for word in segment["words"]]
    word_tokens, word_probabilities = word_token_confidences(words)
    text_tokens = wtoks(window["text"])
    if word_tokens != text_tokens:
        raise SystemExit(
            "decoded text and word timestamps do not normalize to the same token "
            f"sequence: {text_tokens[:8]!r} != {word_tokens[:8]!r}"
        )
    return word_tokens, word_probabilities


def column_confidences(
    columns: list[tuple | list],
    adapter_tokens: list[str],
    adapter_probabilities: list[float],
) -> list[float | None]:
    """Attach the adapter confidence to its aligned contribution, else ``None``."""
    if len(adapter_tokens) != len(adapter_probabilities):
        raise ValueError("adapter token and probability sequences differ in length")
    result: list[float | None] = []
    token_index = 0
    for column in columns:
        adapter_token = column[2]
        if adapter_token is None:
            result.append(None)
            continue
        if token_index >= len(adapter_tokens):
            raise ValueError("alignment consumed more adapter tokens than decoded")
        if adapter_token != adapter_tokens[token_index]:
            raise ValueError(
                f"alignment adapter token {adapter_token!r} does not match "
                f"decoded token {adapter_tokens[token_index]!r}"
            )
        result.append(float(adapter_probabilities[token_index]))
        token_index += 1
    if token_index != len(adapter_tokens):
        raise ValueError(
            f"alignment consumed {token_index} of {len(adapter_tokens)} adapter tokens"
        )
    return result


def _alignment_cache_guard() -> Path:
    cache = F._cache_path()
    if cache.name != EXPECTED_ALIGNMENT_CACHE:
        raise SystemExit(
            "MSA alignment cache key moved: "
            f"{cache.name} != {EXPECTED_ALIGNMENT_CACHE}; refusing to build"
        )
    return cache


def _band(a: list[str], b: list[str], c: list[str]) -> int:
    return max(F.BAND_FLOOR, max(len(a), len(b), len(c)) - min(len(a), len(b), len(c)) + 20)


def _align_one(payload):
    window_id, sequences, pivot = payload
    cols = align3(*sequences, band=_band(*sequences))
    w_tokens, decisions = compose(cols, pivot=pivot)
    return window_id, cols, w_tokens, decisions


def _decode_identity(state: dict) -> dict:
    return {
        key: state.get(key)
        for key in ("model", "model_sha256_16", "config", "environment", "thread_count")
    }


def _new_build_state(decode_state: dict, alignment_cache: Path) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "exp-2026-08-18-conf-substrate",
        "run_id": RUN_ID,
        "providers": list(TRIO),
        "adapter_provider": ADAPTER_PROVIDER,
        "decode_identity": _decode_identity(decode_state),
        "alignment_cache_key": alignment_cache.name,
        "windows": {},
    }


def build() -> Path:
    """Align the two benchmark rows with the RW adapter row and attach confidence."""
    alignment_cache = _alignment_cache_guard()
    decode_state = _read_json(decode_path())
    if decode_state.get("model_sha256_16") != "8a1a3b257d0c1bdb":
        raise SystemExit("decode cache does not carry the approved CT2 model digest")
    if decode_state.get("config") != resolved_config():
        raise SystemExit("decode cache was not produced by the frozen RW config")
    if decode_state.get("alignment_cache_key") not in (None, EXPECTED_ALIGNMENT_CACHE):
        raise SystemExit("decode cache carries an unexpected alignment cache key")

    items = substrate_items()
    dest = build_path()
    if dest.exists():
        state = _read_json(dest)
        if state.get("alignment_cache_key") != alignment_cache.name:
            raise SystemExit("existing confidence substrate has a different MSA cache key")
        if state.get("decode_identity") != _decode_identity(decode_state):
            raise SystemExit("existing confidence substrate was built from a different decode")
    else:
        state = _new_build_state(decode_state, alignment_cache)
        _write_json(dest, state)

    payloads = []
    confidence_by_window = {}
    pivot_by_window = {}
    sequences_by_window = {}
    for item in items:
        window_id = item["item_id"]
        decoded = decode_state["windows"].get(window_id)
        if decoded is None:
            raise SystemExit(f"decode cache is incomplete: {window_id} is missing")
        adapter_tokens, adapter_probabilities = decoded_tokens_and_confidence(decoded)
        hyps = dict(item["hyp"])
        hyps[ADAPTER_PROVIDER] = decoded["text"]
        sequences = [wtoks(hyps[provider]) for provider in TRIO]
        if sequences[2] != adapter_tokens:
            raise SystemExit(f"adapter token sequence mismatch for {window_id}")
        pivot_provider = B.consensus_pick({**item, "hyp": hyps}, TRIO)
        payloads.append((window_id, sequences, TRIO.index(pivot_provider)))
        confidence_by_window[window_id] = (adapter_tokens, adapter_probabilities)
        pivot_by_window[window_id] = pivot_provider
        sequences_by_window[window_id] = sequences

    todo = [payload for payload in payloads if payload[0] not in state["windows"]]
    workers = int(os.environ.get("WORKERS", str(min(8, os.cpu_count() or 4))))
    print(f"building confidence substrate: {len(todo)} windows on {workers} workers", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for number, (window_id, columns, w_tokens, decisions) in enumerate(
                executor.map(_align_one, todo, chunksize=1), 1
            ):
                item = next(row for row in items if row["item_id"] == window_id)
                adapter_tokens, adapter_probabilities = confidence_by_window[window_id]
                confidence = column_confidences(
                    columns, adapter_tokens, adapter_probabilities
                )
                state["windows"][window_id] = {
                    "city": item["city_id"],
                    "meeting": item["meeting_id"],
                    "ref": wtoks(item["ref"]),
                    "hyps": {
                        provider: sequences_by_window[window_id][index]
                        for index, provider in enumerate(TRIO)
                    },
                    "pivot": pivot_by_window[window_id],
                    "cols": [list(column) for column in columns],
                    "decisions": decisions,
                    "w_tokens": w_tokens,
                    "adapter_confidence": confidence,
                }
                _write_json(dest, state)
                if number % 25 == 0 or number == len(todo):
                    print(f"  {number}/{len(todo)}", flush=True)
    print(f"confidence substrate -> {dest}", flush=True)
    return dest


def _absolute_bootstrap(
    rows: list[tuple[int, int, int, int]],
    clusters: list[str],
    component: str,
    n_boot: int,
    seed: int,
) -> dict:
    import numpy as np

    index = {"wer": None, "del_rate": 1, "ins_rate": 2}[component]
    groups: dict[str, list[int]] = {}
    for position, cluster in enumerate(clusters):
        groups.setdefault(cluster, []).append(position)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for number in range(n_boot):
        picked = rng.integers(0, len(keys), len(keys))
        selected = [position for key_index in picked for position in groups[keys[key_index]]]
        denominator = sum(rows[position][3] for position in selected)
        if index is None:
            numerator = sum(sum(rows[position][:3]) for position in selected)
        else:
            numerator = sum(rows[position][index] for position in selected)
        samples[number] = numerator / denominator if denominator else np.nan
    if index is None:
        numerator = sum(sum(row[:3]) for row in rows)
    else:
        numerator = sum(row[index] for row in rows)
    denominator = sum(row[3] for row in rows)
    point = numerator / denominator if denominator else float("nan")
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return {
        "estimate": point,
        "ci95": [float(lo), float(hi)],
        "n_meetings": len(keys),
        "bootstrap_replicates": n_boot,
        "bootstrap_seed": seed,
    }


def _column_coverage(state: dict) -> dict:
    total = with_confidence = 0
    exact2_total = exact2_with_confidence = 0
    for window in state["windows"].values():
        columns = [tuple(column) for column in window["cols"]]
        confidences = window["adapter_confidence"]
        if len(columns) != len(confidences):
            raise SystemExit("substrate column and confidence lengths differ")
        for column, confidence in zip(columns, confidences):
            total += 1
            if confidence is not None:
                with_confidence += 1
            if column_class(column) == "exact_2_of_3":
                exact2_total += 1
                if confidence is not None:
                    exact2_with_confidence += 1
    return {
        "columns_total": total,
        "columns_with_adapter_confidence": with_confidence,
        "exact_2_of_3_equivalent": {
            "frozen_benchmark_total": FROZEN_EXACT_2_OF_3_COLUMNS,
            "new_alignment_total": exact2_total,
            "with_adapter_confidence": exact2_with_confidence,
        },
    }


def measure(n_boot: int | None = None) -> Path:
    """Score W-conf as a separate absolute baseline on the frozen normalizer."""
    state = _read_json(build_path())
    if state.get("alignment_cache_key") != EXPECTED_ALIGNMENT_CACHE:
        raise SystemExit("confidence substrate has an unexpected MSA alignment cache key")
    if len(state.get("windows", {})) != F.N_WINDOWS:
        raise SystemExit(
            f"confidence substrate is incomplete: expected {F.N_WINDOWS} windows, "
            f"got {len(state.get('windows', {}))}"
        )
    score_rows = []
    clusters = []
    for window in state["windows"].values():
        s, d, insertion, n_ref = sdi(
            " ".join(window["ref"]), " ".join(window["w_tokens"])
        )
        score_rows.append((s, d, insertion, n_ref))
        clusters.append(window["meeting"])
    if not score_rows:
        raise SystemExit("confidence substrate contains no windows")

    n_boot = n_boot if n_boot is not None else int(os.environ.get("N_BOOT", "10000"))
    aggregate = rates(score_rows)
    baseline = {
        "label": "W-conf re-decoded adapter baseline",
        "wer": _absolute_bootstrap(score_rows, clusters, "wer", n_boot, BOOTSTRAP_SEED),
        "deletion_rate": _absolute_bootstrap(
            score_rows, clusters, "del_rate", n_boot, BOOTSTRAP_SEED + 1
        ),
        "insertion_rate": _absolute_bootstrap(
            score_rows, clusters, "ins_rate", n_boot, BOOTSTRAP_SEED + 2
        ),
        "counts": aggregate,
    }
    result = {
        "experiment": "exp-2026-08-18-conf-substrate",
        "run_id": RUN_ID,
        "normalizer": "eval/controlled_eval/scoring.py::wtoks",
        "existing_W_baseline": {
            "wer": BASELINE_W,
            "label": "existing W benchmark baseline",
        },
        "w_conf_baseline": baseline,
        "column_coverage": _column_coverage(state),
        "note": (
            "W-conf is a separate baseline because its adapter row was re-decoded; "
            "it is not a delta or an improvement claim against existing W."
        ),
    }
    dest = measure_path()
    _write_json(dest, result)
    print(f"existing W baseline: {BASELINE_W:.5f}", flush=True)
    print(
        f"W-conf baseline: WER {baseline['wer']['estimate']:.5f}, "
        f"deletion {baseline['deletion_rate']['estimate']:.5f}, "
        f"insertion {baseline['insertion_rate']['estimate']:.5f}",
        flush=True,
    )
    print(
        "adapter confidence columns: "
        f"{result['column_coverage']['columns_with_adapter_confidence']} / "
        f"{result['column_coverage']['columns_total']}",
        flush=True,
    )
    print(f"measurement -> {dest}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("--limit", type=int, default=None)

    subparsers.add_parser("build")

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--bootstrap", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "decode":
        decode(args.limit)
    elif args.command == "build":
        build()
    elif args.command == "measure":
        measure(args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
