"""Training-set WER for the v2 clean-pack contiguous adapter, on its own corpus.

The sample was frozen on 2026-08-23 at ``research/train-wer-v2/sample.json``, before
any decode, and is read here rather than re-derived.  It is 300 packs drawn from the
2,476-pack clean-pack contiguous corpus that trained
``artifact-adapter-cleanpack-cont-s47``.

This number is NOT comparable to the v1 training WER of 0.1313: different corpus,
different row length, different reference construction.  The sample says so itself and
the guard is carried into the output.  What it *is* comparable to is the base arm
decoded on the same rows in the same run, which is the whole point of running two arms.

Audio and hypotheses stay under ``~/.cache/oc-public``.  Only aggregates reach git.

    .venv-eval/bin/python -m eval.controlled_eval.train_wer_v2 verify
    .venv-eval/bin/python -m eval.controlled_eval.train_wer_v2 decode v2 --threads 6
    .venv-eval/bin/python -m eval.controlled_eval.train_wer_v2 decode base --threads 6
    .venv-eval/bin/python -m eval.controlled_eval.train_wer_v2 score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import decode_ablation as DA  # noqa: E402
from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap  # noqa: E402

SAMPLE_PATH = ROOT / "research/train-wer-v2/sample.json"
PACK_CACHE = Path.home() / ".cache/oc-public/clean-pack-arm-2026-08"
SEED_NAMESPACE = "train-wer-v2"

# text_pn and text_p are byte-identical on all 300 sampled rows, so the choice cannot
# move a number here; ``verify`` re-checks that rather than trusting this comment.
REFERENCE_FIELD = "text_pn"

ARMS = {
    # Provenance proven by rebuild on 2026-08-25: openai/whisper-large-v3 with the
    # local cont_s47 adapter (adapter_model.safetensors 5e4b55d2803541aa..., the hash
    # the ledger records for artifact-adapter-cleanpack-cont-s47) merged through PEFT
    # merge_and_unload and converted with --quantization int8_float16 reproduces
    # model.bin byte for byte. Two independent rebuilds agreed, so the pipeline is
    # deterministic and the match is not a coincidence.
    "v2": {
        "artifact": "artifact-ct2-cleanpack-cont-s47",
        "model": Path("/home/harold/oc-asr-serve/ct2-v2"),
        "model_sha256_16": "0c6976f120f12f7c",
    },
    # The matched base. NOT the Systran published conversion the v1 harness used:
    # that binary differs (69f74147...) and was converted by someone else, so pairing
    # it against a locally converted v2 would confound adaptation with conversion
    # lineage. ct2-base and ct2-v2 were written five minutes apart on 2026-08-23 by
    # the same local build.
    "base": {
        "artifact": "artifact-ct2-base-large-v3-local",
        "model": Path("/home/harold/oc-asr-serve/ct2-base"),
        "model_sha256_16": "bba445638b80555f",
    },
}

# One of the 2,476 packs in the source manifest carries a 480-token label and was cut
# before training by the 448-token decoder cap, so the executed training set is 2,475.
# The frozen sample was drawn over the 2,476-row manifest; this asserts the dropped
# pack did not land in it, which would otherwise put an unseen row in a training-WER
# measurement. See docs/reports/2026-08-21-clean-pack-screen.md.
EXCLUDED_OVER_CAP = "31cba9e6d6dc28a7"
EXECUTED_TRAINING_ROWS = 2475


def work_dir() -> Path:
    path = Path.home() / ".cache/oc-public/train-wer-v2-2026-08-n300"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows() -> tuple[dict, list[dict]]:
    """Resolve the frozen sample against the local pack cache, hash-checking every row.

    Every row carries its own audio and reference digest, so a silently re-cut clip or
    a re-exported reference fails here rather than quietly changing the result.
    """
    spec = json.loads(SAMPLE_PATH.read_text())
    entries = spec["sample"]
    declared = spec["rule"]["n"]
    if len(entries) != declared:
        raise SystemExit(f"sample holds {len(entries)} rows, rule declares {declared}")
    ids = [entry["pack_id"] for entry in entries]
    if len(set(ids)) != len(ids):
        # A duplicate collapses in the per-key count dict but survives in the bootstrap
        # input list, so the point estimate and the interval would describe different
        # samples.
        raise SystemExit("duplicate pack_id in the frozen sample")

    manifest = PACK_CACHE / "packs.jsonl"
    manifest_hash = sha256_file(manifest)
    expected = spec["population"]["manifest_sha256"]
    if manifest_hash != expected:
        raise SystemExit(f"packs.jsonl hash {manifest_hash} != frozen {expected}")

    packs = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        pack = json.loads(line)
        packs[Path(pack["audio"]).stem] = pack

    if EXCLUDED_OVER_CAP in {entry["pack_id"] for entry in spec["sample"]}:
        raise SystemExit(
            f"{EXCLUDED_OVER_CAP} was cut by the 448-token cap and never trained on; "
            "it cannot appear in a training-WER sample")

    rows = []
    for entry in spec["sample"]:
        pack = packs.get(entry["pack_id"])
        if pack is None:
            raise SystemExit(f"pack {entry['pack_id']} missing from the manifest")
        clip = PACK_CACHE / pack["audio"]
        audio_hash = sha256_file(clip)
        if audio_hash != entry["audio_sha256"]:
            raise SystemExit(f"{entry['pack_id']}: audio hash {audio_hash} != frozen")
        reference = pack[REFERENCE_FIELD]
        if sha256_bytes(reference.encode()) != entry["reference_sha256"]:
            raise SystemExit(f"{entry['pack_id']}: reference hash mismatch")
        # The claim that the field choice cannot move the number is checked, not
        # asserted in prose: the other field must hash to the same frozen digest.
        other = "text_p" if REFERENCE_FIELD == "text_pn" else "text_pn"
        if sha256_bytes(pack[other].encode()) != entry["reference_sha256"]:
            raise SystemExit(
                f"{entry['pack_id']}: {other} differs from {REFERENCE_FIELD}; the "
                "reference field choice now changes the result and must be declared")
        rows.append({
            "key": entry["pack_id"],
            "clip": str(clip),
            "reference": reference,
            "city_id": pack["city_id"],
            "meeting_id": pack["meeting_id"],
            "person_id": pack["person_id"],
            "speech_sec": float(pack["speech_sec"]),
            "span_sec": float(pack["span_sec"]),
            "n_utterances": int(pack["n_utterances"]),
        })
    return spec, rows


def sample_fingerprint(rows: list[dict]) -> str:
    """One digest over the resolved row set, stored in every decode file.

    Guards the case the frozen sample is edited between two arms; without it the two
    arms could be scored against different references and the pairing would be a lie.
    """
    payload = "\n".join(f"{row['key']}|{sha256_bytes(row['reference'].encode())}"
                        for row in rows)
    return sha256_bytes(payload.encode())


def verify() -> None:
    spec, rows = load_rows()
    seconds = sum(row["span_sec"] for row in rows)
    meetings = {(row["city_id"], row["meeting_id"]) for row in rows}
    print(json.dumps({
        "experiment": spec["experiment"],
        "n_rows": len(rows),
        "fingerprint": sample_fingerprint(rows),
        "n_meetings": len(meetings),
        "n_cities": len({row["city_id"] for row in rows}),
        "n_speakers": len({row["person_id"] for row in rows}),
        "audio_seconds": round(seconds, 1),
        "reference_field": REFERENCE_FIELD,
        "arms_present": {
            name: (arm["model"] / "model.bin").exists() for name, arm in ARMS.items()
        },
    }, ensure_ascii=False, indent=1))


def decode(arm_name: str, threads: int) -> None:
    import ctranslate2
    from faster_whisper import WhisperModel

    _, rows = load_rows()
    fingerprint = sample_fingerprint(rows)
    arm = ARMS[arm_name]
    model_hash = sha256_file(arm["model"] / "model.bin")
    if not model_hash.startswith(arm["model_sha256_16"]):
        raise SystemExit(f"{arm_name} model hash mismatch: {model_hash}")

    dest = work_dir() / f"decode-{arm_name}.json"
    state = json.loads(dest.read_text()) if dest.exists() else {
        "arm": arm_name,
        "artifact": arm["artifact"],
        "model_sha256": model_hash,
        "sample_fingerprint": fingerprint,
        "device": DA.DEVICE,
        "compute_type": DA.COMPUTE,
        "cpu_threads": threads,
        "config": DA.CONTROL,
        "rows": {},
    }
    # Everything that defines the stack, not just the sample. Resuming a half-decoded
    # arm on a different device or compute type would silently produce one arm whose
    # rows came from two stacks, which no downstream check could detect.
    expected = {
        "artifact": arm["artifact"],
        "model_sha256": model_hash,
        "sample_fingerprint": fingerprint,
        "device": DA.DEVICE,
        "compute_type": DA.COMPUTE,
        "cpu_threads": threads,
        "config": DA.CONTROL,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise SystemExit(f"stale decode {key}: {state.get(key)!r} != {value!r}")

    todo = [row for row in rows if row["key"] not in state["rows"]]
    print(f"{arm_name}: {len(todo)} rows to decode ({len(state['rows'])} done), "
          f"{threads} threads", flush=True)
    if not todo:
        return
    model = WhisperModel(str(arm["model"]), device=DA.DEVICE, compute_type=DA.COMPUTE,
                         cpu_threads=threads)
    for index, row in enumerate(todo, 1):
        started = time.time()
        record: dict | None = None
        failures = []
        # The frozen rule: the attempt plus two clean retries, then the row stays with
        # an empty hypothesis and the failure is disclosed rather than dropped.
        for attempt in range(3):
            try:
                ctranslate2.set_random_seed(DA.seed_for(SEED_NAMESPACE, row["key"]))
                segments = list(model.transcribe(row["clip"], **DA.CONTROL)[0])
                # Preserve segment boundaries so scoring cannot invent fused words.
                record = {"segments": [segment.text for segment in segments]}
                break
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                failures.append(f"{type(error).__name__}: {error}")
                print(f"  {row['key']} attempt {attempt + 1} failed: {failures[-1]}",
                      flush=True)
        if record is None:
            record = {"segments": [], "failed": True}
        if failures:
            record["failures"] = failures
        record["wall_seconds"] = round(time.time() - started, 3)
        state["rows"][row["key"]] = record
        # Atomic: a kill between truncate and write would otherwise leave a torn file
        # and lose every row decoded so far.
        # Per-process temporary name: two decoders on one arm would otherwise race on
        # a shared tmp file and interleave two partial states into it.
        tmp = dest.with_suffix(f".json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        tmp.replace(dest)
        if index % 10 == 0 or index == len(todo):
            done = len(state["rows"])
            print(f"  {index}/{len(todo)} (total {done}/{len(rows)})", flush=True)


def load_state(arm_name: str) -> dict:
    return json.loads((work_dir() / f"decode-{arm_name}.json").read_text())


def check_stack(states: dict[str, dict]) -> dict:
    """Refuse to score arms that were not decoded on one stack.

    The decode step guards its own resume, but nothing stopped two arms produced weeks
    apart under different settings from being paired here. A paired contrast across
    two stacks is the exact error this project has paid for more than once, so the
    values compared are the ones recorded next to the hypotheses, never the current
    module constants.
    """
    for name, state in states.items():
        if state["config"] != DA.CONTROL:
            raise SystemExit(f"{name}: decoded under a different decode config")
        prefix = ARMS[name]["model_sha256_16"]
        if not state["model_sha256"].startswith(prefix):
            raise SystemExit(f"{name}: decoded with model {state['model_sha256'][:16]}, "
                             f"the record names {prefix}")
    shared = {}
    for field in ("cpu_threads", "device", "compute_type"):
        values = {name: state.get(field) for name, state in states.items()}
        if len(set(values.values())) > 1:
            raise SystemExit(f"arms disagree on {field}: {values}; that is two stacks, "
                             "not one comparison")
        shared[field] = next(iter(values.values()))
    if shared["cpu_threads"] is None:
        raise SystemExit("no recorded cpu_threads; the stack cannot be established")
    return shared


def arm_counts(rows: list[dict], arm_name: str) -> dict[str, tuple[int, int, int, int]]:
    state = load_state(arm_name)
    if state["sample_fingerprint"] != sample_fingerprint(rows):
        raise SystemExit(f"{arm_name}: sample fingerprint mismatch")
    out = {}
    for row in rows:
        decoded = state["rows"].get(row["key"])
        if decoded is None:
            raise SystemExit(f"{arm_name}: incomplete decode at {row['key']}")
        reference = ftoks(row["reference"])
        hypothesis = [token for segment in decoded["segments"] for token in ftoks(segment)]
        out[row["key"]] = (*sdi(reference, hypothesis), len(reference))
    return out


def rates(counts: dict[str, tuple[int, int, int, int]]) -> dict[str, float | int | None]:
    values = list(counts.values())
    if not values:
        return {"n_rows": 0, "ref_tokens": 0, "wer": None}
    sub, delete, insert, n_ref = (sum(row[i] for row in values) for i in range(4))
    return {
        "n_rows": len(values),
        "sub": sub,
        "del": delete,
        "ins": insert,
        "ref_tokens": n_ref,
        "wer": (sub + delete + insert) / n_ref,
        "sub_rate": sub / n_ref,
        "del_rate": delete / n_ref,
        "ins_rate": insert / n_ref,
    }


def slices(rows: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Diagnostic strata.

    The v1 sample split on correction/no_edit, and that split carried the whole result.
    The clean-pack corpus has no such label, so the substitutes are structural: which
    council, how long the pack is, and how many separate turns were concatenated into
    it. A pack of many short turns is a different acoustic object from one long turn.
    """
    by_city: dict[str, list[str]] = {}
    for row in rows:
        by_city.setdefault(row["city_id"], []).append(row["key"])

    # Bins frozen from the manifest, not from any outcome. n_utterances runs 1..17 with
    # median 7 and exactly one single-utterance pack, so a single-versus-multi split
    # would be empty on one side.
    by_turns: dict[str, list[str]] = {}
    for row in rows:
        count = row["n_utterances"]
        label = "1-5" if count <= 5 else "6-9" if count <= 9 else "10+"
        by_turns.setdefault(label, []).append(row["key"])

    # Occupancy = speech / span: how much of the pack is speech rather than pause.
    # Ranges 0.751-0.987 here, so terciles separate pause-heavy from dense packs.
    ordered = sorted(rows, key=lambda row: row["speech_sec"] / row["span_sec"])
    occupancy: dict[str, list[str]] = {}
    for index, row in enumerate(ordered):
        label = f"t{index * 3 // len(ordered) + 1}"
        occupancy.setdefault(label, []).append(row["key"])

    return {"city": by_city, "n_utterances": by_turns, "occupancy_tercile": occupancy}


def dominance(counts: dict[str, tuple[int, int, int, int]],
              rows: list[dict]) -> dict:
    by_key = {row["key"]: row for row in rows}
    total = sum(sum(value[:3]) for value in counts.values())
    top_key, top_count = max(
        ((key, sum(value[:3])) for key, value in counts.items()), key=lambda item: item[1]
    )
    meetings: dict[tuple[str, str], int] = {}
    for key, value in counts.items():
        row = by_key[key]
        block = (row["city_id"], row["meeting_id"])
        meetings[block] = meetings.get(block, 0) + sum(value[:3])
    top_meeting, top_meeting_count = max(meetings.items(), key=lambda item: item[1])
    return {
        "largest_row_key": top_key,
        "largest_row_errors": top_count,
        "largest_row_share": top_count / total if total else None,
        "largest_meeting_key": hashlib.sha256("|".join(top_meeting).encode()).hexdigest()[:16],
        "largest_meeting_errors": top_meeting_count,
        "largest_meeting_share": top_meeting_count / total if total else None,
    }


def contrast_dominance(a: dict[str, tuple[int, int, int, int]],
                       b: dict[str, tuple[int, int, int, int]],
                       rows: list[dict]) -> dict:
    by_key = {row["key"]: row for row in rows}
    contributions = {key: sum(a[key][:3]) - sum(b[key][:3]) for key in a}
    net = sum(contributions.values())
    top_key, top_value = max(contributions.items(), key=lambda item: abs(item[1]))

    meetings: dict[tuple[str, str], int] = {}
    for key, value in contributions.items():
        row = by_key[key]
        block = (row["city_id"], row["meeting_id"])
        meetings[block] = meetings.get(block, 0) + value
    top_meeting, top_meeting_value = max(meetings.items(), key=lambda item: abs(item[1]))

    cities: dict[str, int] = {}
    for key, value in contributions.items():
        city = by_key[key]["city_id"]
        cities[city] = cities.get(city, 0) + value

    def loo(groups: dict) -> dict:
        """Leave-one-out over a grouping: does any single group carry the sign?

        Reversal is ``remaining * net < 0`` so a positive and a negative net are
        judged the same way; ``(value > 0) != (net > 0)`` calls a remainder of exactly
        zero a reversal when net is positive and not when it is negative. Reaching
        zero is reported separately as sign_lost, since it kills the effect without
        turning it around.
        """
        values = list(groups.values())
        if not net:
            return {"range": None, "sign_reversal": None, "sign_lost": None}
        left = [net - value for value in values]
        return {
            "range": [min(left), max(left)],
            "sign_reversal": any(value * net < 0 for value in left),
            "sign_lost": any(value * net <= 0 for value in left),
        }

    # Share statistics are reported two ways on purpose. The absolute-contribution
    # share degenerates when net is near zero and can nominate a row that pushed the
    # other way; the positive-benefit share says how concentrated the gain itself is.
    positive = sum(value for value in contributions.values() if value > 0)
    top_five = sorted(contributions.values(), key=abs, reverse=True)[:5]
    # The largest *positive* row, chosen among rows that actually helped. top_key is
    # picked by absolute size, so on contributions +5, +5, -9 it names the row that
    # hurt; calling that row 90% of the benefit would be backwards.
    gains = {key: value for key, value in contributions.items() if value > 0}
    top_gain_key, top_gain = (
        max(gains.items(), key=lambda item: item[1]) if gains else (None, 0))
    return {
        "net_error_difference": net,
        "rows_better": sum(1 for value in contributions.values() if value > 0),
        "rows_tied": sum(1 for value in contributions.values() if value == 0),
        "rows_worse": sum(1 for value in contributions.values() if value < 0),
        "largest_row_key": top_key,
        "largest_row_error_difference": top_value,
        "largest_row_share_of_net": abs(top_value / net) if net else None,
        "largest_gaining_row_key": top_gain_key,
        "largest_gaining_row_errors": top_gain,
        "largest_gaining_row_share_of_positive": (
            top_gain / positive if positive else None),
        "top_five_row_share_of_net": abs(sum(top_five) / net) if net else None,
        "largest_meeting_share_of_net": abs(top_meeting_value / net) if net else None,
        "largest_city_share_of_net": (
            abs(max(cities.values(), key=abs) / net) if net and cities else None),
        "leave_one_row_out": loo(contributions),
        "leave_one_meeting_out": loo(meetings),
        "leave_one_city_out": loo(cities),
    }


def score(output: Path) -> None:
    spec, rows = load_rows()
    available = [name for name in ARMS if (work_dir() / f"decode-{name}.json").exists()]
    if "v2" not in available:
        raise SystemExit("the v2 arm is required")
    states = {name: load_state(name) for name in available}
    stack = check_stack(states)
    counts = {name: arm_counts(rows, name) for name in available}
    totals = {name: rates(value) for name, value in counts.items()}
    keys = [row["key"] for row in rows]
    clusters = [(row["city_id"], row["meeting_id"]) for row in rows]
    strata = slices(rows)

    levels = {}
    for name in available:
        observed = [(sum(counts[name][key][:3]), counts[name][key][3]) for key in keys]
        zero = [(0, counts[name][key][3]) for key in keys]
        levels[name] = cluster_bootstrap(observed, zero, clusters, n_boot=4000, seed=7)

    wall = {name: sum(row["wall_seconds"] for row in states[name]["rows"].values())
            for name in available}

    result = {
        "experiment": spec["experiment"],
        "metric": "agreement with the labels used to train the v2 adapter; not fidelity to audio",
        "sample": {
            "source": str(SAMPLE_PATH.relative_to(ROOT)),
            "frozen_before": spec["frozen_before"],
            "population": spec["population"],
            "rule": spec["rule"],
            "n_rows": len(rows),
            "fingerprint": sample_fingerprint(rows),
            "n_meetings": len(set(clusters)),
            "n_cities": len({row["city_id"] for row in rows}),
            "n_speakers": len({row["person_id"] for row in rows}),
            "audio_seconds": round(sum(row["span_sec"] for row in rows), 1),
            "speech_seconds": round(sum(row["speech_sec"] for row in rows), 1),
            "executed_training_rows": EXECUTED_TRAINING_ROWS,
            "population_note": (
                f"Selection was frozen over the {spec['population']['rows']}-row source "
                f"manifest, but training executed on {EXECUTED_TRAINING_ROWS} rows: pack "
                f"{EXCLUDED_OVER_CAP} carries a 480-token label and was cut by the "
                "448-token decoder cap. It is not among the 300 selected, so every "
                "scored row was genuinely trained on."
            ),
            "reference_field": REFERENCE_FIELD,
            "reference_field_note": (
                "text_pn and text_p are byte-identical on all 300 sampled rows; verify "
                "re-checks both digests, so the field choice cannot move this number."
            ),
        },
        # Read back from the decode files, not from the module constants, so this
        # block describes the stack that produced the hypotheses.
        "decode": {
            "device": stack["device"],
            "compute_type": stack["compute_type"],
            "cpu_threads": stack["cpu_threads"],
            "config": states["v2"]["config"],
            "normalizer": "eval.controlled_eval.eval_freeze.ftoks",
            "segment_join": "tokenize each segment, then concatenate token lists",
            "seed_policy": f"ctranslate2.set_random_seed(DA.seed_for({SEED_NAMESPACE!r}, pack_id))",
        },
        "arms": {
            name: {
                "artifact": ARMS[name]["artifact"],
                "model_sha256_16": ARMS[name]["model_sha256_16"],
                "decode_wall_seconds": round(wall[name], 1),
                **totals[name],
                "wer_ci95_meeting_clustered": levels[name]["ci95"],
                "failed_rows": sorted(key for key, row in states[name]["rows"].items()
                                      if row.get("failed")),
                "by_slice": {
                    axis: {
                        label: rates({key: counts[name][key] for key in members})
                        for label, members in sorted(groups.items())
                    }
                    for axis, groups in strata.items()
                },
                "dominance": dominance(counts[name], rows),
            }
            for name in available
        },
        "not_comparable_to": spec["not_comparable_to"],
        "interpretation_guard": (
            "Training WER measures agreement with the labels the model saw. It diagnoses "
            "fit and memorisation. It cannot establish label quality, fidelity to audio, "
            "or which adapter to ship, and it must never select an arm."
        ),
    }

    if "base" in available:
        # Every component gets its own paired interval. A WER interval alone cannot
        # say whether a gain is recognition or coverage, which is the one question
        # this project keeps needing answered.
        components = {"wer": (0, 1, 2), "sub": (0,), "del": (1,), "ins": (2,)}
        paired_ci = {}
        for name, fields in components.items():
            paired_ci[name] = cluster_bootstrap(
                [(sum(counts["base"][key][i] for i in fields), counts["base"][key][3])
                 for key in keys],
                [(sum(counts["v2"][key][i] for i in fields), counts["v2"][key][3])
                 for key in keys],
                clusters, n_boot=4000, seed=7,
            )["ci95"]
        result["base_minus_v2"] = {
            "wer_delta": totals["base"]["wer"] - totals["v2"]["wer"],
            "wer_delta_ci95_meeting_clustered": paired_ci["wer"],
            "sub_rate_delta": totals["base"]["sub_rate"] - totals["v2"]["sub_rate"],
            "sub_rate_delta_ci95_meeting_clustered": paired_ci["sub"],
            "del_rate_delta": totals["base"]["del_rate"] - totals["v2"]["del_rate"],
            "del_rate_delta_ci95_meeting_clustered": paired_ci["del"],
            "ins_rate_delta": totals["base"]["ins_rate"] - totals["v2"]["ins_rate"],
            "ins_rate_delta_ci95_meeting_clustered": paired_ci["ins"],
            "by_slice": {
                axis: {
                    label: {
                        metric: result["arms"]["base"]["by_slice"][axis][label][metric]
                        - result["arms"]["v2"]["by_slice"][axis][label][metric]
                        for metric in ("wer", "sub_rate", "del_rate", "ins_rate")
                    }
                    for label in sorted(groups)
                }
                for axis, groups in strata.items()
            },
            "dominance": contrast_dominance(counts["base"], counts["v2"], rows),
            "interpretation": (
                "Positive values mean the base model is worse. Paired on the same frozen "
                "rows in the same run, so it measures what adaptation learned on its own "
                "labels. It does not establish label fidelity or validation improvement."
            ),
        }

    output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(result["arms"], ensure_ascii=False, indent=1))
    if "base_minus_v2" in result:
        print(json.dumps(result["base_minus_v2"], ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    decode_parser = sub.add_parser("decode")
    decode_parser.add_argument("arm", choices=sorted(ARMS))
    decode_parser.add_argument("--threads", type=int, default=6)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--output", type=Path,
                              default=ROOT / "eval/results_train_wer_v2.json")
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "decode":
        decode(args.arm, args.threads)
    else:
        score(args.output)


if __name__ == "__main__":
    main()
