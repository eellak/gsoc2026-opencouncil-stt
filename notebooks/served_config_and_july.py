"""Score the configuration production serves, and the July adapter, on the frozen 39.

`exp-2026-08-17-served-config-and-july-adapter`. Two questions on one harness.

**A. What does the served decode configuration score?** Every decode conclusion in
this project was produced at `beam_size=5, word_timestamps=False`
(`notebooks/decode_ablation.CONTROL`). The deployment is not that. `~/oc-asr-serve/
asr.env` sets `OC_ASR_BEAM=2`, and `word_timestamps` is per-route in
`serve/oc-asr/oc_asr_server.py`: the product routes (`/transcribe`,
`/transcribe/upload`) pass `True`, the OpenAI-compatible route the benchmark calls
(`/v1/audio/transcriptions`) passes `False`. Two served configurations, neither
scored.

**B. Is the July adapter better than the corrected one?** Provenance answers it
first - `artifact-adapter-july-broken` trained through the label-prefix bug
(`exp-2026-07-31-label-prefix-bug`) - but a belief deserves a number, and the number
is cheap because `artifact-ct2-july-broken` is on disk.

## What this is, and what it is not

This is a **decode-option ablation on the frozen evaluation stack**: cpu / int8 /
16 threads, the stack every other number on these 39 windows came from. It is
**not** a production emulation. Three production behaviours are absent, and are
reported as residual gaps rather than papered over:

1. `OC_ASR_CPU_THREADS=8`. Held at 16 so these contrasts stay comparable with every
   existing number on this substrate. Bounded separately by `threadprobe`.
2. `OC_ASR_MAX_INFER_SEC=150`, a streaming guard that stops consuming segments and
   appends a truncation marker. Not emulated. `wallclock()` reports how many windows
   ran past 150 s, which is an *indicator* that the guard is reachable, not a
   counterfactual truncation result - the real guard also starts its timer before it
   acquires the inference lock, so queueing counts against it.
3. Route aggregation. Handled: see "Text assembly" below.

## Text assembly - the thing that had to be fixed before anything ran

`decode_ablation` stores `"".join(segment.text)`. faster-whisper does not always put
a leading space on a segment (505 of 1,677 boundaries had none in
`exp-2026-08-16-adapter-confidence`), so that join fuses the last word of a segment
into the first word of the next - and `word_timestamps` **moves segment boundaries**,
which is exactly the contrast under test here. Production does not do that: it emits
`seg.text.strip()` per utterance and `" ".join(...)` for the full transcript.

So the primary hypothesis for scoring is built **per segment**:

    hyp = [tok for seg in segments for tok in ftoks(seg)]

Segment texts are stored, so any later question can be re-scored without re-decoding.
The legacy raw join is stored too, and used only for the two arms whose decode
predates this experiment and stored nothing else.

## Arms

Every arm is `decode_ablation.CONTROL` with the named keys changed and nothing else;
`config_for()` enforces that against the resolved options, not just the request.

| arm | model | change from CONTROL | role |
|---|---|---|---|
| `R`  | ct2-fixed  | none                                | control, re-decoded today |
| `S1` | ct2-fixed  | `beam_size=2`                       | served benchmark route |
| `S2` | ct2-fixed  | `beam_size=2, word_timestamps=True` | served product route |
| `RW` | ct2-fixed  | `word_timestamps=True`              | the interaction cell |
| `J`  | ct2        | none                                | the July broken adapter |
| `R0` | ct2-fixed  | none                                | cached 2026-08-12 control |
| `CO` | corr ct2   | none                                | cached correction-only arm |

`R` is re-decoded rather than lifted from `$SC/decode-ablation/eval-A.json`. Two
reasons, both load-bearing. The cached file stores only the fused join, so it cannot
be scored the way the primary endpoint is defined. And the paired bootstrap resamples
meetings while treating the hypotheses as fixed - it contains no run-to-run decoder
variance - so a 2026-08-12 baseline cannot carry a causal attribution today, given
that `exp-2026-08-16-adapter-confidence` **withdrew** the bit-exactness claim for this
decode (16 of 18 windows reproduced, not 18 of 18). `R0` is kept as a separate
continuity contrast under the legacy representation, not as a baseline.

`R0` and `CO` are **exploratory**: legacy representation, historical run, reported
beside the primary table and never inside it.

## Endpoints, fixed before any number existed

Primary **WER**; safety endpoint **deletion rate**; **substitution** and **insertion**
rates descriptive. Deltas are paired, meeting-clustered bootstrap (31 blocks, 4000
replicates, seed 7), with a secondary 32-block `(city, meeting_id)` split because the
frozen `meeting_id` key merges two different cities' `apr7_2026` meetings.

Interaction is computed as a difference of differences, `(S2-S1) - (RW-R)`,
bootstrapped directly - not inferred from the fact that the count-based deltas
telescope (they always do; the denominator is the same 11,911 tokens).

Domination is reported three ways per contrast: each unit's **signed** share of the
net error change, its **gross** share of total absolute movement, and leave-one-out
sensitivity. The first two answer "does one window carry this"; the third is a
different question and is labelled as such.

`MDE_POINTS` are historical analogues from other arm shapes, not thresholds this
experiment must clear. The observed interval is what is reported.

Hypothesis text stays under `$SC`. Only aggregates go in the repo.

    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/served_config_and_july.py check
    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/served_config_and_july.py smoke --arm S1
    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/served_config_and_july.py decode --arm S1
    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/served_config_and_july.py threadprobe
    SC=~/.cache/oc-public .venv-eval/bin/python notebooks/served_config_and_july.py score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap  # noqa: E402

import decode_ablation as DA  # noqa: E402

EXPERIMENT = "exp-2026-08-17-served-config-and-july-adapter"

CT2_FIXED = "/home/harold/oc-asr-serve/ct2-fixed"
CT2_JULY = "/home/harold/oc-asr-serve/ct2"
CT2_CORR = "/home/harold/oc-correction-only/ct2"

# model.bin sha256[:16]. The first two are the ledger's `artifact-ct2-fixed` and
# `artifact-ct2-july-broken` values; all three were re-verified on disk 2026-08-17.
MODEL_SHA16 = {
    CT2_FIXED: "8a1a3b257d0c1bdb",
    CT2_JULY: "dfffde80906f2cd5",
    CT2_CORR: "4cdbb59b51560cd5",
}

# arm -> (model dir, overrides on CONTROL, role)
ARMS: dict[str, tuple[str, dict, str]] = {
    "R":  (CT2_FIXED, {},
           "control: frozen research config, re-decoded 2026-08-17"),
    "S1": (CT2_FIXED, {"beam_size": 2},
           "served benchmark route: OC_ASR_BEAM=2, word_timestamps=False"),
    "S2": (CT2_FIXED, {"beam_size": 2, "word_timestamps": True},
           "served product route: OC_ASR_BEAM=2, word_timestamps=True"),
    "RW": (CT2_FIXED, {"word_timestamps": True},
           "interaction cell: beam 5 + word timestamps"),
    "J":  (CT2_JULY, {},
           "artifact-ct2-july-broken (label-prefix bug), frozen research config"),
    "R0": (CT2_FIXED, {},
           "cached 2026-08-12 control (legacy fused join) - continuity only"),
    "CO": (CT2_CORR, {},
           "cached artifact-adapter-correction-only (legacy fused join)"),
}

PRIMARY_ARMS = ("R", "S1", "S2", "RW", "J")
# Arms decoded before this experiment: legacy representation, exploratory only.
LEGACY: dict[str, Path] = {
    "R0": Path.home() / ".cache/oc-public/decode-ablation/eval-A.json",
    "CO": Path.home() / ".cache/oc-public/correction-only/decode.json",
}

CONTRASTS = [
    ("S1", "R", "beam 5 -> beam 2, word timestamps off in both", "primary"),
    ("S2", "S1", "word timestamps off -> on, at the served beam 2", "primary"),
    ("S2", "R", "frozen research config -> served product config", "primary"),
    ("RW", "R", "word timestamps off -> on, at the research beam 5", "primary"),
    ("J", "R", "artifact-adapter-fixed -> artifact-adapter-july-broken", "primary"),
    ("R0", "R", "today's control vs the cached 2026-08-12 control (LEGACY join, "
                "same config, same seeds) - decoder rerun instability", "exploratory"),
    ("CO", "R0", "artifact-adapter-correction-only vs the cached control "
                 "(LEGACY join, both historical)", "exploratory"),
]

# Historical analogues from docs/reports/2026-08-16-harness-coverage-mde.md, in WER
# points. Planning estimates for other arm shapes. NOT thresholds for this experiment.
MDE_POINTS = {"decode_ablation_shape": 0.69, "correction_only_adapter_shape": 1.27,
              "absolute_level_half_width": 3.24}
SERVER_MAX_INFER_SEC = 150.0

# The July artifact's status does not depend on any number below. Emitted into the
# results file so it cannot be dropped between the JSON and the prose.
JULY_GUARD = (
    "artifact-ct2-july-broken remains KNOWN_BROKEN regardless of this contrast's "
    "sign. This is a forensic comparison of two fixed historical binaries on one CPU "
    "decode realization of a repeatedly used 39-window agreement-with-OpenCouncil "
    "slice. It is not evidence that the July training targets, adapter or deployment "
    "are usable. A July win would say only that this broken binary scored better "
    "here; a null would say this design did not resolve a difference, not that the "
    "two are equivalent. The contrast does not isolate the causal effect of fixing "
    "the label prefix, because training-seed variation is unreplicated and was "
    "measured at 2.1 WER points."
)


def log(m):
    print(m, flush=True)


def out_dir() -> Path:
    d = DA.sc() / "served-config-2026-08"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dest_for(arm: str) -> Path:
    return LEGACY.get(arm) or (out_dir() / f"eval-{arm}.json")


def config_for(arm: str) -> dict:
    """CONTROL with this arm's overrides - and nothing else changed."""
    over = ARMS[arm][1]
    unknown = set(over) - set(DA.CONTROL)
    if unknown:
        raise SystemExit(f"arm {arm} overrides keys not in CONTROL: {sorted(unknown)}")
    kw = dict(DA.CONTROL, **over)
    differing = {k for k in kw if kw[k] != DA.CONTROL[k]}
    if differing != set(over):
        raise SystemExit(f"arm {arm} differs from CONTROL in {sorted(differing)}, "
                         f"declared {sorted(over)}")
    return kw


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(Path(path) / "model.bin", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def verify_model(arm: str) -> str:
    model = ARMS[arm][0]
    want, got = MODEL_SHA16[model], sha16(model)
    if got != want:
        raise SystemExit(f"arm {arm}: {model}/model.bin is {got}, expected {want}. "
                         f"An output whose producing model is unknown is not evidence.")
    return got


def environment() -> dict:
    import ctranslate2
    import faster_whisper
    return {"python": sys.version.split()[0],
            "faster_whisper": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__,
            "device": DA.DEVICE, "compute_type": DA.COMPUTE,
            "cpu_threads": DA.THREADS}


def code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True
                              ).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------- representations
def tokens_per_segment(segments: list[str]) -> list[str]:
    """The primary hypothesis. Matches production's per-utterance `.strip()` join."""
    return [t for seg in segments for t in ftoks(seg)]


def tokens_legacy(joined: str) -> list[str]:
    """`"".join(segment.text)` as `decode_ablation` wrote it. Fuses across boundaries."""
    return ftoks(joined)


# ------------------------------------------------------------------------ decode
def decode(arm: str, limit: int | None = None) -> Path:
    import ctranslate2
    from faster_whisper import WhisperModel

    if arm in LEGACY:
        raise SystemExit(f"arm {arm} is a cached historical decode; it is not re-run")
    model_dir, over, role = ARMS[arm]
    kw = config_for(arm)
    digest = verify_model(arm)
    dest = dest_for(arm)

    fresh = {"experiment": EXPERIMENT, "arm": arm, "role": role, "model": model_dir,
             "model_sha256_16": digest, "config": kw, "environment": environment(),
             "manifest_frozen_at": DA.manifest()["frozen_at"],
             "code_sha": code_sha(), "windows": {}}
    state = json.loads(dest.read_text()) if dest.exists() else fresh
    for k in ("config", "model", "model_sha256_16"):
        if state.get(k) != fresh[k]:
            raise SystemExit(f"{dest} was written under a different {k}; delete it "
                             f"rather than extending it")
    if state.get("environment") != fresh["environment"]:
        raise SystemExit(f"{dest} was written in a different environment "
                         f"({state.get('environment')} != {fresh['environment']}); "
                         f"delete it rather than extending it")

    rows = DA.rows("eval")[:limit]
    todo = [r for r in rows if r["window_id"] not in state["windows"]]
    log(f"arm {arm} ({role}): {len(todo)} to decode, {len(state['windows'])} done")
    if not todo:
        return dest

    model = WhisperModel(model_dir, device=DA.DEVICE, compute_type=DA.COMPUTE,
                         cpu_threads=DA.THREADS)
    diag = DA.Diagnostics()
    fw_log = logging.getLogger("faster_whisper")
    prev = fw_log.level
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)
    try:
        for i, r in enumerate(todo, 1):
            wav = DA.sc() / "bench_windows" / f"{r['window_id']}.wav"
            if not wav.exists():
                raise SystemExit(f"missing audio for {r['window_id']}")
            diag.reset()
            # Same per-window seed in every arm: the arms differ by the knob or by the
            # weights, not by the draw. Across beam widths this is bookkeeping, not
            # variance reduction - a different beam consumes a different number of draws.
            ctranslate2.set_random_seed(DA.seed_for("A", r["window_id"]))
            t0 = time.time()
            segments, info = model.transcribe(str(wav), **kw)
            segs = list(segments)
            elapsed = time.time() - t0
            resolved = DA.opts_to_dict(info.transcription_options)
            for k, v in over.items():
                if resolved.get(k) != v:
                    raise SystemExit(f"arm {arm}: resolved {k}={resolved.get(k)!r}, "
                                     f"asked {v!r}")
            state["windows"][r["window_id"]] = {
                "segments": [s.text for s in segs],
                "text_legacy_join": "".join(s.text for s in segs).strip(),
                "n_segments": len(segs),
                "decoded_seconds": round(sum(s.end - s.start for s in segs), 2),
                "audio_seconds": round(info.duration, 2),
                "temperatures_used": sorted({s.temperature for s in segs
                                             if s.temperature is not None}),
                "seed": DA.seed_for("A", r["window_id"]),
                "wall_seconds": round(elapsed, 1),
                **diag.snapshot()}
            state.setdefault("resolved_options", resolved)
            dest.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            log(f"  {arm} {i}/{len(todo)} {r['window_id']} {elapsed:.0f}s "
                f"segs={len(segs)}")
    finally:
        fw_log.removeHandler(diag)
        fw_log.setLevel(prev)
    log(f"arm {arm} -> {dest}")
    return dest


# ------------------------------------------------------------------------- probes
def probe_windows() -> list[dict]:
    """One ordinary temperature-0 window and one known fallback window.

    Chosen from the cached control's diagnostics, not by length: a window that never
    left temperature 0 exercises the deterministic path, one that reached a sampling
    temperature exercises the path where reruns can differ.
    """
    cached = json.loads(LEGACY["R0"].read_text())["windows"]
    rows = {r["window_id"]: r for r in DA.rows("eval")}
    plain = [w for w, v in cached.items() if v["temperatures_used"] == [0.0]]
    fell = [w for w, v in cached.items() if max(v["temperatures_used"] or [0]) > 0.0]
    pick = []
    for group in (plain, fell):
        if group:
            pick.append(min(group, key=lambda w: rows[w]["duration_sec"]))
    return [rows[w] for w in pick]


def smoke(arm: str) -> None:
    """Prove the arm resolves as intended and expose rerun instability, cheaply."""
    import ctranslate2
    from faster_whisper import WhisperModel

    kw = config_for(arm)
    log(f"arm {arm}: {ARMS[arm][2]}")
    log(f"  model   {ARMS[arm][0]}  sha16 {verify_model(arm)}")
    log(f"  changes {ARMS[arm][1]}")
    rows = probe_windows()
    for r in rows:
        wid = r["window_id"]
        wav = DA.sc() / "bench_windows" / f"{wid}.wav"
        ref = ftoks(DA.reference_text(wid))
        runs = []
        for rep in range(2):
            # A fresh model instance per repetition: the question is whether the same
            # config decodes the same audio the same way twice, not whether one
            # warmed-up instance is self-consistent.
            model = WhisperModel(ARMS[arm][0], device=DA.DEVICE,
                                 compute_type=DA.COMPUTE, cpu_threads=DA.THREADS)
            ctranslate2.set_random_seed(DA.seed_for("A", wid))
            t0 = time.time()
            segs, info = model.transcribe(str(wav), **kw)
            segs = list(segs)
            resolved = DA.opts_to_dict(info.transcription_options)
            # `TranscriptionOptions` does not carry every request key (language,
            # task, vad_filter live elsewhere; temperature is `temperatures`), so
            # this checks the ones it does expose.
            unintended = {k: (resolved.get(k), v) for k, v in DA.CONTROL.items()
                          if k not in ARMS[arm][1] and k in resolved
                          and resolved.get(k) != v}
            assert not unintended, unintended
            for k, v in ARMS[arm][1].items():
                assert resolved.get(k) == v, (k, resolved.get(k), v)
            hyp = tokens_per_segment([s.text for s in segs])
            s, d, i = sdi(ref, hyp)
            runs.append((hyp, (s, d, i), round(time.time() - t0, 1),
                         sorted({x.temperature for x in segs
                                 if x.temperature is not None})))
            del model
        (h1, c1, t1, temp), (h2, c2, t2, _) = runs
        log(f"  {wid} temps={temp} {t1}s/{t2}s  "
            f"WER {(sum(c1))/len(ref):.4f} vs {(sum(c2))/len(ref):.4f}  "
            f"S{c1[0]} D{c1[1]} I{c1[2]}  rerun_identical={h1 == h2}")


def threadprobe(arm: str = "S2", threads: int = 8) -> None:
    """Bound the one production gap this design does not close: 8 threads vs 16.

    Not a result. It answers only whether the thread count the deployment uses moves
    the text at all on the two probe windows, and what a production-thread decode
    costs in wall-clock against the server's 150 s guard.
    """
    import ctranslate2
    from faster_whisper import WhisperModel

    kw = config_for(arm)
    verify_model(arm)
    rows = probe_windows()
    res = {"experiment": EXPERIMENT, "arm": arm, "config": kw,
           "threads": [DA.THREADS, threads], "windows": {}}
    for r in rows:
        wid = r["window_id"]
        wav = DA.sc() / "bench_windows" / f"{wid}.wav"
        ref = ftoks(DA.reference_text(wid))
        row = {}
        for n in (DA.THREADS, threads):
            model = WhisperModel(ARMS[arm][0], device=DA.DEVICE,
                                 compute_type=DA.COMPUTE, cpu_threads=n)
            ctranslate2.set_random_seed(DA.seed_for("A", wid))
            t0 = time.time()
            segs = list(model.transcribe(str(wav), **kw)[0])
            hyp = tokens_per_segment([s.text for s in segs])
            row[str(n)] = {"wall_seconds": round(time.time() - t0, 1),
                           "sdi": list(sdi(ref, hyp)), "n_tokens": len(hyp),
                           "audio_seconds": r["duration_sec"]}
            row.setdefault("_hyp", {})[str(n)] = hyp
            del model
        row["identical_text"] = row["_hyp"][str(DA.THREADS)] == row["_hyp"][str(threads)]
        row.pop("_hyp")
        res["windows"][wid] = row
        log(f"  {wid} {json.dumps({k: v for k, v in row.items()})}")
    dest = out_dir() / "threadprobe.json"
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(f"-> {dest}")


def check() -> None:
    """Everything that can fail before a multi-hour decode starts."""
    rows = DA.rows("eval")
    man = DA.manifest()
    sealed = {w["window_id"] for w in man["holdout_windows"]}
    wids = {r["window_id"] for r in rows}
    log(f"windows {len(rows)}  meetings {len({r['meeting_id'] for r in rows})}  "
        f"ref tokens {sum(r['ref_tokens'] for r in rows)}  frozen {man['frozen_at']}")
    assert len(rows) == 39 and sum(r["ref_tokens"] for r in rows) == 11911
    assert not (wids & sealed), f"sealed windows in the eval set: {wids & sealed}"
    log(f"sealed holdout windows excluded: {len(sealed)}  intersection: none")
    missing = [r["window_id"] for r in rows
               if not (DA.sc() / "bench_windows" / f"{r['window_id']}.wav").exists()]
    log(f"missing audio: {missing or 'none'}")
    log(f"environment: {json.dumps(environment())}")
    for arm in ARMS:
        kw = config_for(arm)
        d = dest_for(arm)
        n = len(json.loads(d.read_text())["windows"]) if d.exists() else 0
        tag = "LEGACY" if arm in LEGACY else ("primary" if arm in PRIMARY_ARMS
                                              else "exploratory")
        log(f"  {arm:3s} {tag:11s} {ARMS[arm][0]:36s} sha16 {verify_model(arm)} "
            f"decoded {n:2d}/39  changes={ARMS[arm][1]}")
        assert {k: v for k, v in kw.items() if k not in ARMS[arm][1]} == \
               {k: v for k, v in DA.CONTROL.items() if k not in ARMS[arm][1]}
    log(f"probe windows: {[r['window_id'] for r in probe_windows()]}")


# ------------------------------------------------------------------------- score
def counts(hyps: dict[str, list[str]]) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    for r in DA.rows("eval"):
        wid = r["window_id"]
        if wid not in hyps:
            raise SystemExit(f"incomplete: {wid} missing. No complete-case subsets.")
        ref = ftoks(DA.reference_text(wid))
        out[wid] = (*sdi(ref, hyps[wid]), len(ref))
    return out


def arm_hyps(arm: str) -> dict[str, list[str]] | None:
    """Primary representation for fresh arms, legacy join for the two cached ones."""
    d = dest_for(arm)
    if not d.exists():
        return None
    st = json.loads(d.read_text())
    if len(st["windows"]) < 39:
        if arm in PRIMARY_ARMS:
            raise SystemExit(f"primary arm {arm} is incomplete "
                             f"({len(st['windows'])}/39). No partial report.")
        return None
    if arm in LEGACY:
        return {k: tokens_legacy(v["text"]) for k, v in st["windows"].items()}
    return {k: tokens_per_segment(v["segments"]) for k, v in st["windows"].items()}


def wallclock(arm: str) -> dict | None:
    d = dest_for(arm)
    if not d.exists():
        return None
    ws = sorted(w["wall_seconds"] for w in json.loads(d.read_text())["windows"].values()
                if w.get("wall_seconds") is not None)
    if not ws:
        return None
    return {"n": len(ws), "total_seconds": round(sum(ws), 1),
            "median_seconds": ws[len(ws) // 2], "max_seconds": ws[-1],
            "windows_over_server_guard_150s": sum(1 for w in ws
                                                  if w > SERVER_MAX_INFER_SEC)}


def _groups(units: list[str]) -> dict[str, list[int]]:
    g: dict[str, list[int]] = {}
    for i, u in enumerate(units):
        g.setdefault(u, []).append(i)
    return g


def paired_ci(a, b, wids, units, pick) -> dict:
    ca = [(pick(a[w]), a[w][3]) for w in wids]
    cb = [(pick(b[w]), b[w][3]) for w in wids]
    return cluster_bootstrap(ca, cb, units, n_boot=DA.BOOTSTRAP_REPLICATES,
                             seed=DA.BOOTSTRAP_SEED)


def level_ci(a, wids, units, pick) -> dict:
    """Absolute level with a meeting-clustered interval (not a comparison)."""
    import numpy as np
    arr = np.array([(pick(a[w]), a[w][3]) for w in wids], dtype=float)
    g = _groups(units)
    keys = sorted(g)
    rng = np.random.default_rng(DA.BOOTSTRAP_SEED)
    vals = np.empty(DA.BOOTSTRAP_REPLICATES)
    for i in range(DA.BOOTSTRAP_REPLICATES):
        idx = np.concatenate([g[keys[k]] for k in
                              rng.integers(0, len(keys), len(keys))])
        vals[i] = arr[idx, 0].sum() / arr[idx, 1].sum()
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"value": arr[:, 0].sum() / arr[:, 1].sum(),
            "ci95": [float(lo), float(hi)]}


def dod_ci(a1, b1, a2, b2, wids, units, pick) -> dict:
    """Difference of differences, (a1-b1) - (a2-b2), on the same resampled meetings."""
    import numpy as np
    m = np.array([[pick(a1[w]), pick(b1[w]), pick(a2[w]), pick(b2[w]), a1[w][3]]
                  for w in wids], dtype=float)
    g = _groups(units)
    keys = sorted(g)
    rng = np.random.default_rng(DA.BOOTSTRAP_SEED)
    vals = np.empty(DA.BOOTSTRAP_REPLICATES)
    for i in range(DA.BOOTSTRAP_REPLICATES):
        idx = np.concatenate([g[keys[k]] for k in
                              rng.integers(0, len(keys), len(keys))])
        den = m[idx, 4].sum()
        vals[i] = ((m[idx, 0].sum() - m[idx, 1].sum())
                   - (m[idx, 2].sum() - m[idx, 3].sum())) / den
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    den = m[:, 4].sum()
    point = ((m[:, 0].sum() - m[:, 1].sum()) - (m[:, 2].sum() - m[:, 3].sum())) / den
    return {"delta": float(point), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def domination(a, b, wids, units, pick) -> dict:
    """Who carries this delta - signed share, gross share, and LOO sensitivity.

    The first two are direct contributions. The third answers a different question
    (how much does the estimate move if a unit is dropped) and is not a share.
    """
    per_unit: dict[str, float] = {}
    for w, u in zip(wids, units):
        per_unit[u] = per_unit.get(u, 0.0) + (pick(a[w]) - pick(b[w]))
    net = sum(per_unit.values())
    gross = sum(abs(v) for v in per_unit.values())
    den = sum(a[w][3] for w in wids)
    full = net / den

    def loo(unit):
        keep = [w for w, u in zip(wids, units) if u != unit]
        n = sum(a[w][3] for w in keep)
        return (sum(pick(a[w]) - pick(b[w]) for w in keep) / n) if n else float("nan")

    top_signed = max(per_unit.items(), key=lambda kv: abs(kv[1]))
    shifts = {u: abs(loo(u) - full) for u in per_unit}
    top_shift = max(shifts.items(), key=lambda kv: kv[1]) if shifts else (None, 0.0)
    flips = [u for u in per_unit if (full < 0) != (loo(u) < 0)]
    return {"delta": full, "net_error_change": net, "gross_error_movement": gross,
            "top_unit": top_signed[0], "top_unit_error_change": top_signed[1],
            "top_unit_signed_share": (top_signed[1] / net) if net else None,
            "top_unit_gross_share": (abs(top_signed[1]) / gross) if gross else None,
            "max_loo_shift_unit": top_shift[0], "max_loo_shift": top_shift[1],
            "sign_reversed_by": flips}


def churn(a, b, wids, pick) -> dict:
    """Cancellation check: a small net move can hide large per-window movement."""
    d = [pick(a[w]) - pick(b[w]) for w in wids]
    return {"net": sum(d), "abs_sum": sum(abs(x) for x in d),
            "windows_up": sum(1 for x in d if x > 0),
            "windows_down": sum(1 for x in d if x < 0),
            "windows_unchanged": sum(1 for x in d if x == 0)}


PICKS = {"wer": lambda t: t[0] + t[1] + t[2], "sub_rate": lambda t: t[0],
         "del_rate": lambda t: t[1], "ins_rate": lambda t: t[2]}


def score() -> None:
    rows = DA.rows("eval")
    wids = [r["window_id"] for r in rows]
    meetings = [r["meeting_id"] for r in rows]                  # the frozen 31 blocks
    citymeet = [f"{r['city']}/{r['meeting_id']}" for r in rows]  # 32-block sensitivity

    scores, hyps = {}, {}
    for arm in ARMS:
        h = arm_hyps(arm)
        if h is None:
            log(f"arm {arm}: not decoded - skipped")
            continue
        hyps[arm] = h
        scores[arm] = counts(h)
    for arm in PRIMARY_ARMS:
        if arm not in scores:
            log(f"WARNING primary arm {arm} missing; contrasts using it are omitted")

    res = {"experiment": EXPERIMENT,
           "n_windows": len(wids), "n_meetings": len(set(meetings)),
           "n_city_meetings": len(set(citymeet)),
           "ref_tokens": sum(r["ref_tokens"] for r in rows),
           "environment": environment(),
           "code_sha": code_sha(),
           "representation": "primary: per-segment tokenization (matches the server's "
                             "per-utterance strip+join). LEGACY arms use the fused "
                             "\"\".join written by decode_ablation and are exploratory.",
           "production_gaps": [
               "OC_ASR_CPU_THREADS=8; held at 16 here (see threadprobe.json)",
               "OC_ASR_MAX_INFER_SEC=150 streaming guard not emulated",
               "no request queueing, so the server's pre-lock timer is not modelled"],
           "bootstrap": {"replicates": DA.BOOTSTRAP_REPLICATES,
                         "seed": DA.BOOTSTRAP_SEED, "blocks": "meeting_id",
                         "secondary_blocks": "city/meeting_id"},
           "endpoints": {"primary": "wer", "safety": "del_rate",
                         "descriptive": ["sub_rate", "ins_rate"]},
           "mde_points_historical_analogues": MDE_POINTS,
           "july_guard": JULY_GUARD,
           "arms": {}, "contrasts": [], "interaction": {}}

    for arm in scores:
        res["arms"][arm] = {
            "role": ARMS[arm][2], "model": ARMS[arm][0],
            "model_sha256_16": MODEL_SHA16[ARMS[arm][0]],
            "config_change": ARMS[arm][1],
            "representation": "legacy" if arm in LEGACY else "per_segment",
            "class": "primary" if arm in PRIMARY_ARMS else "exploratory",
            "totals": DA.rates(scores[arm]),
            "level_ci95": {m: level_ci(scores[arm], wids, meetings, p)
                           for m, p in PICKS.items()},
            "wallclock": wallclock(arm)}

    for arm, base, what, cls in CONTRASTS:
        if arm not in scores or base not in scores:
            continue
        row = {"arm": arm, "baseline": base, "measures": what, "class": cls,
               "metrics": {}, "domination_window": {}, "domination_meeting": {},
               "churn": {}}
        for metric, pick in PICKS.items():
            ci = paired_ci(scores[arm], scores[base], wids, meetings, pick)
            ci2 = paired_ci(scores[arm], scores[base], wids, citymeet, pick)
            row["metrics"][metric] = {
                "delta": ci["delta"], "ci95": ci["ci95"],
                "excludes_zero": ci["excludes_zero"],
                "ci95_city_meeting_blocks": ci2["ci95"]}
            row["domination_window"][metric] = domination(scores[arm], scores[base],
                                                          wids, wids, pick)
            row["domination_meeting"][metric] = domination(scores[arm], scores[base],
                                                           wids, meetings, pick)
            row["churn"][metric] = churn(scores[arm], scores[base], wids, pick)
        row["head2head"] = {
            "arm_better": sum(1 for w in wids
                              if sum(scores[arm][w][:3]) < sum(scores[base][w][:3])),
            "tie": sum(1 for w in wids
                       if sum(scores[arm][w][:3]) == sum(scores[base][w][:3])),
            "baseline_better": sum(1 for w in wids
                                   if sum(scores[arm][w][:3]) > sum(scores[base][w][:3]))}
        row["text_identical_windows"] = sum(1 for w in wids
                                            if hyps[arm][w] == hyps[base][w])
        row["emitted_tokens"] = {arm: sum(len(hyps[arm][w]) for w in wids),
                                 base: sum(len(hyps[base][w]) for w in wids)}
        res["contrasts"].append(row)

    if all(a in scores for a in ("S2", "S1", "RW", "R")):
        res["interaction"] = {
            "definition": "(S2-S1) - (RW-R): does turning word timestamps on cost "
                          "the same at beam 2 as at beam 5?",
            "metrics": {m: dod_ci(scores["S2"], scores["S1"], scores["RW"],
                                  scores["R"], wids, meetings, p)
                        for m, p in PICKS.items()}}

    dest = out_dir() / "results.json"
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    log(f"{'arm':4s} {'WER':>8s} {'del':>8s} {'ins':>8s} {'sub':>8s}  class  role")
    for arm, row in res["arms"].items():
        t = row["totals"]
        log(f"{arm:4s} {t['wer']:8.5f} {t['del_rate']:8.5f} {t['ins_rate']:8.5f} "
            f"{t['sub_rate']:8.5f}  {row['class'][:4]}  {row['role']}")
    for row in res["contrasts"]:
        log(f"\n{row['arm']} - {row['baseline']} [{row['class']}]: {row['measures']}")
        for m, v in row["metrics"].items():
            log(f"  {m:9s} {v['delta']:+.5f} [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}]"
                f"{'  excludes zero' if v['excludes_zero'] else ''}")
        log(f"  identical text {row['text_identical_windows']}/39; "
            f"h2h {row['head2head']}; tokens {row['emitted_tokens']}")
        for m in ("wer", "del_rate"):
            d = row["domination_meeting"][m]
            c = row["churn"][m]
            log(f"  {m}: net {c['net']:+d} abs {c['abs_sum']} "
                f"(up {c['windows_up']} down {c['windows_down']} "
                f"same {c['windows_unchanged']}); top meeting {d['top_unit']} "
                f"signed {d['top_unit_signed_share']} gross {d['top_unit_gross_share']}")
    if res["interaction"]:
        log("\ninteraction (S2-S1)-(RW-R)")
        for m, v in res["interaction"]["metrics"].items():
            log(f"  {m:9s} {v['delta']:+.5f} [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}]"
                f"{'  excludes zero' if v['excludes_zero'] else ''}")
    log(f"-> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("smoke")
    p.add_argument("--arm", required=True)
    p = sub.add_parser("decode")
    p.add_argument("--arm", required=True)
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("threadprobe")
    p.add_argument("--arm", default="S2")
    p.add_argument("--threads", type=int, default=8)
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    elif a.cmd == "smoke":
        smoke(a.arm)
    elif a.cmd == "decode":
        for arm in a.arm.split(","):
            decode(arm, a.limit)
    elif a.cmd == "threadprobe":
        threadprobe(a.arm, a.threads)
    else:
        score()


if __name__ == "__main__":
    main()
