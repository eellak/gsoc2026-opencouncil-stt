#!/usr/bin/env python3
"""Arm D of the serving-stack plan: N-best ORACLE SCREEN.

Spec: docs/specs/2026-08-12-serving-stack-plan.md (arm D). Question: when the
temperature-0 beam decode drops words inside confidently-decoded segments (the
mechanism confirmed by the decode ablation), do those words exist in OTHER beam
hypotheses? Gate: if the N-best oracle ceiling is < 0.0040 pooled WER, arm D
dies without further work.

Implementation choice (documented per the plan, decided from the INSTALLED
faster_whisper 1.2.1 in .venv-eval):

faster-whisper's `generate_with_fallback` calls `self.model.generate(...)` and
consumes only `sequences_ids[0]`; on the T=0 beam path it does NOT pass
`num_hypotheses`, so CTranslate2 returns a single hypothesis. Rather than
reimplement feature extraction + prompt building (option (a), exact but
divergence-prone), this module wraps the CT2 model object held by the
WhisperModel instance in a forwarding proxy (option (b)). The proxy intercepts
`generate`, injects `num_hypotheses=8` ONLY on the beam path (identified by
`"beam_size" in kwargs` - the exact discriminator `generate_with_fallback`
itself uses; the sampling path passes `sampling_temperature` instead), records
every hypothesis' token ids and score plus the chunk's no_speech_prob into a
side channel, and returns the CT2 result object unchanged. faster-whisper only
ever reads `sequences_ids[0]`, `scores[0]` and `no_speech_prob`, and CT2 beam
search returns the same top-1 whether it reports 1 or 8 of the beam's
hypotheses - so the normal pipeline output is untouched. That makes control
equivalence testable: `validate` decodes with the proxy off and on and asserts
byte-identical window text.

Oracle unit: the decode CHUNK - one `generate_with_fallback` call, i.e. one
~30s decoder window. That is the unit a real selector could choose at: the 8
full-chunk hypotheses of one beam. (A chunk may split into several output
Segments at timestamp pairs; those splits share one beam and cannot be chosen
independently.) Per hypothesis we keep the CONSUMED portion under the
pipeline's own timestamp rule (`_split_segments_by_timestamps`: an unfinished
trailing segment is dropped and re-decoded in the next chunk), applied to each
hypothesis' own timestamps. Hypothesis 0 therefore reproduces the pipeline
output exactly - asserted per window - and alternatives are not credited with
tail text whose audio the next chunk covers again.

Config: CONTROL + beam_size=8 + temperature=[0.0] (arm D prereg: fixed beam 8,
8 hypotheses, T=0; no sampling fallback). All thresholds stay at control
values. Chunk-level no_speech skips are replicated from the recorded
no_speech_prob/score (cross-checked against faster-whisper's own debug log
count) - a skip is decided per chunk, before any hypothesis is read, so a
skipped chunk is not a choice point for any selector.

Oracle scoring is EXACT, not greedy: minimizing total window edit errors over
per-chunk hypothesis choices is a shortest path through the composition of
per-chunk edit-distance transducers. `oracle_window` runs that DP (state = ref
position at each chunk boundary, O(hyps * hyp_len * ref_len) per chunk) with
backpointers, so the selection is the true minimum for this unit in the
per-chunk ftoks token space. One measured caveat: faster-whisper joins segment
texts with plain concatenation, and a chunk whose text lacks a leading space
MERGES with the previous word (seen in the pipeline's own output:
"ότι"+"Επειδή" -> one frozen token). The reconstruction therefore uses the
pipeline's own join, the REPORTED S/D/I come from the frozen `sdi` on that
reconstructed text (same scoring as the baseline), and the DP-vs-sdi drift
from such boundary merges is recorded per window instead of asserted away.

    SC=~/.cache/oc-public .venv-eval/bin/python scripts/serving_stack/nbest_arm.py validate
    SC=~/.cache/oc-public .venv-eval/bin/python scripts/serving_stack/nbest_arm.py decode
    SC=~/.cache/oc-public .venv-eval/bin/python scripts/serving_stack/nbest_arm.py score

Outputs (never in the repo): $SC/serving-stack/eval-D-nbest.json (resumable
decode state incl. all 8 texts + scores + timestamps per chunk) and
$SC/serving-stack/results-D-eval.json. The aggregate summary goes to
data/reports/finetune-research/nbest-oracle-2026-08-12.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402


def _load_decode_ablation():
    """Import the frozen harness by path (notebooks/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "decode_ablation", ROOT / "notebooks/decode_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DA = _load_decode_ablation()

ARM = "D-nbest"
NUM_HYPS = 8
CEILING_GATE = 0.0040            # pooled WER ceiling below which arm D dies
# A timed decode may share this box: default to 6 threads, override via env.
THREADS = int(os.environ.get("NBEST_THREADS", "6"))
NBEST_CONFIG = dict(DA.CONTROL, beam_size=NUM_HYPS, temperature=[0.0])


def log(m):
    print(m, flush=True)


def out_dir() -> Path:
    d = DA.sc() / "serving-stack"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dest_path(which: str) -> Path:
    return out_dir() / f"{which}-D-nbest.json"


# ----------------------------------------------------------------- capture side
class CaptureSink:
    """Side channel for one window's decode. One entry per CT2 generate call."""

    def __init__(self):
        self.calls: list[dict] = []

    def reset(self):
        self.calls.clear()


class CapturingModel:
    """Forwarding proxy around the CT2 Whisper model held by a WhisperModel.

    Intercepts `generate`; everything else (`encode`, `device`, ...) is
    forwarded untouched. On the beam path it injects `num_hypotheses` and
    records all hypotheses; the returned result object is the original one, so
    the caller's top-1 path is unchanged by construction.
    """

    def __init__(self, inner, sink: CaptureSink, num_hypotheses: int = NUM_HYPS):
        self._inner = inner
        self._sink = sink
        self._n = num_hypotheses

    def generate(self, *args, **kwargs):
        beam_path = "beam_size" in kwargs      # generate_with_fallback's own split
        if beam_path:
            assert kwargs["beam_size"] >= self._n, (
                f"beam_size {kwargs['beam_size']} < num_hypotheses {self._n}")
            kwargs["num_hypotheses"] = self._n
        results = self._inner.generate(*args, **kwargs)
        r = results[0]
        self._sink.calls.append({
            "beam_path": beam_path,
            "sequences_ids": [list(s) for s in r.sequences_ids],
            "scores": [float(x) for x in r.scores],
            "no_speech_prob": float(r.no_speech_prob),
        })
        return results

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ----------------------------------------------- pipeline-rule text extraction
def consumed_segments(tokens: list[int], timestamp_begin: int) -> list[dict]:
    """Replicate `_split_segments_by_timestamps` on one hypothesis' tokens.

    Returns the CONSUMED segments only: with consecutive timestamp pairs and no
    single-timestamp ending, the unfinished tail after the last pair is dropped
    (the pipeline re-decodes that audio in the next chunk). Segment dicts carry
    (start_pos, end_pos) in timestamp-token positions (0.02s units), or None
    for the no-timestamp-pair case.
    """
    single_end = (len(tokens) >= 2
                  and tokens[-2] < timestamp_begin <= tokens[-1])
    consecutive = [i for i in range(1, len(tokens))
                   if tokens[i] >= timestamp_begin
                   and tokens[i - 1] >= timestamp_begin]
    if consecutive:
        slices = list(consecutive)
        if single_end:
            slices.append(len(tokens))
        out, last = [], 0
        for cur in slices:
            sliced = tokens[last:cur]
            out.append({"start_pos": sliced[0] - timestamp_begin,
                        "end_pos": sliced[-1] - timestamp_begin,
                        "tokens": sliced})
            last = cur
        return out
    return [{"start_pos": None, "end_pos": None, "tokens": list(tokens)}]


def used_text(tokens: list[int], tokenizer) -> tuple[str, float | None]:
    """(consumed text, last consumed timestamp in seconds) for one hypothesis.

    Mirrors the yield loop of `generate_segments`: a segment whose decoded text
    is empty, or whose start and end timestamps coincide, contributes nothing.
    """
    parts, end = [], None
    for seg in consumed_segments(tokens, tokenizer.timestamp_begin):
        text = tokenizer.decode(seg["tokens"])
        if seg["start_pos"] is not None and seg["start_pos"] == seg["end_pos"]:
            continue
        if not text.strip():
            continue
        parts.append(text)
        if seg["end_pos"] is not None:
            end = seg["end_pos"] * 0.02
    return "".join(parts), end


def predicts_skip(no_speech_prob: float, score0: float, seq_len: int,
                  cfg: dict = NBEST_CONFIG, length_penalty: float = 1.0) -> bool:
    """Replicate the chunk-skip decision of `generate_segments`.

    should_skip = no_speech_prob > no_speech_threshold, overridden to False
    when avg_logprob > log_prob_threshold. avg_logprob is recovered from the
    top-1 score exactly as `generate_with_fallback` does.
    """
    nst = cfg.get("no_speech_threshold")
    if nst is None or no_speech_prob <= nst:
        return False
    lpt = cfg.get("log_prob_threshold")
    if lpt is not None:
        cum = score0 * (seq_len ** length_penalty)
        avg = cum / (seq_len + 1)
        if avg > lpt:
            return False
    return True


class ProcessingDiag(logging.Handler):
    """Capture per-chunk time offsets from 'Processing segment at <ts>'."""

    RE = re.compile(r"Processing segment at ([\d:.]+)")

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.offsets: list[float] = []

    def emit(self, record):
        if (m := self.RE.search(record.getMessage())):
            parts = [float(x) for x in m.group(1).split(":")]
            sec = 0.0
            for p in parts:
                sec = sec * 60 + p
            self.offsets.append(sec)


# --------------------------------------------------------------------- oracle
def transduce(v_in: list[int], hyp: list[str], ref: list[str]
              ) -> tuple[list[int], list[int]]:
    """One chunk-hypothesis step of the exact oracle DP.

    v_in[p] = best cost with the ref consumed up to position p before this
    chunk. Returns (v_out, entry) where v_out[p] is the best cost after this
    chunk consumed ref[a:p] for the best a (unit-cost edit distance of `hyp`
    against that span), and entry[p] is that a (backpointer for the window
    backtrack). Standard edit-distance transducer composition.
    """
    n = len(ref)
    prev = list(v_in)
    prev_e = list(range(n + 1))
    for p in range(1, n + 1):                       # ref deletions, row 0
        if prev[p - 1] + 1 < prev[p]:
            prev[p] = prev[p - 1] + 1
            prev_e[p] = prev_e[p - 1]
    for i in range(1, len(hyp) + 1):
        h = hyp[i - 1]
        cur = [prev[0] + 1]                         # hyp token inserted
        cur_e = [prev_e[0]]
        for p in range(1, n + 1):
            best = prev[p - 1] + (h != ref[p - 1])  # match / substitution
            be = prev_e[p - 1]
            if prev[p] + 1 < best:                  # insertion of hyp token
                best, be = prev[p] + 1, prev_e[p]
            if cur[p - 1] + 1 < best:               # deletion of ref token
                best, be = cur[p - 1] + 1, cur_e[p - 1]
            cur.append(best)
            cur_e.append(be)
        prev, prev_e = cur, cur_e
    return prev, prev_e


def oracle_window(chunk_hyps: list[list[list[str]]], ref: list[str]
                  ) -> tuple[int, list[int]]:
    """(minimum total edit errors, chosen hypothesis index per chunk).

    chunk_hyps: per non-skipped chunk, the candidate hypotheses as ftoks token
    lists, index 0 being the pipeline's own top-1. Exact joint minimization
    over choices via DP on the ref position at each chunk boundary.
    """
    n = len(ref)
    v = list(range(n + 1))          # ref tokens before the first chunk: deleted
    choices: list[list[int]] = []
    entries: list[list[int]] = []
    for hyps in chunk_hyps:
        outs = [transduce(v, h, ref) for h in hyps]
        v_new, choice_p, entry_p = [], [], []
        for p in range(n + 1):
            k = min(range(len(hyps)), key=lambda k: outs[k][0][p])
            v_new.append(outs[k][0][p])
            choice_p.append(k)
            entry_p.append(outs[k][1][p])
        v = v_new
        choices.append(choice_p)
        entries.append(entry_p)
    total = v[n]
    chosen, p = [], n
    for j in range(len(chunk_hyps) - 1, -1, -1):
        chosen.append(choices[j][p])
        p = entries[j][p]
    chosen.reverse()
    return total, chosen


# --------------------------------------------------------------------- decode
def pending(state: dict, all_rows: list[dict], limit: int | None) -> list[dict]:
    """Windows still to decode; factored out so resumability is testable."""
    return [r for r in all_rows[:limit] if r["window_id"] not in state["windows"]]


def make_tokenizer(model):
    from faster_whisper.tokenizer import Tokenizer
    return Tokenizer(model.hf_tokenizer, True, task=NBEST_CONFIG["task"],
                     language=NBEST_CONFIG["language"])


def decode_window(model, tokenizer, sink: CaptureSink, diag, proc, wav: Path,
                  seed: int) -> dict:
    """Decode one window with capture on; return the state record."""
    import ctranslate2

    sink.reset()
    diag.reset()
    proc.offsets.clear()
    ctranslate2.set_random_seed(seed)
    t0 = time.time()
    segments, info = model.transcribe(str(wav), **NBEST_CONFIG)
    segs = list(segments)
    elapsed = time.time() - t0
    text = "".join(s.text for s in segs).strip()

    resolved = DA.opts_to_dict(info.transcription_options)
    assert resolved.get("beam_size") == NUM_HYPS, resolved.get("beam_size")

    calls = sink.calls
    assert all(c["beam_path"] for c in calls), \
        "sampling fallback fired under temperature=[0.0]"
    assert len(proc.offsets) == len(calls), \
        f"{len(proc.offsets)} chunks logged vs {len(calls)} generate calls"

    chunks, n_skipped = [], 0
    for c, t_off in zip(calls, proc.offsets):
        seqs, scores = c["sequences_ids"], c["scores"]
        assert len(seqs) == NUM_HYPS, f"{len(seqs)} hypotheses, wanted {NUM_HYPS}"
        assert all(math.isfinite(s) for s in scores), scores
        skipped = predicts_skip(c["no_speech_prob"], scores[0], len(seqs[0]))
        n_skipped += skipped
        hyps = []
        for toks, score in zip(seqs, scores):
            htext, hend = used_text(toks, tokenizer)
            hyps.append({"text": htext, "score": score,
                         "end": round(hend, 2) if hend is not None else None})
        chunks.append({"t0": round(t_off, 2),
                       "no_speech_prob": round(c["no_speech_prob"], 4),
                       "skipped": bool(skipped),
                       "n_distinct": len({h["text"] for h in hyps}),
                       "hyps": hyps})

    assert n_skipped == len(diag.no_speech_skips), \
        (f"replicated skip rule disagrees with faster-whisper: "
         f"{n_skipped} vs {len(diag.no_speech_skips)}")
    recon = "".join(ch["hyps"][0]["text"] for ch in chunks
                    if not ch["skipped"]).strip()
    assert recon == text, \
        "hypothesis-0 reconstruction differs from the pipeline output"

    return {
        "text": text,
        "n_segments": len(segs),
        "n_chunks": len(chunks),
        "n_chunks_skipped": n_skipped,
        "audio_seconds": round(info.duration, 2),
        "temperatures_used": sorted({s.temperature for s in segs
                                     if s.temperature is not None}),
        "seed": seed,
        "wall_seconds": round(elapsed, 1),
        "chunks": chunks,
        **diag.snapshot(),
    }


def build_model():
    from faster_whisper import WhisperModel
    return WhisperModel(DA.MODEL_DIR, device=DA.DEVICE, compute_type=DA.COMPUTE,
                        cpu_threads=THREADS)


def install_capture(model, sink: CaptureSink):
    inner = model.model
    model.model = CapturingModel(inner, sink, NUM_HYPS)
    return inner


def decode(which: str, limit: int | None) -> Path:
    dest = dest_path(which)
    state = (json.loads(dest.read_text()) if dest.exists() else
             {"arm": ARM, "set": which, "config": NBEST_CONFIG,
              "num_hypotheses": NUM_HYPS, "cpu_threads": THREADS,
              "windows": {}})
    todo = pending(state, DA.rows(which), limit)
    log(f"arm {ARM} [{which}]: {len(todo)} windows to decode "
        f"({len(state['windows'])} already done, threads={THREADS})")
    if not todo:
        return dest

    model = build_model()
    tokenizer = make_tokenizer(model)
    sink = CaptureSink()
    install_capture(model, sink)

    diag = DA.Diagnostics()
    proc = ProcessingDiag()
    fw_log = logging.getLogger("faster_whisper")
    prev_level = fw_log.level
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)
    fw_log.addHandler(proc)
    try:
        for i, r in enumerate(todo, 1):
            wid = r["window_id"]
            wav = DA.sc() / "bench_windows" / f"{wid}.wav"
            if not wav.exists():
                raise SystemExit(f"missing audio for {wid} - the manifest says "
                                 f"it should be there")
            rec = decode_window(model, tokenizer, sink, diag, proc, wav,
                                DA.seed_for(ARM, wid))
            state["windows"][wid] = rec
            dest.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            log(f"  {ARM} {i}/{len(todo)} {wid} {rec['wall_seconds']:.0f}s "
                f"chunks={rec['n_chunks']} skipped={rec['n_chunks_skipped']} "
                f"distinct={[c['n_distinct'] for c in rec['chunks']]}")
    finally:
        fw_log.removeHandler(diag)
        fw_log.removeHandler(proc)
        fw_log.setLevel(prev_level)
    log(f"arm {ARM} [{which}] -> {dest}")
    return dest


def validate(which: str, n_windows: int = 2) -> None:
    """Capture-off vs capture-on equivalence on the shortest windows.

    Proves: injecting num_hypotheses=8 does not change the top-1 pipeline
    output; every non-skipped chunk carries 8 finite-scored hypotheses; and the
    hypothesis-0 side channel reconstructs the pipeline text exactly (asserted
    inside decode_window).
    """
    import ctranslate2

    picks = sorted(DA.rows(which), key=lambda x: x["duration_sec"])[:n_windows]
    model = build_model()
    tokenizer = make_tokenizer(model)
    inner = model.model

    diag = DA.Diagnostics()
    proc = ProcessingDiag()
    fw_log = logging.getLogger("faster_whisper")
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)
    fw_log.addHandler(proc)
    sink = CaptureSink()
    try:
        for r in picks:
            wid = r["window_id"]
            wav = DA.sc() / "bench_windows" / f"{wid}.wav"
            seed = DA.seed_for(ARM, wid)

            model.model = inner                       # capture OFF
            ctranslate2.set_random_seed(seed)
            t0 = time.time()
            segs_off = list(model.transcribe(str(wav), **NBEST_CONFIG)[0])
            text_off = "".join(s.text for s in segs_off).strip()
            t_off = time.time() - t0

            model.model = CapturingModel(inner, sink, NUM_HYPS)  # capture ON
            rec = decode_window(model, tokenizer, sink, diag, proc, wav, seed)

            assert rec["text"] == text_off, \
                f"{wid}: top-1 changed when num_hypotheses was injected"
            log(f"  {wid}: top-1 identical off/on ({t_off:.0f}s off, "
                f"{rec['wall_seconds']:.0f}s on), chunks={rec['n_chunks']} "
                f"skipped={rec['n_chunks_skipped']} "
                f"distinct/chunk={[c['n_distinct'] for c in rec['chunks']]}")
    finally:
        model.model = inner
        fw_log.removeHandler(diag)
        fw_log.removeHandler(proc)
    log("validate: PASS")


# ---------------------------------------------------------------------- score
def window_oracle(rec: dict, ref: list[str], nbest: int
                  ) -> tuple[tuple[int, int, int, int], str, int]:
    """((S, D, I, n_ref), reconstructed oracle text, dp_total).

    Selection is the exact DP over the first `nbest` hypotheses per chunk; the
    reported S/D/I come from the frozen `sdi` on the reconstruction, which
    joins chunk texts exactly as the pipeline joins segment texts (plain
    concatenation - a missing leading space merges words across the boundary,
    and the baseline was scored with those merges too). dp_total can differ
    from the sdi total by O(1) per merged boundary; the caller records it.
    """
    live = [c for c in rec["chunks"] if not c["skipped"]]
    chunk_hyps = [[ftoks(h["text"]) for h in c["hyps"][:nbest]] for c in live]
    total, chosen = oracle_window(chunk_hyps, ref)
    text = "".join(c["hyps"][k]["text"]
                   for c, k in zip(live, chosen)).strip()
    s, d, i = sdi(ref, ftoks(text))
    return (s, d, i, len(ref)), text, total


def score(which: str) -> dict:
    p = dest_path(which)
    if not p.exists():
        raise SystemExit(f"arm {ARM} [{which}] not decoded: {p} missing")
    state = json.loads(p.read_text())
    all_rows = DA.rows(which)
    wids = [r["window_id"] for r in all_rows]
    for wid in wids:
        if wid not in state["windows"]:
            raise SystemExit(f"arm {ARM} [{which}] is incomplete: {wid} missing. "
                             f"No complete-case subsets.")

    base, oracle8, oracle4 = {}, {}, {}
    per_window = {}
    merged_boundary_windows = []
    for r in all_rows:
        wid = r["window_id"]
        rec = state["windows"][wid]
        ref = ftoks(DA.reference_text(wid))
        # the identity choice must reproduce the pipeline output exactly
        (_, _, _, _), text1, _ = window_oracle(rec, ref, 1)
        assert text1 == rec["text"], \
            f"{wid}: 1-best reconstruction differs from the pipeline output"
        # boundary-merge diagnostic: per-chunk token space vs frozen whole-text
        live = [c for c in rec["chunks"] if not c["skipped"]]
        cat = [t for c in live for t in ftoks(c["hyps"][0]["text"])]
        if cat != ftoks(rec["text"]):
            merged_boundary_windows.append(wid)
        base[wid] = (*sdi(ref, ftoks(rec["text"])), len(ref))
        oracle8[wid], _, dp8 = window_oracle(rec, ref, 8)
        oracle4[wid], _, dp4 = window_oracle(rec, ref, 4)
        per_window[wid] = {
            "n_ref": len(ref),
            "base_sdi": base[wid][:3],
            "oracle8_sdi": oracle8[wid][:3],
            "oracle4_sdi": oracle4[wid][:3],
            "dp_total_8": dp8,
            "dp_vs_sdi_drift_8": sum(oracle8[wid][:3]) - dp8,
            "errors_recovered_8": sum(base[wid][:3]) - sum(oracle8[wid][:3]),
            "deletions_recovered_8": base[wid][1] - oracle8[wid][1],
        }

    t_base, t_o8, t_o4 = DA.rates(base), DA.rates(oracle8), DA.rates(oracle4)
    ceiling8 = t_base["wer"] - t_o8["wer"]
    ceiling4 = t_base["wer"] - t_o4["wer"]
    err_rec = sum(v["errors_recovered_8"] for v in per_window.values())
    del_rec = sum(v["deletions_recovered_8"] for v in per_window.values())
    sub_rec = t_base["sub"] - t_o8["sub"]
    ins_rec = t_base["ins"] - t_o8["ins"]
    drift = sum(v["dp_vs_sdi_drift_8"] for v in per_window.values())

    # does one window dominate the ceiling?
    shares = {w: v["errors_recovered_8"] / err_rec if err_rec else 0.0
              for w, v in per_window.items()}
    w_max = max(shares, key=shares.get) if err_rec else None

    # context: this run's top-1 vs the frozen beam-5 control
    ctrl = DA.per_window(which, "A")
    t_ctrl = DA.rates(ctrl)

    n_chunks = sum(st["n_chunks"] for st in state["windows"].values())
    n_skipped = sum(st["n_chunks_skipped"] for st in state["windows"].values())

    return {
        "arm": ARM, "set": which,
        "config": {"beam_size": NUM_HYPS, "num_hypotheses": NUM_HYPS,
                   "temperature": [0.0], "base": "decode-ablation CONTROL",
                   "oracle_unit": "decode chunk (one generate call, consumed "
                                  "portion per the pipeline timestamp rule)",
                   "oracle_method": "exact DP over per-chunk hypothesis choice "
                                    "(edit-distance transducer composition), "
                                    "not greedy"},
        "n_windows": len(wids),
        "n_chunks": n_chunks, "n_chunks_skipped": n_skipped,
        "boundary_merge_windows": merged_boundary_windows,
        "context": {"eval_A_beam5": {k: t_ctrl[k] for k in
                                     ("wer", "sub", "del", "ins", "del_rate",
                                      "ins_rate", "ref_tokens")},
                    "note": "beam8-top1 vs eval-A is context only; the ceiling "
                            "is oracle vs the SAME-RUN beam8 top-1"},
        "beam8_top1": t_base,
        "oracle_8": t_o8,
        "oracle_4": t_o4,
        "ceiling": {
            "delta_wer_8": ceiling8,
            "delta_wer_4": ceiling4,
            "errors_recovered_8": err_rec,
            "recovered_sdi_8": [sub_rec, del_rec, ins_rec],
            "deletions_recovered_8": del_rec,
            "deletion_share_of_ceiling": del_rec / err_rec if err_rec else None,
            "share_in_first_4": ceiling4 / ceiling8 if ceiling8 else None,
            "dp_vs_sdi_drift_total": drift,
            "drift_note": ("frozen-sdi totals exceed the DP totals by this many "
                           "errors, all from the pipeline's boundary-merge join "
                           "artifact; the DP selection is exact in split-token "
                           "space, so the true frozen-space ceiling is >= the "
                           "reported one"),
        },
        "domination": {"max_share_window": w_max,
                       "max_share": shares.get(w_max) if w_max else None,
                       "per_window_shares": {w: round(s, 4)
                                             for w, s in sorted(
                                                 shares.items(),
                                                 key=lambda x: -x[1])[:10]}},
        "per_window": per_window,
        "gate": {"threshold": CEILING_GATE,
                 "ceiling": ceiling8,
                 "passed": bool(ceiling8 >= CEILING_GATE),
                 "note": ("oracle ceiling clears the screen; arm D may proceed "
                          "to a frozen selector"
                          if ceiling8 >= CEILING_GATE else
                          "ceiling below 0.0040 pooled: arm D dies without "
                          "further work, no benchmark slot")},
        "wall_seconds": round(sum(st["wall_seconds"]
                                  for st in state["windows"].values()), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "decode", "score"):
        s = sub.add_parser(name)
        s.add_argument("--set", dest="which", default="eval",
                       choices=("eval",))   # holdout stays sealed for screens
        if name == "decode":
            s.add_argument("--limit", type=int, default=None)
        if name == "validate":
            s.add_argument("--windows", type=int, default=2)
    a = ap.parse_args()

    if a.cmd == "validate":
        validate(a.which, a.windows)
    elif a.cmd == "decode":
        decode(a.which, a.limit)
    else:
        res = score(a.which)
        dest = out_dir() / f"results-D-{a.which}.json"
        dest.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        summary = {k: res[k] for k in
                   ("arm", "set", "config", "n_windows", "n_chunks",
                    "n_chunks_skipped", "boundary_merge_windows",
                    "context", "beam8_top1", "oracle_8",
                    "oracle_4", "ceiling", "domination", "gate",
                    "wall_seconds")}
        rep = ROOT / "data/reports/finetune-research/nbest-oracle-2026-08-12.json"
        rep.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        for name in ("beam8_top1", "oracle_8", "oracle_4"):
            t = res[name]
            log(f"{name:>10}  WER {t['wer']:.4f}  S {t['sub']} D {t['del']} "
                f"I {t['ins']}")
        c = res["ceiling"]
        log(f"ceiling {c['delta_wer_8']:.4f} (4-best {c['delta_wer_4']:.4f}, "
            f"{c['share_in_first_4']:.0%} in first 4)  "
            f"deletion share {c['deletion_share_of_ceiling']:.0%}")
        log(f"gate: {'PASS' if res['gate']['passed'] else 'FAIL'} - "
            f"{res['gate']['note']}")
        log(f"-> {dest}\n-> {rep}")


if __name__ == "__main__":
    main()
