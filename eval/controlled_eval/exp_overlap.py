#!/usr/bin/env python3
"""Is overlapping speech a marker for where our ASR fails? A screen, not a verdict.

`docs/specs/asr-v2-design.md` lists "how much overlapping speech is there really" as an
open question, and `docs/reports/diarization-conditioned-asr-review.md` sketches an
expensive answer (diarization-conditioned Whisper) whose justification rests on that
unmeasured number. This measures what can honestly be measured with what we have.

WHAT THIS CAN AND CANNOT SAY. Codex reviewed the design at high effort and the verdict
was blunt, so the limits are written into the file rather than left to the reader:

  MAY say: in the benchmark windows with local audio, we measured associations between
  pyannote-ESTIMATED overlap and the error rates of seven existing ASR systems. These
  are descriptive risk-marker associations. They do not estimate the effect of true
  overlap, and they do not estimate what an overlap-targeted system would recover.

  MAY NOT say: "overlap causes X% of our errors", "X% of errors are overlap-attributable",
  "a large share justifies DiCoW", "a small share kills it", "consensus works especially
  well on overlap", or "insertions rising proves the reference omits the interjector".

Overlap is not randomly assigned: crosstalk windows are also noisier, faster and busier,
and pyannote is unvalidated on Greek council audio, so it may over-detect overlap exactly
where the audio is bad. Both push the same way, and neither is fixable by reanalysis.
Error concentration is also not addressability. What decides the DiCoW question is a
paired intervention (same target speech, interjector added at a known level, same
reference) plus a pilot of the treatment itself. See the report for that plan.

What this screen is good for: if overlap is not even a marker, the expensive path is
hard to defend and we stop cheaply.

  Q1  Does the error MIX shift with overlap? Specifically, do insertions rise faster than
      substitutions and deletions? If the reference omits the interjector, a recognizer
      that hears it emits words with nothing to align to. This is a weak signal with real
      confounds (hallucination in noise, omitted disfluencies, alignment ties) and is
      reported as a flag for the listening audit, NEVER as a finding.
  Q2  Do error rates rise with estimated overlap, and how concentrated are errors in the
      high-overlap windows relative to their share of reference words?
  Q3  Free, same windows: does the consensus vote's gain vary with overlap?

Every contrast is a meeting-clustered bootstrap, because 240 windows sit in ~140
meetings. Buckets are frozen from the diarization output alone, before any error is
looked at.

Usage:
  HF_TOKEN=hf_... SC=/path python eval/controlled_eval/exp_overlap.py
  ANALYZE_ONLY=1 ...   reuse the cached diarization

Env: HF_TOKEN (phase 1) SC DEVICE (cuda|cpu) N_ITEMS N_BOOT (10000)
"""
from __future__ import annotations

import collections
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
from eval.controlled_eval import bench_data as B                      # noqa: E402
from eval.controlled_eval.scoring import counts, wer                  # noqa: E402

AUDIO = ROOT / "data/asr/audio"
SC = Path(os.environ.get("SC", "/tmp"))
OUT = Path(__file__).with_name("results_overlap.json")
MODEL_ID = "pyannote/speaker-diarization-community-1"
N_ITEMS = int(os.environ.get("N_ITEMS", "0"))
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
TRIO = ("scribe-v2-clean", "soniox", "oc-minipc-finetune")
BASELINE = "scribe-v2-clean"

# Bumped whenever anything that changes a cached feature changes: the sweep, the
# slicing, the model. A cache entry whose fingerprint differs is recomputed.
FEATURE_VERSION = 2


def truthy(name):
    """`ANALYZE_ONLY=0` must not mean yes. Codex caught this in the first draft."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def write_atomic(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text)
    tmp.replace(path)


# ------------------------------------------------------------------ phase 1: diarize
def slice_audio(src: Path, start: float, dur: float, dst: Path) -> float:
    """16 kHz mono wav of one window. Returns the MEASURED duration of the decoded
    slice: near the end of a recording ffmpeg returns less than asked, and using the
    requested duration as a denominator would understate overlap on exactly those."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)], check=True)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(dst)], check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def read_wav(path: Path):
    """16-bit mono PCM into the (waveform, sample_rate) dict pyannote accepts directly.

    pyannote 4 decodes through torchcodec, whose shared library does not load against
    this torch build. Handing it a waveform bypasses the decoder entirely, and ffmpeg
    has already done the resampling in slice_audio.
    """
    import numpy as np
    import torch
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "expected 16-bit mono"
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    x = torch.from_numpy(pcm.astype("float32") / 32768.0).unsqueeze(0)
    return {"waveform": x, "sample_rate": sr}


def overlap_features(diarization, duration: float) -> dict:
    """Speech and overlap seconds from a pyannote Annotation.

    Intervals are merged PER SPEAKER LABEL first. pyannote can emit several tracks for
    one speaker, and counting raw tracks would score a speaker overlapping themselves as
    crosstalk, which is the single most damaging bug this file could have had.
    """
    per_speaker = collections.defaultdict(list)
    for seg, _, speaker in diarization.itertracks(yield_label=True):
        s, e = max(0.0, seg.start), min(duration, seg.end)
        if e > s:
            per_speaker[speaker].append((s, e))

    events = []
    speaker_time = {}
    for speaker, spans in per_speaker.items():
        merged, total = [], 0.0
        for s, e in sorted(spans):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        for s, e in merged:
            total += e - s
            events.append((s, 1))
            events.append((e, -1))
        speaker_time[speaker] = total

    # Ends before starts at a shared timestamp, so two segments that merely touch are
    # not counted as overlapping.
    events.sort()
    speech = overlap = 0.0
    active, prev = 0, 0.0
    for t, d in events:
        if active >= 1:
            speech += t - prev
        if active >= 2:
            overlap += t - prev
        active += d
        prev = t

    assert -1e-6 <= overlap <= speech + 1e-6 <= duration + 1e-6, \
        f"overlap {overlap} speech {speech} duration {duration}"
    return {"duration": duration, "speech_sec": speech, "overlap_sec": overlap,
            "overlap_frac_of_speech": overlap / speech if speech else 0.0,
            "overlap_frac_of_window": overlap / duration if duration else 0.0,
            "n_detected_speakers": len(per_speaker),
            "n_segments": sum(len(v) for v in per_speaker.values()),
            "speaker_time_sec": speaker_time}


def fingerprint(item, versions):
    return {"feature_version": FEATURE_VERSION, "start": item["_start"],
            "requested_dur": item["_dur"], **versions}


def diarize_all(items) -> dict:
    cache = SC / "overlap_features.json"
    blob = json.loads(cache.read_text()) if cache.exists() else {}
    done = blob.get("features", {})

    import torch
    import pyannote.audio
    versions = {"model": MODEL_ID, "pyannote": pyannote.audio.__version__,
                "torch": torch.__version__}
    # A cached entry computed under different code, model or offsets is not reusable.
    stale = [k for k, v in done.items()
             if v.get("_fp", {}).get("feature_version") != FEATURE_VERSION
             or v.get("_fp", {}).get("model") != MODEL_ID]
    for k in stale:
        del done[k]
    if stale:
        log(f"dropped {len(stale)} stale cache entries")

    todo = [it for it in items if it["item_id"] not in done]
    if not todo:
        log(f"all {len(items)} windows already diarized")
        return done

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            f"HF_TOKEN not set. Accept the terms at https://hf.co/{MODEL_ID} and "
            "https://hf.co/pyannote/segmentation-3.0 with the token's account.")
    from pyannote.audio import Pipeline
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading {MODEL_ID} (pyannote {versions['pyannote']}) on {device}")
    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipeline.to(torch.device(device))

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "w.wav"
        for i, it in enumerate(todo, 1):
            measured = slice_audio(it["_audio"], it["_start"], it["_dur"], wav)
            out = pipeline(read_wav(wav))
            ann = getattr(out, "speaker_diarization", out)   # NOT the exclusive one
            f = overlap_features(ann, measured)
            f["_fp"] = fingerprint(it, versions)
            f["measured_dur"] = measured
            done[it["item_id"]] = f
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                log(f"  {i}/{len(todo)} ({el / i:.1f}s/window, "
                    f"eta {(len(todo) - i) * el / i / 60:.0f}min)")
                write_atomic(cache, json.dumps({"features": done}, ensure_ascii=False))
    write_atomic(cache, json.dumps({"features": done}, ensure_ascii=False))
    log(f"features -> {cache}")
    return done


# ------------------------------------------------------------------ phase 2: analyse
def boot_indices(clusters, n_boot, seed=7):
    """Meeting-resampled index sets. Windows in one meeting share a room and a chair."""
    import numpy as np
    groups = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        yield [i for k in pick for i in groups[keys[k]]]


def ci(samples):
    import numpy as np
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def micro(counts_, idx):
    e = sum(counts_[i][0] for i in idx)
    n = sum(counts_[i][1] for i in idx)
    return e / n if n else float("nan")


def freeze_buckets(fracs):
    """Zero-overlap, then tertiles of the positive values.

    Derived from the diarization output only, before any error is inspected, so the
    boundaries cannot be tuned to make a result appear.
    """
    pos = sorted(f for f in fracs if f > 0)
    if len(pos) < 12:
        return [(0.0, 1e-9, "none"), (1e-9, 1.01, "any")]
    q1, q2 = pos[len(pos) // 3], pos[2 * len(pos) // 3]
    return [(0.0, 1e-9, "none"), (1e-9, q1, "low"), (q1, q2, "mid"), (q2, 1.01, "high")]


def bucket_of(f, buckets):
    for lo, hi, name in buckets:
        if lo <= f < hi:
            return name
    return buckets[-1][2]


def main():
    report = B.load_report()
    providers = B.provider_ids(report)
    items = B.common_items(report, providers)
    by_id = {it["itemId"]: it for it in report["items"]}

    kept, dropped = [], []
    for it in items:
        raw = by_id[it["item_id"]]
        p = AUDIO / f"{it['city_id']}__{it['meeting_id']}.mp3"
        if not p.exists():
            dropped.append(it)
            continue
        it["_audio"], it["_start"], it["_dur"] = p, raw["startSec"], raw["durationSec"]
        kept.append(it)
    if N_ITEMS:
        kept = kept[:N_ITEMS]
    log(f"{len(kept)} windows with local audio, {len(dropped)} without")

    feats = diarize_all(kept) if not truthy("ANALYZE_ONLY") else json.loads(
        (SC / "overlap_features.json").read_text()).get("features", {})
    missing = [it["item_id"] for it in kept if it["item_id"] not in feats]
    if missing:
        raise SystemExit(f"{len(missing)} windows have no cached features; rerun phase 1")

    results = {"model": MODEL_ID, "feature_version": FEATURE_VERSION,
               "n_windows": len(kept), "n_dropped_no_audio": len(dropped),
               "n_boot": N_BOOT, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "claim_limits": (
                   "Descriptive association between pyannote-ESTIMATED overlap and the "
                   "error rates of existing systems. Not causal, not an estimate of what "
                   "an overlap-targeted system would recover. Overlap is confounded with "
                   "general acoustic difficulty and the diarizer is unvalidated on Greek."
               )}

    # Are the windows we had to drop different from the ones we kept? If they are, the
    # screen describes a subset rather than the benchmark.
    if dropped:
        results["dropped_vs_kept"] = {
            p: {"wer_kept": wer(counts([i["ref"] for i in kept],
                                       [i["hyp"][p] for i in kept])),
                "wer_dropped": wer(counts([i["ref"] for i in dropped],
                                          [i["hyp"][p] for i in dropped]))}
            for p in providers}
        d = results["dropped_vs_kept"][BASELINE]
        log(f"dropped-window check ({BASELINE}): kept {d['wer_kept']:.4f} vs "
            f"dropped {d['wer_dropped']:.4f}")

    fr = [feats[it["item_id"]]["overlap_frac_of_speech"] for it in kept]
    ov = [feats[it["item_id"]]["overlap_sec"] for it in kept]
    clusters = [it["cluster"] for it in kept]
    buckets = freeze_buckets(fr)
    tags = [bucket_of(f, buckets) for f in fr]
    results["overlap"] = {
        "buckets": [{"lo": lo, "hi": hi, "name": n} for lo, hi, n in buckets],
        "frac_of_speech": {"mean": statistics.fmean(fr),
                           "median": statistics.median(fr), "max": max(fr)},
        "sec_per_window": {"mean": statistics.fmean(ov),
                           "median": statistics.median(ov)},
        "windows_with_zero_detected_overlap": sum(1 for f in fr if f == 0),
        "mean_detected_speakers": statistics.fmean(
            feats[it["item_id"]]["n_detected_speakers"] for it in kept),
        "n_per_bucket": dict(collections.Counter(tags)),
    }
    log(f"estimated overlap: mean {statistics.fmean(fr):.1%} of speech time, "
        f"median {statistics.median(fr):.1%}, "
        f"{results['overlap']['windows_with_zero_detected_overlap']} windows at zero")
    log(f"buckets: {results['overlap']['n_per_bucket']}")

    refs = [it["ref"] for it in kept]
    hi_name, lo_name = buckets[-1][2], buckets[0][2]
    hi_idx = [i for i, t in enumerate(tags) if t == hi_name]
    lo_idx = [i for i, t in enumerate(tags) if t == lo_name]
    boots = list(boot_indices(clusters, N_BOOT))

    # ---- Q2: do error rates rise with estimated overlap?
    log("=== Q2: error rate by overlap bucket (micro-WER, meeting-clustered CI) ===")
    q2 = {}
    for p in providers:
        c = counts(refs, [it["hyp"][p] for it in kept])
        rows = {name: micro(c, [i for i, t in enumerate(tags) if t == name])
                for _, _, name in buckets}
        diffs = []
        for bidx in boots:
            h = [i for i in bidx if tags[i] == hi_name]
            lo = [i for i in bidx if tags[i] == lo_name]
            diffs.append(micro(c, h) - micro(c, lo) if h and lo else float("nan"))
        err_share = sum(c[i][0] for i in hi_idx) / max(1, sum(x[0] for x in c))
        word_share = sum(c[i][1] for i in hi_idx) / max(1, sum(x[1] for x in c))
        q2[p] = {"wer_by_bucket": rows,
                 f"{hi_name}_minus_{lo_name}": rows[hi_name] - rows[lo_name],
                 **ci(diffs),
                 "share_of_errors_in_high": err_share,
                 "share_of_ref_words_in_high": word_share,
                 "concentration_ratio": err_share / word_share if word_share else None}
        log(f"  {p:28s} " + " ".join(f"{n}={rows[n]:.3f}" for _, _, n in buckets)
            + f" | {hi_name}-{lo_name} {q2[p][f'{hi_name}_minus_{lo_name}']:+.4f} "
              f"CI[{q2[p]['ci95'][0]:+.4f},{q2[p]['ci95'][1]:+.4f}]"
              f" | concentration {q2[p]['concentration_ratio']:.2f}")
    results["Q2_error_rate_vs_overlap"] = q2

    # ---- Q1: does the error MIX shift, i.e. insertions faster than the rest?
    # A difference-in-differences on rates per reference word between the high and low
    # overlap buckets, insertions against substitutions. Comparing "insertions are
    # significant, substitutions are not" would not be a test of a difference.
    log("=== Q1: error-type mix (flag for the listening audit, not a finding) ===")
    q1 = {}
    for p in providers:
        det = []
        for it in kept:
            m = by_id[it["item_id"]]["perProvider"][p]["perMetric"][B.METRIC]
            d = m.get("details")
            if not isinstance(d, dict) or not {"sub", "del", "ins"} <= set(d):
                det = None
                break
            if m["denominator"] <= 0:
                det = None
                break
            det.append((d["sub"], d["del"], d["ins"], m["denominator"]))
        if det is None:
            log(f"  {p:28s} skipped: per-item sub/del/ins not available")
            continue

        def rate(k, idx):
            return sum(det[i][k] for i in idx) / max(1, sum(det[i][3] for i in idx))

        did = (rate(2, hi_idx) - rate(2, lo_idx)) - (rate(0, hi_idx) - rate(0, lo_idx))
        samples = []
        for bidx in boots:
            h = [i for i in bidx if tags[i] == hi_name]
            lo = [i for i in bidx if tags[i] == lo_name]
            samples.append(((rate(2, h) - rate(2, lo)) - (rate(0, h) - rate(0, lo)))
                           if h and lo else float("nan"))
        q1[p] = {"ins_rate": {n: rate(2, [i for i, t in enumerate(tags) if t == n])
                              for _, _, n in buckets},
                 "sub_rate": {n: rate(0, [i for i, t in enumerate(tags) if t == n])
                              for _, _, n in buckets},
                 "del_rate": {n: rate(1, [i for i, t in enumerate(tags) if t == n])
                              for _, _, n in buckets},
                 "ins_minus_sub_diff_in_diff": did, **ci(samples)}
        log(f"  {p:28s} ins/word {q1[p]['ins_rate'][lo_name]:.4f}->"
            f"{q1[p]['ins_rate'][hi_name]:.4f}  sub {q1[p]['sub_rate'][lo_name]:.4f}->"
            f"{q1[p]['sub_rate'][hi_name]:.4f} | DiD {did:+.4f} "
            f"CI[{q1[p]['ci95'][0]:+.4f},{q1[p]['ci95'][1]:+.4f}]")
    results["Q1_error_mix"] = {
        "per_provider": q1,
        "interpretation": (
            "A positive DiD means insertions rise faster than substitutions between the "
            "low and high overlap buckets. That is CONSISTENT with references omitting "
            "the interjector, and equally consistent with hallucination in noisy audio "
            "or with transcribers dropping disfluencies. It decides nothing. Its only "
            "use is to decide whether a blinded listening audit of overlap events is "
            "worth booking."),
    }

    # ---- Q3: does the consensus vote's gain vary with overlap?
    trio = [p for p in TRIO if p in providers]
    if len(trio) == 3:
        picks = [B.consensus_pick(it, trio) for it in kept]
        c_a = counts(refs, [it["hyp"][p] for it, p in zip(kept, picks)])
        c_b = counts(refs, [it["hyp"][BASELINE] for it in kept])
        gains = {n: micro(c_a, [i for i, t in enumerate(tags) if t == n])
                    - micro(c_b, [i for i, t in enumerate(tags) if t == n])
                 for _, _, n in buckets}
        samples = []
        for bidx in boots:
            h = [i for i in bidx if tags[i] == hi_name]
            lo = [i for i in bidx if tags[i] == lo_name]
            samples.append(((micro(c_a, h) - micro(c_b, h))
                            - (micro(c_a, lo) - micro(c_b, lo)))
                           if h and lo else float("nan"))
        results["Q3_consensus_gain_by_overlap"] = {
            "baseline": BASELINE, "gain_by_bucket": gains,
            "interaction_high_minus_low": gains[hi_name] - gains[lo_name], **ci(samples),
            "note": "Exploratory subgroup contrast on the exact paired window set."}
        log("=== Q3: consensus gain by overlap bucket ===")
        for _, _, n in buckets:
            log(f"  {n:5s} {gains[n]:+.4f}")
        log(f"  interaction {results['Q3_consensus_gain_by_overlap']['interaction_high_minus_low']:+.4f} "
            f"CI{results['Q3_consensus_gain_by_overlap']['ci95']}")

    write_atomic(OUT, json.dumps(results, ensure_ascii=False, indent=2))
    log(f"-> {OUT}")


if __name__ == "__main__":
    main()
