#!/usr/bin/env python3
"""`exp-2026-08-16-tse-overlap` — target speaker extraction over simultaneous speech.

Preregistered in `docs/specs/2026-08-16-tse-overlap-prereg.md`. Read that first;
this file only implements it. Nothing here may be tuned on the gold set.

Sub-commands are split because WeSep and faster-whisper cannot share one venv:

  build    (.venv-eval)  select items, write clean/mixture/enrollment wavs, emit a manifest
  decode   (.venv-eval)  decode every arm wav with the FROZEN serving decode config
  score    (.venv-eval)  score every arm against the human-verified reference
  build2/decode2/audit2   run the preregistered Stage 2 case-level audit

Separation itself runs from `scripts/tse/run_extract.py` inside
`/home/harold/wesep-build/.venv`, between `build` and `decode`.

Audio, mixtures and extracted tracks live under ~/.cache/oc-public/tse-2026-08/.
Transcript text and audio never enter git.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gold_set_score import (  # noqa: E402
    load, wtoks, sdi_counts, cluster_ci, paired_delta, region_blocks,
)

WORK = Path(os.environ.get("TSE_WORK", Path.home() / ".cache/oc-public/tse-2026-08"))
CLIPS = Path.home() / ".cache/oc-public/gold-set/clips"
SR = 16_000
SEED = 20260816

# ---- frozen by the preregistration, section 4 -------------------------------
MIN_TARGET_SEC = 2.0
MIN_TARGET_TOK = 5
MIN_ENROLL_SEC = 3.0
MAX_ENROLL_SEC = 10.0
SIRS_DB = (0.0, 5.0)          # primary is 0 dB; both frozen before any number
PEAK_CEIL = 10 ** (-1 / 20)   # -1 dBFS, one common attenuation per item
PAD_SEC = 0.5                 # stage 2 overlap-region padding

STAGE1_ARMS = ("CLEAN", "CLEAN_NORM", "MIX", "MIX_NORM",
               "TSE", "TSE_WRONG", "TSE_ABSENT", "TSE_CLEAN")
# arms that do not depend on the mixing level, so they are built once per item
SIR_FREE = ("CLEAN", "CLEAN_NORM", "TSE_CLEAN")
# arms scored against the MASKER's text rather than the target's
MASKER_REF_ARMS = ("TSE_WRONG",)
OUT_PEAK = 0.9          # exactly what wesep's output_norm does; MIX_NORM matches it
DOMINATION_MAX = 0.50   # frozen: no single item may supply this share of the gain
DEL_HARM_MAX = 0.02     # frozen: G4
G3_MAX = 0.05           # frozen: G3
ACTIVE_DB = 40.0        # active-speech level: frames within 40 dB of the loudest


# --------------------------------------------------------------------- audio io
def read_wav(path: Path) -> np.ndarray:
    """Mono float32 at SR. soundfile only — torchaudio.load needs TorchCodec in 2.11."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path} is {sr} Hz, expected {SR}")
    return x[:, 0].astype(np.float32)


def write_wav(path: Path, x: np.ndarray) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), SR)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def active_rms(x: np.ndarray, frame: int = 320, floor_db: float = ACTIVE_DB) -> float:
    """RMS over active speech only: 20 ms frames within `floor_db` of the loudest.

    Whole-block RMS would let a long pause in one block move the mixing gain, so
    that "0 dB SIR" would not mean equal loudness where both people are talking.
    """
    if x.size < frame:
        return rms(x)
    n = (x.size // frame) * frame
    f = np.square(x[:n].astype(np.float64)).reshape(-1, frame).mean(axis=1)
    f = np.maximum(f, 1e-20)
    keep = f >= f.max() * (10 ** (-floor_db / 10.0))
    return float(np.sqrt(f[keep].mean())) if keep.any() else rms(x)


def peak_norm(x: np.ndarray, peak: float = OUT_PEAK) -> np.ndarray:
    """The exact normalisation wesep applies to its own output, so the _NORM arms
    and the TSE arms go through the same stack."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / m * peak).astype(np.float32) if m > 0 else x.astype(np.float32)


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Both are truncated to the common length."""
    n = min(est.size, ref.size)
    if n == 0:
        return float("nan")
    e, r = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    e = e - e.mean()
    r = r - r.mean()
    rr = float(r @ r)
    if rr <= 0:
        return float("nan")
    proj = (float(e @ r) / rr) * r
    noise = e - proj
    dn = float(noise @ noise)
    return 10 * np.log10(float(proj @ proj) / dn) if dn > 0 else float("inf")


def tile_to(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or tile `x` to exactly n samples. Tiling only happens when the masker
    is shorter than the target; it is a level-preserving repeat, not a stretch."""
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    if x.size >= n:
        return x[:n]
    return np.tile(x, int(np.ceil(n / x.size)))[:n]


def mix_at_sir(target: np.ndarray, masker: np.ndarray, sir_db: float, rng=None):
    """Scale the MASKER so the target-to-masker ACTIVE-speech ratio is `sir_db`.

    The target is never rescaled here, and the common attenuation applied
    afterwards is shared by every arm of the item — so "mixed is worse" can
    never mean "the target got quieter". Rule inherited verbatim from
    docs/specs/synthetic-overlap-preregistration.md.
    """
    if masker.size >= target.size and rng is not None:
        off = int(rng.integers(0, masker.size - target.size + 1))
        m, tiled = masker[off:off + target.size], False
    else:
        m, tiled = tile_to(masker, target.size), masker.size < target.size
    rt, rm = active_rms(target), active_rms(m)
    if rm <= 0 or rt <= 0:
        return target.copy(), np.zeros_like(target), tiled
    g = (rt / rm) * (10 ** (-sir_db / 20.0))
    return target.copy(), (m * g).astype(np.float32), tiled


def common_attenuation(arms: dict[str, np.ndarray]) -> float:
    peak = max((float(np.max(np.abs(a))) if a.size else 0.0) for a in arms.values())
    return PEAK_CEIL / peak if peak > PEAK_CEIL else 1.0


# ------------------------------------------------------------------ item choice
def gold_blocks(ans, cid):
    return sorted(ans[cid]["b"]["blocks"], key=lambda b: (b["s"], b["id"]))


def is_clean_block(b) -> bool:
    """Certain text and not participating in overlap — usable as target or enrollment."""
    return not b.get("text_unc") and not b.get("ov_with")


def enrollment_span(blocks, spk: str, exclude_id: str | None) -> list[tuple[float, float]]:
    """That speaker's certain non-overlap blocks in the same cell, in time order,
    capped at MAX_ENROLL_SEC. Returns [] when the supply is below MIN_ENROLL_SEC —
    the caller must count the exclusion, never drop it silently."""
    segs, total = [], 0.0
    for b in blocks:
        if b["spk"] != spk or not is_clean_block(b) or b["id"] == exclude_id:
            continue
        take = min(b["e"] - b["s"], MAX_ENROLL_SEC - total)
        if take <= 0:
            break
        segs.append((b["s"], b["s"] + take))
        total += take
        if total >= MAX_ENROLL_SEC:
            break
    return segs if total >= MIN_ENROLL_SEC else []


def select_stage1(ans, sel, scored, seed: int = SEED):
    """Every eligible (target, masker, wrong-enrollment) triple. Fully determined
    by the preregistered filters plus one seeded RNG draw; no outcome is consulted."""
    rng = np.random.default_rng(seed)
    cand = []
    for cid in scored:
        bl = gold_blocks(ans, cid)
        meet = sel[cid]["meeting_id"]
        for b in bl:
            if not is_clean_block(b):
                continue
            if (b["e"] - b["s"]) < MIN_TARGET_SEC or len(wtoks(b["text"])) < MIN_TARGET_TOK:
                continue
            enr = enrollment_span(bl, b["spk"], b["id"])
            cand.append({"cid": cid, "meeting": meet, "bid": b["id"], "spk": b["spk"],
                         "s": b["s"], "e": b["e"], "text": b["text"],
                         "n_tok": len(wtoks(b["text"])), "enroll": enr,
                         "eligible": bool(enr)})
    pool = [c for c in cand if c["eligible"]]
    items = []
    for c in sorted(pool, key=lambda x: (x["cid"], x["bid"])):
        others = [d for d in pool if d["meeting"] != c["meeting"]]
        if len(others) < 2:
            continue
        i, j = rng.choice(len(others), size=2, replace=False)
        masker, absent = others[int(i)], others[int(j)]
        items.append({
            **c,
            # the masker is present in the mixture, so ITS enrollment is the
            # targetedness control: it asks whether the model selects between the
            # two voices actually there, not how it behaves out of set
            "masker": {k: masker[k] for k in ("cid", "bid", "spk", "s", "e", "meeting")},
            "masker_enroll": masker["enroll"], "masker_tok": masker["n_tok"],
            "absent": {k: absent[k] for k in ("cid", "bid", "spk", "meeting")},
            "absent_enroll": absent["enroll"], "absent_cid": absent["cid"],
        })
    return items, cand


def cut(clip: np.ndarray, s: float, e: float) -> np.ndarray:
    return clip[max(0, int(round(s * SR))):max(0, int(round(e * SR)))]


def concat_segments(clip: np.ndarray, segs) -> np.ndarray:
    return np.concatenate([cut(clip, s, e) for s, e in segs]) if segs else np.zeros(0, np.float32)


# ------------------------------------------------------------------------ build
def build(out: Path = WORK):
    import hashlib
    man, ans, cells, sel = load()
    scored = man["scored_cell_ids"]
    items, cand = select_stage1(ans, sel, scored)
    clip_cache: dict[str, np.ndarray] = {}

    def clip(cid):
        if cid not in clip_cache:
            clip_cache[cid] = read_wav(CLIPS / f"{cid}.wav")
        return clip_cache[cid]

    audio = out / "stage1"
    rng = np.random.default_rng(SEED + 1)
    recs, n_tiled = [], 0
    for k, it in enumerate(items):
        tid = f"i{k:03d}"
        tgt = cut(clip(it["cid"]), it["s"], it["e"])
        msk_full = cut(clip(it["masker"]["cid"]), it["masker"]["s"], it["masker"]["e"])
        enr = concat_segments(clip(it["cid"]), it["enroll"])
        wenr = concat_segments(clip(it["masker"]["cid"]), it["masker_enroll"])
        aenr = concat_segments(clip(it["absent_cid"]), it["absent_enroll"])
        write_wav(audio / f"{tid}.enroll.wav", enr)
        write_wav(audio / f"{tid}.enroll_wrong.wav", wenr)
        write_wav(audio / f"{tid}.enroll_absent.wav", aenr)
        rec = {"item": tid, "cid": it["cid"], "meeting": it["meeting"], "bid": it["bid"],
               "spk": it["spk"], "n_tok": it["n_tok"], "dur": round(it["e"] - it["s"], 3),
               "masker": it["masker"], "absent": it["absent"],
               "enroll_sec": round(enr.size / SR, 3),
               "enroll_wrong_sec": round(wenr.size / SR, 3),
               "enroll_absent_sec": round(aenr.size / SR, 3), "sirs": {}}
        # one common attenuation per item, over every raw arm at every SIR, so no
        # arm is normalised on its own; the _NORM arms then carry exactly the
        # peak normalisation wesep applies to its own output
        mixes = {}
        for sir in SIRS_DB:
            t, m, tiled = mix_at_sir(tgt, msk_full, sir, rng)
            mixes[f"{sir:g}"] = (t + m, m, tiled)
            n_tiled += bool(tiled and sir == SIRS_DB[0])
        att = common_attenuation({"CLEAN": tgt, **{k2: v[0] for k2, v in mixes.items()}})
        write_wav(audio / f"{tid}.CLEAN.wav", tgt * att)
        write_wav(audio / f"{tid}.CLEAN_NORM.wav", peak_norm(tgt))
        # exact sources, kept for SI-SDR; never decoded
        write_wav(audio / f"{tid}.src_target.wav", tgt * att)
        for tag, (mx, m, tiled) in mixes.items():
            write_wav(audio / f"{tid}.MIX.{tag}.wav", mx * att)
            write_wav(audio / f"{tid}.MIX_NORM.{tag}.wav", peak_norm(mx))
            write_wav(audio / f"{tid}.src_masker.{tag}.wav", m * att)
            rec["sirs"][tag] = {"sir_db": float(tag), "tiled": bool(tiled)}
        recs.append(rec)

    manifest = {
        "protocol": "tse-overlap-2026-08-16", "stage": 1, "seed": SEED,
        "answers_sha256": man["answers_sha256"], "cells_sha256": man["cells_sha256"],
        "sirs_db": list(SIRS_DB), "primary_sir": f"{SIRS_DB[0]:g}",
        "arms": list(STAGE1_ARMS),
        "thresholds": {"G3_max": G3_MAX, "G4_del_harm_max": DEL_HARM_MAX,
                       "domination_max": DOMINATION_MAX},
        "n_candidate_blocks": len(cand),
        "n_candidate_eligible": sum(1 for c in cand if c["eligible"]),
        "n_items": len(recs), "n_tiled_items": n_tiled,
        "n_meetings": len({r["meeting"] for r in recs}),
        "tokens": sum(r["n_tok"] for r in recs),
        "items": recs,
    }
    body = json.dumps(manifest, ensure_ascii=False, indent=1)
    manifest["self_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    (out / "manifest_stage1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"stage1: {len(recs)} items / {manifest['n_meetings']} meetings / "
          f"{manifest['tokens']} tokens / {n_tiled} tiled -> {out}")
    print("manifest sha256", manifest["self_sha256"])
    return manifest


# ----------------------------------------------------------------------- decode
def decode(work: Path = WORK):
    """The FROZEN serving decode config, identical for every arm. No tuning."""
    from faster_whisper import WhisperModel
    model = WhisperModel("/home/harold/oc-asr-serve/ct2-fixed", device="cpu",
                         compute_type="int8",
                         cpu_threads=int(os.environ.get("TSE_ASR_THREADS",
                                                        os.cpu_count() or 8)))
    audio = work / "stage1"
    outdir = work / "hyp_stage1"
    outdir.mkdir(parents=True, exist_ok=True)
    ignored = (".enroll.wav", ".enroll_wrong.wav", ".enroll_absent.wav",
               ".src_target.wav", ".src_masker.0.wav", ".src_masker.5.wav")
    wavs = sorted(p for p in audio.glob("*.wav")
                  if not p.name.endswith(ignored))
    for i, p in enumerate(wavs):
        f = outdir / (p.stem + ".json")
        if f.exists():
            continue
        segs, _info = model.transcribe(str(p), language="el", beam_size=5,
                                       word_timestamps=True,
                                       condition_on_previous_text=False)
        words, text = [], []
        for s in segs:
            text.append(s.text.strip())
            for w in (s.words or []):
                words.append({"w": w.word, "s": round(float(w.start), 3)})
        f.write_text(json.dumps({"text": " ".join(text).strip(), "words": words},
                                ensure_ascii=False))
        if i % 20 == 0:
            print(f"decoded {i}/{len(wavs)}", flush=True)
    print(f"decoded {len(wavs)} wavs -> {outdir}")


def decode_stage2(work: Path = WORK):
    """Decode only the current Stage 2 manifest, excluding stale cache files."""
    from faster_whisper import WhisperModel
    model = WhisperModel("/home/harold/oc-asr-serve/ct2-fixed", device="cpu",
                         compute_type="int8",
                         cpu_threads=int(os.environ.get("TSE_ASR_THREADS",
                                                        os.cpu_count() or 8)))
    man = json.loads((work / "manifest_stage2.json").read_text())
    audio = work / "stage2"
    outdir = work / "hyp_stage2"
    outdir.mkdir(parents=True, exist_ok=True)
    wavs = []
    for case in man["cases"]:
        gid = case["case"]
        wavs.extend([audio / f"{gid}.BASELINE.wav",
                     audio / f"{gid}.MIX_NORM.wav"])
        enrolled = [s for s in case["speakers"] if s["enrollable"]]
        wavs.extend(audio / f"{gid}.TSE.{s['spk']}.wav" for s in enrolled)
        if len(enrolled) > 1:
            wavs.extend(audio / f"{gid}.TSE_WRONG.{s['spk']}.wav"
                        for s in enrolled)
    wavs = [p for p in wavs if p.exists()]
    for i, p in enumerate(sorted(set(wavs))):
        f = outdir / (p.stem + ".json")
        if f.exists():
            continue
        segs, _info = model.transcribe(str(p), language="el", beam_size=5,
                                       word_timestamps=True,
                                       condition_on_previous_text=False)
        words, text = [], []
        for s in segs:
            text.append(s.text.strip())
            for w in (s.words or []):
                words.append({"w": w.word, "s": round(float(w.start), 3)})
        f.write_text(json.dumps({"text": " ".join(text).strip(), "words": words},
                                ensure_ascii=False))
        print(f"decoded stage2 {i + 1}/{len(wavs)}", flush=True)
    print(f"decoded {len(wavs)} stage2 wavs -> {outdir}")


# ------------------------------------------------------------------------ score
def arm_files(item: str, sir: str) -> dict[str, str]:
    """Arm -> wav stem. SIR_FREE arms carry no SIR tag; the others do."""
    out = {a: f"{item}.{a}" for a in SIR_FREE}
    for a in STAGE1_ARMS:
        if a not in SIR_FREE:
            out[a] = f"{item}.{a}.{sir}"
    return out


def score(work: Path = WORK, n_boot: int = 10000):
    man = json.loads((work / "manifest_stage1.json").read_text())
    hyp = work / "hyp_stage1"
    audio = work / "stage1"
    _m, ans, _c, _s = load()
    ref_of, masker_ref_of = {}, {}
    for it in man["items"]:
        b = [x for x in ans[it["cid"]]["b"]["blocks"] if x["id"] == it["bid"]][0]
        ref_of[it["item"]] = wtoks(b["text"])
        mb = [x for x in ans[it["masker"]["cid"]]["b"]["blocks"]
              if x["id"] == it["masker"]["bid"]][0]
        masker_ref_of[it["item"]] = wtoks(mb["text"])

    out = {"protocol": man["protocol"], "manifest_sha256": man.get("self_sha256"),
           "n_items": man["n_items"], "n_meetings": man["n_meetings"],
           "tokens": man["tokens"], "n_tiled_items": man["n_tiled_items"],
           "primary_sir": man["primary_sir"], "sirs": {}, "missing": []}

    for sir in [f"{x:g}" for x in man["sirs_db"]]:
        rows = {a: [] for a in STAGE1_ARMS}
        drows = {a: [] for a in STAGE1_ARMS}
        irows = {a: [] for a in STAGE1_ARMS}
        raw = {a: {"S": 0, "D": 0, "I": 0, "N": 0} for a in STAGE1_ARMS}
        wrong_vs_masker, per_item, sisdr = [], {}, {}
        for it in man["items"]:
            files = arm_files(it["item"], sir)
            got = {}
            for arm, stem in files.items():
                f = hyp / f"{stem}.json"
                if not f.exists():
                    out["missing"].append(stem)
                    continue
                got[arm] = wtoks(json.loads(f.read_text())["text"])
            if len(got) != len(STAGE1_ARMS):
                continue
            ref = ref_of[it["item"]]
            per_item[it["item"]] = {}
            for arm, h in got.items():
                c = sdi_counts(ref, h)
                rows[arm].append((it["meeting"], c["err"], c["N"]))
                drows[arm].append((it["meeting"], c["D"], c["N"]))
                irows[arm].append((it["meeting"], c["I"], c["N"]))
                for k in ("S", "D", "I", "N"):
                    raw[arm][k] += c[k]
                per_item[it["item"]][arm] = c
            # the masker-enrolled extraction, scored against the MASKER's own text:
            # a positive control for "did it extract the speaker it was asked for"
            cm = sdi_counts(masker_ref_of[it["item"]], got["TSE_WRONG"])
            wrong_vs_masker.append((it["meeting"], cm["err"], cm["N"]))
        res = {"n_scored": len(per_item), "arms": {}, "paired": {}, "raw_sdi": raw}
        for arm in STAGE1_ARMS:
            if rows[arm]:
                res["arms"][arm] = {"wer": cluster_ci(rows[arm], n_boot=n_boot),
                                    "del_rate": cluster_ci(drows[arm], n_boot=n_boot),
                                    "ins_rate": cluster_ci(irows[arm], n_boot=n_boot)}
        if wrong_vs_masker:
            res["tse_wrong_vs_masker_text"] = cluster_ci(wrong_vs_masker, n_boot=n_boot)
        for a, b in (("TSE", "MIX_NORM"), ("TSE_WRONG", "MIX_NORM"),
                     ("TSE_ABSENT", "MIX_NORM"), ("MIX_NORM", "MIX"),
                     ("CLEAN_NORM", "CLEAN"), ("TSE_CLEAN", "CLEAN_NORM"),
                     ("MIX_NORM", "CLEAN_NORM"), ("TSE", "CLEAN_NORM")):
            if rows[a] and rows[b]:
                res["paired"][f"{a}-{b}"] = paired_delta(rows[a], rows[b], n_boot=n_boot)
        res["si_sdr_db"] = si_sdr_table(audio, man, sir)
        res["domination"] = domination(per_item, "TSE", "MIX_NORM")
        res["gates"] = gates(res, primary=(sir == man["primary_sir"]))
        out["sirs"][sir] = res
    (ROOT / "eval" / "results_tse_overlap.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    return out


def si_sdr_table(audio: Path, man: dict, sir: str) -> dict:
    """SI-SDR of each extracted track against the exact target and masker sources.

    WER can show that preprocessing helped an ASR; only this can show that
    target-speaker separation happened.
    """
    out = {}
    for arm in ("TSE", "TSE_WRONG", "TSE_ABSENT", "MIX_NORM"):
        vt, vm = [], []
        for it in man["items"]:
            stem = f"{it['item']}.{arm}.{sir}"
            f = audio / f"{stem}.wav"
            t = audio / f"{it['item']}.src_target.wav"
            m = audio / f"{it['item']}.src_masker.{sir}.wav"
            if not (f.exists() and t.exists() and m.exists()):
                continue
            est = read_wav(f)
            vt.append(si_sdr(est, read_wav(t)))
            vm.append(si_sdr(est, read_wav(m)))
        if vt:
            out[arm] = {"vs_target_median": float(np.nanmedian(vt)),
                        "vs_masker_median": float(np.nanmedian(vm)), "n": len(vt)}
    return out


def gates(res: dict, primary: bool) -> dict:
    """G1-G4 exactly as preregistered. Evaluated at the primary SIR only.

    Deterministic robustness criteria, not significance claims: with six meeting
    clusters a bootstrap CI is not a credible inferential gate, so the CIs are
    reported for description and these decide.
    """
    A = res["arms"]
    g = {"evaluated": bool(primary)}
    if not primary:
        g["note"] = "descriptive SIR; cannot rescue a failed gate"
        return g
    need = {"TSE", "TSE_WRONG", "MIX_NORM", "CLEAN_NORM", "TSE_CLEAN"}
    if not need <= A.keys():
        g["error"] = f"missing arms: {sorted(need - A.keys())}"
        return g
    B = A["MIX_NORM"]["wer"]["point"]
    d_right = B - A["TSE"]["wer"]["point"]
    d_wrong = B - A["TSE_WRONG"]["wer"]["point"]
    loo = res["paired"]["TSE-MIX_NORM"]["leave_one_meeting_out"]
    dom = res["domination"]
    g["G1_separates"] = {
        "d_right": d_right,
        "all_loo_negative_delta": all(v is not None and v < 0 for v in loo.values()),
        "loo": loo,
        "top_item_share": dom.get("top_share"),
        "pass": bool(d_right > 0
                     and all(v is not None and v < 0 for v in loo.values())
                     and (dom.get("top_share") is None
                          or dom["top_share"] < DOMINATION_MAX))}
    g["G2_targets_not_enhances"] = {
        "d_right": d_right, "d_wrong": d_wrong,
        "contrast_C": d_wrong - 0.5 * d_right,
        "pass": bool(d_right > 0 and (d_wrong - 0.5 * d_right) < 0)}
    cost = A["TSE_CLEAN"]["wer"]["point"] - A["CLEAN_NORM"]["wer"]["point"]
    g["G3_safe_off_target"] = {"cost_on_clean": cost, "max": G3_MAX,
                               "pass": bool(cost <= G3_MAX)}
    harm = A["TSE"]["del_rate"]["point"] - A["MIX_NORM"]["del_rate"]["point"]
    g["G4_no_deletion_purchase"] = {"del_harm": harm, "max": DEL_HARM_MAX,
                                    "pass": bool(harm <= DEL_HARM_MAX)}
    g["all_pass"] = all(v["pass"] for k, v in g.items()
                        if k.startswith("G") and isinstance(v, dict))
    return g


def domination(per_item: dict, a: str, b: str) -> dict:
    """Largest single item's share of the total a-b error difference.

    One window has supplied 67% of a headline effect in this project, so this is
    reported beside every delta rather than on request.
    """
    diffs = {k: v[a]["err"] - v[b]["err"] for k, v in per_item.items()
             if a in v and b in v}
    tot = sum(diffs.values())
    if not diffs or tot == 0:
        return {"total_err_diff": tot, "top_item": None, "top_share": None}
    top = min(diffs, key=lambda k: diffs[k]) if tot < 0 else max(diffs, key=lambda k: diffs[k])
    return {"total_err_diff": tot, "top_item": top, "top_item_diff": diffs[top],
            "top_share": diffs[top] / tot}


# ======================================================================= stage 2
# Real gold overlap. NOT a measurement: 6 blocks, ~23 post-mask reference tokens,
# 2 meetings. Preregistration section 5 forbids any aggregate WER, delta or
# bootstrap here. What this produces is a per-case failure-mode audit.

def overlap_groups(blocks: list[dict]) -> list[list[dict]]:
    """Maximal transitively-connected sets of `ov_with`-linked blocks."""
    by_id = {b["id"]: b for b in blocks}
    seen, out = set(), []
    for b in sorted(blocks, key=lambda x: (x["s"], x["id"])):
        if b["id"] in seen or not b.get("ov_with"):
            continue
        grp, stack = {b["id"]}, [b["id"]]
        while stack:
            for nb in by_id[stack.pop()].get("ov_with", []):
                if nb in by_id and nb not in grp:
                    grp.add(nb)
                    stack.append(nb)
        seen |= grp
        out.append([by_id[g] for g in sorted(grp, key=lambda g: (by_id[g]["s"], g))])
    return out


def overlap_seconds(block: dict, by_id: dict[str, dict]) -> float:
    """Union duration where a block overlaps any of its linked partners."""
    intervals = []
    for partner_id in block.get("ov_with", []):
        partner = by_id.get(partner_id)
        if partner is None:
            continue
        start = max(block["s"], partner["s"])
        end = min(block["e"], partner["e"])
        if end > start:
            intervals.append((start, end))
    # The gold set has at most one simultaneous partner in the primary cases.
    # Keep the merge explicit so the estimate remains correct if that changes.
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def build_stage2(out: Path = WORK):
    """Write one region wav per overlap group plus a per-speaker enrollment.

    Speakers below MIN_ENROLL_SEC are recorded as `enrollable: false` and their
    reference words stay in the case — the frozen estimand is all groups, with an
    unenrollable speaker's words counted as unavoidable TSE deletions.
    """
    man, ans, _cells, sel = load()
    audio = out / "stage2"
    clip_cache: dict[str, np.ndarray] = {}

    def clip(cid):
        if cid not in clip_cache:
            clip_cache[cid] = read_wav(CLIPS / f"{cid}.wav")
        return clip_cache[cid]

    cases = []
    for cid in man["scored_cell_ids"]:
        blocks = gold_blocks(ans, cid)
        by_id = {b["id"]: b for b in blocks}
        core = region_blocks(blocks, "core_envelope")
        eligible_ids = {
            b["id"] for b in core
            if not b.get("text_unc")
            and any(by_id.get(pid) is not None
                    and not by_id[pid].get("text_unc")
                    for pid in b.get("ov_with", []))
        }
        w = clip(cid)
        dur = w.size / SR
        for gi, full_grp in enumerate(overlap_groups(blocks)):
            # Stage 2 is deliberately the tiny real-overlap substrate: retain
            # only core blocks whose linked overlap partner is also text-certain.
            # Full block text remains available for the per-case audit; the
            # duration-weighted estimate below is the preregistered ~23-token
            # post-mask denominator.
            grp = [b for b in full_grp if b["id"] in eligible_ids]
            if not grp:
                continue
            gid = f"{cid}__g{gi}"
            s = max(0.0, min(b["s"] for b in grp) - PAD_SEC)
            e = min(dur, max(b["e"] for b in grp) + PAD_SEC)
            region = cut(w, s, e)
            write_wav(audio / f"{gid}.BASELINE.wav", region)
            write_wav(audio / f"{gid}.MIX_NORM.wav", peak_norm(region))
            spks = []
            for spk in sorted({b["spk"] for b in grp}):
                segs = enrollment_span(blocks, spk, None)
                enr = concat_segments(w, segs)
                ok = bool(segs)
                if ok:
                    write_wav(audio / f"{gid}.{spk}.enroll.wav", enr)
                spks.append({"spk": spk, "enrollable": ok,
                             "enroll_sec": round(enr.size / SR, 3)})
            block_ref_tokens = sum(len(wtoks(b["text"])) for b in grp)
            overlap_sec = sum(overlap_seconds(b, by_id) for b in grp)
            ref_token_estimate = sum(
                len(wtoks(b["text"])) * overlap_seconds(b, by_id)
                / max(b["e"] - b["s"], 1e-9)
                for b in grp
            )
            cases.append({
                "case": gid, "cid": cid, "meeting": sel[cid]["meeting_id"],
                "s": round(s, 3), "e": round(e, 3),
                "speakers": spks,
                "blocks": [{"id": b["id"], "spk": b["spk"], "s": b["s"], "e": b["e"],
                            "text_unc": bool(b.get("text_unc")),
                            "n_tok": len(wtoks(b["text"]))} for b in grp],
                "block_ref_tokens": block_ref_tokens,
                "overlap_sec": round(overlap_sec, 3),
                "ref_tokens_estimate": round(ref_token_estimate, 2),
            })
    m2 = {"protocol": "tse-overlap-2026-08-16", "stage": 2,
          "n_cases": len(cases),
          "n_meetings": len({c["meeting"] for c in cases}),
          "ref_tokens": round(sum(c["ref_tokens_estimate"] for c in cases)),
          "block_ref_tokens": sum(c["block_ref_tokens"] for c in cases),
          "overlap_sec": round(sum(c["overlap_sec"] for c in cases), 3),
          "speakers_total": sum(len(c["speakers"]) for c in cases),
          "speakers_enrollable": sum(1 for c in cases for s in c["speakers"]
                                     if s["enrollable"]),
          "cases": cases}
    (out / "manifest_stage2.json").write_text(json.dumps(m2, ensure_ascii=False, indent=1))
    print(f"stage2: {len(cases)} cases / {m2['n_meetings']} meetings / "
          f"{m2['ref_tokens']} certain ref tokens; enrollment coverage "
          f"{m2['speakers_enrollable']}/{m2['speakers_total']} speakers")
    return m2


def audit_stage2(work: Path = WORK):
    """Per-case S/D/I and named failure modes. No aggregate, by preregistration."""
    m2 = json.loads((work / "manifest_stage2.json").read_text())
    _m, ans, _c, _s = load()
    hyp = work / "hyp_stage2"
    rows = []
    for c in m2["cases"]:
        blocks = {b["id"]: b for b in ans[c["cid"]]["b"]["blocks"]}
        ref = [t for b in c["blocks"] if not b["text_unc"]
               for t in wtoks(blocks[b["id"]]["text"])]
        row = {"case": c["case"], "meeting": c["meeting"], "n_ref": len(ref),
               "enrollment": {s["spk"]: s["enroll_sec"] for s in c["speakers"]},
               "unenrollable": [s["spk"] for s in c["speakers"] if not s["enrollable"]],
               "arms": {}, "failures": []}
        base = None
        enrolled = [s for s in c["speakers"] if s["enrollable"]]
        arms = ["BASELINE", "MIX_NORM"] + [f"TSE.{s['spk']}" for s in enrolled]
        if len(enrolled) > 1:
            arms += [f"TSE_WRONG.{s['spk']}" for s in enrolled]
        for arm in arms:
            f = hyp / f"{c['case']}.{arm}.json"
            if not f.exists():
                row["failures"].append(f"{arm}: no output")
                continue
            h = wtoks(json.loads(f.read_text())["text"])
            cnt = sdi_counts(ref, h)
            row["arms"][arm] = {k: cnt[k] for k in ("S", "D", "I", "M", "N", "err")}
            row["arms"][arm]["n_hyp"] = len(h)
            if arm == "BASELINE":
                base = set(h)
            if not h:
                row["failures"].append(f"{arm}: empty transcript")
            if arm.startswith(("TSE.", "TSE_WRONG.")):
                wav = work / "stage2" / f"{c['case']}.{arm}.wav"
                if wav.exists() and not np.any(read_wav(wav)):
                    row["failures"].append(f"{arm}: extraction returned silence")
            if base is not None and arm.startswith("TSE."):
                # words the baseline did not have that ARE in the human reference:
                # a count, reported as a count
                new = [t for t in h if t not in base]
                row["arms"][arm]["new_words"] = len(new)
                row["arms"][arm]["new_words_in_reference"] = sum(1 for t in new if t in ref)
        rows.append(row)
    extraction_failures = []
    failure_file = work / "extract_stage2_failures.json"
    if failure_file.exists():
        extraction_failures = json.loads(failure_file.read_text())
    out = {"protocol": m2["protocol"], "stage": 2,
           "n_cases": m2["n_cases"], "n_meetings": m2["n_meetings"],
           "ref_tokens": m2["ref_tokens"],
           "enrollment_coverage": [m2["speakers_enrollable"], m2["speakers_total"]],
           "note": ("case-level failure-mode audit; no aggregate WER, delta or "
                    "bootstrap is computed from this substrate, by preregistration"),
           "extraction_failures": extraction_failures,
           "cases": rows}
    (ROOT / "eval" / "results_tse_overlap_stage2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "decode":
        decode()
    elif cmd == "build2":
        build_stage2()
    elif cmd == "decode2":
        decode_stage2()
    elif cmd == "audit2":
        r = audit_stage2()
        print(json.dumps({k: v for k, v in r.items() if k != "cases"}, ensure_ascii=False))
    elif cmd == "score":
        r = score()
        print(json.dumps({k: v for k, v in r.items() if k != "sirs"}, ensure_ascii=False))
        for sir, res in r["sirs"].items():
            print(f"SIR {sir} dB  n={res['n_scored']}")
            for arm, v in res["arms"].items():
                w, d, i = v["wer"], v["del_rate"], v["ins_rate"]
                print(f"  {arm:10s} WER {w['point']:.4f} "
                      f"[{w['ci95_meeting_cluster'][0]:.4f},{w['ci95_meeting_cluster'][1]:.4f}] "
                      f"D {d['point']:.4f}  I {i['point']:.4f}")
            print("  si-sdr:", json.dumps(res.get("si_sdr_db", {}), ensure_ascii=False))
            print("  gates:", json.dumps(res["gates"], ensure_ascii=False))
    else:
        raise SystemExit(f"unknown command {cmd}")
