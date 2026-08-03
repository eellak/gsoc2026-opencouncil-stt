#!/usr/bin/env python3
"""Build the paired synthetic-overlap mixtures.

Design and its limits are frozen in `docs/specs/synthetic-overlap-preregistration.md`.
Read that first; this file only implements it.

The one thing worth restating here, because it is the easiest thing to get wrong: no arm
is normalised on its own. Per-arm loudness matching would move the TARGET speaker's level
between arms, and then "mixed is worse" could just mean "quieter target". Every arm of an
item is written through one common attenuation chosen so the loudest arm clears -1 dBFS,
and the clean arm goes through the identical decode/write path.

Audio and mixtures are PII (council speech). Everything is written under $SC or
~/.cache/oc-overlap, never into the repo.

Usage:
  SC=~/.cache/oc-overlap python eval/controlled_eval/synth_overlap_build.py
Env: SC  N_ITEMS (cap, for the QC batch)  OUT (mixture dir)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as B  # noqa: E402

ROOT = Path("/home/harold/opencouncil-fine-tuning")
AUDIO = ROOT / "data/asr/audio"
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
OUT = Path(os.environ.get("OUT", SC / "mixtures"))
N_ITEMS = int(os.environ.get("N_ITEMS", "0"))

SR = 16_000
DONOR_MIN, DONOR_MAX = 1.5, 3.0        # seconds
PEAK_CEIL = 10 ** (-1 / 20)            # -1 dBFS
SIR_TOL = 0.25                         # dB, QC gate
BUILD_VERSION = 1

# arm -> (kind, sir_db). "clean" ignores sir; "gain" uses sir as a plain dB gain.
ARMS = {
    "A": ("clean", None),
    "B": ("donor", 15.0),
    "C": ("donor", 5.0),               # primary
    "D": ("donor", 0.0),
    "E": ("noise", 5.0),               # envelope-matched speech-shaped noise
    "F": ("reversed", 5.0),
    "G": ("gain", +3.0),
    "H": ("gain", -3.0),
}


def log(*a):
    print(*a, flush=True)


def h32(*parts) -> int:
    """Stable documented hash. Python's hash() is salted per process."""
    d = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(d[:8], "big")


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(h32(*parts) % (2 ** 32))


# ------------------------------------------------------------------- audio plumbing
def decode(src: Path, start: float, dur: float) -> np.ndarray:
    """mp3 -> float64 mono 16 kHz. One decode path for every arm, clean included."""
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "w.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
             "-i", str(src), "-ac", "1", "-ar", str(SR), "-f", "wav", str(w)],
            check=True)
        raw = np.frombuffer(w.read_bytes()[44:], dtype="<i2")
    return raw.astype(np.float64) / 32768.0


def write_wav(path: Path, x: np.ndarray):
    assert np.abs(x).max() < 1.0, "clipped"
    pcm = np.round(x * 32767.0).astype("<i2")
    hdr = (b"RIFF" + (36 + pcm.nbytes).to_bytes(4, "little") + b"WAVEfmt "
           + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
           + (1).to_bytes(2, "little") + SR.to_bytes(4, "little")
           + (SR * 2).to_bytes(4, "little") + (2).to_bytes(2, "little")
           + (16).to_bytes(2, "little") + b"data" + pcm.nbytes.to_bytes(4, "little"))
    path.write_bytes(hdr + pcm.tobytes())


# ------------------------------------------------------------------- speech activity
def frame_rms(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-20)


def active_mask(x: np.ndarray, hop: int = SR // 100) -> np.ndarray:
    """Per-frame speech-active mask, threshold relative to the clip's own level.

    Deliberately crude. It only has to answer "is the main speaker talking here", and
    pyannote is not usable for it: the diarization cache keeps summary features, not
    segment boundaries. Threshold is 12 dB below the 90th-percentile frame, with a
    two-frame hysteresis so single quiet frames inside a word do not split it.
    """
    r = frame_rms(x, hop)
    thr = np.percentile(r, 90) * 10 ** (-12 / 20)
    m = r > thr
    for _ in range(2):                       # close 1-frame gaps
        m[1:-1] |= m[:-2] & m[2:]
    return m


def active_rms(x: np.ndarray, hop: int = SR // 100) -> float:
    m = active_mask(x, hop)
    if not m.any():
        return float(np.sqrt((x ** 2).mean() + 1e-20))
    idx = np.repeat(m, hop)[:len(x)]
    return float(np.sqrt((x[idx] ** 2).mean() + 1e-20))


def longest_active_run(x: np.ndarray, want: float, hop: int = SR // 100):
    """Start sample of a `want`-second stretch that is speech-active throughout."""
    m = active_mask(x, hop)
    need = int(round(want * SR / hop))
    if need > len(m):
        return None
    cs = np.concatenate([[0], np.cumsum(m.astype(int))])
    run = cs[need:] - cs[:-need]
    best = int(np.argmax(run))
    if run[best] < need:                      # no fully-active stretch
        return None
    return best * hop


# ------------------------------------------------------------------- donor mining
def mine_donor(item, dur: float):
    """A single-speaker excerpt from a zero-overlap window, fully speech-active."""
    x = decode(item["_audio"], item["_start"], item["_dur"])
    s = longest_active_run(x, dur)
    if s is None:
        return None
    d = x[s:s + int(dur * SR)]
    # fade 10 ms so the insertion has no click of its own to be detected by
    f = int(0.01 * SR)
    ramp = np.linspace(0, 1, f)
    d[:f] *= ramp
    d[-f:] *= ramp[::-1]
    return d


def speech_shaped_noise(donor: np.ndarray, rng) -> np.ndarray:
    """Noise with the donor's long-term spectrum and broad amplitude envelope.

    The energy control. It is matched on everything that is not linguistic content:
    duration, spectrum, and envelope. It is NOT "the donor without the voice" — no such
    signal exists — and the report describes it operationally.
    """
    n = len(donor)
    spec = np.abs(np.fft.rfft(donor))
    phase = np.exp(2j * np.pi * rng.random(len(spec)))
    y = np.fft.irfft(spec * phase, n=n)
    # impose the donor's envelope (20 ms smoothing) so onsets/offsets line up
    hop = SR // 50
    env = np.repeat(frame_rms(donor, hop), hop)[:n]
    env = np.concatenate([env, np.full(n - len(env), env[-1] if len(env) else 1.0)])
    ycur = np.repeat(frame_rms(y, hop), hop)[:n]
    ycur = np.concatenate([ycur, np.full(n - len(ycur), ycur[-1] if len(ycur) else 1.0)])
    return y * (env / np.maximum(ycur, 1e-12))


# ------------------------------------------------------------------- the build
def build_item(item, donors, out_dir: Path) -> dict | None:
    target = decode(item["_audio"], item["_start"], item["_dur"])
    n = len(target)

    dur = float(rng_for("dur", item["item_id"]).uniform(DONOR_MIN, DONOR_MAX))
    pos = longest_active_run(target, dur + 0.2)
    if pos is None:
        return None                              # no fully-active stretch to talk over
    pos += int(0.1 * SR)

    # donor from a different city, chosen by a stable hash of the item id
    pool = [d for d in donors if d["city_id"] != item["city_id"]]
    if not pool:
        return None
    donor = pool[h32("donor", item["item_id"]) % len(pool)]
    d = mine_donor(donor, dur)
    if d is None:
        return None

    local = target[pos:pos + len(d)]
    t_rms = active_rms(local)
    rng = rng_for("noise", item["item_id"])

    variants = {"donor": d, "reversed": d[::-1].copy(),
                "noise": speech_shaped_noise(d, rng)}

    arms, achieved = {}, {}
    for arm, (kind, val) in ARMS.items():
        if kind == "clean":
            arms[arm] = target.copy()
        elif kind == "gain":
            arms[arm] = target * (10 ** (val / 20))
        else:
            v = variants[kind]
            g = (t_rms / active_rms(v)) * 10 ** (-val / 20)
            y = target.copy()
            y[pos:pos + len(v)] += v * g
            arms[arm] = y
            achieved[arm] = 20 * np.log10(t_rms / max(active_rms(v * g), 1e-12))

    # ONE common attenuation across every arm of this item, clean included.
    peak = max(float(np.abs(a).max()) for a in arms.values())
    gain = min(1.0, PEAK_CEIL / peak) if peak > 0 else 1.0

    bad = [a for a, got in achieved.items() if abs(got - ARMS[a][1]) > SIR_TOL]
    if bad:
        return None                              # QC gate: achieved SIR off spec

    ref = (arms["A"] * gain)
    for arm, y in arms.items():
        write_wav(out_dir / f"{item['item_id']}__{arm}.wav", y * gain)
    write_wav(out_dir / f"{item['item_id']}__donor.wav", d * (PEAK_CEIL / max(np.abs(d).max(), 1e-9)))

    # the target must be untouched outside the event, up to the common gain
    for arm in ("B", "C", "D", "E", "F"):
        a = arms[arm] * gain
        assert np.allclose(a[:pos], ref[:pos], atol=1e-9), f"{arm} altered audio before the event"
        assert np.allclose(a[pos + len(d):], ref[pos + len(d):], atol=1e-9), \
            f"{arm} altered audio after the event"

    return {
        "item_id": item["item_id"], "city_id": item["city_id"],
        "meeting_id": item["meeting_id"], "ref": item["ref"],
        "donor_item": donor["item_id"], "donor_city": donor["city_id"],
        "donor_meeting": donor["meeting_id"],
        "event_start_sec": pos / SR, "event_dur_sec": len(d) / SR,
        "window_dur_sec": n / SR,
        "common_gain_db": 20 * float(np.log10(gain)),
        "achieved_sir_db": {k: round(float(v), 3) for k, v in achieved.items()},
        "peak_dbfs": round(20 * float(np.log10(peak * gain)), 3),
    }


def main():
    feats = json.loads((SC / "overlap_features.json").read_text())["features"]
    report = B.load_report()
    providers = B.provider_ids(report)
    items = B.common_items(report, providers)
    by_id = {it["itemId"]: it for it in report["items"]}

    pool = []
    for it in items:
        p = AUDIO / f"{it['city_id']}__{it['meeting_id']}.mp3"
        f = feats.get(it["item_id"])
        if not p.exists() or f is None or f.get("overlap_sec", 0) > 0:
            continue
        raw = by_id[it["item_id"]]
        it["_audio"], it["_start"], it["_dur"] = p, raw["startSec"], raw["durationSec"]
        pool.append(it)
    log(f"{len(pool)} zero-overlap windows with local audio")

    targets = pool[:N_ITEMS] if N_ITEMS else pool
    OUT.mkdir(parents=True, exist_ok=True)

    # Donors should be one voice. A zero-overlap window can still hold two speakers
    # taking turns, so prefer windows the diarizer saw a single speaker in; the
    # contiguous fully-active stretch mine_donor takes then cannot straddle a handover.
    solo = [it for it in pool if feats[it["item_id"]].get("n_detected_speakers", 9) == 1]
    donors = solo if len(solo) >= 10 else pool
    log(f"donor pool: {len(donors)} windows ({'single-speaker' if donors is solo else 'all zero-overlap'})")

    built, failed = [], []
    for i, it in enumerate(targets, 1):
        try:
            m = build_item(it, donors, OUT)
            err = "qc gate or no usable speech stretch"
        except (AssertionError, subprocess.CalledProcessError) as e:
            m, err = None, f"{type(e).__name__}: {e}"
        if m is None:
            failed.append({"item_id": it["item_id"], "why": err})
        else:
            built.append(m)
        if i % 10 == 0 or i == len(targets):
            log(f"  {i}/{len(targets)}  built={len(built)} failed={len(failed)}")

    manifest = {
        "build_version": BUILD_VERSION, "sr": SR, "arms": {k: list(v) for k, v in ARMS.items()},
        "n_built": len(built), "n_failed": len(failed), "failed": failed,
        "n_target_meetings": len({m["meeting_id"] for m in built}),
        "n_donor_meetings": len({m["donor_meeting"] for m in built}),
        "items": built,
    }
    (SC / "synth_overlap_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    log(f"manifest -> {SC / 'synth_overlap_manifest.json'}")
    log(f"{len(built)} items, {manifest['n_target_meetings']} target meetings, "
        f"{manifest['n_donor_meetings']} donor meetings, wavs in {OUT}")


if __name__ == "__main__":
    main()
