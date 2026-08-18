#!/usr/bin/env python3
"""Cut the 299 s page audio from the source recording, and prove it is the same audio.

Why not just concatenate the two cached benchmark WAVs: they leave a 169 ms hole at
2096.065..2096.234 (see `coords.SEAM`), exactly where a boundary error would live.
Butt-joining them puts a discontinuity in the one place the page exists to inspect.

Why the proof: an extraction that is 40 ms off, or resampled differently, or at a
different gain, would make every timestamp on the page wrong by a constant nobody
would notice. So the extraction is cross-correlated against each cached window WAV
and the lag is reported. numpy only; scipy is not installed here.
"""
from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.tsfusion.coords import PAGE_DURATION, T0, WINDOWS

SR = 16000


def read_wav_mono16(path: Path, offset_s: float = 0.0, dur_s: float | None = None):
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, (path, w.getparams())
        sr = w.getframerate()
        w.setpos(int(round(offset_s * sr)))
        n = w.getnframes() if dur_s is None else int(round(dur_s * sr))
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return x, sr


def extract(source: Path, out_wav: Path, start: float = T0,
            duration: float = PAGE_DURATION, sr: int = SR) -> Path:
    """Sample-accurate cut of [start, start+duration) as 16 kHz mono PCM."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-accurate_seek", "-ss", f"{start:.6f}",
         "-i", str(source), "-t", f"{duration:.6f}",
         "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(out_wav)],
        check=True)
    return out_wav


def to_mp3(src_wav: Path, out_mp3: Path, kbps: int = 64) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src_wav),
                    "-b:a", f"{kbps}k", "-ac", "1", str(out_mp3)], check=True)
    return out_mp3


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples) of `b` relative to `a` maximising normalised cross-correlation.

    Positive lag means `b` occurs LATER in `a`'s clock. Returns (lag, peak) where
    peak is the Pearson correlation at that lag, so a bad match is visible as a low
    number rather than as a confident wrong lag.
    """
    a = a - a.mean()
    b = b - b.mean()
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    cc = np.fft.irfft(fa * np.conj(fb), n)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = cc[lags % n]
    k = int(np.argmax(vals))
    lag = int(lags[k])
    # Pearson at that lag, computed on the actually overlapping samples
    if lag >= 0:
        x, y = a[lag:lag + len(b)], b[:len(a) - lag]
    else:
        x, y = a[:len(b) + lag], b[-lag:]
    m = min(len(x), len(y))
    x, y = x[:m], y[:m]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    peak = float(np.dot(x, y) / denom) if denom else 0.0
    return lag, peak


@dataclass
class AlignCheck:
    item_id: str
    probe_at_window_s: float
    probe_seconds: float
    expected_page_offset_s: float
    lag_samples: int
    lag_ms: float
    peak_corr: float
    rms_ratio: float
    ok: bool
    note: str = ""


PROBES = (2.0, 20.0, 75.0, 140.0)


def verify_against_windows(page_wav: Path, windows_dir: Path,
                           probes: tuple[float, ...] = PROBES,
                           probe_seconds: float = 5.0,
                           max_lag_ms: float = 250.0) -> list[AlignCheck]:
    """Cross-correlate several slices of each cached window WAV against the extraction.

    One probe per window is not enough: a constant offset and a drift look the same
    from a single point, and the boundary is exactly where a boundary error hides. So
    probes are taken near the head, in the body and near the tail of each window, and
    every lag is reported.
    """
    page, sr = read_wav_mono16(page_wav)
    out = []
    for w, probe_at in [(w, p) for w in WINDOWS for p in probes]:
        wav = windows_dir / f"{w.item_id}.wav"
        ref, wsr = read_wav_mono16(wav, probe_at, probe_seconds)
        assert wsr == sr, (wsr, sr)
        expected = (w.start - T0) + probe_at            # page seconds
        max_lag = int(max_lag_ms / 1000 * sr)
        lo = max(0, int((expected - max_lag_ms / 1000) * sr))
        hi = min(len(page), int((expected + probe_seconds + max_lag_ms / 1000) * sr))
        seg = page[lo:hi]
        lag, peak = xcorr_lag(seg, ref, max_lag)
        # lag is relative to `seg`'s start; convert to an error against `expected`
        err = (lo + lag) / sr - expected
        rms_ref = float(np.sqrt(np.mean(ref ** 2))) or 1.0
        if rms_ref < 1.0:                       # silence: correlation is meaningless
            out.append(AlignCheck(
                item_id=w.item_id, probe_at_window_s=probe_at,
                probe_seconds=probe_seconds, expected_page_offset_s=round(expected, 6),
                lag_samples=0, lag_ms=0.0, peak_corr=0.0, rms_ratio=0.0, ok=True,
                note="probe is silence, skipped"))
            continue
        matched = page[int(round((expected + err) * sr)):
                       int(round((expected + err) * sr)) + len(ref)]
        rms_page = float(np.sqrt(np.mean(matched ** 2))) if len(matched) else 0.0
        out.append(AlignCheck(
            item_id=w.item_id, probe_at_window_s=probe_at,
            probe_seconds=probe_seconds, expected_page_offset_s=round(expected, 6),
            lag_samples=int(round(err * sr)), lag_ms=round(err * 1000, 3),
            peak_corr=round(peak, 6), rms_ratio=round(rms_page / rms_ref, 4),
            ok=bool(abs(err * 1000) <= 5.0 and peak >= 0.95)))
    return out


def build(source_mp3: Path, out_dir: Path, windows_dir: Path) -> dict:
    wav = extract(source_mp3, out_dir / "page.wav")
    mp3 = to_mp3(wav, out_dir / "page.mp3")
    checks = verify_against_windows(wav, windows_dir)
    report = {
        "source": str(source_mp3),
        "page_wav": str(wav),
        "page_mp3": str(mp3),
        "start_abs": T0,
        "duration": PAGE_DURATION,
        "sample_rate": SR,
        "checks": [asdict(c) for c in checks],
        "all_ok": all(c.ok for c in checks),
    }
    (out_dir / "audio_check.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path.home() / ".cache/oc-public/tsfusion-2026-08"
    r = build(out / "source.mp3", out, Path.home() / ".cache/oc-public/bench_windows")
    print(json.dumps(r, indent=1))
