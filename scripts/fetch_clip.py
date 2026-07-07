"""Fetch the audio for one dataset row WITHOUT downloading the whole meeting mp3.

data.opencouncil.gr serves HTTP range requests (206 Partial Content), so ffmpeg
can seek to the utterance and pull only the ~clip-length segment (~90 kB) instead
of the full meeting recording (tens of MB). The cut uses the boundary-corrected
span `start_adj`/`end_adj` (falling back to the raw `start`/`end` when those are
null, e.g. align_failed), so no syllable is clipped.

CLI:
  python scripts/fetch_clip.py data/hf-dataset/public/validation.parquet <utterance_id> [out.wav]
Library:
  from scripts.fetch_clip import fetch_clip, clip_span
  fetch_clip(url, start, end, "out.wav")     # writes a 16 kHz mono wav, returns the path
  samples, sr = fetch_clip(url, start, end)  # returns a numpy array + sample rate
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile

SR = 16000
PAD_S = 0.15  # extra safety margin around the (already padded) corrected span


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def clip_span(row: dict) -> tuple[float, float]:
    """The span to cut on: boundary-corrected (start_adj/end_adj) if present,
    otherwise the raw start/end."""
    sa, ea = row.get("start_adj"), row.get("end_adj")
    if _ok(sa) and _ok(ea):
        return float(sa), float(ea)
    return float(row["start"]), float(row["end"])


def fetch_clip(audio_url: str, start: float, end: float,
               out_path: str | None = None, *, sr: int = SR, pad: float = PAD_S):
    """Download only [start-pad, end+pad] via ffmpeg HTTP range-seeking, as
    `sr`-Hz mono. Writes a wav to out_path (returns the path) or, if out_path is
    None, returns (samples, sr)."""
    s = max(0.0, float(start) - pad)
    dur = float(end) - float(start) + 2 * pad
    if dur <= 0:
        raise ValueError(f"non-positive clip duration for {audio_url}")
    target = out_path
    tmp_fd = None
    if target is None:
        tmp_fd, target = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error",
           "-ss", f"{s:.3f}", "-i", audio_url, "-t", f"{dur:.3f}",
           "-ar", str(sr), "-ac", "1", "-y", target]
    subprocess.run(cmd, check=True)
    if out_path is not None:
        return out_path
    import soundfile as sf
    data, got_sr = sf.read(target)
    os.unlink(target)
    return data, got_sr


def _main() -> None:
    import pandas as pd
    if len(sys.argv) < 3:
        sys.exit("usage: fetch_clip.py <parquet> <utterance_id> [out.wav]")
    parquet, uid = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else f"{uid}.wav"
    df = pd.read_parquet(parquet)
    hit = df[df.utterance_id == uid]
    if hit.empty:
        sys.exit(f"utterance {uid} not found in {parquet}")
    row = hit.iloc[0].to_dict()
    s, e = clip_span(row)
    kind = "adj" if _ok(row.get("start_adj")) else "raw"
    fetch_clip(row["audio_url"], s, e, out)
    print(f"{out}  ({e - s:.2f}s, {kind}-span [{s:.2f},{e:.2f}], "
          f"status={row.get('boundary_status')})")


if __name__ == "__main__":
    _main()
