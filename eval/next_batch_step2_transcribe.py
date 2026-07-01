"""Next-batch Step 2b — extract clips + Soniox re-transcription (calibration).

Runbook Step 2. For each sampled candidate: HTTP-seek the meeting mp3 with ffmpeg
(padded +/-0.3s), re-encode to 16k mono WAV, verify the decoded duration, then
re-transcribe with the Soniox realtime path (file_transcribe.py --lang el — mints
its own temp key, no API key). Resumable + every failure logged (per plan review:
failed clips must be visible, not silently dropped from calibration).

Artifacts under data/next-batch/calib/:
  clips/<utterance_id>.wav      extracted audio
  asr/<utterance_id>.json       {soniox_text, durations, ok flags, error}
  transcribe_manifest.csv       one row per processed item (append/rebuild)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "data" / "next-batch" / "calib"
CLIPS = CALIB / "clips"
ASR = CALIB / "asr"
SONIOX_DIR = Path("/home/harold/projects/soniox-tools")
SONIOX_PY = SONIOX_DIR / ".venv" / "bin" / "python"
PAD = 0.3
MARKER = "===== TRANSCRIPT ====="


def _ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def extract_clip(url: str, start: float, end: float, out: Path) -> dict:
    pstart = max(0.0, start - PAD)
    dur = (end + PAD) - pstart
    cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{pstart:.3f}", "-i", url,
           "-t", f"{dur:.3f}", "-ac", "1", "-ar", "16000", "-f", "wav", str(out)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=180)
    except subprocess.TimeoutExpired:
        return {"extract_ok": False, "error": "ffmpeg_timeout",
                "expected_dur": dur, "decoded_dur": None}
    if p.returncode != 0 or not out.exists():
        return {"extract_ok": False, "error": f"ffmpeg_rc={p.returncode}: " + p.stderr.strip()[-300:],
                "expected_dur": dur, "decoded_dur": None}
    decoded = _ffprobe_duration(out)
    # reject gross mismatches (server ignored range / returned whole file / empty)
    ok = decoded is not None and (dur * 0.5 - 0.2) <= decoded <= (dur + 2.0)
    return {
        "extract_ok": bool(ok),
        "error": None if ok else f"duration_mismatch expected={dur:.2f} decoded={decoded}",
        "expected_dur": round(dur, 3),
        "decoded_dur": None if decoded is None else round(decoded, 3),
        "pstart": round(pstart, 3),
        "ffmpeg_s": round(time.time() - t0, 1),
    }


def transcribe(wav: Path, expected_dur: float) -> dict:
    timeout = max(90.0, expected_dur * 5 + 40)
    try:
        p = subprocess.run(
            [str(SONIOX_PY), "file_transcribe.py", str(wav), "--lang", "el"],
            capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(SONIOX_DIR),
        )
    except subprocess.TimeoutExpired:
        return {"asr_ok": False, "soniox_text": "", "error": "asr_timeout"}
    if MARKER not in p.stdout:
        return {"asr_ok": False, "soniox_text": "",
                "error": f"no_marker rc={p.returncode}: " + (p.stderr.strip()[-300:] or p.stdout.strip()[-300:])}
    text = p.stdout.split(MARKER, 1)[1].strip()
    keyline = next((l for l in p.stderr.splitlines() if l.startswith("[key]")), "")
    return {"asr_ok": True, "soniox_text": text, "error": None, "key_meta": keyline.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process at most N (0=all)")
    args = ap.parse_args()

    CLIPS.mkdir(parents=True, exist_ok=True)
    ASR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_parquet(CALIB / "sample.parquet")

    todo = []
    for _, r in sample.iterrows():
        jp = ASR / f"{r['utterance_id']}.json"
        if jp.exists():
            try:
                if json.loads(jp.read_text()).get("asr_ok"):
                    continue
            except Exception:
                pass
        todo.append(r)
    if args.limit:
        todo = todo[: args.limit]

    print(f"[step2] {len(sample)} sampled, {len(todo)} to process", flush=True)
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        uid = r["utterance_id"]
        wav = CLIPS / f"{uid}.wav"
        rec = {"utterance_id": uid, "city_id": r["city_id"], "meeting_id": r["meeting_id"],
               "audio_url": r["audio_url"], "utterance_start": float(r["utterance_start"]),
               "utterance_end": float(r["utterance_end"]), "duration": float(r["duration"]),
               "clip_path": str(wav.relative_to(ROOT))}
        ext = extract_clip(r["audio_url"], float(r["utterance_start"]),
                           float(r["utterance_end"]), wav)
        rec.update(ext)
        if ext["extract_ok"]:
            rec.update(transcribe(wav, ext["expected_dur"]))
        else:
            rec.update({"asr_ok": False, "soniox_text": "", "error": ext["error"]})
        (ASR / f"{uid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

        el = time.time() - t0
        rate = i / el
        status = "ok" if rec.get("asr_ok") else f"FAIL({rec.get('error','')[:40]})"
        print(f"[{i}/{len(todo)}] {uid} {status}  {el/60:.1f}m  "
              f"ETA {(len(todo)-i)/rate/60:.1f}m", flush=True)

    # rebuild manifest from all asr json
    rows = []
    for jp in sorted(ASR.glob("*.json")):
        try:
            rows.append(json.loads(jp.read_text()))
        except Exception:
            pass
    man = pd.DataFrame(rows)
    man.to_csv(CALIB / "transcribe_manifest.csv", index=False)
    n_ok = int(man["asr_ok"].sum()) if len(man) else 0
    print(f"[step2] done. {n_ok}/{len(man)} transcribed -> {CALIB/'transcribe_manifest.csv'}",
          flush=True)


if __name__ == "__main__":
    main()
