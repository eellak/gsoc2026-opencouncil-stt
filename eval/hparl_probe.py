"""HParl faithfulness probe — do the official minutes match the audio?

`ddamianos/hparl` pairs parliamentary audio with the Βουλή's official minutes,
force-aligned. Official minutes are *edited* text, so the standing hypothesis
(docs/reports/2026-08-11-improvement-research.md §"Το πολιτικό dataset") is that
a non-trivial share of rows is not a faithful transcript of its own audio.

This probe measures that on a random sample of one shard:

  1. read N random rows from a parquet shard (audio is a float32 array column)
  2. encode each to mono 16 kHz MP3 (small enough to keep 120h on disk)
  3. transcribe with Soniox (independent ASR, the same tool used for gap2/gap3)
  4. align Soniox vs the dataset `sentence` on greek_normalize tokens and split
     the errors by DIRECTION, which is the whole point:
       - ref_only  (deletions): in the minutes, not in what Soniox heard
                    -> editorial text with no audio behind it. Poisons training.
       - hyp_only  (insertions): heard, but absent from the minutes
                    -> the minutes dropped speech. Teaches the model to delete.
       - sub: both heard something, they disagree (ASR error or paraphrase).

Nothing here decides whether Soniox is right; it is a second opinion, and rows
where two independent sources agree are the ones we can trust cheaply.

Audio and text stay under ~/.cache/oc-public/ — never in git.

Run:  .venv-eval/bin/python eval/hparl_probe.py --n 60
"""
from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.scoring import greek_normalize  # noqa: E402

REPO = "ddamianos/hparl"
WORK = Path.home() / ".cache/oc-public/hparl"
HF_CACHE = WORK / "hf"
CLIPS = WORK / "clips"
ASR = WORK / "asr"

SONIOX_DIR = Path("/home/harold/projects/soniox-tools")
SONIOX_PY = SONIOX_DIR / ".venv" / "bin" / "python"
MARKER = "===== TRANSCRIPT ====="


# ---------- sampling ----------
def _open_shard(shard: str, remote: bool):
    """ParquetFile for one shard. remote=True reads it over HTTP range requests
    (only the row groups we touch are fetched) instead of pulling ~240 MB."""
    if remote:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        return pq.ParquetFile(fs.open(f"datasets/{REPO}/{shard}", "rb"))
    path = hf_hub_download(REPO, shard, repo_type="dataset", cache_dir=str(HF_CACHE))
    return pq.ParquetFile(path)


def sample_rows(shard: str, n: int, seed: int, remote: bool = False) -> list[dict]:
    """N random rows from one parquet shard, reading only the row groups that
    contain them (the audio column expands ~2x in memory, so never read whole)."""
    pf = _open_shard(shard, remote)
    md = pf.metadata
    total = md.num_rows
    rng = np.random.default_rng(seed)
    want = sorted(int(i) for i in rng.choice(total, size=min(n, total), replace=False))

    # map global row index -> (row_group, offset)
    bounds, acc = [], 0
    for g in range(md.num_row_groups):
        acc += md.row_group(g).num_rows
        bounds.append(acc)

    by_group: dict[int, list[int]] = {}
    for idx in want:
        g = int(np.searchsorted(bounds, idx, side="right"))
        start = bounds[g - 1] if g else 0
        by_group.setdefault(g, []).append(idx - start)

    out: list[dict] = []
    for g, offs in sorted(by_group.items()):
        tbl = pf.read_row_group(g, columns=["path", "sentence", "utt_id", "audio"])
        d = tbl.take(offs).to_pylist()
        for r in d:
            out.append({
                "utt_id": r["utt_id"],
                "path": r["path"],
                "sentence": r["sentence"],
                "sr": int(r["audio"]["sampling_rate"]),
                "wav": np.asarray(r["audio"]["array"], dtype=np.float32),
            })
    return out


def to_mp3(pcm: np.ndarray, sr: int, out: Path) -> float:
    """Raw float32 -> 32 kbps mono MP3. Returns duration in seconds."""
    dur = len(pcm) / float(sr)
    if out.exists() and out.stat().st_size > 0:
        return dur
    tmp = out.with_suffix(".part")
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1",
         "-i", "pipe:0", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k",
         "-f", "mp3", str(tmp)],
        input=pcm.tobytes(), capture_output=True, timeout=120)
    if p.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ffmpeg rc={p.returncode}: {p.stderr[-300:]!r}")
    tmp.rename(out)
    return dur


# ---------- ASR ----------
def transcribe(mp3: Path, dur: float) -> dict:
    timeout = max(90.0, dur * 5 + 40)
    try:
        p = subprocess.run([str(SONIOX_PY), "file_transcribe.py", str(mp3),
                            "--lang", "el"],
                           capture_output=True, text=True, errors="replace",
                           timeout=timeout, cwd=str(SONIOX_DIR))
    except subprocess.TimeoutExpired:
        return {"asr_ok": False, "soniox_text": "", "error": "asr_timeout"}
    if MARKER not in p.stdout:
        return {"asr_ok": False, "soniox_text": "",
                "error": f"no_marker rc={p.returncode}: "
                         + (p.stderr.strip()[-300:] or p.stdout.strip()[-300:])}
    return {"asr_ok": True, "soniox_text": p.stdout.split(MARKER, 1)[1].strip(),
            "error": None}


def transcribe_cached(mp3: Path, dur: float, key: str) -> dict:
    jp = ASR / f"{key}.json"
    if jp.exists():
        try:
            r = json.loads(jp.read_text())
            if r.get("asr_ok"):
                return r
        except Exception:
            pass
    r = transcribe(mp3, dur)
    if not r.get("asr_ok"):
        time.sleep(5)
        r = transcribe(mp3, dur)
    jp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return r


# ---------- alignment ----------
def compare(ref: str, hyp: str) -> dict:
    tr, th = greek_normalize(ref).split(), greek_normalize(hyp).split()
    sm = difflib.SequenceMatcher(a=tr, b=th, autojunk=False)
    sub = dele = ins = 0
    ref_only: list[str] = []
    hyp_only: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            dele += i2 - i1
            ref_only += tr[i1:i2]
        elif tag == "insert":
            ins += j2 - j1
            hyp_only += th[j1:j2]
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            sub += n
            if (i2 - i1) > n:
                dele += (i2 - i1) - n
                ref_only += tr[i1 + n:i2]
            if (j2 - j1) > n:
                ins += (j2 - j1) - n
                hyp_only += th[j1 + n:j2]
    nref = len(tr)
    return {
        "n_ref": nref, "n_hyp": len(th),
        "sub": sub, "del": dele, "ins": ins,
        "wer": (sub + dele + ins) / nref if nref else (1.0 if th else 0.0),
        "del_rate": dele / nref if nref else 0.0,
        "ins_rate": ins / nref if nref else 0.0,
        "ref_only": ref_only[:40], "hyp_only": hyp_only[:40],
    }


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="data/test-00000-of-00006.parquet")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--remote", action="store_true",
                    help="range-read the shard from HF instead of downloading it")
    ap.add_argument("--out", default=str(WORK / "probe.jsonl"))
    args = ap.parse_args()

    for d in (HF_CACHE, CLIPS, ASR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[probe] sampling {args.n} rows from {args.shard}", flush=True)
    rows = sample_rows(args.shard, args.n, args.seed, remote=args.remote)
    print(f"[probe] got {len(rows)} rows; encoding MP3", flush=True)

    items = []
    for r in rows:
        mp3 = CLIPS / f"{r['utt_id']}.mp3"
        dur = to_mp3(r["wav"], r["sr"], mp3)
        items.append({"utt_id": r["utt_id"], "path": r["path"],
                      "sentence": r["sentence"], "dur": dur, "mp3": mp3})
    total_audio = sum(i["dur"] for i in items)
    mp3_bytes = sum(i["mp3"].stat().st_size for i in items)
    print(f"[probe] {total_audio/60:.1f} min audio, "
          f"{mp3_bytes/1e6:.1f} MB mp3 "
          f"({mp3_bytes/total_audio/1000:.1f} kB/s)", flush=True)

    out_path = Path(args.out)
    lock = threading.Lock()
    done = 0
    t0 = time.time()

    def work(it: dict) -> dict:
        a = transcribe_cached(it["mp3"], it["dur"], it["utt_id"])
        rec = {"utt_id": it["utt_id"], "path": it["path"], "dur": round(it["dur"], 2),
               "sentence": it["sentence"], "soniox_text": a["soniox_text"],
               "asr_ok": a["asr_ok"], "error": a["error"]}
        if a["asr_ok"]:
            rec.update(compare(it["sentence"], a["soniox_text"]))
        return rec

    recs = []
    with out_path.open("w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it["utt_id"] for it in items}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"utt_id": futs[fut], "asr_ok": False,
                       "error": f"worker: {type(e).__name__}: {e}"}
            with lock:
                done += 1
                recs.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if done % 10 == 0 or done == len(items):
                    el = time.time() - t0
                    print(f"[{done}/{len(items)}] {el/60:.1f}m", flush=True)

    ok = [r for r in recs if r.get("asr_ok")]
    print(f"\n[probe] {len(ok)}/{len(recs)} transcribed -> {out_path}")
    if not ok:
        return
    nref = sum(r["n_ref"] for r in ok)
    print(f"  pooled WER      {sum(r['sub']+r['del']+r['ins'] for r in ok)/nref:.3f}")
    print(f"  pooled del rate {sum(r['del'] for r in ok)/nref:.3f}  "
          f"(minutes text with no audio behind it)")
    print(f"  pooled ins rate {sum(r['ins'] for r in ok)/nref:.3f}  "
          f"(speech the minutes dropped)")
    print(f"  pooled sub rate {sum(r['sub'] for r in ok)/nref:.3f}")
    wers = sorted(r["wer"] for r in ok)
    q = lambda p: wers[min(len(wers) - 1, int(p * len(wers)))]  # noqa: E731
    print(f"  per-utt WER     p10={q(.10):.2f} median={q(.50):.2f} "
          f"p90={q(.90):.2f}")
    for thr in (0.0, 0.10, 0.20, 0.50):
        n = sum(1 for r in ok if r["wer"] <= thr)
        print(f"  WER <= {thr:.2f}: {n}/{len(ok)} ({n/len(ok):.0%})")


if __name__ == "__main__":
    main()
