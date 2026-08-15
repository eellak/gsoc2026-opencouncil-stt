"""HParl (Elormiden mirror) — MP3 conversion + Soniox alignment filter.

`Elormiden/Hellenic-greek-parliamentary-speech` is the same CLARIN HParl corpus as
`ddamianos/hparl` but processed for ML: 92,133 clips, transcripts are **accented**
(99.9% of rows) and carry no `[UNK]`. What they do carry is `<spoken_noise>` markers
on ~59% of rows — audio events the transcript does not spell out. Still lowercase and
unpunctuated. See docs/reports/2026-08-14-hparl-audio-text-probe.md for the earlier
mirror.

Pipeline, per row:

  1. read from a parquet shard over HTTP range requests (only touched row groups)
  2. decode the embedded audio to mono 16 kHz 32 kbps MP3 (~4 kB/s -> 120h = ~1.8 GB)
  3. transcribe with Soniox (independent second opinion, cached per row id)
  4. strip `<...>` tags from the transcript, align against Soniox on
     greek_normalize tokens, and KEEP the row when alignment >= --min-align
     (alignment = 1 - WER, so 0.95 means at most 5% token errors)

Errors are split by direction, because they mean different things:
  - ref_only : in the transcript, not heard  -> text without audio behind it
  - hyp_only : heard, not in the transcript  -> what `<spoken_noise>` was hiding
  - sub      : both sides have a word, they disagree

Audio and text stay under ~/.cache/oc-public/hparl2/ — never in git.

Run:  .venv-eval/bin/python eval/hparl2_filter.py --n 150
      .venv-eval/bin/python eval/hparl2_filter.py --shard data/train-00005-of-00022.parquet --n 300
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.scoring import greek_normalize  # noqa: E402

REPO = "Elormiden/Hellenic-greek-parliamentary-speech"
WORK = Path.home() / ".cache/oc-public/hparl2"
CLIPS = WORK / "clips"
ASR = WORK / "asr"

SONIOX_DIR = Path("/home/harold/projects/soniox-tools")
SONIOX_PY = SONIOX_DIR / ".venv" / "bin" / "python"
MARKER = "===== TRANSCRIPT ====="

TAG_RE = re.compile(r"<[^>]*>")


def strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", text or "")).strip()


# ---------- sampling ----------
def open_shard(shard: str):
    """ParquetFile over HTTP range requests — a shard is ~500 MB and we only need
    the row groups our sample falls in."""
    from huggingface_hub import HfFileSystem
    return pq.ParquetFile(HfFileSystem().open(f"datasets/{REPO}/{shard}", "rb"))


def _rows_from(tbl, shard: str, base: int, offs: list[int] | None = None) -> list[dict]:
    tag = shard.rsplit("/", 1)[-1].replace(".parquet", "")
    sel = tbl.take(offs) if offs is not None else tbl
    idxs = offs if offs is not None else range(tbl.num_rows)
    return [{
        "row_id": f"{tag}_{base + i:06d}",
        "src_path": r["audio"].get("path"),
        "transcription": r["transcription"],
        "audio_bytes": r["audio"]["bytes"],
    } for i, r in zip(idxs, sel.to_pylist())]


def iter_row_groups(shard: str):
    """Every row of a shard, one row group at a time (a shard is ~500 MB; holding
    all of its audio bytes at once is not worth it)."""
    pf = open_shard(shard)
    md = pf.metadata
    base = 0
    for g in range(md.num_row_groups):
        tbl = pf.read_row_group(g, columns=["audio", "transcription"])
        yield _rows_from(tbl, shard, base)
        base += tbl.num_rows


def sample_rows(shard: str, n: int, seed: int,
                skip_ids: set[str] | None = None) -> list[dict]:
    pf = open_shard(shard)
    md = pf.metadata
    rng = np.random.default_rng(seed)
    want = sorted(int(i) for i in
                  rng.choice(md.num_rows, size=min(n, md.num_rows), replace=False))

    # Row ids follow from the indices alone, so a resumed run can decide a shard is
    # already done from the parquet footer, without pulling ~460 MB of row groups.
    if skip_ids:
        tag = shard.rsplit("/", 1)[-1].replace(".parquet", "")
        if all(f"{tag}_{i:06d}" in skip_ids for i in want):
            return []

    bounds, acc = [], 0
    for g in range(md.num_row_groups):
        acc += md.row_group(g).num_rows
        bounds.append(acc)

    by_group: dict[int, list[int]] = {}
    for idx in want:
        g = int(np.searchsorted(bounds, idx, side="right"))
        by_group.setdefault(g, []).append(idx - (bounds[g - 1] if g else 0))

    out: list[dict] = []
    tag = shard.rsplit("/", 1)[-1].replace(".parquet", "")
    for g, offs in sorted(by_group.items()):
        tbl = pf.read_row_group(g, columns=["audio", "transcription"])
        for off, r in zip(offs, tbl.take(offs).to_pylist()):
            gidx = (bounds[g - 1] if g else 0) + off
            out.append({
                "row_id": f"{tag}_{gidx:06d}",
                "src_path": r["audio"].get("path"),
                "transcription": r["transcription"],
                "audio_bytes": r["audio"]["bytes"],
            })
    return out


def to_mp3(raw: bytes, out: Path) -> float:
    """Encoded audio bytes -> 32 kbps mono 16 kHz MP3. Returns duration (s)."""
    if not (out.exists() and out.stat().st_size > 0):
        tmp = out.with_suffix(".part")
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3", str(tmp)],
            input=raw, capture_output=True, timeout=120)
        if p.returncode != 0 or not tmp.exists():
            raise RuntimeError(f"ffmpeg rc={p.returncode}: {p.stderr[-300:]!r}")
        tmp.rename(out)
    return ffprobe_duration(out)


def ffprobe_duration(path: Path) -> float:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, timeout=30)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


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
    wer = (sub + dele + ins) / nref if nref else (1.0 if th else 0.0)
    return {
        "n_ref": nref, "n_hyp": len(th), "sub": sub, "del": dele, "ins": ins,
        "wer": wer, "align": max(0.0, 1.0 - wer),
        "ref_only": ref_only[:40], "hyp_only": hyp_only[:40],
    }


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="data/test-00000-of-00003.parquet",
                    help="one shard path, or a comma-separated list")
    ap.add_argument("--full", action="store_true",
                    help="every row of every listed shard (streams row groups), "
                         "instead of an --n row sample")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-align", type=float, default=0.95)
    ap.add_argument("--out", default=str(WORK / "filtered.jsonl"))
    ap.add_argument("--append", action="store_true",
                    help="append to --out and skip row_ids already in it")
    args = ap.parse_args()

    for d in (CLIPS, ASR):
        d.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    seen: set[str] = set()
    if args.append and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["row_id"])
            except Exception:
                pass

    shards = [s for s in args.shard.split(",") if s]

    def chunks():
        """(shard, rows) batches. --full walks every row group of every shard so a
        whole-corpus pass never holds more than one row group of audio in memory."""
        if args.full:
            for sh in shards:
                for grp in iter_row_groups(sh):
                    yield sh, grp
        else:
            for sh in shards:
                rows = sample_rows(sh, args.n, args.seed, skip_ids=seen)
                if not rows:
                    print(f'[filter] {sh}: already done, skipped', flush=True)
                    continue
                yield sh, rows

    def work(it: dict) -> dict:
        a = transcribe_cached(it["mp3"], it["dur"], it["row_id"])
        ref = strip_tags(it["transcription"])
        rec = {"row_id": it["row_id"], "src_path": it["src_path"],
               "dur": round(it["dur"], 2), "shard": it["shard"],
               "transcription_raw": it["transcription"], "ref": ref,
               "n_tags": len(TAG_RE.findall(it["transcription"] or "")),
               "soniox_text": a["soniox_text"], "asr_ok": a["asr_ok"],
               "error": a["error"], "mp3": str(it["mp3"])}
        if a["asr_ok"]:
            rec.update(compare(ref, a["soniox_text"]))
            rec["keep"] = rec["align"] >= args.min_align
        else:
            rec["keep"] = False
        return rec

    recs: list[dict] = []
    lock = threading.Lock()
    done = 0
    audio_s = 0.0
    t0 = time.time()
    with out_path.open("a" if args.append else "w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        for shard, rows in chunks():
            rows = [r for r in rows if r["row_id"] not in seen]
            if not rows:
                continue
            items = []
            for r in rows:
                mp3 = CLIPS / f"{r['row_id']}.mp3"
                try:
                    dur = to_mp3(r["audio_bytes"], mp3)
                except Exception as e:
                    print(f"  encode failed {r['row_id']}: {e}", flush=True)
                    continue
                items.append({**r, "shard": shard, "mp3": mp3, "dur": dur})
                r["audio_bytes"] = None      # release before the ASR wait
            audio_s += sum(i["dur"] for i in items)

            futs = {ex.submit(work, it): it["row_id"] for it in items}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"row_id": futs[fut], "asr_ok": False, "keep": False,
                           "error": f"worker: {type(e).__name__}: {e}"}
                with lock:
                    done += 1
                    recs.append(rec)
                    seen.add(rec["row_id"])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    if done % 25 == 0:
                        el = time.time() - t0
                        kept_so_far = sum(1 for r in recs if r.get("keep"))
                        print(f"[{done}] {el/3600:.1f}h {done/el*60:.0f} rows/min "
                              f"keep={kept_so_far/done:.0%} "
                              f"audio={audio_s/3600:.1f}h", flush=True)

    ok = [r for r in recs if r.get("asr_ok")]
    if not ok:
        print("[filter] nothing transcribed")
        return
    kept = [r for r in ok if r["keep"]]
    nref = sum(r["n_ref"] for r in ok)
    print(f"\n[filter] {len(ok)}/{len(recs)} transcribed -> {out_path}")
    print(f"  pooled WER {sum(r['sub']+r['del']+r['ins'] for r in ok)/nref:.3f} "
          f"(sub {sum(r['sub'] for r in ok)/nref:.3f} / "
          f"del {sum(r['del'] for r in ok)/nref:.3f} / "
          f"ins {sum(r['ins'] for r in ok)/nref:.3f})")
    print(f"  KEEP align>={args.min_align:.2f}: {len(kept)}/{len(ok)} "
          f"({len(kept)/len(ok):.0%}), "
          f"{sum(r['dur'] for r in kept)/60:.1f} min of {audio_s/60:.1f} min")
    tagged = [r for r in ok if r["n_tags"] > 0]
    untag = [r for r in ok if r["n_tags"] == 0]
    for label, grp in (("rows with <tags>", tagged), ("rows without tags", untag)):
        if grp:
            k = sum(1 for r in grp if r["keep"])
            print(f"    {label:20s} n={len(grp):4d} keep={k/len(grp):.0%}")
    for thr in (0.80, 0.90, 0.95, 1.00):
        k = sum(1 for r in ok if r["align"] >= thr)
        hrs = sum(r["dur"] for r in ok if r["align"] >= thr) / audio_s
        print(f"  align>={thr:.2f}: {k}/{len(ok)} ({k/len(ok):.0%}) "
              f"= {hrs:.0%} of audio")


if __name__ == "__main__":
    main()
