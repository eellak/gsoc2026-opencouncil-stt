"""Generic external-source Soniox alignment filter (STOMA, EuroSpeech-el, ...).

Same per-row pipeline as eval/hparl2_filter.py (that script stays as-is for the
running hparl2 pilot): parquet shard over HTTP range requests -> mono 16 kHz 32 kbps
MP3 -> cached Soniox transcription -> token alignment against the source transcript.
What differs per source is only the wiring in SOURCES: repo, columns, extras, cache
dir. Both current sources ship accented+punctuated+cased targets, so there is no
punctuation-repair stage; the training label (`final`) is the source text verbatim
and rows are directly consumable by scripts/build_training_pack.py.

Beyond hparl2, every row also records **boundary flags** (first/last reference token
missing from the ASR hypothesis). An edge deletion on an otherwise-passing row means
clipped audio or a clipped transcript — exactly the truncation poison the deletion
work is fighting — so `keep` requires clean edges as well as the alignment gate.

For eurospeech, rows carry the corpus's own `ds_wer`/`ds_cer` (their Whisper-Turbo
pipeline score). The pilot samples RANDOMLY and the summary reports our keep rate by
ds_wer bin, which is the calibration that decides whether their score may later
pre-select what gets sent to Soniox. Their score never admits a row by itself.

Run:  .venv-eval/bin/python eval/ext_filter.py --source stoma --n 150
      .venv-eval/bin/python eval/ext_filter.py --source eurospeech --n 30 --workers 8
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.scoring import greek_normalize                      # noqa: E402
from eval.hparl2_filter import compare, to_mp3, transcribe    # noqa: E402

OC_CACHE = Path.home() / ".cache/oc-public"

# Per-source wiring. `extras` maps output-field -> parquet-column.
SOURCES: dict[str, dict] = {
    "stoma": {
        "repo": "aangelakis/STOMA",
        "work": OC_CACHE / "stoma",
        "audio_col": "audio",
        "text_col": "text",
        "default_shards": [f"data/train-{i:05d}-of-00015.parquet" for i in range(15)],
        "extras": {"speaker_id": "speaker_id", "section": "section",
                   "session": "session"},
        # isolated read sentences; recorded per-row anyway via the heuristic
    },
    "eurospeech": {
        "repo": "disco-eth/EuroSpeech",
        "work": OC_CACHE / "eurospeech-el",
        "audio_col": "audio",
        "text_col": "human_transcript",
        # train split ONLY (their splits are session-disjoint; never consume
        # validation/test — they may serve as benchmarks later). Five spread-out
        # shards for the pilot; pass --shard for anything else.
        "default_shards": [f"greece/train-{i:05d}-of-00527.parquet"
                           for i in (0, 105, 210, 315, 420)],
        "extras": {"ds_wer": "wer", "ds_cer": "cer", "video_id": "video_id",
                   "transcript_id": "transcript_id",
                   "ds_dur": "duration_seconds"},
    },
    "cv": {
        # Common Voice Scripted Speech 26.0 el — LOCAL tar extract (CC0; the tar
        # needs an authenticated Mozilla Data Collective download, no HF mirror).
        # Policy: all community-validated rows EXCEPT the official dev/test splits,
        # which stay clean as potential third-party benchmarks. Original MP3 bytes
        # are kept as-is (no second lossy transcode).
        "kind": "local_cv",
        "root": OC_CACHE / "cv-el/cv-corpus-26.0-2026-06-12/el",
        "work": OC_CACHE / "cv-el",
        "extras": {"client_id": "client_id", "sentence_id": "sentence_id",
                   "sentence_domain": "sentence_domain",
                   "up_votes": "up_votes", "down_votes": "down_votes"},
    },
}

TERMINAL = ".;!?…»"          # Greek question mark is ';'

# Latin -> Greek homoglyphs. STOMA (at least) ships sentences whose first letter is
# a LATIN lookalike ('Tο', 'Nα', 'Aρχικά'), which breaks token equality after
# normalization and poisons both the alignment and the training label. Deterministic
# repair: map only inside tokens that also contain Greek letters, so genuine Latin
# words (code-switching) are never touched.
_HOMOGLYPHS = str.maketrans(
    "ABEZHIKMNOPTYXaeiknopstuvxy",
    "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧαεικνορστυνχγ")
_GREEK_RE = re.compile(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊϋΐΰς]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def fix_homoglyphs(text: str) -> str:
    def fix_token(tok: str) -> str:
        if _GREEK_RE.search(tok) and _LATIN_RE.search(tok):
            return tok.translate(_HOMOGLYPHS)
        return tok
    return " ".join(fix_token(t) for t in (text or "").split())


def complete_sentence(text: str) -> bool:
    t = (text or "").strip()
    if not t or t[-1] not in TERMINAL:
        return False
    first = next((c for c in t if c.isalpha()), "")
    return first.isupper() if first else False


def edge_flags(ref: str, hyp: str) -> dict:
    """Is the first/last reference token matched by the hypothesis? A miss on an
    edge means clipped audio or transcript, independent of the aggregate score."""
    tr, th = greek_normalize(ref).split(), greek_normalize(hyp).split()
    if not tr:
        return {"first_ref_missing": False, "last_ref_missing": False}
    matched: set[int] = set()
    sm = difflib.SequenceMatcher(a=tr, b=th, autojunk=False)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            matched.update(range(i1, i2))
    return {"first_ref_missing": 0 not in matched,
            "last_ref_missing": (len(tr) - 1) not in matched}


# ---------- parquet access ----------

def open_shard(repo: str, shard: str):
    from huggingface_hub import HfFileSystem
    # skip_instance_cache: the fsspec-cached singleton's httpx client has been
    # observed closed after ~2h of a long run (killed both the stoma and the
    # eurospeech pass mid-flight). A fresh instance per shard bounds the blast
    # radius to one shard.
    fs = HfFileSystem(skip_instance_cache=True)
    return pq.ParquetFile(fs.open(f"datasets/{repo}/{shard}", "rb"))


def canonical_rows(pf, shard: str, cfg: dict, base: int,
                   offs: list[int] | None = None, group: int | None = None) -> list[dict]:
    """Canonical records from one row group (or a whole small table)."""
    tag = shard.rsplit("/", 1)[-1].replace(".parquet", "")
    cols = [cfg["audio_col"], cfg["text_col"]] + sorted(set(cfg["extras"].values()))
    tbl = pf.read_row_group(group, columns=cols) if group is not None \
        else pf.read(columns=cols)
    sel = tbl.take(offs) if offs is not None else tbl
    idxs = offs if offs is not None else range(tbl.num_rows)
    out = []
    for i, r in zip(idxs, sel.to_pylist()):
        audio = r[cfg["audio_col"]]
        out.append({
            "row_id": f"{tag}_{base + i:06d}",
            "src_path": (audio or {}).get("path"),
            "transcription": r[cfg["text_col"]],
            "audio_bytes": (audio or {}).get("bytes"),
            "extras": {k: r[c] for k, c in cfg["extras"].items()},
        })
    return out


def sample_indices(num_rows: int, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in
                  rng.choice(num_rows, size=min(n, num_rows), replace=False))


def preselect_indices(pf, cfg: dict, n: int, seed: int, thr: float,
                      explore_frac: float) -> list[int]:
    """Indices worth a Soniox call: rows the source itself scores <= thr, plus an
    exploration lane from the mediocre band (thr, 0.35] so the eventual pack is not
    exclusively source-model-easy audio. The source score NEVER admits a row."""
    col = cfg["extras"].get("ds_wer", "wer")
    scores = pf.read(columns=[col]).column(0).to_pylist()
    good = [i for i, s in enumerate(scores) if s is not None and s <= thr]
    band = [i for i, s in enumerate(scores) if s is not None and thr < s <= 0.35]
    rng = np.random.default_rng(seed)
    n_exp = min(len(band), int(n * explore_frac))
    n_good = min(len(good), n - n_exp)
    pick = ([good[i] for i in rng.choice(len(good), n_good, replace=False)]
            if n_good < len(good) else good)
    pick += ([band[i] for i in rng.choice(len(band), n_exp, replace=False)]
             if n_exp < len(band) else band)
    return sorted(int(i) for i in pick)


def sample_rows(cfg: dict, shard: str, n: int, seed: int,
                skip_ids: set[str] | None = None,
                preselect: float | None = None,
                explore_frac: float = 0.18) -> list[dict]:
    pf = open_shard(cfg["repo"], shard)
    md = pf.metadata
    if preselect is not None:
        want = preselect_indices(pf, cfg, n, seed, preselect, explore_frac)
    else:
        want = sample_indices(md.num_rows, n, seed)
    if not want:
        return []

    tag = shard.rsplit("/", 1)[-1].replace(".parquet", "")
    if skip_ids and all(f"{tag}_{i:06d}" in skip_ids for i in want):
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
    for g, offs in sorted(by_group.items()):
        base = bounds[g - 1] if g else 0
        out.extend(canonical_rows(pf, shard, cfg, base, offs=offs, group=g))
    return out


# ---------- Common Voice local adapter ----------

def _read_tsv(path: Path) -> list[dict]:
    import csv
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def cv_rows(cfg: dict) -> list[dict]:
    """validated.tsv minus dev/test, with durations from clip_durations.tsv.
    Rows come back in file order; sampling happens on top with sample_indices."""
    root: Path = cfg["root"]
    held_out = {r["path"] for split in ("dev.tsv", "test.tsv")
                for r in _read_tsv(root / split)}
    dur_ms = {r["clip"]: float(r["duration[ms]"])
              for r in _read_tsv(root / "clip_durations.tsv")}
    out = []
    for r in _read_tsv(root / "validated.tsv"):
        if r["path"] in held_out:
            continue
        mp3 = root / "clips" / r["path"]
        if not mp3.exists():
            continue
        out.append({
            "row_id": Path(r["path"]).stem,
            "src_path": r["path"],
            "transcription": r["sentence"],
            "dur": dur_ms.get(r["path"], 0.0) / 1000.0,
            "mp3_src": mp3,
            "extras": {k: r.get(c) for k, c in cfg["extras"].items()},
        })
    return out


# ---------- ASR ----------

def transcribe_cached(asr_dir: Path, mp3: Path, dur: float, key: str) -> dict:
    jp = asr_dir / f"{key}.json"
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


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--shard", default=None,
                    help="comma-separated shard list; default: source's pilot set")
    ap.add_argument("--n", type=int, default=30, help="rows sampled PER SHARD")
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-align", type=float, default=0.95)
    ap.add_argument("--preselect-ds-wer", type=float, default=None,
                    help="only send rows the source scores <= this to Soniox "
                         "(plus an exploration lane); admission is still our gate")
    ap.add_argument("--explore-frac", type=float, default=0.18)
    ap.add_argument("--out", default=None)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    cfg = SOURCES[args.source]
    work: Path = cfg["work"]
    clips, asr_dir = work / "clips", work / "asr"
    for d in (clips, asr_dir):
        d.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else work / "filtered.jsonl"

    seen: set[str] = set()
    if args.append and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["row_id"])
            except Exception:
                pass

    shards = ([s for s in args.shard.split(",") if s] if args.shard
              else cfg.get("default_shards", []))

    def work_one(it: dict) -> dict:
        a = transcribe_cached(asr_dir, it["mp3"], it["dur"], it["row_id"])
        ref = fix_homoglyphs(re.sub(r"\s+", " ", it["transcription"] or "").strip())
        rec = {"row_id": it["row_id"], "src_path": it["src_path"],
               "dur": round(it["dur"], 2), "shard": it["shard"],
               "ref": ref, "final": ref,        # no repair stage: label = source text
               "complete": complete_sentence(ref),
               "soniox_text": a["soniox_text"], "asr_ok": a["asr_ok"],
               "error": a["error"], "mp3": str(it["mp3"]), **it["extras"]}
        if a["asr_ok"]:
            rec.update(compare(ref, a["soniox_text"]))
            rec.update(edge_flags(ref, a["soniox_text"]))
            rec["keep"] = (rec["align"] >= args.min_align
                           and not rec["first_ref_missing"]
                           and not rec["last_ref_missing"])
        else:
            rec["keep"] = False
        return rec

    def batches():
        if cfg.get("kind") == "local_cv":
            allr = cv_rows(cfg)
            picked = [allr[i] for i in sample_indices(len(allr), args.n, args.seed)]
            yield "local", picked
        else:
            for shard in shards:
                yield shard, sample_rows(cfg, shard, args.n, args.seed,
                                         skip_ids=seen,
                                         preselect=args.preselect_ds_wer,
                                         explore_frac=args.explore_frac)

    recs: list[dict] = []
    lock = threading.Lock()
    done, audio_s, t0 = 0, 0.0, time.time()
    with out_path.open("a" if args.append else "w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        for shard, rows in batches():
            rows = [r for r in rows if r["row_id"] not in seen]
            if not rows:
                print(f"[filter] {shard}: already done, skipped", flush=True)
                continue
            items = []
            for r in rows:
                if r.get("mp3_src"):           # local source: keep original bytes
                    items.append({**r, "shard": shard, "mp3": r["mp3_src"]})
                    continue
                mp3 = clips / f"{r['row_id']}.mp3"
                try:
                    dur = to_mp3(r["audio_bytes"], mp3)
                except Exception as e:
                    print(f"  encode failed {r['row_id']}: {e}", flush=True)
                    continue
                items.append({**r, "shard": shard, "mp3": mp3, "dur": dur})
                r["audio_bytes"] = None
            audio_s += sum(i["dur"] for i in items)

            futs = {ex.submit(work_one, it): it["row_id"] for it in items}
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
                        kept = sum(1 for r in recs if r.get("keep"))
                        print(f"[{done}] {el/3600:.1f}h {done/el*60:.0f} rows/min "
                              f"keep={kept/done:.0%} audio={audio_s/3600:.1f}h",
                              flush=True)

    summarize(args, recs, audio_s, out_path)


def summarize(args, recs: list[dict], audio_s: float, out_path: Path) -> None:
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
    edge = sum(1 for r in ok if r.get("first_ref_missing") or r.get("last_ref_missing"))
    print(f"  KEEP (align>={args.min_align:.2f} + clean edges): {len(kept)}/{len(ok)} "
          f"({len(kept)/len(ok):.0%}), edge-flagged {edge}/{len(ok)}")
    print(f"  complete sentences: "
          f"{sum(1 for r in ok if r.get('complete'))}/{len(ok)}")
    for thr in (0.80, 0.90, 0.95, 1.00):
        k = sum(1 for r in ok if r["align"] >= thr)
        hrs = sum(r["dur"] for r in ok if r["align"] >= thr) / audio_s if audio_s else 0
        print(f"  align>={thr:.2f}: {k}/{len(ok)} ({k/len(ok):.0%}) = {hrs:.0%} of audio")

    if args.source == "eurospeech":
        print("  keep rate by source ds_wer bin (calibration):")
        bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 9.9)]
        for lo, hi in bins:
            grp = [r for r in ok if r.get("ds_wer") is not None
                   and lo <= r["ds_wer"] < hi]
            if grp:
                k = sum(1 for r in grp if r["keep"])
                print(f"    ds_wer [{lo:.2f},{hi:.2f}): n={len(grp):4d} "
                      f"keep={k/len(grp):.0%}")
    if args.source == "stoma":
        by_spk: dict[str, list] = {}
        for r in ok:
            by_spk.setdefault(r.get("speaker_id") or "?", []).append(r)
        for spk, grp in sorted(by_spk.items()):
            k = sum(1 for r in grp if r["keep"])
            print(f"    speaker {spk:3s} n={len(grp):4d} keep={k/len(grp):.0%}")


if __name__ == "__main__":
    main()
