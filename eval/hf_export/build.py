"""HF dataset export pipeline — spec docs/specs/hf-dataset-export.md.

Stages (checkpointed, run in order):
  rows      fetch live export -> filter -> join speakers -> split
            -> data/hf-dataset/rows.parquet (+ reports)
  boundary  audio download + forced alignment + VAD per utterance
            -> data/hf-dataset/boundary.jsonl (resumable)
  finalize  merge boundary results -> train/validation parquet + stats + card

Usage:
  .venv-eval/bin/python -m eval.hf_export.build rows
  .venv-eval/bin/python -m eval.hf_export.build boundary [--limit-meetings N]
  .venv-eval/bin/python -m eval.hf_export.build finalize
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.exclusions import load_excluded_keys            # noqa: E402
from eval.hf_export import boundary as _bnd               # noqa: E402
from eval.hf_export.overlap import has_overlap_marker, notes_report_rows  # noqa: E402
from eval.hf_export.split import (DEFAULT_SEED, VAL_CITIES, SplitError,    # noqa: E402
                                  assign_splits)

EXPORT_URL = "https://79-76-114-184.sslip.io/api/export"
SPEAKERS_PARQUET = ROOT / "data" / "eval" / "speakers.parquet"
OUT = ROOT / "data" / "hf-dataset"
ROWS_PARQUET = OUT / "rows.parquet"
SPLIT_JSON = OUT / "split_assignments.json"
BOUNDARY_JSONL = OUT / "boundary.jsonl"
TEMPORAL_TEST_FROM = "2026-06-01"

# Bump when the boundary algorithm or its thresholds change: it is folded into
# row_sig so stale cached boundary results are ignored (Codex review #1).
BOUNDARY_ALGO_VERSION = "1"
_BOUNDARY_PARAMS = "|".join(str(x) for x in (
    BOUNDARY_ALGO_VERSION, _bnd.MARGIN_S, _bnd.EDGE_TOL_S, _bnd.BLEED_MIN_S,
    _bnd.PAD_S, _bnd.OK_SHIFT_S, _bnd.MIN_MEAN_SCORE))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_sig(r: dict) -> str:
    """Signature binding a boundary result to the exact span+text+algo it saw."""
    key = (f"{r['audio_url']}|{float(r['start']):.3f}|{float(r['end']):.3f}|"
           f"{r['final_after_text']}|{_BOUNDARY_PARAMS}")
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ---------- pure helpers (unit-tested) ----------

def filter_rows(export_rows: list[dict],
                excluded_keys: set[tuple[str, str]]) -> tuple[list[dict], dict]:
    """Keep publishable include rows. Returns (kept, drop_counts). No silent drops."""
    drops = collections.Counter()
    seen = set()
    kept = []
    for r in export_rows:
        key = (r["city_id"], r["meeting_id"], r["utterance_id"])
        if key in seen:
            raise ValueError(f"duplicate composite key {key}")
        seen.add(key)
        if r.get("include_status") != "include":
            drops["not_include"] += 1
            continue
        if (r["city_id"], r["meeting_id"]) in excluded_keys:
            drops["denylist"] += 1
            continue
        date = str(r.get("meeting_date") or "")
        if len(date) < 10 or not date[:4].isdigit():
            drops["bad_date"] += 1     # unknown date: cannot prove pre-June -> drop
            continue
        if date >= TEMPORAL_TEST_FROM:
            drops["temporal_test"] += 1
            continue
        try:
            s, e = float(r["start"]), float(r["end"])
        except (TypeError, ValueError):
            drops["bad_span"] += 1
            continue
        if not (e > s >= 0):
            drops["bad_span"] += 1
            continue
        r["error_categories"] = [str(c) for c in (r.get("error_categories") or [])]
        kept.append(r)

    # publication-safety invariants (Codex review): utterance_id must be
    # globally unique (joins + boundary keying rely on it), and each meeting
    # must map to exactly one audio_url (boundary stage decodes one file).
    uids = collections.Counter(r["utterance_id"] for r in kept)
    dups = [u for u, n in uids.items() if n > 1]
    if dups:
        raise ValueError(f"utterance_id not globally unique: {dups[:5]}")
    urls = collections.defaultdict(set)
    for r in kept:
        urls[(r["city_id"], r["meeting_id"])].add(r["audio_url"])
    multi = {k: v for k, v in urls.items() if len(v) > 1}
    if multi:
        raise ValueError(f"multiple audio_urls per meeting: {list(multi)[:3]}")
    return kept, dict(drops)


def join_speakers(rows: list[dict],
                  spk_by_utt: dict[str, str]) -> tuple[list[dict], int]:
    """Attach speaker_id (None if unknown). Returns (rows, n_missing)."""
    n_missing = 0
    for r in rows:
        spk = spk_by_utt.get(r["utterance_id"])
        if spk is None:
            n_missing += 1
        r["speaker_id"] = spk
    return rows, n_missing


def _as_list(v) -> list:
    """error_categories may arrive as a list (JSON) or numpy array (parquet);
    normalise to a list so `or []`-style truthiness never trips on arrays."""
    if v is None:
        return []
    return list(v)


def build_stats(rows: list[dict]) -> dict:
    """Aggregate hours/percentages/counters for stats.json (pure, tested)."""
    total_h = sum(r["duration_s"] for r in rows) / 3600
    by_split: dict[str, dict] = {}
    for split in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == split]
        h = sum(r["duration_s"] for r in sub) / 3600
        cats = collections.Counter(c for r in sub for c in _as_list(r["error_categories"]))
        by_split[split] = {
            "rows": len(sub),
            "hours": round(h, 2),
            "pct_hours": round(100 * h / total_h, 1) if total_h else 0.0,
            "cities": dict(collections.Counter(r["city_id"] for r in sub)),
            "speakers": len({r["speaker_id"] for r in sub if r.get("speaker_id")}),
            "error_categories": dict(cats.most_common()),
        }
    return {
        "total_rows": len(rows),
        "total_hours": round(total_h, 2),
        "by_split": by_split,
        "overlap_rows": sum(1 for r in rows if r.get("has_overlap")),
        "boundary_status": dict(collections.Counter(
            r.get("boundary_status") or "pending" for r in rows)),
    }


# ---------- rows stage ----------

def fetch_export(snapshot: Path) -> list[dict]:
    if snapshot.exists():
        log(f"using existing snapshot {snapshot.name}")
    else:
        log(f"fetching {EXPORT_URL} ...")
        req = urllib.request.Request(EXPORT_URL, headers={"User-Agent": "oc-hf-export/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            snapshot.write_bytes(resp.read())
        log(f"snapshot -> {snapshot} ({snapshot.stat().st_size/1e6:.1f} MB)")
    return [json.loads(l) for l in snapshot.open() if l.strip()]


def load_speaker_map() -> dict[str, str]:
    """utterance_id -> person_id. Fails on ambiguity (Codex review #13/#14)."""
    import pandas as pd
    df = pd.read_parquet(
        SPEAKERS_PARQUET,
        columns=["utterance_id", "city_id", "meeting_id", "person_id"])
    df = df.dropna(subset=["person_id"])
    dup = df[df.duplicated("utterance_id", keep=False)]
    conflicts = dup.groupby("utterance_id")["person_id"].nunique()
    if (conflicts > 1).any():
        raise ValueError(
            f"speakers.parquet: conflicting person_id for utterances "
            f"{list(conflicts[conflicts > 1].index[:5])}")
    return dict(zip(df["utterance_id"], df["person_id"]))


def stage_rows(args) -> None:
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%Y-%m-%d")
    snapshot = OUT / f"raw-export-{date}.jsonl"
    export_rows = fetch_export(snapshot)
    log(f"export rows: {len(export_rows)}")

    kept, drops = filter_rows(export_rows, load_excluded_keys())
    for k, v in sorted(drops.items()):
        log(f"  dropped {v}: {k}")
    log(f"kept include rows: {len(kept)}")

    kept, n_missing = join_speakers(kept, load_speaker_map())
    log(f"speaker join: {len(kept) - n_missing} matched, {n_missing} without "
        f"speaker_id (train-only, never validation)")

    for r in kept:
        r["duration_s"] = round(float(r["end"]) - float(r["start"]), 3)
        r["has_overlap"] = has_overlap_marker(r.get("reviewer_notes"))
        r["source"] = "correction"

    try:
        res = assign_splits(kept, seed=args.seed)
    except SplitError as ex:
        log(f"SPLIT GATE: {ex}")
        raise SystemExit(
            "validation share landed outside 18-22% of hours — this is the "
            "designed human gate, not a bug. Review the numbers above and "
            "decide (adjust window / seed / held-out cities) before proceeding."
        ) from ex
    for r, s in zip(kept, res.splits):
        r["split"] = s

    # overlap eyeball report
    matched, unmatched = notes_report_rows(kept)
    rep = ["# Overlap notes report", "",
           f"Rule: standalone Latin C in reviewer_notes. Matched {len(matched)}, "
           f"unmatched non-empty notes {len(unmatched)}.", "", "## Matched", ""]
    rep += [f"- `{r['utterance_id']}`: {r['reviewer_notes']!r}" for r in matched]
    rep += ["", "## Unmatched (non-empty)", ""]
    rep += [f"- `{r['utterance_id']}`: {r['reviewer_notes']!r}" for r in unmatched]
    (OUT / "overlap-notes-report.md").write_text("\n".join(rep) + "\n")

    hours = collections.defaultdict(float)
    for r in kept:
        hours[r["split"]] += r["duration_s"] / 3600
    # held-out-city vs seeded-speaker validation hours, reported separately so
    # the composition of the val set is auditable (Codex review #5)
    val_heldout_h = sum(r["duration_s"] for r in kept
                        if r["city_id"] in VAL_CITIES) / 3600
    val_seeded_h = hours["validation"] - val_heldout_h
    log(f"split: train {hours['train']:.2f}h / validation {hours['validation']:.2f}h "
        f"({res.val_share:.1%}) — held-out {val_heldout_h:.2f}h + seeded "
        f"{val_seeded_h:.2f}h; {len(res.val_speakers)} mixed-city val speakers, "
        f"{len(res.skipped_speakers)} skipped as overshoot")

    import numpy as np
    SPLIT_JSON.write_text(json.dumps({
        "seed": args.seed, "snapshot": snapshot.name,
        "snapshot_sha256": _sha256(snapshot),         # reproducibility (Codex #3)
        "val_cities": sorted(VAL_CITIES),
        "val_speakers": sorted(res.val_speakers),
        "skipped_speakers": res.skipped_speakers,
        "speaker_durations": {s: round(res.speaker_durations[s], 3)
                              for s in (list(res.val_speakers)
                                        + res.skipped_speakers)},
        "val_share_hours": round(res.val_share, 4),
        "val_heldout_hours": round(val_heldout_h, 3),
        "val_seeded_hours": round(val_seeded_h, 3),
        "temporal_test_from": TEMPORAL_TEST_FROM,
        "drop_counts": drops, "n_missing_speaker": n_missing,
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__},
    }, ensure_ascii=False, indent=2))

    cols = ["utterance_id", "city_id", "meeting_id", "meeting_date", "speaker_id",
            "audio_url", "start", "end", "duration_s", "initial_before_text",
            "final_after_text", "error_categories", "has_overlap", "source",
            "split", "reviewer_notes"]
    pd.DataFrame([{c: r.get(c) for c in cols} for r in kept]).to_parquet(ROWS_PARQUET)
    log(f"-> {ROWS_PARQUET} ({len(kept)} rows), {SPLIT_JSON}, overlap-notes-report.md")


# ---------- boundary stage ----------

def stage_boundary(args) -> None:
    import numpy as np
    import pandas as pd
    import torch
    from ctc_forced_aligner import (generate_emissions, get_alignments,
                                     get_spans, load_alignment_model,
                                     postprocess_results, preprocess_text)
    from silero_vad import get_speech_timestamps, load_silero_vad

    from eval.autoresearch.prepare_asr import SR, decode_pcm, download_mp3
    from eval.hf_export.boundary import MARGIN_S, classify_boundary

    df = pd.read_parquet(ROWS_PARQUET)
    # staleness guard (Codex #19/#1): a boundary line only counts as done if it
    # was computed for the SAME span+text+algo as the current rows.parquet
    expected = {r["utterance_id"]: _row_sig(r) for r in df.to_dict("records")}
    done = set()
    stale = 0
    if BOUNDARY_JSONL.exists():
        for l in BOUNDARY_JSONL.open():
            d = json.loads(l)
            if expected.get(d["utterance_id"]) == d.get("row_sig"):
                done.add(d["utterance_id"])
            else:
                stale += 1
        log(f"resume: {len(done)} already classified, {stale} stale lines ignored")

    align_model, tokenizer = load_alignment_model("cpu", dtype=torch.float32)
    vad_model = load_silero_vad()

    by_mtg = collections.defaultdict(list)
    for r in df.to_dict("records"):
        if r["utterance_id"] not in done:
            by_mtg[(r["city_id"], r["meeting_id"])].append(r)
    meetings = sorted(by_mtg)
    if args.limit_meetings:
        meetings = meetings[: args.limit_meetings]
        log(f"--limit-meetings {args.limit_meetings}: processing "
            f"{sum(len(by_mtg[m]) for m in meetings)} utterances")

    def _fail(r, note):
        return json.dumps({"utterance_id": r["utterance_id"],
                           "row_sig": _row_sig(r),
                           "boundary_status": "align_failed",
                           "start_adj": r["start"], "end_adj": r["end"],
                           "mean_score": 0.0, "note": note}) + "\n"

    out = BOUNDARY_JSONL.open("a")
    for mi, (city, meeting) in enumerate(meetings, 1):
        rows = by_mtg[(city, meeting)]
        mp3 = download_mp3(rows[0]["audio_url"], city, meeting)
        if mp3 is None:
            for r in rows:
                out.write(_fail(r, "audio download failed"))
            out.flush()
            continue
        pcm = decode_pcm(mp3)
        if pcm is None:
            for r in rows:
                out.write(_fail(r, "audio decode failed"))
            out.flush()
            continue
        total = len(pcm)
        n_ok = 0
        for r in rows:
            s, e = float(r["start"]), float(r["end"])
            a = max(0, int((s - MARGIN_S) * SR))
            b = min(total, int((e + MARGIN_S) * SR))
            clip = pcm[a:b]
            clip_dur = len(clip) / SR
            left_margin = s - a / SR  # actual margin (clamped near file start)
            try:
                wav = torch.from_numpy(np.ascontiguousarray(clip))
                emissions, stride = generate_emissions(align_model, wav, batch_size=4)
                tok_star, txt_star = preprocess_text(
                    r["final_after_text"], romanize=True, language="ell",
                    split_size="word", star_frequency="edges")
                segs, scores, blank = get_alignments(emissions, tok_star, tokenizer)
                spans = get_spans(tok_star, segs, blank)
                words = [w for w in postprocess_results(txt_star, spans, stride, scores)
                         if w.get("text") != "<star>"]
            except Exception as ex:  # noqa: BLE001 — any aligner failure = align_failed
                words = []
                log(f"  align FAIL {r['utterance_id']}: {type(ex).__name__} {str(ex)[:60]}")
            vad = get_speech_timestamps(clip, vad_model, sampling_rate=SR,
                                        return_seconds=True)
            res = classify_boundary(words, vad, raw_dur=e - s,
                                    clip_dur=clip_dur, margin_s=left_margin)
            out.write(json.dumps({
                "utterance_id": r["utterance_id"],
                "row_sig": _row_sig(r),
                "boundary_status": res.status,
                "start_adj": round(a / SR + res.start_off, 3),
                "end_adj": round(a / SR + res.end_off, 3),
                "mean_score": round(res.mean_score, 4),
            }) + "\n")
            n_ok += 1
        out.flush()
        del pcm
        log(f"[{mi}/{len(meetings)}] {city}/{meeting}: {n_ok} clips classified")
    out.close()
    log(f"-> {BOUNDARY_JSONL}")


# ---------- finalize stage ----------

def _load_boundary(expected: dict[str, str]) -> tuple[dict[str, dict], int]:
    """Merge boundary.jsonl keyed by utterance_id; ignore stale, error on
    conflicting duplicates (Codex review #2)."""
    bnd: dict[str, dict] = {}
    stale = 0
    if not BOUNDARY_JSONL.exists():
        return bnd, stale
    for l in BOUNDARY_JSONL.open():
        d = json.loads(l)
        uid = d["utterance_id"]
        if expected.get(uid) != d.get("row_sig"):
            stale += 1
            continue
        prev = bnd.get(uid)
        if prev is not None:
            same = all(prev.get(k) == d.get(k) for k in
                       ("boundary_status", "start_adj", "end_adj"))
            if not same:
                raise ValueError(
                    f"conflicting boundary results for {uid}: {prev} vs {d}")
        bnd[uid] = d
    return bnd, stale


def stage_finalize(args) -> None:
    import pandas as pd
    df = pd.read_parquet(ROWS_PARQUET)
    expected = {r["utterance_id"]: _row_sig(r) for r in df.to_dict("records")}
    bnd, stale = _load_boundary(expected)
    if stale:
        log(f"ignored {stale} stale boundary lines (row_sig mismatch)")
    missing = [u for u in df["utterance_id"] if u not in bnd]
    if missing and not args.allow_pending_boundary:
        raise SystemExit(
            f"{len(missing)} rows lack boundary results — finish the boundary "
            f"stage or pass --allow-pending-boundary (status becomes 'pending')")
    log(f"boundary results: {len(bnd)} matched, {len(missing)} pending")

    df["boundary_status"] = [
        bnd.get(u, {}).get("boundary_status", "pending") for u in df["utterance_id"]]
    df["start_adj"] = [bnd.get(u, {}).get("start_adj") for u in df["utterance_id"]]
    df["end_adj"] = [bnd.get(u, {}).get("end_adj") for u in df["utterance_id"]]

    # audit CSV: everything not ok/adjusted, for sampled human review
    audit = df[~df["boundary_status"].isin(["ok", "adjusted"])]
    audit_cols = ["utterance_id", "city_id", "meeting_id", "audio_url", "start",
                  "end", "start_adj", "end_adj", "boundary_status",
                  "final_after_text"]
    audit[audit_cols].to_csv(OUT / "boundary-audit.csv", index=False)
    log(f"boundary audit rows: {len(audit)} -> boundary-audit.csv")

    # adjusted spans are only trustworthy for ok/adjusted rows (Codex #10) —
    # in the PUBLISHED files suspects/failures get start_adj/end_adj = null so
    # nobody uses them blindly (the proposals stay in boundary-audit.csv above)
    trusted = df["boundary_status"].isin(["ok", "adjusted"])
    df.loc[~trusted, ["start_adj", "end_adj"]] = None
    # keep the adjusted-offset columns as nullable float, not object (Codex #7)
    df["start_adj"] = pd.to_numeric(df["start_adj"], errors="coerce")
    df["end_adj"] = pd.to_numeric(df["end_adj"], errors="coerce")

    # published columns — reviewer_notes stays internal (free text, PII risk)
    pub_cols = ["utterance_id", "city_id", "meeting_id", "meeting_date",
                "speaker_id", "audio_url", "start", "end", "start_adj",
                "end_adj", "duration_s", "boundary_status",
                "initial_before_text", "final_after_text", "error_categories",
                "has_overlap", "source", "split"]
    pub = df[pub_cols].rename(columns={"initial_before_text": "before_text",
                                       "final_after_text": "text"})
    # pre-publish lint (Codex #9): the published frame must contain exactly the
    # whitelist columns and never the internal free-text notes.
    expected_pub = [("before_text" if c == "initial_before_text"
                     else "text" if c == "final_after_text" else c)
                    for c in pub_cols]
    assert list(pub.columns) == expected_pub, list(pub.columns)
    assert "reviewer_notes" not in pub.columns
    # ONLY data/hf-dataset/public/ is ever uploaded to HF (Codex #15) — the
    # internal artifacts (reviewer-notes report, audit CSV, raw snapshot with
    # notes) live one level up and must never enter the published repo
    pub_dir = OUT / "public"
    pub_dir.mkdir(exist_ok=True)
    for split in ("train", "validation"):
        part = pub[pub["split"] == split].drop(columns=["split"])
        part.to_parquet(pub_dir / f"{split}.parquet", index=False)
        part.to_json(pub_dir / f"{split}.jsonl", orient="records", lines=True,
                     force_ascii=False)
        # roundtrip sanity: the file must read back with the same columns/count
        back = pd.read_parquet(pub_dir / f"{split}.parquet")
        assert len(back) == len(part) and list(back.columns) == list(part.columns)
        log(f"-> public/{split}.parquet / .jsonl ({len(part)} rows)")
    (pub_dir / "split_assignments.json").write_text(SPLIT_JSON.read_text())

    stats = build_stats(df.to_dict("records"))
    stats["versions"] = {"python": platform.python_version()}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    md = ["# HF dataset export — stats", "",
          f"Total: {stats['total_rows']} rows, {stats['total_hours']} h", ""]
    for split, s in stats["by_split"].items():
        md += [f"## {split}", "",
               f"- rows {s['rows']}, hours {s['hours']} ({s['pct_hours']}%), "
               f"speakers {s['speakers']}",
               f"- cities: {s['cities']}",
               f"- categories: {s['error_categories']}", ""]
    md += [f"overlap rows: {stats['overlap_rows']}",
           f"boundary status: {stats['boundary_status']}"]
    (OUT / "stats.md").write_text("\n".join(md) + "\n")
    log(f"stats: {stats['total_hours']}h total; "
        + "; ".join(f"{k} {v['hours']}h ({v['pct_hours']}%)"
                    for k, v in stats["by_split"].items()))
    write_dataset_card(stats)
    log("-> README.md (dataset card draft)")


def write_dataset_card(stats: dict) -> None:
    split_lines = "\n".join(
        f"| {k} | {v['rows']} | {v['hours']} h | {v['pct_hours']}% | {v['speakers']} |"
        for k, v in stats["by_split"].items())
    seed = (json.loads(SPLIT_JSON.read_text())["seed"]
            if SPLIT_JSON.exists() else "N/A")
    (OUT / "public" / "README.md").write_text(f"""---
language: [el]
task_categories: [automatic-speech-recognition]
pretty_name: OpenCouncil Greek municipal-council ASR corrections
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
---

# OpenCouncil Greek ASR corrections (train/validation)

Human-curated `(audio span, corrected text)` pairs from Greek municipal-council
recordings (opencouncil.gr), built for Whisper fine-tuning. This release is
**metadata-only**: each row carries the source `audio_url` plus `start`/`end`
offsets (and boundary-checked `start_adj`/`end_adj`); audio clips are not
embedded. A clip-embedded release may follow once licensing is confirmed.

| split | rows | hours | % hours | speakers |
|---|---|---|---|---|
{split_lines}

## Fields

- `before_text` — raw ASR output; `text` — the human-corrected target.
- `start`/`end` — raw utterance span in the meeting audio; `start_adj`/`end_adj`
  — spans snapped via forced alignment + VAD with ±0.2 s padding.
- `boundary_status` — `ok`/`adjusted` (usable as-is) vs `suspect_cut_start`/
  `suspect_cut_end`/`suspect_bleed_in`/`align_failed` (inspect before use;
  `start_adj`/`end_adj` are null for these). `suspect_bleed_in` is a
  *conservative* flag: VAD speech inside the span that the alignment did not
  cover — often neighbouring speech, but sometimes fillers/hesitations the
  corrected text omits.
- `has_overlap` — a reviewer marked overlapping speech (someone else audible)
  in this span. Boolean only; the overlapping speech is not transcribed.
- `error_categories` — reviewer labels for the correction type.

## Split methodology

Validation = two whole held-out cities (orestiada, argos) plus whole seeded
speakers (>= 3 min speech in-dataset) from the remaining cities up to ~20% of
total hours; speaker-disjoint from train. A temporal test set (meetings from
June 2026 on) is withheld entirely. Split map: `split_assignments.json`
(seed {seed}).

## Caveats

- Corrections were hand-curated during review (inclusion bias toward
  interesting errors); this is not a random sample of council speech.
- `meeting_id` slugs collide across cities — always key by
  `(city_id, meeting_id)`.
- Rows with `speaker_id = null` (~no diarization identity) are train-only.
- License: pending confirmation with OpenCouncil; audio remains at
  data.opencouncil.gr and is not redistributed here.
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p_rows = sub.add_parser("rows")
    p_rows.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_bnd = sub.add_parser("boundary")
    p_bnd.add_argument("--limit-meetings", type=int, default=0)
    p_fin = sub.add_parser("finalize")
    p_fin.add_argument("--allow-pending-boundary", action="store_true")
    args = ap.parse_args()
    if args.stage == "rows":
        stage_rows(args)
    elif args.stage == "boundary":
        stage_boundary(args)
    elif args.stage == "finalize":
        stage_finalize(args)


if __name__ == "__main__":
    main()
