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
from eval.hf_export.sources import dedupe_by_span, slug_date  # noqa: E402
from eval.hf_export.split import (DEFAULT_SEED, VAL_CITIES, SplitError,    # noqa: E402
                                  assign_splits)

EXPORT_URL = "https://79-76-114-184.sslip.io/api/export"
SPEAKERS_PARQUET = ROOT / "data" / "eval" / "speakers.parquet"
OUT = ROOT / "data" / "hf-dataset"
ROWS_PARQUET = OUT / "rows.parquet"
SPLIT_JSON = OUT / "split_assignments.json"
BOUNDARY_JSONL = OUT / "boundary.jsonl"
TEMPORAL_TEST_FROM = "2026-06-01"

# --- extra sources (combined dataset) ---
LEFTOVER_IDS = ROOT / "data" / "next-batch" / "final_audio" / "nb2audio_ids.json"
SELECTED_EDITS = ROOT / "data" / "next-batch" / "selected_edits.jsonl"
TRUST_TSV = ROOT / "data" / "reports" / "meeting-edit-fraction" / "distribution.tsv"
TRUST_FRAC = 0.15               # frac_user gate: "review-exposed" meeting
BACKBONE_TARGET = 20000         # ~20k trusted no-edit utterances
BACKBONE_CAP_TRAIN = 2200       # per train-city cap
BACKBONE_CAP_HELDOUT = 1200     # smaller cap on held-out cities (all -> val)
BACKBONE_DUR = (1.0, 15.0)      # utterance duration window (seconds)
# _rank for span-dedupe: include < leftover < backbone (lower wins)
_RANK = {"include": 0, "leftover": 1, "backbone": 2}

# Bump when the boundary algorithm or its thresholds change: it is folded into
# row_sig so stale cached boundary results are ignored (Codex review #1).
BOUNDARY_ALGO_VERSION = "1"
_BOUNDARY_PARAMS = "|".join(str(x) for x in (
    BOUNDARY_ALGO_VERSION, _bnd.MARGIN_S, _bnd.EDGE_TOL_S, _bnd.BLEED_MIN_S,
    _bnd.PAD_S, _bnd.OK_SHIFT_S, _bnd.MIN_MEAN_SCORE))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _boundary_files() -> list[Path]:
    """Legacy single-writer file + any sharded part files (Codex: merge set)."""
    return ([BOUNDARY_JSONL] if BOUNDARY_JSONL.exists() else []) + \
        sorted(OUT.glob("boundary.part-*.jsonl"))


def _iter_boundary_lines(path: Path):
    """Yield parsed JSON records, tolerating a truncated trailing line."""
    for l in path.open():
        l = l.strip()
        if not l:
            continue
        try:
            yield json.loads(l)
        except json.JSONDecodeError:
            log(f"  skip malformed line in {path.name}")


def _load_aligner(threads: int):
    """(model, tokenizer). threads>0 -> capped ONNX session for parallel shards."""
    if threads <= 0:
        from ctc_forced_aligner import AlignmentSingleton
        a = AlignmentSingleton()
        return a.model, a.tokenizer
    import os

    import onnxruntime
    from ctc_forced_aligner import MODEL_URL, Tokenizer, ensure_onnx_model
    mp = os.path.join(os.path.expanduser("~"), "ctc_forced_aligner", "model.onnx")
    ensure_onnx_model(mp, MODEL_URL)
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(mp, sess_options=so), Tokenizer()


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
            "sources": dict(collections.Counter(
                r.get("source") or "correction" for r in sub)),
            "error_categories": dict(cats.most_common()),
        }
    by_source: dict[str, dict] = {}
    for source in sorted({r.get("source") or "correction" for r in rows}):
        sub = [r for r in rows if (r.get("source") or "correction") == source]
        by_source[source] = {
            "rows": len(sub),
            "hours": round(sum(r["duration_s"] for r in sub) / 3600, 2),
            "by_split": dict(collections.Counter(r["split"] for r in sub)),
        }
    return {
        "total_rows": len(rows),
        "total_hours": round(total_h, 2),
        "by_split": by_split,
        "by_source": by_source,
        "overlap_rows": sum(1 for r in rows if r.get("has_overlap")),
        "boundary_status": dict(collections.Counter(
            r.get("boundary_status") or "pending" for r in rows)),
    }


# ---------- extra-source loaders ----------

def _iso_to_date(dt: str | None) -> str | None:
    """'2025-12-09T14:00:00.000Z' -> '2025-12-09 14:00:00'."""
    if not dt or len(dt) < 19:
        return None
    return dt[:10] + " " + dt[11:19]


def build_date_map(export_rows: list[dict]) -> dict:
    return {(r["city_id"], r["meeting_id"]): r.get("meeting_date")
            for r in export_rows}


def _date_for(city: str, meeting: str, date_map: dict) -> str | None:
    d = date_map.get((city, meeting))
    if d and str(d)[:4].isdigit():
        return str(d)
    return slug_date(meeting)


def _filter_extra(rows: list[dict],
                  excluded_keys: set) -> tuple[list[dict], dict]:
    """Denylist + temporal + bad date/span guards for non-export sources
    (the live export already went through filter_rows). No silent drops."""
    drops = collections.Counter()
    out = []
    for r in rows:
        if (r["city_id"], r["meeting_id"]) in excluded_keys:
            drops["denylist"] += 1
            continue
        date = str(r.get("meeting_date") or "")
        if len(date) < 10 or not date[:4].isdigit():
            drops["bad_date"] += 1
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
        out.append(r)
    return out, dict(drops)


def load_leftover_corrections(exclude_uids: set, date_map: dict) -> list[dict]:
    """NB2 audio-verified balanced leftover edits -> export-schema rows."""
    if not (LEFTOVER_IDS.exists() and SELECTED_EDITS.exists()):
        log("leftover source missing -> 0 leftover rows")
        return []
    ids = set(json.loads(LEFTOVER_IDS.read_text()))
    rows, seen = [], set()
    for l in SELECTED_EDITS.open():
        d = json.loads(l)
        uid = d["utterance_id"]
        if uid not in ids or uid in exclude_uids or uid in seen:
            continue
        seen.add(uid)
        rows.append({
            "utterance_id": uid, "city_id": d["city_id"],
            "meeting_id": d["meeting_id"],
            "meeting_date": _date_for(d["city_id"], d["meeting_id"], date_map),
            "audio_url": d["audio_url"],
            "start": float(d["utterance_start"]), "end": float(d["utterance_end"]),
            "initial_before_text": d.get("input_raw") or "",
            "final_after_text": d.get("gold_final") or "",
            "error_categories": [str(d["category"])] if d.get("category") else [],
            "reviewer_notes": None, "source": "correction",
            "_rank": _RANK["leftover"],
        })
    return rows


def load_review_exposed_meetings(excluded_keys: set) -> set:
    """(city, meeting) with frac_user >= TRUST_FRAC or humanReview, not denylisted.

    NB: this is a *review-exposed* gate (the meeting saw human editing), not a
    proof every no-edit utterance was individually verified — documented in the
    dataset card as the backbone's known limitation."""
    import csv
    trusted = set()
    with TRUST_TSV.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (r["city"], r["meeting"])
            if key in excluded_keys:
                continue
            try:
                frac = float(r["frac_user"])
            except (TypeError, ValueError):
                frac = 0.0
            if frac >= TRUST_FRAC or r["humanReview"].strip().lower() == "true":
                trusted.add(key)
    return trusted


def load_backbone(trusted_meetings: set, spk_by_utt: dict, date_map: dict, *,
                  seed: int) -> list[dict]:
    """Trusted no-edit ASR utterances, per-city capped to ~BACKBONE_TARGET.

    Text is the uncorrected ASR from the meeting JSON (before_text == text).
    Rows without a speaker_id are skipped (can't guarantee split disjointness)."""
    import numpy as np

    from eval.autoresearch.prepare_asr import fetch_meeting_json, noedit_utts
    lo, hi = BACKBONE_DUR
    by_city: dict[str, list] = collections.defaultdict(list)
    n_meet = 0
    for (city, meeting) in sorted(trusted_meetings):
        d = fetch_meeting_json(city, meeting)
        if d is None:
            continue
        mt = d.get("meeting") or {}
        audio_url = mt.get("audioUrl")
        if not audio_url:
            continue
        n_meet += 1
        mdate = _iso_to_date(mt.get("dateTime")) or _date_for(city, meeting, date_map)
        for uid, s, e, txt in noedit_utts(d):
            if not (lo <= (e - s) <= hi):
                continue
            if not spk_by_utt.get(uid):
                continue
            by_city[city].append({
                "utterance_id": uid, "city_id": city, "meeting_id": meeting,
                "meeting_date": mdate, "audio_url": audio_url,
                "start": float(s), "end": float(e),
                "initial_before_text": txt, "final_after_text": txt,
                "error_categories": [], "reviewer_notes": None,
                "source": "no_edit", "_rank": _RANK["backbone"],
            })
    log(f"backbone: {n_meet} review-exposed meetings, "
        f"{sum(len(v) for v in by_city.values())} candidate no-edit utts")
    rng = np.random.default_rng(seed)
    out = []
    for city in sorted(by_city):
        cand = by_city[city]
        cap = BACKBONE_CAP_HELDOUT if city in VAL_CITIES else BACKBONE_CAP_TRAIN
        idx = np.arange(len(cand))
        rng.shuffle(idx)
        out.extend(cand[i] for i in idx[:cap])
    return out


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


def _normalize_meeting_urls(rows: list[dict]) -> int:
    """One audio_url per (city, meeting) — prefer a correction row's URL so the
    boundary stage downloads a single mp3 per meeting. Returns #rows rewritten."""
    best_rank: dict[tuple, int] = {}
    pref: dict[tuple, str] = {}
    for r in rows:
        key = (r["city_id"], r["meeting_id"])
        if key not in best_rank or r["_rank"] < best_rank[key]:
            best_rank[key] = r["_rank"]      # include < leftover < backbone
            pref[key] = r["audio_url"]
    changed = 0
    for r in rows:
        u = pref[(r["city_id"], r["meeting_id"])]
        if r["audio_url"] != u:
            r["audio_url"] = u
            changed += 1
    return changed


def stage_rows(args) -> None:
    import numpy as np
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%Y-%m-%d")
    snapshot = OUT / f"raw-export-{date}.jsonl"
    export_rows = fetch_export(snapshot)
    log(f"export rows: {len(export_rows)}")
    excluded = load_excluded_keys()
    date_map = build_date_map(export_rows)
    spk_map = load_speaker_map()

    # 1) includes (corrections) from the live export
    kept, drops = filter_rows(export_rows, excluded)
    for k, v in sorted(drops.items()):
        log(f"  export dropped {v}: {k}")
    for r in kept:
        r["source"] = "correction"
        r["_rank"] = _RANK["include"]
    include_uids = {r["utterance_id"] for r in kept}
    log(f"includes kept: {len(kept)}")

    # 2) leftover corrections (NB2 audio-verified, not already included)
    leftover, ld = _filter_extra(
        load_leftover_corrections(include_uids, date_map), excluded)
    log(f"leftover corrections: {len(leftover)} (drops {ld})")

    # 3) no-edit backbone (trusted review-exposed meetings)
    trusted = load_review_exposed_meetings(excluded)
    backbone, bd = _filter_extra(
        load_backbone(trusted, spk_map, date_map, seed=args.seed), excluded)
    log(f"backbone no-edit: {len(backbone)} (drops {bd})")

    # combine + span-dedupe (correction beats no_edit; include beats leftover)
    combined = kept + leftover + backbone
    n_url = _normalize_meeting_urls(combined)
    combined, n_dup = dedupe_by_span(combined)
    log(f"combined {len(kept)}+{len(leftover)}+{len(backbone)} -> {len(combined)} "
        f"after span-dedupe ({n_dup} dropped; {n_url} audio_urls normalized)")

    # global utterance_id uniqueness across all sources (join/boundary key)
    uids = collections.Counter(r["utterance_id"] for r in combined)
    dup_uids = [u for u, n in uids.items() if n > 1]
    if dup_uids:
        raise ValueError(f"utterance_id not globally unique: {dup_uids[:5]}")

    combined, n_missing = join_speakers(combined, spk_map)
    log(f"speaker join: {len(combined) - n_missing} matched, {n_missing} without "
        f"speaker_id (train-only, never validation)")

    for r in combined:
        r["duration_s"] = round(float(r["end"]) - float(r["start"]), 3)
        r["has_overlap"] = has_overlap_marker(r.get("reviewer_notes"))

    try:
        res = assign_splits(combined, seed=args.seed)
    except SplitError as ex:
        log(f"SPLIT GATE: {ex}")
        raise SystemExit(
            "validation share landed outside 18-22% of hours — this is the "
            "designed human gate, not a bug. Review the numbers above and "
            "decide (adjust caps / window / seed / held-out cities) first."
        ) from ex
    for r, s in zip(combined, res.splits):
        r["split"] = s

    # overlap eyeball report (only correction rows carry reviewer_notes)
    matched, unmatched = notes_report_rows(combined)
    rep = ["# Overlap notes report", "",
           f"Rule: standalone Latin C in reviewer_notes. Matched {len(matched)}, "
           f"unmatched non-empty notes {len(unmatched)}.", "", "## Matched", ""]
    rep += [f"- `{r['utterance_id']}`: {r['reviewer_notes']!r}" for r in matched]
    rep += ["", "## Unmatched (non-empty)", ""]
    rep += [f"- `{r['utterance_id']}`: {r['reviewer_notes']!r}" for r in unmatched]
    (OUT / "overlap-notes-report.md").write_text("\n".join(rep) + "\n")

    hours = collections.defaultdict(float)
    src = collections.Counter()
    for r in combined:
        hours[r["split"]] += r["duration_s"] / 3600
        src[(r["source"], r["split"])] += 1
    val_heldout_h = sum(r["duration_s"] for r in combined
                        if r["city_id"] in VAL_CITIES) / 3600
    val_seeded_h = hours["validation"] - val_heldout_h
    log(f"split: train {hours['train']:.2f}h / validation {hours['validation']:.2f}h "
        f"({res.val_share:.1%}) — held-out {val_heldout_h:.2f}h + seeded "
        f"{val_seeded_h:.2f}h; {len(res.val_speakers)} seeded val speakers, "
        f"{len(res.skipped_speakers)} skipped")
    log(f"source x split: {dict(src)}")

    def _h(p):
        return _sha256(p) if p.exists() else None
    SPLIT_JSON.write_text(json.dumps({
        "seed": args.seed, "snapshot": snapshot.name,
        "snapshot_sha256": _sha256(snapshot),
        "sources": {
            "includes": len(kept), "leftover": len(leftover),
            "backbone": len(backbone), "combined_after_dedupe": len(combined),
            "span_dupes_dropped": n_dup,
        },
        "input_hashes": {
            "nb2audio_ids": _h(LEFTOVER_IDS),
            "selected_edits": _h(SELECTED_EDITS),
            "trust_tsv": _h(TRUST_TSV),
        },
        "backbone_config": {
            "trust_frac": TRUST_FRAC, "dur_window": list(BACKBONE_DUR),
            "cap_train": BACKBONE_CAP_TRAIN, "cap_heldout": BACKBONE_CAP_HELDOUT,
            "review_exposed_meetings": len(trusted),
        },
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
        "drop_counts": {"export": drops, "leftover": ld, "backbone": bd},
        "n_missing_speaker": n_missing,
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__},
    }, ensure_ascii=False, indent=2))

    cols = ["utterance_id", "city_id", "meeting_id", "meeting_date", "speaker_id",
            "audio_url", "start", "end", "duration_s", "initial_before_text",
            "final_after_text", "error_categories", "has_overlap", "source",
            "split", "reviewer_notes"]
    pd.DataFrame([{c: r.get(c) for c in cols} for r in combined]).to_parquet(ROWS_PARQUET)
    log(f"-> {ROWS_PARQUET} ({len(combined)} rows), {SPLIT_JSON}, overlap-notes-report.md")


# ---------- boundary stage ----------

def stage_boundary(args) -> None:
    import math

    import numpy as np
    import pandas as pd
    import torch
    # NOTE: the PyPI `ctc-forced-aligner` (deskpai, ONNX) is used, NOT
    # MahmoudAshraf's torch package the plan first named — same MMS CTC pipeline
    # (preprocess_text/generate_emissions/get_alignments/get_spans/
    # postprocess_results, uroman romanization for Greek), reached here because
    # the git-installed package is not available and the spec leaves the aligner
    # open. AlignmentSingleton auto-downloads the ONNX model on first use.
    from ctc_forced_aligner import (generate_emissions, get_alignments,
                                     get_spans, postprocess_results,
                                     preprocess_text)
    from silero_vad import get_speech_timestamps, load_silero_vad

    from eval.autoresearch.prepare_asr import SR, decode_pcm, download_mp3
    from eval.hf_export.boundary import MARGIN_S, classify_boundary

    df = pd.read_parquet(ROWS_PARQUET)
    # staleness guard (Codex #19/#1): a boundary line only counts as done if it
    # was computed for the SAME span+text+algo as the current rows.parquet.
    # Done-set spans the legacy file AND every shard part file (Codex: workers
    # must skip clips finished by any worker).
    expected = {r["utterance_id"]: _row_sig(r) for r in df.to_dict("records")}
    done = set()
    stale = 0
    for path in _boundary_files():
        for d in _iter_boundary_lines(path):
            if expected.get(d["utterance_id"]) == d.get("row_sig"):
                done.add(d["utterance_id"])
            else:
                stale += 1
    log(f"resume: {len(done)} already classified, {stale} stale lines ignored")

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    align_model, tokenizer = _load_aligner(args.threads)
    vad_model = load_silero_vad()

    by_mtg = collections.defaultdict(list)
    for r in df.to_dict("records"):
        if r["utterance_id"] not in done:
            by_mtg[(r["city_id"], r["meeting_id"])].append(r)
    meetings = sorted(by_mtg)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        meetings = meetings[k::n]      # deterministic meeting partition
        log(f"--shard {args.shard}: {len(meetings)} meetings, "
            f"{sum(len(by_mtg[m]) for m in meetings)} utterances")
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

    out_path = OUT / args.out if args.out else BOUNDARY_JSONL
    out = out_path.open("a")
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
                wav = np.ascontiguousarray(clip, dtype=np.float32)  # ONNX: 1D numpy
                emissions, stride = generate_emissions(
                    align_model, wav, window_length=args.window, batch_size=4)
                tok_star, txt_star = preprocess_text(
                    r["final_after_text"], romanize=True, language="ell",
                    split_size="word", star_frequency="edges")
                segs, scores, blank = get_alignments(emissions, tok_star, tokenizer)
                spans = get_spans(tok_star, segs, blank)
                words = []
                for w in postprocess_results(txt_star, spans, stride, scores):
                    if w.get("text") == "<star>":
                        continue
                    # deskpai returns each word's score as a SUM of per-frame
                    # log-probs (negative). Normalise to a per-frame geometric
                    # mean probability in (0,1] so boundary.MIN_MEAN_SCORE (a
                    # probability floor) means what it says.
                    n_frames = max(1.0, (w["end"] - w["start"]) * 1000.0 / stride)
                    w["score"] = math.exp(w["score"] / n_frames)
                    words.append(w)
            except Exception as ex:  # noqa: BLE001 — any aligner failure = align_failed
                words = []
                log(f"  align FAIL {r['utterance_id']}: {type(ex).__name__} {str(ex)[:60]}")
            vad = get_speech_timestamps(torch.from_numpy(np.array(clip, dtype=np.float32)),
                                        vad_model, sampling_rate=SR,
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
    log(f"-> {out_path}")


# ---------- finalize stage ----------

def _load_boundary(expected: dict[str, str]) -> tuple[dict[str, dict], int]:
    """Merge boundary.jsonl keyed by utterance_id; ignore stale, error on
    conflicting duplicates (Codex review #2)."""
    bnd: dict[str, dict] = {}
    stale = 0
    for path in _boundary_files():
        for d in _iter_boundary_lines(path):
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

    # backbone alignment gate (Codex review): a no-edit row whose ASR text does
    # not align to its audio is not a trustworthy positive ("alignment-passed
    # no-edit ASR", a minimum-viability gate — NOT verified-correct). Drop it
    # from the published set; the frozen sample is the gated set.
    gate = (df["source"] == "no_edit") & (df["boundary_status"] == "align_failed")
    n_gated = int(gate.sum())
    if n_gated:
        df[gate][["utterance_id", "city_id", "meeting_id", "audio_url", "start",
                  "end", "final_after_text"]].to_csv(
            OUT / "backbone-dropped.csv", index=False)
    df = df[~gate].reset_index(drop=True)
    log(f"backbone align-gate: dropped {n_gated} no_edit align_failed rows "
        f"-> {len(df)} rows")
    # re-check val share over the gated set (must stay in the agreed window)
    if not df.empty and (df["boundary_status"] != "pending").any():
        vh = df[df.split == "validation"].duration_s.sum() / 3600
        th = df.duration_s.sum() / 3600
        share = vh / th if th else 0.0
        if not (0.18 <= share <= 0.22):
            raise SystemExit(
                f"val share {share:.1%} left the 18-22% window after the "
                f"backbone gate — human decision (adjust caps/seed).")
        log(f"val share after gate: {share:.1%}")

    # audit CSV: everything not ok/adjusted, for sampled human review
    audit = df[~df["boundary_status"].isin(["ok", "adjusted"])]
    audit_cols = ["utterance_id", "city_id", "meeting_id", "audio_url", "start",
                  "end", "start_adj", "end_adj", "boundary_status",
                  "final_after_text"]
    audit[audit_cols].to_csv(OUT / "boundary-audit.csv", index=False)
    log(f"boundary audit rows: {len(audit)} -> boundary-audit.csv")

    # start_adj/end_adj = the aligner-corrected span (aligned words +/- PAD_S) —
    # the RECOMMENDED span to cut clips on. Provided for EVERY row that aligned,
    # INCLUDING suspect_cut_*/bleed_in, so no clip cuts a syllable (raw CSV spans
    # do: ~16% of curated corrections end too early). Only align_failed / pending
    # (no reliable alignment) get null; boundary_status still records what the
    # raw span's problem was, for transparency.
    no_span = df["boundary_status"].isin(["align_failed", "pending"])
    df.loc[no_span, ["start_adj", "end_adj"]] = None
    # keep the adjusted-offset columns as nullable float, not object (Codex #7)
    df["start_adj"] = pd.to_numeric(df["start_adj"], errors="coerce")
    df["end_adj"] = pd.to_numeric(df["end_adj"], errors="coerce")
    # the hand-picked included corrections (for the sync report) = the utterance
    # ids in the live-export snapshot
    inc_uids = set()
    snaps = sorted(OUT.glob("raw-export-*.jsonl"))
    if snaps:
        for l in snaps[-1].open():
            try:
                inc_uids.add(json.loads(l)["utterance_id"])
            except json.JSONDecodeError:
                pass
    write_boundary_report(df, inc_uids)

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
          f"Total: {stats['total_rows']} rows, {stats['total_hours']} h", "",
          "## By source", ""]
    for source, s in stats.get("by_source", {}).items():
        md += [f"- **{source}**: {s['rows']} rows, {s['hours']} h "
               f"(split {s['by_split']})"]
    md += [""]
    for split, s in stats["by_split"].items():
        md += [f"## {split}", "",
               f"- rows {s['rows']}, hours {s['hours']} ({s['pct_hours']}%), "
               f"speakers {s['speakers']}",
               f"- sources: {s.get('sources', {})}",
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


def write_boundary_report(df, include_uids: set | None = None) -> None:
    """Team-facing plain-language report on span/sync quality (boundary pass)."""
    import collections as _c

    def block(sub, title):
        done = sub[sub["boundary_status"] != "pending"]
        n = len(done)
        st = _c.Counter(done["boundary_status"])
        def p(k):
            return f"{st.get(k, 0)} ({100 * st.get(k, 0) / n:.1f}%)" if n else "0"
        clean = st.get("ok", 0) + st.get("adjusted", 0)
        lines = [
            f"### {title}", "",
            f"- checked: **{n}** / {len(sub)} rows",
            f"- **in-sync / clean** (ok + adjusted): **{100 * clean / n:.0f}%**" if n else "",
            f"- ends too early, last syllable cut (`suspect_cut_end`): {p('suspect_cut_end')}",
            f"- starts too early (`suspect_cut_start`): {p('suspect_cut_start')}",
            f"- neighbouring phrase audible in span (`suspect_bleed_in`): {p('suspect_bleed_in')}",
            f"- could not align (`align_failed`): {p('align_failed')}", ""]
        return n, st, clean, lines

    corr = df[df["source"] == "correction"]
    inc = corr[corr["utterance_id"].isin(include_uids)] if include_uids else corr
    left = corr[~corr["utterance_id"].isin(include_uids)] if include_uids else corr.iloc[0:0]
    ni, sti, cleani, inc_lines = block(inc, "Curated *included* corrections (your hand-picked set)")
    ce, cs = sti.get("suspect_cut_end", 0), sti.get("suspect_cut_start", 0)
    lean = ("lean to *ends-too-early* (last syllable clipped)" if ce > 1.3 * cs
            else "lean to *starts-too-early*" if cs > 1.3 * ce
            else "roughly balanced between the two edges")

    lines = [
        "# Span / sync quality report", "",
        "**What this checks.** We force-align each utterance's *text* to its "
        "*audio* and run VAD (speech detection). That tells us whether the stored "
        "`start`/`end` timestamps really bracket the spoken words, or clip a "
        "syllable / swallow a neighbour. Every row gets a `boundary_status`; "
        "aligned rows also get a corrected span `start_adj`/`end_adj` (aligned "
        "words ± 0.2 s padding) — **the span to cut clips on.**", "",
        "## Headline (plain words)", "",
        f"- About **{100 * cleani / ni:.0f}% of your hand-picked corrections are "
        f"already in-sync**; the other ~{100 - round(100 * cleani / ni)}% clip a "
        "word at one edge or catch a neighbour." if ni else "",
        f"- Those errors {lean} — so it is **loose timestamps, not one constant "
        "audio offset** (the no-edit backbone actually leans the other way).",
        "- **The fix is automatic:** use `start_adj`/`end_adj` and no syllable is "
        "clipped, whichever edge was loose. No manual re-sync needed.", "",
        "## Numbers", "",
    ] + inc_lines
    if include_uids:
        _, _, _, left_lines = block(left, "Leftover corrections (NB2 audio-verified)")
        lines += left_lines
    _, _, _, bk_lines = block(df[df["source"] == "no_edit"], "No-edit backbone")
    lines += bk_lines
    lines += [
        "## What to do", "",
        "- **Cut training clips on `start_adj`/`end_adj`, not `start`/`end`.** "
        "They include the whole word plus padding, so no syllable is clipped.",
        "- `align_failed` rows (rare) have null `*_adj` — fall back to raw + padding.",
        "- `suspect_bleed_in` = a neighbouring phrase sits inside the raw span; "
        "the corrected span covers only the labelled words.", ""]
    (OUT / "boundary-sync-report.md").write_text("\n".join(lines) + "\n")


def write_dataset_card(stats: dict) -> None:
    split_lines = "\n".join(
        f"| {k} | {v['rows']} | {v['hours']} h | {v['pct_hours']}% | {v['speakers']} |"
        for k, v in stats["by_split"].items())
    source_lines = "\n".join(
        f"| {k} | {v['rows']} | {v['hours']} h | {v['by_split']} |"
        for k, v in stats.get("by_source", {}).items())
    seed = (json.loads(SPLIT_JSON.read_text())["seed"]
            if SPLIT_JSON.exists() else "N/A")
    (OUT / "public" / "README.md").write_text(f"""---
language: [el]
license: cc-by-sa-4.0
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

## Sources (`source` column)

| source | rows | hours | by split |
|---|---|---|---|
{source_lines}

- **correction** — human-curated `(before → after)` edit pairs (review-UI
  includes + audio-verified NB2 leftover).
- **no_edit** — trusted backbone: ASR utterances from review-exposed meetings
  that a reviewer left unchanged (`before_text == text`). This is
  *alignment-passed no-edit ASR* — each clip's text force-aligns to its audio —
  **not** independently verified-correct transcription; treat it as a
  high-precision-but-not-perfect positive. Provided so training is not
  corrections-only (avoids the over-editing bias).

## Fields

- `source` — `correction` (edit pair) or `no_edit` (trusted backbone).
- `before_text` — raw ASR output; `text` — the human-corrected target.
- `start`/`end` — raw utterance span in the meeting audio. **`start_adj`/
  `end_adj` — the recommended span to cut on**: the force-aligned words ± 0.2 s
  padding, so no syllable is clipped. Provided for every row that aligned
  (including the `suspect_*` ones); null only for `align_failed`.
- `boundary_status` — quality of the *raw* span: `ok`/`adjusted` (raw was fine)
  vs `suspect_cut_start`/`suspect_cut_end` (raw clipped a word — corrected in
  `*_adj`), `suspect_bleed_in` (a neighbouring phrase is audible inside the raw
  span; `*_adj` covers only the labelled words), `align_failed` (no reliable
  alignment). ~16% of curated corrections had `suspect_cut_end` — the raw spans
  systematically end a touch early — which `start_adj`/`end_adj` fix.
- `has_overlap` — a reviewer marked overlapping speech (someone else audible)
  in this span. Boolean only; the overlapping speech is not transcribed.
- `error_categories` — reviewer labels for the correction type.

## Split methodology

The 80/20 (by hours) split is computed **once over the whole combined sample**
(corrections + no-edit backbone), so it holds across sources and future batches
inherit it by speaker. Validation = two whole held-out cities (orestiada, argos)
plus whole seeded speakers (>= 3 min speech in-dataset) from the remaining
cities up to ~20% of total hours; **speaker-disjoint** from train. Held-out-city
backbone is per-city-capped so val stays ~20%. A temporal test set (meetings
from June 2026 on) is withheld entirely. Note the val set mixes held-out-city
and seeded-speaker holdout (not a pure random-speaker holdout) — see
`split_assignments.json` (seed {seed}) for the exact map and per-source counts.

## Caveats

- Corrections were hand-curated during review (inclusion bias toward
  interesting errors); this is not a random sample of council speech.
- `meeting_id` slugs collide across cities — always key by
  `(city_id, meeting_id)`.
- Rows with `speaker_id = null` (~no diarization identity) are train-only.
- License: **CC-BY-SA-4.0** (attribution + share-alike), mirroring OpenCouncil's
  AGPL-3.0 copyleft. Source: OpenCouncil / Schema Labs (opencouncil.gr), derived
  from public Greek municipal-council proceedings. This release is
  **metadata-only** — audio is **not** redistributed (stays at
  data.opencouncil.gr). Attribution: "Transcripts © OpenCouncil / Schema Labs,
  CC-BY-SA-4.0". Confirm the attribution wording with Schema Labs before pushing.
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p_rows = sub.add_parser("rows")
    p_rows.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_bnd = sub.add_parser("boundary")
    p_bnd.add_argument("--limit-meetings", type=int, default=0)
    p_bnd.add_argument("--shard", type=str, default="",
                       help="K/N deterministic meeting shard for parallel workers")
    p_bnd.add_argument("--out", type=str, default="",
                       help="output file name under data/hf-dataset/ "
                            "(e.g. boundary.part-0.jsonl); default boundary.jsonl")
    p_bnd.add_argument("--threads", type=int, default=0,
                       help="cap ONNX/torch threads per process (for sharding)")
    p_bnd.add_argument("--window", type=int, default=30,
                       help="forced-align window seconds; 20 is ~29%% faster and "
                            "lossless for clips <=18s (backbone), default 30")
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
