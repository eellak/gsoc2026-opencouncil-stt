# HF Dataset Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One reproducible pipeline that turns the live curated includes (~5,000 edits on the VPS) into Hugging Face-ready `train`/`validation` parquet files with per-split hour accounting, a `has_overlap` flag from reviewer notes, and a boundary-quality pass over utterance spans.

**Architecture:** A small Python package `eval/hf_export/` with three pure, unit-tested modules (`overlap.py`, `split.py`, `boundary.py`) and one CLI orchestrator (`build.py`) with three checkpointed stages: `rows` (fetch + filter + join + split, fast), `boundary` (audio download + forced alignment + VAD, long, resumable), `finalize` (stats + parquet + dataset card). Spec: `docs/specs/hf-dataset-export.md`.

**Tech Stack:** Python 3 in `.venv-eval` (pandas/pyarrow/numpy/torch already present), `soundfile`, `ctc-forced-aligner` (MMS CTC model, Greek via `romanize=True, language="ell"`, CPU float32), `silero-vad` (pip package, CPU), ffmpeg (already used by `eval/autoresearch/prepare_asr.py`).

**Key facts (verified in this repo, don't re-derive):**
- Export endpoint: `GET https://79-76-114-184.sslip.io/api/export` → JSONL, one row per utterance. Fields used: `utterance_id, city_id, meeting_id, meeting_name, meeting_date, audio_url, start, end, initial_before_text, final_after_text, error_categories, include_status, reviewer_notes`.
- Speaker join source: `data/eval/speakers.parquet` (571,124 rows; columns `utterance_id, meeting_id, city_id, person_id, person_name, speaker_label, dur_s, last_modified_by, n_chars`). Join on `utterance_id` (verify `(city_id, meeting_id)` agree; log mismatches).
- Denylist: `eval/exclusions.py::load_excluded_keys()` → set of `(city_id, meeting_id)`.
- Leakage guard: `eval/splits.py::assert_no_leakage(train, ev, key=...)` (works on any dict key).
- Audio helpers to reuse (import, don't copy): `eval/autoresearch/prepare_asr.py::download_mp3, decode_pcm` (16kHz mono float32 PCM, shared cache in `data/asr/audio/`).
- ctc-forced-aligner API (verified via DeepWiki 2026-07-03): `load_alignment_model(device, dtype=torch.float32)` → `(model, tokenizer)`; `generate_emissions(model, waveform_tensor, batch_size=4)` → `(emissions, stride)`; `preprocess_text(text, romanize=True, language="ell", split_size="word", star_frequency="edges")` → `(tokens_starred, text_starred)`; `get_alignments(emissions, tokens_starred, tokenizer)` → `(segments, scores, blank_token)`; `get_spans(tokens_starred, segments, blank_token)`; `postprocess_results(text_starred, spans, stride, scores)` → `[{"text","start","end","score"}, ...]` in seconds. Default model `MahmoudAshraf/mms-300m-1130-forced-aligner` (CC-BY-NC-4.0 — fine: internal QA tooling only, aligner output is not published or deployed). Pass the PCM slice as a `torch.Tensor` directly to `generate_emissions` (skip `load_audio`, which only takes file paths).
- silero-vad API (verified via DeepWiki 2026-07-03): `from silero_vad import load_silero_vad, get_speech_timestamps`; `get_speech_timestamps(tensor_or_ndarray, model, sampling_rate=16000, return_seconds=True, min_silence_duration_ms=100, speech_pad_ms=30)` → `[{"start": s, "end": s}, ...]` (floats, seconds). CPU by default.
- Constants: `VAL_CITIES = {"orestiada", "argos"}`; temporal-TEST filter = `meeting_date >= "2026-06-01"` (string compare works, dates are `YYYY-MM-DD HH:MM:SS`); SEED = 20260703; val window 18–22% of hours; speaker floor 180s within-dataset.

---

### Task 0: Dependencies + package skeleton

**Files:**
- Create: `eval/hf_export/__init__.py` (empty)

- [ ] **Step 1: Install missing deps into `.venv-eval`**

```bash
cd /home/harold/opencouncil-fine-tuning
.venv-eval/bin/pip install soundfile silero-vad ctc-forced-aligner pytest
```

Expected: all install cleanly (torch/transformers already present, so `ctc-forced-aligner` should not pull a new torch).

- [ ] **Step 2: Smoke-check imports**

```bash
.venv-eval/bin/python -c "
import soundfile
from silero_vad import load_silero_vad, get_speech_timestamps
from ctc_forced_aligner import load_alignment_model, generate_emissions, preprocess_text, get_alignments, get_spans, postprocess_results
print('imports OK')
"
```

Expected: `imports OK`. If `ctc_forced_aligner` fails to import, capture the error and stop — do not work around it silently.

- [ ] **Step 3: Create the package dir**

```bash
mkdir -p eval/hf_export && touch eval/hf_export/__init__.py
git add eval/hf_export/__init__.py
git commit -m "hf-export: package skeleton + deps installed in .venv-eval"
```

---

### Task 1: `overlap.py` — standalone-«C» detector + notes report

**Files:**
- Create: `eval/hf_export/overlap.py`
- Test: `eval/tests/test_hf_overlap.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the reviewer-note overlap convention (standalone Latin C)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.hf_export.overlap import has_overlap_marker, notes_report_rows


def test_bare_c_matches():
    assert has_overlap_marker("C") is True
    assert has_overlap_marker("c") is True
    assert has_overlap_marker("  C  ") is True


def test_standalone_c_inside_text_matches():
    assert has_overlap_marker("C και θόρυβος") is True
    assert has_overlap_marker("κακό κόψιμο, C") is True
    assert has_overlap_marker("overlap (C) στο τέλος") is True


def test_c_inside_latin_word_does_not_match():
    assert has_overlap_marker("check this") is False
    assert has_overlap_marker("ASCII") is False
    assert has_overlap_marker("cut") is False


def test_none_and_empty_do_not_match():
    assert has_overlap_marker(None) is False
    assert has_overlap_marker("") is False
    assert has_overlap_marker("καθαρό") is False


def test_notes_report_partitions_matched_unmatched():
    rows = [
        {"utterance_id": "u1", "reviewer_notes": "C"},
        {"utterance_id": "u2", "reviewer_notes": "κάτι άλλο"},
        {"utterance_id": "u3", "reviewer_notes": None},
        {"utterance_id": "u4", "reviewer_notes": "c θόρυβος"},
    ]
    matched, unmatched = notes_report_rows(rows)
    assert [r["utterance_id"] for r in matched] == ["u1", "u4"]
    assert [r["utterance_id"] for r in unmatched] == ["u2"]  # u3 empty -> excluded
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-eval/bin/python -m pytest eval/tests/test_hf_overlap.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'eval.hf_export.overlap'`

- [ ] **Step 3: Implement `overlap.py`**

```python
"""Reviewer-note overlap convention.

The reviewer marks overlapping speech (someone else talking over the
utterance) with a standalone Latin letter C in `reviewer_notes`. Standalone =
not adjacent to another Latin letter, so "C", "c, κακό" match but "check"
does not. Greek text around it is fine. See docs/specs/hf-dataset-export.md.
"""
from __future__ import annotations

import re

_STANDALONE_C = re.compile(r"(?<![A-Za-z])[cC](?![A-Za-z])")


def has_overlap_marker(note: str | None) -> bool:
    if not note:
        return False
    return bool(_STANDALONE_C.search(note))


def notes_report_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition rows with non-empty reviewer_notes into (matched, unmatched).

    Feeds the human-eyeball report published before the dataset: every note
    the rule flagged as overlap, and every note it did not.
    """
    matched, unmatched = [], []
    for r in rows:
        note = r.get("reviewer_notes")
        if not note or not str(note).strip():
            continue
        (matched if has_overlap_marker(note) else unmatched).append(r)
    return matched, unmatched
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-eval/bin/python -m pytest eval/tests/test_hf_overlap.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/hf_export/overlap.py eval/tests/test_hf_overlap.py
git commit -m "hf-export: standalone-C overlap marker + notes report partition"
```

---

### Task 2: `split.py` — seeded speaker split to ~20% hours

**Files:**
- Create: `eval/hf_export/split.py`
- Test: `eval/tests/test_hf_split.py`

Rule (spec): validation = all rows of `orestiada`+`argos`, then whole seeded
speakers from the train cities (≥180s speech within the dataset) added until
validation ≥ 20% of total hours; a speaker that would push past 22% is skipped
and the walk continues; final share must land in [18%, 22%] or the build stops.
Null-speaker rows from the *train* cities can never be moved to validation
(held-out-city rows go to validation regardless of speaker, including nulls).
Known bias (Codex review, accepted): skip-on-overshoot makes very-high-hour
speakers less likely to be picked near the target — record chosen/skipped
speaker durations in `split_assignments.json` so the composition is auditable.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the seeded speaker split."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.hf_export.split import SplitError, assign_splits


def _row(uid, city, spk, dur):
    return {"utterance_id": uid, "city_id": city, "speaker_id": spk,
            "duration_s": float(dur)}


def test_heldout_cities_always_validation():
    rows = [_row("a", "orestiada", "s1", 10), _row("b", "argos", None, 10),
            _row("c", "athens", "s2", 80)]
    res = assign_splits(rows, target_frac=0.20, window=(0.10, 0.30), seed=1)
    assert res.splits[0] == "validation"
    assert res.splits[1] == "validation"
    assert res.splits[2] == "train"


def test_speakers_added_until_target_and_disjoint():
    # held-out = 10s of 200s total (5%) -> needs ~30s more from speakers
    rows = [_row("v", "argos", "sv", 10)]
    rows += [_row(f"a{i}", "athens", "spkA", 10) for i in range(4)]   # 40s
    rows += [_row(f"b{i}", "sparta", "spkB", 10) for i in range(4)]   # 40s
    rows += [_row(f"c{i}", "chania", "spkC", 11) for i in range(10)]  # 110s
    res = assign_splits(rows, target_frac=0.20, window=(0.18, 0.45), seed=7,
                        floor_s=30.0)  # fixture speakers are 40-110s
    val_spk = {rows[i]["speaker_id"] for i, s in enumerate(res.splits)
               if s == "validation"} - {"sv"}
    train_spk = {rows[i]["speaker_id"] for i, s in enumerate(res.splits)
                 if s == "train"}
    assert val_spk, "at least one train-city speaker moved to validation"
    assert not (val_spk & train_spk), "speaker-disjoint violated"
    assert res.val_share >= 0.18


def test_floor_excludes_short_speakers():
    rows = [_row("v", "argos", "sv", 100)]
    rows += [_row("a", "athens", "tiny", 5)]           # below 180s floor
    rows += [_row(f"c{i}", "chania", "big", 60) for i in range(6)]  # 360s
    res = assign_splits(rows, target_frac=0.20, window=(0.15, 0.35),
                        seed=1, floor_s=180.0)
    assert "tiny" not in res.val_speakers


def test_null_speaker_rows_stay_in_train():
    rows = [_row("v", "argos", "sv", 20),
            _row("n", "athens", None, 80)]
    res = assign_splits(rows, target_frac=0.20, window=(0.18, 0.22), seed=1)
    assert res.splits[1] == "train"


def test_same_seed_same_split_different_seed_may_differ():
    rows = [_row("v", "argos", "sv", 10)]
    rows += [_row(f"u{i}", "athens", f"s{i}", 30) for i in range(20)]
    a = assign_splits(rows, target_frac=0.20, window=(0.10, 0.40), seed=42)
    b = assign_splits(rows, target_frac=0.20, window=(0.10, 0.40), seed=42)
    assert a.splits == b.splits and a.val_speakers == b.val_speakers


def test_out_of_window_raises():
    # held-out cities alone are 50% of hours -> way past 22% cap
    rows = [_row("v", "argos", "sv", 100), _row("t", "athens", "s1", 100)]
    with pytest.raises(SplitError):
        assign_splits(rows, target_frac=0.20, window=(0.18, 0.22), seed=1)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-eval/bin/python -m pytest eval/tests/test_hf_split.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `split.py`**

```python
"""Seeded speaker split: held-out cities + whole speakers to ~20% of hours.

Recipe (mentor sync 2026-06-23 + val-split report 2026-06-24, spec
docs/specs/hf-dataset-export.md): validation = all rows of the held-out
cities, then whole seeded speakers from the train cities (>= floor_s speech
within the dataset) until validation reaches target_frac of total hours.
Speaker-disjoint by construction; null-speaker rows never go to validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VAL_CITIES = frozenset({"orestiada", "argos"})
DEFAULT_SEED = 20260703


class SplitError(RuntimeError):
    """Validation share landed outside the acceptable window — human call."""


@dataclass
class SplitResult:
    splits: list[str]              # per input row: "train" | "validation"
    val_speakers: set[str]         # train-city speakers moved to validation
    val_share: float               # validation hours / total hours
    skipped_speakers: list[str] = field(default_factory=list)  # would overshoot


def assign_splits(rows: list[dict], *, val_cities: frozenset = VAL_CITIES,
                  seed: int = DEFAULT_SEED, target_frac: float = 0.20,
                  floor_s: float = 180.0,
                  window: tuple[float, float] = (0.18, 0.22)) -> SplitResult:
    total = sum(r["duration_s"] for r in rows)
    if total <= 0:
        raise SplitError("empty dataset")

    val_s = sum(r["duration_s"] for r in rows if r["city_id"] in val_cities)

    spk_dur: dict[str, float] = {}
    for r in rows:
        spk = r.get("speaker_id")
        if r["city_id"] in val_cities or not spk:
            continue
        spk_dur[spk] = spk_dur.get(spk, 0.0) + r["duration_s"]

    eligible = sorted(s for s, d in spk_dur.items() if d >= floor_s)
    rng = np.random.default_rng(seed)
    order = [eligible[i] for i in rng.permutation(len(eligible))]

    chosen: set[str] = set()
    skipped: list[str] = []
    cur = val_s
    for spk in order:
        if cur / total >= target_frac:
            break
        if (cur + spk_dur[spk]) / total > window[1]:
            skipped.append(spk)
            continue
        chosen.add(spk)
        cur += spk_dur[spk]

    share = cur / total
    if not (window[0] <= share <= window[1]):
        raise SplitError(
            f"validation share {share:.1%} outside window "
            f"[{window[0]:.0%}, {window[1]:.0%}] — stop for a human decision "
            f"(held-out cities alone: {val_s / total:.1%})")

    splits = [
        "validation" if (r["city_id"] in val_cities
                         or (r.get("speaker_id") or "") in chosen)
        else "train"
        for r in rows
    ]
    return SplitResult(splits=splits, val_speakers=chosen, val_share=share,
                       skipped_speakers=skipped)
```

Also add `spk_dur` to the result so the caller can persist chosen/skipped
speaker durations: add a field `speaker_durations: dict[str, float] =
field(default_factory=dict)` to `SplitResult` and return
`speaker_durations=spk_dur` (the stage writes `{spk: dur}` for chosen +
skipped speakers into `split_assignments.json`).

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-eval/bin/python -m pytest eval/tests/test_hf_split.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/hf_export/split.py eval/tests/test_hf_split.py
git commit -m "hf-export: seeded speaker split to ~20% validation hours"
```

---

### Task 3: `build.py` stage `rows` — fetch, filter, join, split

**Files:**
- Create: `eval/hf_export/build.py`
- Test: `eval/tests/test_hf_build_rows.py` (pure helpers only)

The stage: snapshot the live export → keep `include_status == "include"` →
drop denylist + temporal-TEST → join `speaker_id` → compute durations +
`has_overlap` → `assign_splits` → write `data/hf-dataset/rows.parquet` +
`overlap-notes-report.md` + `split_assignments.json` + a drop log. Every
filter logs its count (repo rule: no silent drops).

- [ ] **Step 1: Write the failing test for the pure helpers**

```python
"""Tests for build.py pure helpers (filtering/joining), no network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.hf_export.build import filter_rows, join_speakers


def _exp(uid, city, mtg, date, inc="include", start=0.0, end=2.0):
    return {"utterance_id": uid, "city_id": city, "meeting_id": mtg,
            "meeting_date": date, "include_status": inc,
            "start": start, "end": end, "audio_url": "http://x/a.mp3",
            "initial_before_text": "b", "final_after_text": "a",
            "error_categories": [], "reviewer_notes": None}


def test_filter_keeps_only_includes_and_logs():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u2", "athens", "m1", "2025-01-01 10:00:00", inc="exclude"),
            _exp("u3", "athens", "m1", "2025-01-01 10:00:00", inc="uncertain")]
    kept, drops = filter_rows(rows, excluded_keys=set())
    assert [r["utterance_id"] for r in kept] == ["u1"]
    assert drops["not_include"] == 2


def test_filter_drops_denylist_and_temporal():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u2", "rhodes", "jul17_2025", "2025-07-17 10:00:00"),
            _exp("u3", "athens", "m2", "2026-06-05 10:00:00")]
    kept, drops = filter_rows(rows, excluded_keys={("rhodes", "jul17_2025")})
    assert [r["utterance_id"] for r in kept] == ["u1"]
    assert drops["denylist"] == 1 and drops["temporal_test"] == 1


def test_filter_drops_missing_or_malformed_dates():
    rows = [_exp("u1", "athens", "m1", None),
            _exp("u2", "athens", "m1", "unknown"),
            _exp("u3", "athens", "m1", "2025-01-01 10:00:00")]
    kept, drops = filter_rows(rows, excluded_keys=set())
    assert [r["utterance_id"] for r in kept] == ["u3"]
    assert drops["bad_date"] == 2


def test_filter_rejects_multiple_audio_urls_per_meeting():
    import pytest
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u2", "athens", "m1", "2025-01-01 10:00:00")]
    rows[1]["audio_url"] = "http://x/OTHER.mp3"
    with pytest.raises(ValueError):
        filter_rows(rows, excluded_keys=set())


def test_filter_drops_bad_spans():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00", start=5.0, end=5.0),
            _exp("u2", "athens", "m1", "2025-01-01 10:00:00", start=9.0, end=5.0),
            _exp("u3", "athens", "m1", "2025-01-01 10:00:00")]
    kept, drops = filter_rows(rows, excluded_keys=set())
    assert [r["utterance_id"] for r in kept] == ["u3"]
    assert drops["bad_span"] == 2


def test_filter_raises_on_duplicate_composite_key():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u1", "athens", "m1", "2025-01-01 10:00:00")]
    import pytest
    with pytest.raises(ValueError):
        filter_rows(rows, excluded_keys=set())


def test_join_speakers_matches_and_counts_missing():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u2", "athens", "m1", "2025-01-01 10:00:00")]
    spk = {"u1": "person-9"}
    joined, n_missing = join_speakers(rows, spk)
    assert joined[0]["speaker_id"] == "person-9"
    assert joined[1]["speaker_id"] is None
    assert n_missing == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-eval/bin/python -m pytest eval/tests/test_hf_build_rows.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `build.py` (helpers + `rows` stage CLI)**

```python
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
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.exclusions import load_excluded_keys            # noqa: E402
from eval.hf_export.overlap import has_overlap_marker, notes_report_rows  # noqa: E402
from eval.hf_export.split import DEFAULT_SEED, VAL_CITIES, assign_splits  # noqa: E402

EXPORT_URL = "https://79-76-114-184.sslip.io/api/export"
SPEAKERS_PARQUET = ROOT / "data" / "eval" / "speakers.parquet"
OUT = ROOT / "data" / "hf-dataset"
ROWS_PARQUET = OUT / "rows.parquet"
SPLIT_JSON = OUT / "split_assignments.json"
TEMPORAL_TEST_FROM = "2026-06-01"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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

    res = assign_splits(kept, seed=args.seed)
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
    log(f"split: train {hours['train']:.2f}h / validation {hours['validation']:.2f}h "
        f"({res.val_share:.1%}) — {len(res.val_speakers)} mixed-city val speakers, "
        f"{len(res.skipped_speakers)} skipped as overshoot")

    SPLIT_JSON.write_text(json.dumps({
        "seed": args.seed, "snapshot": snapshot.name,
        "val_cities": sorted(VAL_CITIES),
        "val_speakers": sorted(res.val_speakers),
        "skipped_speakers": res.skipped_speakers,
        "val_share_hours": round(res.val_share, 4),
        "temporal_test_from": TEMPORAL_TEST_FROM,
        "drop_counts": drops, "n_missing_speaker": n_missing,
    }, ensure_ascii=False, indent=2))

    cols = ["utterance_id", "city_id", "meeting_id", "meeting_date", "speaker_id",
            "audio_url", "start", "end", "duration_s", "initial_before_text",
            "final_after_text", "error_categories", "has_overlap", "source",
            "split", "reviewer_notes"]
    pd.DataFrame([{c: r.get(c) for c in cols} for r in kept]).to_parquet(ROWS_PARQUET)
    log(f"-> {ROWS_PARQUET} ({len(kept)} rows), {SPLIT_JSON}, overlap-notes-report.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p_rows = sub.add_parser("rows")
    p_rows.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    if args.stage == "rows":
        stage_rows(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests, then the real stage**

```bash
.venv-eval/bin/python -m pytest eval/tests/test_hf_build_rows.py -v
```

Expected: 5 passed.

```bash
.venv-eval/bin/python -m eval.hf_export.build rows
```

Expected: logs with per-filter drop counts, a split line showing hours and a
val share inside 18–22%, and `data/hf-dataset/rows.parquet` written. If
`SplitError` is raised (share outside window), STOP and report the numbers to
the user — that is the designed human gate, not a bug to code around.

Sanity-check the output:

```bash
.venv-eval/bin/python - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/hf-dataset/rows.parquet")
print(len(df), "rows")
print(df.groupby("split")["duration_s"].agg(["count", "sum"]))
v = df[df.split == "validation"]; t = df[df.split == "train"]
assert set(v[v.city_id.isin({"orestiada","argos"})].city_id) <= {"orestiada","argos"}
assert not (set(v.speaker_id.dropna()) & set(t.speaker_id.dropna())), "speaker leakage!"
assert t[t.city_id.isin({"orestiada","argos"})].empty, "held-out city in train!"
print("leakage guards OK; overlap flags:", int(df.has_overlap.sum()))
EOF
```

Expected: row count ≈ the ~5,000 includes; guards pass.

- [ ] **Step 5: Commit**

```bash
git add eval/hf_export/build.py eval/tests/test_hf_build_rows.py
git commit -m "hf-export: rows stage — fetch, filter, speaker join, seeded split"
```

---

### Task 4: `boundary.py` — classification logic (pure) 

**Files:**
- Create: `eval/hf_export/boundary.py`
- Test: `eval/tests/test_hf_boundary.py`

Geometry: each clip is sliced with `MARGIN_S = 1.0` extra context on both
sides, so inside the extended clip the raw span is `[margin, margin + raw_dur]`.
The aligner places the label words inside the extended clip; VAD gives speech
segments. `classify_boundary` is pure — it never touches audio or models.

Rules:
- no words or mean score < `MIN_MEAN_SCORE` → `align_failed`
- first word starts `> EDGE_TOL_S` *before* the raw start → `suspect_cut_start`
  (the CSV span cut into the first word); symmetric for `suspect_cut_end`
- VAD speech of `≥ BLEED_MIN_S` inside the raw span but *outside* the aligned
  words' span → `suspect_bleed_in` (neighbour speech bracketed by the span)
- otherwise `ok` if the proposed adjustment moves either edge ≤ `OK_SHIFT_S`,
  else `adjusted`
- proposed offsets: aligned span ± `PAD_S`, clamped to the extended clip

- [ ] **Step 1: Write the failing test**

```python
"""Tests for boundary classification (pure geometry, no audio/models)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.hf_export.boundary import classify_boundary

MARGIN = 1.0


def _w(start, end, score=0.9):
    return {"text": "λέξη", "start": start, "end": end, "score": score}


def test_clean_clip_is_ok():
    # raw span [1.0, 4.0] in extended clip; words snugly inside; VAD agrees
    words = [_w(1.05, 2.0), _w(2.1, 3.9)]
    vad = [{"start": 1.0, "end": 3.95}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "ok"
    assert r.start_off < 1.05 and r.end_off > 3.9  # padded outward


def test_cut_start_detected():
    # aligner finds the first word starting well BEFORE the raw start
    words = [_w(0.55, 1.4), _w(1.5, 3.9)]
    vad = [{"start": 0.5, "end": 3.95}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "suspect_cut_start"


def test_cut_end_detected():
    words = [_w(1.05, 2.0), _w(2.1, 4.6)]  # last word ends past raw end 4.0
    vad = [{"start": 1.0, "end": 4.7}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "suspect_cut_end"


def test_bleed_in_detected():
    # words occupy [2.0, 3.9] but VAD hears speech from 1.0 — someone else
    # talks inside the raw span before the label starts
    words = [_w(2.0, 3.0), _w(3.1, 3.9)]
    vad = [{"start": 1.0, "end": 3.95}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "suspect_bleed_in"


def test_no_words_is_align_failed():
    r = classify_boundary([], [{"start": 1.0, "end": 4.0}],
                          raw_dur=3.0, clip_dur=5.0)
    assert r.status == "align_failed"


def test_low_score_is_align_failed():
    words = [_w(1.05, 3.9, score=0.05)]
    r = classify_boundary(words, [{"start": 1.0, "end": 3.95}],
                          raw_dur=3.0, clip_dur=5.0)
    assert r.status == "align_failed"


def test_large_shift_is_adjusted():
    # aligned span sits deep inside the raw span -> big inward shift, no flags
    words = [_w(1.9, 2.5), _w(2.6, 3.0)]
    vad = [{"start": 1.9, "end": 3.0}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "adjusted"


def test_invalid_and_unsorted_words_are_sanitized():
    # unsorted + zero-length + NaN words must not corrupt the span
    words = [_w(3.0, 3.9), _w(2.0, 2.0), _w(float("nan"), 1.5), _w(1.05, 1.9)]
    vad = [{"start": 1.0, "end": 3.95}]
    r = classify_boundary(words, vad, raw_dur=3.0, clip_dur=5.0)
    assert r.status == "ok"          # effective span = [1.05, 3.9]


def test_raw_span_truncated_by_clip_end_is_clamped():
    # raw span extends past the decoded audio: raw_hi clamps to clip_dur
    words = [_w(1.05, 3.4)]
    vad = [{"start": 1.0, "end": 3.5}]
    r = classify_boundary(words, vad, raw_dur=4.0, clip_dur=3.5)
    assert r.status in ("ok", "adjusted")  # must not flag a phantom cut_end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-eval/bin/python -m pytest eval/tests/test_hf_boundary.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `boundary.py` (pure part)**

```python
"""Boundary-quality classification for utterance clips.

Closes the 2026-07-03 known issue in docs/decisions/audio.md: raw CSV
utterance spans can cut mid-syllable or bracket neighbouring speech. Each clip
is sliced with MARGIN_S extra context per side; the label is force-aligned
inside that extended clip and VAD provides speech segments. This module holds
the pure geometry; the model calls live in the boundary stage of build.py.
"""
from __future__ import annotations

from dataclasses import dataclass

MARGIN_S = 1.0        # extra context sliced on each side of the raw span
EDGE_TOL_S = 0.15     # how far a word may poke past the raw edge before "cut"
BLEED_MIN_S = 0.25    # min un-labelled VAD speech inside the span -> bleed
PAD_S = 0.2           # padding added around the aligned span
OK_SHIFT_S = 0.30     # max per-edge adjustment still called "ok"
MIN_MEAN_SCORE = 0.15 # below this mean alignment score -> align_failed


@dataclass
class BoundaryResult:
    status: str        # ok|adjusted|suspect_cut_start|suspect_cut_end|suspect_bleed_in|align_failed
    start_off: float   # proposed start, seconds relative to the EXTENDED clip
    end_off: float
    mean_score: float


def _speech_overlap(vad: list[dict], lo: float, hi: float) -> float:
    """Total VAD speech seconds inside [lo, hi]."""
    return sum(max(0.0, min(seg["end"], hi) - max(seg["start"], lo))
               for seg in vad)


def _sane_words(words: list[dict], clip_dur: float) -> list[dict]:
    """Drop star/NaN/zero-length/out-of-clip words; sort by start (Codex #8)."""
    import math
    out = []
    for w in words:
        s, e = w.get("start"), w.get("end")
        if s is None or e is None:
            continue
        if isinstance(s, float) and math.isnan(s):
            continue
        if isinstance(e, float) and math.isnan(e):
            continue
        if not (0.0 <= s < e <= clip_dur + 1e-6):
            continue
        if w.get("text") == "<star>":
            continue
        out.append(w)
    return sorted(out, key=lambda w: (w["start"], w["end"]))


def classify_boundary(words: list[dict], vad: list[dict], *, raw_dur: float,
                      clip_dur: float, margin_s: float = MARGIN_S) -> BoundaryResult:
    raw_lo = margin_s
    raw_hi = min(margin_s + raw_dur, clip_dur)   # clamp: span may outrun audio
    fallback = BoundaryResult("align_failed", raw_lo, raw_hi, 0.0)
    words = _sane_words(words, clip_dur)
    if not words:
        return fallback
    scores = [w.get("score") for w in words if w.get("score") is not None]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    if mean_score < MIN_MEAN_SCORE:
        return BoundaryResult("align_failed", raw_lo, raw_hi, mean_score)

    a_lo = min(w["start"] for w in words)
    a_hi = max(w["end"] for w in words)
    # plausibility: aligned span wildly shorter/longer than the raw span means
    # the alignment cannot be trusted (Codex #9)
    if (a_hi - a_lo) < 0.3 * (raw_hi - raw_lo) or (a_hi - a_lo) > clip_dur:
        return BoundaryResult("align_failed", raw_lo, raw_hi, mean_score)
    start_off = max(0.0, a_lo - PAD_S)
    end_off = min(clip_dur, a_hi + PAD_S)

    if a_lo < raw_lo - EDGE_TOL_S:
        status = "suspect_cut_start"
    elif a_hi > raw_hi + EDGE_TOL_S:
        status = "suspect_cut_end"
    elif (_speech_overlap(vad, raw_lo, a_lo) >= BLEED_MIN_S
          or _speech_overlap(vad, a_hi, raw_hi) >= BLEED_MIN_S):
        status = "suspect_bleed_in"
    elif max(abs(start_off - raw_lo), abs(end_off - raw_hi)) <= OK_SHIFT_S:
        status = "ok"
    else:
        status = "adjusted"
    return BoundaryResult(status, start_off, end_off, mean_score)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv-eval/bin/python -m pytest eval/tests/test_hf_boundary.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/hf_export/boundary.py eval/tests/test_hf_boundary.py
git commit -m "hf-export: pure boundary classification (align + VAD geometry)"
```

---

### Task 5: `build.py` stage `boundary` — resumable audio pass

**Files:**
- Modify: `eval/hf_export/build.py` (add stage; no new pure logic — the logic
  is Task 4's tested module, this is plumbing)

Reuses `download_mp3` + `decode_pcm` from `eval/autoresearch/prepare_asr.py`
(shared mp3 cache `data/asr/audio/`). Loads the aligner + VAD once; iterates
meeting-by-meeting (decode once, slice all clips); appends one JSON line per
utterance to `data/hf-dataset/boundary.jsonl`; on restart, already-done
utterance ids are skipped. Run under `tmux` — with ~200 meetings this is a
long, download-heavy job.

- [ ] **Step 1: Add the stage to `build.py`**

Add these imports near the top (after the existing ones):

```python
BOUNDARY_JSONL = OUT / "boundary.jsonl"


def _row_sig(r: dict) -> str:
    """Signature binding a boundary result to the exact span+text it saw."""
    import hashlib
    key = f"{r['audio_url']}|{float(r['start']):.3f}|{float(r['end']):.3f}|{r['final_after_text']}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
```

Add the stage function:

```python
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
    # staleness guard (Codex #19): a boundary line only counts as done if it
    # was computed for the SAME span+text as the current rows.parquet
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

    out = BOUNDARY_JSONL.open("a")
    for mi, (city, meeting) in enumerate(meetings, 1):
        rows = by_mtg[(city, meeting)]
        mp3 = download_mp3(rows[0]["audio_url"], city, meeting)
        if mp3 is None:
            for r in rows:
                out.write(json.dumps({"utterance_id": r["utterance_id"],
                                      "row_sig": _row_sig(r),
                                      "boundary_status": "align_failed",
                                      "start_adj": r["start"], "end_adj": r["end"],
                                      "mean_score": 0.0,
                                      "note": "audio download failed"}) + "\n")
            out.flush()
            continue
        pcm = decode_pcm(mp3)
        if pcm is None:
            for r in rows:
                out.write(json.dumps({"utterance_id": r["utterance_id"],
                                      "row_sig": _row_sig(r),
                                      "boundary_status": "align_failed",
                                      "start_adj": r["start"], "end_adj": r["end"],
                                      "mean_score": 0.0,
                                      "note": "audio decode failed"}) + "\n")
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
```

Wire it into `main()`:

```python
    p_bnd = sub.add_parser("boundary")
    p_bnd.add_argument("--limit-meetings", type=int, default=0)
```

and

```python
    elif args.stage == "boundary":
        stage_boundary(args)
```

- [ ] **Step 2: Smoke-run on 2 meetings**

```bash
.venv-eval/bin/python -m eval.hf_export.build boundary --limit-meetings 2
```

Expected: downloads (or reuses) 2 mp3s, logs `[1/2] ... clips classified`,
appends lines to `data/hf-dataset/boundary.jsonl`. Inspect a few:

```bash
head -5 data/hf-dataset/boundary.jsonl
```

Expected: JSON lines with plausible statuses (mostly `ok`/`adjusted`,
`start_adj` within ~1s of the raw start) — if everything is `align_failed`,
stop and debug the aligner call before scaling.

- [ ] **Step 3: Verify resume works**

```bash
.venv-eval/bin/python -m eval.hf_export.build boundary --limit-meetings 2
```

Expected: `resume: N utterances already classified` and the same 2 meetings now
process 0 new utterances (or are absent from the worklist).

- [ ] **Step 4: Commit**

```bash
git add eval/hf_export/build.py
git commit -m "hf-export: boundary stage — resumable align+VAD pass over meetings"
```

- [ ] **Step 5: Launch the full pass in tmux (long-running; report, don't wait)**

```bash
tmux new-session -d -s hfboundary \
  'cd /home/harold/opencouncil-fine-tuning && .venv-eval/bin/python -m eval.hf_export.build boundary 2>&1 | tee data/hf-dataset/boundary-run.log'
```

Tell the user it is running and roughly how many meetings it will process
(count from the stage's first log lines). Finalize (Task 6) can be implemented
meanwhile and run once this finishes.

---

### Task 6: `build.py` stage `finalize` — outputs, stats, dataset card

**Files:**
- Modify: `eval/hf_export/build.py` (add stage)
- Test: extend `eval/tests/test_hf_build_rows.py` with the stats helper test

- [ ] **Step 1: Write the failing test for the stats helper**

Append to `eval/tests/test_hf_build_rows.py`:

```python
def test_build_stats_hours_and_percentages():
    from eval.hf_export.build import build_stats
    rows = [
        {"split": "train", "duration_s": 3600.0, "city_id": "athens",
         "error_categories": ["homophone"], "has_overlap": False,
         "boundary_status": "ok", "speaker_id": "s1"},
        {"split": "validation", "duration_s": 1800.0, "city_id": "argos",
         "error_categories": [], "has_overlap": True,
         "boundary_status": "suspect_bleed_in", "speaker_id": "s2"},
    ]
    st = build_stats(rows)
    assert st["total_hours"] == 1.5
    assert st["by_split"]["train"]["hours"] == 1.0
    assert st["by_split"]["validation"]["pct_hours"] == 33.3
    assert st["overlap_rows"] == 1
    assert st["boundary_status"]["suspect_bleed_in"] == 1
    assert st["by_split"]["train"]["speakers"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv-eval/bin/python -m pytest eval/tests/test_hf_build_rows.py::test_build_stats_hours_and_percentages -v
```

Expected: FAIL — `ImportError: cannot import name 'build_stats'`.

- [ ] **Step 3: Implement `build_stats` + `stage_finalize` in `build.py`**

```python
def build_stats(rows: list[dict]) -> dict:
    """Aggregate hours/percentages/counters for stats.json (pure, tested)."""
    total_h = sum(r["duration_s"] for r in rows) / 3600
    by_split: dict[str, dict] = {}
    for split in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == split]
        h = sum(r["duration_s"] for r in sub) / 3600
        cats = collections.Counter(c for r in sub for c in (r["error_categories"] or []))
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


def stage_finalize(args) -> None:
    import pandas as pd
    df = pd.read_parquet(ROWS_PARQUET)
    expected = {r["utterance_id"]: _row_sig(r) for r in df.to_dict("records")}
    bnd = {}
    stale = 0
    if BOUNDARY_JSONL.exists():
        for l in BOUNDARY_JSONL.open():
            d = json.loads(l)
            if expected.get(d["utterance_id"]) == d.get("row_sig"):
                bnd[d["utterance_id"]] = d
            else:
                stale += 1
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

    # published columns — reviewer_notes stays internal (free text, PII risk)
    pub_cols = ["utterance_id", "city_id", "meeting_id", "meeting_date",
                "speaker_id", "audio_url", "start", "end", "start_adj",
                "end_adj", "duration_s", "boundary_status",
                "initial_before_text", "final_after_text", "error_categories",
                "has_overlap", "source", "split"]
    pub = df[pub_cols].rename(columns={"initial_before_text": "before_text",
                                       "final_after_text": "text"})
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
        log(f"-> public/{split}.parquet / .jsonl ({len(part)} rows)")
    (pub_dir / "split_assignments.json").write_text(SPLIT_JSON.read_text())

    stats = build_stats(df.to_dict("records"))
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
```

`write_dataset_card(stats)` renders `OUT / "public" / "README.md"` — an HF dataset card
with YAML header. Include verbatim:

```python
def write_dataset_card(stats: dict) -> None:
    split_lines = "\n".join(
        f"| {k} | {v['rows']} | {v['hours']} h | {v['pct_hours']}% | {v['speakers']} |"
        for k, v in stats["by_split"].items())
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
(seed {json.loads((ROOT / 'data/hf-dataset/split_assignments.json').read_text())['seed'] if (ROOT / 'data/hf-dataset/split_assignments.json').exists() else 'N/A'}).

## Caveats

- Corrections were hand-curated during review (inclusion bias toward
  interesting errors); this is not a random sample of council speech.
- `meeting_id` slugs collide across cities — always key by
  `(city_id, meeting_id)`.
- Rows with `speaker_id = null` (~no diarization identity) are train-only.
- License: pending confirmation with OpenCouncil; audio remains at
  data.opencouncil.gr and is not redistributed here.
""")
```

Wire into `main()`:

```python
    p_fin = sub.add_parser("finalize")
    p_fin.add_argument("--allow-pending-boundary", action="store_true")
```

and

```python
    elif args.stage == "finalize":
        stage_finalize(args)
```

- [ ] **Step 4: Run tests, then finalize (with pending boundary if the tmux run is still going)**

```bash
.venv-eval/bin/python -m pytest eval/tests/ -v -k "hf_"
```

Expected: all hf_ tests pass.

```bash
.venv-eval/bin/python -m eval.hf_export.build finalize --allow-pending-boundary
```

Expected: `public/{train,validation}.{parquet,jsonl}`, `public/README.md`,
`public/split_assignments.json` plus internal `stats.{json,md}` and
`boundary-audit.csv` under `data/hf-dataset/`; the log line shows hours per
split with the validation share inside 18–22%. Re-run `finalize` WITHOUT the
flag after the boundary tmux run completes — `--allow-pending-boundary` output
is an internal preview, never the published artifact.

- [ ] **Step 5: Commit**

```bash
git add eval/hf_export/build.py eval/tests/test_hf_build_rows.py
git commit -m "hf-export: finalize stage — parquet splits, stats, dataset card"
```

---

### Task 7: Verification, docs, and handoff

**Files:**
- Modify: `CURRENT.md` (Next Concrete Steps), `docs/decisions/audio.md`
  (known-issue note), `docs/progress.md` (HF-publish row), `docs/decisions/data.md`
  (short entry: published-split recipe + overlap flag convention)

- [ ] **Step 1: Full test suite + leakage re-check on the real output**

```bash
.venv-eval/bin/python -m pytest eval/tests/ -v
.venv-eval/bin/python - <<'EOF'
import json, pandas as pd
t = pd.read_parquet("data/hf-dataset/public/train.parquet")
v = pd.read_parquet("data/hf-dataset/public/validation.parquet")
assert not (set(t.speaker_id.dropna()) & set(v.speaker_id.dropna()))
assert t[t.city_id.isin({"orestiada","argos"})].empty
assert (t.meeting_date < "2026-06-01").all() and (v.meeting_date < "2026-06-01").all()
assert t.utterance_id.is_unique and v.utterance_id.is_unique
assert not (set(t.utterance_id) & set(v.utterance_id))
assert (t.boundary_status != "pending").all() and (v.boundary_status != "pending").all(), \
    "pending boundary rows in the publishable files — rerun finalize after the boundary pass"
for df_ in (t, v):
    assert "reviewer_notes" not in df_.columns
stats = json.load(open("data/hf-dataset/stats.json"))
share = stats["by_split"]["validation"]["pct_hours"]
assert 18 <= share <= 22, share
print(f"OK: {len(t)} train / {len(v)} val rows, val {share}% of hours")
EOF
```

Expected: all pass, printed summary matches `stats.md`.

- [ ] **Step 2: Update the vault docs**

- `CURRENT.md` → add to *Next Concrete Steps*: HF publish pending manual
  `huggingface-cli upload` after boundary audit + license confirmation.
- `docs/decisions/audio.md` → under the 2026-07-03 known issue, append: *"Addressed
  for the published dataset by `eval/hf_export` (forced-align + VAD
  `boundary_status`, adjusted spans); training-clip builds should consume
  `start_adj`/`end_adj`."*
- `docs/progress.md` → move the "publish reproducible dataset on HuggingFace"
  row to in-progress `[~]` with a pointer to `data/hf-dataset/`.
- `docs/decisions/data.md` → short *Accepted* entry: published split recipe
  (held-out cities + seeded ≥3-min speakers to ~20% hours, temporal test
  withheld, seed 20260703) and the standalone-«C» → `has_overlap` convention.

- [ ] **Step 3: Ask the user to eyeball the two human-gate reports**

Point the user at `data/hf-dataset/overlap-notes-report.md` (does the C rule
match their marking habits?) and `data/hf-dataset/boundary-audit.csv` (sample
a few suspects in the review UI). Do NOT push to HF — the card documents the
manual `huggingface-cli upload` command; the push is the user's call.

- [ ] **Step 4: Commit docs**

```bash
git add CURRENT.md docs/decisions/audio.md docs/decisions/data.md docs/progress.md
git commit -m "docs: HF dataset export landed — split recipe, overlap flag, boundary status"
```

---

## Self-review notes

**Codex plan review (2026-07-03, effort=high) incorporated:** fixed the split
test floor bug; word sanitization + raw-span clamping + plausibility guard in
`classify_boundary`; adjusted spans nulled for suspect/failed rows in the
published files; `row_sig` staleness guard on boundary resume/finalize;
`public/` upload-only subdir so internal reports (reviewer notes!) can't leak
to HF; global-uniqueness assertions for `utterance_id`, conflict check in the
speaker map, one-`audio_url`-per-meeting invariant; malformed-date drop
counter; `error_categories` normalized to `list[str]`; verification asserts
no `pending` rows and no `reviewer_notes` column in publishable files.
Accepted-with-note (not changed): skip-on-overshoot bias against very long
speakers (documented, durations persisted); `suspect_bleed_in` kept as a
conservative flag (documented in the card).

- Spec coverage: inputs (T3), row schema + overlap (T1/T3), split + guards
  (T2/T3/T7), boundary pass (T4/T5), outputs + stats + card (T6), no-silent-drops
  (T3 drop counters), manual publish (T6 card + T7). Soniox cross-check
  (`--soniox-crosscheck`) is spec-optional and deferred — noted here explicitly
  rather than half-implemented.
- The aligner model is CC-BY-NC: used only as internal QA tooling; its output
  (timestamps/flags) ships, the model does not. Noted in T0/T4 rationale.
- `margin_s` passed to `classify_boundary` is the *actual* left margin (clamped
  at file start) — right-margin clamping only affects `clip_dur`, already
  handled by the `min(clip_dur, ...)` in the offset computation.
