# HF dataset export: split + publish script

Status: **APPROVED design** — 2026-07-03. Supersedes nothing; implements the
publish step planned in [dataset-split-and-publish-plan](dataset-split-and-publish-plan.md) §3
under the split agreed at the [2026-06-23 mentor sync](../meetings/2026-06-23-mentor-sync.md)
and refined in `data/reports/finetune-research/val-split-whole-city-vs-stratified.md`.

## Goal

One reproducible script that takes the current curated includes (~5,000 edits,
live on the VPS) and produces Hugging Face-ready `train` / `validation` files,
with per-split audio-hour accounting, an audio-overlap flag derived from
reviewer notes, and a boundary-quality pass over the utterance spans.

First release is **metadata-only** (text pairs + audio URLs + offsets). Audio
clips embedded as an HF `Audio` feature come in a second release, after the
license/PII question is confirmed with OpenCouncil. The temporal TEST
(meetings reviewed June 2026+) is removed from the pool and **never published**.

## Inputs

- Live `/api/export` JSONL from the VPS — keep rows with
  `include_status == "include"` only. Snapshot the raw export to
  `data/hf-dataset/raw-export-<date>.jsonl` so the run is reproducible.
- Cached meeting JSON / `data/eval/speakers.parquet` — `person_id` + timing per
  utterance (speaker identity for the split).
- `data/exclusions/unreviewed_meetings.json` via `eval/exclusions.py` —
  canonical meeting denylist, keyed `(city_id, meeting_id)`.

## Row schema (one row per utterance)

| field | source / rule |
| --- | --- |
| `utterance_id`, `city_id`, `meeting_id`, `meeting_date` | export |
| `speaker_id` | meeting JSON `speakerTag.personId`; nullable |
| `audio_url` | export (original OpenCouncil URL) |
| `start`, `end`, `duration_s` | export offsets (raw CSV spans) |
| `start_adj`, `end_adj` | boundary pass output (VAD-snapped ±0.2s) |
| `boundary_status` | `ok / adjusted / suspect_cut_start / suspect_cut_end / suspect_bleed_in / align_failed` |
| `before_text` | raw ASR (initial_before_text) |
| `text` | `final_after_text` — the training target |
| `error_categories` | export labels |
| `has_overlap` | standalone «C» in `reviewer_notes` (see below) |
| `source` | `"correction"` (review includes + NB2 leftover) or `"no_edit"` (backbone) |
| `split` | `train` / `validation` |

### Sources (combined dataset — 2026-07-05)

The published set now combines three sources, all mapped to this schema, split
by ONE frozen speaker-disjoint map (see Split):

1. **review-UI includes** (`/api/export`, `include_status=="include"`) — `correction`.
2. **NB2 audio-verified leftover** (`data/next-batch/final_audio/nb2audio_ids.json`
   → text/span from `selected_edits.jsonl`), excluding rows already in (1) —
   `correction`.
3. **no-edit backbone** — alignment-passed no-edit ASR from *review-exposed* meetings
   (`frac_user ≥ 0.15` or `humanReview`, minus denylist), no-edit utterances
   (`lastModifiedBy is null`) 1–15 s, text from meeting JSON, `before==text`,
   per-city capped to ~20k. `source="no_edit"`. Gate: an utterance whose text
   fails to force-align to its audio (`align_failed`) is dropped — this is
   *alignment-passed no-edit ASR*, a minimum-viability gate, **not** verified
   correct. Provided so training is not corrections-only.

### `has_overlap` derivation

Reviewer marks overlapping speech with a standalone letter C in
`reviewer_notes`. Rule: case-insensitive Latin `c` not adjacent to another
Latin letter (regex `(?i)(?<![a-z])c(?![a-z])`). The script emits
`data/hf-dataset/overlap-notes-report.md` listing every non-empty note split
into matched / unmatched, for a human eyeball pass before publishing.
`reviewer_notes` itself is NOT published (free text, may contain PII/asides).

This is deliberately a boolean only — infrastructure for future overlap work.
Transcribing *what* the other speaker says is out of scope.

## Split (seeded, reproducible)

1. Drop denylisted meetings; drop temporal-TEST meetings (`meeting_date >=
   2026-06-01`, the operational form of "TEST = Jun 2026+") — logged, not
   published.
1b. The split runs **once over the whole combined sample** (corrections +
   backbone), so 80/20-by-hours holds across sources and future batches inherit
   it by speaker. Held-out-city backbone is per-city-capped so val stays ~20%.
2. `validation` = all rows from held-out cities `orestiada`, `argos` (whole
   cities → speaker-disjoint by construction; 0 cross-city speakers measured).
3. Then add **whole seeded speakers** from the train cities: eligibility floor
   ≥ 3 minutes of speech **within the dataset rows** (not corpus-wide),
   seeded shuffle, add speaker-by-speaker until validation reaches ~20% of
   total dataset hours.
4. Rows with `speaker_id == null` (~9.5% of corpus utterances) can never go to
   validation — speaker disjointness can't be guaranteed for them; they stay in
   train.
5. Guards: `assert_no_leakage` on `speaker_id` (excluding nulls), plus no
   held-out-city row in train. Note: meetings in the *train* cities are NOT
   meeting-disjoint across splits — a meeting can contribute train rows (its
   train speakers) and validation rows (its val speakers). That is inherent to
   the agreed hybrid split (whole speakers, not whole meetings, from the mixed
   cities); the held-out cities remain fully meeting-disjoint.
   Uniqueness on `(city_id, meeting_id, utterance_id)`; if validation lands
   outside 18–22% of hours the script **reports and stops** for a human
   decision instead of silently accepting.
6. The seed, snapshot date, and full speaker→split map are written to
   `data/hf-dataset/split_assignments.json` and published alongside the data
   (auditability requirement from the publish plan).

## Boundary pass (closes the 2026-07-03 known issue in decisions/audio.md)

Raw CSV `utterance_start/end` can cut mid-syllable or bracket neighbouring
speech. Per row:

1. Slice the clip in memory (decode once per meeting file, slice PCM — the
   `prepare_asr.py` pattern).
2. **Forced alignment** of the label `text` onto the clip (local CTC forced
   aligner, CPU-capable, free — exact library pinned at implementation time).
   - first/last token aligns flush against a clip edge, or fails to align →
     `suspect_cut_start` / `suspect_cut_end`;
   - VAD-positive speech in the clip beyond the aligned span edges →
     `suspect_bleed_in`;
   - aligner returns nothing usable → `align_failed`.
3. **VAD snap + padding**: propose `start_adj/end_adj` snapped to silence with
   ±0.2s padding, clamped to available audio.
4. Nothing is dropped or auto-corrected beyond the snap. Suspects go to
   `data/hf-dataset/boundary-audit.csv` for sampled human review. Pulling
   neighbouring-utterance text into the label ("stitch the bled syllable") is
   an explicitly deferred second iteration.
5. Optional `--soniox-crosscheck N`: re-transcribe a random N suspects via the
   async API (per-token timestamps) as an accuracy spot-check — off by default
   (API spend).

## Outputs (`data/hf-dataset/`)

- `train.parquet`, `validation.parquet` (+ `.jsonl` mirrors)
- `split_assignments.json` (seed, snapshot date, speaker→split)
- `stats.json` + `stats.md` — total hours; hours + % per split; per-city and
  per-category breakdowns per split; speaker counts; `has_overlap` count;
  `boundary_status` histogram; every filter's drop count with reason
- `overlap-notes-report.md`, `boundary-audit.csv`
- `README.md` — HF dataset card draft: size/hours, cities/meetings/speakers,
  split methodology (held-out cities + seeded ≥3-min speakers, ~80/20 by
  hours, temporal test withheld), category distribution, provenance, caveats
  (curation bias, boundary status semantics, overlap flag semantics, license
  pending confirmation).

Publishing to the HF Hub (`huggingface_hub`) is a **separate, manual step** —
the script prepares the folder; the push command is documented in the README
but never run automatically.

## Error handling / invariants

- No silent caps or drops: every filter logs counts + reason (repo-wide rule).
- Composite key uniqueness enforced; duplicate → hard error.
- Val-share outside 18–22% → stop for decision.
- Export snapshot + seed pinned → identical rerun gives identical splits.

## Out of scope (this iteration)

- Embedded audio clips (release 2, license-gated).
- Transcribing overlapping speech; neighbour-text stitching.
- Any DB/VPS write-back.
