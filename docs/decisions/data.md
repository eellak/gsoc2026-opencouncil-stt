# Data decisions

CSV ingest, content categorisation, stable IDs, task version, dataset split.

## Accepted

### 2026-06-19 - Drop degenerate ingest bins from review + export (query-time)

The corpus carries ~10.5k utterances whose **latest edit** falls in a
content-problematic `ingest_category` (tagged at build time by `categorise()` in
`scripts/lib/csv-clean.ts`). Three of those bins are degenerate — they hold no
real correction signal — and are now excluded from the review queue **and** the
export so the final dataset is clean:

- **Dropped by default** (`DROP_INGEST_CATEGORIES`, default `noop_edit,empty_after,whitespace_only`): **8,121** utterances (2.82%).
- **Kept** (likely legit): `empty_before` (1,731 — insertion from nothing, may be a real ASR miss), `multiline_text` (623), `embedded_reasoning` (20 — already cleaned at ingest).

Applied as a **query-time view filter**, the same reversible posture as
[meeting-eligibility](#2026-06-03---exclude-meetings-with--10-human-corrected-utterances-from-review)
— **not** a rebuild of the 579 MB SQLite index. Both repos subtract degenerate
ids from `eligibleOrderedIds()` after the meeting filter; `/api/export` applies
the same policy independently (Codex review: two layers, one shared policy).
`getGroup(id)` still resolves dropped utterances and nothing in `.state/` is
touched, so it is fully reversible (`DROP_INGEST_CATEGORIES=""` disables it).
Unknown categories fail closed (throw) rather than silently shipping unfiltered.

Of the 8,121 dropped, only **2** carried a human decision — both `whitespace_only`
already marked `exclude`, so hiding them is consistent with the reviewer. The
`.state/` review work was backed up first to
`ui/.state-backups/state-2026-06-19.tar.gz` (302,501 events + labels, verified).

Files: `ui/src/lib/server/state/ingest-filter.ts`, `repo/{sqlite,file}-repo.ts`
(`applyIngestCategoryFilter`), `routes/api/export/+server.ts`. Audit report +
machine-readable sidecar: `data/reports/ingest-filter-2026-06-19.{md,json}`
(regenerate via `bun scripts/report-ingest-filter.ts`).

This is distinct from the runtime **skip** mechanism (private/404 + below-threshold
meetings) discussed for the prefetch bug — degenerate bins were never the cause of
the audio latency, and this change does not touch the prefetch/audio path.

### 2026-06-16 - Split mechanics: temporal test set + seeded automated train/val

Refines [Split by whole meeting](#2026-06---split-trainvaltest-by-whole-meeting-not-by-utterance)
with the concrete mechanics agreed in the [2026-06-16 sync](../meetings/2026-06-16.md):

- **Test set = meetings/cities reviewed from 1 Jun 2026 onward.** A temporal
  hold-out: training never touches current review work, so it is naturally
  uncontaminated, and it includes newly-onboarded cities and unseen speakers —
  which answers the business question "does it work on a council we onboard
  tomorrow?".
- **Train/Val = pre-June data, split by a seeded, automated program** that reads
  the data and assigns whole meetings + whole speakers to a set. **No human
  picking** of speakers or meetings (Christos): letting reviewers nominate "hard"
  speakers biases the set; we want objective metrics. Reproducible via seed.
- Hold out **entire jurisdictions** where affordable (e.g. all speakers of one
  city) into the held-out set; take train from what remains.
- Target ratio **~30/70**; the program must ensure **both sets have sufficient
  data** (enough hours to validate/test on).

Open detail: whether 30/70 is train/val or val/test, and the exact speaker/meeting
membership, is specified next week. See [open questions](#open).

### 2026-06-16 - Cost is not a deciding factor for the provider

Christos: optimise for transcription **quality**, not price. On the
[`2026-06-10-oc-benchmark`](../meetings/2026-06-16.md) run, **Scribe v2 (13.4%
WER)** is the best provider and stays the current choice; Gladia (14.7%, best CER)
is what runs on prod today; zero-shot **whisper-large-v3 (15.0%)** is the finetune
baseline to beat. Soniox landed between Gladia and Scribe, so being cheaper does
not change the choice. A naive Greek finetune on HF (sam8000) scored 48.7% — 3×
worse than zero-shot — evidence that careful data, not just "finetune on Greek",
is what matters.

### 2026-06 - Split train/val/test by whole meeting, not by utterance

Team consensus (sync 2026-06-02 + Discord follow-up). The train/val/test split
unit is the **whole meeting**: every utterance of a meeting belongs entirely to
one split. Random per-utterance shuffling puts the same speaker and mic in both
train and test, so a good score would measure memorisation, not generalisation.

Where affordable, go further: hold out specific **speakers** entirely, and hold
out one entire **municipality** as test (answers "does this work on a council we
onboard tomorrow?"). Background and rationale:
[../reference/finetuning-101.md](../reference/finetuning-101.md#our-dataset).
This is consistent with the GSoC-proposal plan (two held-out municipalities +
date-based test split).

### 2026-06 - Baseline first, before any finetuning

Measure the baseline before training anything: (1) how **Gladia** (current
provider) does, and (2) how **zero-shot Whisper** does without finetuning. WER is
the first metric; richer metrics (NE-WER, per-category WER, corrections/hour)
come next. Reason (Christos): we can't claim an improvement without a measured
starting point, and the eval harness is the highest-leverage thing to build
first. See [finetuning-101 → what to measure](../reference/finetuning-101.md#what-to-measure).

### 2026-06 - GPU strategy: rent for training, mini PC for eval/dev

Real training runs go on **rented GPUs** (Runpod/Vast: 4090 ~$0.34/hr for dev,
A100 ~$1.50/hr for runs; first LoRA runs ~$10–50). Angelos's mini PC (7840HS /
780M iGPU / 64 GB) is **not** used for training — the 780M (gfx1103) lacks
official ROCm support and shared-RAM bandwidth/compute make a training run take
weeks. It IS used, for free, for the **eval harness + baseline WER** (CPU
faster-whisper/CTranslate2 INT8), **pipeline smoke tests**, **LoRA dev/debug** on
tiny subsets, and **data prep**. Optimise for engineering throughput, not GPU
cost. Details: [finetuning-101 → mini PC](../reference/finetuning-101.md#where-the-mini-pc-fits-local-compute).

### 2026-06-03 - Exclude meetings with < 10 human-corrected utterances from review

The review queue, filters, and stats now only include meetings with at least **10 human-corrected utterances** — distinct utterances carrying ≥1 edit with `edited_by === 'user'`. Applied as a query-time eligibility filter, **not** a DB rewrite: a one-time scan computes the eligible meeting set, persists `ui/.state/meeting-eligibility.snapshot.json` (keyed by `cache_hash` + `threshold`), and the repo derives `eligibleOrderedIds` from it. `getGroup(id)` stays unfiltered so direct links still resolve; nothing is deleted and labels/flags in `.state/` are untouched — fully reversible (set `MEETING_MIN_HUMAN_UTTERANCES=0` to disable). Threshold default 10 via `MEETING_MIN_HUMAN_UTTERANCES`.

**`meeting_id` is not unique — key by (city_id, meeting_id).** The `meeting_id` slug (e.g. `feb26_2025`) is reused across cities; the corpus has **105** such collisions, so there are **411** real meetings under only 242 distinct slugs. Eligibility counts (and the eligible set) are therefore keyed by the `(city_id, meeting_id)` pair via `meetingKey()` — keying by the slug alone merged distinct meetings and **kept** ones that should be excluded (e.g. Athens `feb26_2025` "6η Ειδική Συνεδρίαση Λογοδοσίας", 9 human utterances, was wrongly retained because it shares the slug with the eligible Chania `feb26_2025`).

On the live corpus (correct keying): **384/411 meetings eligible, 27 excluded, 18 123 utterances removed** (269 482 / 287 605 remain). The 27 excluded include 14 fully task-only meetings (0 human corrections) and 13 with 1–9.

Reason: meetings whose edits are essentially all task-generated, with few human corrections, aren't useful review/training targets. Filtering them at query time keeps reviewer effort on corrected meetings without re-uploading the 579 MB SQLite index or losing any labeling already done.

Files: `ui/src/lib/server/state/meeting-eligibility.ts` (incl. `meetingKey`), `repo/{sqlite,file}-repo.ts`, `state/stats-cache.ts`, `routes/api/review/ids/+server.ts`.

### 2026-05-12 - Do not block on the CSV export query

The exact external query that produced `utterance-edits-may12-26.csv` is not required before we can proceed.

Reason: the CSV already has edit text, timestamps, audio URL, meeting name, and meeting date. We can use these to join against cached meeting JSON for the prototype.

### 2026-05-12 - Task version is not required for first exploration

`TaskStatus.version` is useful for rigorous historical baseline analysis, but not required for the first dataset exploration UI.

Reason: the current job is to inspect corrections and classify error types, not to prove model-version regressions.

### 2026-05-12 - Waiting on a new corrections export with stable IDs

`meeting_name` + `meeting_date` is **not** sufficient to identify a meeting uniquely. Mentors have been asked for a new export that includes stable identifiers (likely `utterance_id`, `meeting_id`, `city_id`, `speakerSegmentId`).

Implication: until that export arrives, any matcher we write is a workaround. Once it arrives, matching becomes a direct ID join.

### 2026-05-16 - Categorise all CSV rows instead of dropping

All 379 194 rows are loaded into the database regardless of quality. Each row receives an `ingest_category` column (`clean`, `noop_edit`, `whitespace_only`, `empty_before`, `empty_after`, `embedded_reasoning`, `reversed_timestamps`, `multiline_text`) and a `cleaning_applied` tag list.

Reason: dropping rows at ingest time is irreversible and makes it impossible to inspect what was excluded. With categories in the DB, the stats UI can show counts per category and the user can browse even the rejected bins. Future re-ingestion (when the new CSV arrives with stable IDs) will re-run the same categorisation.

CSV is structurally clean (RFC-4180 quoted, parseable by `csv-parse`). Issues are content-level only: 5 219 no-ops, 1 384 multiline, 1 219 whitespace-only, 1 950 empty-before, 3 812 empty-after, 27 embedded-reasoning. 96.4% categorise as `clean`.

Scripts: `ui/scripts/analyze-csv.ts` (read-only report) and `ui/scripts/ingest-csv.ts` (full ingest with categorisation). Library of pure transforms in `ui/scripts/lib/csv-clean.ts` with unit tests.

### 2026-05-19 - Stable IDs export arrived

A second export landed at the repo root (`data-1779206108158.csv`, ~246 MB, 397 556 rows) carrying the IDs that were pending in [2026-05-12 - Waiting on a new corrections export with stable IDs](#2026-05-12---waiting-on-a-new-corrections-export-with-stable-ids):

- `utterance_id` (the stable utterance identifier)
- `meeting_id` (FK into the new normalised `meetings` table)
- `city_id`

`ui/scripts/ingest-csv-v2.ts` reads this CSV and upserts directly into the Postgres schema. The first ingest landed 393 970 rows after CSV-level filtering.

### 2026-05-19 - Normalise meetings out of corrections

`meeting_name`, `meeting_date`, `city_id`, `audio_url`, `youtube_url`, and `audio_cdn_url` are no longer stored per-correction. They live on a new `meetings` table (PK `meeting_id`, 242 rows) and `corrections` joins via `meeting_id`.

Reason: the unnormalised layout duplicated those columns across hundreds of thousands of rows for only 242 distinct meetings — roughly 110 MB of redundant text plus the index overhead. With normalisation the corrections table dropped from 463 MB to ~106 MB and stays well under the Supabase free-tier ceiling.

Implementation note: `corrections.audio_cdn_url` was moved to `meetings.audio_cdn_url`; `scripts/apply-audio-cdn-map.ts` still writes the same key, just on a different table.

### 2026-05-19 - Keep only the latest edit per utterance

`corrections` stores **one row per `utterance_id`** — the most recent edit, ordered by `COALESCE(edit_updated_at, edit_timestamp) DESC, edit_id DESC`. The 106 365 superseded chain edits are not kept in the live DB.

Reason: for the training/evaluation dataset the only useful signal is the final corrected text. Intermediate edits in a chain are noise: they capture transient states (e.g. mid-typo, accidental space, partial paste) that the reviewer themselves discarded in the next edit. Loading the chain into the review UI also wastes reviewer time on rows that are already known to be superseded.

Numbers from the CSV (see [data/reports/latest-per-utterance.md](../../data/reports/latest-per-utterance.md) for distribution and worked examples):

- 287 605 unique utterances total
- 70.2 % had a single edit (no chain)
- 27 % had a 2- or 3-edit chain
- ~0.4 % had 5+ edits; the longest chain is 27 edits on one utterance

Implication: if at some point we want the full chain for audit or to study reviewer behaviour, we re-ingest `data-1779206108158.csv` into a separate `corrections_history` table — the CSV is the source of truth. The decision is reversible without data loss.

Implementation: `latest_per_utterance` flag computed via window function in `ui/scripts/ingest-csv-v2.ts` follow-up, non-latest rows deleted in batches, table compacted via `VACUUM FULL` to bring DB size from 568 MB to 215 MB.

## Open

### Training-unit granularity: utterances vs larger / speaker segments (raised 2026-06-23)

New question from the [mentor sync](../meetings/2026-06-23-mentor-sync.md). Edits
arrive **per utterance** from the review UI, so the obvious unit is the utterance.
But many ASR errors come from the audio being cut slightly before/after the
utterance (missing words at the start/end that are clear if you shift the boundary)
— Whisper self-segments 30-min/hour audio, the utterance boundaries are arbitrary.
So a larger unit may train better: a context window (±neighbouring utterances) or
the **whole speaker segment** the utterance belongs to, with **duplicate avoidance**
(if utterance −3 is also in the set, don't double-count). Decide from the
literature + what others do + Claude. Affects dataset build and the manifest.

### Meeting-trust cutoff: `humanReview` flag is unreliable (raised 2026-06-23)

Christos: `taskStatus.humanReview` is a "fake" status — some meetings lack it yet
are corrected (old / 2025 ones), so the flag alone over- and under-counts. Refine
the [humanReview gate](metric-hir.md): combine the flag with a **distribution of
the human-edit *fraction* (edits/total) per meeting** and pick a **threshold**
below which a meeting is not trusted (expect a cluster ~20–30% and a tail <5% to
drop — and don't sample "no-edit as ground truth" from the low-fraction ones).
Supersedes relying solely on the flag; complements the ≥20 human-edit count gate.

### Reviewer curation subjectivity (raised 2026-06-23)

Angelos rejects ~3 of 4 corrections during review. Different reviewers judge "worth
training on" differently, which biases the included set. Discuss with the whole team
(Eliza, Thanos) — calendar item. Relates to the 2026-06-16 "no human picking of
speakers/meetings" principle: the same bias concern applies to per-correction curation.

### Can we trust "non-corrected" utterances as ground truth?

The dataset plan wants ~50–70% non-corrected utterances from **fully-reviewed**
meetings as silent positives. Risk: if a reviewer can skip/skim, "non-corrected"
may just mean "not yet looked at." Before bulk-including them we need to verify
in the pipeline which meetings were *fully* reviewed (vs partially). Until then,
treat non-corrected utterances from partially-reviewed meetings as unlabelled.

### Audio normalization (raised 2026-06-02)

Do we normalize loudness in production? Per whole meeting, or per interval too?
Do we normalize for the finetuning training set as well? Whatever we choose, the
training feature extraction and inference must match (use HF
`WhisperFeatureExtractor` consistently). Undecided.

### Benchmark: fixed meetings vs random test set (raised 2026-06-02)

**Resolved 2026-06-16.** Two distinct things, not one:

- The `2026-06-10-oc-benchmark` run is a **provider comparison** on a frozen
  random sample across all meetings — kept as a scoreboard, never used as the
  finetune test set (its windows leak across train/test).
- The **finetune test set is a fixed temporal hold-out** (post-June meetings),
  per [2026-06-16 split mechanics](#2026-06-16---split-mechanics-temporal-test-set--seeded-automated-trainval).
  A separate benchmark restricted to those meetings gives release-defensible
  numbers, re-run at the end to compare baseline → finetuned.

### Human-edit threshold for eligible meetings: 10 → 20 (resolved 2026-06-16)

**Resolved: use 20.** The review tool previously treated a meeting as reviewable
at **≥10** human-edited utterances (`MEETING_MIN_HUMAN_UTTERANCES`, see
[exclude meetings with < 10](#2026-06-03---exclude-meetings-with--10-human-corrected-utterances-from-review)),
while the `2026-06-10-oc-benchmark` used **20**. Aligning on **20**: the cleaned
SQLite index (prefetch-bug fix) keeps only public meetings from the 10 public
cities with **≥20** human-edited utterances; private edits and below-threshold
meetings are dropped so no runtime skip is needed.

### Correction-bias mix ratio

Final ratio of corrected vs non-corrected utterances in the training set
(starting point ~30–50% corrections / ~50–70% non-corrected). To be tuned once
the eval harness can measure the effect. See
[finetuning-101 → correction-bias trap](../reference/finetuning-101.md#our-dataset).
