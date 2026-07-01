# On-box brief: next training batch via auto-selection + Soniox faithfulness filter

**Audience:** a Claude Code agent running on the mini PC (`ssh minipc`), in the
synced repo `~/opencouncil-fine-tuning`. **Owner stays in the loop** — calibration
thresholds and any DB/VPS write are human-gated (Angelos).

**What this is:** build a *second* training batch by **automatically** selecting
the most useful corrections from the un-curated remainder, instead of hand-picking
them. Two engines: (1) a candidate-selection filter over the corrections CSV, and
(2) a Soniox re-transcription "faithfulness" filter that keeps only examples whose
final corrected text is trustworthy against the audio.

This aligns with the team's "no human picking of speakers/meetings" objectivity
principle (`docs/decisions/data.md`, 2026-06-16).

---

## 0. Read first (don't skip — grounds every choice below)

- Research conclusion (the metric/threshold design): `data/reports/finetune-research/soniox-faithfulness-threshold.md`
- Data facts & exclusions: `docs/decisions/data.md`
- Already-decided segment length: `docs/reference/training-unit-granularity.md` (20–30s, **not** bare 10s)
- Error-type division (finetune vs LLM): `docs/specs/error-division.md`
- Existing reusable code in `eval/` (read signatures before calling):
  - `eval/scoring.py` → `greek_normalize()` **is exactly the normalization the research wants** (NFD accent-strip, ς→σ, punctuation, lowercase, whitespace). Reuse it. Add a CER helper next to it.
  - `eval/exclusions.py` + `data/exclusions/unreviewed_meetings.json` → canonical meeting denylist, keyed by `(city_id, meeting_id)`.
  - `eval/chains.py`, `eval/build_chain_report.py` → edit-chain logic (chain length / iterations). Reuse, don't reinvent.
  - `eval/segment.py` → utterance→segment grouping (for the 20–30s unit).
  - `eval/sample.py`, `eval/splits.py` → sampling / split helpers.
- **Note:** `eval/backends.py` is LLM text backends (claude/codex/gemini), **not ASR**. ASR re-transcription comes from `~/projects/soniox-tools/` (below).

## Inventory (verified present on the box, 2026-06-30)

- Corrections CSV: `~/opencouncil-fine-tuning/data-1779206108158.csv` (246 MB, ~393,970 rows = **all edit-chain rows**, not just latest-per-utterance).
  Columns: `edit_id, utterance_id, edit_timestamp, edit_updated_at, before_text, after_text, edited_by, utterance_start, utterance_end, audio_url, youtube_url, meeting_name, meeting_date, meeting_id, city_id`.
  → `edited_by` (user vs LLM), per-utterance chain (multiple `edit_id` per `utterance_id`), and the audio span (`utterance_start/end` + `audio_url`) are all here.
- Python env: `~/opencouncil-fine-tuning/.venv-eval` (pandas, rapidfuzz present).
- Soniox tools: `~/projects/soniox-tools/` (separate dir).
  - `file_transcribe.py` — **realtime path, works headless now** (mints its own Perplexity temp key via Chrome+curl_cffi). Use for the **calibration sample** (no API key needed). ~1× audio wall-time; key auto-renews. See its `RUN.md`.
  - `async_transcribe.py` — **batch path, needs a real `SONIOX_API_KEY`** in `~/projects/soniox-tools/.env`. Use for **full bulk** (faster, returns per-token confidence + timestamps). Set the key before scaling.

## Prerequisites / guardrails

- Ensure repo + this brief are current on the box (owner rsyncs from the laptop).
- **Never run git on both machines at once** (laptop ↔ minipc).
- Run long jobs in `tmux` so they survive disconnects.
- **No silent caps:** every filter step must log how many rows it dropped and why.
- **Do not touch the live DB or the VPS export** (step 7) without an explicit human go.
- Checkpoint intermediate outputs to `data/next-batch/` so a crash doesn't lose work.

---

## Plan (calibration-first; thresholds are set by a human, not guessed)

### Step 1 — Build the candidate pool (CSV → features)
From `data-1779206108158.csv`, produce one row per `utterance_id` (latest edit as
the target) **plus** chain features. Apply these filters, logging drop counts:

1. **Drop LLM-only edits.** Keep an utterance only if a human touched it: its chain
   contains `edited_by == 'user'` (user-only, or LLM→user). Drop chains that are
   purely LLM. *(This is the "more interesting corrections" subset.)*
2. **Drop punctuation/whitespace-only & degenerate edits.** Reuse the ingest
   categorisation rationale in `docs/decisions/data.md` (`noop_edit, empty_after,
   whitespace_only`, and punctuation-only diffs under `greek_normalize`).
3. **Apply meeting exclusions** via `eval/exclusions.py` (the unreviewed-meeting denylist).
4. **Compute features** per utterance: chain length (# edits / iterations),
   `char_diff = levenshtein(before, after)` (raw, surface), normalized word-diff,
   audio span duration (`utterance_end - utterance_start`), `edited_by` path.

Output: `data/next-batch/candidates.parquet` + a one-page summary (counts dropped
per filter, distribution of chain length / duration / char_diff).

**Acceptance:** counts reconcile with `docs/decisions/data.md` numbers; no row kept
that an exclusion should remove.

### Step 2 — Calibration sample + Soniox re-transcription
- Draw a **stratified sample of ~200–500** candidates (stratify by chain length and
  char_diff so the audit covers easy→hard). Use `eval/sample.py` if it fits.
- For each: download `audio_url`, slice `[utterance_start, utterance_end]` with
  `ffmpeg`, re-transcribe with **`file_transcribe.py --lang el`** (realtime path —
  no key needed). Cache clips + transcripts under `data/next-batch/calib/`.

**Acceptance:** ≥95% of the sample transcribed without error; failures logged with reason.

### Step 3 — Compute the faithfulness metric → BUCKETS (not binary)
For each calibration item, under `greek_normalize`:
- `cer_soniox = CER(soniox_text, after_text)`
- `cer_before = CER(before_text, after_text)`

Emit a **bucket**, not a keep/drop boolean (Codex: the "both-high" bin hides genuine
hard cases AND bad spans — never auto-trust it). Starting gates (calibrate in step 4):
- **DROP / bad-label:** `cer_before ≤ 0.08` AND `cer_soniox ≥ 0.18` AND `cer_soniox ≥ 2.5 × max(cer_before, 0.04)`.
- **KEEP / trusted-hard:** `cer_before ≥ 0.12` AND `cer_soniox ≤ 0.12–0.15`.
- **AUDIT / both-high:** `cer_before ≥ 0.15` AND `cer_soniox ≥ 0.15` → keep ONLY if aux signals good.
- **BACKBONE / no-edit:** `cer_before ≤ 0.03` AND `cer_soniox ≤ 0.08–0.10` (feeds step 6 backbone).
- **SHORT-SPECIAL:** if `after` has `<20` normalized chars or `<5` words, route to a
  separate short-utterance path — plain CER is unstable; use exact/near-exact lexical
  match + Soniox confidence, and treat names/numbers/acronyms as high-weight (one char = big meaning).

**Aux signals** (compute alongside; used to break ties and weight the AUDIT bucket):
- duration/word-rate guard — flag if speech rate `<1` or `>4.5` words/s, or clip `<1.5s` / `>30s` (span misalignment is the biggest real failure mode).
- **Soniox per-token confidence** — `async_transcribe.py` returns it. High-confidence Soniox
  disagreement with the label is strong bad-label evidence; low-confidence disagreement is weak →
  soften those. (Do NOT replace CER with confidence at first; use it to weight.)
- length ratio, insertion/deletion asymmetry, whether disagreement lands on names/numbers.

Add `cer(a, b)` next to `eval/scoring.py:greek_normalize` using `rapidfuzz.distance.Levenshtein` on characters of the normalized strings.

### Step 4 — HUMAN GATE: set thresholds
Emit `data/next-batch/calib_audit.csv` (clip path, before/after/soniox, the two CERs,
the rule's verdict) for **Angelos to hand-audit**. Plot the CER distributions; pick
the cut where good/bad separate (ROC-style). **Do not hard-code 0.10 / 15% — those
are starting guesses; the audited cut wins.** Record the chosen thresholds back into
this runbook and `docs/decisions/data.md`.

### Step 5 — Scale to full bulk (after thresholds locked)
- Put a real `SONIOX_API_KEY` in `~/projects/soniox-tools/.env`; switch to
  `async_transcribe.py` (batch, per-token confidence).
- Re-transcribe the full candidate pool (or the most promising slice), apply the
  locked thresholds + guard. Checkpoint heavily; this is the long run (tmux).

### Step 6 — Final selection (~5k corrections / ~30h speech) + segmentation
From the faithful candidates (KEEP + vetted AUDIT), select the batch targeting
**~30 hours** of audio, composed roughly as:
- **50–60%** high-confidence hard corrections (high chain-length ≥3, larger char_diff)
- **20–30%** under-represented correction *families* (novelty)
- **10–20%** common high-impact recurring errors (the model still needs repetition)

**Diversity weighting (coverage-first, then controlled repetition — do NOT hard-exclude
common types):**
- Characterize each correction by a **deterministic edit-signature** first (accent/ς/casing/
  punctuation, word-substitution, insert/delete, NE change, number/date, inflection ending,
  function-word, compound split/merge, homophone, hallucination-removal, dropped-phrase). Use
  diff-of-before→after, not full-sentence embeddings (those cluster by topic). Embeddings only
  as a secondary pass to find semantic families.
- Diminishing-returns weight per type vs (batch-1 + already-selected):
  `type_weight = 1 / sqrt(1 + count_type)`. "Enough" per type: orthographic/punct ~20–50;
  lexical sub ~50–150; morphological endings ~100–300 across varied contexts; named entities —
  broaden coverage rather than repeat one (5–30/entity). Tie to `docs/specs/error-division.md`.

**Segmentation to the training unit (mixed lengths, not 10s-only):**
- target ~**60–70% at 15–30s**, **20–30% at 5–15s**, **5–10% near-30s**; avoid `<3s` except
  names/interjections/vote responses.
- concatenate neighbouring utterances **within the same speaker turn only**, boundaries on
  silence, no overlap, `≤30s`. Reuse `eval/segment.py`.
- **First run: train WITHOUT timestamp tokens** (`--without_timestamps`) — timestamp quality
  is a separate, later experiment with clean labels.

**Backbone:** also assemble **20–30% trusted no-edit utterances** (the BACKBONE bucket from
step 3, fully-reviewed meetings only — never random untouched ASR), stratified (short/long/
names/numbers/varied speakers). Track them **separately** from the 5k corrections; the *training
mix* must not be corrections-only (correction-bias trap → over-editing). Final mix at train time:
batch-1 includes + batch-2 corrections + 20–30% backbone.

Output: `data/next-batch/selected.jsonl` (+ `backbone.jsonl`) + stats (hours, type distribution,
overlap with batch-1).

**Acceptance:** total audio within ~±10% of target hours; type distribution reported and
visibly broader than batch-1's; length mix within the bands above; every selected item passed
the faithfulness gate; backbone sampled only from fully-reviewed meetings.

### Step 7 — Write-back (DEFERRED — separate sub-project, human-gated)
Add a faithfulness score/flag to the DB and (carefully, reversibly) the VPS export.
**Do not start without an explicit go** — it touches production. Mirror the
reversible, query-time posture used for exclusions in `docs/decisions/data.md`.

---

## Resolved decisions (research + Codex review, 2026-07-01)
- **Segment length:** MIX, not 10s-only — ~60–70% @15–30s, 20–30% @5–15s, 5–10% near-30s;
  concatenate within speaker turn, ≤30s; **no timestamp tokens in the first run**. (Folded into step 6.)
- **Diversity:** deterministic edit-signature buckets primary, embeddings secondary;
  diminishing-returns weighting `1/sqrt(1+count)`, NOT hard caps; coverage-first then
  controlled repetition. (Folded into step 6.)
- **Non-correction utterances:** include **20–30% trusted no-edit backbone** (fully-reviewed
  meetings only), tracked separately; training mix must not be corrections-only. (Folded into step 6.)
- **Faithfulness:** bucketed verdicts (not binary) + short-utterance special path + Soniox
  confidence weighting + duration guards; starting CER gates in step 3. (Folded into step 3.)

**Still genuinely open (needs Angelos / step-4 audit):** the *exact* CER thresholds (the step-3
gates are Codex starting points); final batch-2 size if it overshoots ~30h; whether to mix in a
little generic Greek ASR data only if domain overfitting appears.

## Status hook
When a step lands, update `docs/progress.md` and leave a short note in
`docs/logs/` per the vault's history rules. Keep `data/next-batch/` outputs and
this runbook the source of truth for what ran.
