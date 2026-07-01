# 2026-07-01 — Auto-selected batch-2 via faithfulness + interestingness

Ad-hoc log: an implementation milestone landed (automated batch-2 selection), and
the Soniox faithfulness metric was validated on real data. Runbook and
`data/next-batch/` outputs are the source of truth; this is the short history note.

## What ran (on-box, `harold-venusseries`, repo `~/opencouncil-fine-tuning`)

End-to-end automated selection of a second training batch from the un-curated
remainder, per [the runbook](../runbooks/next-batch-selection-onbox-brief.md). No
human hand-picking of speakers/meetings — aligns with the objectivity principle in
[decisions/data.md](../decisions/data.md).

Pipeline (code under `eval/next_batch_step*.py`, reuses `eval/scoring.py`,
`eval/categorize.py`, `eval/exclusions.py`, `eval/chains.py`):

1. **Candidate pool** — `candidates.parquet`, **93,584** utterances. Funnel:
   393,970 raw → 287,605 utterances → 156,863 human-touched → drop 35,628
   normalisation-only + 5,284 empty + 245 denylist + **22,030 held-out eval
   meetings (leakage)** = 93,584. Every drop logged (`step1_summary.md`).
2. **Soniox calibration** — stratified 320-clip sample, ffmpeg HTTP-seek + Soniox
   realtime re-transcription. **320/320 transcribed, 0 failures.**
3. **Faithfulness metric (validated)** — `cer_before` = how much the human changed
   the STT; `cer_soniox` = how far a fresh independent ASR is from the human label
   (both under `greek_normalize`; new `cer()`/`wer()` helpers in `eval/scoring.py`).
   Buckets separate cleanly: **KEEP** surfaced confirmed fixes
   (`στατηρικης→στρατηγικης`, Soniox agrees); **DROP** surfaced bad labels where the
   audio disagrees with the human edit. Human-gate audit sheet + CER plot emitted
   (`calib/calib_audit.csv`, `calib/cer_distributions.png`) — exact thresholds are
   still Angelos's to lock.
4. **Interestingness ranking** (formula reviewed with Codex) — free text-only score
   (category weight × correction magnitude × chain length × duration × rewrite
   guard) + deterministic edit-signatures + lazy-greedy diverse selection →
   **15,000 shortlist** (14,370 distinct edits, 11 cities, 178 meetings).
5. **Sonnet text-plausibility triage** — `claude -p` sonnet judged all 15,000:
   **7,364 keep**, 5,967 reject, 1,669 unsure. Removes semantic rewrites,
   formatting-only, and implausible labels the text-only score can't catch.
6. **Final list** — `selected_edits.jsonl`, **7,364 edits**, 99.8% acoustic, 7,173
   distinct specific corrections, 177 meetings, 11 cities, **~5.4 h** raw span audio
   (→ ~30 h target after 15–30 s speaker-turn concatenation). Category mix inside
   Codex's bands (`step7_summary.md`).

## Decisions / notes

- The old `data/asr/train_manifest.csv` (93,864 rows, 119 meetings) is an **ad-hoc
  export with no committed builder** — it covered fewer meetings AND leaked 17,237
  rows from 30 held-out test meetings. The new pool is broader (178 meetings) and
  leakage-free. Nothing valuable was lost.
- Soniox realtime path works headless on this box (temp key auto-mint). The async
  **batch key is present** in `~/projects/soniox-tools/.env`, so the paid gold
  faithfulness pass at scale is unblocked.

## Open / next (human-gated)

- **Soniox gold-faithfulness pass** on the 7,364 (or the top ~12k shortlist) to
  audio-verify each label and drop the DROP-bucket cases — real API spend, awaiting
  Angelos go. This is the brief's Step 5.
- Lock the exact CER thresholds from the calibration audit (brief Step 4).
- Segmentation to 15–30 s units + no-edit backbone assembly (brief Step 6) before
  training.
