# Progress

Planned timeline lives in the [GSoC proposal](reference/gsoc-proposal.md#timeline-12-weeks--350-hours). This file tracks where we actually are.

Status: `not started` · `in progress` · `done` · `blocked`.

GSoC weeks 1–12 are the coding-period weeks. Pre-coding work (vault setup, dataset exploration, review UI) runs before week 1 and is tracked separately below.

**Review progress (2026-06-09):** ~700 utterances reviewed. Revised target
**~6k by mid-July** (the May-26 cadence aimed at ~2k by Jun 12; we are behind it
but the mid-July dataset target is the binding one). Finetuning background and
plan: [reference/finetuning-101.md](reference/finetuning-101.md). Split, baseline,
and GPU decisions: [decisions/data.md](decisions/data.md).

**Batch-2 auto-selection (2026-07-01):** end-to-end automated selection landed —
**7,364 fine-tune edits** chosen from the 93,584-candidate un-curated remainder via
a validated Soniox faithfulness metric + Codex-designed interestingness ranking +
Sonnet text triage. Outputs under `data/next-batch/`, code under
`eval/next_batch_step*.py`. See
[logs/2026-07-01-next-batch-selection.md](logs/2026-07-01-next-batch-selection.md).
Pending human-gated steps: lock CER thresholds, Soniox gold-faithfulness pass,
segmentation + no-edit backbone.

## Pre-coding (community bonding / preparation)

| Focus                                                                                                  | Status        | Evidence                                                                                                                                |
| ------------------------------------------------------------------------------------------------------ | ------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Vault organisation, decisions, roadmap, agent instructions                                             | done          | [logs/2026-05-12-vault-setup.md](logs/2026-05-12-vault-setup.md)                                                                        |
| Full CSV ingest with content categorisation                                                            | done          | [decisions/data.md](decisions/data.md#2026-05-16---categorise-all-csv-rows-instead-of-dropping)                                         |
| Local SQLite/JSONL prototype, Supabase Postgres hosted state                                           | done          | [decisions/storage.md](decisions/storage.md)                                                                                            |
| Correction-review UI (diff, audio, labels, stats, export)                                              | in progress   | [specs/exploration-ui.md](specs/exploration-ui.md), [ui/README.md](../ui/README.md)                                                     |
| Cached meeting JSON + correction-to-utterance matching                                                 | not started   | [decisions/matching.md](decisions/matching.md#correction-to-utterance-matching-strategy)                                                |
| Audio CORS / decoding workaround (proxy + URL map)                                                     | in progress   | [decisions/audio.md](decisions/audio.md#2026-05-16---audio-workaround-via-vercel-proxy-and-fixed-file-map-pending-proper-fix); 2026-06-20: fixed slow-link audio — prefetch pool was over-warming (~24 `<audio preload=auto>` each pulling a whole meeting mp3, starving the link); now bounded to current+neighbours with only current+next at `preload=auto` (commit `e1be71d`) |
| Stable-ID corrections export from mentors                                                              | done          | [decisions/data.md](decisions/data.md#2026-05-19---stable-ids-export-arrived) — landed 2026-05-19, ingested via `ingest-csv-v2.ts`      |
| Latest-per-utterance reduction (live DB stores one row per utterance)                                  | done          | [decisions/data.md](decisions/data.md#2026-05-19---keep-only-the-latest-edit-per-utterance) |

## GSoC weeks 1–12 (planned)

| Week | Planned focus (from proposal)                                                                                 | Status      | Evidence |
| ---- | ------------------------------------------------------------------------------------------------------------- | ----------- | -------- |
| 1    | Dataset work: error auto-classification, audio preprocessing (noise reduction, VAD), 2–5 min segments         | not started |          |
| 2    | Dataset work continued: train/val/test split with two held-out municipalities + date-based test split         | not started |          |
| 3    | Dataset finalisation; publish reproducible dataset on HuggingFace                                             | not started |          |
| 4    | Baseline evaluation: Gladia WER/CER reconstructed from `UtteranceEdit.beforeText`                             | not started |          |
| 5    | Baseline evaluation: zero-shot Whisper-large-v3, Charalampos/whisper-medium-el, Cohere transcribe. **M1**     | in progress | zero-shot large-v3 baseline measured on held-out cities (val_corr WER 33.4, val_reg 27.1) in the 2026-06-24 smoke run |
| 6    | LoRA fine-tuning, hyperparameter sweep (rank, alpha, learning rate)                                           | in progress | turbo sweep ran 6.3/9 configs before the 12h guard stopped it (per-epoch flush kept all 19 rows). Every config beat baseline on BOTH sets — val_corr 24.0→~21, val_reg 23.0→~14 (no regression to manage). LR/rank within eval noise (~654 words); pick `lr1e-4 r32, 2 epochs`. Confirm on large-v3 (sweep was turbo proxy). Sweep + seeds/CIs still pending |
| 7    | LoRA fine-tuning continued; evaluate on held-out municipalities                                               | not started |          |
| 8    | Target morphological errors and domain terminology                                                            | not started |          |
| 9    | Ablation studies (concatenation, data subsets). **M2: ≥15% relative DS-WER over Gladia baseline**             | not started |          |
| 10   | CTranslate2 conversion; benchmark inference (target RTF < 0.5)                                                | not started |          |
| 11   | Plug `WhisperTranscriber` into production pipeline; end-to-end tests on new meetings                          | not started |          |
| 12   | Documentation, reproducibility scripts, final evaluation report. **Final milestone**: merged into OpenCouncil | not started |          |

Update Status / Evidence as work lands. The Planned focus column mirrors the proposal — change it there first, then here.
