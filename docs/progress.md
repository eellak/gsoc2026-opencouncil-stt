# Progress

Planned timeline lives in the [GSoC proposal](reference/gsoc-proposal.md#timeline-12-weeks--350-hours). This file tracks where we actually are.

Status: `not started` · `in progress` · `done` · `blocked`.

GSoC weeks 1–12 are the coding-period weeks. Pre-coding work (vault setup, dataset exploration, review UI) runs before week 1 and is tracked separately below.

## Pre-coding (community bonding / preparation)

| Focus                                                                                                  | Status        | Evidence                                                                                                                                |
| ------------------------------------------------------------------------------------------------------ | ------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Vault organisation, decisions, roadmap, agent instructions                                             | done          | [logs/2026-05-12-vault-setup.md](logs/2026-05-12-vault-setup.md)                                                                        |
| Full CSV ingest with content categorisation                                                            | done          | [decisions/data.md](decisions/data.md#2026-05-16---categorise-all-csv-rows-instead-of-dropping)                                         |
| Local SQLite + JSONL state, Turso hosted state                                                         | done          | [decisions/storage.md](decisions/storage.md)                                                                                            |
| Correction-review UI (diff, audio, labels, stats, export)                                              | in progress   | [specs/exploration-ui.md](specs/exploration-ui.md), [ui/README.md](../ui/README.md)                                                     |
| Cached meeting JSON + correction-to-utterance matching                                                 | not started   | [decisions/matching.md](decisions/matching.md#correction-to-utterance-matching-strategy)                                                |
| Audio CORS / decoding workaround (proxy + URL map)                                                     | in progress   | [decisions/audio.md](decisions/audio.md#2026-05-16---audio-workaround-via-vercel-proxy-and-fixed-file-map-pending-proper-fix)           |
| Stable-ID corrections export from mentors                                                              | blocked       | [decisions/data.md](decisions/data.md#2026-05-12---waiting-on-a-new-corrections-export-with-stable-ids)                                 |

## GSoC weeks 1–12 (planned)

| Week | Planned focus (from proposal)                                                                                 | Status      | Evidence |
| ---- | ------------------------------------------------------------------------------------------------------------- | ----------- | -------- |
| 1    | Dataset work: error auto-classification, audio preprocessing (noise reduction, VAD), 2–5 min segments         | not started |          |
| 2    | Dataset work continued: train/val/test split with two held-out municipalities + date-based test split         | not started |          |
| 3    | Dataset finalisation; publish reproducible dataset on HuggingFace                                             | not started |          |
| 4    | Baseline evaluation: Gladia WER/CER reconstructed from `UtteranceEdit.beforeText`                             | not started |          |
| 5    | Baseline evaluation: zero-shot Whisper-large-v3, Charalampos/whisper-medium-el, Cohere transcribe. **M1**     | not started |          |
| 6    | LoRA fine-tuning, hyperparameter sweep (rank, alpha, learning rate)                                           | not started |          |
| 7    | LoRA fine-tuning continued; evaluate on held-out municipalities                                               | not started |          |
| 8    | Target morphological errors and domain terminology                                                            | not started |          |
| 9    | Ablation studies (concatenation, data subsets). **M2: ≥15% relative DS-WER over Gladia baseline**             | not started |          |
| 10   | CTranslate2 conversion; benchmark inference (target RTF < 0.5)                                                | not started |          |
| 11   | Plug `WhisperTranscriber` into production pipeline; end-to-end tests on new meetings                          | not started |          |
| 12   | Documentation, reproducibility scripts, final evaluation report. **Final milestone**: merged into OpenCouncil | not started |          |

Update Status / Evidence as work lands. The Planned focus column mirrors the proposal — change it there first, then here.
