# Progress

The historical proposal timeline lives in the [GSoC proposal](reference/gsoc-proposal.md#timeline-12-weeks--350-hours). This file tracks the plan we are actually working from now.

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

## GSoC weeks 1–12 (working plan)

| Week | Working focus                                                                                                      | Status      | Evidence |
| ---- | ------------------------------------------------------------------------------------------------------------------ | ----------- | -------- |
| 1    | Data joining: cache meeting JSON, match CSV corrections to utterances, report matched/ambiguous/unmatched rows     | not started |          |
| 2    | Taxonomy and review UI: validate error taxonomy, review sample rows with audio/context, refine include/exclude      | not started |          |
| 3    | Curated dataset v1: export ASR fine-tuning candidates, define split policy, check city/meeting distribution         | not started |          |
| 4    | Evaluation harness: Greek-aware WER/CER, Domain WER, baseline reconstruction from `before_text`                     | not started |          |
| 5    | Base-model benchmark and **M1: Curated ASR Dataset and Base Model**                                                 | not started |          |
| 6    | First LoRA fine-tuning run on the selected base model                                                              | not started |          |
| 7    | Hypertuning sweep: rank, alpha, learning rate, dataset subset, segment length/concatenation                         | not started |          |
| 8    | Model selection: compare checkpoints on fixed metrics, domain terms, hallucinations, and runtime                    | not started |          |
| 9    | Final fine-tuned checkpoint selection and reproducibility report                                                    | not started |          |
| 10   | Optimization and deployment: faster-whisper/CTranslate2 packaging if feasible, inference benchmarks                 | not started |          |
| 11   | OpenCouncil tasks integration: transcriber adapter, configuration, end-to-end held-out meeting test                 | not started |          |
| 12   | Final docs, deployment notes, evaluation report, **Final: Deployed Fine-Tuned ASR Transcriber for OpenCouncil**     | not started |          |

Update Status / Evidence as work lands. The proposal stays as history; this table is the working plan.
