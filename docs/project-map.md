# Project Map

This is the working structure for the OpenCouncil ASR dataset exploration project. Links are plain relative Markdown links so they work in both Obsidian and GitHub.

## Core References

- [Current state](../CURRENT.md): first file to read; summarizes what matters now.
- [Agent instructions](../CLAUDE.md): guidance for coding/LLM agents working in this vault (`AGENTS.md` is a symlink to the same file).
- [Correction data quality](../data/reports/data_quality.md): what was cleaned, rejected, and why.
- [Roadmap](roadmap.md): phased plan from vault organization to UI, labels, dataset selection, and evaluation.
- [Progress](progress.md): current status against the GSoC plan (planned focus lives in the [proposal](reference/gsoc-proposal.md)).
- [Decisions index](decisions/_index.md): accepted decisions and open questions, split by theme (`data`, `storage`, `ui`, `audio`, `matching`).

## Meeting Notes

- [Meetings index](meetings/_index.md)
- [2026-05-08 - First sync](meetings/2026-05-08-first-sync.md)
- [2026-05-12 - Dataset exploration sync](meetings/2026-05-12-dataset-exploration-sync.md)
- [2026-05-19 - Notes](meetings/2026-05-19.md): midterm target, v2 CSV, nearby utterances API, seeded review order.

## Specs

- [Exploration UI spec](specs/exploration-ui.md): prototype UI behavior, local labels, and stats.
- [Local data model](specs/local-data-model.md): local records/tables for CSV corrections, cached JSON, matches, labels, and history.
- [UI prototype](../ui/README.md): implemented SvelteKit review app, CSV ingest, Supabase-backed review state, and export behavior.

## Runbooks

- [Claude parallel LLM categorization](runbooks/claude-parallel-llm-categorization.md): token-efficient subagent loop for categorizing remaining edit groups.
- [Self-host file-backed UI on Fly.io](runbooks/self-host-fly.md): deploy the local-only file-backed review UI on one persistent Fly machine.

## Reference Notes

- [GSoC proposal](reference/gsoc-proposal.md): original scope, timeline, risks, and appendix analysis.
- [Error taxonomy and routing](reference/error-taxonomy.md): how correction pairs should be split between ASR fine-tuning, LLM post-correction, rules, and review.
- [UI error categories](reference/ui-error-categories.md): proposed dropdown values for the exploration UI (Greek labels + English IDs + examples).
- [Audio segmentation](reference/audio-segmentation.md): how `audio_url`, timestamps, and corrected text can become training examples.
- [Latest-per-utterance report](../data/reports/latest-per-utterance.md): why the live DB keeps one correction per utterance.
- [Dynamic vocabulary and entities](reference/dynamic-vocabulary-and-entities.md): how municipal names, people, places, acronyms, and legal terms should feed the post-correction stage.
- [Mentor meeting questions](mentor-meeting-questions.md): concrete questions to resolve before implementation.
- [OpenCouncil meeting JSON](reference/opencouncil-meeting-json.md): shape of the large meeting JSON endpoint and how it can be used to match CSV corrections to utterances.

## History Logs

See [logs/_index.md](logs/_index.md) for cadence rules (weekly digest, skip empty weeks).

- [2026-05-12 - Vault setup](logs/2026-05-12-vault-setup.md): consolidated record of vault organization, agent instructions, meeting-notes skill, Mermaid diagrams, and PRD todo notation.
- [2026-05-18 - Weekly digest](logs/2026-05-18-weekly.md): full CSV ingest, audio CORS workaround, decisions split, weekly cadence adopted.

Older daily-normalization entries (May 13–18) are archived under [`archive/logs/2026-W20-daily-normalizations/`](../archive/logs/2026-W20-daily-normalizations/).

## Data Folders

- [`../data/clean`](../data/clean): analysis-ready CSV outputs.
- [`../data/reports`](../data/reports): data-quality reports and rejected rows.
- [`../scripts`](../scripts): reproducible preprocessing scripts.
- [`../archive`](../archive): raw, superseded, or non-current material.

## Immediate Workflow

1. Keep [Current state](../CURRENT.md) updated.
2. Use [Roadmap](roadmap.md), [Progress](progress.md), and the [Decisions index](decisions/_index.md) to avoid losing context.
3. Use the implemented [UI prototype](../ui/README.md) to review v2 CSV correction pairs.
4. Extend the [local data model](specs/local-data-model.md) with cached meeting JSON and matched utterances.
5. Match the v2 CSV rows to utterances from the large meeting JSON.
6. Then run dataset profiling and taxonomy validation with the UI-backed labels.
