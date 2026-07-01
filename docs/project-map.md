# Project Map

Where things live. This is a navigational map by domain, not an exhaustive file
listing — for the current state read [CURRENT.md](../CURRENT.md) first. Links are
plain relative Markdown so they work in both Obsidian and GitHub.

## Start Here

- [Current state](../CURRENT.md) — first file to read; where the project is now.
- [Progress vs GSoC plan](progress.md) — status against the plan (the plan itself is the [proposal](reference/gsoc-proposal.md)).
- [Roadmap](roadmap.md) — phases and current direction.
- [Decisions index](decisions/_index.md) — accepted decisions and open questions, by theme.
- [Agent instructions](../CLAUDE.md) — guidance for agents (`AGENTS.md` symlinks to it).

## Where Each Kind of Thing Lives

| Area | Holds | Notes |
| ---- | ----- | ----- |
| `CURRENT.md` | Operational status — where we are today | Keep short and current-facing |
| `docs/decisions/` | ADR-like accepted decisions + open questions | One file per theme |
| `docs/specs/` | Product and implementation specs | The largest doc area |
| `docs/reference/` | Stable technical references and schemas | |
| `docs/runbooks/` | Repeatable operational procedures | |
| `docs/reports/` | Committed report snapshots (curated) | Regenerated artifacts stay in `data/` |
| `docs/meetings/` | Normalized meeting notes | |
| `docs/logs/` | Weekly digests (skip empty weeks) | See [cadence rules](logs/_index.md) |
| `docs/issues/` | Issue working notes | **Local only — gitignored** |
| `eval/` | Fix-task eval + autoresearch code (Python) | |
| `ui/` | SvelteKit review app | [ui/README.md](../ui/README.md) |
| `scripts/` | Reproducible preprocessing/ops scripts | |
| `notebooks/` | Kaggle fine-tune / sweep notebooks | |
| `data/` | Generated data outputs and reports | **Local only — gitignored** |
| `archive/` | Superseded / raw material | **Local only — gitignored** |

## Decisions (by theme)

[Index](decisions/_index.md) · [data](decisions/data.md) · [storage](decisions/storage.md) ·
[ui](decisions/ui.md) · [audio](decisions/audio.md) · [matching](decisions/matching.md) ·
[metric-hir](decisions/metric-hir.md)

## Specs

Fine-tuning / dataset:

- [Dataset split and publish plan](specs/dataset-split-and-publish-plan.md)
- [Whisper hyperparameter sweep](specs/whisper-hyperparam-sweep.md)
- [Fine-tuning dry-run plan](specs/finetuning-dryrun-plan.md)
- [Meeting trust cutoff plan](specs/meeting-trust-cutoff-plan.md)
- [Mini-PC fine-tune autoresearch](specs/minipc-finetune-autoresearch.md)
- [Error division: fine-tune vs LLM](specs/error-division.md) · [finetune-vs-llm error division](specs/finetune-vs-llm-error-division.md)

Fix-task / eval:

- [Fix-task eval harness](specs/fix-task-eval-harness.md)
- [Fix-task improvement loop](specs/fix-task-improvement-loop.md)

Review UI:

- [Exploration UI](specs/exploration-ui.md)
- [Local data model](specs/local-data-model.md)
- [Context in index](specs/context-in-index.md)

## Reference

- [GSoC proposal](reference/gsoc-proposal.md) — scope, timeline, risks, appendix analysis.
- [Fine-tuning 101](reference/finetuning-101.md) — background and plan.
- [OpenCouncil meeting JSON](reference/opencouncil-meeting-json.md) — the large meeting JSON shape and CSV join path.
- [Error taxonomy and routing](reference/error-taxonomy.md) — where each error is best fixed.
- [Dynamic vocabulary and entities](reference/dynamic-vocabulary-and-entities.md) — names/places/acronyms feeding post-correction.
- [Fix-task prompt v2 (verbatim)](reference/fix-task-prompt-v2.md) — the exact current fixTranscript prompt.
- [Audio segmentation](reference/audio-segmentation.md) · [Training-unit granularity](reference/training-unit-granularity.md) · [UI error categories](reference/ui-error-categories.md)
- [Disaster recovery](reference/disaster-recovery.md) · [Oracle VM](reference/oracle-vm.md)

## Reports (committed snapshots)

- [Month #1 — June 2026](reports/month-1-2026-06.md) — narrative progress journal (part 1 of 3).
- [Diarization-conditioned ASR review](reports/diarization-conditioned-asr-review.md)
- [Fix-task experiment report (HTML)](reports/fix-task-experiment-report.html) — self-contained public report of the fix-task chain.
- [Research findings (simple)](reports/research-findings-simple.md)

See [docs/reports/README.md](reports/README.md) for what belongs here vs. in `data/`.

## Runbooks

- [Eval harness on-box brief](runbooks/eval-harness-onbox-brief.md)
- [Next-batch selection on-box brief](runbooks/next-batch-selection-onbox-brief.md)
- [VPS inventory](runbooks/vps-inventory.md)

## Meetings & Logs

- [Meetings index](meetings/_index.md) — normalized sync notes (2026-05-08 → 2026-06-23).
- [Logs index](logs/_index.md) — weekly digests; cadence rules there.

## Immediate Workflow

1. Keep [CURRENT.md](../CURRENT.md) current-facing.
2. Use [Roadmap](roadmap.md), [Progress](progress.md), and the [Decisions index](decisions/_index.md) to avoid losing context.
3. Fine-tuning work: [dataset split plan](specs/dataset-split-and-publish-plan.md) + [hyperparameter sweep](specs/whisper-hyperparam-sweep.md).
4. Review throughput continues via the [UI](../ui/README.md) toward the dataset target.
