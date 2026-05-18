# OpenCouncil Greek ASR Fine-Tuning Notes

This repository is the working vault for a [GSoC 2026 project](docs/reference/gsoc-proposal.md) to fine-tune Whisper for Greek municipal council transcription, with LLM post-correction. It holds the dataset exploration notes, decisions, specs, and the local review-UI prototype that feeds into the training work.

The repo is currently in **dataset exploration mode**, not training mode.

Start here:

- [Current state](CURRENT.md)
- [Progress vs GSoC plan](docs/progress.md)
- [Agent instructions](CLAUDE.md) (`AGENTS.md` is a symlink to the same file)
- [Project map](docs/project-map.md)
- [Roadmap](docs/roadmap.md)
- [Decisions index](docs/decisions/_index.md)
- [Meetings index](docs/meetings/_index.md)
- [Logs index](docs/logs/_index.md)
- [Exploration UI spec](docs/specs/exploration-ui.md)
- [Local data model](docs/specs/local-data-model.md)
- [OpenCouncil meeting JSON notes](docs/reference/opencouncil-meeting-json.md)
- [GSoC proposal](docs/reference/gsoc-proposal.md)
- [UI prototype](ui/README.md)

## Vault Rules

- `CURRENT.md` is the first file to read and should stay short.
- `docs/decisions/` records accepted decisions and open questions, split by theme. See [decisions index](docs/decisions/_index.md).
- `docs/progress.md` shows where we are against the GSoC plan. The plan itself lives in the [proposal](docs/reference/gsoc-proposal.md).
- `docs/roadmap.md` tracks phases and current direction.
- `docs/meetings/` stores normalized meeting notes.
- `docs/specs/` stores product and implementation specs.
- `docs/logs/` keeps weekly digests. See [logs index](docs/logs/_index.md) for cadence rules — no daily logs.
- `docs/reference/` stores stable technical reference notes.
- `archive/` stores raw, superseded, or non-current material.
- Data outputs live under `data/`; scripts live under `scripts/`.
- `CLAUDE.md` is the single source of truth for assistant instructions; `AGENTS.md` is a symlink to it for tools that read that filename.

## Current Dataset Outputs

- Full May 12 export: [`utterance-edits-may12-26.csv`](utterance-edits-may12-26.csv)
- Historical sample (archived): [`archive/old-data/corrections-sample.csv`](archive/old-data/corrections-sample.csv)
- Clean CSV: [`data/clean/corrections_clean.csv`](data/clean/corrections_clean.csv)
- Rejected rows: [`data/reports/corrections_rejected.csv`](data/reports/corrections_rejected.csv)
- Summary JSON: [`data/reports/corrections_summary.json`](data/reports/corrections_summary.json)

Regenerate the cleaned data:

```bash
rtk python3 scripts/preprocess_corrections.py
```

## Local UI Prototype

The SvelteKit correction-review prototype lives in [`ui/`](ui/). It ingests the May 12 CSV into local SQLite and supports review labels, timestamp adjustments, stats, and included-row export. It does not yet join rows to cached meeting JSON for utterance IDs, speaker context, or surrounding transcript context.
