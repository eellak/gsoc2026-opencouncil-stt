# OpenCouncil Greek ASR Fine-Tuning Notes

Working vault for a [GSoC 2026 project](docs/reference/gsoc-proposal.md) to fine-tune Whisper on Greek municipal council speech, with LLM post-correction. Holds dataset exploration notes, decisions, specs, and the local review-UI prototype.

No training yet — dataset exploration comes first.

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
- `docs/decisions/` — accepted decisions and open questions, split by theme. See [decisions index](docs/decisions/_index.md).
- `docs/progress.md` — where we are against the GSoC plan. The plan itself is in the [proposal](docs/reference/gsoc-proposal.md).
- `docs/roadmap.md` — phases and current direction.
- `docs/meetings/` — normalized meeting notes.
- `docs/specs/` — product and implementation specs.
- `docs/logs/` — weekly digests only. See [logs index](docs/logs/_index.md) for cadence rules.
- `docs/reference/` — stable technical references.
- `archive/` — superseded material, local only (gitignored).
- Data outputs live under `data/`; scripts live under `scripts/`.
- `CLAUDE.md` is the single source of truth for assistant instructions; `AGENTS.md` is a symlink to it for tools that read that filename.

## Current Dataset Outputs

- Full May 12 export: [`utterance-edits-may12-26.csv`](utterance-edits-may12-26.csv)
- Historical sample (archived): [`archive/old-data/corrections-sample.csv`](archive/old-data/corrections-sample.csv)
- Clean CSV: [`data/clean/corrections_clean.csv`](data/clean/corrections_clean.csv)
- Rejected rows: [`data/reports/corrections_rejected.csv`](data/reports/corrections_rejected.csv)

(`data/` is gitignored — these are regenerated locally by the preprocessing script below.)

Regenerate the cleaned data:

```bash
rtk python3 scripts/preprocess_corrections.py
```

## Local UI Prototype

The SvelteKit correction-review prototype lives in [`ui/`](ui/). It ingests the May 12 CSV into local SQLite and supports review labels, timestamp adjustments, stats, and included-row export. Meeting JSON matching (utterance IDs, speaker context, surrounding transcript) is the next gap to close.
