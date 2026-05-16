# OpenCouncil Greek ASR Fine-Tuning Notes

This vault organizes the OpenCouncil Greek ASR dataset exploration and fine-tuning work.

Start here:

- [Current state](CURRENT.md)
- [Agent instructions](CLAUDE.md) (`AGENTS.md` is a symlink to the same file)
- [Project map](docs/project-map.md)
- [Roadmap](docs/roadmap.md)
- [Decisions](docs/decisions.md)
- [Meetings index](docs/meetings/_index.md)
- [Exploration UI spec](docs/specs/exploration-ui.md)
- [Local data model](docs/specs/local-data-model.md)
- [OpenCouncil meeting JSON notes](docs/reference/opencouncil-meeting-json.md)
- [UI prototype](ui/README.md)

## Vault Rules

- `CURRENT.md` is the first file to read and should stay short.
- `docs/decisions.md` records accepted decisions and open questions.
- `docs/roadmap.md` tracks phases and current direction.
- `docs/meetings/` stores normalized meeting notes.
- `docs/specs/` stores product and implementation specs.
- `docs/logs/YYYY-MM-DD-*.md` keeps dated vault/implementation history.
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
