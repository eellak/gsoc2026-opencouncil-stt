# 2026-05-12 - Vault Setup

Consolidated log of the vault organization work done on 2026-05-12. Replaces six separate per-topic logs.

## Context

We clarified the immediate need is **not** training and **not** a new API endpoint — it is organizing the vault and defining how to combine the corrections CSV with existing OpenCouncil meeting JSON for an exploration UI.

Key clarifications captured:

- The exact query that produced the CSV is unavailable and should not block progress.
- The large meeting JSON endpoint already contains what is needed for the first prototype.
- Range requests and `TaskStatus.version` are not first-milestone concerns.
- The first UI focuses on correction review, audio playback, categorization, include/exclude decisions, and stats.

## Vault Normalization

Canonical homes established:

- current truth and next action: `CURRENT.md`
- decisions/open questions: `docs/decisions.md` (later split into `docs/decisions/` on 2026-05-18)
- phases/tasks: `docs/roadmap.md`
- meeting history: `docs/meetings/`
- product/implementation specs: `docs/specs/`
- stable technical references: `docs/reference/`
- maintenance history: `docs/logs/`
- raw or superseded material: `archive/`

Created `docs/meetings/`, `docs/specs/`, `archive/superseded-docs/`, `archive/old-notes/`, `archive/artifacts/audio-samples/`. Moved the raw first meeting note, superseded May 12 next-step note, exploration UI spec, local data model spec, and the error taxonomy / audio segmentation / dynamic vocabulary / GSoC reference notes into their canonical homes. Added normalized meeting notes for the 2026-05-08 first sync and the 2026-05-12 dataset exploration sync.

## Agent Instructions

Authored `CLAUDE.md` (single source of truth) with `AGENTS.md` as a symlink for tools that read the latter. Both instruct assistants to:

- start from `CURRENT.md`, `docs/decisions.md`, `docs/roadmap.md`, `docs/project-map.md`;
- keep the project in exploration mode unless asked otherwise;
- update current state, decisions, and dated logs when meaningful context changes;
- avoid artificial blockers; distinguish "needed now" from "useful later";
- support the correction exploration UI.

## Meeting Notes Skill

Created local Codex skill at `~/.codex/skills/opencouncil-meeting-notes/SKILL.md` so future raw notes are processed consistently: normalized notes under `docs/meetings/`, canonical homes updated, raw/superseded material archived. Agent files reference the skill.

## Mermaid Diagrams as Project Memory

Added canonical diagrams (kept synchronized with surrounding text on every edit):

- `CURRENT.md`: current project flow from CSV and meeting JSON to UI and stats.
- `docs/roadmap.md`: timeline from vault organization to evaluation/training.
- `docs/reference/opencouncil-meeting-json.md`: CSV-to-meeting-JSON join path.
- `docs/specs/exploration-ui.md`: review state/update loop.

Update the relevant diagram when project flow, roadmap phases, matching strategy, storage/review states, or UI workflow change.

## PRD-Style Todo Notation

Added markdown checkbox notation to `docs/roadmap.md` and `CURRENT.md`:

- `[ ]` not started, `[x]` done, `[~]` in progress, `[?]` blocked

Update checkbox state in the same edit as the supporting work. Do not mark tasks done unless they actually are.

## Next Action After Setup

Define and implement the local prototype data model: corrections table from CSV, cached meeting JSON, matched correction-to-utterance table, review labels, JSONL history log.
