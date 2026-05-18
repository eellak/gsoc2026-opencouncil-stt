# Agent Instructions

This vault is for the OpenCouncil Greek ASR fine-tuning and dataset exploration project.

## Role

Be a practical project assistant. Help keep the work organized, current, and actionable.

The project is currently in dataset exploration mode, not training mode. Prioritize:

- clarifying the current state;
- organizing notes;
- preserving decisions and open questions;
- turning vague meeting notes into concrete next steps;
- keeping the vault readable by humans and LLMs;
- supporting the planned correction exploration UI.

## Start Here

Always read these first:

1. `CURRENT.md`
2. `docs/decisions/_index.md`
3. `docs/progress.md`
4. `docs/roadmap.md`
5. `docs/project-map.md`

Then follow only the links needed for the task.

## Meeting Notes Skill

When processing raw meeting notes, transcripts, mentor sync notes, or project discussion notes, use the `opencouncil-meeting-notes` skill if available.

That skill defines how to:

- create normalized meeting notes under `docs/meetings/`;
- update `CURRENT.md`, the relevant file under `docs/decisions/`, and `docs/roadmap.md`;
- update specs and reference notes without duplication;
- keep Mermaid diagrams and PRD todos synchronized;
- archive raw or superseded notes.

## Current Project Direction

The immediate goal is to combine:

- `utterance-edits-may12-26.csv`
- the large OpenCouncil meeting JSON endpoint

so the project can build a local exploration UI that shows:

- `before_text` / `after_text` diff;
- audio playback for the utterance span;
- meeting, city, speaker, and surrounding utterance context;
- error category;
- include/exclude/uncertain decision;
- aggregate stats over all corrections and selected corrections.

Do not treat training, benchmarking, or model selection as the next step unless the user explicitly asks.

## Vault Organization Rules

- `CURRENT.md` is the short entry point. Update it when the actual current direction changes.
- `docs/decisions/` holds accepted decisions and open questions, split by theme (`data.md`, `storage.md`, `ui.md`, `audio.md`, `matching.md`) with an `_index.md`. Keep each file concise.
- `docs/progress.md` is the current status snapshot against the GSoC plan. The plan itself lives in `docs/reference/gsoc-proposal.md` — do not duplicate it.
- `docs/roadmap.md` is for phases and next milestones.
- `docs/meetings/` is for normalized meeting notes.
- `docs/specs/` is for product and implementation specs.
- `docs/logs/` is for weekly digests and ad-hoc incident logs only. See `docs/logs/_index.md` for cadence rules.
- `docs/reference/` is for stable technical references and schemas.
- `archive/` is for raw, superseded, or non-current material.
- `data/` is for generated data outputs and reports.
- `scripts/` is for reproducible processing scripts.

Do not scatter important state across random notes. If something changes the plan, update `CURRENT.md` or the relevant file under `docs/decisions/`.

## History Tracking

Default cadence is a **weekly digest** under `docs/logs/YYYY-MM-DD-weekly.md`, written on Fridays. Skip the week entirely if none of the following happened:

- a canonical doc was updated (`CURRENT.md`, `docs/decisions/**`, `docs/roadmap.md`, `docs/specs/**`);
- a decision was added, changed, or superseded;
- an implementation milestone landed;
- a real ambiguity surfaced that needs a future decision.

Routine "I reviewed the diff and nothing structural changed" days do not deserve a log entry. Use an ad-hoc dated log (`YYYY-MM-DD-<slug>.md`) only for a significant incident or one-off decision that does not fit a weekly digest.

When something does warrant a log:

1. Update the relevant canonical note (decisions, spec, roadmap, current).
2. Add it to the week's digest.
3. Link the digest from `docs/logs/_index.md` and, if structurally important, from `docs/project-map.md`.

Also update the relevant row in `docs/progress.md` when an implementation milestone changes status. Prefer short decision entries over long narrative.

## Mermaid Diagrams

The vault uses Mermaid diagrams as compact project memory. Keep them synchronized with the surrounding text.

Canonical diagrams:

- `CURRENT.md`: current project flow.
- `docs/roadmap.md`: timeline/phases.
- `docs/reference/opencouncil-meeting-json.md`: CSV-to-meeting-JSON join path.
- `docs/specs/exploration-ui.md`: review state/update loop.

Update the relevant diagram when:

- the main project flow changes;
- roadmap phases are added, removed, renamed, or reordered;
- the correction-to-utterance matching strategy changes;
- local storage or review states change;
- the UI workflow changes.

Do not add decorative diagrams. Add or edit diagrams only when they clarify state, flow, dependencies, or decisions.

## PRD-Style Todos

The vault uses markdown checkboxes as lightweight PRD/task notation.

Use:

- `[ ]` not started
- `[x]` done
- `[~]` in progress or partially done
- `[?]` blocked or needs decision

When editing roadmap, specs, or next-step docs:

- preserve markdown todo notation;
- update task status when work is completed or blocked;
- keep acceptance criteria near the relevant task list;
- do not convert checklists into prose;
- do not mark tasks done unless the work is actually complete;
- add new tasks as checkboxes when they represent concrete work.

## Working Style

- Be concrete and concise.
- If the user is confused, explain the practical difference and why it matters.
- Do not introduce blockers unless they are real blockers.
- Separate "needed now" from "useful later".
- Prefer local prototype steps over production architecture unless the user asks otherwise.
- Preserve uncertainty explicitly instead of pretending it is resolved.

## Current Clarifications

- We do not need the exact external query that produced the CSV before moving forward.
- We do not need a new API endpoint for the first prototype.
- We can use the large meeting JSON endpoint and cache it locally.
- Range requests are not a first milestone.
- `TaskStatus.version` is not needed for the first exploration UI.
- The first UI should work with CSV edit pairs as `before_text -> after_text`.

## Shell Commands

This workspace follows the local RTK instruction:

```bash
rtk <command>
```

Use `rtk` before shell commands when working in this vault.

## Editing Rules

- Keep docs in plain Markdown.
- Prefer relative links inside Markdown notes.
- Keep canonical notes short; put details in reference notes or dated logs.
- Do not rewrite unrelated notes just for style.
- Do not delete uncertainty or context unless it has been superseded and recorded.
