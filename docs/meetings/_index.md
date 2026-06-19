# Meetings Index

This folder contains normalized meeting notes. Raw notes or transcripts should either be summarized into this format or archived under `archive/old-notes/`.

| Date | Meeting | Status | Canonical outputs |
| --- | --- | --- | --- |
| 2026-05-08 | [First sync](2026-05-08-first-sync.md) | normalized from raw notes | metrics, initial backlog, mentor questions |
| 2026-05-12 | [Dataset exploration sync](2026-05-12-dataset-exploration-sync.md) | normalized from conversation notes | exploration-first plan, UI direction, data-access clarifications |
| 2026-05-19 | [Notes](2026-05-19.md) | normalized notes | midterm target, new CSV export, nearby utterances API, seeded review order |
| 2026-05-26 | [Coding period kickoff](2026-05-26.md) | normalized from raw notes | review-UI todos, no-timestamp-fix decision, midterm dataset target, review cadence |
| 2026-06-02 | [Finetuning attempt + split discussion](2026-06-02.md) | normalized from raw notes + Discord | end-to-end pipeline, split-by-meeting, baseline-first, audio-normalization + benchmark open questions |
| 2026-06-16 | [Benchmark walkthrough + split mechanics](2026-06-16.md) | normalized from sync transcript | provider results (Scribe best), temporal test set, seeded train/val split, smoke-pipeline next steps, prefetch-bug fix plan |

## Meeting Note Rules

- Keep one normalized note per meeting.
- Put raw transcript/rough notes under `## Raw Notes` only when they are short enough to be useful; otherwise archive them and link to the archive path.
- Decisions that remain true after the meeting belong in the relevant file under `../decisions/` (see [decisions index](../decisions/_index.md)).
- Action items that affect execution belong in `../roadmap.md` or `../../CURRENT.md`.
- Stable technical details belong in `../reference/`.
- Product or implementation requirements belong in `../specs/`.
