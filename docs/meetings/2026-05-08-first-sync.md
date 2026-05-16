# 2026-05-08 - First Sync

## Metadata

- Date: 2026-05-08
- Source: archived raw notes at `../../archive/old-notes/first-meeting-notes-0805.raw.md`
- Status: normalized

## Summary

The first sync framed the project around correction pairs, evaluation metrics, and early GSoC planning. The notes introduced HIR, WER, CER, and DS-WER, and sketched how correction data could support ASR fine-tuning, evaluation, and post-correction analysis.

## Decisions

- [x] Treat correction pairs as multi-purpose data: training candidates, evaluation references, and error-analysis material.
- [x] Use HIR as an operational metric alongside WER and domain-specific metrics.
- [x] Separate ASR-fixable errors from post-processing and rule-based cleanup candidates.

## Action Items

- [ ] Keep metric definitions in the roadmap/evaluation notes when evaluation work begins.
- [ ] Preserve mentor questions that are still relevant in `../mentor-meeting-questions.md`.
- [ ] Use the archived raw notes only as historical context, not as a source of current truth.

## Open Questions

- [?] Which metric matters most to mentors operationally: WER, DS-WER, HIR, or review-time reduction?
- [?] Which municipalities or meetings should be held out for generalization evaluation?
- [?] What data sharing/licensing constraints apply to correction/audio data?

## Project Updates Needed

- [x] Core evaluation ideas captured in `../roadmap.md`.
- [x] Taxonomy ideas captured in `../reference/error-taxonomy.md`.
- [x] Raw notes archived to reduce root-level clutter.

## Raw Notes

See `../../archive/old-notes/first-meeting-notes-0805.raw.md`.
