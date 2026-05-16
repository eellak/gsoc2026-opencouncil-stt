# 2026-05-12 - Dataset Exploration Sync

## Metadata

- Date: 2026-05-12
- Source: user meeting summary and follow-up clarifications
- Status: normalized

## Summary

The meeting clarified that the next phase should focus on dataset exploration, not training. The useful next step is to combine the May 12 corrections CSV with OpenCouncil meeting JSON so a local exploration UI can review corrections with audio, context, labels, and include/exclude decisions.

## Decisions

- [x] Do not start model training yet.
- [x] Build intuition about data quality, correction types, timestamps, and training usefulness first.
- [x] Use the large meeting JSON endpoint for the first prototype instead of requiring a new API endpoint.
- [x] Do not block on finding the exact external query that generated `utterance-edits-may12-26.csv`.
- [x] Do not treat range requests or `TaskStatus.version` as first-milestone requirements.
- [x] Build a lightweight exploration UI before dataset selection.

## Action Items

- [ ] Define correction-to-utterance matching rules.
- [ ] Identify example meeting JSON URLs for rows in `utterance-edits-may12-26.csv`.
- [ ] Decide local storage shape for matched corrections and review labels.
- [ ] Build the first exploration UI prototype.
- [ ] Add stats for all corrections and included corrections.
- [ ] Later, run LLM pre-classification and taxonomy validation.

## Open Questions

- [?] Is `meeting_name` + `meeting_date` + `audio_url` enough to identify a meeting uniquely?
- [?] How often do CSV timestamps differ from meeting JSON utterance timestamps?
- [?] Should local review state use SQLite plus JSONL history, or a simpler sidecar format?
- [?] What exact error taxonomy should appear in the UI select field?

## Project Updates Needed

- [x] Current direction captured in `../../CURRENT.md`.
- [x] PRD phases captured in `../roadmap.md`.
- [x] UI behavior captured in `../specs/exploration-ui.md`.
- [x] Meeting JSON shape captured in `../reference/opencouncil-meeting-json.md`.
- [x] Superseded meeting-next-steps note archived at `../../archive/superseded-docs/meeting-next-steps-may12.md`.

## Raw Notes

Raw discussion context is preserved in the conversation and the archived superseded note:

- `../../archive/superseded-docs/meeting-next-steps-may12.md`
