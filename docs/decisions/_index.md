# Decisions

This folder tracks accepted decisions and open questions, split by theme. Keep each file short and current. Move long reasoning into dated logs or reference notes.

## Files

- [data.md](data.md) — CSV ingest, content categorisation, stable IDs, task version
- [storage.md](storage.md) — local SQLite/JSONL state, Turso + Vercel hosting
- [ui.md](ui.md) — exploration-before-training stance, waveform/peaks-cache
- [audio.md](audio.md) — CORS workaround, Vercel proxy, fixed-file URL map
- [matching.md](matching.md) — meeting JSON usage, `utterance.text` semantics, open matching/taxonomy questions

## Index

### Accepted

- [Exploration before training](ui.md#2026-05-12---exploration-before-training) — 2026-05-12
- [Use the large meeting JSON for the first prototype](matching.md#2026-05-12---use-the-large-meeting-json-for-the-first-prototype) — 2026-05-12
- [Do not block on the CSV export query](data.md#2026-05-12---do-not-block-on-the-csv-export-query) — 2026-05-12
- [Range requests are not a first milestone](audio.md#2026-05-12---range-requests-are-not-a-first-milestone) — 2026-05-12
- [Task version is not required for first exploration](data.md#2026-05-12---task-version-is-not-required-for-first-exploration) — 2026-05-12
- [`utterance.text` is the corrected text, not the original](matching.md#2026-05-12---utterancetext-is-the-corrected-text-not-the-original) — 2026-05-12
- [Waiting on a new corrections export with stable IDs](data.md#2026-05-12---waiting-on-a-new-corrections-export-with-stable-ids) — 2026-05-12
- [Local storage: SQLite + JSONL event log](storage.md#2026-05-12---local-storage-sqlite--jsonl-event-log) — 2026-05-12
- [Turso + Vercel for hosted review state](storage.md#2026-05-15---turso--vercel-for-hosted-review-state) — 2026-05-15
- [Waveform bars + peaks-cache prefetch](ui.md#2026-05-15---waveform-bars--peaks-cache-prefetch) — 2026-05-15
- [Categorise all CSV rows instead of dropping](data.md#2026-05-16---categorise-all-csv-rows-instead-of-dropping) — 2026-05-16
- [Audio workaround via Vercel proxy and fixed-file map](audio.md#2026-05-16---audio-workaround-via-vercel-proxy-and-fixed-file-map-pending-proper-fix) — 2026-05-16

### Open

- [Correction-to-utterance matching strategy](matching.md#correction-to-utterance-matching-strategy)
- [Error taxonomy](matching.md#error-taxonomy)
- [Include/exclude semantics](matching.md#includeexclude-semantics)
- [Training/evaluation pair](matching.md#trainingevaluation-pair)
- [Diagram maintenance](#diagram-maintenance) (meta — see below)

## Meta

### Diagram maintenance

The vault contains Mermaid diagrams in:

- `CURRENT.md`
- `docs/roadmap.md`
- `docs/reference/opencouncil-meeting-json.md`
- `docs/specs/exploration-ui.md`

When the project flow, roadmap phases, matching strategy, or review state model changes, update the relevant diagram in the same edit as the text.
