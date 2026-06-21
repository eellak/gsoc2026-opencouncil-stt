# Dynamic Vocabulary and Entities

This note tracks the non-ASR side of the project: improving LLM post-correction through vocabulary and entity awareness.

Related notes:

- [Error taxonomy and routing](error-taxonomy.md)
- [Audio segmentation](audio-segmentation.md)
- [Dataset exploration sync](../meetings/2026-05-12-dataset-exploration-sync.md)
- [GSoC proposal](gsoc-proposal.md)

## Entity Classes

| Class | Scope | Likely source | Correction stage |
| --- | --- | --- | --- |
| Administrative acronyms | Global or municipality-specific | curated list, correction pairs | rule/LLM |
| Legal/administrative terms | Mostly global | correction frequency, domain docs | ASR + LLM |
| Council member names | Municipality/time-specific | OpenCouncil DB | LLM dynamic vocabulary |
| Local place names | Municipality-specific | DB, agenda parsing, transcripts | LLM dynamic vocabulary |
| Agenda/session entities | Meeting-specific | agenda documents, NER | LLM dynamic vocabulary |

## Working Principle

Stable domain terms can help both Whisper fine-tuning and LLM correction. Volatile entities, especially names of current council members and local places, should usually stay in the dynamic LLM vocabulary rather than being baked into the ASR model.

## Current OpenCouncil Behavior

Confirmed 2026-06-20 via `deepwiki-research` against `schemalabz/opencouncil`:

The transcription request already sends dynamic hints to Gladia
(`src/lib/tasks/transcribe.ts`):

- `customVocabulary`: `city.name`, person names, and party names, split into words.
- `customPrompt`: Greek prompt containing the municipality name and meeting date.

The same DB-derived context (`cityName`, `administrativeBodyName`,
`partiesWithPeople`, `topicLabels`) is also forwarded to the **fixTranscript LLM**
via `getRequestOnTranscriptRequestBody()`.

**The gap (the feature to build):** there is **no dedicated glossary / acronym /
custom-vocabulary Prisma model**, and **no admin UI** to manage per-city terms.
Vocabulary is derived *implicitly* from `Person` / `Party` / `City` names only.
Acronyms (ΕΕΤΑ, ΚΕΔΕ, ΔΕΥΑΟ…), toponyms beyond agenda titles, organisations, and
legal terms are **not** captured anywhere — so the fix LLM never sees them.

**Scope for us (decided 2026-06-20):** the value we invest in is feeding a
curated global + per-city term store into the
[fix-task prompt](fix-task-prompt-v2.md#key-finding-for-our-work) and measuring
whether it catches more errors. The STT-side path (`customVocabulary`/Gladia) is
**out of scope** — Gladia is being replaced (our own finetuned Whisper later), so
optimising its custom vocabulary is not worth our time. The glossary store stays
provider-agnostic, so a future STT can read it too, but we don't wire or measure
that.

Implication for evaluation: domain WER should distinguish terms already available to Gladia through `customVocabulary` from terms missing from the current DB-derived vocabulary.

## Proposed Vocabulary Layers

1. Global civic vocabulary: terms common across Greek municipal councils.
2. Municipality vocabulary: local officials, neighborhoods, municipal bodies, recurring local projects.
3. Meeting vocabulary: agenda items, invited speakers, legal references, ΑΔΑ/protocol numbers.

## Next Extraction Tasks

- Build frequency lists from `before_text -> after_text` substitutions.
- Extract candidate all-caps Greek acronyms.
- Extract capitalized multi-token names from `after_text`.
- Compare candidate entities across meetings to identify global vs local terms.
- Feed high-confidence entity lists into LLM correction prompts and DS-WER term lists.
