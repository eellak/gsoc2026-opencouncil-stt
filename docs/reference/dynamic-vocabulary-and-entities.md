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

The transcription request already sends dynamic hints to Gladia:

- `customVocabulary`: `city.name`, person names, and party names, split into words.
- `customPrompt`: Greek prompt containing the municipality name and meeting date.

There is no separate glossary file today. The vocabulary source is the live OpenCouncil DB.

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
