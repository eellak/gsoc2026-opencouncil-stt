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

## Empirical findings (eval harness, 2026-06-21)

Measured on our text-only fix-task eval (1000-row stratified A/B + 100-row
segment proxy, sonnet; verbatim task-v2 prompt). See
[fix-task eval harness](../specs/fix-task-eval-harness.md).

A naively injected global glossary block is **net-negative**:

| effect | result |
| --- | --- |
| named_entity (names/orgs/places) | **+9.9pp** correction rate (A/B); **+11pp** in segments (36.8%→47.9%) |
| acronym_abbreviation | −6.5pp |
| number_date | −6.0pp |
| accent/morph | −5.9pp |
| **HIR overall** | **+1.9pp (worse)** — 60.0% → 61.9% |
| WER overall | 0.153 → 0.151 (flat) |

On the harder dev split (42 residual rows, sonnet) `v1_glossary` scored **identical**
to baseline (HIR 0.8095, overcorrection 0.03 both) — no gain.

**Why:** dumping terms into the prompt invites retrieval-noise overcorrection —
the model bends an unrelated but phonetically-near word toward a glossary entry.
The named_entity win is real; it is just swamped by harm elsewhere.

### Is it needed? — current verdict

**Conditionally yes, scoped — not as a global dump.** A glossary is worth shipping
only if it is shaped to keep the named_entity win without the collateral damage:

- **Precise per-utterance retrieval, not dump-all.** Only surface terms that fuzzy-match
  a token in the current utterance (rapidfuzz; length ≥ 4 guard; high cutoff
  ~88/90 word/phrase). A whole-list block is what caused the regression.
- **Scope the instruction to name/entity normalisation only.** The block must say
  "use ONLY to fix a word that is clearly the same name misspelled — never force an
  unrelated word to match the list" (the seed `v1_glossary`/`v4_combo` wording).
- **Pair with an anti-overcorrection guardrail** in the prompt (change as few words
  as possible) to offset the retrieval-noise harm.
- **Type the entries** (acronym / toponym / org / legal / person / place) and carry
  ASR-misspelling aliases — retrieval against aliases is more precise than against
  surface forms alone.

Open question (pending): whether the *refined* glossary block (precise retrieval +
guardrail) beats baseline HIR on the production model. Being measured now via the
[improvement loop](../specs/fix-task-improvement-loop.md) (codex gpt-5.5-low
selection → sonnet validation). Until that confirms a net HIR drop, treat the
glossary as an **entity-only** assist, not a global correctness lever.

## NER-gated deterministic name correction (2026-06-21)

A separate experiment line: can we fix misspelled names **deterministically**
(without spending LLM quota) by fuzzy-matching ASR tokens against a name list —
but **only on tokens that are actually names**, so we don't bend unrelated
phonetically-near words? The gate is a Greek NER model:
`amichailidis/bert-base-greek-uncased-v1-finetuned-ner` (GreekBERT) or GLiNER
(`urchade/gliner_multi-v2.1`). Code: `eval/{fuzzy_correct,ner_gate_eval,roster_gate_eval,fetch_rosters}.py`.

The point of the NER gate is exactly the thing plain fuzzy matching gets wrong:
plain fuzzy "corrects" any token close to a list entry, so it rewrites correct,
unrelated words. Gating on entity spans should let it touch only names.

**Findings:**
- **The NER gate alone is not enough.** Against the dense global glossary
  (~5,894 names), deterministic fuzzy — *even NER-gated* — still corrupts clean
  text. Too many real words sit phonetically near *some* name in a 5,894-entry list.
- **The candidate-set size is the real lever.** Swapping the global glossary for a
  **small per-meeting roster** (the names actually spoken in that meeting, ~tens)
  raises precision and clean-text retention monotonically.
- **With the real per-meeting roster** (fetched from
  `/api/cities/{city}/meetings/{meeting}` `partiesWithPeople`): precision **0.42**,
  collateral damage **1.2 / 100** utterances, clean-text retention **0.955** (vs
  **0.855** with the global glossary), recall **~0.12**.
- **Verdict: not deployable standalone** (~95.5% retention still rewrites ~1 in 20
  clean utterances, and ~0.12 recall misses most). Its best role is a
  **conservative first pass alongside the LLM**, not a replacement — and it
  confirms the same lesson as the glossary A/B: **scope tightly (small per-meeting
  candidate set), never dump a big list.**

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
