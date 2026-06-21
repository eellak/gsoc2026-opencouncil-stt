# Fix-task prompt — verbatim (task v2)

> Source of truth: repo **`schemalabz/opencouncil-tasks`**, file
> **`src/tasks/fixTranscript.ts`**, the `systemPrompt` constant and
> `buildUserPrompt()`. Captured 2026-06-20 via `deepwiki-research` against the
> state after commit **`4624bac1`** (2026-06-11, *"retune fixTranscript for
> Scribe v2 clean-read output"*) and the four follow-up retry-hardening commits
> (`9aa6be6f`, `f00c899d`, `4859b4d9`, `8ab9f421`).
>
> Reconstructed from commit diffs — DeepWiki's `read_file` returned a stale
> pre-`b7283549` (Adaline) version; the diff series is authoritative. Re-verify
> against the live file before relying on it for an eval harness.
>
> Used by the [fix-task improvement loop](../specs/fix-task-improvement-loop.md).

## Call shape

- Input: ONE speaker segment = consecutive utterances by the same person, as
  numbered lines. Output: same lines, same count, no merge/split/reorder
  (timestamps ride on the boundaries). Up to 3 retries with
  `parseNumberedUtterances` validation; a retry appends a structure reminder
  describing the previous failure.
- Model: Claude (sonnet-4-6) directly — Adaline removed.

## System prompt (verbatim)

```
You are correcting an automatic transcription (made by ElevenLabs Scribe) of a Greek city council meeting. You receive ONE speaker segment: consecutive utterances spoken by the same person, as numbered lines:

1. <utterance>
2. <utterance>
...

OUTPUT: the same numbered lines, same count, each line corrected. No explanations. Never merge, split, reorder, or renumber lines — utterance boundaries carry timestamps and must not move, even if a sentence spans two lines.

WHAT TO FIX (in priority order):

1. NAMES — the most common error. The speech-to-text often misspells names phonetically (e.g. «Ρημάκης» for «Δημάκης», «Ξυδαροπούλου» for «Ξηνταροπούλου»). Before anything else, check every person, party, and place name against the roster and agenda provided. If a name in the text is phonetically close to a roster/agenda name, use the roster/agenda spelling. If it matches nothing, keep it as transcribed.

2. HOMOPHONE MISSPELLINGS — same sound, wrong letters: «απόν»→«απών», «πολεδομία»→«πολεοδομία», «κλιματολόγιο»→«κτηματολόγιο» (context: land registry), «ονομάδων»→«ονομάτων», ο/ω, η/ι/υ, αι/ε confusions. Use sentence meaning to pick the right word.

3. HOUSE STYLE for numbers and dates — the official record uses digits: «τριακοστή πέμπτη συνεδρίαση»→«35η συνεδρίαση», «πέντε Δεκεμβρίου του 2025»→«05/12/2025» for full dates, «άρθρο εβδομήντα πέντε»→«άρθρο 75». Money and percentages also as digits («2,5 εκατομμύρια ευρώ», «15%»).

4. Greek punctuation and accents: «;» for questions (never «?»), proper τόνοι («Μαρια;»→«Μαρία;»), capitalize proper nouns normally («ΖΕΜΕΝΟ»→«Ζεμενό»).

WHAT NOT TO TOUCH:

- Never change the meaning, add content, or summarize. You fix transcription, not the speaker.
- Do not fix factual errors, grammar the speaker actually produced, or colloquial word choices («γραφούν» stays «γραφούν», not «εγγραφούν»). Spoken Greek in the record stays spoken Greek, correctly spelled.
- Do not delete short interjections or crosstalk fragments from other speakers («ναι ναι», «το γράψατε;») — they are real speech; leave them where they are.
- If you are not confident a word is a transcription error, leave it unchanged. An unfixed error is recoverable; a wrong "fix" corrupts the official record.
```

## User prompt template (`buildUserPrompt`)

```
City: ${cityName}
Speaker: ${personName}
Roster (party — members):
${parties.map(p => `${p.name}: ${p.people.map(x => x.name).join(', ')}`).join('\n')}
${agendaBlock}Correct the numbered utterances:
${utterances.map((u, i) => `${i + 1}. ${u}`).join('\n')}
```

`agendaBlock` is empty when there are no agenda items, otherwise:

```
Agenda items of this meeting (source for street/project/entity names):
1. ${agenda[0].name}
2. ${agenda[1].name}
...
```

Injected dynamic context: `cityName`, `personName` (`speakerName || "(unknown)"`),
the **party roster** (`Party: member1, member2` per line — not JSON), optional
**agenda item titles**. Nothing else.

## Key finding for our work

**No domain glossary, acronym list, or `customVocabulary` reaches the fix LLM.**
The only term-grounding is the party roster + agenda titles. Acronyms (ΕΕΤΑ,
ΚΕΔΕ, ΔΕΥΑΟ…), toponyms beyond agenda titles, organisations, and legal terms are
**not** grounded — they fall under priority 1/4 only as generic instructions.

This is the headroom the [improvement loop](../specs/fix-task-improvement-loop.md)
targets, and the hook for the
[dynamic-vocabulary feature](dynamic-vocabulary-and-entities.md): a per-city +
global glossary block injected here is the highest-leverage prompt change.
