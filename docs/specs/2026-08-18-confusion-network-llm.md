# Confusion-network arbitration over whole segments

Status: DESIGN, nothing run
Supersedes the design of `exp-2026-08-17-llm-f1-arbiter` (not its results)

## Why the previous design failed, in one paragraph

F1 asked the model about **one aligned column at a time**, in isolation, with 20
tokens of masked context, and forced a binary answer. Measured on 2,223 resolved
questions: overrides were correct 42.4% of the time against a 19.4% base rate, so
the model has real discrimination, but 42.4% is below the 50% break-even and the
arm made W **worse** — delta WER +0.000467, CI [+0.000082, +0.000849], excludes
zero, damage concentrated in substitutions. Asking it to grade its own certainty
did not rescue it: 81% of returned confidences were 98 or 99, so the dial had no
travel. The failure is **not** the prompt — the prompt already forbade preferring
the fluent or shorter option — and it is not the model's Greek. It is that a single
column, stripped of its sentence, is not enough evidence to beat two ASR systems
that heard the audio.

## The change

Three things move at once, because they are one idea.

**One: the unit becomes a whole speaker segment, not a column.** The model reads
continuous speech with its neighbours intact and decides every disputed position in
it at once.

**Two: the disputed positions carry acoustic probabilities.** We already own a
confusion network and never rendered it as one — the 3-way MSA alignment *is* the
network, and per-word confidence now exists natively for the adapter row
(`exp-2026-08-18-conf-substrate`, 73,560 of 80,717 columns covered, 91.1%).

**Three: high-confidence segments never reach the model.** Routing is the whole
economy: fewer calls, and none of them spent where there is nothing to decide.

## What the model sees

```
ΘΕΜΑ: αναπλάσεις πλατείας και κυκλοφοριακές ρυθμίσεις
ΟΜΙΛΗΤΕΣ: Παπαδόπουλος, Λιόλιος, Καραγιάννη, Βλαχοδήμος
ΟΡΟΙ: Ερμού, ΣΒΑΚ, Δημοτική Επιτροπή Ποιότητας Ζωής

…και θα ήθελα να πω ότι η [1: επιτροπή:0.91 | επιτροπής:0.09] αποφάσισε
για τα [2: δύο:0.72 | δυο:0.23 | δις:0.05] έργα ⟨3: μόνο-Soniox: «στην οδό Ερμού»⟩
στις [4: οκτώ:0.82 | οχτώ:0.18] το βράδυ.
```

- Plain text is **frozen**. It is what all systems agreed on and it is not editable.
- `[n: …]` is a disputed position with its candidates and their acoustic weights.
- `⟨n: μόνο-Soniox: …⟩` is a span **only one system heard** — a candidate insertion,
  marked as such, with the system named. This is the channel the previous design
  had no way to express: `msa.vote_column` discards a token proposed by one system
  before any vote, so 4,460 such columns were structurally invisible.

## What the model returns

A list of `position -> choice`, nothing else:

```json
[{"n": 1, "pick": "επιτροπή"}, {"n": 2, "pick": "δύο"},
 {"n": 3, "pick": "ΝΑΙ"}, {"n": 4, "pick": "ΑΠΟΧΗ"}]
```

**It reads prose and returns choices.** This is deliberate and non-negotiable. The
one time a model in this project was allowed to emit free text, 2 outputs of 150
added commentary and cost 13 WER points; a whole-window selector picked the shortest
candidate 103 times in 129 and raised deletions 41%. Full context is what it needed;
a free pen is not. `ΑΠΟΧΗ` stays a first-class answer, and for a `μόνο-Soniox` span
the answer is `ΝΑΙ` (it was said, keep it) or `ΟΧΙ`.

## Routing

A segment is sent to the model only if it has at least one disputed position **and**
at least one of those positions is below a confidence threshold `tau`. Everything
else keeps W's output untouched and costs nothing.

`tau` is fitted **out of fold** on the SEARCH cities and applied unchanged to
CONFIRM, exactly as every other tunable in this project. It is not chosen by
looking at the result.

## Prompt

Few-shot, with worked examples chosen to teach the failure modes we have measured
rather than the task in general:

1. a morphological pair where the grammar decides — the one bucket that already
   works, 64.6% precision
2. a **number**, where the correct answer is `ΑΠΟΧΗ`, because measured precision
   there is 20.0% against a 20.7% base rate: literally no information
3. a `μόνο-Soniox` span that **is** real speech, since 40.8% of Soniox's scored
   insertions sit on words a human heard
4. a case where the fluent option is wrong

## Model

`gpt-5.6-luna` at **maximum reasoning effort**, through the codex bridge. The pilot
showed 0% invalid answers at every batch size from 6 to 48, so transport is not the
constraint and effort can go up.

## Preregistered gate

Primary is WER against W on the same substrate, meeting-clustered paired bootstrap.
An arm passes only if the WER delta's 95% upper bound is below zero **and** the
deletion rate does not rise (upper bound <= +0.0005) **and** the insertion rate does
not rise (upper bound <= +0.0005).

Reported alongside, never as the gate: the count of edits, precision against the
column oracle per bucket, and the share of segments routed.

## What would make this fail

Named honestly, before running. The confusion network gives the model more evidence
per decision but does not change the arithmetic of the floor: about **75 net correct
edits** on 74,917 tokens are needed to clear the ship test. The previous design
produced 126 correct against 171 wrong. Segments and probabilities have to move
precision above 50% *and* keep the volume, and only the first of those is argued
for here.
