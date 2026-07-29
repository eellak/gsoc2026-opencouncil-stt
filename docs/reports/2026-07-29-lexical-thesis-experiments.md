# Testing the "lexical costume" thesis — three cheap experiments (2026-07-29)

The [asr-v2 design note](../specs/asr-v2-design.md) claims the correction dataset is
"a lexical and textual signal (names, domain terms, punctuation, casing) wearing an
acoustic costume". That claim rested on one p=0.021 measurement and a 0.155-vs-0.158
comparison with a reference-anchoring problem. These three CPU-scale experiments try to
falsify it directly. Plan reviewed by Codex (gpt-5.6-sol) — its distractor-control and
CI suggestions are incorporated; its glossary-arm suggestion was skipped on leakage
grounds (the glossary's provenance vs the corrections is unverified).

Scripts: `eval/controlled_eval/exp_edit_taxonomy.py`, `exp_llm_postedit.py`,
`exp_oracle_hotwords.py` · raw: `results_edit_taxonomy.json`,
`results_llm_postedit.json`, `results_oracle_hotwords.json`.

## A. What kind of edits are the corrections? (n = 3,832 includes)

Word-level diff of `initial_before_text` → `final_after_text` over every included
correction in `data/asr/export.jsonl`; each edit op classified against DF-filtered
per-meeting rosters and per-city glossaries. (First attempt used the global glossary
and mis-attributed 48% of ops to it — it is polluted with common words ("κάνει",
"πράγμα"); dropped after sampling.)

| class | ops | % ops | % changed-token volume |
|---|---|---|---|
| spelling-tweak (small char-distance fix, incl. inflection) | 2,088 | 43.1% | 32.6% |
| **other-lexical** (ordinary-word substitution — plainly acoustic) | 1,948 | 40.2% | 45.6% |
| glossary-term (per-city, DF-filtered) | 408 | 8.4% | 10.2% |
| number | 228 | 4.7% | 6.9% |
| roster-name | 176 | 3.6% | 4.8% |

Utterance level: **8.8%** of corrections are pure formatting (invisible after WER
normalization). Of the rest, 1,858 / 3,495 (53%) contain *only* contextual/textual ops
(spelling, name, glossary, number), 1,174 (34%) contain *only* other-lexical ops, 463
(13%) are mixed.

**Read:** the thesis survives only in a weakened form. Names + glossary + numbers —
the part contextual biasing can address — are just **17% of edit ops (~22% of token
volume)**. The largest class is spelling/inflection tweaks (43%), which are
text-recoverable but *arise* acoustically; and a full 40% of ops are ordinary-word
misrecognitions, which are acoustic, full stop. "Mostly lexical wearing an acoustic
costume" overstates it; "roughly half context/text-recoverable, half genuinely
acoustic" is what the data says. Caveat: the classifier is a lever-attribution
heuristic (regex + edit distance), not ground truth.

## B. LLM post-edit on the corrected subset (n = 50)

<!-- RESULTS-B -->

## C. Oracle candidate-selection bound for roster biasing (n = 59, 114 gold names)

<!-- RESULTS-C -->

## Verdict on the thesis

<!-- VERDICT -->

## Confounds this does not remove

- **Reference anchoring (A, B):** the human references were produced by editing the
  original ASR output, so both `scribe_before` and an LLM that edits it are
  structurally favored over independently-decoded systems. Only the gold eval set
  (asr-v2 note §1) removes this.
- **Selection bias (B):** only known-corrected clips are scored; over-editing on
  clean utterances is unmeasured.
- n = 50/59 with utterance-level pairing; deltas smaller than a couple of points are
  noise even with CIs.
