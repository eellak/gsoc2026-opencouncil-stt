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

Same 50 clips, same normalizing scorer as the capstone A/B. claude-sonnet via the
on-box CLI (`eval/backends.py`), zero-shot, temperature defaults, prompt frozen in
`exp_llm_postedit.py`. In 2/150 outputs the model appended meta-commentary despite
instructions; it is stripped (text after the first blank line) uniformly across
configs before scoring — both raw and stripped numbers are in the results JSON.

| config | WER | vs its source (95% CI on delta) |
|---|---|---|
| scribe_before (original ASR) | 0.1552 | — |
| base whisper | 0.1576 | — |
| ours (LoRA fine-tune) | 0.1761 | — |
| llm_plain (scribe, no context) | 0.1318 | [−0.052, +0.012] · 25/9/16 |
| **llm_roster (scribe + roster)** | **0.1195** | **[−0.056, −0.014]** · 27/4/19 |
| llm_base (base whisper + roster, stripped) | 0.1367 | **[−0.038, −0.001]** · 21/7/22 |

**Read:** the LLM post-editor is the first thing in this project that beats
`scribe_before` on its own subset — **0.155 → 0.119, −23% relative, CI excludes
zero** — where 8h31m of LoRA training produced a regression (0.176). The roster adds
a real but modest increment over plain cleanup (12/7/31 head-to-head). Two caveats
cut opposite ways: reference anchoring favors an editor of `scribe_before` — but the
same editor also improves **base whisper** output (0.158 → 0.137, CI excludes zero),
where anchoring works *against* the LLM, so the recoverability is not purely an
anchoring artifact. And the 2 chatter outputs show the failure mode that matters in
production: when the LLM does go off-script it destroys the utterance (+63 edits on a
17-word reference); a deployment needs an output-validity gate (length ratio /
newline check), which is trivial but mandatory.

## C. Oracle candidate-selection bound for roster biasing (n = 59, 114 gold names)

Same 59 clips and scorer as `ab_hotwords_names.py`; base and full-roster rows reused
from the 2026-07-25 run. Oracle = only the roster names actually present in each
reference (canonical roster surface forms, median 1 term/clip). Distractor = the same
number of full names from the same roster that are *not* in the reference.

| config | WER | name recall | false name insertions |
|---|---|---|---|
| base | 0.3412 | 27.2% (31/114) | — |
| base + full roster | 0.3460 | 36.0% (41/114) | — |
| base + oracle | 0.3356 | **36.0% (41/114)** | 0 |
| base + distractor | 0.3578 | 32.5% (37/114) | 1 |

**Read:** two findings, both deflating for biasing-as-panacea.

1. **The hotwords mechanism is saturated.** Perfect knowledge of which name is spoken
   buys *zero* recall over dumping the whole roster (41/114 both). The 64% of names
   still missed are not a candidate-selection problem — the decoder cannot produce
   them even when told the answer. Getting past ~36% on this audio needs either a
   stronger biasing mechanism (shallow fusion / word boosting) or better acoustics —
   the whisper prompt is exhausted.
2. **About half the biasing gain is generic, not informational.** Wrong names alone
   lift recall 27.2% → 32.5% (+6 names), presumably by pushing decoding toward
   name-like outputs; the true roster adds only ~4 more (+10 total). The +8.8pp
   headline from 2026-07-25 is real but its mechanism is cruder than "the model reads
   the roster". On the plus side, biasing is safe: 0–1 false insertions of supplied
   names across both arms.

## Verdict on the thesis

The thesis survives **only in a weakened, more useful form**:

- **Confirmed:** the corrections are substantially text-recoverable. An LLM
  post-editor with the roster — untried until today, ranked #5 in the postmortem — is
  now the **single best system on the corrected subset (0.119 WER vs scribe's 0.155
  and the fine-tune's 0.176)**, and it also improves base whisper output where
  reference anchoring works against it. The design note's instinct that this lever
  was "probably underrated" was right.
- **Refuted (strong version):** "mostly lexical" is wrong as an accounting statement.
  40% of edit ops are ordinary-word misrecognitions and the audio-hard name problem
  is real: even oracle hotwords leave 64% of names untranscribed. The acoustic
  frontier exists; it is just not where the fine-tune was pointed.
- **Revised model of the win:** whisper-prompt biasing is a saturated, partly generic
  mechanism worth its +8.8pp but no more; the scalable context lever is the LLM
  post-editor (and, if pursued, decode-time boosting in a non-whisper stack).

Priority implication for the roadmap: the LLM post-editor moves from idea #5 to the
top of the no-retrain track, behind only the gold eval set — measured on *these* 50
clips it beats everything, and its two failure modes (meta-commentary, over-editing
clean text) are gateable/testable next.

## Confounds this does not remove

- **Reference anchoring (A, B):** the human references were produced by editing the
  original ASR output, so both `scribe_before` and an LLM that edits it are
  structurally favored over independently-decoded systems. Only the gold eval set
  (asr-v2 note §1) removes this.
- **Selection bias (B):** only known-corrected clips are scored; over-editing on
  clean utterances is unmeasured.
- n = 50/59 with utterance-level pairing; deltas smaller than a couple of points are
  noise even with CIs.
- **The fine-tune numbers quoted here are subset-specific.** The same-day
  [gain-decomposition report](2026-07-29-finetune-gain-decomposition.md) (parallel
  work, n=300) shows the n=50 "regression" does not replicate once the ≥4s/≥6-word
  filter is dropped — the LoRA wins −4.35pp overall, driven by short-clip
  hallucination suppression. "Beats the fine-tune" in section B means *on this
  filtered subset*; it does not reinstate the postmortem's verdict against the LoRA.
