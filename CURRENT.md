# Current Work

Last updated: 2026-08-16

**The endgame plan is finished.** All four workstreams of
[`docs/specs/2026-08-11-endgame-handoff-plan.md`](docs/specs/2026-08-11-endgame-handoff-plan.md)
are closed, 2026-08-12. Read
[`docs/reports/2026-08-20-final-report.md`](docs/reports/2026-08-20-final-report.md)
first — it is the answer to the project's question, with its limits.

What is left is the queue below: one experiment (name lexicon) and one gap
(fidelity-to-audio). The publish decision was carried out on 2026-08-16.

Canonical research state: [`research/ledger.json`](research/ledger.json).
Agent protocol: [`CLAUDE.md`](CLAUDE.md).

**Η σειρά εκτέλεσης μέχρι τις 23/8 ζει πλέον στον χάρτη**
[«Το καλύτερο δυνατό μοντέλο + serving harness μέχρι 23/8»](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/3)
(GitHub Issues, label `wayfinder:map`). Το queue παρακάτω μένει ως περιγραφή
κατάστασης· ο χάρτης είναι αυτός που λέει τι πιάνεται μετά και με ποια σειρά.
Το `exp-2026-08-14-hparl-probe` βγήκε **εκτός scope** για αυτόν τον κύκλο (ανοιχτό
νομικό ζήτημα CLARIN 1602)· το record μένει OPEN στο ledger για μετά το GSoC.

The long narrative that used to live here is at
[`archive/current/2026-08-10-CURRENT.md`](archive/current/2026-08-10-CURRENT.md).
It is history, not state.

## Objective

Deliver a defensible answer to one question for GSoC: does domain fine-tuning of
whisper-large-v3 improve Greek council transcription enough to matter? The corrected
adapter (`artifact-adapter-fixed`) is the candidate. Everything before 2026-08-01 was
trained through the label-prefix bug and cannot answer it.

## Work queue

0. **Serving-stack ladder: CLOSED 2026-08-12** — one survivor.
   [`docs/reports/2026-08-12-serving-stack-ladder.md`](docs/reports/2026-08-12-serving-stack-ladder.md).
   E (post-hoc name repair) is real: −0.25 on validation, −0.08 pooled / −0.28
   unseen on the one frozen benchmark look, all CIs exclude zero. B, C, D all
   rejected with evidence. Standing finding: **the deletions live in the
   weights** (thresholds never fire, audio is covered, words absent from all 8
   beam hypotheses) — no serving-time technique reaches them; Scribe is not
   beatable without targeted retraining. Remaining task: deploy E behind the
   Βήμα-3 shadow gates (`exp-2026-08-11-name-repair`, OPEN).
1. Publish `artifact-adapter-fixed` to HuggingFace — **done 2026-08-16**, commit
   `e214de71` at `opencouncil/whisper-large-v3-el-council-lora`. The hub weights are
   the corrected ones.
1b. `exp-2026-08-13-targeted-deletion-training` — OPEN, **its first screen came
   back negative, 2026-08-16**: the deletion-targeted mix *raised* the deletion
   rate (0.0600 → 0.0788 per reference token, CI excludes zero) while lowering
   substitutions, with WER flat, and the external-pack stage-1 of
   `exp-2026-08-14-external-packs` (RUN 2) changed nothing detectable on top of
   it (every paired CI includes zero). Both are single-seed screens against a
   2.1-point per-seed spread. `artifact-adapter-fixed` keeps the candidate slot;
   the frozen tree's branch is **no blind retry, error analysis of the
   deletion-hard supply first**.
   [`docs/reports/2026-08-16-screens-eval.md`](docs/reports/2026-08-16-screens-eval.md).
   **That error analysis ran the same day and closed as
   `exp-2026-08-16-deletion-hard-coverage`: neither frozen gate was established,
   so the user's training freeze holds for this cycle.** New deletions are 68–72%
   "ordinary speech" (the residual bucket, not a mechanism); the largest
   mechanical category is names at 18.0%/11.8% against a >40% gate. Uncovered
   speech >1.0 s reaches 10.75% of the rows that have a Soniox witness — under the
   15% gate, and ~8–9% if the measured seconds-proxy calibration transports
   (not established) — but 1,260 of the
   3,921 deletion-hard rows are the pre-wave human-reviewed stratum with **no
   witness at all**, so a cohort-level failure is not provable either. The
   realized mixture is the bigger anomaly: names got 2.6% against a designed 10%,
   and names are the one category enriched (6.8x/4.4x) in the new deletions.
   [`docs/reports/2026-08-16-deletion-hard-coverage-audit.md`](docs/reports/2026-08-16-deletion-hard-coverage-audit.md).
   Prior state: **candidate supply
   unblocked 2026-08-13**: user audited 228 gap3 items (94.7% accept overall);
   the calibrated strict stratum (found_frac≥0.85 ∧ n_added≥5, 97.4% agreement)
   was bulk-accepted (324 items, username=auto-verifier) with user consent. The
   full Soniox wave then finished and 2,643 rule passers were auto-accepted; the
   built `deletion_hard` bucket is 3,921 rows / 5.4 h.
   Recipe stays frozen (Codex-reviewed 55/25/10/10, 3 seeds/arm, deletion gate +
   insertion guard) and **unused**; +78h trusted clean backbone still available at
   zero human cost.
2. Name work continues as the **post-hoc roster repair** arm of
   `exp-2026-08-11-name-repair` (inside the ladder above). Decode-time hotword
   biasing is closed: it lifts name recall 51→65% but costs +0.34 WER in
   deletions and fails its preregistered gate. **2026-08-16:** the same repair
   now has a number on top of the fusion vote across all 10 benchmark cities —
   −0.083 WER points, CI excludes zero, both rate gates pass identically because
   it only moves substitutions (`exp-2026-08-16-roster-grounded-selection`).
   **Later the same day:** that measurement sits on top of the whole-window vote,
   which `exp-2026-08-16-composition-over-selection` has now displaced as the fusion
   arm. The repair has not been re-measured on top of the per-column composition and
   its −0.083 does not transfer to it unexamined.
3. Decide whether fidelity-to-audio changes this ranking. The benchmark measures
   agreement-with-OpenCouncil; the one time both were measured, the ranking flipped.
   **`exp-2026-08-16-gold-set` CLOSED 2026-08-16** — the corrected adapter now has a
   fidelity-to-audio number on untouched meetings: **0.284** [0.169, 0.455] against
   a human who listened, deletions 0.116, on 27 cores in 6 meetings / 6 cities.
   The published OpenCouncil pipeline scores 0.198 on the same audio, but **the
   ranking flips between scoring regions and no system ordering is claimed**.
   [Report](docs/reports/2026-08-16-gold-set-findings.md).
   Three things it did settle: 4 of every 5 words a second system has and ours
   lacks were really said (53/66); the production pipeline leaves **5.8% of
   spoken blocks with no published utterance at all** and loses 28.6% of certain
   words inside overlap; and agreement-WER is a different quantity from
   fidelity-WER, not a correctable offset.
   **Still open:** W itself was never run on this audio — there is no ElevenLabs
   credential in this environment — so the question the gold set was built for
   ("are the words the fusion recovers real?") is answered only for the candidate
   pool, not for W. Getting a Scribe v2 key is the unblocking step.

## Product decision: answered 2026-08-12 — transcripts are **clean**

Filled pauses («εεε») and false starts are **stripped**. Asked and answered once, per
the handoff plan. This is the standard future listening hours should be produced
against, so that they are compatible with each other.

Nothing frozen was changed on the strength of it: the benchmark normalizer, the
existing labels and the 2026-08 evaluation freeze all stay exactly as they were. The
decision governs new data collection, not any number already measured.

## Blockers

- Item 1 is done: the corrected weights are on the hub since 2026-08-16.
- The dataset itself remains on **legal hold** (DPO, 2026-07-17): text-level PII
  removal does not anonymise it because each row links audio carrying the voice. See
  [decisions/data.md](docs/decisions/data.md).

## Three closed doors — do not reopen without a reason

Each cost real time this cycle and each is answered in the ledger:

- **Decode thresholds** (`exp-2026-08-12-decode-ablation`). The no-speech gate fires
  **zero** times on the 39 windows, so it cannot be causing our deletions; removing
  the temperature fallback makes every metric worse.
- **Label purity** (`exp-2026-08-13-correction-only`). Dropping all `no_edit` rows
  moves WER by +0.0015, against a known 2.1-point per-seed spread.
- **Data scale** (`exp-2026-08-11-wer-levers-research`). The dominant residual error
  is homophone orthography the audio cannot decide; ~1300h buys ~0.5 points.

The 2026-08 evaluation freeze
([`manifest.json`](research/eval-freeze-2026-08/manifest.json)) is the substrate for
all three: 39 validation windows, 31 meetings, 11,911 reference tokens. Its 7
temporal holdout windows are **still sealed** — no arm ever passed a gate that would
have released them.

## Recently changed

- `exp-2026-08-16-autoresearch-harness` closed: **the idea loop exists and its first run
  found nothing**, which is the honest outcome for a smoke test. 11 ideas registered, 11
  evaluated, 1 refused as a cosmetic variant, 0 through the screen, **0 of 5
  confirmations spent — the confirmation partition has never been read.** The 10 cities
  are cut once by an outcome-blind token-balance rule into a 6-city search partition and
  a sealed 4-city confirmation partition, and the API enforces the split; exactly one
  confirmation batch may be frozen per cycle, before any confirmation number exists; the
  p-value is a null-imposed studentized wild cluster bootstrap-t, not the percentile
  tail; and the ship gate is a **one-sided minimum-effect test** (H0: dWER ≥ −0.0010)
  under Holm, with BH reported beside it, because a monotone arm touching 4 meetings
  excites a percentile CI at any effect size. Measured under the null: 40 ideas give "some idea significant" 87% of the time
  and "some idea ships" 5.5%. All three ways of overriding a 2-of-3 majority — the class
  holding 25% of the gap in hindsight — came back **worse**, CI excluding zero on the
  wrong side, in all six search cities. Any further idea search on this substrate goes
  through the harness.
  [`docs/reports/2026-08-16-autoresearch-harness.md`](docs/reports/2026-08-16-autoresearch-harness.md).
- `exp-2026-08-16-soniox-confidence` closed: **Soniox per-word confidence does predict
  human-verified errors** — mean within-meeting AUROC **0.8167** (preregistered GO
  threshold 0.60, null 0.5), all six meetings 0.78–0.90, permutation null topping out at
  0.602. But it is conditional on the word being *emitted*: **22.8% of edits in the
  scored region are deletions confidence cannot see**, and insertion detection — the
  thing that fails the occupancy gate — is the weakest arm at 0.773. The production
  `< 0.5` threshold gets its first ever calibration: precision 0.706, recall 0.164.
  Decision is **GO for one ~$0.82 `stt-async-v5` run** to test whether this transports
  off the free `stt-rt-v4` path; no fusion arm until it returns. Zero spend so far.
  [`docs/reports/2026-08-16-soniox-confidence-probe.md`](docs/reports/2026-08-16-soniox-confidence-probe.md).
- `exp-2026-08-16-char-vote-homophones` closed: **the columns are not there.** A census
  run before either arm was built found 34 strict-homophone columns out of 80,659
  (0.042%), so the homophone arm was **not built**; the per-character vote was built
  and out-of-fold (leave-one-city-out) gives 0.10046 → 0.10038, CI [−0.00026, +0.00009],
  which includes zero. What survives is structural: only 1,396 unresolved columns have
  W differing from the column oracle, so word-choice arbitration between the three
  transcripts can close at most ~35% of the 5.3-point gap to 0.0475, and a hindsight
  replay over every unresolved column closes 12.7% — against 25.0% for overriding
  2-of-3 token majorities and 14.2% for occupancy columns, the latter still failing
  the insertion gate. The mass sits where the systems agree, or agree 2-of-1, and are
  wrong together.
  [`docs/reports/2026-08-16-char-vote-homophones.md`](docs/reports/2026-08-16-char-vote-homophones.md).
  Reusable evaluator: `eval/controlled_eval/fusion_lab.py`.
- `exp-2026-08-16-composition-over-selection` closed: **stop selecting, compose.**
  An exact three-way word alignment of scribe + soniox + `artifact-adapter-fixed`
  with a per-column vote — no LLM, no audio, no speaker information — takes WER
  0.1201 → **0.1005** and lowers deletions, insertions *and* substitutions at once,
  every CI excluding zero, both rate gates passing, no LOO sign flip over windows,
  meetings or cities. It lands **below** the whole-window trio oracle (0.1064),
  because its output is a text none of the three systems produced: whole-window
  selection is not an upper bound on composition. The new per-position ceiling is
  the alignment-conditional column oracle at **0.0475** (range 0.0461–0.0479 across
  alignments — not an attainable ceiling). An LLM arbiter restricted to tie-broken
  columns shows no detected benefit; the length guard and the pyannote-grounded
  restoration of dropped text both fail the insertion gate, 2026-08-16.
- `exp-2026-08-16-roster-grounded-selection` closed: a closed term list inside the
  fusion selector helps only through the FREE phonetic repair (−0.083 WER points,
  CI excludes zero, both rate gates pass identically, 6% of the trio-oracle gap).
  The LLM selector fails its deletion gate on every variant and loses a full WER
  point: it picks Scribe on 215 of 244 windows, the shorter text in 103 of its 129
  deviations, and imports Scribe's deletions. Text-only selection is closed,
  2026-08-16.
- `exp-2026-08-14-hparl-probe` closed: HParl's minutes *are* faithful to their audio
  (6.1% Soniox disagreement on placeholder-free rows), but ~55% of rows carry `[UNK]`
  over real speech and no row anywhere carries an accent or punctuation mark. Usable
  audio, unusable targets; the corpus stays deprioritised, 2026-08-14. Addendum: the
  ML-processed mirror (`Elormiden/…`) is accented and `[UNK]`-free; a Soniox
  alignment filter at ≥0.95 keeps 49% of rows (≈60h), punctuation restored by
  `gpt-5.6-luna` under a word guard (74/74 clean, only 26% of segments are complete
  sentences). Record reopened OPEN: 10k-row pilot filter running, preregistration at
  [`docs/specs/2026-08-14-hparl-stage1-prereg.md`](docs/specs/2026-08-14-hparl-stage1-prereg.md),
  no GPU spend authorised, 2026-08-14.
- `exp-2026-08-20-final-report` closed: the answer is yes-but-modestly, and what
  remains points at names rather than at audio, 2026-08-12.
- `exp-2026-08-13-correction-only` closed: dropping the unverified half buys nothing,
  labelled suggestive; `artifact-adapter-correction-only` registered, 2026-08-12.
- `exp-2026-08-12-decode-ablation` closed: no threshold ships; six arms collapse into
  two behaviours, 2026-08-12.
- `exp-2026-08-12-ds-wer` closed: on domain terms we sit at 0.488 against Soniox
  0.328 and Scribe 0.372, far worse than our tie with them on overall WER, and our
  name errors are substitutions not deletions, 2026-08-12.
- `exp-2026-08-10-benchmark-fixed-adapter` closed: on unseen cities the corrected
  adapter is indistinguishable from Scribe and Soniox and beats its broken
  predecessor by 1.77 points, CI excludes zero, 2026-08-10.
- `exp-2026-08-10-packed-training` closed STOP; four overstated claims corrected
  after a Codex audit, 2026-08-10.
- `exp-2026-08-08-same-stack` gained a provenance erratum: its fine-tune arm was the
  broken adapter, 2026-08-10.
- `exp-2026-08-08-mixture-ratio` closed by user instruction, 2026-08-09.
