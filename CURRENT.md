# Current Work

Last updated: 2026-08-17

**The endgame plan is finished.** All four workstreams of
[`docs/specs/2026-08-11-endgame-handoff-plan.md`](docs/specs/2026-08-11-endgame-handoff-plan.md)
are closed, 2026-08-12. Read
[`docs/reports/2026-08-20-final-report.md`](docs/reports/2026-08-20-final-report.md)
first — it is the answer to the project's question, with its limits.

What is left is the queue below: the name-lexicon deployment, one unresolved
fidelity gap, and the external resource research ticket. The publish decision was
carried out on 2026-08-16. The adapter-confidence screen closed on 2026-08-17.

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
   beatable without targeted retraining. **`exp-2026-08-11-name-repair` CLOSED
   2026-08-17**: E was re-measured on W and survives (−0.00075
   [−0.00109, −0.00044]), but four of the six Βήμα-3 gates cannot be evaluated
   without a name-level adjudication that has never been done for W, and the
   ≥300-activatable-points gate is unassessable on a holdout with no rosters.
   Shadow only. [Report](docs/reports/2026-08-17-name-repair-on-w.md).
1. Publish `artifact-adapter-fixed` to HuggingFace — **done 2026-08-16**, commit
   `e214de71` at `opencouncil/whisper-large-v3-el-council-lora`. The hub weights are
   the corrected ones.
1b. `exp-2026-08-13-targeted-deletion-training` — **CLOSED 2026-08-16** after
   its first screen came back negative: the deletion-targeted mix *raised* the deletion
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
   arm. **2026-08-17: it has now been re-measured on W and it transfers.**
   W 0.10046 → 0.09971, −0.00075 [−0.00109, −0.00044], directional primary endpoint
   met, both rate gates unchanged *and* zero of 247 windows show any change in
   deletions or insertions, no single window or meeting carrying more than 8.9% of
   the 56 net edits. The firing rule had to be re-specified from a token-set rule to
   an MSA-column rule first (protect `agree` columns), and that restriction is what
   the number depends on: firing on unanimous columns as well gives −0.00031 with a
   CI including zero. **Roster coverage is 94%, not 21.5%** — the untried-inventory's
   figure was the fetch-failure count. Record CLOSED, shadow only.
   [Report](docs/reports/2026-08-17-name-repair-on-w.md).
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
4. `exp-2026-08-16-tse-overlap` — **CLOSED 2026-08-16**. All four synthetic
   mechanism gates passed, but the real-overlap audit had only 1/6 enrollable
   speakers and showed no recovered reference word. TSE is not a serving candidate.
   [`docs/reports/2026-08-16-tse-overlap.md`](docs/reports/2026-08-16-tse-overlap.md).
5. `exp-2026-08-16-adapter-confidence` — **CLOSED 2026-08-17**. Our own per-word
   probability predicts our own errors: mean within-meeting AUROC **0.8151** on the
   gold set (permutation null 0.4999, p = 0.0005; LOO 0.800–0.837; dropping the
   error-dominant meeting raises it to 0.820), and it is **not the same signal** as
   Soniox's — the errors co-occur strongly (56 against 21.0 expected) while the
   bottom-decile flags overlap at 8 against 8.54, i.e. chance. **No combination gain
   is established** (4/6 meetings positive, both CIs include zero), and 41.1% of our
   edit operations are deletions, so confidence reaches only 58.9% of our damage
   against Soniox's 77.2%. `exp(avg_logprob)` is not a substitute (0.611, worse in
   all six meetings). **The gate failed**: asking for word timestamps changes the
   transcript in 101 of 102 paired windows at a pooled 7.7%, so these confidences
   belong to their own decode and **cannot be attached to the frozen fusion input W**.
   The near-zero deletion delta is cancellation — 82 of 102 windows move, absolute
   values summing to 613 against a net −5, with whole passages appearing and
   disappearing. Read alongside the `exp-2026-08-16-w-rt-confidence` negative below:
   **no downstream fusion gain from either confidence source is established, and
   neither experiment identifies a binding constraint.** Two caveats travel with it:
   the decode's frozen 23:45 stop was **mechanically violated** (the watchdog died
   with the session, the passes overran ~1 h, timestamps were not recorded, 102 of
   247 paired) — blindness survived and the binary gate result is invariant, but
   every magnitude from those 102 is descriptive; and the earlier claim that our
   local CPU decode is **bit-exactly reproducible is withdrawn** (16 of 18, not 18 of
   18). [`docs/reports/2026-08-16-adapter-confidence.md`](docs/reports/2026-08-16-adapter-confidence.md).
6. `exp-2026-08-16-open-task-resources` — **OPEN, research index started**. The
   primary-source resource index is at
   [`docs/reference/external-resources/2026-08-16-open-task-resources.md`](docs/reference/external-resources/2026-08-16-open-task-resources.md).
   Re-run the MacBook Grok loop when its composer is healthy, reconcile citations,
   then graduate only concrete decisions into Wayfinder tickets.

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

- `exp-2026-08-17-confirmation-audit` closed: **the autoresearch confirmation partition
  is invalid as confirmation for the LLM-composer family F1, and no confirmation is
  spent.** F1 was selected by reading oracle counts, the majority-error taxonomy and the
  "25% of the gap" figure — all computed by scripts that call `fusion_lab.load_substrate()`
  bare, which has no city filter, so all four result JSONs read `n_windows 247 /
  n_cities 10` and 27,665 confirmation tokens are in every denominator. Earlier
  experiments quoted per-city outcomes over all 10 cities as a habit. The harness itself
  is clean (journal: 16 registered / 16 searched, all at 153 windows, zero
  `CONFIRM_BATCH_FROZEN`) and the split is by **city**, coarser than meeting, so nothing
  straddles — but "never read" is true of the harness, not of analyst knowledge. A fresh
  holdout cannot be carved: the only unread material is the 7 sealed eval-freeze windows
  at 2,101 tokens, where the ship floor is 2.1 edits against 75. **F1 may run, reports
  exploratory results only, no confirmatory CI, budget stays 5 of 5.** Settled beside it:
  the harness permits **one** confirmation batch ever per `PROTOCOL_VERSION`, holding at
  most 5 ideas — the report's "budget of 5" is the per-batch idea count and the code
  refuses a second batch outright.
  [`docs/reports/2026-08-17-confirmation-audit.md`](docs/reports/2026-08-17-confirmation-audit.md).
- `exp-2026-08-11-name-repair` closed: **the project's one measured positive survives
  its substrate change, and the coverage story that would have killed it was wrong.**
  Arm E on W: 0.10046 → 0.09971, −0.00075 [−0.00109, −0.00044], preregistered
  directional endpoint met, S 3200 → 3144 with D and I byte-identical in *every* one
  of the 247 windows, LOO stable over windows, meetings and cities, largest window and
  largest meeting each 8.9% of the 56 net edits. The crux was re-specifying "act only
  where the three systems disagree" from a token-set rule (meaningless once the output
  is composed per column) to "protected iff the token's MSA column is class `agree`";
  the paired contrast against firing everywhere is +0.00044 [+0.00020, +0.00069], so
  the restriction is load-bearing. Roster coverage is **232/247 windows (94%)**, not
  the 21.5% the untried-inventory reported — 56 was the fetch-failure count in
  `data/pii/fetch_rosters.log`, and that report now carries an erratum. What is real
  from it: all 7 sealed holdout meetings genuinely have no roster. **The Βήμα-3 gates
  are unassessable, not failed:** four need a name-level adjudication never done for W,
  and ≥300 activatable points would need 13–15 h of untouched roster-covered audio at
  the measured 0.61–0.72 firings per window. Shadow only.
  [`docs/reports/2026-08-17-name-repair-on-w.md`](docs/reports/2026-08-17-name-repair-on-w.md).
- `exp-2026-08-17-majority-error-taxonomy` closed: **the 25% class is not one thing,
  and 27.7% of it is definitely not a selection failure, with up to 39.6% not cleanly
  attributable to one.** Of 6,645 `exact_2_of_3`
  columns 1,719 have a wrong majority; read off the oracle DP's optimal-support set
  rather than one backtrace, 1,038 are selection failures (the minority token *is* the
  reference word), 318 coverage, 205 ambiguous, 158 spurious — and 719 of the 1,719
  (41.8%) have zero marginal benefit with W's other choices frozen. The census's 1,245
  "W differs from the oracle" is a different set (1,215 overlap, 30 tie-breaks, 504
  wrong majorities missing), and "the oracle takes the lone dissenter 1,245/1,245" is a
  candidate-set identity, not recoverability. Largest linguistic bucket is
  function-word pairs (19.9%, ~28% of the measurable hindsight gain); **Greek
  morphology does not dominate** (10.9%). The entity cross-check answers what
  `exp-2026-08-16-error-mined-terms` left open: of 99 wrong majorities whose correct
  word is a frozen term, **93 are in the own-city file and 83 survive the roster
  gate**, so coverage is not the constraint — the frozen `name_repair.select()` fires
  on 36 and the largest attrition is the minimum-length eligibility gate (21 of 33
  `no_candidate`). But only 28 of the 83 are recoverable at all, capping this funnel at
  0.037 WER points. Hindsight throughout; no arm, no gate.
  [`docs/reports/2026-08-17-majority-error-taxonomy.md`](docs/reports/2026-08-17-majority-error-taxonomy.md).
- `exp-2026-08-16-w-rt-confidence` closed: **no confidence arm met its criteria, and
  the two with the most room never fired.** The free realtime path re-transcribed all
  247 windows with per-token confidence in 34 minutes at zero cost, which forced a
  parallel substrate (**W-rt**, `stt-rt-v4` in place of the cached paid `stt-async-v5`)
  because Soniox is one of W's three voters. Inside it, the occupancy arm and the
  majority-override arm both fitted "never fire" in **all ten** leave-one-city-out
  folds; the asymmetric weighted vote moved WER by −0.00035 with both intervals
  including zero and both rate gates failing. The ungated control says the occupancy
  material is real and unreachable: firing on everything cuts deletions 37% and nearly
  doubles insertions. Post-hoc, confidence's AUROC on the decisions a fusion arm
  actually makes is **0.587–0.703**, against 0.8167 on the gold-set error-detection
  task — weakest where the mass is. Descriptive only: W-rt scores 0.09931 against old
  W's 0.10046 on these windows, which is a **model swap, not a result**. Consequence:
  the ~$0.82 `stt-async-v5` run proposed the same morning is **not** justified by the
  fusion-arm hypothesis.
  [`docs/reports/2026-08-16-w-rt-confidence.md`](docs/reports/2026-08-16-w-rt-confidence.md).
- `exp-2026-08-16-overlap-speaker-arms` closed: **one negative that must not be
  overstated, and one positive.** The inside-overlap speaker advantage of
  `exp-2026-08-16-pyannote-transcription` (−0.00558 on top of whole-window selection)
  was carried onto W under preregistration, on a cut-independent mask with a
  dose-matched placebo, and **was not demonstrated**: turn minus placebo +0.00094, CI
  [−0.00600, +0.00833] — which still contains −0.00558, so this is a failed
  demonstration, not a demonstrated failure (power ≈33% against that effect, 80% MDE
  ≈0.0103, only 43 of 103 meetings informative). What is established: W is the best
  thing inside its own overlap neighbourhood, the three selection-shaped patches cost
  +0.0022 to +0.0036 WER with CIs excluding zero, the two composition-shaped ones are
  unresolved, **all five failed the search screen and zero of five confirmations were
  spent** — and none could have been, because the hypothesis was generated on all 247
  windows including the sealed confirmation cities. Separately, the per-speaker
  omission rule the parent refused to try was preregistered and **works**: the obvious
  density form provably cannot fire on the one-lost-speaker case, so the quantity became
  a missing-speaker count; recall goes 0.1075 → 0.2020 (CI excludes zero) and against a
  duration-only detector at a matched alert budget precision is +0.0543
  [+0.0042, +0.1089]. The price is 2.27× the alerts at an unresolved 5.3-point precision
  loss. The gold set gave 4 flags in 27 cells, below its own preregistered floor, so the
  withdrawn "lower bound" label stays withdrawn. Zero pyannote calls, zero GPU.
  [`docs/reports/2026-08-16-overlap-speaker-arms.md`](docs/reports/2026-08-16-overlap-speaker-arms.md).
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
