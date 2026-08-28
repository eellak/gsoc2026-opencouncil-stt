# Current Work

Last updated: 2026-08-28

## PAST GSOC

**The GSoC deliverable is submitted and `FINAL_REPORT.md` is frozen.** Do not edit
it again. One exception, 2026-08-28: three corrections after mentor review, and
nothing else. The review UI was missing from section 2; the stated reason for
dropping the human intervention rate did not match the recorded one
([`docs/decisions/metric-hir.md`](docs/decisions/metric-hir.md), mentor sync
2026-06-23); and section 4.1 now says in full that base `whisper-large-v3` is the
only open-weights system measured, so the table is not a ranking of open Greek ASR.

Work from 2026-08-24 onward is research past the programme: it lands in
dated reports under `docs/reports/`, in `research/ledger.json`, and here. Three
questions closed on 2026-08-24 after the report was frozen (near-miss vote,
two-system fusion Stage 0, deletion runs); their conclusions live in the ledger
and their reports, and are deliberately absent from `FINAL_REPORT.md`.

The open direction is stated in
[`docs/specs/2026-08-24-acoustic-ceiling.md`](docs/specs/2026-08-24-acoustic-ceiling.md):
where the acoustic limit of this approach actually sits, and how much of the
remaining gap is decoding rather than hearing.

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

## Open 2026-08-26 — training round 2, ceiling first

`exp-2026-08-26-training-round-2` is **OPEN, spec written, nothing measured**:
[`docs/specs/2026-08-26-training-round-2.md`](docs/specs/2026-08-26-training-round-2.md),
settled in a 15-decision grill. Order: three CPU ceiling measurements (phoneme-skeleton
fold of the 391, N-best oracle vs candidate budget on both adapters, base large-v3 on
the 39) while Harold listens to ~2 h of targeted clips outside every evaluation
meeting; then one $3 screen, continued-from-`cont_s{13,29,47}` (50/50 replay) against
from-scratch against the untouched control, six ladder gates. Frozen-encoder probe or
DPO on whisper's decoder only afterwards, chosen by the N-best curve. $50 GPU cap.
Beam-8 N-best headroom is already known to be 0.0071 (`exp-2026-08-12-serving-stack`).

**Amended 2026-08-27 after reading Fun-ASR (arXiv 2509.12508), reviewed by Codex
`a471616b`.** One idea was tried as an experiment and **closed the same day**:
`exp-2026-08-27-retrieval-hotwords` (spec §3.4) retrieved at most three roster surnames
per window by phoneme edit distance against the unbiased pass-1 output and re-decoded
only where the set was non-empty. **`STOP`** — 19 of 39 windows fired, mean prompt 11.1
tokens, and it bought +15 correct name mentions of 246 against a required 18, with the
deletion-rate CI upper endpoint still at +0.0118.
[`docs/reports/2026-08-27-retrieval-hotwords.md`](docs/reports/2026-08-27-retrieval-hotwords.md).
The decoy control is the finding worth carrying forward: one irrelevant surname added to
the same prompts costs a third of the name gain and more than doubles the deletion
penalty **while never being emitted** — the harm is suppression, not hallucination.
`exp-2026-07-25-hotwords` stays CLOSED and is confirmed. Name work stays in post-hoc
roster repair.
Three things were closed rather than added: noise augmentation is OFF TARGET (deletions
and substitutions rise at the same rate with overlap, and the causal experiment is
already negative); the paper's multi-witness mining rule cannot run on our pools, so the
five-witness benchmark windows are **calibration-only** and never supply a training clip
(user decision, 2026-08-27; **99 of the original 137** after the release of 2026-08-28); and Table 8 supplies no threshold for the DPO branch.
Two defects in the spec were fixed in the same change: §4.2 named the wrong benchmark
for the 391 exclusions, and §4.1 overstated the witnessed `deletion_hard` rows
(2,677 of 3,921) and named the wrong Soniox model.

**Step 2 is built and the listening queue is cut.** The exclusion manifest
(`~/.cache/oc-public/exclusion-manifest-2026-08/manifest.json`, sha256 `a45de4c6…`, 283
meetings and 5 whole cities off limits) is hash-pinned by the new cutter
`eval/controlled_eval/cut_listening_clips.py`, and `artifact-listen-queue-2026-08` holds
**721 clips / 2.16 h** over 95 meetings in 4 cities. The blind page is
`eval/controlled_eval/blind_listen.py`; the procedure is
[`docs/runbooks/2026-08-27-blind-listening-session.md`](docs/runbooks/2026-08-27-blind-listening-session.md).

Three things about that queue that decide how its result may be read:

- **It is four municipalities and they were bought.** The 2026-08-27 queue was Chania
  307 + Athens 216 only, because `calibration_260` alone blocked 1,305 of the 2,661
  candidates. On 2026-08-28 Harold released `chalandri` and `vrilissia` from that source
  (`CALIBRATION_RELEASED_CITIES`), which added 198 clips and cost **38 of the 137
  five-witness calibration windows** and **36 of the 247 `composition-rt-2026-08` fusion
  substrate windows** — a round-3 adapter may no longer be scored on those 36. Published
  fusion results are unaffected. `zografou` was left excluded as the worst trade, and
  there is no free subset: every meeting blocked only by `calibration_260` holds a
  calibration window (spec §4.2b).
- **108 rows were dropped because their span has been edited since Soniox heard it**,
  one by 13.25 s: their witness text and their cached wav belong to different audio.
  Found by Codex `4cf77e98` and confirmed by measurement before anything was published.
- **The overlap rule is applied literally after all.** An earlier draft of the cutter
  argued it could not be; the measurement says 724 of 899 clips touch no other training
  row. The 175 that do are dropped, which also removes label supersession as a worry.

Labels are collected under the **frozen** listening protocol (2026-08-09) rather than
§4.1's clean-transcript line — false starts stay in, and the uncertainty states, the
three-listen limit and the five hidden repeats are enforced by the page (spec §4.2c).
Pool 2 of §4.1 has no inputs on disk and is not built.

**The session is split, by Harold's decision of 2026-08-28.** Typing 721 clips from
nothing is 9–17 hours and he said plainly he would not do it; correcting Soniox instead
makes every label a derivative of the witness it is meant to judge. So: a **blind core
of 120 clips** typed from the audio alone, which is the only material a measurement may
use, then the remaining 601 with the Soniox text prefilled and edited, marked
`assisted`. The five hidden repeats are drawn from the core, the core comes first in the
session, `/prefill` returns 403 for a blind clip, and `training_rows(mode="blind")` is
how any number asks for the clean set.

**Every Soniox run this project has made was served by `stt-rt-v5`, not `stt-rt-v4`.**
Soniox removed v4 on 2026-06-30 and routes the name; all our runs are August. No
comparison is invalidated — everything we have is the same model — but the label was
wrong wherever it appeared, and the client now requests v5 by name.

## Closed 2026-08-28 — slowing the audio does not help Soniox in crosstalk

`exp-2026-08-28-tempo-probe` ran Harold's question — play the audio slower so the model
catches the words people talk over — as five arms on the gold set's 18 overlap episodes,
three interleaved cycles, 225 free `stt-rt-v5` sessions.
**STOP on all four arms** ([report](docs/reports/2026-08-28-tempo-probe.md)): the control
is the best arm, at 0.8756 overlap-episode recall against 0.8710 (`raw`, `pace070`), 0.8679
(0.85×) and 0.8587 (0.70×). The noise envelope is 0.0092; three arms sit inside it on the
losing side and **0.70× sits outside it**, while also being the only arm to fail the
collateral guard — recall outside the overlap falls 2.1 points. Slowing is not neutral, it
is harmful.

Harold found a real defect by reading the adjudication page: Soniox streams **sub-word
pieces**, and the first scorer treated each piece as a word, giving recall ≈ 0.32 for
everything. `group_words` in `eval/soniox_confidence_probe.py` already did it right; the
probe now reuses it instead of its own loop. The adjudication itself was **not run** — after
the fix no arm gains, so there is nothing for it to rescue.

## Open 2026-08-26 — integrating the fusion into the OpenCouncil pipeline

`exp-2026-08-26-fusion-integration` is **OPEN, spec written, nothing built**:
[`docs/specs/2026-08-26-fusion-integration.md`](docs/specs/2026-08-26-fusion-integration.md),
settled in an 18-decision grill. The shipped arm is **R1+R2 + the vote/negation
guard** (0.11154 on 391 against Scribe's 0.13771); the LLM chooser is a bench-only
flag until the deletion caveat closes. The code lives in `opencouncil-tasks` as a
stdlib-only Python package behind a conformance test, exposed as an audio-in
`openai-compatible` route so the bench can call it, then run in shadow beside Scribe.
Word timestamps come from Scribe only. Order: reproduce → package → bench run on a
RunPod CPU pod → shadow → flip.

Side-effect: the chooser code (`cats.py`, `run_chooser.py`, `reconstruct.py`,
`freeze.json` …) lived only in an ephemeral `/tmp` scratchpad and was **rescued** into
`eval/controlled_eval/chooser/`; data with transcript text went to
`~/.cache/oc-public/chooser-2026-08-25/`. Reproduction from the new location is step 1.

Blocked on Harold: where `RUNPOD_API_KEY` lives, ElevenLabs and Anthropic API keys.

## Closed 2026-08-26 — training WER for v2, and where the learning goes

`exp-2026-08-26-train-wer-v2` is **CLOSED**. The 300-pack sample frozen on
2026-08-23 was finally decoded: base 0.0824 against v2 0.0666, one CPU int8
4-thread stack, both models hash-checked, 300/300 rows on both arms, zero failures.

**The adapter learns its own labels, and the learning is entirely substitutions**
(0.0554 → 0.0393, CI [+0.0124, +0.0205]). Deletions move the wrong way inside noise.
On the 39 unseen validation windows the same two models show the mirror image:
substitutions unchanged, deletions halved. Comparing the decompositions only — never
the levels — the training gain and the validation gain are **disjoint**. The recipe
does teach new words; they do not transfer. What transfers is coverage.

That rules out "the recipe never taught it new words" and leaves transfer as the
open mechanism. It does not distinguish memorised labels from a skill too narrow to
generalise, and this measurement cannot.

Provenance side-effect: `/home/harold/oc-asr-serve/ct2-v2` had no record anywhere.
It is now `artifact-ct2-cleanpack-cont-s47`, proven by rebuild — cont_s47 merged into
whisper-large-v3 and converted with `--quantization int8_float16` reproduces model.bin
byte for byte. Its matched base is `artifact-ct2-base-large-v3-local`, **not** the
Systran conversion, which is now marked `MISSING` because its local snapshot is a
76-byte stub.

Report: [`docs/reports/2026-08-26-train-wer-v2.md`](docs/reports/2026-08-26-train-wer-v2.md).

## Where the fusion line stands, 2026-08-26

**0.10581 WER on the 391-window benchmark**, against Scribe v2 at 0.13771. Held out
on five untouched cities: **0.12431**. The policy is an algorithmic split of
disagreement islands into 21 categories, a deterministic rule in 19 of them and an
LLM chooser in 2.

**It looks immovable from here by text alone.** `exp-2026-08-26-three-composite-compose`
is CLOSED negative: the largest remaining prize, 0.0211 of WER in `THREE|COMPOSITE`,
did not yield to instructions, Codex revision loops, few-shot counts from 0 to 20,
context widths from 8 to 30 tokens, free composition, two structured edit formats, or
Opus instead of Sonnet. The frozen rule at 42.5% was never beaten reproducibly. The
model cannot see which word is spurious, because all three candidates are equally
plausible Greek and the question is acoustic.

The next thing with new information is **acoustic rescoring against the audio**, with
an independent CTC recogniser and a wrong-audio control. Before that,
`exp-2026-08-26-chooser-deletion-safety` is OPEN and should be settled: the policy
deletes 2.4x more than Scribe and the 36-clip vote-and-negation listening test exists
unrun.

## Closed 2026-08-26 — the per-category chooser

`exp-2026-08-26-category-chooser` is **CLOSED**. Disagreement islands are assigned
algorithmically to 21 categories; a deterministic rule runs where one works and an
LLM (Opus) where none does. Confirmed once on the five held-out cities, freeze
`6537f8e2451e7dbd`, second runs refused by the code.

End-to-end WER on rebuilt confirmation windows: **policy 0.12431** against R1+R2
0.12825, the vote 0.13798 and Scribe 0.15210. Both contrasts exclude zero and no
cluster carries more than 15.5% of 47.

The number that matters is the comparator: **+21.0 points against the vote, +3.0
against the best trivial rule.** The LLM only pays where every rule is a coin flip —
2 of 6 tested categories, 1,696 of 6,115 islands.

**Adopt nothing yet.** Deletions rose 76% over the vote (0.01434 → 0.02524, 2.4x
Scribe). `exp-2026-08-26-chooser-deletion-safety` is **OPEN** and holds that caveat:
a 36-clip listening test over the vote and negation deletions exists and has not
been run. Every number in the closed record is also ONE run of a nondeterministic
policy; three repeats were recommended before freezing and were not done.

[Report](docs/reports/2026-08-26-category-chooser.md).

## Closed 2026-08-23 — the held-out benchmark, both adapters

`exp-2026-08-23-post-june-held-out` is **CLOSED**. 391 held-out post-June windows,
117 meetings, both adapters served from one pod through one decoder stack with the
weights hash-checked before each arm.

The clean-pack contiguous adapter reaches 0.1827 against the incumbent's 0.1867 —
**−0.0040, 95% CI [−0.0078, +0.0002]**. The interval crosses zero, so the
pre-declared rule is not met and the adapter is **not promoted**. What it does buy
is a deletion rate of 0.0313 against 0.0525, CI [−0.0251, −0.0174]: roughly 40%
less of the meeting silently dropped, paid for in insertions and substitutions.

Caveat that travels with the WER number: the five largest windows carry 44.2% of
the net difference. The deletion rate does not share that weakness.

The GPU pod is terminated and the watchdog stood down.

## Active 2026-08-21

- `exp-2026-08-18-chunking-aware-decoding`: **arm P is now a served decoder and the
  contiguous-audio gap is closed.** `serve/decode_p.py`
  (`artifact-decode-policy-p`, policy `5ae98472227696e0`, capability
  `cap-decode-p-local`) reproduces the measured arm byte-for-byte on three pinned
  conformance windows, and a whole 12.69-minute meeting decodes end to end with no
  tiny pieces and no dropped speech.
  **P is the best measured arm by WER (0.14751 vs the control's 0.15893, CI
  [−0.02025, −0.00195]); it FAILED that experiment's preregistered gate, which was
  on deletions.** Three new whole-meeting limits — a forced-cut rate that runs from
  5% to 41% depending on the meeting, subtitle-credit hallucinations at the true
  start and end, and segment timestamps that overshoot their own piece at a third
  of the segments.
  Next: the 247-window GPU run on one stack, which is the last thing between P and
  "production policy".
  [Report](docs/reports/2026-08-21-decode-p-served.md),
  [runbook](docs/runbooks/decode-p-policy.md).

- `exp-2026-08-21-fusion-production` is **OPEN, spec written, nothing measured**.
  Production fusion with **two** ASR systems and no LLM:
  [`docs/specs/2026-08-21-fusion-production.md`](docs/specs/2026-08-21-fusion-production.md).
  The pair is `artifact-ct2-fixed` + Soniox `stt-async-v5`, named in the spec as the
  **cost-constrained** pair, not the accuracy-optimal one — Scribe v2 beats our
  adapter on WER, deletions and substitutions, and is excluded because no ElevenLabs
  credential exists here and because confidence must come from the same decode pass
  as the text (0 of 133 windows text-joinable). The merge rule is **one base
  transcript plus conservative patches over disagreement islands**, occupancy
  default **DROP** with a frozen restore gate, identity by phonetic closed-list
  repair then frozen priority; calibrated confidence is an arm, never an assumption.
  Two Codex reviews shaped it (`8dbcf232` rewrote the merge rule away from
  per-column tiers; `e2771f4c` supplied the LLM/confusion-network and extra-speaker
  future work).
  Next: the free Stage 0 screen on the cached 247 windows, and one cheap probe of
  what per-word fields `stt-async-v5` actually returns — every Soniox timestamp and
  confidence number this project holds is `stt-rt-v4` and the two must not be mixed.

## Active 2026-08-20

- `exp-2026-08-20-seam-repair` is **OPEN, mechanism pilot done, 39-window run not
  run**. Three arms on top of arm P — **F** (cut at the lowest-mean Silero
  probability valley in the last 2 s instead of cutting blind), **C** (64-*token*
  `initial_prompt` carried across the seam), **W** (overlapping windows with
  word-level ownership) — all **PASS** their mechanism checks on the 5 windows where
  P cuts blind. F turns 7 of P's forced cuts into 4 valley cuts and 3 remaining
  blind ones. Nothing is adopted: the 5-window rates are description, not evidence.
  The one structural result is that **word-level ownership, not word timestamps, is
  what arm E was missing** (E 0.33969 → E-WT 0.30338 → W 0.17538).
  A fourth arm **X** runs all three mechanisms in one decode: it posts the best WER
  of any arm (0.16062, (S+D)/N 0.14585) but **fails** its mechanism check — its 25.5 s
  ceiling manufactures blind boundaries faster than the valley search repairs them,
  so its win is not coming from the mechanism it advertises.
  Next: the 39-window decode, `eval/seam_repair.py decode --arm F|C|W|X` then `score`.
  [Spec](docs/specs/2026-08-20-seam-repair-prereg.md).

## Active 2026-08-19

- `exp-2026-08-19-dense-reference-repair` is **CLOSED, stopped in use**. Both page
  designs were tried and rejected by the reviewer: the two windows are continuous
  interruption and simultaneous speech, so every interval needed additions and the
  resulting reference would not be trustworthy. No repaired reference exists and the
  dense arm's insertion evidence stays unresolved. The finding is redirected: select
  training data by overlap-freedom instead of auditing references after the fact.
  [Spec](docs/specs/2026-08-19-dense-reference-repair.md).
- `exp-2026-08-19-training-residual-audit` is **CLOSED**. All 38 blind items were
  reviewed: 7/36 selected training clips were jointly label-faithful and
  boundary-usable, one was a definite boundary failure and 28 remain unresolved.
  Both insertion-heavy validation references have material omissions. Next:
  audio-faithfully re-reference those two windows, then freeze the hybrid clean-core
  plus protected lane. [Report](docs/reports/2026-08-19-training-listening-audit.md).
- The dense 300-step paired-seed screen is **CLOSED: `SCREEN — STOP`**. Dense B
  improved mean validation WER 15.31%→14.76% and deletions in all three seeds, but
  failed the insertion and dominance guards. One window explains 75.8% of net extra
  insertions, yet excluding it post hoc is forbidden and does not fully repair the
  gate. No medium/full stage is authorized. The recovery pod is terminated; network
  volume `qzw88vdwv2` remains retained.
  [Report](docs/reports/2026-08-19-dense-screen-300.md).
- The post-screen decisions are frozen in
  [`training-evidence.md`](docs/decisions/training-evidence.md#2026-08-19--agreed-decision-tree-for-the-remaining-training-work)
  and [`data.md`](docs/decisions/data.md#2026-08-19---strict-validation-and-hybrid-data-contract):
  boundary audit, strict audio-faithful validation, then one hybrid-data screen;
  no automatic GPU escalation.

Agent taking over the 18–23 August project/training push reads
[`2026-08-19-handoff-claude.md`](docs/runbooks/2026-08-19-handoff-claude.md) first.

**Η σειρά εκτέλεσης μέχρι τις 23/8 ζει πλέον στον χάρτη**
[«Το καλύτερο δυνατό μοντέλο + serving harness μέχρι 23/8»](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/3)
(GitHub Issues, label `wayfinder:map`). Το queue παρακάτω μένει ως περιγραφή
κατάστασης· ο χάρτης είναι αυτός που λέει τι πιάνεται μετά και με ποια σειρά.
Το `exp-2026-08-14-hparl-probe` βγήκε **εκτός scope** για αυτόν τον κύκλο (ανοιχτό
νομικό ζήτημα CLARIN 1602)· το record μένει OPEN στο ledger για μετά το GSoC.

Δεύτερος χάρτης, με ορίζοντα **πέρα** από τις 23/8:
[«Αρχιτεκτονική μετά-το-ASR: fusion ως provider, διαφωνίες ως δεδομένα»](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/50).
Σχέδιο: [`docs/specs/2026-08-21-postasr-architecture.md`](docs/specs/2026-08-21-postasr-architecture.md),
από τον ακατέργαστο κατάλογο ιδεών
[`docs/specs/2026-08-21-postasr-architecture-braindump.md`](docs/specs/2026-08-21-postasr-architecture-braindump.md).
Μέρος του ανήκει στο προϊόν OpenCouncil και δηλώνεται `PRODUCT`.

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
   it (every paired CI includes zero). Both are single-seed screens without a
   matched paired-seed confirmation; the later control calibration cannot turn
   either into a result. `artifact-adapter-fixed` keeps the candidate slot;
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
   **2026-08-17, the mirror question is now answered for the systems we can run**
   (`exp-2026-08-17-insertion-fidelity`): **23.7% of the adapter's scored
   insertions and 40.8% of Soniox's sit on words the human has and the published
   text cannot be reading**, with a further 36.8% / 12.3% undecidable. Insertions
   both systems emit are 0.72–0.80 human-supported — the class the cut F3 family
   would have deleted is the worst possible class to delete. Same limits as its
   parent: description only, no ranking, no transport to the benchmark.
   [Report](docs/reports/2026-08-17-insertion-fidelity.md).
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
  moves WER by +0.0015 in a single-seed arm whose interval crosses zero and whose
  sign reverses when one window is removed.
- **Data scale** (`exp-2026-08-11-wer-levers-research`). The dominant residual error
  is homophone orthography the audio cannot decide; ~1300h buys ~0.5 points.

The 2026-08 evaluation freeze
([`manifest.json`](research/eval-freeze-2026-08/manifest.json)) is the substrate for
all three: 39 validation windows, 31 meetings, 11,911 reference tokens. Its 7
temporal holdout windows are **still sealed** — no arm ever passed a gate that would
have released them.

## Recently changed

- `exp-2026-08-17-insertion-fidelity` closed: **the insertion metric is partly
  charging us for being right, and most for the system that inserts most.** On the
  frozen gold set, 18 of 76 (23.7%) of `artifact-adapter-fixed`'s scored insertions
  and 53 of 130 (40.8%) of Soniox `stt-rt-v4`'s are matched to a certain gold
  occurrence the published text fails to match under *every* minimum-cost
  alignment; 36.8% / 12.3% are undecidable, 31.6% / 28.5% unsupported, and the
  residual 7.9% / 14.6% are supported but on occurrences the published text does
  match — duplications, not omissions. As
  rates, 0.0199 of the adapter's 0.0840 and 0.0586 of Soniox's 0.1436 are
  reference-omission-consistent. **Insertions echoed by the other system are
  0.72–0.80 gold-supported at every window size** — the F3 family the composer
  draft cut would have deleted exactly those. The direct overlap test, on a gold
  denominator, recapitulates the pipeline-loss finding: gold occurrences whose
  block touches simultaneous speech go unmatched by the published text at
  25/74 = 0.338 against 140/925 = 0.151 elsewhere. **No transport to the 247-window
  benchmark**: the levels do not carry, and the only permitted statement about it
  is that insertion headroom must not be equated with hallucination headroom. Two
  Codex reviews, one before the design was coded and one on the findings; the
  second found a rate-denominator error that had been flattering us.
  [`docs/reports/2026-08-17-insertion-fidelity.md`](docs/reports/2026-08-17-insertion-fidelity.md).
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
