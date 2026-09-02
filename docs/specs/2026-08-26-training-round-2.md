# Training round 2: continued adapter vs retrain, with the ceiling measured first

Written 2026-08-26 after a 15-decision grill with Harold, then revised on the
Codex plan review `49fcd40d` (screen confound, N-best trigger, probe claim, leakage). This is the plan an agent
executes; every decision below was put to the user and settled. Nothing here is
measured yet. Ledger record: `exp-2026-08-26-training-round-2`.

Read first: [`CLAUDE.md`](../../CLAUDE.md), the evidence ladder in
[`training-evidence.md`](../decisions/training-evidence.md), and the direction in
[`2026-08-24-acoustic-ceiling.md`](2026-08-24-acoustic-ceiling.md).

## 1. The question

Two adapters exist and neither is a confirmed win over the other:

| artifact | recipe | best number | status of that number |
|---|---|---|---|
| `artifact-adapter-fixed` (HF tag v1) | 28,967 short clips, 7,242 steps | 0.1867 on the 391 held-out windows | incumbent |
| `artifact-adapter-cleanpack-cont-s47` (HF tag v2) | 2,476 single-speaker packs, 18.59 h, 619 steps | 0.1827 on the same 391, deletions 0.0313 vs 0.0525 | −0.0040, CI [−0.0078, +0.0002], **one seed**; the three seeds of this recipe spread 0.048 DS-WER; at 1,800 steps the gain dissolved (`exp-2026-08-22-clean-pack-medium`) |

Harold's question has three parts, and the grill settled each into something measurable:

1. **Can we keep the v2 adapter and train it further on new data, instead of
   retraining from scratch?** Technically yes, today: `notebooks/train_runpod.py`
   lines 538–553 load `INIT_ADAPTER` with `is_trainable=True` and continue training the
   *same* LoRA weights. No second LoRA on a merged model is needed. Whether it is
   *better* than retraining is not a matter of best practice; it is one screen
   (section 5).
2. **What new data is worth collecting?** Corrections of the kind already in the
   corpus do not transfer: `exp-2026-08-26-train-wer-v2` shows the training gain is
   entirely substitutions and the validation gain entirely deletions, and
   `exp-2026-08-13-correction-only` shows the correction rows alone buy nothing. New
   data must be different in kind (section 4).
3. **Before turning to decoding, how do we know the encoder cannot improve further?**
   Two measurements bound it, both already named in the acoustic-ceiling spec and
   neither run to completion: a candidate-budget sweep of the same-model N-best
   oracle, and a frozen-encoder probe. A third, phoneme-space rescoring, separates
   spelling from content at zero cost (section 3).

## 2. Decisions (the grill record)

| # | decision | choice | why |
|---|---|---|---|
| Q1 | primary endpoint | single-adapter WER on the frozen 39 validation windows, per the evidence ladder | the fusion line is a different system; its gain from a new adapter is reported as a secondary oracle number and decides nothing |
| Q2 | continued vs from-scratch | **both as arms in one 300-step screen**, 3 paired seeds | the question is a measurement, and the screen costs about $3 |
| Q3 | data source | Harold listens to targeted clips where our adapter disagrees with an independent witness; plus whatever new OpenCouncil UI corrections arrive; **never** from a meeting in the 391 benchmark, the 39 validation windows or the five confirmation cities | the old correction pool is already measured as useless on its own |
| Q4 | order | ceiling measurements (CPU, free) **before** any GPU training | the N-best sweep is also a precondition for every rescoring or DPO idea |
| Q5 | "encoder cannot improve" criterion | unfrozen-encoder arm beats frozen-encoder arm with a meeting-clustered 95% CI excluding zero **and** deletions not worse; otherwise the encoder is not the constraint | a point threshold would be moved after seeing the number |
| Q6 | what "RLHF for decoding" means | DPO on whisper's own decoder over its own N-best, **only if** the N-best sweep shows > 2 WER points between top-1 and oracle at some budget | with < 2 points there is nothing for a preference model to choose |
| Q7 | budget | **$50 GPU cap** for the round; no full training before screen and medium pass | ladder rule |
| Q8 | deletion gate | kept: deletion rate must not rise in any screen | gold set: 53/66 words another system has and we lack were really said; arm P already paid for this |
| Q9 | DPO reward | agreement with OpenCouncil text for training pairs; fidelity checked on the gold set only | the only reference that exists at scale |
| Q10 | listening supply | Harold, about 10 hours of work, target ~2 h of speech | below 1 h the screen cannot see past seed noise |
| Q11 | control | the existing cont_s13 / cont_s29 / cont_s47 adapters, untouched | zero cost; the delta then measures the new data only |
| Q12 | N-best sweep windows | the 39 validation windows and the 300 frozen training packs | known vs unknown data in one plot; the 391 only if headroom appears |
| Q13 | hours | as in Q10 | — |
| Q14 | replay ratio in the continued arm | 50 % old packs / 50 % new | new-only is the known failure mode of continued training and would show as deletions rising. Fun-ASR (§4.4.3) is external precedent for *retaining regular ASR data during post-training to mitigate forgetting*; their regular-ASR subset is one of five equal 20K subsets — 20 %, in a different training regime — so it does **not** validate 50/50 and does not transfer from GRPO to continued SFT |
| Q15 | where the plan lives | this spec + ledger record + `CURRENT.md` line | — |

## 3. Step 1 — the ceiling measurements (CPU, zero GPU cost)

Run all four before any training. Expected wall clock: 2–3 days of local CPU.
3.4 was added on 2026-08-27 from Fun-ASR (arXiv 2509.12508) and has its own ledger
record; it does not gate the others and can run in any order among them.

### 3.1 Phoneme-space WER on the 391 benchmark

Re-score reference, ours (`artifact-ct2-cleanpack-cont-s47`), Soniox `stt-async-v5`
and Scribe v2 after folding every token to its phonetic skeleton (ω/ο, η/ι/υ/ει/οι,
αι/ε, double consonants, final ς). `folded_pair`
(`eval/controlled_eval/exp_near_miss_vote.py:159-168`) is only a pairwise column
predicate; the fold itself is `eval/controlled_eval/greek_phonetics.phon:93-131` and
that is what a whole-transcript fold imports. Three fold implementations exist in the
repo and they disagree — `greek_phonetics.phon` is canonical (frozen tests),
`eval/categorize.py:_phonetic_skeleton` and
`eval/controlled_eval/classify_bench_errors.py:phon` are not. Choose STRICT or LOOSE
in writing before scoring: `Μαυρίδης`/`Μαβρίδης` collide under LOOSE and not under
STRICT. Score with the frozen scorer (`eval/controlled_eval/exp_same_stack.sdi`) and the
meeting-clustered bootstrap. The four transcripts are in
`~/.cache/oc-public/chooser-2026-08-25/` (see `eval/controlled_eval/chooser/reconstruct.py`).

Output: the 4.18-point gap to Scribe, before and after folding, with CIs.

Reading rule, frozen now: if the gap shrinks by less than 1 point, we lose **content**,
not spelling, and the decoder-only direction stays bounded at the 0.77 points the
homophone oracle already gave. If it shrinks by more than 2 points, that prediction was
wrong and decoder-side work moves up the queue. No g2p model is needed for this;
skeleton folding is the cheap version and is enough to answer the question.

### 3.2 N-best oracle as a function of candidate budget

**Already measured once, do not repeat that point.** `exp-2026-08-12-serving-stack`
arm D captured 8 beam hypotheses per window from `artifact-ct2-fixed` on the 39
validation windows through `scripts/serving_stack/nbest_arm.py` (`NUM_HYPS = 8`,
exact oracle DP, tested in `eval/tests/test_nbest_arm.py`). Oracle headroom over
top-1: **0.0071 WER**. A KenLM selector recovered 0 % of it.

What has not been measured is the *curve*. Reuse `nbest_arm.py` and sweep:

| budget | how |
|---|---|
| beam 8 (existing) | reproduce the 0.0071 as the anchor, now on **both** adapters |
| beam 16, 32 | `num_hypotheses` on the beam path only |
| sampling | 16 samples at temperature 0.4 and 0.8, in addition to the beam set |
| alternative chunking | the arm P policy (`serve/decode_p.py`) as a separate candidate set, merged with the frozen-config set |

Score the oracle at each budget on the 39 validation windows and on the 300 training
packs in `~/.cache/oc-public/train-wer-v2-2026-08-n300/` (both models already
decoded there). Plot oracle WER against candidate count; the two curves are the
result.

Headroom is defined as **top-1 WER − oracle WER** (positive when the oracle is
better). Reading rule, frozen now:

- headroom **< 2 points at every budget** → the right words are not being
  emitted; the limit is generation, not selection. DPO (section 7) is **not built**.
  This is the expected outcome given the 0.0071 anchor.
- **≥ 2 points at some budget** is necessary, not sufficient. Before DPO is
  authorised, all of: the meeting-clustered 95% CI of the headroom excludes 1 point
  (not a few windows supplying the two points); the oracle's gain is not mainly
  deletion or length selection (report S/D/I of oracle vs top-1); and a
  **cross-validated lightweight reranker** (features: log-prob, length, no-speech,
  compression ratio, KenLM score), trained on training-pack captures and tested on
  held-out meetings, recovers **≥ 25 %** of the headroom. A reference-informed
  oracle proves a better string exists; only the reranker shows it is findable.
  If the oracle has headroom and the reranker recovers nothing, the 2026-08-12
  result repeats and DPO is not motivated.

Captures from the 39 validation windows are **eval-only** and never supply
training pairs. Wider candidate sets also inflate the oracle by multiple
comparison and can contain reference-like hallucinations; the curve is read with
that in mind.

Either way the sweep also tells whether the training packs saturate at a different
place from the validation windows, which is the memorised-versus-narrow question the
train-WER report could not answer.

### 3.3 Base whisper-large-v3 on the same buckets

The acoustic-ceiling spec lists this as open. `artifact-ct2-base-large-v3-local` is
already decoded on the 300 training packs; decode it on the 39 validation windows
with the frozen config (about 40 minutes CPU) and run the same S/D/I and homophone
decomposition as `exp-2026-08-23-decoder-only-screen`. This bounds how much of each
bucket is ours to fix.

### 3.4 Retrieval-biased hotwords (CPU, zero GPU cost)

> **RAN 2026-08-27. Verdict `STOP`; the record is CLOSED.** 19 of 39 windows fired,
> mean prompt 11.1 tokens, +15 correct name mentions of 246 against a required 18,
> deletion-rate 95 % upper endpoint +0.0118. The section below is the plan as frozen
> before the run and is kept unedited; the result is in
> [`docs/reports/2026-08-27-retrieval-hotwords.md`](../reports/2026-08-27-retrieval-hotwords.md)
> and `exp-2026-08-27-retrieval-hotwords`.

A new question, and therefore a new record: `exp-2026-08-27-retrieval-hotwords`.
`exp-2026-07-25-hotwords` stays **CLOSED** and its conclusion — "do not ship
decode-time hotword biasing" — is not edited, reopened or contradicted here.

What the closed arm did: a **static**, hash-ordered, 160-token prompt of ~20 roster
surnames, applied unconditionally to every window
(`scripts/serving_stack/roster_hotwords.py:67`, `:227-236`). On the 39 frozen
validation windows it gave correctly matched name mentions **126 → 161 of 246**,
all-substitutions 938 → 856, and deletions **+133**, del_rate +0.0112
[+0.0020, +0.0207] with the CI excluding zero. Net WER +0.34 [−0.61, +1.31].

What this arm changes, adapted from Fun-ASR §5.5: the prompt becomes **conditional on
what the model already produced**.

- **Pass 1** is the cached frozen-config decode `~/.cache/oc-public/decode-ablation/eval-A.json`
  (`artifact-ct2-fixed`, model hash `8a1a3b257d0c1bdb`), reused byte-for-byte. It is
  not re-derived: `exp-2026-08-16-adapter-confidence` withdrew the bit-exact
  reproducibility claim for this local CPU decode (16 of 18, not 18 of 18).
- **Retrieval** folds pass-1 tokens with `greek_phonetics.phon` and retrieves lexicon
  entries within a phoneme edit distance computed by `eval/controlled_eval/scoring.edist`
  (Levenshtein over any sequence, so it works on skeletons as-is).
- **Pass 2** re-decodes the same window with the retrieved entries as the
  faster-whisper `hotwords` kwarg, everything else identical. Empty retrieval → pass 2
  is not run, and the pass-1 text stands as an exact zero-delta observation.

**Model: `artifact-ct2-fixed` (v1), not v2.** The +35 comparator is valid only on the
stack that produced it. Putting v2 in requires a contemporaneous static arm.

**Lexicon for the primary arm: the roster-present `person_surname` universe of the
closed arm** (`roster_hotwords.py:119-132`), *not* `roster_lexicon.py`. The broader
list carries error-mined terms whose mining fold overlaps the benchmark; using it
would change coverage and application at the same time. A broader-lexicon run is a
separate and explicitly exploratory arm. Note also that `data/pii/rosters_full.json`
is constant per city, so "per-meeting retrieval" is per-city retrieval filtered by
pass-1 content.

**The frozen retrieval parameters.** Written 2026-08-27, before any decode has run
and before any number exists (Codex review `a471616b` required these to be fixed in
writing rather than left as "typically 0-3 names"):

| parameter | value | why this one |
|---|---|---|
| fold | **STRICT** (`greek_phonetics.phon(loose=False)`) | corrected 2026-08-27, before any decode, after `plan` showed the lexicon must carry each surname's recorded inflections (see below). The first choice was LOOSE, argued from canonical-to-canonical distances; with the inflections in the lexicon that argument inverts. LOOSE merges β/υ, so a genuine misspelling of an inflected name (`Μαβρίδη` against the recorded alias `μαυριδη`) collapses to distance 0 and is skipped as already-correct — precisely the error class this arm exists to catch. STRICT keeps them one edit apart. With the fix the two folds fire on the same 19 of 39 windows and differ in one, so this is a structural choice and not a count |
| lexicon surface forms | canonical **plus the `aliases` of the DS-WER term file** | found by `plan` before any decode: matching the nominative alone makes every ordinary genitive (`Μαυρίδη` for `Μαυρίδης`) look like a one-edit error, and the prompt fired on 24 of 39 windows mostly for names the model had already written correctly. With the inflections in, it fires on 19 |
| `max_distance` | **1** | one edit in skeleton space. The recorded name-error profile is "wrong by one or two characters"; distance 1 is the conservative half of that, and the deletion gate is the binding constraint. Distance 2 is the **pre-declared secondary**, run only if the primary passes, and reported as such |
| `min_skeleton_len` | **4** | below 4 the skeletons of ordinary Greek function words collide — `και`→`ke`, `να`→`na`, `πω`→`po`, `ότι`→`oti` — and a lexicon entry that short would fire on every window |
| `k_max` | **3** | the spec's hard cap, enforced at config construction so it cannot be widened by a caller (CodeRabbit, 2026-08-27) |
| `token_budget` | **24** | three surnames fit comfortably; 160 is what the closed arm used and is the length this arm exists to get away from. Setting the budget an order of magnitude below it makes the short prompt structural rather than incidental |
| `include_exact` | **False** | a name the model already spells correctly needs no prompt, and prompting for it spends prev-context on every 30-second frame |
| matching | token-level, not contiguous n-gram | multi-token surnames are not in the primary lexicon; an n-gram pass would change the candidate inventory and belongs to the exploratory arm |
| tie-breaking | `(distance, normalised surface, surface)` | deterministic and independent of lexicon order, asserted by test |

Retrieval never receives the reference and cannot read scorer data. That is a
structural guard, not a convention: `retrieve` takes `(pass1_text, lexicon, cfg)` and
a test asserts its signature carries nothing else, because a reference-informed
retrieval would measure an oracle rather than a policy.

**The hypothesis, corrected.** An earlier draft of this section predicted the deletion
penalty would shrink in proportion to prompt length. That claim is **withdrawn**, for
two independent reasons found by separate reviewers. First, `hotwords` occupies the
`sot_prev` slot on *every* 30-second frame of a `transcribe()` call (faster-whisper
1.2.1, `transcribe.py:1200-1206`, `:1542-1550`), so a shorter prompt lowers the token
cost but does not remove the prefix from frames that contain no retrieved name; only a
piecewise decode such as the arm-P path could localise it. Second, R differs from the
static arm in relevance, length, candidate inventory and treatment frequency at once,
so no single cause is identified. What is left, and what this arm actually tests:

> Targeted retrieval retains at least half the closed arm's name benefit while passing
> deletion safety.

Identifying prompt length as the cause would need its own matched padded-prompt arm.
That arm is not built here.

**WER does not decide here.** The 80 %-power MDE for a hotword-shaped arm on these 39
windows is **1.37 points** (`exp-2026-08-16-harness-mde`), and the closed arm's whole
WER effect was +0.34 [−0.61, +1.31]. A gentler policy must produce a smaller effect
than one this harness already cannot resolve. The deciding statistics are the name and
deletion endpoints, where the closed arm's signal was large and CI-clean.

**Advance rule, frozen now.** The contrast is **R − A over all 39 windows**.
Restricting the headline to retrieved, changed or near-miss windows is forbidden:
those subsets are enriched for baseline errors and may be reported as descriptive
only. All of:

1. overall ΔWER point estimate < 0;
2. ΔWER 95 % upper endpoint < +0.0010 (the serving-stack non-inferiority margin);
3. Δdel_rate 95 % **upper endpoint ≤ 0** — a point estimate below zero with an
   interval crossing into positive **fails**;
4. Δinsertion < +0.0005;
5. at least **18 additional correctly matched name mentions** on the frozen
   246-mention denominator (half of the closed arm's +35);
6. name precision does not fall, and out-of-roster or wrong-person emissions do not
   rise;
7. no leave-one-window-out sign reversal **of the deciding statistics** (gate 5's
   name-mention gain and gate 3's deletion delta), not of ΔWER;
8. no window and no meeting supplies more than 25 % of the **name-mention gain** of
   gate 5.

All intervals meeting-clustered, 4,000 replicates, seed 7, blocks `(city, meeting_id)`.
Gate 5 is what stops an all-no-op retrieval policy from passing gate 3 trivially.

**Why gates 7 and 8 are anchored to the name gain and not to ΔWER.** ΔWER here is
near zero by design — that is the whole content of the MDE paragraph above — and
"25 % of a quantity indistinguishable from zero" is undefined, the same trap section 5
already names for its own gate 6. The name-mention gain is a discrete positive count
when the arm works, so domination is well posed on it. If gate 5 fails, gates 7 and 8
are reported `n/a`, not passed: there is no gain to dominate.

**Deletion verdict, two tiers.** Gate 3 is deliberately the strictest gate in this
spec, because the project's standing rule is that a model which lowers WER by omitting
hard passages is worse. It is kept as written for `ADVANCE`. But a policy that halves
the closed arm's penalty — say +0.005 [−0.002, +0.012] — would fail gate 3 while
carrying real information, and discarding that would waste the run. So the verdict
vocabulary for this arm is:

| verdict | requires |
|---|---|
| `ADVANCE` | all eight gates, gate 3 at upper endpoint ≤ 0 |
| `PARTIAL` | gates 1, 2, 4, 5, 6 pass; gate 3's **point estimate ≤ 0** and its upper endpoint ≤ +0.0028 (a quarter of the closed arm's +0.0112) |
| `STOP` | anything else |

`PARTIAL` authorises **nothing** — no shipping, no promotion, no further budget. It
licenses exactly one follow-up: the piecewise arm-P localisation named in the
withdrawn-hypothesis paragraph, which is the design the mechanism actually points at.
`PARTIAL` is not a weakening of `ADVANCE`; it is a named place to put a result that is
neither a pass nor nothing.

**Decoy control, as a separate pass.** R+D is the identical true retrieved set plus
one deterministically assigned decoy surname from another municipality, same audio,
same config, same seed. An event is a normalised decoy form present in R+D and absent
from R. Reported: exposures, incremental emissions, R+D − R S/D/I and WER, and whether
a true name was displaced. It is called **prompt-induced decoy emission**, not
hallucination — a councillor from another municipality can genuinely be mentioned, and
each emission needs audio adjudication before being called unsupported. With ~39
opportunities, even zero emissions is a weak upper bound: this is a stress test, not a
measurement. Decoys are **never** mixed into the primary R decode; that would change
prompt length and candidate competition and would score a different policy. Fun-ASR's
five irrelevant contexts (§4.3) are a contextual-SFT regulariser, not an
inference-time decoy evaluation, so the paper does not validate mixed-decoy scoring.

**Cost**: pass 1 is free (cached); pass 2 is ≤ 76 minutes CPU and less in proportion
to windows that retrieve nothing. Zero GPU. The model hash is verified before each
pass with the `eval/chunking_decode.py:155-169` check, which the closed arm never did.

Endpoints here are **agreement-with-OpenCouncil**. They can advance an engineering
screen; they cannot authorise shipping. Any advance requires fidelity adjudication of
the changed spans.

## 4. Step 2 — the new data (Harold, in parallel with step 1)

### 4.1 What the clips are

Short clips (5–15 s) where our adapter is wrong and an independent witness has
words we do not. Two pools, both already built:

1. **`deletion_hard`**: 3,921 rows / 5.4 h, of which **2,677 carry a Soniox witness**.
   The other 1,244 (31.7 %) have **no witness at all** and are not eligible — the same
   fact `CURRENT.md` already records. 1,260 rows are the earlier human-reviewed
   stratum; listen to the *unreviewed witnessed* remainder first. The witness for this
   pool is Soniox **`stt-rt-v4`**, the free real-time model invoked by
   `eval/gap2_verify_audio.py:159`, **not** `stt-async-v5`. They are different models
   (`eval/controlled_eval/w_rt.py:6-7`) and their outputs are never pooled.
2. **Adapter-vs-Soniox disagreement clips on training-side meetings.**
   `eval/controlled_eval/chooser/cutclips.py` **cannot be reused**: its rows come from
   `A.CONFIRM_CITIES`, every one of which §4.2 excludes, and its clip position is a
   column-fraction proxy rather than a timestamp (`cutclips.py:3-8`, `:45`). A new
   cutter is needed.

**One witness, and the Fun-ASR multi-witness rule does not apply here.** Fun-ASR
§4.4.3 mines hard cases where several external systems agree with each other and
differ from the model under test. That rule **cannot be computed on either pool**:
both carry exactly one external witness, and no second one exists on training-side
audio. The only local alternative is the whisper family, which is not independent —
base whisper shares **42.2 %** of the v2 adapter's long deletion runs, and on our long
runs it kept the span 53 times against Scribe's 102 and Soniox's 86
(`eval/results_deletion_runs.json`, `inheritance`). A base-versus-adapter agreement is
close to an autocorrelation. Codex review `a471616b` reaches the same conclusion from
the rule's side: base and v1 are one correlated signal, not two witnesses.

**So the rule is used for calibration instead, on windows that never supply a clip.**
137 windows of the 2026-08-10 benchmark run (5.58 h, 118 meetings) survive the §4.2
exclusions and carry five usable independent witnesses (`scribe-v2-clean`, `soniox`,
`gladia-prod`, `gpt-4o-transcribe`, `hf-openai-whisper-large-v3`); `pyannote-parakeet`
0.3567 and `greek-whisper-v3-turbo` 0.4823 fail the frozen ≤0.25 voter threshold and
are not witnesses. **USER DECISION 2026-08-27: these 137 windows are
calibration-only. No training clip is ever cut from them**, because training on
evaluation windows would contaminate the 2026-06-10 and 2026-08-10 runs and the
247-window fusion substrate, and that is not reversible. **AMENDED 2026-08-28: 38 of
them were spent** on the chalandri + vrilissia release described in §4.2b, so the
calibration runs on **99** windows and the substrate it may be read against is 211.
The amendment does not change the rule — the 99 that remain are still never cut from. What they buy is the
**pollution rate of single-witness mining**: of the spans a Soniox-only rule would
select, what fraction do the other witnesses attribute to the witness's error rather
than to ours. That number is reported with the listening batch and travels with every
row as a declared weakness. Zero cost, zero contamination, no new decode.

Read the calibration number against this: on the gold set our adapter and Soniox are
**both wrong 56 times against a meeting-stratified independence expectation of 21.0**
(odds ratio 17.8); 43 % of our errors and 65 % of Soniox's are shared
(`docs/reports/2026-08-16-adapter-confidence.md:175-183`). Witnesses agreeing with
each other is weaker evidence in this domain than Fun-ASR assumes.

If an ElevenLabs key appears, Scribe becomes a genuine second witness on training-side
audio and the mining rule in this section changes; the record must note the date.

**Listening is blind.** Harold transcribes the clip without seeing either
hypothesis; the witness text is revealed only afterwards, for disagreement coding.
Seeing Soniox first would anchor the label and turn Soniox into a teacher. The
listening page must enforce this order, not rely on discipline.
`eval/controlled_eval/chooser/listen_server.py:29-31` shows the judge the left/right
context **and** the deleted words, so it does not satisfy this; a new page is required.

The task is not to decide who is right. It is to write what was said. Label
convention: ~~the clean-transcript standard of 2026-08-12 (fillers and false starts
stripped)~~ — **superseded 2026-08-27 by §4.2c**. The frozen listening protocol of
2026-08-09 keeps intelligible false starts and repetitions, and it governs every blind
label this project has taken; only fillers are dropped. The raw label is collected under
that protocol and the cleaned target is derived separately.

### 4.2 Exclusions, enforced before a single clip is shown

No clip from a meeting that appears in any of:

- the 391 held-out windows — the meeting set is
  `~/.cache/oc-public/bench_2026-08-22-post-june-held-out-test-clean-pack-cont-.json`
  (`items[].cityId` / `items[].meetingId`, 117 meetings, 11 cities). **Not**
  `~/.cache/oc-public/bench_windows`: that directory holds the 260 wavs of the
  2026-06-10 / 2026-08-10 run and matches **0 of the 391**. An earlier draft of this
  line named it, and a builder written from it would have excluded the wrong benchmark
  and let the 391's meetings straight into training;
- the 39 frozen validation windows — `research/eval-freeze-2026-08/manifest.json`,
  `eval_windows[].city` / `.meeting_id` (32 meetings), built by
  `eval/controlled_eval/eval_freeze.py:84-98`;
- the five confirmation cities of `exp-2026-08-26-category-chooser` —
  `eval/controlled_eval/chooser/autoprompt.py:40`, city-level, no meeting;
- the gold set (`~/.cache/oc-public/gold-set-2026-08/manifest.json`,
  `scored_cell_ids` + `warmup_cell_ids`, 6 meetings), whose ids are packed as
  `gc_<city>_<meeting_id>_<offset_ms>` and must be parsed, not string-matched.

The four sources use three different id shapes, so the manifest builder normalises
them. The canonical key is **`(city, meeting)`**, never meeting alone: meeting ids
repeat across cities and collapse 117 real meetings into 66
(`eval/controlled_eval/exp_composition_postjune.py:212-216`).
`eval/exclusions.py` is a different mechanism but supplies the reusable
`is_excluded(city, meeting, keys)` helper.

The filter is a meeting-ID set difference and must be asserted in the clip builder,
not applied by hand. The exclusion manifest (canonical meeting IDs **and** source
audio hashes) is written and hashed **before** the first clip is cut, and its hash
goes in the ledger record; IDs alone miss renamed or duplicated meetings. Clips
overlapping in time with any existing training or evaluation clip are dropped. Each
row carries municipality, meeting, date, speaker (if known), source recording and
clip span, so repeated agenda language and cross-meeting audio reuse can be audited.
UI corrections with unknown meeting identity are quarantined, and repeated
corrections of the same span are deduplicated.

### 4.2b What the exclusions actually cost, measured 2026-08-27

The manifest was built, hashed and applied. The funnel, from
`eval/controlled_eval/cut_listening_clips.py plan`, in the order the filters run:

| stage | rows (2026-08-27) | rows (2026-08-28, after the release) |
|---|---:|---:|
| `deletion_hard` bucket | 3,921 | 3,921 |
| with a `VERIFIED` Soniox witness | 2,661 | 2,661 |
| outside every excluded meeting | 714 | 1,050 |
| within the 15 s ceiling | 676 | 1,007 |
| whose witness describes the row's **current** span | 605 | 899 |
| touching no other training row | 524 | 724 |
| after the same-meeting overlap rule | **523** | **721** |

The queue of record is the right-hand column: 721 clips, **2.16 h of clip audio** over 95
meetings in 4 cities, against the §4.3 target of about 2 h; 1.02 h if only the deletion
spans themselves are counted. The left-hand column is the 2026-08-27 queue, superseded
by the release described below and kept here because the two differ only in the
exclusion manifest.

Two of those filters were added after the plan was reviewed (Codex `4cf77e98`), because
the numbers behind them were measured rather than assumed:

- **108 rows have been edited since Soniox ran on them** (71 in the narrower
  2026-08-27 cut). `gap-verify/api/<row>.json` records the span that was actually
  verified; those 108 now differ from it, one by 13.25 s. Their `soniox_text` describes different audio and their cached wav is a
  different clip, so they are dropped rather than re-cut — a re-cut would produce a clip
  carrying somebody else's witness.
- **The overlap rule is applied literally**, which an earlier draft of this section
  argued was impossible. It is not: 724 of the 899 clips that reach that stage touch no
  other training row at all and only 175 touch one or more, because a meeting's training
  rows are a sample of it rather than a covering. Dropping those 175 also removes the
  supersession problem — no new human label can contradict an existing row's target,
  because no clip overlaps one.

**The 2026-08-27 queue came from two municipalities**, Chania 307 and Athens 216. The
pool had eight cities; `samothraki`, `sparta` and `xylokastro` are confirmation cities
and go city-wide, and `chalandri`, `vrilissia` and `zografou` were removed **entirely by
`calibration_260`** — every one of chalandri's 19 meetings is a meeting of the
2026-06-10 / 2026-08-10 benchmark run. That source alone blocked 1,305 of the 2,661
candidates, against 642 for the confirmation cities and 0 blocked only by
`held_out_391`, `validation_39` or `gold_set`.

Two municipalities is a confound the §5 gates do not test: a screen cannot separate "the
new data helps" from "the new data matches Chania and Athens".

### The release of 2026-08-28, and what it cost

Harold released **`chalandri` and `vrilissia` from the `calibration_260` source only**
(`CALIBRATION_RELEASED_CITIES` in `exclusion_manifest.py`; manifest rebuilt, sha256
`a45de4c6c670124d…`, the superseded file kept beside it). The four options were measured
on the real selection before the decision, not estimated:

| released | clips | h | cities | five-witness windows kept | fusion-substrate windows kept |
|---|---:|---:|---:|---:|---:|
| nothing | 523 | 1.56 | 2 | 137 / 137 | 247 / 247 |
| chalandri | 631 | 1.89 | 3 | 117 | 229 |
| **chalandri + vrilissia** | **721** | **2.16** | **4** | **99** | **211** |
| all three | 787 | 2.36 | 5 | 80 | 193 |

`zografou` was not released: it buys 66 clips for 19 calibration and 18 substrate
windows, the worst trade of the three.

There is **no free subset**. Every meeting that `calibration_260` blocks alone holds at
least one calibration window, so any widening spends evaluation material. What is spent
is of two kinds, and the second was found only after the first list of options had been
put to Harold:

- **38 of the 137 five-witness windows**, which is the material that measures the
  pollution rate of single-witness mining. 99 remain.
- **36 of the 247 windows of the `composition-rt-2026-08` fusion substrate**, which is
  the same 2026-08-10 run (253 windows common to all nine providers, less the 6 sealed).
  Published fusion, composition and chooser results are unaffected — they were measured
  before any adapter saw this audio — but a **round-3 adapter may no longer be scored on
  those 36**, and the substrate for that purpose is 211 windows.

The §5 screen itself is unaffected: it scores on the 39 frozen validation windows, which
no source releases.

The release drops keys from `calibration_260` only, so a released meeting that
`held_out_391`, `validation_39` or `gold_set` also names stays excluded through the
union; two of the 33 released meetings are in that position.

### 4.2c The label convention, and the conflict with the frozen protocol

§4.1 above says the listening label follows the clean-transcript standard of 2026-08-12,
fillers **and false starts** stripped. `docs/specs/listening-protocol.md`, frozen
2026-08-09 and governing every blind label this project has taken, says the opposite for
false starts: intelligible repetitions and false starts are written **as spoken**, and
only fillers are dropped.

Resolved in favour of the frozen document, because labels taken under different rules do
not belong to the same set and cannot be separated afterwards. `blind_listen.py`
collects the **raw** label under the frozen protocol — including its uncertainty and
exclusion states, its three-listen limit, and its five hidden repeat presentations for
the listener's own noise floor. Deriving the cleaned training target is a separate,
versioned step and is deliberately not done at collection time: a false start that was
never written cannot be recovered, while one that was written can always be stripped.

### 4.2d From raw label to training row — the contract, NOT YET WRITTEN

`blind_listen.py` writes `labels.json`: one raw answer per **presentation**, under the
frozen protocol. That file is not training data and must not be fed to §5 as if it were.
The transform between them is a separate, versioned step, and it does not exist yet.
Naming it here so that nobody writes it twice, differently, in a hurry:

| input | `~/.cache/oc-public/listen-2026-08/labels.json` + `current/queue.json` |
|---|---|
| output | one JSONL row per accepted clip, with `clip_id`, `text`, `source=listen-2026-08`, `witness`, `city`, `meeting_id`, `clip_start`, `clip_end`, `audio_sha256`, `transform_version` |

Rules the transform owes:

1. **Only `state == "transcribed"` rows become training rows.** `uncertain` and
   `excluded` are dropped, and their share per reason is reported — the exclusion rate
   is itself a finding, not a loss.
2. **Repeat presentations never become rows.** They are the consistency control; the two
   answers for a repeated clip go to the noise-floor calculation and the first hearing
   supplies the row.
3. **Cleaning is explicit and versioned.** The raw label keeps intelligible false starts
   and repetitions; the training convention of 2026-08-12 strips them. Whatever rule
   does that stripping is written down, applied once, and stamped into
   `transform_version`, so two batches cleaned differently are distinguishable
   afterwards. A false start that was never written cannot be recovered; one that was
   written can always be stripped, which is why collection keeps them.
4. **The noise floor travels with the batch.** The listener's disagreement with
   themselves over the five repeats is reported wherever these rows are used. A screen
   effect smaller than it is not an effect.
5. **The single-witness pollution rate travels too**, from the §4.1 calibration on the
   99 five-witness windows that survive the 2026-08-28 release, as a declared weakness
   of every row.

Until this exists, `artifact-listen-queue-2026-08` is a listening queue and nothing more.

### 4.3 Target and stop rule

Target about 2 h of speech, roughly 10 h of listening at 5x real time. Stop early
only if agreement between Harold's label and the witness drops below 60 % over any
200-clip run; that means the pool is wrong, not that Harold is.

Every new row carries `source=listen-2026-08`, the witness system, and the clip's
meeting ID, so the screen can report per-source contribution.

### 4.4 OpenCouncil UI corrections

Whatever new corrections arrive through the OpenCouncil pipeline on meetings outside
the exclusions join the same pool with `source=oc-ui`. They are not targeted and are
expected to contribute less; they are kept because they cost nothing.

## 5. Step 3 — the screen (GPU, about $3)

Three seeds, 13 / 29 / 47, paired. All arms use the clean-pack recipe from
`notebooks/train_runpod.py` as frozen for `artifact-adapter-cleanpack-cont-s47`
(r 32, lr 1e-4, effective batch 8, `pack_arm=pn`), one GPU float16
faster-whisper stack for scoring, the 39 validation windows.

| arm | start | data | steps |
|---|---|---|---|
| control | `~/.cache/oc-public/adapters-2026-08-22/cont_s{13,29,47}` (619 steps, the v2 recipe; s47 **is** v2) | — | none, scored as is |
| A continued | same three adapters via `INIT_ADAPTER` | 50 % old packs / 50 % new rows, packed the same way | +300 |
| A0 continued, old only | same three adapters via `INIT_ADAPTER` | old packs only, same sampler and minibatch order as A | +300 |
| B from scratch | base whisper-large-v3 | old packs + new rows, same 50/50 sampling | 919 (= 619 + 300) |

**A − A0 is the number that isolates the new data**; A − control alone would
mix 150 old-replay updates and 300 extra optimisation steps into it (Codex review
`49fcd40d`). A0 costs one more 300-step run per seed, about $1.50 in total.

A vs B compares initialisation, curriculum and exposure at once (B sees new data
throughout, A only after step 619; per update, A is roughly 769 old / 150 new and
B 460 / 460). It therefore selects an **operational recipe** — the question Harold
asked — and does not answer the causal question "is continuing better than
retraining". Codex proposed making B copy A's curriculum (619 old-only, then 300
mixed); that was declined because it collapses B into A with a fresh seed and
answers neither question. The record must state the reading as recipe selection.

Minibatch order is paired across arms as well as seeds. Three seeds are a screening
sample: uncertainty is aggregated over seeds and meetings together, and the 2-of-3
gate alone is weak against the known 0.048 seed spread.

Gates, all six from `training-evidence.md`, evaluated per arm against the control:

1. mean paired ΔWER < 0;
2. at least 2 of 3 seeds with ΔWER < 0;
3. mean deletion delta ≤ 0;
4. insertion delta < +0.0005;
5. no sign reversal leave-one-window-out;
6. no window above 25 % of the net gain — undefined when the net gain is near zero;
   if |net gain| is below the seed spread the gate is reported as `n/a`, not passed.

Verdict vocabulary is the ladder's: `SCREEN — ADVANCE`, `SCREEN — STOP`, never
"win". An arm that passes goes to the 1,800-step medium rung (about $7) only on a
new explicit decision by Harold; the clean-pack medium screen is the reason that
rung is not automatic.

Secondary, reported and not deciding: the per-source contribution (`listen-2026-08`
vs `oc-ui`) by ablation on one seed if budget allows; and the fusion-side oracle
gain on the 39 windows with the new adapter as the third system.

Pod rules: watchdog with hard deadline **before** upload, pod ID in the ledger
record, checkpoints by step, both adapters' hashes verified before scoring (the
training pod checklist in memory lists the eight ways this went wrong before).

## 6. Step 4 — the frozen-encoder probe (GPU, only if section 3.2 says generation)

Run only if the N-best sweep puts the limit in generation and the screen has not
already answered the question. Budget: whatever remains of the $50 after the screen
and, if taken, the medium rung. If that is under $20, do not start; record the
probe as unaffordable this round.

Design, preregistered here:

- **Arm F (frozen)**: LoRA on decoder q/v only, encoder frozen, clean-pack data,
  sweep steps 300 / 900 / 1,800 until validation WER stops moving by more than the
  seed spread. The plateau is the frozen-encoder ceiling for this data.
- **Arm U (unfrozen)**: same, plus encoder q/v LoRA, same steps, same seeds.
- Criterion (Q5): U beats F with a meeting-clustered 95% CI excluding zero **and**
  deletions not worse. Trainable parameters are matched approximately (F's decoder
  rank raised so the two arms have the same LoRA parameter count), so the contrast
  is placement, not capacity.
- What a result licenses, narrowed after Codex review: a positive U says "encoder
  q/v LoRA gives incremental benefit at this data and budget"; a null U says "it did
  not", which could be data, optimisation, rank or placement. **Neither is a
  ceiling.** A true representation test is a frozen-encoder phone/CTC probe
  trained on training meetings and scored on held-out ones (phoneme error and
  deletion recall); that is listed in section 10, not built here. A null U does
  not route the project to DPO; DPO is authorised only by section 3.2.
- "Stops moving by more than the seed spread" is a stopping heuristic for the
  sweep, not a convergence claim.

The whole probe is a screen at 300 steps first ($3), then a medium rung on decision.

## 7. Step 5 — DPO on whisper's decoder (only if section 3.2 says selection)

Not designed in detail until the trigger fires. Skeleton, so the data step keeps
what it needs.

External context, carrying no threshold: industrial-scale GRPO produced a modest
average improvement in Fun-ASR (Table 8, 13.18 → 12.70 average WER), qualitatively
consistent with keeping DPO behind our own preregistered N-best and reranker gates.
That number is an average over different proprietary datasets, a different model, a
different stack and a different training regime, reported without seeds or CIs; it
**cannot** be compared numerically with our 0.71-point eight-best oracle headroom, and
RL can change generation rather than only choose among the old eight hypotheses. The
paper says GRPO with a WER reward is "similar to" MWER, not equivalent, and its summed
scalar reward is not structurally the same object as our six gates, which are
non-compensatory and additionally enforce seeds, leave-one-out stability and
dominance.

- pairs come from the N-best capture: (chosen = the hypothesis with the lowest
  sdi against the reference, rejected = top-1) on windows where they differ;
- reward is agreement with OpenCouncil text (Q9); fidelity is checked once on the
  gold set, never used for training;
- the reference LoRA is the adapter under test; DPO trains a new LoRA on top of the
  merged weights, because the continued-training path in `train_runpod.py` is SFT
  and DPO needs its own loop — 2–4 days of code, no library ships it for whisper;
- gates are the same six; a DPO arm that lowers WER by deleting is a `STOP`.

Because of this, **keep every N-best capture** from section 3.2 on disk under
`~/.cache/oc-public/nbest-2026-08/`, keyed by model hash and budget.

## 8. What this plan does not do

- It does not promote either adapter. Neither is a confirmed win and this round
  does not try to settle v1 against v2; the control is v2's recipe because it is
  the one whose deletion advantage is inside its CI.
- It does not open the sealed test windows.
- It does not use Scribe as a teacher. Pseudo-labels from the fusion output were
  offered as an arm and rejected for this round (leakage toward the 391, and
  distillation of a system we are trying to measure against).
- It does not run any GPU step before the CPU measurements in section 3 are on
  disk with their reading rules applied.
- **It does not add a noise-augmentation arm.** Fun-ASR §5.2/§5.6 was screened on
  2026-08-27 and is OFF TARGET here, on three independent grounds. (1) The causal
  experiment already exists and is CLOSED negative: `exp-2026-08-03-synthetic-overlap`
  measured a real interjector at +5 dB at +0.0016 [−0.0049, +0.0076] against a
  preregistered 2.0-point gate, and envelope-matched noise at +5 dB at −0.0035
  [−0.0110, +0.0024]. (2) Deletions do concentrate in high-overlap windows
  (+0.0339 [+0.0167, +0.0502] for `artifact-adapter-fixed` over 232 windows, no
  single-item domination, surviving every confound tested), but substitutions
  (+0.0346) and insertions (+0.0344) rise at the same rate — deletions minus
  substitutions is −0.0007 [−0.0255, +0.0231]. Overlap marks hard windows, not
  deletions, and Fun-ASR's "early termination" mechanism predicts the opposite.
  (3) The mass is not there: mean overlap is 2.2 % of speech time, and a perfect
  overlap fix moves the corpus deletion rate 0.0440 → 0.0372, about 0.0068 WER.
  Two caveats travel with this closure: arm E of the 2026-08-03 experiment ran on base
  whisper-large-v3, not on v2 (Codex review `a471616b` would keep a $1.50 v2 screen
  open on that ground); and no per-window SNR artifact exists anywhere in the repo, so
  the *noise* predictor as distinct from *overlap* has not been measured directly. A
  per-window noise-floor proxy would cost 20–40 CPU-minutes and $0 if the question is
  reopened. Where our deletions actually live is already recorded in
  `exp-2026-08-24-deletion-runs`: whole published utterances and speaker turns, not
  noise.

## 9. Order of execution and cost

| step | who | cost | output |
|---|---|---|---|
| 3.1 phoneme fold | agent | 0, ~2 h CPU | gap before/after fold, CIs |
| 3.2 N-best sweep | agent | 0, ~1–2 days CPU | two oracle-vs-budget curves |
| 3.3 base on 39 | agent | 0, ~40 min CPU | base bucket decomposition |
| 3.4 retrieval hotwords | agent | 0, ≤ 76 min CPU | R−A on 8 frozen gates, plus the R+D decoy pass |
| 4 listening | Harold | ~10 h | ~2 h labelled speech, `source=listen-2026-08` |
| 5 screen | agent + pod | ~$3 | A vs B vs control, six gates, three seeds |
| 6 probe or 7 DPO | decided by 3.2 | ≤ remaining of $50 | ceiling location, or a selection screen |

Steps 3.x and 4 run in parallel. Step 5 waits for both. Step 6/7 waits for 5 and
for a new decision by Harold.

## 10. Open, not decided here

- A frozen-encoder phone/CTC probe as the real representation test (Codex review);
  not costed, not scheduled.

- Whether the `listen-2026-08` rows should also be re-listened by a second person
  for a small agreement estimate. Not required for the screen; required before any
  claim that the labels are fidelity-grade.
- Whether the 391-window oracle sweep is worth its day of CPU. Only if 3.2 shows
  headroom on the 39.
- Whether an ElevenLabs key will exist for the two-witness clip pool. If it does,
  the pool definition in 4.1 changes and the record must note the date.
