# Draft: dataset split + publish plan (meeting prep)

Status: **DRAFT for discussion** — 2026-06-23. Grounded in the eval data on the
mini PC (`data/eval/`, `data/reports/fix-task-eval/`, `data/improve_loop/`) and
the Notion meeting notes (May 26 → Jun 16). Pressure-tested with Grok + Codex
(see end). Not yet a decision; the `[?]` items are for the meeting.

---

## 0. Framing — two artifacts, do not conflate them

The work splits into **two separate things** that have been getting mixed up:

1. **The fix-task** — an LLM post-processor that cleans raw ASR text (already in
   production, Sonnet). The on-box A/B harness measured *prompt levers* on this.
   Headline result: a retrieved-glossary block **does not help overall**
   (52.8% → 51.3% edit-application) and only helps `named_entity` (+9.9pp);
   prompt-tuning gave **no statistically significant** gain on held-out
   (HIR −2.0pp, McNemar p=0.44, CI crosses 0).
2. **The ASR fine-tuning dataset** — `(audio, corrected_text)` pairs to fine-tune
   the acoustic model (likely Whisper-v3-large). This is the HuggingFace **post**
   and the thing the train/eval/test **split** is for.

The meeting is about #2. #1 informs #2 in one way: it tells us *which residual
errors the LLM can already fix* (so they are lower-value as ASR targets) vs
*which are genuinely acoustic* (high-value ASR targets).

---

## 0b. Empirical findings (2026-06-23, measured on the full source CSV)

Computed from `data-1779206108158.csv` (397k edits, has `meeting_date`,
`utterance_start/end`, `audio_url`) joined with the public/private availability
report. **These change the plan — read before the meeting.**

- **There is NO data after 2026-06-01 yet.** Latest meetings are **May 2026**.
  → The Notion "test = data after June 1" set is currently **empty**; it can only
  exist as a **future/rolling** benchmark that accumulates. (Confirms Codex #1,
  but stronger: it's not thin, it's zero today.)
- **May-2026 slice** (temporal-ish dev-test fallback): 26 meetings, 20 public,
  ~1249 public corrected-minutes (~21h of *corrected* audio).
- **Volume is not the constraint.** Total ~**365h of corrected audio** across
  411 meetings / 22 cities (this is only the edited spans; total audio is more).
  A 20–30h fine-tuning target is trivially covered. The hard part is the **split
  + label quality**, not quantity.
- **The proposed val cities are mostly PRIVATE:**
  - `vrilissia`: 32 meetings but **only 10 public / 22 private**.
  - `zografou`: 54 meetings, **only 16 public / 38 private**.
  - `argos`: 33 meetings, **32 public** ✅ (fine as a public val anchor).
  - `chania`: **105 meetings, all public, ~93h** — the biggest city, not even in
    the Notion plan; natural train backbone (or a strata source).
  - Good **public** val candidates: argos (32), sparta (37), xylokastro (26),
    chalandri (21), orestiada (14), samothraki (10).
  → **Reconsider "Argos + Vrilissia → val":** Argos OK, **Vrilissia is a poor
    *published* val** (mostly private). Swap in a public-heavy city.
- **No `speaker_id` and no per-speaker audio in the CSV.** Speaker-disjoint
  splitting and per-speaker data floors **require the meeting JSON** (utterances
  carry speaker + timing there). Blocker for the Notion speaker-level split until
  we fetch it.
- **Residual-WER on no-edit cannot be auto-computed** — the dataset is
  edits-only (no no-edit rows) and has no independent gold. Needs a **human audit
  / stronger-model re-transcription** of a sampled no-edit set from the meeting JSON.

---

## 0c. Speaker + audio facts (from the meeting JSON, 2026-06-23)

Fetched per-utterance speaker + duration for all **327 public meetings** via
`GET /api/cities/{city}/meetings/{meeting}` (`eval/fetch_speakers.py` →
`data/eval/speakers.parquet`, 571,124 utterances, ~378h). Real numbers we did
not have before:

- **Composition by `lastModifiedBy` (raw, all public meetings):** no-edit
  329k utts / ~168h / 44.3%; `user` 143k / ~122h / 32.1%; `task` 99k / ~89h / 23.6%.
- **⚠️ But the `humanReview` gate changes this materially** (see §0d): only **212
  of 327** public meetings are `taskStatus.humanReview=true`. Restricting to those:
  trustworthy no-edit backbone = **~109h** (not 168h), human-verified = ~93h,
  task-final = ~52h; the other **~124h (31% of utts) are un-reviewed → exclude**.
  Even the 109h needs a residual-WER audit before trusting as ground truth.
- **90.5% of utterances have an identified `person_id`** → speaker-disjoint
  splitting is feasible (only ~2,226 min unidentified).
- **0 speakers appear in more than one city** (of 544 identified). → **A split by
  city is automatically speaker-disjoint across cities** (no cross-city leakage).
  Within a city, speakers recur across meetings (so unseen-speaker-within-seen-
  city needs a speaker-level split inside that city).
- **Per-city real audio (public):** chania ~115h (42 spk), athens ~75h (122 spk),
  sparta ~29h, chalandri ~25h, zografou ~20h, samothraki ~19h, vrilissia ~18h
  (only 10 public meetings — rest private), xylokastro ~18h, orestiada ~17h,
  argos ~17h, then a long tail of 1-meeting cities (<7h each).
- **Speaker floors:** 254 speakers ≥10min, 141 ≥30min, 87 ≥60min.

### Concrete split proposal (data-backed, public-only)
- **Unseen-city stratum (val + test):** hold out a few mid-size public cities
  *entirely* — e.g. `orestiada` (~17h), `samothraki` (~19h), `argos` (~17h).
  Automatically speaker+meeting disjoint; tests city/acoustic generalization.
  (Replaces "Vrilissia" — which is mostly private — with public cities.)
- **Unseen-speaker / seen-city stratum (val):** inside the big cities
  (`chania`, `athens`, `sparta`), hold out ~30% of **speakers** (with a ≥10min
  floor) → tests speaker generalization with the acoustic domain seen.
- **Train backbone:** `chania` + `athens` remainder + the rest, with the no-edit
  corpus as the bulk and human-verified corrections upweighted.
- **Future-temporal test:** rolling, **currently empty** (no post-Jun-1 data);
  freeze the policy, accumulate.

---

## 0d. The `humanReview` gate (Discord question → general rule, 2026-06-23)

Question: `thessaloniki/apr1_2026` has >10 human-reviewed utterances but isn't
fully corrected — exclude it? **Yes**, and there's a clean general signal:
`taskStatus.humanReview`. In a meeting nobody finished reviewing, a `no-edit`
utterance means "nobody looked", not "ASR was right" — so its label is untrusted.

- **Rule:** include a meeting in the trusted dataset / HIR denominator **only if
  `humanReview=true`**. (`thessaloniki/apr1_2026`: `humanReview=false`, 37 stray
  edits / 9,282 utts → excluded.)
- Of 327 public meetings: **212 reviewed, 115 not** (124h / 31% of utts dropped).
- This gate also makes the **HIR metric** well-defined — see
  [metric-hir.md](../decisions/metric-hir.md). Baseline **HIR = 28.1%** (FPY 71.9%).
- Per-meeting coverage is now browsable in the UI: **`/stats/coverage`** (reads
  `data/reports/coverage.json`).

---

## 1. The split — proposal

### Test set (held-out)
> **Policy (reconciled with reviewer note #1):** the temporal test is a *rolling*
> benchmark that keeps accumulating until it meets the adequacy minimums, then is
> **frozen** and never touched again. "Never touched until the end" applies only
> *after* it is frozen.

- **Temporal holdout: all data after 2026-06-01.** (Notion Jun 16.) This forces
  the test set to contain new meetings + new speakers → measures real
  generalization, not memorization.
- `[?]` **Feasibility check needed.** Today is 2026-06-23, so only ~3 weeks of
  post-Jun-1 meetings exist. We must confirm there is *enough* post-Jun-1
  **public** audio to make WER statistically meaningful. If thin, fall back to a
  hybrid: temporal where possible + a speaker-disjoint slice.

### Train vs Validation (no leakage)
- **Speaker- and meeting-disjoint:** every meeting AND every speaker lives
  exclusively in train OR val. (Notion Jun 16.) This is the correct standard for
  ASR — a random per-utterance split leaks speaker acoustics and inflates scores.
- Concrete starting proposal (Notion):
  - Argos + Vrilissia (all their speakers) → validation.
  - From the remaining cities, ~30% of speakers → validation; ~70% → train.
  - Enforce a **minimum minutes-per-speaker** floor in each side so both have
    enough acoustic coverage.
- `[?]` **Wrinkle we just found:** many **Vrilissia and Argos meetings are
  private** (84/411 meetings are private; Vrilissia/Argos heavily represented).
  Private audio probably can't go in a *published* val set. Either (a) pick val
  cities that are mostly public, or (b) keep a private val internally but publish
  only the public subset. Decide before fixing the val cities.

### One canonical split, two consumers
- `[?]` The existing `data/eval/split.json` is the **fix-task eval** split
  (per-meeting + per-city, eval cities: argithea, athens, kalamata, sparta,
  zografou) — built to keep the glossary leakage-safe. It is **not** the ASR
  split. Proposal (precise form per reviewer note #7): define **one canonical
  OUTER partition** (`meeting_id`/`speaker → train|val|test`) and have both the
  ASR fine-tuning and the fix-task eval consume it, each applying its own
  *eligibility filters* on top. So a meeting is never ASR-train + fix-task-eval at
  once, but the two eval *datasets* need not be identical. This is the single
  contract; "canonical split" everywhere in this doc means this shared partition.

---

## 2. Dataset composition (what rows go into ASR training)

Source: 287,605 correction *chains*, 22 cities, 242 meetings (`dataset_stats.json`).
Chain types: pure_correction 235k, mixed 37k, semantic_rewrite 15k.
Provenance (1000-row sample, `by_ebclass.md`): ~half are `task_only` (LLM did the
edit, no human), the rest involve a human (`task_then_user`, `user_only`).

Proposed include/exclude for the **ASR target labels**:

- **✅ No-edit utterances = the bulk / backbone.** Raw ASR the human left
  unchanged = clean, abundant, representative `(audio, text)`. Prevents
  catastrophic forgetting and overfitting to hard cases. This should be the
  majority of the corpus.
- **✅ Human-verified finals** (`user_only` + `task_then_user`) — high-value,
  genuinely-missed errors. Candidate for mild **upweighting**, not for being the
  whole training set.
- **❌ `task_only`** (LLM-edited, no human sign-off) — labels are unvalidated and
  we measured the task **overcorrects**. Training the acoustic model on these
  risks baking in the LLM's mistakes. **Proposed default: exclude from the trusted
  target set.** Reviewer note #5 refines this to a *weak/down-weighted tier*
  (possibly only NE/acronym `task_only`); **ablation A is what settles exclude vs
  weak-keep** — until then, exclude is the working assumption, not a final decision.
- `[?]` **Acoustic-vs-style filter.** A "user edit" is not always an ASR error.
  For ASR targets we want *acoustic* errors. Use the categories as triage:
  `homophone / word_boundary / named_entity / acronym` ≈ acoustic (keep);
  pure `morph_grammar / style / punctuation` ≈ not necessarily acoustic (filter
  or down-weight). Categories are a text-only heuristic, so this is triage not truth.
- `[?]` Target size (Notion May 26): ~25k utterances / ~30h audio. Revisit now
  that we have real counts — the no-edit backbone could make it much larger.

### Validate, don't guess — the ablation
Everything above is testable. Run an ablation: fine-tune **with vs without
`task_only`**, and **with vs without** the acoustic-only filter, measure WER on
the held-out test. If "without task_only" doesn't hurt (or helps), the exclusion
is confirmed. (This is the decision note + ablation sketch that was requested in
the paused mini-PC session and not yet written — fold it into `docs/decisions/data.md`.)

---

## 3. The post (HuggingFace dataset publication)

- **Card contents:** size/hours, #cities/#meetings/#speakers, split methodology
  (temporal test + speaker-disjoint train/val), category distribution, provenance
  (no-edit vs human-verified vs excluded task-only), and the honest caveats.
- `[?]` **License + privacy.** Council meetings are public proceedings, but:
  speaker names = PII; **84 private meetings must be excluded** from publication;
  confirm OpenCouncil's license for derived audio+transcripts before pushing.
- `[?]` **Audio packaging:** clips per utterance vs full-meeting audio + offsets.
  Per-utterance clips are simplest for a published ASR set.
- **Reproducibility:** publish the split file (meeting/speaker → train/val/test)
  and the build script so the split is auditable.

---

## 4. Open questions still unresolved (open since June 2)

- `[?]` **Audio normalization.** Do we normalize volume for the training set?
  Per-meeting or per-interval? Must match (or deliberately differ from)
  production. Untouched since Jun 2.
- `[?]` **"Specific WER"** for toponyms / named entities / legal terms — how do
  we measure it separately? (Most useful metric per Notion is the
  human-interaction rate, HIR, which we already compute.)
- `[?]` Enough data per speaker after the disjoint split? Needs the per-speaker
  minutes histogram before fixing val speakers.

---

## 5. What is already clarified (don't re-litigate)

- [x] Primary metric = WER (+ HIR as the human-facing metric).
- [x] Test = temporal holdout (post 2026-06-01).
- [x] Train/val = speaker+meeting disjoint, not per-utterance.
- [x] Don't fix timestamps (~10% of errors → discard those).
- [x] Likely fine-tune Whisper-v3-large.
- [x] Glossary is **not** a general win for the fix-task (only `named_entity`).

---

## 6. Concrete deliverables before / at the meeting

1. [ ] Confirm post-Jun-1 **public** data volume → decide test feasibility.
2. [ ] Per-speaker minutes histogram → pick val speakers with the data floor.
3. [ ] Decide val cities given the private-meeting constraint.
4. [ ] Produce the canonical split CSV (meeting + speaker → train/val/test).
5. [ ] Write `docs/decisions/data.md` entry: include/exclude rule + ablation plan.
6. [ ] Draft the HF card skeleton + resolve license/privacy.

---

## Reviewer notes (Grok best-practices + Codex review, 2026-06-23)

### Grok — confirms direction, adds method
- Speaker-disjoint + temporal test is the standard; both, not either.
- Composition ~**70–80% clean / 20–30% corrected**; oversample hard cases only
  mildly.
- **Exclude LLM-only labels** as gold (use at most as weak supervision) — confirmed.
- Anti-forgetting toolkit: **LoRA (r=16–32) over full fine-tune**, LR ~1e-5 +
  warmup + early-stop on val WER; **rehearsal** with ~5–10% general Greek
  (Common Voice / FLEURS / **HParl**) mixed in; **SpecAugment / speed-perturb /
  reverberation** augmentation; optional staged/curriculum (easy→hard).
- Greek benchmarks to cite/compare: **HParl, Mosel (EP Greek), GPC**.

### Codex — sharper critique (raises the bar; fold these in)
1. **Temporal test is a *rolling* benchmark, not yet a frozen test.** 3 weeks is
   too thin + can drift (mic/pipeline/UI/LLM-version changes) and gets inspected
   during dev → leaks. Action: freeze the *policy* now, keep accumulating until
   **minimum adequacy** met (≥20–30 meetings, several cities, enough NE/acronym
   events, no single meeting >X%). Use a separate **dev-test** (older,
   speaker+meeting-disjoint) for development. Three explicit sets:
   `dev-validation` / `dev-test` / frozen `future-temporal`.
2. **Don't dump whole cities into val.** Argos/Vrilissia-as-val confounds
   unseen-speaker vs unseen-meeting vs unseen-city. Use **explicit strata**:
   `seen-city/unseen-speaker`, `unseen-city`, `future-temporal`.
3. **Public-only canonical benchmark.** Private data = a separate *internal
   stress test*, **never** used for model selection (else published results
   aren't reproducible). Resolves the private-Vrilissia/Argos wrinkle cleanly.
4. **"No-edit" is NOT automatically clean ground truth** — it can mean nobody
   reviewed / error missed / tolerated / editor stopped early. **Measure residual
   WER on a random no-edit sample BEFORE** treating it as the backbone. (Sharpest
   single correction to our plan.)
5. **Three label tiers, don't discard `task_only`:** (1) human-verified finals =
   trusted; (2) reviewed no-edit = backbone; (3) `task_only` = weak supervision,
   flagged + downweighted (maybe keep only NE/acronym `task_only` where the
   fix-task showed value).
6. **Category filter is risky** — Greek morphology is often a *genuine* acoustic
   error, and categories are noisy. Filter on **"is the final text acoustically
   supported"** (audited per-category samples), not on category label alone.
7. **One canonical OUTER partition; per-consumer eligibility filters.** ASR needs
   audio+aligned+acoustically-valid target; fix-task eval needs hyp+edit-chain.
   Don't force the two eval datasets to be identical — just keep the
   meeting/speaker partition shared so nothing is train-here/test-there.
8. **Leakage checklist:** speaker identity across cities + spelling variants;
   near-duplicate audio; templated agenda/intro text; **glossary/lexicon built
   from train only**; **recording-local** audio normalization (never global
   stats from test); tokenizer/vocab never adapted on test; **cluster stats by
   meeting** (utterances in a meeting are not independent).
9. **Named-entity metric:** define denominator + matching rules *in advance*;
   report entity error rate (S/D/I), performance on **unseen** entities, and
   micro+macro by meeting. Aggregate WER hides sparse-NE regressions.

### Two ablations to fold into `docs/decisions/data.md`
- **A — Label-source:** same split/budget. (A) no-edit only; (B) A + human-verified;
  (C) B + downweighted `task_only`; (D) B + NE/acronym `task_only` only.
  Success: **B beats A on correction/entity metrics without natural-WER
  regression**; C/D must beat B with meeting-clustered CIs.
- **B — Generalization matrix:** evaluate the chosen model on seen-city/unseen-
  speaker, unseen-city, future-temporal, and the public-only subset. Require gains
  **consistent across meetings** (macro-by-meeting deltas + clustered bootstrap),
  not driven by one city.

### Biggest risk (Codex) — keep front of mind
> Treating correction provenance and "no-edit" status as reliable labels when
> they are **workflow artifacts** → an easy-looking aggregate gain that actually
> reflects selection bias / templated text / city composition, not better ASR.

**Simplest defensible result to aim for:**
> *Human-verified, acoustically-supported corrections improve targeted Greek ASR
> errors over a reviewed no-edit backbone, under meeting- and speaker-disjoint
> evaluation.*

Keep `task_only`, private-data effects, and the short temporal benchmark as
**secondary analyses** until their validity is demonstrated.
