# LLM composer — plan, revision 2 after review

Status: **revised draft, exploratory only.** Reviewed independently by Fable and by
Codex (sol, high effort) on 2026-08-17. **Both cut it to one family.** Still not
preregistered. **The §0 contamination audit ran on 2026-08-17 and failed: F1 gets no
confirmatory CI and spends no confirmation.** Read §0 before anything below.

Substrate: 247-window fixed benchmark, `run_id
2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`, trio `scribe-v2-clean` /
`soniox` / `oc-runpod-fixed-2026-08-10`. Baseline **W = 0.10046**. References are
agreement-with-OpenCouncil, never fidelity-to-audio.

Revision 1 proposed four families (F1 majority override, F2 restore singleton,
F3 drop two-agree insertion, F4 tie control). **F2 and F3 are cut. F4 is not rerun.**

## 0. Blocking audit — RUN 2026-08-17, and it FAILED

Codex raised a failure that would void the confirmatory claim entirely: this plan was
designed by reading benchmark-wide oracle disagreements, error taxonomies and bucket
sizes. **If any of those calculations included the confirmation partition, those
confirmations are already spent** — the labels influenced which family was selected,
and splitting the data now does not restore blindness.

They did. The audit is
[`docs/reports/2026-08-17-confirmation-audit.md`](../reports/2026-08-17-confirmation-audit.md),
ledger record `exp-2026-08-17-confirmation-audit`.

| question | answer |
|---|---|
| Did the oracle counts and the majority-error taxonomy exclude confirmation meetings? | **No — they included them.** `column_census.py:38`, `exp_majority_taxonomy.py:365`, `exp_char_homophone.py:169` and `exp_composition.py:295` all call the bare `fusion_lab.load_substrate()`, whose only filter (`fusion_lab.py:128-138`) is the 6 sealed eval-freeze windows. All four result JSONs read `n_windows 247 / n_cities 10 / ref_tokens 74917`. |
| Did earlier experiments expose confirmation outcomes? | **Yes, as a habit** — LOO over all 10 cities in composition-over-selection, "9 of 10 cities negative" in name-repair-on-w, "5 cities better / 4 worse" in char-vote-homophones. |
| Did roster and term-list construction use confirmation references? | **Partly.** v1 base lists are externally sourced, but 38 error-mined candidates route to confirm cities and one v2 stoplist rule reads confirm reference frequency. Contaminates entity numbers on confirm, optimistically. |
| Is the split by meeting, not by window? | **Yes — by city**, strictly coarser than meeting (`autoresearch.py:102-103`, `:203`). The one clean answer. |
| Has the confirmation partition been read through the harness? | **No.** Journal: 16 registered / 16 searched, all at 153 windows, zero `CONFIRM_BATCH_FROZEN`, zero `CONFIRM_RESULT`. True of the harness, not of analyst knowledge. |

**Verdict (b): CONFIRM is invalid as confirmation for F1.** Not universally poisoned —
unusable for *this* hypothesis, because F1's eligibility class, its abstention success
condition and the elimination of F2/F3 are each traceable to specific
reference-conditioned numbers computed with confirmation labels in the denominator.
The project already made this exact call once, for
`exp-2026-08-16-overlap-speaker-arms`.

**(c) is refused.** All 247 windows have been through at least six reference-conditioned
passes, so carving from them restores no information independence; the only genuinely
unread material is the 7 sealed eval-freeze temporal-holdout windows at 2,101 reference
tokens, where the −0.0010 ship floor is **2.1 edits** against 75 on the full substrate —
underpowered by ~36× and sealed by a rule no arm has passed a gate to release.

**Therefore, binding on everything below:**

- F1 may be built and run. **No confirmation batch is frozen; no confirmation is spent.**
  The budget stays 5 of 5 and the single available batch stays available.
- Every F1 number is **exploratory**. Intervals are labelled descriptive — never
  confirmatory, never gate-valid, never multiplicity-controlled. F1 may not be described
  as having passed the ship gate whatever it measures.
- Any entity or name number carries the term-list leak caveat in the same sentence.
- If F1 runs, run it as a prospective-design exercise: freeze composer, prompt,
  thresholds, abstention policy and analysis before evaluating; ablate the leaky term
  resources out; report abstentions, failures, net edits and per-city heterogeneity; use
  the output for effect-size and power planning against a future sealed set.

Related, and settled by the same audit: Fable's reading is **correct** — the harness
permits **one confirmation batch, ever** per `PROTOCOL_VERSION`
(`freeze_confirmation_batch` raises if any batch record exists,
`autoresearch.py:819-825`), holding at most 5 ideas (`CONFIRM_BUDGET = 5`, `:112`). The
harness report's "budget of 5" means five ideas inside that one batch and says so
explicitly at `:67-73`; the five-sequential-batches / 22.6% scenario is what the code
forbids, so there is no conflict. Moot for F1 either way: the one shot is not being
taken, and it would have landed on a harder partition (W 0.11354 there against 0.09280
on search) where a failure is ambiguous between overfitting and city heterogeneity.

## 1. Why F3 was cut — the plan's disqualifying flaw

F3 asked a text-only model to delete words that **two independent systems both heard**.

- Two systems agreeing is close to the strongest acoustic evidence this pipeline has.
  The taxonomy's symmetric figure: when a majority is wrong it is a pure insertion
  only **9.2%** of the time. Correlated hallucination of the *same* word by two
  systems is rare.
- Our own reference omits speech. On the human gold set, **80.3%** of
  published-not-in-adapter words were confirmed really said, and **5.8%** of speech
  blocks have no published utterance at all. So much of the 1,532-column "insertion
  headroom" is the editor's cleaning, not ASR error.
- Therefore F3 would be rewarded for **reproducing editorial omission** and the result
  would be reported as WER improvement. The number goes down while the transcript gets
  worse.
- **No gate catches this**, because both rate gates are scored against the same
  omitting reference. Revision 1 said "if F3 fails its deletion gate it fails". The
  dangerous outcome is F3 **passing** while deleting real speech.
- The prompt was internally incoherent: the anti-fluency instruction "prefer the
  reading that carries more of what was said" contradicts the family's only available
  action.

Codex adds the estimand argument: F3 is coherent only as *"predict whether the
published text omitted this word"*, which is reference-style prediction, not an
acoustic claim — and asking "was this said?" while scoring against an editorial
reference mixes the two estimands the project keeps separate.

**What survives:** a descriptive gold-set study of what the insertion headroom actually
contains. Sample F3-eligible columns, check against human ground truth, measure the
fraction that is really-said speech. Costs no confirmation, and Fable judges it
possibly more publishable than any WER delta available in six days.

## 2. Why F2 was cut

- "Was this said?" is an acoustic question. Neighbouring timestamps give a
  duration-gap signal at best: *something* may be there, never *this word*.
- Base rates: 715 wrong drops among 4,460 singletons, so **84% of singletons are
  correctly dropped**. The harness's `occupancy_restore_singleton` already measured
  what indiscriminate restoration costs: **+2.85 points, insertions doubled**.
- Decisive: **even the oracle replay over occupancy columns fails the frozen insertion
  gate** (0.0374 → 0.0391). A model strictly worse than the oracle would be sent
  against a gate the oracle itself fails.
- The reference-omission problem cuts here too, in reverse: a correct restoration of a
  really-said word the editor omitted is **scored as an insertion**. The metric
  punishes F2 for being right.
- Soniox's low deletion rate is a prior, not evidence for an individual decision, and
  routing a prior through an LLM obscures a rule that could be tested directly.

## 3. F1 — the only family that runs

### 3.1 Eligibility must be reference-blind

Revision 1 repeatedly spoke of "the 1,245 wrong majorities". **That set is only
knowable from the reference.** At inference time F1 operates on **every observably
eligible `exact_2_of_3` column** — all 6,645 — and any narrowing rule must be
computable without the reference. Otherwise the experiment is directly label-leaked.

### 3.2 Success looks like abstention, not picking

Three text-only override rules already came back significantly **negative** (+0.16 to
+0.27 points, all six search cities worse). The taxonomy says **39.6–41.8%** of wrong
majorities carry zero or negative marginal value. So the success condition is not
"picks well" but "**abstains on roughly 40% of its own eligible set**".

Preregister the abstention rate as a gate-level diagnostic. A model that overrides
frequently is the shortest-picker in new clothes, whatever the order randomisation.

### 3.3 Controls that actually bite

Both reviewers agree order randomisation is **cosmetic**: the measured bias is
*content* preference (shorter, smoother), and shuffling positions does nothing to a
model that recognises the fluent token wherever it sits. It remains necessary hygiene
because the "in doubt pick 0" anchor was real, but it is not the countermeasure.

What replaces it:

- **Both candidate orders, every question.** Accept only if the choice is
  order-invariant; otherwise **abstain**. This diagnoses position sensitivity instead
  of assuming it away.
- **Masked slot.** Show the sentence with the decision position blanked. Showing the
  already-composed W text makes its current token look grammatically inevitable.
- **Abstain is an explicit modal option**, not a fallback. Invalid or inconsistent
  output becomes an abstention and is counted; it must never silently fall back to W
  while being counted as a valid model decision.
- Only existing token candidates. **No epsilon, no generated text.**

### 3.4 One of everything

One eligibility rule, one model and reasoning setting, one prompt, one context
construction, one abstention rule, one confirmatory hypothesis. Variants may be
developed on search data, but **only one frozen variant reaches confirmation**.

Model: `gpt-5.6-luna` at the highest effort the bridge allows —
`-c model=gpt-5.6-luna -c model_reasoning_effort=high` (the worker normalises `xhigh`
→ `high` outside the execute lane, and `high` routes to `sol` unless a model is passed
explicitly).

**Verify the unbatching hypothesis before committing to it.** Revision 1 asserted that
batching 24 questions caused the 6.8% invalid rate. That is untested. Check on a
100-question sample; small related-question batches may be fine, and one-decision-per-
call multiplies wall-clock by 24.

Cache keyed on **(question id × prompt hash × seed)**, or prompt iteration silently
reuses stale picks. Do not cache failures as no-ops — that bug already inflated an
invalid count once (`exp_composition.py`, caught by CodeRabbit).

Inherit from `exp_fusion.py`: neutral labels Α/Β/Γ so the model never sees provider
names, and interleaved execution so service drift cannot align with an arm. Inherit
`parse_json_array` from `exp_composition.py:200`. Inherit the reference-free gate from
`exp_postedit_gate.py:88-111` and its accounting framing ("analysis of the gate, never
an input to it").

## 4. The floor arithmetic that decides what can ship

Fable's arithmetic, and it settles the names question:

The ship test is H₀: ΔWER ≥ −0.0010, i.e. roughly **75 net edits** on 74,917 tokens.

- **Named entities: perfect play yields 41 edits.** The roster funnel caps one-token
  replacement at 28. **A name-targeted arm cannot clear the floor even if it never
  misses.** Names can therefore only be reported descriptively, never gated.
- **Function words: 276 edits**, the highest yield per column, and the only bucket that
  can carry a shippable effect.

But be honest about what winning there means. στη/στην, final-ν and particle choice are
largely **OpenCouncil house orthographic style**, which an LLM trained on written Greek
knows. A gain is real under agreement-with-OpenCouncil and must be framed as **style
normalisation**, never headlined as ASR improvement.

Codex dissents mildly and is right to: do not swap in a name-weighted metric *after*
observing that entities are 6.4%. Keep frozen WER primary; add named-entity error rate
and the two rate gates as **mandatory secondary**; report the bucket breakdown as
descriptive.

## 5. The ceilings in revision 1 were not inferential quantities

WER is globally aligned and non-additive: changing several column decisions can alter
the whole hypothesis–reference alignment. So `W ≠ oracle` column counts are diagnostics
only; summing them is not a reachable improvement, and the 0.67-point and 2.43-point
"ceilings" must not be used inferentially. Score every resulting full transcript
through the frozen WER implementation. Freeze MSA tie-breaking too — several equally
optimal alignments would change the column classes and hence eligibility.

## 6. Scope of any conclusion

One fixed trio, one benchmark, one realization. The conclusion is explicitly
conditional on that and cannot establish that the composer helps ASR systems generally.
Per-seed WER spread on training is 2.1 points, larger than the effect sought here.
This is the **seventh pass** over the same 247 windows; the search/confirm split
absorbs test multiplicity, not the adaptivity of having proposed families by reading a
reference-conditioned taxonomy. Say so.

Contamination note that must ride in the same sentence as any entity gain: the term
lists and rosters were mined from material overlapping this benchmark — the source
reports call it "optimistic and leaky".

## 7. Order of dropping, if six days are not enough

Wall-clock is the binding constraint, not confirmations. Drop in this order: F3
(already cut), F2 (already cut), all prompt and model variants, any name-weighted
optimization, a new F4 run. **The §0 audit failed, so the confirmatory claim is already
dropped** — F1 is exploratory only, whatever else survives the clock.

F4's existing number (W 0.10046 → W+L 0.10024) is quoted as the control. It is not
rerun: an improved F4 is a new arm, not a control.
