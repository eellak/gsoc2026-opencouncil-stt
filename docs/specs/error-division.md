# Who fixes what: fine-tuning vs the LLM, by error type

Status: **report + experiment plan** (2026-06-23). Purpose: decide — with numbers —
which error types the Whisper fine-tune should *own*, which we leave to the LLM
fix-task, and therefore what the **error-type distribution of the fine-tuning data**
should look like. So we can tell the team what we focus on and why.

> **TL;DR (GR):** Δεν χρειάζεται το Whisper να μάθει τα πάντα. Κάποια λάθη (στίξη,
> κεφαλαία, τόνοι, αριθμοί/μορφή, γραμματική, ακριβής ορθογραφία ονομάτων) τα
> διορθώνει ήδη αξιόπιστα το LLM — άρα είναι **σπατάλη** να τα κυνηγάει και το
> fine-tuning. Το fine-tuning πρέπει να εστιάσει στα **ακουστικά** λάθη που το LLM
> **δεν μπορεί** να σώσει (δεν άκουσε τον ήχο): φωνητικές αντικαταστάσεις, ομόηχα,
> όρια λέξεων, και να φέρει τα ονόματα **φωνητικά κοντά**. Άρα η κατανομή των
> δεδομένων fine-tuning πρέπει να γέρνει προς αυτά. Το πείραμα παρακάτω το δείχνει
> με νούμερα (WER ανά κατηγορία, σε κάθε στάδιο του pipeline).

## The principle (from the literature)

The split is not arbitrary — it follows from *where the information lives*:

- The **acoustic model owns what needs the audio**: phonetic confusions, word
  boundaries, getting names phonetically close. An LLM downstream literally cannot
  recover these — by the time it runs, the sound is gone. Multiple papers show
  generic LLMs *miss* phonetic confusions and can even *degrade* already-good ASR
  by overcorrecting (Revisiting ASR Error Correction, arXiv 2405.15216).
- The **LLM owns what's recoverable from text + context**: punctuation,
  capitalization, Greek diacritics (τόνοι), number/date formatting, grammar and
  agreement, and the *exact* spelling of a known entity given a roster. These are
  exactly the things LLM/GEC post-correction does well, and most are *conventionally
  excluded from WER* anyway (Whisper's own normalizer strips punctuation/case —
  arXiv 2409.02449).

| Error type | Owner | Why |
|---|---|---|
| substitution_phonetic | **Fine-tune** | needs the audio; LLM can't hear it |
| homophone (η/ι/ει…) | **Fine-tune** | acoustic; context helps a little |
| word_boundary | **Fine-tune** | prosody/acoustics |
| person/place/org name | **Shared** | FT gets it phonetically close; LLM+roster lands exact spelling |
| acronym_abbreviation | **LLM** | needs a vocabulary list; volatile per city |
| number_date | **LLM** | formatting / inverse-text-normalization |
| punctuation_capitalization | **LLM** | post-processing; excluded from WER |
| accent_tonos | **LLM** | orthographic, recoverable from text |
| verb_inflection / noun_case / article | **LLM (GEC)** | grammar/context |
| semantic_rewrite / disfluency | neither (curation) | not an ASR target |

Two caveats the literature insists on, both of which we've already seen ourselves:
1. A **generic, zero-shot** LLM on already-good text can *hurt* — the safe post-step
   is a corrector grounded on our real errors + an entity list, with an abstain/copy
   guard. (Matches our glossary result below.)
2. Training **only** on corrections (a skewed distribution) causes forgetting and
   out-of-distribution regressions — keep a no-edit / general-speech backbone in the
   mix (arXiv 2412.15726). (Matches our auto-research result below.)

## What our own data already says

**The corpus is dominated by exactly two things** (training clips,
`data/asr/dataset_stats.json`): `substitution_phonetic` **46.6%** and
`punctuation_capitalization` **20.4%**; then a long tail (noun_case, verb_inflection,
homophone, word_boundary each ~5–7%, names/acronyms/numbers each ~2–5%). So ~1 in 5
human corrections is *punctuation/casing* — a category we're arguing Whisper
shouldn't chase at all.

**The LLM fix-task already covers the "leave to LLM" column — and overcorrects when
pushed past it.** From the fix-task A/B ([dynamic-vocabulary-and-entities](../reference/dynamic-vocabulary-and-entities.md)):
a glossary block helped **named_entity +9.9pp** but *hurt* `acronym` −6.5pp,
`number_date` −6.0pp, `accent/morphology` −5.9pp, net HIR **+1.9pp worse**. Reading:
the LLM is already handling the textual categories; shoving more at it backfires.
The one place it genuinely needs help is exact entity spelling — which is the
"shared" row.

**Fine-tuning helps ordinary speech a lot and the textual stuff is noise-limited.**
From the tiny-model auto-research ([finetune-research report](../../data/reports/finetune-research/report.md)):
fine-tuning dropped `val_reg` (normal speech) ~24 points reliably, while per-category
`val_corr` moved within noise (≤27 clips/category). Directionally, the **hardest**
categories for the acoustic model were morphology-heavy (`verb_inflection` ~0.81,
`word_boundary` ~0.76) — and morphology is a *grammar* problem we're routing to the
LLM anyway. Phonetic/homophone/punctuation sat near the mean.

Put together, the working hypothesis is: **bias the fine-tuning data toward acoustic
errors (phonetic, homophone, boundary, names-for-phonetic-closeness), down-weight
punctuation/caps/number/accent/grammar corrections, and keep a ~50% no-edit
backbone.** The experiment below is to prove it instead of asserting it.

## Grok live-search cross-check (2026-06-23)

Ran the two core claims through `grok-research`. Both came back well-supported,
with sharper numbers:

- **Targeting the FT data at specific error types works — measurably.** A
  contextual-biasing fine-tune (Whisper, ~670h Common Voice, focused on
  mis-transcribed rare words) reported **+45.6% rare-word WER** and **+60.8%
  OOV-word WER** across 11 datasets (arXiv 2502.11572). Direct support for
  oversampling phonetic/entity errors in our mix.
- **Training only on corrections degrades general speech — with a scary number.**
  Yoruba-only fine-tuning sent English WER from ~22% to **147%** (catastrophic
  forgetting); the fix is mixing in unchanged/general data (rehearsal, LoRA,
  balanced ratios). Confirms the ~50% no-edit backbone is not optional
  (medium.com/@ccibeekeoc42; arXiv 2412.15726).
- **Generic LLMs overcorrect already-good ASR**, especially in low-WER regimes —
  so the post-step should be uncertainty-aware: let the LLM rewrite only
  **low-confidence / N-best** spans, not everything (arXiv 2405.15216;
  isca-archive interspeech_2024/manhtienanh24). This is exactly the N-best
  confidence design in [finetuning-dryrun-plan §6](finetuning-dryrun-plan.md) —
  the two ideas reinforce each other.
- **Punctuation/casing are excluded from standard WER** (normalized away before
  scoring; HEWER is the variant that re-includes them) — so de-prioritizing them
  in the acoustic FT costs nothing on the headline metric (Apple "Humanizing WER";
  speechmatics; gladia.io/blog/what-is-wer).

Net: the routing table above is the mainstream position, and the lever we control
(error-type composition of the FT data + a no-edit backbone + an uncertainty-gated
LLM) is exactly what the literature recommends.

## The experiment (mini PC, whisper-base, CPU)

Goal: produce a single table that shows, **per error category, the WER at each
pipeline stage**, so we can see (a) how much the LLM removes on its own, (b) where
fine-tuning adds value the LLM can't, and (c) what's left for humans.

### Design A — the 2×2 pipeline matrix (the headline)

Run all four stages on the **same held-out val set** (orestiada + argos, enlarged
per the val-size fix), decoding params identical throughout:

| | no LLM | + LLM fix-task |
|---|---|---|
| **baseline ASR** | A | B |
| **fine-tuned ASR** | C | D |

Report **WER per error category** (raw + Greek-normalized + CER) for A/B/C/D, with
meeting-clustered bootstrap CIs. Then read it off:

- **How necessary is the LLM** = the A→B drop (and C→D). Big drop on a category ⇒
  the LLM is doing the work there ⇒ fine-tuning needn't.
- **Where fine-tuning earns its keep** = categories where C ≪ A **and** B is still
  high (LLM couldn't fix them) — i.e. acoustic errors. These are the FT targets.
- **Safe to skip in FT** = categories where B ≈ C ≈ D (LLM already flattens them).
  Spending FT capacity there is wasted; D is the same whether or not Whisper learned it.
- **The real residual cost** = D per category = what a human still has to fix after
  the *full* pipeline. That's the number the project actually wants to shrink, and
  it tells us where to push next.

The fix-task runs through the existing on-box harness (`eval/fix_call.py` /
`run_ab.py`) over each ASR output; the on-box `claude` CLI covers the LLM with no
API key.

### Design B — does focusing the FT distribution help? (the composition sweep)

Fine-tune whisper-base on the same budget but three different **error-type mixes**,
each + the no-edit backbone, and score with Design A's stage-C/D table:

1. `natural` — categories as they occur (phonetic-heavy, lots of punctuation).
2. `acoustic_focus` — oversample phonetic/homophone/boundary/name, **down-weight**
   punctuation/caps/accent/number/grammar.
3. `balanced` — flatten categories.

Success criterion (this is the claim to the team): **`acoustic_focus` matches or
beats `natural` on the post-LLM full-pipeline WER (stage D), while using little/no
FT capacity on the textual categories** — i.e. dropping the style errors from the FT
data doesn't hurt the final transcript, because the LLM covers them. If true, we
train Whisper on a deliberately acoustic-skewed diet and let the LLM finish.

### Guards (so the numbers are trustworthy)

- Enlarge `val_corr` first — the current 56-clip noise floor makes per-category
  ranking meaningless (see auto-research next steps).
- ≥3 seeds for the fine-tuned stage (C); report mean + min/max, not one run.
- Same decoding + same Greek normalization across all four stages; version the rules.
- Report **`n_clips` and `n_meetings`** next to every cell; flag any cell with
  `n_clips < 30` **or** `n_meetings < 5` as directional only.

### What it produces

- `data/reports/error-division/matrix_categorized.md` — the A/B/C/D per-category
  table over the 160 export corrections, with CIs and per-cell `n_clips`/`n_meetings`.
- `data/reports/error-division/overall_val_reg.md` — no-edit regression / overall
  WER sanity slice (kept separate; no error categories).
- A one-paragraph "what we focus on and why" for the team, backed by that table.

## Implementation plan (hardened by Codex review, 2026-06-23)

### Data reality (corrected)

The "~9.9k orestiada+argos pool" in the auto-research report is **meeting-JSON
corrected utterances** (`lastModifiedBy != null`) — large but **un-categorized**.
The `/api/export` gives only **160 categorized corrections** in the two val cities
(34 meetings, 27 `humanReview`). The per-category matrix **needs** `error_categories`,
so it is capped at those 160. Consequence: this is a **"directional per-category +
aggregate-significant"** result, not a broad per-category claim — every cell is
bounded by `n_clips`, `n_meetings`, and CI width.

### Step 1 — enlarge the val set (the blocker)

New script `eval/autoresearch/prepare_val.py` building a **separate** frozen
`data/asr/val_manifest.jsonl` (does **not** touch the train manifest), reusing the
decode-once-then-slice + clip-validate path from `prepare_asr.py`:

- `val_corr_cat` = all 160 export corrections in orestiada+argos, `error_categories`
  kept, reference = `final_after_text`. Download+decode the ~28 missing val meetings
  once. **Leakage check:** assert none of these clips/utterance-ids/meeting+span/
  text-hashes appear in the train manifest.
- `val_reg` = no-edit utts (`lastModifiedBy is null`) from the `humanReview` val
  meetings, target ~400, capped per meeting to spread across meetings, deduped vs
  corrections. **Separate slice — never mixed into the per-category denominators.**
- Defer a large uncategorized overall-WER val (meeting-JSON corrected utts) unless
  160 proves too noisy.

### Step 2 — Design A harness `eval/autoresearch/matrix.py`

- Factor the **decoding path** out of `experiment.evaluate()` so stages A/C return
  **raw per-item Whisper hypotheses** (row-stable), then score raw WER, normalized
  WER, normalized CER from the *same* raw hyp.
- Stage A: zero-shot whisper-base over `val_corr_cat`. Stage C: train the round-1
  composition keeper (`corr+backbone_1x`) deterministically, ≥3 seeds.
- Stage B/D: on-box fix-task (`fix_call`, numbered-batch like `run_ab.py`, verbatim
  fix-task-prompt-v2 system prompt) over `hyp_A` / `hyp_C`. **No leakage:** batch
  input carries **only stable IDs + the ASR hypothesis** — no reference, category,
  metadata, or original text. **Strict parser:** require exactly the same IDs back,
  no missing/duplicate/reordered lines; **fail closed**.
- Scoring: per category, **micro-averaged** WER (summed edit-distance / summed
  reference length), **meeting-clustered bootstrap** (resample meetings, not clips)
  for CIs, plus **paired clustered bootstrap** for the deltas `C−A`, `B−A`, `D−C`,
  `D−B`. Report A/B once; C/D as mean over seeds + min/max.

### Acceptance test (write first — TDD)

`eval/autoresearch/test_matrix.py`: a tiny frozen fixture (2 meetings, 2 categories,
~8 clips) with **mocked** Whisper hyps (A/C) and **mocked** `fix_call` outputs (B/D).
Run the full `matrix.py` path and assert: same item IDs across A/B/C/D; references
unchanged; per-category raw/norm WER + norm CER match hand-computed values; bootstrap
resamples by meeting; a missing/reordered LLM line **fails loudly**; the report
includes `n_clips`, `n_meetings`, and the low-n flags. This proves the mechanics
before any multi-hour CPU run.

## To decide / next

- [ ] Enlarge the val set (shared blocker with the auto-research round 2).
- [ ] Run Design A (2×2 matrix) — highest signal, do first.
- [ ] Run Design B (composition sweep) once the val set is big enough to rank.
- [ ] Fold the resulting routing into the dataset build's category weights and into
      the [error taxonomy](../reference/error-taxonomy.md) `asr_finetune` vs
      `llm_post_correction` buckets.

Related: [error taxonomy](../reference/error-taxonomy.md),
[dynamic vocabulary & entities](../reference/dynamic-vocabulary-and-entities.md),
[auto-research report](../../data/reports/finetune-research/report.md),
[finetuning dry-run plan](finetuning-dryrun-plan.md).
