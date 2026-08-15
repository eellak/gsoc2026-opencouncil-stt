# Current Work

Last updated: 2026-08-14

**The endgame plan is finished.** All four workstreams of
[`docs/specs/2026-08-11-endgame-handoff-plan.md`](docs/specs/2026-08-11-endgame-handoff-plan.md)
are closed, 2026-08-12. Read
[`docs/reports/2026-08-20-final-report.md`](docs/reports/2026-08-20-final-report.md)
first — it is the answer to the project's question, with its limits.

What is left is the queue below: one decision (publish), one experiment (name
lexicon), one gap (fidelity-to-audio).

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
1. Publish `artifact-adapter-fixed` to HuggingFace. The benchmark now prices the
   delay: the published weights cost **1.77 WER points on unseen cities**. **Done
   when** the hub weights are the corrected ones.
1b. `exp-2026-08-13-targeted-deletion-training` — OPEN, **candidate supply
   unblocked 2026-08-13**: user audited 228 gap3 items (94.7% accept overall);
   the calibrated strict stratum (found_frac≥0.85 ∧ n_added≥5, 97.4% agreement)
   was bulk-accepted (324 items, username=auto-verifier) with user consent —
   ~540 verified deletion examples so far. Soniox verification of the remaining
   ~10.9k unreviewed deletion-shaped rows is running (expected +1,500–2,000).
   Recipe is frozen (Codex-reviewed 55/25/10/10, 3 seeds/arm, deletion gate +
   insertion guard); +78h trusted clean backbone available at zero human cost.
   Next: preregistration spec once the verification wave lands.
2. Name work continues as the **post-hoc roster repair** arm of
   `exp-2026-08-11-name-repair` (inside the ladder above). Decode-time hotword
   biasing is closed: it lifts name recall 51→65% but costs +0.34 WER in
   deletions and fails its preregistered gate.
3. Decide whether fidelity-to-audio changes this ranking. The benchmark measures
   agreement-with-OpenCouncil; the one time both were measured, the ranking flipped.
   **Done when** the corrected adapter has a fidelity-to-audio number on unseen
   meetings.

## Product decision: answered 2026-08-12 — transcripts are **clean**

Filled pauses («εεε») and false starts are **stripped**. Asked and answered once, per
the handoff plan. This is the standard future listening hours should be produced
against, so that they are compatible with each other.

Nothing frozen was changed on the strength of it: the benchmark normalizer, the
existing labels and the 2026-08 evaluation freeze all stay exactly as they were. The
decision governs new data collection, not any number already measured.

## Blockers

- Item 1 is blocked on nothing technical — it is a decision. The corrected weights
  exist locally and are now measured.
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
