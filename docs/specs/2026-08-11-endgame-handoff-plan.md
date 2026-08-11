# Endgame handoff plan — last 12 days

> **For agentic workers:** if the superpowers plugin is installed, use
> superpowers:subagent-driven-development or superpowers:executing-plans to run this
> task-by-task. If it is not installed, execute the tasks in order yourself — the
> plugin is optional, the order and the gates are not. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Close the GSoC deliverable with four workstreams that need no new full
retrain: (1) decode-threshold ablation against deletions, (2) the DS-WER metric
promised in the proposal, (3) one correction-only dataset ablation, (4) the final
report with limits declared.

**Reviewed by Codex 2026-08-11 (job 70c12950), corrections incorporated.**

**Read first (mandatory):** `CLAUDE.md`, `CURRENT.md`, ledger records
`exp-2026-08-11-error-analysis`, `exp-2026-08-11-name-repair`,
`exp-2026-08-10-benchmark-fixed-adapter`, and
`docs/reports/2026-08-11-training-brief.md`.

**Hard rules that apply to every task here** (from `CLAUDE.md`):

- Transcript text and audio never go in git. Caches live under `~/.cache/oc-public/`.
- Freeze decode configs, term lists, manifests, and gates **before** seeing a number.
- Same machine, same decoder stack, same normalization for any A/B. CPU int8 and
  CUDA int8 produce different tokens — never mix devices inside one comparison.
- Watch the deletion rate; a lower WER bought by deletions is a regression.
- GPU pods bill from creation: arm a watchdog with a hard deadline before uploading
  anything, record the pod ID.
- Finish protocol per task: update the ledger record in the same change, run
  `python3 scripts/check-research-state.py`. Never edit `research/ledger.json` or
  `CURRENT.md` from two parallel tasks — serialize those edits.
- If a step fails, run a cheap smoke test and retry only after recording a distinct
  root cause. Two identical failures = stop and write down what broke.

**Experiment IDs are fixed in this plan** (no placeholders): `exp-2026-08-12-ds-wer`,
`exp-2026-08-12-decode-ablation`, `exp-2026-08-13-correction-only`,
`exp-2026-08-20-final-report`. If a record with that ID already exists in the ledger
when you start, stop and reconcile — do not mint a variant ID.

---

## Task 0 — Day-0 freeze (before any decoding or scoring)

**Files:** Create `research/eval-freeze-2026-08/manifest.json`; create
`docs/specs/2026-08-12-decode-ablation-prereg.md`.

- [ ] Write the frozen evaluation manifest: the exact 39 clean validation window IDs
  (argos + orestiada minus the 7 temporal-test windows enumerated in the caveats of
  `exp-2026-08-10-benchmark-fixed-adapter` — the caveat names one,
  `win_argos_jun3_2026_855222`; recover the full list of 7 from the benchmark run
  data per `docs/runbooks/benchmark-api.md`, they are the windows with meeting date
  ≥ 2026-06-01). For each window record: window ID, city, meeting ID, reference
  token count. The meeting ID is the resampling block.
- [ ] Record in the same manifest: `faster-whisper` and `ctranslate2` exact installed
  versions, Python version, `artifact-ct2-fixed` content hash (from its ledger
  record — verify against the file on disk), device and compute type to be used.
- [ ] Keep the 7 temporal-test window IDs in a separate `holdout` list in the
  manifest. They are touched **exactly once**, in Task 1.4, and never tuned on.
- [ ] Commit the manifest and the prereg (contents specified in Task 1.1) before any
  arm is decoded.

---

## Workstream 1 — Decode-threshold ablation (`exp-2026-08-12-decode-ablation`)

**Why:** `serve/oc-asr/oc_asr_server.py:158-160` passes only `beam_size=5` and
`condition_on_previous_text=False`. Everything else is faster-whisper defaults:
`no_speech_threshold=0.6`, `log_prob_threshold=-1.0`,
`compression_ratio_threshold=2.4`, temperature fallback `(0.0→1.0)` active,
`vad_filter=False`. Our deletion:insertion ratio is 3.3:1 vs Scribe's 1.6:1
(`exp-2026-08-11-error-analysis`).

**faster-whisper semantics you must respect** (verified against upstream by review):

- The no-speech skip fires only when `no_speech_prob > no_speech_threshold` **AND**
  `avg_logprob < log_prob_threshold`. Setting `log_prob_threshold=None` therefore
  ALSO makes the no-speech gate inert — it is a compound change.
- `compression_ratio_threshold` and `log_prob_threshold` trigger a retry at the next
  fallback temperature; they do not themselves discard output.
- Positive fallback temperatures **sample randomly**: call
  `ctranslate2.set_random_seed(seed)` with a stable seed derived from
  `(arm, window_id)` before each window, or results are irreproducible.
- Use `WhisperModel.transcribe` (the class the server uses), NOT
  `BatchedInferencePipeline` — the batched pipeline ignores these thresholds.
- Consume the segment generator fully before recording any result.

### Task 1.1: Preregister (goes in `docs/specs/2026-08-12-decode-ablation-prereg.md`, Task 0)

- [ ] Arms, one behavioural change each, everything else pinned to the control:
  - **A** control: exact current server config (decode fresh — do NOT reuse the
    2026-08-10 benchmark hypotheses; package/device differences change tokens).
  - **B** `no_speech_threshold=0.8` — anti-deletion family.
  - **C** `no_speech_threshold=None` — anti-deletion family, **primary**.
  - **D** `temperature=[0.0]` (no fallback) — anti-insertion family, **primary**.
  - **E** `log_prob_threshold=None` — labelled "compound: disables low-logprob
    fallback AND makes the no-speech gate inert". Exploratory.
  - **F** `compression_ratio_threshold=None` — exploratory.
- [ ] Every arm sets ALL transcribe options explicitly (language=el, beam_size=5,
  condition_on_previous_text=False, word_timestamps=False, vad_filter=False, plus
  the four thresholds and temperature sequence). Save the resolved
  `TranscriptionOptions` object per arm into the results, not just the arguments.
- [ ] Estimands, defined before running:
  - Primary: **micro-WER** = sum(S+D+I)/sum(ref tokens) over the 39 windows,
    wer-nofillers normalization (import the benchmark normalizer, do not copy it).
  - **Deletion rate** = sum(D)/sum(ref tokens); **insertion rate** analogous.
  - Delta = arm − control (negative is good).
- [ ] Uncertainty: paired block bootstrap, resampling **meetings** (not windows),
  ≥2000 replicates, seeded. If the 39 windows span too few meetings for a stable
  bootstrap (<8 meetings), report the CI anyway and label it descriptive.
- [ ] Gates:
  - B/C ship-candidate if: deletion-rate delta 95% upper bound < 0 AND micro-WER
    delta upper bound ≤ 0 (non-inferiority margin 0.0 — WER must not rise).
  - D/E/F ship-candidate if: insertion-rate delta upper bound < 0 AND deletion-rate
    and micro-WER non-inferior (upper bound ≤ 0).
  - C and D are the primary arms; B/E/F are exploratory and cannot ship alone.
- [ ] Influence check: leave-one-window-out on the pooled delta; flag domination if
  removing one window reverses the primary conclusion. (This replaces any
  "%-of-delta" rule.)
- [ ] Per-window diagnostics to record: which thresholds fired, which temperatures
  were attempted, segments skipped. A null result must distinguish "knob
  ineffective" from "condition never occurred".

### Task 1.2: Implement the decode script

**Files:** Create `notebooks/decode_ablation.py`; smoke test first.

- [ ] Locate the concrete decode+scoring code behind `artifact-bench-hyps-fw-finetune`
  (its ledger record lists provenance) and reuse its audio-loading and normalization
  imports. Name the exact functions you reuse in a comment.
- [ ] Smoke test: decode ONE short window under every arm on the chosen device;
  verify diagnostics land in the output JSON. Fix before scaling.
- [ ] Decode the 39 manifest windows only (not all 240 — avoidable cost), all six
  arms, one device for everything. CPU mini-PC is acceptable offline (no proxy
  timeout); an RTX A4000 pod per `docs/runbooks/runpod-training-pod.md` is faster —
  either way ALL arms on that one device.
- [ ] Hypotheses and per-window scores go under `~/.cache/oc-public/decode-ablation/`,
  aggregates only in the repo.

### Task 1.3: Evaluate against the prereg

- [ ] Apply the gates exactly as preregistered. No post-hoc arm mixing: if two knobs
  both look good, combining them is a NEW confirmation on the holdout, not a free
  merge.
- [ ] Write `docs/reports/2026-08-12-decode-ablation.md` with per-arm micro-WER,
  deletion/insertion rates, CIs, fire-counts, influence check.

### Task 1.4: One-shot holdout confirmation, then ship or record the negative

- [ ] If a primary arm passed: decode ONLY the winning arm + control on the 7
  temporal-test holdout windows, once. It ships only if the deltas keep their sign.
  No tuning after this — if it fails, record the failure, do not iterate.
- [ ] To ship: make the thresholds env-configurable in
  `serve/oc-asr/oc_asr_server.py` `model.transcribe(...)` following the existing
  `OC_ASR_BEAM` pattern (parse `"none"` → `None` explicitly), redeploy per
  `docs/runbooks/self-hosted-asr-endpoint.md`, hit `/health` and one real
  transcription as regression check, and note the served-config change in the
  ledger record and in `cap-self-hosted-asr`.
- [ ] Either way: create+close `exp-2026-08-12-decode-ablation` in the ledger, and
  update `exp-2026-08-11-name-repair` to note its decoder arm is now answered here
  (the name-lexicon arm keeps its own gate in
  `docs/specs/2026-08-11-name-repair-plan.md` and is NOT part of this plan).
- [ ] Run `python3 scripts/check-research-state.py`. Commit.

---

## Workstream 2 — DS-WER (`exp-2026-08-12-ds-wer`, closes proposal Milestone 2)

**Definition source:** `docs/reference/gsoc-proposal.md:100` — Levenshtein restricted
to domain-critical words (council member names, local acronyms). Milestone 2 target:
**≥15% relative improvement over the Gladia baseline**, defined as
`(DSWER_gladia − DSWER_ours) / DSWER_gladia` on point estimates, with the CI
reported alongside.

**Honesty note for the report:** the term lists are being specified after the error
analysis existed, so this is a *retrospectively specified* metric evaluation, and the
reference is OpenCouncil's published text (agreement-with-OpenCouncil, not
fidelity-to-audio). Both statements go in the report verbatim.

### Task 2.1: Freeze the term lists

**Files:** Create `research/ds_wer/terms/<city>.json`, one per benchmark city.

- [ ] Sources: meeting rosters/agendas (the roll-call windows are named in
  `docs/reports/2026-08-11-error-analysis-vs-scribe.md`) — NOT any provider
  hypothesis text.
- [ ] Schema per file: `{city, source: {url_or_doc, retrieved}, version, terms:
  [{id, canonical, aliases: [...]}]}` — UTF-8, NFC. Aliases enumerate accepted
  surface forms explicitly (nominative/genitive of surnames, spaced/undotted
  acronym forms). **No stemming.** Case-insensitive, tonos/dialytika-insensitive
  matching only if the imported benchmark normalizer already does that — the term
  matcher uses the SAME normalizer as scoring.
- [ ] Rule stated in each file: which classes are included (elected members'
  surnames, place names, local acronyms) and excluded (generic civic vocabulary).
  Duplicate/collision rule: an alias may map to exactly one term ID; collisions are
  resolved before freezing or the alias is dropped.
- [ ] **Never add an alias after seeing a provider hypothesis.** Commit the lists
  (names of elected officials in public meetings are public record; the lists
  contain terms only, no transcript text). Only after this commit may scoring run.

### Task 2.2: Implement the metric (TDD)

**Files:** Create `scripts/ds_wer.py`; test `tests/test_ds_wer.py`.

- [ ] Normative algorithm (implement exactly this):
  1. Normalize reference, hypothesis, and terms with the imported benchmark
     normalizer.
  2. Replace multiword/overlapping term matches with atomic term IDs,
     longest-match-leftmost, deterministic.
  3. Unit-cost Levenshtein alignment over the full transformed sequences,
     documented tie-break (prefer match > substitution > deletion > insertion).
  4. Count `D_domain` (ref term → ε), `S_domain` (ref term → anything else,
     including another term — one substitution, never S+I), `I_domain` (hyp term →
     ε or → non-term ref token).
  5. `N_domain` = reference term occurrences.
     `DS-WER = (S+D+I)_domain / N_domain`. Return **NA when `N_domain == 0`**, never 0.
- [ ] Failing tests first, covering at minimum: exact match / substitution /
  deletion; domain vs irrelevant insertion; term-for-term substitution counted
  once; repeated and reordered terms; multiword and overlapping terms; Greek final
  sigma, tonos, case, hyphens; `N_domain == 0` → NA; numerator > denominator via
  insertions; deterministic ties; wrong-city term list yields no matches; duplicate
  alias rejected at load.
- [ ] Run tests → red → implement minimal → green. Commit.

### Task 2.3: Score providers

- [ ] Pull per-provider hypotheses via the benchmark API
  (`docs/runbooks/benchmark-api.md`; `report.json` is public). Providers: ours
  (the 2026-08-10 run of `artifact-ct2-fixed`), Gladia, Scribe v2, Soniox, base
  whisper. **Require complete hypotheses for all 39 manifest windows from every
  provider, or fail loudly** — no silent complete-case subsets. Hypotheses stay in
  `~/.cache/oc-public/`.
- [ ] Compute DS-WER per provider on the 39-window manifest. Primary analysis: all
  39 windows. Sensitivity analysis: excluding the roll-call windows. Report both,
  decided in that order **before** looking — never keep whichever passes.
- [ ] Uncertainty: same meeting-block bootstrap as Workstream 1. Report `N_domain`
  total and per city, S/D/I breakdown, and the milestone ratio with its CI. If
  Gladia's DS-WER is 0, the ratio is undefined — report absolute deltas instead.
- [ ] Write `docs/reports/2026-08-12-ds-wer.md` including the two honesty notes
  above. Create+close `exp-2026-08-12-ds-wer`; update `CURRENT.md` queue (Milestone
  2 status). Run the checker. Commit.

---

## Workstream 3 — Correction-only dataset ablation (`exp-2026-08-13-correction-only`)

**What this can and cannot answer** (state verbatim in the prereg): removing the
`no_edit` rows changes label provenance AND dataset size AND optimizer steps AND
example difficulty AND speaker/city mix simultaneously. Under the fixed two-epoch
recipe this measures *the effect of dropping all `no_edit` rows*, not "do unverified
labels hurt" in isolation. It prices the data; it does not crown a winner.

**Legal-hold check:** the DPO hold (2026-07-17, `docs/decisions/data.md`) blocks
**publication** of the dataset; on-pod training with the same data-handling as the
2026-08-01 run of `artifact-adapter-fixed` is existing practice. Re-read
`docs/decisions/data.md` before provisioning; if it prohibits processing (not just
publication), this workstream is blocked — surface it and skip.

### Task 3.1: Preregister + data audit (before any GPU)

**Files:** Create `docs/specs/2026-08-13-correction-only-preregistration.md`
(pattern: `docs/specs/mixture-ratio-preregistration.md`).

- [ ] Verify what `correction` labels actually are (a human edited the row) vs
  `no_edit` per `docs/reports/2026-08-11-training-brief.md` — confirm against the
  actual manifest field, and record the field name and manifest hash in the prereg.
- [ ] Audit the filtered subset before spending: expected ≈11.16h / ≈13.9k rows;
  record hours, rows, meetings, cities, speakers, date range; verify zero overlap
  with argos/orestiada and with meetings ≥ 2026-06-01. Fail closed on mismatch.
- [ ] Arm: `correction` rows only, exact `artifact-adapter-fixed` recipe (r=32,
  α=64, dropout 0.05, q/v only, lr 1e-4, batch 2 × grad_acc 4, 2 epochs, same
  seed). Control: `artifact-adapter-fixed` itself — no second run.
- [ ] Decision rule: paired micro-WER + deletion rate on the 39-window manifest,
  meeting-block bootstrap, both arms decoded fresh in the same environment.
  The 2.1-point per-seed spread (`exp-2026-08-08-mixture-ratio`) is reported as
  context, not used as a threshold; the single-run outcome is **suggestive** either
  way and is labelled so.
- [ ] The filtered manifest contains transcript text → it stays out of git; the
  prereg records its hash only. Commit prereg.

### Task 3.2: Train

- [ ] Pod per `docs/runbooks/runpod-training-pod.md`. Watchdog with hard deadline
  BEFORE upload; pod ID into the prereg file. Record: base-model revision, source +
  filtered manifest hashes, code SHA, seed, steps, GPU type.
- [ ] Train, download the adapter, kill the pod, **verify the pod is dead**.
- [ ] Register the adapter in the ledger with content hash before any scoring.

### Task 3.3: Score

- [ ] Merge + convert with `serve/oc-asr/build_model.sh` (int8). Decode BOTH the new
  build and `artifact-ct2-fixed` fresh, same device, same process environment,
  under the **control decode config** (arm A) regardless of Workstream 1's outcome.
- [ ] Paired analysis per the prereg. Report, create+close the ledger record, run
  the checker, commit.

---

## Workstream 4 — Final report (`exp-2026-08-20-final-report`)

Last; consumes 1–3 but does not block on 3 if the GPU run slips.

- [ ] Structure: (1) the question from `CURRENT.md`; (2) headline from
  `exp-2026-08-10-benchmark-fixed-adapter` with its six caveats verbatim; (3) the
  error taxonomy — convention vs acoustic gap, and that roughly half the residual
  errors are homophone spelling the audio cannot decide; (4) DS-WER vs Milestone 2
  with the retrospective-specification note; (5) decode-ablation outcome; (6)
  correction-only outcome (labelled suggestive); (7) limits written from ledger
  caveats, not memory: validation-set reuse/adaptive bias, single seed,
  agreement-vs-OpenCouncil vs fidelity-to-audio, cross-stack caveats, legal hold.
- [ ] Every number cites its experiment ID; every number names its metric.
- [ ] Record the unresolved product decision (below) in the report and `CURRENT.md`.
- [ ] Update ledger + `CURRENT.md`, run checker, commit.

---

## Not in scope (deliberately)

- New full retrains for a better headline; more data; prompt-engineering (measured
  saturated); encoder unfreezing; cWER.
- The name-lexicon feasibility audit (`exp-2026-08-11-name-repair`) — own plan, own
  gate.
- HF publication of `artifact-adapter-fixed` — a user decision, not agent work
  (queue item 1, blocked on nothing technical).

## Unresolved product decision (do NOT block on it, do NOT decide it)

Do transcripts keep filled pauses («εεε») and false starts? This decides what a
public record is; every future listening hour produces incompatible data until it is
answered. The executor records it prominently in `CURRENT.md` and the final report,
asks the user once at start of work, and proceeds regardless — never changing the
frozen benchmark normalizer or existing labels on its own.

## Suggested order

Day 0: Task 0 (freeze everything). Day 1: Tasks 2.1–2.2 (CPU) + 1.1–1.2 smoke,
launch the 39×6 decode batch unattended. Day 2: 1.3, 2.3. Day 3: 1.4. Days 3–5:
Workstream 3. Then Workstream 4.
