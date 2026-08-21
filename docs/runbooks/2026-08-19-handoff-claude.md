# Handoff to Claude: project scope and remaining path to 23/8

Date: 2026-08-19. Repository: `/home/harold/opencouncil-fine-tuning`.
Current git HEAD: `11976273`; the working tree contains intentional uncommitted
experiment work. Preserve it. This document covers the whole 18–23 August push,
not only the last dense experiment.

## Mission and scope

The GSoC question is: **does domain fine-tuning `whisper-large-v3` improve Greek
council transcription enough to matter?** The near-term deliverable is the best
defensible model plus serving/evaluation harness by **23/8**, with an honest account
of what improved, what did not, and which metric was measured.

The work has three connected layers:

1. **Weights/data:** find one evidence-backed training change that improves actual
   council transcription, not merely the training loss.
2. **Evaluation:** separate agreement with OpenCouncil's published text from fidelity
   to what was spoken; guarantee meeting and known-speaker separation for strict
   validation.
3. **Serving fallback:** preserve the already measured three-ASR fusion/name-repair
   stack if no new training arm earns promotion.

This push does not reopen the old UI roadmap, publish the training dataset, solve
every historical research ticket, or spend the sealed holdout. Dataset publication
remains on DPO/legal hold. HParl/CLARIN 1602 remains outside this cycle until licence
permission is resolved.

## Start here

Read, in order:

1. `CLAUDE.md` for repository protocol and authority rules.
2. `CURRENT.md` for the active queue and blockers.
3. `research/ledger.json` for canonical experiment/artifact/capability state.
4. `docs/reports/2026-08-20-final-report.md` for the project's answer before this
   last training push and the serving fallback.
5. `docs/decisions/training-evidence.md` for the evidence ladder and GPU gates.
6. `docs/decisions/data.md`, “Strict validation and hybrid-data contract”.
7. The linked reports below only when their branch is active.

Run `git status --short` before editing. The dirty tree is shared work; do not reset
or discard it. Transcript text, review notes and audio stay under
`~/.cache/oc-public/`, never git.

The execution map is GitHub issue
[“Το καλύτερο δυνατό μοντέλο + serving harness μέχρι 23/8”](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/3)
with `wayfinder:map`; its child tickets cover measurement, training WER, window
density, clean data, CPT and overlap. Use the map for sequencing and the ledger for
truth. If they disagree, update the map after establishing the ledger state.

## Where to look

| Need | Source of truth |
|---|---|
| What is active/blocked | `CURRENT.md` |
| Experiment status, artifacts, caches, capabilities | `research/ledger.json` |
| Whole-project conclusion and shipping fallback | `docs/reports/2026-08-20-final-report.md` |
| Training terminology, gates and budget ladder | `CONTEXT.md`, `docs/decisions/training-evidence.md` |
| Validation split and hybrid-data guarantees | `docs/decisions/data.md` |
| Model/serving decisions already settled | `docs/decisions/modeling.md` |
| Reusable GPU procedure | `docs/runbooks/runpod-training-pod.md` |
| Portable bundle/network storage | `docs/runbooks/pod-training-bundle.md` |
| Current training results | reports linked in “Completed work” below |
| Raw transcripts/audio/hypotheses | private paths recorded in matching ledger artifacts |

Reports are dated evidence, not live state. Search the ledger before following one.

## Decisions agreed with the user on 18–19 August

- Treat each proposed advantage as one **mechanism candidate**. Run zero-GPU local
  feasibility/data/leakage checks first; use tiny proxy/auto-research only as a veto;
  buy extended `large-v3` training only after a real 300-step paired-seed screen.
- The primary endpoint is validation WER. Training WER must be reported because the
  mentor asked for it, but it is diagnostic and never promotes an arm.
- Cleaner data means a **hybrid**: reproducible clean core plus an audited protected
  lane for names, fast/hard speech and boundary examples. Wholesale strict filtering
  is rejected because it removes exactly those cases.
- Fast iteration may use the frozen 39-window agreement set. A shipping claim needs
  a strict audio-faithful validation set, meeting-disjoint and known-`person_id`-
  disjoint from training. The target improvement is at least **0.5 absolute WER
  points** without protected-slice regression.
- Every meeting and every known speaker belongs exclusively to train or validation.
  This is a metadata guarantee, not biometric proof about unidentified people.
- Each GPU stage needs fresh explicit approval after exact arms, causal claim,
  seeds/steps, ETA and cost are shown. A passing screen authorizes discussion of the
  next stage, not the stage itself.
- Long jobs are unattended and event-driven: detached runner, durable logs, terminal
  marker and hard billing deadline. The user explicitly does not want repeated
  polling or token-heavy status checks.
- Fresh external research runs by SSH on the MacBook with Grok Research. Existing
  repository evidence remains authoritative for questions already measured.

## Evaluation substrates: keep them separate

| Substrate | Purpose | Status/guard |
|---|---|---|
| Frozen 300-row training sample | Training WER diagnostic | 136 meetings, 9 cities; never selects an arm |
| Frozen 39 windows | Fast screen | Argos + Orestiada, 31 meetings, 11,911 tokens; agreement with published OpenCouncil text |
| Strict validation | Shipping decision | Must be newly frozen from the whole-city pool, audio-faithful, meeting- and known-speaker-disjoint |
| Seven temporal windows | Branch-specific final holdout | Part of the repository's 16 locked windows; remain sealed and require explicit user decision |

The immediately auditable strict-validation pool has 4,016 rows / 3.094 h / 30,236
normalized tokens / 55 known speakers / 45 meetings after null-speaker removal. The
exact strict subset size and construction must be preregistered before reading a
candidate's output. Repairing the two dense outlier references is necessary for the
dense diagnosis, but **two repaired windows are not the strict validation set**.

## Workstream map

| Workstream | State | Meaning / next move |
|---|---|---|
| Measurement calibration | CLOSED | Current control range measured; use paired effects, not the old “2.1 variance” shorthand |
| Training WER | CLOSED | Adapter learned corrections; residual is concentrated in correction rows |
| Window-density mechanism | CLOSED screen | Real WER/deletion direction, but insertion/dominance gates stop this recipe |
| Human boundary/reference audit | CLOSED | Hard training sample mostly unresolved; two dense validation references materially incomplete |
| Strict audio-faithful validation | DESIGN/FREEZE NEXT | Freeze selection/protocol now; complete human references only for a screen survivor |
| Hybrid clean-core + protected lane | PENDING DESIGN | CPU manifests/audits first; one arm, equal source hours |
| Next GPU screen | NOT AUTHORIZED | Ask after hybrid manifests and strict selection/protocol are frozen; use the 39-window fast screen |
| Serving/fusion fallback | AVAILABLE | Existing three-ASR composition plus roster repair remains the fallback if training stops |
| Temporal holdout | SEALED | No access until a candidate passes screen and strict validation gates |

If the properly constructed hybrid screen also stops, training research ends for this
cycle and the existing model/fusion stack is shipped. Do not create a third blind
training idea to avoid that branch.

### Wayfinder ticket routing

- `#44` measurement rule: evidence completed; do not redefine the gate.
- `#36` training WER: completed and now a required diagnostic for future runs.
- `#41` 30-second/dense training: completed as a stopped screen plus diagnosis.
- `#42` cleaner dataset: current critical path, now narrowed to hybrid construction.
- `#40` CPT/broad adaptation: not the next GPU arm; one isolated mechanism only, and
  HParl remains licence-blocked.
- `#43` overlap/speaker separation and `#39` pyannote are separate workstreams; they
  do not displace strict validation + hybrid data before the deadline.
- `#37` mentor communication should consume the final measured training WER and
  screen/audit outcome; check the issue before assuming whether it was already sent.

The GitHub map may lag the local completion recorded here. Update issue text/status
after the corresponding ledger record, not instead of it.

## Responsibility boundaries

| Need | Who/where |
|---|---|
| Code, manifests, tests, cache inspection, pod preparation | Claude should do it directly |
| Listening or writing audio-faithful reference text | Ask the user through a prepared private UI/workflow |
| New GPU spend, medium/full stage, sealed holdout | Explicit user decision |
| Current external literature/product facts | `ssh laptop`, Grok Research skill |
| HParl licence or dataset publication | Licence owner/DPO; engineering cannot self-authorize |
| GitHub sequencing | Wayfinder issue map; evidence still lives in the ledger |

## Current truth

There is a real directional training signal, but no promoted new model.

- Dense 300-step training reduced validation WER **15.31% → 14.76%**
  (`Δ=-0.00551`) and deletions by `-0.00568`; all three paired seeds improved.
- It increased insertions by `+0.00369` and violated the dominance guard. Frozen
  status: **`SCREEN — STOP`**. No 1,800-step or full stage is authorized.
- One window supplied 100/132 net extra insertions, but post-hoc exclusion is
  forbidden and still would not pass the insertion gate.
- Blind listening judged both insertion-heavy validation references to have
  **material omissions**. Apparent insertions there mix recovered speech with model
  errors until the references are repaired.
- Of 36 selected hard training clips, **7** were jointly label-faithful and
  boundary-usable, **1** had a definite unusable boundary, and **28** are unresolved.
  `unsure` is unresolved, not bad. This outcome-enriched sample does not estimate
  whole-corpus prevalence.

The branch is now **reference repair → hybrid/control manifests + strict selection
freeze → GPU approval request**, not another dense retry. Full strict annotation is
bought only if the short screen advances.

## Completed work since the 18/8 handoff

### Measurement and training WER

- Current control seed range on 39 validation windows is 0.285 absolute WER points,
  not the old shorthand “2.1”. Three paired seeds remain mandatory for screens;
  control-outcome SD is not paired treatment-effect SD.
  See `docs/reports/2026-08-18-training-measurement-calibration.md`.
- Frozen 300-row training sample: fixed-adapter WER `0.1313`; correction rows
  `0.2261`, no-edit rows `0.0385`. Base large-v3 is `0.2728`, so the adapter did
  learn the corrections. Training WER stays diagnostic only.
  See `docs/reports/2026-08-18-training-wer.md`.

### Density and clean-data preflights

- Training examples expose mean 3.553 s intended audio in 30 s (88.16% digital
  padding); inference windows average 91.19% source-audio occupancy. The distribution
  mismatch is measured.
- The packer builds 3,877 packs, mean 26.15 s, retains every row and inserts 0.4 s
  labelled silence between utterances.
  See `docs/reports/2026-08-18-training-window-density.md`.
- External-pack L2 filtering retains 30,044 rows / 60.069 h but disproportionately
  removes names, fast speech and all defined hard examples. Wholesale L2 replacement
  is vetoed. Use **L2 clean core + protected audited lane**, source-equalized.
  See `docs/reports/2026-08-18-clean-data-filter-census.md`.

### Dense GPU screen

- Preregistration: `docs/specs/2026-08-19-dense-screen-prereg.md`.
- Aggregate: `eval/results_dense_screen_300.json`.
- Report: `docs/reports/2026-08-19-dense-screen-300.md`.
- A = isolated rows; B = the same rows in dense Pn packs. Fixed 300 updates expose B
  to roughly 7× more labelled speech, so the causal claim is packing plus useful-token
  density, not context alone.
- Timestamp-supervised packing was worse in the older experiment. Extra silence is
  not a new repair because the packer already uses 0.4 s. These are not next arms.

### Residual and human audit

- CPU localization: `docs/reports/2026-08-19-training-residual-audit.md`.
- Frozen protocol: `docs/specs/2026-08-19-training-listening-audit.md`.
- Completed result: `docs/reports/2026-08-19-training-listening-audit.md` and
  `eval/results_training_listening_audit.json`.
- Private free-text notes are boundary-heavy. They guide repair but were not a scored
  field and cannot relabel `unsure` automatically.

## Exact next steps

### 1. Repair the two validation references

Resolve the two audit ids through the private hidden key and create audio-faithful
references while preserving the original OpenCouncil references. Keep two metrics:
agreement-with-published-text and fidelity-to-audio. Construct the corrected
references without inspecting model hypotheses.

The current review page recorded only completeness categories and notes; it did not
collect full replacement transcripts. Claude should prepare a private audio+editable-
reference page or equivalent deterministic workflow, then ask the user only for the
listening/transcription judgement the agent cannot perform. Keep model outputs hidden.

Completion criterion: both references have a human-audited private text/token
artifact, git contains a content-free manifest with hashes, and the scoring procedure
was frozen before any model is rescored.

Private inputs:

- `~/.cache/oc-public/training-listening-audit-2026-08/hidden-key.json`
- `~/.cache/oc-public/training-listening-audit-2026-08/served/answers.json`
- `~/.cache/oc-public/dense-eval-recovery-cx9ra8iluhr225/corrected-input/eval/validation/`

### 2. Freeze the hybrid training manifest

Build one treatment, not a search:

- clean core from reproducible L2 external-pack rules;
- protected lane for audited names, fast/hard speech and verified boundaries;
- admit the 7 reviewed `yes/yes` rows as-is;
- route the one definite failure to correction;
- withhold 28 unresolved rows until repaired or re-reviewed;
- equalize baseline and treatment hours separately per source;
- keep model recipe, updates and compute budget identical.

The external census is a feasibility result, not blanket authorization to mix every
pack. HParl remains outside the cycle because of CLARIN 1602. HParl and EuroSpeech
also cannot be silently combined because source-domain overlap was not measured. The
OpenCouncil in-domain backbone lacks an independent witness on every row. Record the
exact role of each source and what “control” means before building manifests.

Completion criterion: manifests, source-hour table, overlap/leakage checks, protected
slice counts and hashes exist before any pod is created. Freeze gates and cost in a
new spec.

### 3. Freeze strict selection and protocol, not all annotation yet

Use the accepted whole-city pool and allocate by whole meeting plus known
`person_id`, so no known meeting/speaker crosses train and validation. Freeze the
number of windows, selection seed, meeting/speaker roster, normalization, S/D/I
implementation and protected slices before inspecting a new candidate. Keep selected
references private and unscored.

Completion criterion before the short screen: content-free selection manifest and
hashes in git, private package in cache, zero meeting overlap, zero known-person
overlap, and all locked windows absent. Full audio-faithful human correction may wait
until an arm passes the 39-window screen; this preserves speed without selecting the
strict set from model outcomes.

### 4. Ask before the 300-step GPU screen

Give the user one concise proposal: exact arms, supported causal claim, steps/seeds,
GPU-hours/dollars and stop gates. A fresh explicit approval is required before pod
creation. Score first on the frozen 39-window fast substrate.

### 5. Only if the hybrid screen advances

Complete human audio-faithful references for the already frozen strict selection,
then score the candidate there. It must reach `ΔWER <= -0.005` without material
protected-slice regression before any medium/full proposal. Request separate approval
for each later stage. The repository globally has 16 locked evaluation windows; the
seven temporal windows relevant to this branch remain sealed throughout this screen.

## Serving and stopping rule

The corrected adapter `artifact-adapter-fixed` is published on Hugging Face but is not
the deployed production recommendation by itself. The established fallback is the
serving/fusion work summarized in `docs/reports/2026-08-20-final-report.md` and
`CURRENT.md`: three-ASR composition (`W`, approximately 0.10046 agreement-WER on its
fixed substrate) with conservative roster-name repair (approximately 0.09971,
shadow-only under its unresolved name-level gates). Treat those numbers under their
own ledger caveats; do not compare them directly with the 39-window dense CUDA result.

If the hybrid arm stops, finish the cycle with the existing corrected artifact,
fusion harness and documented shadow name repair. Shipping work then means packaging,
deployment checks and the final evidence narrative, not another training search.

Closed directions that need a new mechanism before reopening include mixture-ratio
tuning, targeted-deletion mixtures, external packs stacked on that failed mixture,
decode-threshold tweaks, timestamp-supervised packing, extra synthetic silence,
confidence fusion and wholesale L2 filtering. Broad HParl/CPT is outside this cycle.

## GPU, storage and unattended execution

- Local Ryzen 7840HS CPU is for manifests, audits, tests and slow offline decode. No
  reliable ROCm/iGPU training path or measured local training ETA was established;
  do not spend the deadline porting the trainer unless cloud GPU becomes impossible.
- Use a **VFM/Secure RunPod** GPU for approved training. The dense screen ran on an
  RTX A4000 at $0.250/h; historical 300-step paired screens cost roughly 6–7
  GPU-hours / about $3. These are estimates, not approval.
- Reuse network volume `qzw88vdwv2` (100 GB, EUR-IS-1) only after confirming it still
  exists and its contents/hash. It has a recorded $7/month idle ceiling and continues
  billing until deletion.
- Build/stage the content-addressed private bundle with `scripts/pod_bundle.py` and
  `scripts/stage_pod_bundle.sh`; see `docs/runbooks/pod-training-bundle.md`. The
  current archive is recorded as `artifact-pod-bundle-dense-screen-v1` in the ledger.
- `notebooks/train_runpod.py` now keys reusable clip/feature caches by source parquet,
  audio-byte, label and processor/tokenizer content, and resume fingerprints include
  batch/accumulation/max-steps/LR. Preserve these guards when adding the hybrid arm;
  they prevent a same-path stale cache/checkpoint from silently entering a GPU run.
- Download the base model directly from Hugging Face on the pod or into the volume's
  `HF_HOME`; do not relay the multi-GB model through the mini-PC. Reuse per-arm
  preprocessing/feature caches on the volume.
- Cloudflare R2 is not currently a verified transport: available Cloudflare
  credentials are for Tunnel, and RunPod's volume S3 endpoint needs separate S3 keys.
  Prefer the existing network-volume workflow unless credentials are explicitly
  established.
- Before upload/setup, arm the local hard-deadline watchdog. Run sequential paired
  arms on one pod when stack identity matters; only parallelize independent CPU/data
  preflights. Completion is signalled by a terminal artifact/marker. Wait without
  polling, then retrieve logs/results and terminate the pod immediately.

## Private artifacts and integrity

- Human answers SHA-256:
  `0a49fdc66fc6979e448aa4076828535e0da2c7da0d402fd8589d9bc5c9c0523c`
- Human hidden key SHA-256:
  `ba12a69d67f4cf26ac485033f427a2006df647b85715b384c201193ca3d50236`
- Dense raw results SHA-256:
  `544aa9c4fcfb5239e6bbbb80d55c356ec6172c3a18cfc502d76391c9ff8a1b4a`
- Dense private diagnosis SHA-256:
  `54f577e3b631ed077421f795b2275bc41b8e53a1c9f337aa8033a4751ee48d17`

No RunPod pod was active when this handoff was written (`runpodctl get pod` returned
empty on 19/8). The private audit server is transient service
`oc-training-listening-audit.service` on port 8792. Keep it until the hashed answers
have another safe copy.

## Reproduction and checks

```bash
cd /home/harold/opencouncil-fine-tuning
.venv-eval/bin/python -m eval.controlled_eval.score_training_listening_audit
.venv-eval/bin/python -m pytest -q \
  eval/tests/test_build_training_listening_page.py \
  eval/tests/test_score_training_listening_audit.py \
  eval/tests/test_dense_insertion_analysis.py \
  eval/tests/test_dense_screen_eval.py \
  eval/tests/test_pod_bundle.py
.venv-eval/bin/python scripts/check-research-state.py
git diff --check
```

The scorer emits aggregates only. Never commit answers, hidden keys, hypotheses,
notes or audio.

## User operating preferences

- Communicate concisely, preferably in Greek/Greeklish.
- Waiting is event-driven: no repeated polling or status chatter.
- Every GPU stage needs explicit approval after cost and ETA.
- For fresh external research, SSH to the MacBook and use Grok Research there.
  Repository evidence is authoritative for already measured claims.
