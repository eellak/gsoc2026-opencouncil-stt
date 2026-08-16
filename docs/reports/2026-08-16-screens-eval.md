# Two training screens, evaluated: substitutions bought with deletions

2026-08-16. `exp-2026-08-14-external-packs` (RUN 2) and
`exp-2026-08-13-targeted-deletion-training` (RUN 1). Procedure and the frozen
decision tree: [`../specs/2026-08-16-screens-handoff.md`](../specs/2026-08-16-screens-handoff.md);
design: [`../specs/2026-08-15-external-packs-screens-prereg.md`](../specs/2026-08-15-external-packs-screens-prereg.md).

**These are single-seed screens.** The measured per-seed spread in this project is
2.1 WER points (`exp-2026-08-08-mixture-ratio`), larger than every difference
below. A screen decides nothing by itself, and nothing here crowns a candidate.

## What was run

| arm | what it is | trained on |
|---|---|---|
| control | `artifact-adapter-fixed`, the current corrected adapter | — |
| RUN 1 | stage-2 only: the in-domain targeted-deletion mix, seed 101, 10,552 steps | A40 |
| RUN 2 | stage-1 LoRA on the balanced external mix (STOMA + Common Voice scripted + EuroSpeech), 5,190 steps, then the **identical** stage-2 as RUN 1, same seed, same 10,552 steps | A5000 |

The shared stage-2's **realized** mixture over 42,204 presentations was
68.9/18.6/2.6/10.0 (backbone / deletion-hard / names / other). The deletion
prereg's design was 55/25/10/10 — so what was trained is a *weaker* dose of
deletion-hard and a much weaker dose of names than intended. Every number below
belongs to the executed mixture, and the gap is itself one of the things the
error analysis has to weigh.

RUN 2's stage-1 adapter was scored too, as an extra; see the last section.

Everything was decoded on the local minipc stack only: CTranslate2 int8, CPU,
16 threads, the frozen control decode configuration, the same per-window random
seeds as the control (common random numbers), the same normalizer. No GPU-stack
number appears in this report.

### CT2 fingerprints (`model.bin` sha256)

| arm | sha256 |
|---|---|
| control (`artifact-ct2-fixed`) | `8a1a3b257d0c1bdb71877f36db902a46c14697ff587766b91d6c47973f8fb85b` |
| RUN 1 | `d808868ac7e893bc7fd73e59fb6dda65dde0f3ce0d0a453548efa1a8e870e5a1` |
| RUN 2 stage-1 | `96088a400285721d7ec1bd617a7454ea59a87f17327c7743c0c8a9a755c3d0bd` |
| RUN 2 stage-2 | `444de4e963742227654a39bb3eabac45541279177ce59b69ff4669d7787e0cac` |

Four distinct binaries; the arms provably decode different weights. The source
adapters were sha256-verified against the pod before the pods were removed
(`run2-artifacts/pod-sha256.txt`). Fingerprints prove distinct artifacts, not
correct ancestry — ancestry is checked separately below.

## Results on the 39 frozen windows

39 windows, 31 meetings, 11,911 reference tokens. Paired bootstrap over meetings,
4,000 replicates, seed 7. Rates are per reference token.

| arm | WER | del | ins | sub | S / D / I |
|---|---:|---:|---:|---:|---|
| control | 0.1589 | 0.0600 | 0.0201 | 0.0788 | 938 / 715 / 240 |
| RUN 1 | 0.1641 | 0.0788 | 0.0174 | 0.0680 | 810 / 938 / 207 |
| RUN 2 stage-2 | 0.1591 | 0.0756 | 0.0163 | 0.0672 | 801 / 900 / 194 |

Deltas vs control (positive = worse):

| metric | RUN 1 Δ | CI95 | RUN 2 Δ | CI95 |
|---|---:|---|---:|---|
| WER | +0.0052 | [−0.0102, +0.0240] | **+0.0002** | [−0.0124, +0.0165] |
| **del_rate** | **+0.0187** | **[+0.0012, +0.0414]** | **+0.0155** | **[+0.0016, +0.0341]** |
| ins_rate | −0.0028 | [−0.0080, +0.0013] | −0.0039 | [−0.0099, +0.0010] |
| **sub_rate** | **−0.0107** | **[−0.0195, −0.0039]** | **−0.0115** | **[−0.0186, −0.0062]** |

Both WER deltas include zero. Both deletion and both substitution deltas exclude it.

**RUN 2 vs RUN 1, paired on the same windows: every CI includes zero.**

| metric | Δ (RUN2 − RUN1) | CI95 |
|---|---:|---|
| WER | −0.0050 | [−0.0195, +0.0090] |
| del_rate | −0.0032 | [−0.0193, +0.0121] |
| ins_rate | −0.0011 | [−0.0040, +0.0016] |
| sub_rate | −0.0008 | [−0.0045, +0.0034] |

Head-to-head windows: RUN2 19 / tie 4 / RUN1 16.

### Single-item domination

- RUN 1 vs control, del_rate: 0.0187 → 0.0139 without the most influential window
  (`win_orestiada_apr28_3_2026_1291909`), **no sign reversal on any metric**.
- RUN 2 vs control, del_rate: 0.0155 → 0.0110 without `win_argos_sep10_2025_573077`,
  **no sign reversal**. Same for ins and sub.
- RUN 2 vs control, **WER**: the sign is flipped by 13 of 39 individual windows.
  That delta is +0.0002 — it is noise around zero and its sign carries no
  information. It is quoted as "flat", never as "better".
- RUN 2 vs RUN 1, del_rate: the sign flips when one window is dropped. Another
  reason that comparison resolves nothing.

## DS-WER (domain terms)

Primary cut is v2 `entities` (274 occurrences). Never decides alone — the
occurrence count is small and the CIs are wide, and v2 is retrospectively
specified, so these are the first *prospective* v2 numbers.

| arm | v1 | v2 all | **v2 entities** | v2 person_surname | v2 procedural |
|---|---:|---:|---:|---:|---:|
| control | 0.512 | 0.3474 | **0.5365** | 0.5122 | 0.1579 |
| RUN 1 | 0.596 | 0.4357 | **0.6058** | 0.5976 | 0.2632 |
| RUN 2 stage-2 | 0.536 | 0.3934 | **0.5657** | 0.5366 | 0.2180 |

The composition tells the same story as the overall metric. On the entities cut,
S/D goes 114/26 (control) → 70/93 (RUN 1) → 90/58 (RUN 2). Domain-term
substitutions fall and domain-term deletions rise, independently of the pooled
WER. No arm improves on the control's entities number, and no paired uncertainty
was computed here, so this ranking is corroborating, not deciding.

## Reading

**Both stage-2-containing screens produced a substitution/deletion tradeoff
against the control.** Substitutions fell (both CIs exclude zero) and deletions
rose (both CIs exclude zero), and the two moves roughly cancel in WER. The mix
that was built to *reduce* deletions is followed by *more* deletions in both
arms. The models got quieter, and a pooled-WER-only look would have shown
nothing at all — the exact failure mode `CLAUDE.md` warns about, with the sign
reversed.

Stated as causally as the design allows: this is an association observed on one
seed per arm, with no ablation separating data composition from anything else in
the training run. The aggregate counts also do not demonstrate that particular
substitutions literally became deletions; that is shorthand for the direction of
the two deltas.

**RUN 2's bundled change was not detectably different from RUN 1.** RUN 2 differs
from RUN 1 in three things at once: external stage-1 data, 5,190 extra optimizer
updates, and the training GPU (A5000 vs A40, which affects the checkpoint through
numerical nondeterminism, not the evaluation — every arm was decoded on the same
CPU stack). This design cannot attribute anything among those factors. "Not
detected" is not "no effect": the CIs cover meeting-sampling uncertainty for these
fixed checkpoints and say nothing about training-seed uncertainty, which is known
to be larger than every delta here.

### What the frozen tree says

- **RUN 2 vs RUN 1**: no CI excludes zero → **both advance**, worded as "RUN 2
  recipe vs RUN 1" with the dose + data + GPU confound recorded.
- **vs control**: the primary criterion was a deletion-rate drop without
  insertion/WER regression. Deletions rose in both arms with CIs excluding zero.
  **Neither arm meets it; the control keeps the candidate slot.**
- Branch reached: **both worse on deletions → no blind retry. Error analysis
  first.**
- WER is +0.005 and +0.000 against the control — nowhere near the +2-point
  pipeline-bug threshold, so nothing here needs to be re-run before it can be read.

The earlier prereg's point-estimate promotion rule (WER Δ < 0 **and** del Δ ≤ 0
vs the baseline arm) is technically satisfied by RUN 2 against RUN 1
(−0.0050 / −0.0032). Recorded for completeness, and it governs only *promotion of
RUN 2 relative to RUN 1*; it says nothing about replacing the control, and both
of its point estimates sit inside CIs that include zero.

## Pipeline-bug check (done before interpreting)

No positive sign of a scoring or decoding bug:

- Four distinct CT2 fingerprints; the two RUN 2 adapters were sha256-matched to
  the pod before deletion.
- Ancestry: `run2-artifacts/stage2.log` records `stage continuation: LoRA
  initialized from /workspace/stage1/adapter (192 modules, 384 tensors, 192
  nonzero lora_B; fresh optimizer/scheduler over max_steps=10552)` — stage 2 did
  start from *trained* stage-1 weights.
- Stage-2 substrate identical across runs: `n_train` 42,204 and `max_steps`
  10,552 in both `run_meta.json` files.
- S+D+I reproduce the quoted WERs exactly; the control arm is the same
  `eval-A.json` decode the other frozen-window experiments use.
- The deletion increases survive leave-one-window-out.

What remains worth auditing is the **data**, not the harness: whether the
deletion-hard rows themselves carry truncated targets, or whether the selection
criterion that produced them favours examples where omission is the correct
label.

## Cost

RUN 1 ≈ $6.20 (A40, pod `t6ugwl9f4efu23`), RUN 2 $6.74 (24.96 h × $0.27, A5000,
pod `hwydnokhc60y2f`). Screen total **≈ $12.94** against the $22 ceiling. Both
pods removed, both watchdogs killed, zero pods alive. Evaluation cost $0: local
CPU.

## Extra: RUN 2 stage-1 alone

Scored only because the machine was free. It is an intermediate checkpoint — the
externals with no in-domain stage-2 — and it is **not** RUN 2's candidate; RUN 2's
candidate is stage-2. Numbers are in
`~/.cache/oc-public/train-screens-2026-08/run2-eval-stage1/results.json` if the
decode completed; it answers "what did the external packs alone do to the frozen
windows", which no gate in either prereg refers to.

## Artifacts

Outside git (transcript text never enters the repo):

- `~/.cache/oc-public/train-screens-2026-08/run1-eval/{decode,results}.json`
- `~/.cache/oc-public/train-screens-2026-08/run2-eval-stage2/{decode,results}.json`
- `~/.cache/oc-public/train-screens-2026-08/pair-run2-stage2-vs-run1.json`
- CT2 builds: `~/oc-run1-screen/ct2`, `~/oc-run2-stage2/ct2`, `~/oc-run2-stage1/ct2`
  (each with its `ct2.sha256`)

In git: [`notebooks/screens_score.py`](../../notebooks/screens_score.py) (decode,
score-vs-control, paired arm-vs-arm) and
[`scripts/build_ct2_local.sh`](../../scripts/build_ct2_local.sh).

## Next

Not a retry. The deletion-hard supply is what the next look should be aimed at:
which rows it contains, what the losing meetings have in common, and whether the
label for a deletion-hard row is ever itself truncated. Any confirmation run
after that needs 3 seeds and matched GPUs, per the frozen gates.
