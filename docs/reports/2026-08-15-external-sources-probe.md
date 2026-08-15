# Three more external sources: STOMA, EuroSpeech-el, Common Voice Scripted Greek

2026-08-14/15. Ledger record: `exp-2026-08-14-external-packs`. Pipeline contract:
[`docs/reference/external-source-packs.md`](../reference/external-source-packs.md);
procedure: [`docs/runbooks/external-corpus-to-training-pack.md`](../runbooks/external-corpus-to-training-pack.md).

Goal: does adding external Greek speech as **stage-1 adaptation** (never mixed into
the in-domain corrections) move agreement-WER on the 39 frozen validation windows,
without a deletion-rate regression? Prior expectation is small-or-zero
(`exp-2026-08-11-wer-levers-research`: ~1300 h buys ~0.5 points).

## Step-0 vetting (no audio downloaded)

| | STOMA | EuroSpeech greece | CV Scripted Greek 26.0 |
|---|---|---|---|
| repo | `aangelakis/STOMA` (authors' own) | `disco-eth/EuroSpeech` | Mozilla Data Collective |
| licence | CC-BY-4.0, no conflict seen | **`other`** → for Greece cites ν.2121/1993 art. 2(5) + 25(1)(b): a **statutory exception**, not an NC licence — see the correction below | CC0; no re-hosting, no speaker re-identification |
| size | ~23 h, 6 speakers, 15 shards | ~2,395 h at CER<20%, 562k rows, 527 train shards | 20.2 h validated (17,528 clips), 454 speakers |
| domain | studio read speech (Harvard + B2/C1/C2 exam texts) | Hellenic Parliament plenary | read prompts, community recordings |
| text | accented, punctuated, cased, **complete sentences** | accented, punctuated, cased; slices of official minutes, honest mid-sentence edges; median ~15 s | punctuated prompts (to verify on download) |
| repair needed | none | none (completeness recorded, not invented) | none expected |

Two findings worth keeping:

- **EuroSpeech solves both HParl defects** (3 s fragment boundaries, stripped
  orthography) in the same parliament domain, at 20× the hours. If its keep rate
  holds, it supersedes hparl2 as the parliament source. The two must never be used
  together without cross-pack dedupe — same chamber, possibly same sessions.
- EuroSpeech ships per-row `wer`/`cer` from **its own Whisper-Turbo alignment
  pipeline**. Convenient, and biased toward Whisper-easy audio — the exact model
  family we train. It may only ever pre-select what is sent to Soniox; admission is
  always our gate.
- The HF mirror `h-gajdov/mozzila_common_voice` suggested as a CV shortcut is
  **Macedonian** (Cyrillic, «литературен»), not Greek. Unusable. The Greek CV tar.gz
  requires an authenticated Mozilla Data Collective download.

### Licence correction, 2026-08-14: EuroSpeech is not blocked the way HParl is

This report first filed EuroSpeech's licence as "needs the same legal call as HParl".
That was wrong, and the difference matters more than either entry.

- **HParl** is restricted by the **CLARIN/ILSP packaging** — the compilation is offered
  under CC BY-NC. The restriction attaches to the dataset, not to what the Βουλή said.
- **EuroSpeech** claims no licence of its own over the material. For Greece it cites
  **ν.2121/1993 art. 2(5)** — official texts of the State fall outside copyright
  protection — and **art. 25(1)(b)**. Those are statutory exceptions on the underlying
  proceedings. There is **no NC term to clear**.

So the blocker was never "Greek parliamentary speech is unusable". It was one
compilation's terms. The same domain is available through EuroSpeech at 20× the hours,
with commercial use open.

Two things a lawyer should still confirm, and they are narrower than the HParl
question: (a) that the cited articles cover the **audio**, not only the transcripts;
(b) that the disco-eth compilation imposes nothing itself — its card explicitly
disclaims responsibility for the accuracy of the licence table.

## Codex plan review (2026-08-14, effort high) — adopted / declined

Adopted: train splits only (EuroSpeech splits are session-disjoint; validation/test
may serve as future benchmarks); calibrate the ds_wer preselect on a stratified
sample instead of assuming a threshold, and report keep-rate by ds_wer bin; a 15–20%
exploration lane of mediocre-ds_wer rows so the pack is not exclusively
Whisper-easy; **boundary-edge admission flags** (first/last reference token missing
from the ASR hypothesis ⇒ clipped audio/transcript ⇒ reject — truncation is the
failure this project is fighting); all validated CV rows with no LLM importance
selection (20 h is too small to curate down; diversity is the value); per-source
dedupe keys (STOMA: text+speaker — the same sentence by another speaker is a
legitimate sample); screens as paired-seed with a matched stage-2-only baseline and
dose-equalized stage 1.

Declined for pilot scale, revisit before any large pass: audio-fingerprint dedupe,
fake-ASR end-to-end test harness, contiguous-window concatenation for long-context
coverage, punctuation-F1/diacritic side metrics.

## Pilot filters (150 rows each, seed 20260814)

`eval/ext_filter.py` — the generic filter (tests:
`eval/tests/test_ext_filter.py`). Same per-row pipeline as `hparl2_filter.py` plus
edge flags; label = source text verbatim, no repair stage. STOMA: 10 rows × 15
shards. EuroSpeech: 30 rows × 5 spread train shards, random (not preselected), so
the keep-by-ds_wer-bin table is an honest calibration.

### Results

| source | n | pooled WER | sub | del | ins | median ref tokens | keep @0.95 |
|---|---|---|---|---|---|---|---|
| STOMA | 150 | 0.054 | 0.038 | 0.014 | 0.002 | 9 | **72%** |
| CV Scripted el | 150 | 0.075 | 0.061 | 0.004 | 0.009 | 6 | **65%** |
| EuroSpeech greece | 150 | 0.285 | 0.048 | 0.109 | 0.129 | 31 | **23%** |
| *(hparl2, for scale)* | 9,062 | — | 0.052 | 0.031 | 0.017 | 10 | 46% |

**A corpus defect the pilot caught: Latin homoglyphs.** Many STOMA sentences begin
with a *Latin* lookalike letter ('Tο', 'Nα', 'Aρχικά' with Latin T/N/A). Normalized
tokens then never match, which both deflates alignment and — worse — would have
shipped Latin letters inside Greek training labels. Fixed deterministically
(`fix_homoglyphs`: map Latin→Greek only inside tokens that also contain Greek
letters, so genuine code-switching like «fake news» is untouched) and applied to
every source; the three pilot files were rescored offline from the cached Soniox
outputs. STOMA moved 67%→72% keep and its edge-flag rate fell 15%→9%; the other two
sources were unaffected.

Two read-speech corpora behave as expected: near-zero deletions, errors almost entirely
substitutions, i.e. Soniox mishearing rather than text-audio mismatch. Their keep rates
are the highest we have seen.

**EuroSpeech's 23% is not what it looks like, and the edge flags earned their place.**
Deletions (0.109) and insertions (0.129) are both high and roughly symmetric, which is
the signature of a *window* problem, not an editorial one — the audio span and the text
span do not cover the same speech. Splitting on the boundary flags:

| slice | n | keep @0.95 | pooled WER | del |
|---|---|---|---|---|
| clean-edge | 100 (67%) | **35%** | 0.102 | **0.019** |
| edge-flagged | 50 (33%) | **0%** | 0.627 | 0.275 |

Every rejected-for-clipping row is genuinely unusable, and once they are gone the
deletion rate is **0.019** — the same 1.9% measured on HParl's placeholder-free rows.
So EuroSpeech's official minutes are about as faithful to their audio as HParl's; what
sank the headline number was alignment slop at segment boundaries. The last reference
token is missing from the hypothesis on 26% of rows, the first on 7% — clipping is
mostly at the **end**.

Note the gate is not comparable across these sources: at a median of 6–10 reference
tokens (STOMA, CV, hparl2) `align ≥ 0.95` is effectively exact match, while at
EuroSpeech's median 31 tokens it tolerates one or two. EuroSpeech is being judged on a
looser gate and still keeps fewer rows.

**EuroSpeech ds_wer preselect calibration** (their Whisper-Turbo score vs our gate):

| their ds_wer | n | our keep rate |
|---|---|---|
| < 0.05 | 3 | 100% |
| 0.05–0.10 | 7 | 43% |
| 0.10–0.20 | 48 | 35% |
| ≥ 0.20 | 92 | 13% |

Monotone, so it is usable as a **preselect** to spend Soniox calls where they pay —
but a third of the rows it scores well still fail our gate, so it can never admit a
row. The n in the top two bins is 10; do not fit a threshold on this until the bins
are populated.

All three numbers are pilots (n = 50/150/150) and none has been checked by a human
listening to the audio.

### The gate these were scored against is wrong (2026-08-14)

The blind calibration on hparl2 (62 clips, hidden controls at both extremes — see
[the hparl report](2026-08-14-hparl-audio-text-probe.md) §"Blind round") found that
`align ≥ 0.95` mostly measures **whether Soniox succeeded**, not whether the label is
right, and that it discards rows a human accepts at roughly a 1:1 rate with what it
keeps. The preregistered replacement is `align ≥ 0.50 ∧ duration ≥ 1.5 s`, with
edge-clipping used to down-weight rather than reject.

Rescoring these same three pilots against it:

| source | keep @0.95 | @0.50 + 1.5 s | and dropping clipped edges |
|---|---|---|---|
| STOMA | 70% | **100%** | 88% |
| CV Scripted el | 65% | **99%** | 85% |
| EuroSpeech greece | 23% | **85%** | 66% |

EuroSpeech's headline 23% was the most distorted of the three, exactly as the boundary
analysis above predicted: once short-clip ASR failures stop being counted as label
defects, 85% of its rows survive, and the honest cost of its real defect — clipped
alignment windows — is the drop from 85% to 66%.

**Drop-vs-weight for clipped edges is a per-source call, not a global one.** On hparl2
the human said clipped rows still say their text, so they are admitted at half weight.
On EuroSpeech the edge-flagged slice sits at WER 0.627 and del 0.275 — a flag there
means a large span is missing, because its segments are ~31 reference tokens against
hparl2's ~10. Until a human judges EuroSpeech clips, its pack **drops** clipped rows.

These numbers are re-scorings of the existing pilots, not new runs. No human has
judged a STOMA, CV or EuroSpeech clip; the calibration that justifies the new gate was
done on hparl2 only, and it is being carried across on the argument that the failure
mode (Soniox on short clips) is a property of the verifier, not of the corpus.

## Full-pass results (finished 2026-08-15; harmonized gate: align≥0.5, dur≥1.5 s, edges kept at weight 0.5)

| pack | rows | hours | sha256 (prefix) |
|---|---|---|---|
| `stoma-v2` | 14,407 | 24.19 | `8f455a66` |
| `cv-v2` | 12,971 | 15.11 | `c2f60869` |
| `eurospeech-v2` | 14,136 | 58.57 | `931a8958` |
| **total external** | **41,514** | **97.9** | |

Findings from the full passes, beyond the pilots:

- **EuroSpeech regional cliff**: shards 000–399 pass the harmonized gate at 97–99%,
  shards 400+ at **24%**. A regime change, not a gradient; cause unresolved (source
  audio vs their alignment). Admission is per-row so the pack is safe, but its hours
  come mostly from the <400 region.
- **Infra failure recorded (failed twice the same way)**: both long HF-parquet
  passes died after ~2–7 h with `httpx client has been closed` from the
  fsspec-cached `HfFileSystem` singleton. Fix: `skip_instance_cache=True` per shard
  in `open_shard`; resumes were lossless via `--append`.
- **Overlap eurospeech↔hparl2**: 7/1000 sampled hparl2 kept segments appear
  verbatim inside eurospeech-v2 (~0.7%) — negligible, but the two packs still must
  not share an arm without dedupe.
- **Trainer gap**: the `weight` field on edge-flagged rows is not consumed by
  `train_runpod.py`; until it is, edge rows enter at full weight and no
  down-weighting claim may be made.

## Full passes (launched 2026-08-14 ~17:50)

1. **STOMA**: all 14,552 rows.
2. **CV**: all ~13.9k validated-minus-dev/test rows (the official train split is
   only 2,037 rows; dev/test stay untouched as potential third-party benchmarks).
   Original MP3 bytes are kept as-is — no second lossy transcode.
3. **EuroSpeech**: 59 shards spread across the train split (`range(0,527,9)`),
   `--preselect-ds-wer 0.15 --explore-frac 0.18`, ~19k Soniox calls targeting
   ~30 h admitted — enough for a dose-equalized screen arm; scaling past that is
   only worth it if the screen survives. The preselect threshold rides the pilot
   calibration; the exploration lane samples ds_wer ∈ (0.15, 0.35] so the pack is
   not exclusively Whisper-easy audio.

Then: build packs, register artifacts, and run the preregistered screens
([`docs/specs/2026-08-15-external-packs-screens-prereg.md`](../specs/2026-08-15-external-packs-screens-prereg.md)):
one arm per pack + one combined balanced arm, each stage-1 → frozen stage-2, vs a
matched stage-2-only baseline, frozen decode, 39 validation windows. Deletion rate
is the watched failure. 3-seed confirmation only for survivors. Cheapest RunPod
GPU, watchdog armed before upload.
