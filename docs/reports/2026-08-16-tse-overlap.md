# Target-speaker extraction over overlap

**Experiment:** `exp-2026-08-16-tse-overlap`  
**Status:** `CLOSED`  
**Protocol:** [`docs/specs/2026-08-16-tse-overlap-prereg.md`](../specs/2026-08-16-tse-overlap-prereg.md)

## Decision

The WeSep BSRNN/ECAPA checkpoint passes the preregistered Stage 1 mechanism
screen on the frozen additive-mixture set, but it is **not a deployment
candidate** for natural council overlap. The real-overlap Stage 2 audit had only
one of six participating speakers with sufficient enrollment; its one extracted
track tied the normalized mixture baseline and recovered no missing reference
word. No GPU run, paid API call, second architecture, or benchmark-window run is
authorised by this result.

## Stage 1

The frozen manifest contained 37 items, 484 reference tokens, six meetings and
481 decoded arm files. Extraction produced 224 tracks with zero failures; the
ASR decode completed 481/481 files. The primary gate was 0 dB SIR.

| arm | WER | raw S/D/I | deletion rate |
|---|---:|---:|---:|
| `CLEAN_NORM` | 0.2376 | 56/37/22 | 0.0764 |
| `MIX_NORM` | 0.6384 | 197/81/31 | 0.1674 |
| `TSE` | 0.2934 | 94/39/9 | 0.0806 |
| `TSE_WRONG` | 0.9421 | 312/125/19 | 0.2583 |
| `TSE_ABSENT` | 0.6529 | 182/119/15 | 0.2459 |
| `TSE_CLEAN` | 0.2500 | 68/30/23 | 0.0620 |

All four gates passed:

- G1 separation: `D(TSE)=0.3450`; all six leave-one-meeting-out deltas were
  negative; largest item share was 0.0838.
- G2 targeting: `D(TSE_WRONG)=-0.3037`; contrast `C=-0.4762`.
- G3 clean safety: cost `+0.0124`, below `+0.05`.
- G4 deletion harm: `-0.0868`, below `+0.02`.

The source metric supports the same mechanism interpretation: median SI-SDR of
`TSE` was 13.20 dB against the target and -24.78 dB against the masker. This
screen is favourable to TSE: additive mixtures, oracle-clean enrollment, and
different-room sources. It does not establish transfer to reverberant Greek
council overlap.

## Stage 2

The real substrate contained three cases, two meetings, 10.57 overlap seconds,
24 duration-weighted reference-token estimate, and six participating speakers.
Only one speaker was enrollable. There were no extraction failures.

The table reports raw `S/D/I` counts and total errors. It is deliberately
case-level; no aggregate WER, delta, or bootstrap is computed from this
substrate.

| case | enrollment | `BASELINE` | `MIX_NORM` | TSE |
|---|---|---:|---:|---:|
| Samothraki `...715000__g0` | 0/2 speakers | 12/5/0 (17) | 7/7/2 (16) | not available |
| Xylokastro `...3250000__g0` | B only, 1/2 | 1/0/2 (3) | 1/0/1 (2) | B: 1/0/1 (2) |
| Xylokastro `...4210000__g0` | 0/2 speakers | 1/4/0 (5) | 0/4/0 (4) | not available |

The available `TSE.B` track introduced one new token relative to the baseline,
but zero of those new tokens were in the human reference. This is not evidence
of a real-overlap improvement; the enrollment coverage is too sparse for that
claim.

## Provenance

- Stage 1 manifest: `944d379daa9b27598d2782612b3e66dedcbbe40942210ef046f62e21115241cd`
- Stage 2 manifest: `92d1f641d809262c12c81380906f25518e9bca550ff00efcf006987dfae96eb0`
- Stage 1 result: [`eval/results_tse_overlap.json`](../../eval/results_tse_overlap.json)
- Stage 2 result: [`eval/results_tse_overlap_stage2.json`](../../eval/results_tse_overlap_stage2.json)
- WeSep source commit: `99eca54b`
- Checkpoint: `avg_model.pt`, SHA-256
  `3d050217e1ab31b7cf25835f35d1969b415bb95f2ac52e2c5e2a743ebd8f90e5`

All audio, extracted tracks, hypotheses and transcript text remain under
`~/.cache/oc-public/tse-2026-08/`; none is stored in git.
