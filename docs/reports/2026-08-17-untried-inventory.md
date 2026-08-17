# What has not been tried — an inventory, 2026-08-17

Archaeology over 56 ledger records, 59 reports, 44 specs and 26 GitHub issues. No
experiment was run for this. Two numbers in it are new and were measured here on
cached data (roster coverage, §A2; the `exact_2_of_3` column breakdown, §A5); every
other number is a citation.

**The ranking is my judgement, not a measurement.** There is no experiment behind the
ordering. What is defensible is each row's evidence column and each row's blocker.

> **ERRATUM 2026-08-17 — §A2's roster-coverage number is wrong.** "56 of 260 benchmark
> windows (21.5%) and 43 of 203 meetings" does not replicate. Counting non-empty
> `data/pii/rosters_full.json` entries gives **238 of 260 windows (91.5%)** and **183 of
> 203 meetings (90.1%)**; on the 247-window analysis substrate, 232 (93.9%) and 133 of
> 144 meetings. 56 is the *fetch-failure* count in the last line of
> `data/pii/fetch_rosters.log`. Arm E is therefore **not** a structural no-op on 78.5%
> of the benchmark — it is a no-op on 6%. What survives: all 7 sealed temporal-holdout
> meetings do have zero-length rosters, which is why E fired zero times there. Measured
> and corrected in
> [`2026-08-17-name-repair-on-w.md`](2026-08-17-name-repair-on-w.md); §A1 of this
> inventory is closed by the same report.

Deadline for the main ranking is 2026-08-23. Anything that cannot produce a
defensible number by then is in [§C](#c-after-gsoc), not in the ranking.

---

## The one thing that reframes everything else

The project spent weeks on deletions. On the fusion output they are nearly gone.

Decomposing W → alignment-conditional column oracle, from
`eval/controlled_eval/results_composition.json` (verified here, arm `W` vs
`oracle_column`):

| component | W | column oracle | headroom |
|---|---|---|---|
| substitutions | 0.04271 | 0.01865 | **2.41 pts** |
| insertions | 0.03743 | 0.01312 | **2.43 pts** |
| deletions | 0.02032 | 0.01571 | 0.46 pts |

This is stated in [`2026-08-16-advisory-what-remains.md`](2026-08-16-advisory-what-remains.md)
("Κυνηγούσαμε διαγραφές επί εβδομάδες. Στο W έχουν σχεδόν εξαντληθεί") and it is
**not in the ledger**. The standing finding "the deletions live in the weights"
([`2026-08-12-serving-stack-ladder.md`](2026-08-12-serving-stack-ladder.md)) is true
and is a *pre-fusion* statement. It describes `artifact-adapter-fixed` decoding
alone. It does not describe W.

That report established the claim three ways, and it is worth knowing exactly how,
because it forecloses a lot:

1. decode thresholds fire zero times on 39 windows (`exp-2026-08-12-decode-ablation`);
   the two anti-deletion arms came out byte-identical to control — a closed channel,
   not a weak null;
2. the audio is covered — in the 3 dominant windows the uncovered audio fits ≤19
   words while 75–116 are missing (`c-closure-vad-2026-08-12.json`), **n = 3 windows**;
3. the missing words are absent from all 8 beam hypotheses — 22 of 741 (3%) exist
   anywhere in the beam.

What it forecloses, per [`2026-08-16-asr-landscape-2026.md`](2026-08-16-asr-landscape-2026.md)
§Γ, which lists these explicitly "so they are not proposed again": every
VAD-coverage / forced-alignment-gap guard (WhisperX, `ctc-forced-aligner`, MMS-FA) —
"the method looks for gaps; we have no gaps"; coverage/length/EOS penalties in beam
search — "reranking does not produce tokens"; every LLM GER or n-best rescoring over
*our* n-best (HyPoradise, Whispering-LLaMA, RobustGER, ClozeGER); and every published
Whisper hallucination detector, which detect extra text, not missing spans.

So the honest picture: deletions are closed at serving time and nearly closed on W.
The remaining mass is substitutions and insertions on the fused text, and it sits in
columns where the three systems **agree, or agree 2-of-1, and are wrong together**
([`2026-08-16-char-vote-homophones.md`](2026-08-16-char-vote-homophones.md)).

---

## A. Ranked — can produce a number before 2026-08-23

| # | Direction | Evidence it might pay | Why not done | Cost | Cheapest falsifier |
|---|---|---|---|---|---|
| A1 | Re-measure arm E (phonetic roster repair) on top of **W** instead of V | −0.00083 [−0.00119,−0.00049] on V, both rate gates pass to the digit (`exp-2026-08-16-roster-grounded-selection`). E only moves substitutions, and W's sub_rate is 0.0427 vs V's 0.0512 | Never proposed. W landed the same day and displaced V; `CURRENT.md` item 2 already says the number "does not transfer to it unexamined" | CPU-only, ~2–4 h | The measurement *is* the falsifier. If the paired CI on W includes zero, the only measured positive this project has stops being one |
| A2 | Audit roster availability, then decide whether arm E can ship at all | **New, measured here: 56 of 260 benchmark windows (21.5%), 43 of 203 meetings, have a per-meeting roster.** Arm E is a structural no-op on the other 78.5%. It fired 0 times on all 7 holdout meetings for exactly this reason (serving-stack ladder) | Never proposed. "Roster coverage" appears nowhere in the ledger | ~1 h + a read against the OpenCouncil API | Count meetings the production endpoint can fetch a roster for. If it is ~20%, the Βήμα-3 gate ("≥300 activatable points on untouched meetings") is unreachable and E ships as shadow-only by definition |
| A3 | Decompose the SCHOINA failure: coverage vs **utilisation** vs marginal discovery | SCHOINA is in Chania's frozen list *including the inflected alias humans corrected* — verified in `research/ds_wer/terms/chania.json` — and was still corrected 11 times by hand. 26 terms fell to "already present" carrying ≥2 human corrections each | Deliberately redirected: `exp-2026-08-16-error-mined-terms.next_action` says establish this *before* admitting any of the 147 mined terms. Nobody followed the redirect. The report names the three quantities and says "**δεν έχουν μετρηθεί εδώ**" | CPU-only, ~3–4 h, from `data/asr/export.jsonl` + cached hypotheses | Take the 26 already-present high-correction terms; count how often each appears correctly in the adapter's own output on windows where the reference has it. If utilisation is already high, the 147-term queue and half of arm E's premise are worth less than assumed |
| A4 | Measure the frozen config against the config production actually serves | `OC_ASR_BEAM=2` and `word_timestamps=True` in the endpoint; every decode conclusion is on beam 5, `word_timestamps=False` — [`2026-08-12-decode-ablation.md`](2026-08-12-decode-ablation.md) says so and says nothing transfers. Independently, `exp-2026-08-16-adapter-confidence` measured that asking for word timestamps changes the transcript in **101 of 102 paired windows at a pooled 7.7%**, with 82 windows moving and absolute deltas summing to 613 against a net −5 | Never proposed. The deliverable is a serving harness, and no experiment has scored the thing it serves | CPU decode of 39 frozen windows, ~2–3 h wall | Decode the 39 windows at beam 2 / `wt=True` and score against the frozen config. If WER moves less than the 0.69 MDE for a decode ablation (`exp-2026-08-16-harness-mde`), the mismatch is cosmetic and this closes |
| A5 | Taxonomy of the 1,245 `exact_2_of_3` columns where W ≠ column oracle | The class holds **25.0%** of the W→oracle gap, twice every unresolved column combined (char-vote report). **New, measured here on the cached alignment:** the oracle takes the lone dissenter in 1,245/1,245, never epsilon, never a novel token; 808 (64.9%) are within char-distance ≤2; 104 strict + 108 loose homophones; 72 have a single-character majority token | Never proposed as a taxonomy. Three *blind override* arms were run and all came back worse with CIs on the wrong side in all six search cities (autoresearch). Nobody asked what the errors are | CPU-only, ~3 h. All inputs cached: `~/.cache/oc-public/fusion_lab/align_65b1c4d64618a429.json` (9.3 MB), `bench_2026-08-10-*.json` (9.8 MB). No API, no GPU | Build the breakdown. If no subclass exceeds ~15% of the 1,245 with a rule expressible without audio, the class is confirmed unreachable by text and this whole line closes for good — which is itself the deliverable |
| A6 | An **insertion-side** arm family through the autoresearch harness | Insertions carry 2.43 pts of oracle headroom (table above) — the largest single component and never attacked directly. All 5 confirmations remain unspent; the harness enforces the search/confirm split | Partly tried, honestly: `two_present_oov_drop` moved insertions 0.03486→0.03304 on the search partition but was **+0.00216 worse overall**, and `singleton_oov_drop` never fired. Recorded in `research/autoresearch/journal.jsonl` as "a hypothesis to beat, not a finding" | CPU-only, ~4 h | Run 4–6 occupancy-drop variants through `autoresearch.py`. One search screen kills them all in an afternoon |
| A7 | Obtain a Scribe v2 credential and run W on the gold-set audio | The gold set's headline question — "are the words the fusion recovers real?" — is unanswered. `exp-2026-08-16-gold-set` conclusion opens with "**THE HEADLINE QUESTION WAS NOT ANSWERED**"; the adapter has fidelity-WER 0.284 [0.169, 0.455] and W has none | **Blocked on a credential.** No ElevenLabs key in `.env`, in the shell environment, or in the OpenCouncil repos (I re-verified: `.env` holds only `BENCH_API_KEY`, `MDC_API_KEY`). The benchmark API only runs on its own windows and will not take our audio | Human hours to get a key; then a few dollars of Scribe API. Zero GPU | Nothing to falsify — it is a blocked measurement, not a hypothesis. Either the key arrives before 23/8 or this moves to §C |
| A8 | The ~$0.82 `stt-async-v5` run, for a different question | `exp-2026-08-16-soniox-confidence.next_action`: "it would need a different question." One exists and only one: W's Soniox voter *is* the cached paid async-v5 transcript, so confidences from that path would attach to W with **no substrate swap** — the exact gate that killed both `w-rt-confidence` (free rt-v4 is a different model) and `adapter-confidence` (word timestamps change the decode) | Deliberately not spent. `exp-2026-08-16-w-rt-confidence` decision: do not spend it on the fusion-arm hypothesis | Paid API, ~$0.82 | **My honest read: the gain here is already measured and it is near zero.** Confidence AUROC on the decisions a fusion arm actually makes is 0.587–0.703, weakest where the mass is, and the arms with the most room fitted "never fire" in all ten LOO folds. The $0.82 buys an *attachment* check, not a gain. Rank it low and spend it only if A6 turns up an arm that needs a confidence |
| A9 | Three state fixes, not experiments | (a) The autoresearch record and `results_autoresearch.json` say **11 registered / 11 searched**; `research/autoresearch/journal.jsonl` holds **16 / 16** — the five overlap-speaker arms were appended later. The BH fishing diagnostic is reported against the wrong denominator. (b) `train_runpod.py` does not consume the `weight` field, so the harmonized-gate claim "edge rows kept at weight 0.5" describes something the trainer does not do ([`2026-08-15-external-sources-probe.md`](2026-08-15-external-sources-probe.md)). (c) `2026-08-20-final-report.md` is dated 2026-08-12 in its own first lines | Never noticed | <1 h | n/a |

### Notes on the ranking

**Why A1 is first.** It is the only measured positive this project owns that has never
been deployed, and the number it rests on was invalidated the same day it was
measured — not by a contrary result but by a substrate change. `restricted_repair()`
in `eval/controlled_eval/exp_roster_selection.py` is a text-in/text-out function and
W's composed text is already produced by `exp_composition.py`, so this is an
afternoon. One design question needs answering first, not after: E's firing rule in
that experiment was "acts only where the three systems disagree", and W's output is a
text **none of the three produced**, so "disagree" has to be re-specified against
columns rather than against token sets. Freeze that choice before seeing a number.

**Why A6 is ranked mid, not high.** The instruction was not to rank another text-only
arbitration idea highly without saying why it differs. It differs on one axis and I
want to be precise about it: the char vote, the LLM arbiter and the homophone census
were all **which-word** arbitration, and the structural finding that killed them is
specifically about which-word ("only 1,396 unresolved columns have W differing from
the column oracle → at most ~35% of the gap"). Occupancy — *should there be a word
here at all* — is a different decision and carries the larger headroom. But it is
only *partly* virgin: two occupancy-drop ideas already went through the harness and
neither worked. That is why it sits at 6 and not at 2.

**What A5 buys, being clear.** It produces a diagnosis, not a shipped arm. This
project's rule is to prefer directions that produce a number, and a category
breakdown with counts *is* a number — it is what tells you whether the 25% class has
any separable structure or whether three failed override arms plus this taxonomy
close the door permanently. The 64.9%-within-2-characters figure I measured suggests
these are near-miss spellings rather than semantic errors, which is a testable
handle. The limit is stated in the char-vote report and I will not soften it: a share
of the 1,245 are *reference* errors, not model errors, and the substrate cannot
separate them. Only audio can, and the window `.wav` files are on disk
(`~/.cache/oc-public/bench_windows/`, 261 files, 1.2 GB) if someone wants to
adjudicate a sample by ear.

---

## B. Finished — there is nothing here

Saying so is part of the answer. Each of these cost real time and each is closed on
evidence, not on fatigue.

- **Decode-time thresholds.** `no_speech_threshold` fires zero times on 39 windows;
  removing the temperature fallback makes every metric worse. Two arms came out
  byte-identical to control.
- **Text-only whole-window selection by an LLM.** Loses a full WER point, picks
  Scribe on 215 of 244 windows and the shorter text in 103 of 129 deviations. The
  mechanism (a reader who cannot hear prefers the transcript that swallowed the hard
  passage) is a testable explanation, not a measured cause — but the deletion gate
  catches it every time.
- **Per-character vote and homophone arbitration.** 34 strict-homophone columns out
  of 80,659 (0.042%), seven times below the standing 0.3% do-not-build line; the
  char vote fired on 136 columns for −0.00008 with the CI including zero.
- **Overriding a 2-of-3 majority.** Failed four separate times on this substrate,
  the last time with a confidence signal the previous three did not have.
- **W+len and W+D.** Both fail the insertion gate. W is *supposed* to be shorter.
- **Target-speaker extraction.** All four synthetic gates passed; on real overlap
  1 of 6 speakers was enrollable and no reference word was recovered.
- **Decode-time hotword biasing.** Lifts name recall 51→65% and costs +0.34 WER in
  deletions; fails its preregistered gate. (Three cheap knobs inside it were never
  tried — see §C.)
- **Label purity, mixture ratio, data scale.** +0.0015 against a 2.1-point per-seed
  spread; closed by user instruction; ~1300 h buys ~0.5 points.
- **Attaching our own adapter's per-word confidence to W.** The signal is real
  (AUROC 0.8151) and the gate is a conjunction that already failed. More windows
  cannot reverse it.

---

## C. After GSoC

Real directions. None can produce a defensible number by 2026-08-23.

**The training levers nobody has touched.**
[`2026-08-11-training-brief.md`](2026-08-11-training-brief.md) §7 is the canonical
never-tried list: larger LoRA rank, unfreezing the encoder, targeting beyond
`q_proj`/`v_proj`, more than 2 epochs after the label fix, cleaning the 15,038
`no_edit` labels, more Greek data. Separately,
`docs/specs/window-shape-preregistration.md`: "**no run of this project has ever
varied** the window shape" — and that spec has no ledger record at all. All of it is
gated the same way: `exp-2026-08-16-deletion-hard-coverage` established neither
proof-to-reopen gate, so the training freeze holds; and the advisory prices a frozen-
protocol confirmation at 3 seeds × 2 arms = **$36–40 against ~$9 remaining**. Money
is the binding constraint, not conviction.

**HParl / CLARIN — the legal blocker, stated exactly.** The problem is not the
parliamentary proceedings, which are freely usable under Greek copyright law
n.2121/1993 art. 2(5) with commercial use open. The problem is the **CLARIN/ILSP
compilation ID 1602**, distributed under **CC BY-NC 4.0** — non-commercial — and the
HF mirror contradicts itself (YAML `cc-by-4.0`, README body `CC BY-NC 4.0`). The
licence was never verified at source; the prereg carries an unticked
`[ ] Licence resolved at source`. What clearing it would unblock: `artifact-pack-hparl2-v2`
is **13.16 h built** (9,334 rows, sha256 `29f4fe806b88e80a`), and the calibrated
gate extrapolates to **~117 h of the ~120 h corpus** if the full filter pass runs
(~12 h of ASR at 16 workers). Expected value against the measured ~1300h → 0.5-point
curve: small. And note the honest ordering — **even with the licence cleared, the
training freeze and the $9 stop this**. The licence is not currently the binding
constraint. If it never clears, `2026-08-16-asr-landscape` says the answer is to
switch parliament sources (EuroSpeech reaches the same speech), not to abandon the
domain.

**External models named, costed and never run.** `ilsp/VoxKrikri-21-full` (HParl
13.01 vs 16.99 base; ~24 GB VRAM; **Apache-2.0 on HF vs Llama 3.1 Community in the
paper — resolve before anything product-facing**); `nvidia/canary-1b-v2` (FLEURS el
9.21 vs whisper-large-v3 27.03 — zero-shot only, and NeMo has no official LoRA recipe
for Canary/Parakeet); CopyNE and TCPGen (public code, NE-CER −50–55%, **zero ledger
mentions**); DeRAGEC (public code, 28% relative); CTC-WS / TurboBias acoustic-level
biasing, which is the actual explanation for why the failed arm B failed. Every one
needs GPU.

**Two structural gaps the project scored itself on and never closed.**
[`2026-08-11-vertical-domain-practice.md`](2026-08-11-vertical-domain-practice.md):
stage 2, "199 hours / 138 meetings / 13 cities were downloaded — **the test set was
never built** ⚠ half-finished, this is the barrier"; stage 3, curation of the 15,038
`no_edit` rows "**not done** ❌ the biggest gap", with the boundary audit finding
7 of 10 defective in a stratified sample. The report notes these are exactly the two
the literature scores highest per unit of effort.

**The conversion ladder.** [`2026-08-09-longform-preflight.md`](2026-08-09-longform-preflight.md)
specifies a discriminating test (HF base → HF base + live LoRA → HF merged → CT2 fp16
→ CT2 int8 over 30–50 fixed 30 s chunks) and then says "**Δεν έγινε**". Until it
runs, the long-form finding is attributed to training format "as the most likely
cause, not the proven one". Related and unexplained since July: the decoder stack
matters enormously for the fine-tune (22.66 faster-whisper vs 28.18 HF generate) and
barely at all for base (27.08 vs 28.25) — an asymmetry the report says "deserves
attention **before any number from either harness is trusted**".

**One publishable asset.** The landscape scan looked specifically and found that no
named-entity accuracy comparison of commercial vs open ASR exists for Greek. Our
DS-WER is one of the few such measurements anywhere. Publishing it is writing, not
experimentation — but it inherits two caveats: the metric was **defined after** the
error analysis, and the term list is a lower bound (roster surnames plus two
municipality names, no communities, settlements or acronyms).

**Dataset release.** Blocked by the DPO legal hold of 2026-07-17. `docs/specs/2026-08-16-link-only-dataset-proposal.md`
names the concrete work: Art. 14 subject notification, a DPIA, contract amendments.
Legal, not technical.

---

## D. Loose ends found, not fixed here

- The advisory's oracle decomposition (§ top) is not in the ledger and it displaces
  the operative reading of "the deletions live in the weights". That standing finding
  should carry a scope note: *pre-fusion, about `artifact-adapter-fixed` alone*.
- GitHub **#1** ("Exploration UI for human review") is OPEN, pre-dates the wayfinder
  labels, and has **no corresponding ledger record**. Its checklist was overtaken by
  the corrections mining, the deletion-hard build and the auto-verifier bulk accepts.
  It should be closed or given a record.
- **#6** (arm E deployment) still says "only the installation is missing" and has not
  been updated with the V→W substrate change. A2's roster number belongs in it too.
- CodeRabbit's free allowance for this repository is exhausted (3 of 3 used), so the
  post-change review step in `CLAUDE.md` is currently unexecutable. Codex stood in
  once (job `59c9564`).
- The 2026-08-16 advisory's two full model texts were never written down; only the
  summary survives.
- `~/.cache/oc-public/train-screens-2026-08/run2-eval-stage1/` holds output from a
  decode that was still running when its report was written, and no ledger record
  points at that path.
- `soniox-tools` at `/home/harold/projects/soniox-tools` is not a git repository, so
  the tooling `exp-2026-08-16-w-rt-confidence` depends on has no citable revision.
- EuroSpeech shards 000–399 pass the harmonized gate at 97–99% and shards 400+ at
  24%. A regime change, cause unresolved.
