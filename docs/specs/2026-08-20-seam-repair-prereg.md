# Seam repair on arm P — preregistration

Status: FROZEN 2026-08-20, before any arm was decoded
Experiment: `exp-2026-08-20-seam-repair` (OPEN)
Builds on, and does not edit, the frozen
[2026-08-17 chunking-aware decoding preregistration](2026-08-17-chunking-aware-decoding-prereg.md).

## Everything here is EXPLORATORY

The 39 frozen validation windows **generated** the hypothesis this spec tests. The
forced-cut count that motivates arm F was read out of `eval-P.json`, which is a
decode of those same 39 windows. Re-running on them is therefore **not independent
confirmation**, and no number of bootstrap replicates makes it one. Every result
below is exploratory and must be labelled exploratory wherever it is quoted.
Confirmation requires the 247-window benchmark or fresh windows, on the same stack,
in a separate preregistration.

**Sealed data stays sealed.** `research/eval-freeze-2026-08/manifest.json` holds
39 `eval_windows` and **7** `holdout_windows` (a temporal holdout, 5 meetings, 2,101
reference tokens); separately, the project's 16 locked benchmark evaluation windows
are sealed by `CLAUDE.md`. **Neither set is touched here** and no arm below may be
evaluated on either. (The "16 sealed temporal holdout" phrasing in the task brief
conflates the two; the temporal holdout is 7 windows.)

## What is already known

Arm P (pre-split each window at detected silence, decode pieces independently,
concatenate) is the project's best decoding result on the 39 frozen windows with
`artifact-ct2-fixed`, local CPU int8:

| | WER | delta vs A | CI95 | (S+D)/N |
|---|---|---|---|---|
| A (control) | 0.15893 | — | — | 0.13878 |
| P | 0.14751 | -0.01142 | [-0.02025, -0.00195] | 0.12795 |

Unlike arm V, this is real recovery: substitutions fell 938 -> 860 and (S+D)/N fell.

**But P cuts blind where speech does not stop.** `split_accumulating` accumulates to
a 29.5 s deadline and cuts at the last VAD silence of >= 0.5 s that leaves a legal
piece. When no such silence exists it takes a *forced cut* at the deadline — exactly
the blind boundary the control has. Counted from `eval-P.json`:

**16 forced cuts across 14 of the 39 windows.**

Arm E (overlapping windows, keep central tiles) attacked the same seam and **failed
badly**: WER 0.31081, insertions 0.02015 -> 0.13425. The cause is documented: with
`word_timestamps=false` the smallest timed unit is a whole segment, so two
neighbouring windows transcribe the same speech with different wording and
centre-ownership keeps both.

## Arms

All arms: `artifact-ct2-fixed` (sha256[:16] `8a1a3b257d0c1bdb`), local CPU int8,
`DEVICE=cpu COMPUTE=int8 THREADS=16`, the 39 frozen evaluation windows, the frozen
evaluation normalizer (`ftoks`) and alignment (`sdi`), common random numbers
(`DA.seed_for("A", window_id)`). Everything not named below stays at the frozen
CONTROL config.

### Arm F — repair the forced cut, do not cut blind

Identical to arm P in every respect except the branch that today emits a forced cut.
When the splitter finds no VAD silence of >= 0.5 s in `[cursor + 5.0, deadline]`:

1. Take Silero's **frame-level speech probabilities** from
   `faster_whisper.vad.get_vad_model()(padded_audio)` — one value per 512 samples,
   a 32 ms hop at 16 kHz. This is the primary signal.
2. Search **only the slack region** `[deadline - 2.0, deadline]`. No global search:
   the global RMS minimum over the whole legal range can land at 5 s and shred the
   piece, and low energy is not a phoneme boundary — unvoiced consonants are quiet.
3. A **valley** is a maximal run of >= 4 consecutive frames (>= 128 ms, satisfying
   the >= 100 ms floor) with speech probability `< 0.35`. That threshold is Silero's
   own `neg_threshold` default (`threshold - 0.15` at `threshold=0.5`), i.e. the
   hysteresis floor the library already treats as "definitely not speech". This
   mirrors Silero's `max_speech_duration_s` strategy, which cuts at the last silence
   of more than 100 ms before the ceiling.
4. Among valleys **entirely inside** the slack region, pick the one with the lowest
   mean speech probability. Ties within 1e-6 break to the lower mean RMS energy —
   energy is a **tie-break only**. Remaining ties break to the later valley.
5. Cut at the valley midpoint, clamped into `[cursor + 5.0, deadline]`.
6. If no valley qualifies, cut blind at the deadline and count it as `blind_cuts`.

Frozen constants: `SLACK_SECONDS = 2.0`, `VALLEY_PROB_THRESHOLD = 0.35`,
`MIN_VALLEY_FRAMES = 4`, hop 512 samples. No other change to the splitter, and the
non-forced path is byte-identical to P's.

### Arm C — chain context across the seam

Arm P's segmentation **unchanged**. Piece 1 decodes exactly as in P. For each piece
k >= 2, pass `initial_prompt` = the tail of the concatenated text of pieces 1..k-1,
capped at **64 tokens** measured with the model's own tokenizer
(`model.hf_tokenizer`, `tokenizer.json` shipped inside `artifact-ct2-fixed`), not at
a character count — Greek characters cut words in half.

`condition_on_previous_text` stays `False`, as frozen. This is not a contradiction:
each P piece is at most 29.5 s, so a piece is exactly **one** Whisper window, and
faster-whisper 1.2.1 puts `initial_prompt` into `all_tokens` *before* the first
window and only sets `prompt_reset_since = len(all_tokens)` *after* that window is
decoded. The prompt therefore reaches the decoder as `sot_prev` context on the only
window that exists. Verified by reading `WhisperModel.generate_segments`; asserted
by instrumentation in the pilot (below).

Known risk, recorded rather than dismissed: a prompt can induce repetition looping
and hallucinated continuation. The insertion rate is the place that shows up.

### Arm W — overlapping windows with WORD-level ownership

The geometry of arm E, unchanged: 30 s decoder windows, 15 s stride, each window's
central 15 s tile owned exactly once, tiles covering `[0, duration]` exactly.
One declared config change: **`word_timestamps=True`**.

Ownership is decided **per word, by that word's own timestamp**, not by its
segment's midpoint. A word belongs to the window whose kept tile contains the word's
midpoint; a word is emitted only by the window that owns it. Because the tiles are
an exact cover, this yields **exactly one owner per word by construction**. This is
what whisperX and the HF ASR pipeline do, and it is the ingredient E lacked.

Per-word ownership is **not** a guaranteed-correct merge and this spec does not
claim it is. Two decodes of the same boundary word can carry different timestamps
and land on opposite sides of a tile edge, producing either a duplicate or a hole at
that one word. The seam-disagreement count is reported as a diagnostic, not
suppressed.

### Arm E-WT — derived free from W's decodes

Same audio, same decodes, same word timestamps, but merged with the **old
segment-level rule** (`merge_overlap_segments`, nearest-centre by segment midpoint).
Costs no extra decoding: W's per-window raw segments are cached and merged twice.

Without E-WT we cannot tell whether W wins because the merge got finer or merely
because word timestamps were on — turning them on changes faster-whisper's own
segmentation and timestamp attribution, so it is a confound in its own right.

## Pilot: 5 windows, judging MECHANISM only

Pilot windows are the 5 with the most forced cuts in `eval-P.json`, ties broken by
window id ascending — read from the cache, not guessed:

1. `win_orestiada_dec11_2025_600052` (2 forced cuts)
2. `win_orestiada_feb11_2026_149776` (2)
3. `win_argos_apr7_2026_960810` (1)
4. `win_argos_aug29_2025_2731295` (1)
5. `win_argos_dec23_2025_581922` (1)

**The pilot's pass/fail is whether each arm does what it claims. It is NOT a WER
screen.** Deciding whether to continue by peeking at pilot WER would be optional
stopping and would poison the 39-window numbers. Pilot WER may be computed and
reported, but it is **description only** and gates nothing.

| arm | PASS requires |
|---|---|
| **F** | every piece in `[5.0, 29.5]` s; `speech_dropped == 0`; pieces tile the window exactly; every repaired cut sits inside a recorded valley of >= 100 ms with mean speech probability < 0.35; and `blind_cuts` < P's forced-cut count on the same windows (the mechanism fired at least once) |
| **C** | for every piece k >= 2 the instrumented `model.get_prompt` receives `previous_tokens` equal to the recorded prompt tokens, `1 <= len <= 64`; for piece 1 it receives `[]`; no decode raises |
| **W** | the emitted words are an exact partition of the candidates: every candidate word is either kept by exactly one window or dropped as a non-owner, kept + dropped == total, no word emitted twice, every kept word's midpoint lies in its own window's kept tile |

An arm that fails its mechanism check does **not** proceed to the 39 windows; the
failure is reported as the result.

## Scoring on the 39 windows

Identical to arms V, P and E so the numbers sit in the same table:

- control is the existing `decode-ablation/eval-A.json` cache;
- tokens by `eval.controlled_eval.eval_freeze.ftoks`;
- alignment by `eval.controlled_eval.exp_same_stack.sdi`;
- meeting-clustered paired bootstrap, 4000 replicates, seed 7, blocks `meeting_id`;
- reported per arm: WER and its CI95, deletion / insertion / substitution rates,
  (S+D)/N, emitted tokens, and leave-one-window-out sign stability.

The four frozen 2026-08-17 gate conditions are computed and printed for continuity
with the existing table. **They do not decide adoption here**, because this
experiment is exploratory by construction (see the top of this spec). Adoption of
any arm alongside P requires a confirmation run outside these 39 windows.

Arms are compared **against control A**, as P was. A descriptive comparison against
P is also reported; it is not the estimand.

## What this cannot show

The 39-window harness measures **agreement with OpenCouncil's published text**, not
fidelity to audio. An arm that recovers real speech the published text omits is
scored as an insertion and punished (`exp-2026-08-17-insertion-fidelity`: 23.7% of
scored insertions sit on words a human heard). A passing arm is credible; a failing
arm is ambiguous.

Single decode per arm. No training, no seeds, no GPU, no money, no external API.
Audio and hypotheses stay under `~/.cache/oc-public/`; only aggregates enter git.

## Cost

Three decoded arms (F, C, W) over 5 pilot windows, then over 39 windows, on local
CPU. E-WT is free. Prior arms took roughly 4.5 hours for three arms over 39 windows.

---

## Amendment 1, 2026-08-20 — Codex plan review, before any arm was decoded

Reviewed by `gpt-5.6-sol` (codex-bridge job `7d83b0bc42cd4bd793f8de6987ed99a7`,
`model_reasoning_effort=high`) against the plan above. No arm had been decoded and
no new code had been written when this was folded in. The corrections adopted are
binding; the declined ones are named with a reason, not silently dropped.

### Adopted — arm F

1. **Clip valleys to the slack region, do not reject those that cross its edge.**
   Restrict the frame array to `[deadline - 2.0, deadline]` **first**, then find
   maximal runs inside it. Finding runs globally and requiring containment throws
   away a long valley that merely straddles `deadline - 2.0`.
2. **Every final boundary carries a `cut_kind`**, one of `vad_silence`,
   `probability_valley`, or `blind`, plus its integer sample coordinate, and the
   counters are recomputed **from the final piece layout**. The existing short-tail
   repair can move or delete a boundary and can itself invent a midpoint cut; a
   diagnostic taken before that repair can pass a broken implementation.
3. **Coverage is asserted at integer samples**, not by summing floats: the piece
   slices must concatenate to exactly `audio[0:n]`, every sample covered once.
4. **VAD input contract**: pad to a multiple of 512, flatten the returned array,
   and assert all probabilities are finite.
5. **Two overclaims removed.** F is *not* a reproduction of Silero's
   `max_speech_duration_s` split: Silero takes the *latest* qualifying silence, F
   takes the *lowest-mean* valley and cuts at its midpoint; and four 32 ms frames
   span 128 ms of duration but only 96 ms of first-to-last frame starts, so the
   ">= 100 ms" claim is about run duration, which is what F measures. F is a
   related heuristic, not the same rule. Likewise, "the non-forced path is
   byte-identical to P" holds only up to the **first** repaired boundary; moving
   that cursor changes every later deadline and can change later VAD choices.
6. **`min_speech` is a preference, not a requirement** in the existing splitter: if
   no candidate clears 2.5 s of voiced audio it still takes the latest in-range
   silence. That is P's behaviour and F inherits it unchanged.

### Adopted — arm C

7. **Pass token IDs, not a string.** `initial_prompt` receives
   `model.hf_tokenizer.encode(text, add_special_tokens=False).ids` truncated to the
   last 64. A string prompt is stripped, space-prefixed and *re-encoded* by
   faster-whisper, so a character-free cap measured on our side would not be the cap
   the decoder sees.
8. **A <= 29.5 s input is not guaranteed to be one decoder iteration.** Timestamp
   splitting can advance `seek` short of the input end, and word-timestamp mode can
   move it again. The pilot oracle is therefore: piece 1 — *every* `get_prompt` call
   receives `[]`; piece k >= 2 — the **first** call receives exactly the recorded
   prompt IDs and every later call receives `[]`; and the constructed prompt begins
   with `sot_prev`, contains the prompt IDs, then `sot_sequence`.
9. **An empty prompt is a correct prompt.** If every preceding piece emitted no
   text the right length is 0. The criterion is `0 <= len <= 64`, and `len == 0` is
   only legal when the preceding text is empty.
10. **Whole-word truncation.** Take the longest suffix of complete whitespace-
    delimited words whose tokenization is at most 64 tokens, so the cap never begins
    mid-word — the same failure the token cap was introduced to avoid.

### Adopted — arm W

11. **The original W pilot criterion was not red-capable**: an implementation that
    drops every candidate satisfies "kept + dropped == total, nothing twice, every
    kept midpoint in its tile" vacuously. The oracle is strengthened to a two-way
    equivalence: a candidate is kept **if and only if** its own window owns its
    midpoint, and `kept > 0`.
12. **Frozen edge convention.** Word timestamps are rounded to centiseconds, so
    exact tile edges will occur. Tiles are `[0, 15]`, `(15, 30]`, `(30, 45]`, ... —
    the earlier window wins a tie, preserving arm E's rule.
13. **Stable candidate identity and order.** Every candidate carries
    `(window_index, segment_index, word_index)`; emitted words sort by
    `(start, window_index, segment_index, word_index)`. Sorting equal timestamps by
    *text* can reorder words.
14. **Three separate drop counts** — `non_owner`, `outside_audio`, `kept` — so
    hallucinations in the zero-padded region of the first and last windows are not
    hidden inside the seam-duplicate count.
15. **`seam_disagreement` gets a frozen definition** (it was previously named but
    undefined): at each tile edge, take the raw candidate words of both adjacent
    windows whose midpoints fall within +/- 1.0 s of the edge, normalize with
    `ftoks`, and report the multiset symmetric difference plus the count retained
    0, 1 and 2 times. Candidate conservation alone cannot see a hole or a duplicate.
16. **Word timestamps are not passive metadata** — they run attention alignment,
    alter segment boundaries and can change `seek`. So the three contrasts mean:
    `W - E-WT` = word-level vs segment-level merge, holding the decode fixed;
    `E-WT - E` = the total effect of enabling word timestamps under one merge rule;
    neither is a clean "timestamps only" contrast.

### Adopted — measurement

17. **Report the incremental contrast against P, not only against A.** F and C sit
    on top of P, so an A-relative number mostly re-measures P's known benefit.
    Primary descriptive contrasts: `F - P`, `C - P`, `W - P`, plus `W - E-WT` and
    `E-WT - E`. A-relative numbers stay, for continuity with the existing table.
18. **Cache identity must cover the algorithm, not just the config.** F, C and P
    share the same static decode config, so today's identity check
    (model / config / environment) would silently accept a cache written by an
    older splitter. Every new arm's cache carries an `algorithm_manifest_sha256`
    over the arm's constants and merge rules, and a mismatch is refused.
19. **Leave-one-meeting-out is added** next to leave-one-window-out, since the
    bootstrap blocks on `meeting_id`. F touches few clusters, so its interval will
    be discrete and window-level LOO alone would misrepresent stability.
20. **The holdout provenance in this spec was wrong** and is corrected above:
    7 temporal-holdout windows in the freeze manifest, 16 locked benchmark windows
    in `CLAUDE.md`. Neither is touched.
21. **Pilot WER is not computed at all.** Calling it "description only" does not
    stop it influencing which bug gets fixed or how a threshold is read. The pilot
    emits mechanism diagnostics and nothing else.

### Declined, with reasons

- **Re-decode arm A under a repaired text-assembly convention.** Codex found, on
  the cached P decode, that `combine_piece_transcripts` joins segment strings with
  no separator, and that per-segment `strip` + space-join changes tokenization in
  7 of 39 windows, recovering seven fused tokens. That is a real defect and it is
  recorded as its own open question. It is **not** fixed here: changing the
  assembly convention would make every number in this spec incomparable with the
  published V / P / E table, which is exactly the table the user asked these arms
  to sit inside. Mitigation for the asymmetry Codex raises — that W assembles from
  words while P assembles from segments — is test-enforced instead: W joins the
  word strings faster-whisper emits (which carry their own leading spaces), and a
  test asserts that joining a segment's words reproduces that segment's own text
  token-for-token under `ftoks` before ownership filtering.
- **Reset an arm-neutral seed immediately before each piece.** This would decouple
  the random stream from the number of `transcribe` calls, which is right in
  principle — but P's cache was produced with **one** `ctranslate2.set_random_seed`
  per window before the piece loop, and changing the scheme would make F and C
  non-comparable to the cached P they are built on. F and C keep P's exact seeding.
  Recorded as a caveat: with temperature fallback, differing piece counts consume
  different random streams, so "common random numbers" is weaker than the phrase
  suggests.
- **Repeat runs to quantify decoder nondeterminism.** Correct in principle and out
  of budget: it doubles a multi-hour CPU pass for an exploratory result. Recorded
  as a caveat — the bootstrap covers window/meeting sampling, not decoder
  nondeterminism.
- **The full proposed test list.** The high-value subset is implemented: F's frame
  and tie-break edges, the F/P differential on no-forced-cut fixtures, integer
  tiling, C's Greek/empty/64/65-token tokenizer cases and the `get_prompt` oracle,
  W's `14.99` / `15.00` / `15.01` edges, the W two-window duplicate/hole oracle, the
  raw-W word-join invariant, and the algorithm-manifest cache rejection. The
  remainder is deferred as scope.

---

## Amendment 2, 2026-08-20 — scope cut to the pilot, by the user

The 39-window decode of arms F, C and W is **not run**. The user capped this work at
the small test: the 5 pilot windows, all three arms in one pass.

Consequences, stated so no later reader mistakes what exists:

- There are **no 39-window numbers** for F, C or W, and therefore no row for them in
  the V / P / E table. That run remains the documented next action.
- The mechanism verdicts stand as the result of this spec: they were always the
  pilot's job and they do not depend on the 39-window pass.
- Amendment 1 item 21 ("pilot WER is not computed at all") existed to stop optional
  stopping from poisoning the 39-window numbers. With that run cancelled there are
  no numbers left to poison, so the 5-window rates ARE reported — as **description
  of five windows and four meetings**, with no bootstrap interval quoted as
  inferential, no gate evaluated, and no arm adopted on their basis. They are not a
  small version of the 39-window answer; they are five windows chosen precisely
  because they are unrepresentative (they are the ones where P cuts blind).

---

## Amendment 3, 2026-08-20 — CodeRabbit review, pilot re-run under the fixed code

`coderabbit --agent` on the new files raised three findings; all three are fixed and
the pilot was **decoded again from scratch** under the corrected code rather than
being blessed retrospectively.

1. *(major)* `_w_verdict` rebuilt the overlap layout and `zip`ped it against the
   cached windows with no length or order check, so a cache from a different layout
   would have been silently truncated instead of failing. It now records a failure
   and stops, matching `derive_e_wt`.
2. *(minor)* Arm C accumulated carried text as `carried + text`. Whisper segment
   text normally begins with a space, but where it does not, the last word of piece
   k-1 and the first of piece k fuse into one "word" for the whitespace split that
   `prompt_ids_for` uses — the very mid-word truncation the token cap exists to
   prevent. The carry now joins with an explicit separator.
3. *(minor)* A W ownership test carried an `or not inside` escape clause that let
   out-of-audio candidates satisfy the assertion either way. Removed.

Because (2) changes arm C's behaviour, the algorithm manifest gained
`prompt_carry_join`. The manifest is now **per arm**, so correcting a rule only one
arm uses no longer invalidates the others' caches — but the manifests themselves
changed shape, so all three pilot caches were refused (the guard working as
intended) and all three arms were decoded again.

---

## Amendment 4, 2026-08-20 — arm X, all three mechanisms at once

Added at the user's request, after F, C and W each passed their mechanism check
separately and before arm X was decoded.

**Arm X = F + C + W in one decode.** The three mechanisms compose only under one
hard constraint: a decoder input must still be **one** Whisper window. A core piece
plus an overlap margin on each side has to stay under 30 s, so X cuts at
`X_MAX_CHUNK_SECONDS = 25.5` instead of P's 29.5 and spends the difference on
`X_MARGIN_SECONDS = 2.0` of context per side.

1. **F** chooses the boundary: VAD silence where one exists, otherwise the
   lowest-mean probability valley in the last 2 s, otherwise blind.
2. **C** passes the previous pieces' **core** text as a 64-token `initial_prompt`.
   The margin's words are deliberately excluded from the carry, so the prompt never
   contains a word that a neighbouring piece also emitted.
3. **W** resolves the overlap: each decoded word is owned by the **core** whose
   half-open span `[start, end)` contains its own midpoint (the final core's end is
   closed). The margin is context the decoder sees and never text it contributes.

X's segmentation is therefore **not** P's or F's — the ceiling differs — so X is not
a strict superset of F and a difference between them is not attributable to the
margin alone.

**Mechanism criteria (all must hold at once):** F's — legal core lengths, exact
integer-sample tiling, every repaired cut inside a real valley; C's — the recorded
prompt ids reach the decoder on the first `get_prompt` call of every piece k >= 2
and every later call is reset; W's — kept **iff** the emitting piece owns the
midpoint, candidates conserved, `kept > 0`.

The same exploratory status applies: 5 windows chosen because P cuts blind in them,
no gate, nothing adopted on their basis.

---

## Amendment 5, 2026-08-20 — arm X's criterion was wrong, and its margin was misspent

Arm X **failed** its first mechanism check (7 blind cuts against P's 7, valley repair
fired once in 8) while posting the best WER of any arm. Two measurements taken from
that decode, before anything was changed, explain both halves and neither is a
number this spec is allowed to optimise:

**1. A blind CORE cut is not a blind DECODE cut.** In F, a blind cut truncates the
decoder's input — that is the whole harm. In X, both neighbours hear across the seam
through their margins and word ownership then picks one side. Seam disagreement in
the first X decode, by boundary kind: **blind 0.551, silence 0.608, valley 0.600**.
Blind seams were the *least* disagreeing. So "zero blind cuts", inherited from F, is
measuring the wrong property for X.

The X criterion is therefore replaced — **not relaxed** — with what X actually
claims, and the replacement is red-capable (a test proves it fails on a seam
starved of context):

- every internal seam is decoded with at least `X_LOOKBACK_SECONDS` before the core
  and `X_LOOKAHEAD_SECONDS` after it; the first input starts at 0 and the last
  reaches the audio end;
- the lookahead is **used**, not merely supplied: pieces must actually transcribe
  past their own core, or the margin bought nothing;
- word ownership stays an exact if-and-only-if with `kept > 0`;
- F's and C's checks are unchanged and still apply.

**2. The margin was spent on the wrong side.** Words emitted into each margin, per
boundary: the left piece over-runs its core by **3.7–5.3 words**; the right piece
transcribes only **0.32–1.43 words** before its core start. A symmetric 2 + 2 margin
was paying 2 s of ceiling for a lookback that is almost never transcribed.

So the margin becomes asymmetric — `X_LOOKBACK_SECONDS = 0.5`,
`X_LOOKAHEAD_SECONDS = 2.0` — which buys the ceiling back:
`X_MAX_CHUNK_SECONDS` rises 25.5 → **27.0** (27.0 + 0.5 + 2.0 = 29.5, still one
window), and `X_SLACK_SECONDS` rises to 3.0. The valley **definition** is untouched:
still >= 128 ms below 0.35, Silero's own `neg_threshold`.

**Honest limit on this tuning.** The constants were chosen on the pilot windows' VAD
geometry — model-free, and no WER was consulted while choosing them — but they were
still chosen on these five windows, so the blind-cut counts they produce will not
transfer exactly to the 39. A model-free sweep over the same five windows showed
zero blind cuts is **unreachable** with the strict valley definition: the floor is
1 of 33 boundaries, because some speech genuinely has no 128 ms sub-0.35 gap in
three seconds. Requiring zero would have forced loosening the valley to Silero's
*speech* threshold, which was refused.
