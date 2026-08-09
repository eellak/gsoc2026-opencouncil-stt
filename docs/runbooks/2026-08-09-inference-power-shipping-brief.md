# On-box brief — inference, power, and the two shipping candidates (2026-08-09)

> **You are Claude Code running on the mini PC (`harold-venusseries`).** No prior
> conversation context — this file is your instruction set. Repo: `~/opencouncil-fine-tuning`.
> Tasks A–D below are independent. Run them in separate worktrees if running more than one.

## What is already settled — do not re-open

Read `CURRENT.md` and `docs/decisions/modeling.md` first. Short version:

- **The continuous-30-second-window retrain is CLOSED (2026-08-09) as an explanation of the
  deletions.** Two independent negative checks. Its motivating evidence (+1.54 deletion
  points) was a decoder artifact: same-stack it is +0.19, 90% [−0.50, +0.85]. The blinded
  listening test reversed the predicted mechanism. Do not propose it again without a new,
  independent motivation.
  **The condition was met later the same day, and clip shape is now Task E.** The new
  motivation does not involve deletions at all: the boundary audit found 4 of 20 uniformly
  sampled training clips cut or lose one of their own reference words, all at ≤1.06s, and
  25.8% of the corpus is that short. Clip shape has never been varied in any experiment
  here, so it is an uncontrolled factor under every result to date. Scope, metrics and
  frozen gates: [preregistration](../specs/window-shape-preregistration.md). Deletions are
  not a criterion in it.
- **The mixture ratio is not the lever.** 20/80 vs 50/50 indistinguishable. Seed spread
  (2.1 points) is ten times the effect.
- **The one surviving gain is substitutions, −0.61 points, CI excludes zero.** The
  fine-tune puts the right word in more often. That is the thing to build on.
- **The benchmark's "human reference" is the published OpenCouncil transcript** (WER
  0.0008 between them). Every WER against it measures agreement with our own output.
  The audio-faithful reference (`eval/controlled_eval/score_audio_faithful.py`, 32 dev
  windows + 16 locked) is the only honest scorer. Use it.
- **Standing rule:** never compare two models decoded by two different engines.

## The binding constraint

With 1,236 reference words, **no difference under ~4.5 WER points is visible.** Every
effect measured so far is under 1 point. More GPU runs cannot resolve anything at this
sample size. This is why Task B exists and why it outranks any retraining idea.

---

## Task A — Inference that bills only while in use

**Problem.** A meeting is transcribed for ~3 hours a month; an always-on VM bills for 720.
OpenCouncil currently runs on DigitalOcean. We need a serving shape that scales to zero.

**Already known (do not re-derive):**
- RunPod Serverless endpoint `o1jda6sxo85dnk` exists, image `ghcr.io/angelospk/oc-asr-serverless`,
  runs float16 (`int8_float16` throws CUBLAS_STATUS_NOT_SUPPORTED there). Per-second billing,
  scale-to-zero. A 3h meeting at GPU speed costs well under $0.50.
- Mini-PC CPU endpoint `https://asr.haroldpoi.dev` exists (faster-whisper int8, Cloudflare
  tunnel). Runbook: `docs/runbooks/self-hosted-asr-endpoint.md`. RTF ~0.2–0.4x real-time.
- DigitalOcean GPU droplets bill **while the droplet exists, even powered off**. They are
  therefore disqualified as the primary serving shape regardless of hourly rate.

**Deliverables:**

1. **Cold-start measurement on the RunPod serverless endpoint.** Fire 10 requests spaced
   so that at least 5 hit a cold worker. Report: p50/p95 cold-start seconds, p50/p95 warm
   latency, cost per 1h of audio. Write to `docs/reports/2026-08-XX-serverless-cost.md`.
   This is the number that decides whether scale-to-zero is acceptable for OpenCouncil's
   UX, and nobody has measured it.
2. **A cost table** over three volume scenarios (10 / 50 / 200 meeting-hours per month)
   comparing: RunPod Serverless GPU, RunPod Serverless CPU, DigitalOcean CPU-Optimized
   8vCPU/16GB (~$168/mo), Hetzner dedicated AX42-class (~€50–150/mo), and the mini PC
   (free, but not an SLA). Include the break-even volume where always-on beats per-second.
   `eval/controlled_eval/breakeven.py` already exists — read it before writing new code.
3. **A one-page recommendation** for OpenCouncil, written for someone who is not us.
   State plainly which shape wins at their actual volume and what the migration costs.
   Do NOT recommend a provider migration on price alone — include the operational cost
   of moving off DigitalOcean, since they already run there.

**Out of scope:** actually migrating anything. This is a decision document.

---

> **Task B below is SUPERSEDED (2026-08-09, second half of the day).** The version that
> governs is in the
> [windows-and-listening handoff](2026-08-09-windows-and-listening-handoff.md): a hybrid
> correct-don't-transcribe set, because the 5.5 listening hours this one assumes are not
> available. The hybrid method has to pass an 80% omission-recall calibration before it may
> produce reference data at all. Rules are frozen in
> [listening-protocol.md](../specs/listening-protocol.md). The text below is kept for the
> protocol requirements in points 1, 3 and 4, which still hold.

## Task B — Buy statistical power (highest value in the project right now)

**Problem.** Every claim is stuck at ±4.5 points because the audio-faithful reference is
32 dev windows / 1,236 words from one listener.

**Deliverable:** extend the audio-faithful reference set.

1. Read `eval/controlled_eval/build_independent_reference.py` and
   `docs/reports/2026-08-04-audio-faithful-reference.md` first. **Reproduce the existing
   protocol exactly** — same 20-second window shape, same instruction to transcribe all
   speakers, same fitting alignment at the edges, same locked/dev split discipline.
2. Sample **new public meetings after 2026-05-16 that have never been used in training or
   evaluation.** Verify against the training manifest AND the cached meeting JSON, the way
   the original script does. Target: enough windows to bring dev to **~4,000 reference
   words** (roughly 100 dev windows), which moves the detectable difference from ~4.5
   points to ~2.5.
3. Keep the locked/dev ratio. Do not open the locked split.
4. Produce the listening package in the same form the human listener already used.
   **This task ends with a package ready for a human to transcribe — do not generate the
   references with a model.** That would reintroduce exactly the circularity that
   `docs/reports/2026-08-03-the-reference-problem.md` documents.

Report estimated human listening hours required.

---

## Task C — Ship the two things that already passed a fair test

Both of these are already measured and both survived meeting-clustered confidence
intervals. Neither has shipped. This task is about making them deployable, not about
re-measuring them.

**C1 — Roster hotwords.** Decision `docs/decisions/modeling.md` 2026-07-25: name recall
27.2% → 36.0%, McNemar p = 0.021, no WER cost. The deployment shape is already written in
that decision (per-meeting `hotwords`, roster full names first then single tokens,
truncated to ~180 tokens). Deliverable: implement it in `oc_asr_server.py` behind a
request parameter, plus a test that the truncation never exceeds the prompt window.
Re-run `eval/controlled_eval/ab_hotwords_names.py` afterwards to confirm no regression.

**C2 — Gated LLM post-editor.** `docs/reports/2026-08-01-postedit-gate.md`: gated
post-editing moves Scribe 0.1529 → 0.1144, CI [−0.0487, −0.0279], nowhere near zero.
The gate rejected 4 of 98 outputs and that alone was worth 1.8 points — the gate IS the
result. Deliverable: package `eval/controlled_eval/exp_postedit_gate.py`'s gate + prompt
as a reusable module with the gate as a hard precondition (it must be impossible to call
the post-editor ungated). Then score it against the **audio-faithful** reference, not the
published-transcript reference — this has never been done, and it is the one open
question about C2.

**Note:** OpenCouncil already runs a fix-task over transcripts, and there is prior work at
`docs/specs/fix-task-eval-harness.md` + `docs/reports/fix-task-experiment-report.html`.
Read both before writing a new post-edit prompt. Do not build a second parallel system.

---

## Task D — Push

The mini PC is 13+ commits ahead of `origin/main` and the laptop has been reading stale
state as a result. When a task above completes, push. If a push is not appropriate,
say so explicitly in the report rather than leaving it local.

---

## Rules for all tasks

- Preregister the gate before running anything that produces a number. Follow the pattern
  in `docs/specs/mixture-ratio-preregistration.md`.
- Report confidence intervals, clustered by meeting. A point estimate without an interval
  is not a result.
- Never compare across decoding stacks.
- If a result is inconclusive by the frozen rule, say inconclusive. Do not reinterpret.
