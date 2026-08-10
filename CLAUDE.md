# Agent Protocol

OpenCouncil Greek ASR research (whisper-large-v3 + LoRA). The reports in this repo
are **evidence of what was concluded on a date**. They are not current project state.

## Required start

Before doing any work, read in this order:

1. `CURRENT.md` — the active queue and real blockers. Short by design.
2. `research/ledger.json` — the authoritative state.

Then search the ledger for the task's question, model, adapter, cache, dataset, or
service. Follow only the reports, specs, and runbooks linked from matching records.

Do not read `archive/` unless the ledger points there.

## Sources of authority

| File | Authoritative for |
|---|---|
| `research/ledger.json` | experiment status, current conclusions, artifact identity and validity, available capabilities |
| `CURRENT.md` | what is being worked on right now, and what is blocked |
| `docs/reports/` | immutable narrative evidence — a conclusion here may have been superseded |
| `docs/reference/` | stable interfaces and concepts |
| `docs/runbooks/` | reusable procedures |
| `docs/decisions/` | product and process decisions |

**When files disagree, the ledger wins, and you fix the stale file in the same
change.** A stale caveat in an index that contradicts its own source document has
already cost this project real time.

## Experiment states

- `OPEN` — unresolved. Work may be proposed. Must have `next_action`.
- `CLOSED` — answered, **including negative and inconclusive-by-gate results**. Do
  not rerun without explicitly reopening.
- `SUPERSEDED` — the conclusion must not be used. `superseded_by` is mandatory.

An experiment that finishes but raises a new question is `CLOSED`; the new question
gets its own `OPEN` record.

**Before proposing or spending money on an experiment**, search the ledger, check
the matching artifact and capability records, and read the linked reports. Reuse
existing audio, hypotheses, and results when their provenance matches. A whole
GPU-funded proposal has already been drafted for an experiment that was finished two
days earlier and cached on disk.

## Artifacts

Refer to models, adapters, datasets, and caches by ledger **artifact ID**, not by
path or display name alone. Several adapters exist and some are known broken.

Never write "the fine-tune" when more than one adapter exists. Name the artifact.

Any artifact behind a conclusion needs a content hash, a validity status, and its
caveats recorded. An output file whose producing model is unknown is not evidence.

## External services

Check `capabilities` in the ledger **before** claiming an operation needs a human or
that no API exists. For an `AVAILABLE` capability, read its runbook and run the
documented cheap read-only smoke check before declaring it unavailable.

Credentials live in `.env` (gitignored). The ledger records only the variable name.

## Reading numbers honestly

This project's expensive mistakes were all measurement mistakes, not code mistakes:

- **Never compare two models across two stacks.** Same machine, same decoder, same
  normalization, or it is not a comparison.
- **Freeze the decode config before you see a number.** Do not adopt whichever beam
  size wins.
- **Check single-item domination** before quoting any delta. One window has supplied
  67% of a headline effect here.
- **Distinguish the two metrics and never merge them:** *fidelity-to-audio* (WER vs
  a human who listened — this decides) and *agreement-with-OpenCouncil* (WER vs our
  own published text — records product compatibility, decides nothing).
- **Watch the deletion rate.** A model that lowers WER by omitting hard passages
  looks better and is worse.

## Finish protocol

1. Update the matching ledger record in the same change as the work.
2. Mark displaced conclusions `SUPERSEDED` — do not just add contradicting prose.
3. Update `CURRENT.md` only if the queue or blockers actually changed.
4. Run `python3 scripts/check-research-state.py`.
5. Do not create routine diary, handoff, or progress-log files.

## Hard rules

- **Transcript text and audio never go in git.** Same PII category as the
  2026-07-21 history purge. Caches live under `~/.cache/oc-public/`.
- **The 16 locked evaluation windows stay sealed.**
- **A GPU pod bills from creation.** Arm a watchdog with a hard deadline *before*
  uploading anything, and record the pod ID.
- **If something fails twice the same way, stop.** Kill the pod, write down what
  broke.
- Use `rtk <command>` for shell commands in this workspace.

## Writing

Plain Markdown, relative links. Keep canonical notes short and put detail in dated
reports. Preserve uncertainty explicitly instead of resolving it in prose. Preserve
markdown checkbox notation (`[ ]`, `[x]`, `[~]`, `[?]`) where it exists. Do not
rewrite unrelated documents for style.

## Post-change review

After a complex change (multiple files, new dependencies, schema changes, or
anything crossing module boundaries), run `coderabbit --agent`. Fix `critical` and
`important` findings before declaring done; report `nit` findings without applying
them. Skip for typo and comment edits.
