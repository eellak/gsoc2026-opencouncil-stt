# External source packs

A **pack** is how an outside speech corpus enters a fine-tune here. One shape for
every source, so adding the second and third source costs a filter script and a
dictionary entry, not a new pipeline.

Packs are supplementary. They sit next to the project's own human corrections; they do
not replace them, and a pack from an out-of-domain corpus is a stage-1 adaptation
input, not extra in-domain data. See
[`docs/specs/2026-08-14-hparl-stage1-prereg.md`](../specs/2026-08-14-hparl-stage1-prereg.md)
for why the two stages stay separate.

**Step-by-step procedure**: [`docs/runbooks/external-corpus-to-training-pack.md`](../runbooks/external-corpus-to-training-pack.md).
This file is the format contract; the runbook is how you get there.

## Layout

```
~/.cache/oc-public/training-sets/<pack-id>/
  audio/          mono 16 kHz MP3, ~4 kB/s, one clip per row
  train.jsonl     one row per clip
  meta.json       counts, hours, gate, hashes, licence, caveats
  README.md       generated: how this pack was built, what not to trust
```

Never in git — same PII category as the 2026-07-21 history purge, and the same rule as
every other cache under `~/.cache/oc-public/`.

## `train.jsonl` row

| field | meaning |
|---|---|
| `id` | stable, unique within the pack |
| `audio` | **absolute** path; the trainer verifies every one before starting |
| `text` | the training target |
| `text_pn` | same target, named for the no-timestamps training arm |
| `dur` | seconds |
| `source` | source key, so mixed packs stay separable in a manifest |
| `align` | agreement with the independent ASR that admitted the row |
| `complete_sentence` | whether the clip ends a sentence — fragments must not be punctuated as if they did |
| `text_dataset`, `text_asr` | both inputs, kept so any label can be re-derived or audited later |

Consumed by `notebooks/train_runpod.py` as
`PACK_MANIFEST=<...>/train.jsonl PACK_ARM=pn`. `PACK_ARM=p` needs Whisper timestamp
tokens in the target and is only available for sources that ship word-level timings.

## The four steps a source goes through

1. **Convert** — decode to mono 16 kHz MP3. 32 kbps is ~4 kB/s, so 120 h is ~1.8 GB;
   size is never the reason to skip this step.
2. **Verify against an independent ASR** — transcribe every clip and align against the
   corpus's own transcript. This is what separates *the corpus says it* from *the audio
   says it*. Split the errors by direction: text-without-audio and audio-without-text
   mean different things and only one of them is editorial.
3. **Repair the target** — most corpora ship targets in a different convention than
   ours (no punctuation, no case, placeholder tokens). Repair with a hard guard: the
   repaired target's normalized token sequence must equal the source's, or the row
   falls back to a deterministic transform. A repair step that may silently reword the
   target is not usable unattended.
4. **Pack** — `scripts/build_training_pack.py`, which applies the admission gate,
   drops duplicates and short rows, and writes `meta.json` with the hash and the
   caveats attached to the data rather than to somebody's memory.

## Adding a source

1. Write `eval/<source>_filter.py` (steps 1–2) and, if the targets need it,
   `eval/<source>_punctuate.py` (step 3). Both exist for `hparl2` and are the working
   examples.
2. Add an entry to `SOURCES` in `scripts/build_training_pack.py`: title, path to the
   punctuated jsonl, licence, domain, report, caveats.
3. Register the pack as an artifact in `research/ledger.json` with its
   `train_jsonl_sha256`, and link its report.

## Rules that are not negotiable

- **Licence is resolved at the original source**, not at a HuggingFace mirror. Mirrors
  have been observed contradicting themselves and the original within one card.
- **Every pack carries its caveats in `meta.json`.** A pack whose caveats live only in
  a chat log is an output file whose producing method is unknown, which this project
  does not treat as evidence.
- **A pack is never merged into the in-domain corrections silently.** It enters as its
  own arm, with a preregistered gate, so its effect can be measured and reversed.
- **Deletion rate is the metric to watch** when a pack is made of short fragments.
  A corpus that lowers WER by teaching the model to stop early looks better and is
  worse.
