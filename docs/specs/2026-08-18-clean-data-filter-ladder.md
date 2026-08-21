# Clean-data filter ladder: frozen CPU preflight

Frozen 2026-08-18 before running the census. This is a data audit, not a model
result. It may veto a GPU experiment; it cannot establish that filtering improves
WER.

## Scope

The population is the current `v2` external-pack supply: `hparl2-v2`, `stoma-v2`,
`cv-v2`, and `eurospeech-v2`. The packs are counted separately. In particular,
HParl and EuroSpeech are not silently combined: they overlap in source domain and
their own artifact caveats forbid doing that without deduplication.

The in-domain OpenCouncil backbone is outside this census. It has no independent
ASR witness for every row, so putting it into L0 and then requiring a witness at L1
would measure metadata availability rather than label quality.

## Frozen levels

Levels are nested. A row that cannot be evaluated at a level is reported as
`unmeasurable`, never as a filter failure.

| Level | Frozen rule |
|---|---|
| L0 | Row is present in the current `v2` pack. The existing pack floor was `align >= 0.50`, duration `1.5..30 s`, at least 3 whitespace tokens, with clipped-edge rows retained and flagged. |
| L1 | L0 plus a usable Soniox witness: non-empty `text_asr` and finite `align`. |
| L2 | L1 plus `align >= 0.95`, duration `1.5..30 s`, at least 3 normalized Greek tokens, and no clipped-edge flag. |
| L3 | L2 plus a second independent ASR witness agreeing with the target at WER `<= 0.05`. The only existing second witness is EuroSpeech's corpus-provided Whisper-Turbo `ds_wer`; other sources are `unmeasurable`, not failed. |

The thresholds are not tuned from this census. L0 is the current harmonized pack
gate; L2 is the earlier strict Soniox-agreement/clean-edge policy with the current
duration floor; L3 operationalizes the ticket's two-system requirement.

## Frozen retention diagnostics

- `known_roster_name`: normalized full-phrase match against the existing public
  council rosters. This is a high-precision person-name proxy, not entity recall;
  inflected or external names absent from those rosters are missed.
- `capitalized_noninitial`: at least one non-initial capitalized Greek token. This is
  a broader, noisier name proxy and is unreliable for labels whose casing was
  generated or normalized.
- `fast_speech`: at least 3.0 normalized tokens per second.
- `hard_example`: Soniox alignment in `[0.50, 0.85)` or a clipped-edge flag.
- `edge_clipped`: the existing pack flag.

For every level, report rows and audio hours by source and for every diagnostic.
Retention is always against the same diagnostic population at L0.

## Equal-hours comparison

The interpretable comparison is L0 versus L2 independently within each source.
For each paired training seed, deterministically shuffle L0 rows and sample the same
number of audio seconds as that source's L2 arm. Keep optimizer updates, sampling
exposure, seed, recipe, and decode fixed. Do not match the diagnostic categories:
their changed prevalence is part of what filtering does and must be reported.

L1 is not a separate arm if its census is identical to L0. L3 is eligible only for
EuroSpeech and only as a proxy screen: its second witness came from Whisper-Turbo,
so it preferentially selects examples already easy for the base-model family.

No GPU stage is authorized by this spec. Any later run follows
[`training-evidence.md`](../decisions/training-evidence.md) and requires explicit
user approval.
