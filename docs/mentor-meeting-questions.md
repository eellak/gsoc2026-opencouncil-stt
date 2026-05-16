# Mentor Meeting Questions

Use this as the agenda for the next mentor sync.

## Data Access and Sharing

- Can the cleaned correction dataset be shared publicly, or only scripts/model adapters?
- Are `audio_url` files licensed for redistribution, or should the pipeline download them from authenticated/internal storage?
- Are there privacy constraints around council member names, speakers, or meeting-specific entities?
- Should malformed/rejected CSV rows be re-exported from the database instead of manually repaired?

## Dataset Construction

- Are utterance timestamps reliable enough for direct FFmpeg cuts?
- Should the first dataset use individual utterances or grouped 2-5 minute windows?
- Which municipalities should be held out entirely for generalization evaluation?
- Should `task`-edited rows be used as training references, or only `user`-edited rows?

## Taxonomy and Routing

- Do mentors agree with the four routing buckets in [Error taxonomy and routing](reference/error-taxonomy.md)?
- Which error types are explicitly out of scope for Whisper fine-tuning?
- Should proper nouns be split between stable public/domain names and volatile local names?
- What manual review sample size is enough before trusting automated labels?

## Evaluation

- Which metric matters most operationally: WER, CER, DS-WER, HIR, or review-time reduction?
- What should be the first milestone report: data quality, baseline WER, or taxonomy validation?
- Are there known difficult meetings that should become stress tests?

## Infrastructure

- What GPU/VRAM budget is realistic?
- Should experiments use W&B, simple CSV logs, or both?
- Where should inference benchmarks run?
