# Is our error a listening error or a spelling error? (2026-08-23)

A free screen, run before any GPU is bought, on the proposal to fine-tune **only the
decoder**. The premise is that our adapter's substitutions are near-misses — the word
is there, wrong by a character or two — which would be a language-model failure rather
than an acoustic one, and the only kind decoder-only training can fix.

Substrate: the 391 held-out post-June windows of `exp-2026-08-23-post-june-held-out`,
110,610 reference tokens, no meeting from any training pack. Every substitution pair
was extracted from the reference alignment. No GPU, no audio, no API calls.

## The premise holds, at the level it was stated

Substitutions on reference words of 4+ characters, by character edit distance:

| distance | ours (clean-pack) | Scribe v2 | Soniox |
|---|---|---|---|
| 1 char | 27.3% | 24.2% | 24.0% |
| 2 chars | 17.3% | 16.9% | 14.8% |
| 3 chars | 12.1% | 10.4% | 10.7% |
| 4+ chars | 43.3% | 48.5% | 50.6% |
| **near-miss (1–2)** | **44.6%** | 41.1% | 38.8% |

So yes: nearly half our long-word substitutions are within two characters of the right
word. But so are Scribe's and Soniox's. **The shape of the error is not what
distinguishes our adapter — the volume is.** We make 11,708 substitutions where Scribe
makes 5,803.

## The sharper question, and the number that decides it

A near-miss is not automatically a decoder error. Some near-misses are genuinely
mis-heard. The errors a decoder *could* fix and an acoustic model *never* can are the
ones where the two spellings sound identical.

Greek is unusually rich in these: ω and ο are one sound; η, ι, υ, ει, οι are all /i/;
αι is /e/. Folding those, a substitution whose two words collapse to the same skeleton
was not misheard. It was misspelled.

| system | substitutions | homophone misspellings | share | WER if all were fixed |
|---|---|---|---|---|
| ours (clean-pack) | 11,708 | 855 | 7.3% | **−0.0077** |
| Soniox | 6,648 | 296 | 4.5% | −0.0027 |
| Scribe v2 | 5,803 | 168 | 2.9% | −0.0015 |

The most common ones are council vocabulary, and they are the same handful over and
over: `απών → απόν` and `παρών → παρόν` (present/absent in a roll call, 46 times
between them), `πίστωσης → πίστοσης`, `Αργιθέας → Αργυθέας`, `ΑμεΑ → αμαία`.

**Our adapter misspells homophones 2.5 times more often than Scribe does.** That is a
real, specific, decoder-shaped defect, and it is exactly what Harold's proposal
predicted.

## And it is smaller than it looks

A perfect oracle that fixed every homophone misspelling would move our adapter from
0.1795 to 0.1718. Scribe is at 0.1377. The gap to close is 4.18 WER points; homophone
spelling accounts for about **0.6 of them, roughly 15%**.

For scale, on the same 391 windows the gap between three-system composition (0.1202)
and the per-column oracle on its own alignment (0.0611) is **5.91 WER points**
(`exp-2026-08-23-fusion-postjune`) — nearly eight times the entire homophone ceiling.

## What this screen decides

- **The decoder-only proposal is not refuted.** It names a real defect with a
  measurable signature, and it is the cheapest training arm left: one seed, one rung,
  roughly $0.40 on the pod already used tonight.
- **It cannot be sold as the way to catch Scribe.** Its ceiling is bounded at under one
  WER point, and that ceiling assumes a perfect fix.
- **A cheaper variant exists and should be screened first.** The frequent confusions are
  a short, closed list of council vocabulary. A post-hoc spelling repair against that
  list costs no GPU at all and has the same ceiling. If the repair cannot capture the
  0.0077, retraining the decoder to capture it is not a good bet either.

Nothing here justifies GPU spend tonight. It justifies the next screen.
