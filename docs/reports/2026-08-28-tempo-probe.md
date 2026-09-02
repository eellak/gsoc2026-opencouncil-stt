# Feeding Soniox slower does not recover words lost to crosstalk — it costs words

2026-08-28. Record: `exp-2026-08-28-tempo-probe`. Preregistration:
[`docs/specs/2026-08-28-tempo-probe.md`](../specs/2026-08-28-tempo-probe.md), written and
frozen before the first session. Plan reviewed by Codex `db4d8ac8`.

## Verdict

**STOP on all four arms.** The control is the best arm in the table. Every manipulation
loses recall, and the aggressive stretch loses more than the instrument's own noise, both
inside the overlap and outside it.

225 free `stt-rt-v5` sessions: 15 gold cells over 5 meetings, 5 arms, 3 interleaved
randomized cycles. 18 linked overlap episodes, 217 certain human words per cycle
(651 across the three).

| arm | what it changes | recall | Δ vs control | meetings better | non-overlap Δ | verdict |
|---|---|---:|---:|---:|---:|---|
| `proc10` | control, `atempo=1.0` | **0.8756** | — | — | — | — |
| `raw` | untouched file | 0.8710 | −0.0046 | 0 / 5 | +0.001 | STOP |
| `pace070` | fed at 0.7× wall clock | 0.8710 | −0.0046 | 1 / 5 | 0.000 | STOP |
| `t085` | `atempo=0.85` | 0.8679 | −0.0077 | 1 / 5 | −0.005 | STOP |
| `t070` | `atempo=0.70` | 0.8587 | **−0.0169** | 1 / 5 | **−0.021** | STOP |

The noise envelope is **0.0092**, the widest spread of a single arm across its own three
cycles. Three arms sit inside it, on the losing side. **`t070` sits outside it**, and it
is the only arm that also fails the collateral guard: recall on the non-overlap blocks of
the same cells falls 2.1 points. Stretching to 0.70 is not neutral — it is harmful, and
it is harmful everywhere, not only where people talk over each other.

`raw` against `proc10` is the sanity check that makes the rest readable: passing the file
through `atempo=1.0` and not passing it at all differ by 0.0046, which is what this
instrument's disagreement with itself looks like.

## What the arms did do

Slowing changes the output; it just does not improve it.

| comparison | new words | words lost |
|---|---:|---:|
| control against itself, cycle to cycle | 4.3 (mean of three pairs) | — |
| `raw` against control | 13 | 8 |
| `t085` against control | 12 | 16 |
| `pace070` against control | 15 | 11 |
| `t070` against control | 16 | 11 |

Three times the churn of pure nondeterminism, and roughly as many words lost as gained.
The change is **lateral** — different words, not more correct ones — which is what the
prior said: time stretching does not unmix sources or raise SNR, and crosstalk loss is
source competition.

## The measurement bug Harold caught

The first version of this report gave recall ≈ 0.32 for every arm. That number was wrong,
and how it was caught matters more than the number.

Soniox's realtime stream returns **sub-word pieces**, not words: `Κ`, `υ`, `ρ`, `ία`,
` Αγ`. The scorer treated each piece as a word, so human WORDS were matched against ASR
PIECES, and only the pieces that happened to spell a short Greek word could ever match.
Nothing in the recall table looked odd — all five arms were equally wrong, so the
comparison still ran and still said STOP.

What exposed it was the **adjudication page**, which showed Harold the individual words
the arms disagreed about. He said they were not words. Of the 52 candidates, 43 were three
characters or shorter, and the list held fragments like `γραφ`, `ειται`, `νται`. The
artefact built for a human to judge is what made the defect visible, because a human
finally looked at the units the metric was built from.

The fix is not a new loop. `eval/soniox_confidence_probe.py:group_words` already did this
correctly, mirroring the production Go client (`soniox-core`,
`internal/soniox/soniox.go`): finals only, explode each piece into runes carrying its
confidence, cut words on whitespace. `words_from_tokens` now calls it and adds only the
tempo mapping. Writing a second implementation of the same rule is how the two drift apart
again; the first version of this module is what that drift looks like.

One thing had to be added on top: `group_words` silently skips a word with no confidence
or no start time, which is right for a confidence study and wrong for a recall study,
where a missing word becomes a deletion charged to whichever arm produced it. It now
raises instead. On the 225 cached sessions it never fired.

Re-scoring was free — the sessions were cached — and the verdict did not change; it
hardened, because the control turned out to be the best arm rather than the middle of the
pack.

## The adjudication was not run, on purpose

It exists (`eval/controlled_eval/adjudicate_tempo.py`) and it carried its own null class:
words the control disagrees with **itself** about, mixed in unlabelled, so a supported-rate
would have had a yardstick. It was abandoned for two reasons, in this order: the candidate
set it produced was mostly sub-word fragments, which is the bug above; and after the fix no
arm gains recall, so the question it answers — whether the extra words turn out to be real
— cannot rescue an arm that is already behind the control on the words we can check.

## Limits, stated before the numbers were seen

- 217 certain words over 5 meetings is a **direction**, not a decision. Three cycles cut
  measurement noise; they do not enlarge the sample.
- The endpoint is **overlap-associated episode recall**. The human blocks carry no
  per-word timings, so an episode is the union of the linked blocks' spans rather than the
  strictly simultaneous interval.
- A result here is about the **witness**, not about our adapter. Soniox is a hosted
  streaming model; a gain would have meant better mining labels, not a decode change we
  can ship.
- `rubberband` was held back as a second-implementation check on surviving episodes.
  Nothing survived, so it was not run.
