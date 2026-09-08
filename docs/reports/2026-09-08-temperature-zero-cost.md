# What removing the temperature ladder cost

2026-09-08. `exp-2026-09-07-decode-nondeterminism`.

[The decoder does not repeat itself](2026-09-07-decode-nondeterminism.md)
established that faster-whisper's temperature ladder decodes by unseeded
sampling, and that `temperature=0.0` removes both the nondeterminism and most of
the runtime. It deliberately did **not** establish what the ladder was doing when
it fired. The ladder exists to retry segments that trip the log-probability and
compression-ratio thresholds, and a speed fix that quietly drops speech is not a
fix. This measures that.

## How it was set up

60 windows, the pre-declared prefix of the 80-window live cohort under the frozen
shuffle — no window was chosen for this experiment. Both conditions ran on the
**same execution stack** (local CPU int8); comparing the CPU ladder against the
GPU zero would have priced two changes as one.

Only `ours` differs. Scribe and Soniox are read from the component cache in both
conditions, so their rows are a control: if they move, the comparison is broken.

## What it cost

| arm | ladder | temperature=0.0 | delta | better/tied/worse |
|---|---:|---:|---:|---|
| Scribe | 0.16809 | 0.16803 | −0.00006 | 2/54/4 |
| Soniox | 0.15261 | 0.15277 | +0.00017 | 1/57/2 |
| ours | 0.14454 | 0.14564 | **+0.00111** | 2/57/1 |
| majority vote | 0.12492 | 0.12470 | −0.00022 | 4/50/6 |
| **rules_on** | 0.11707 | 0.11768 | **+0.00061** | 4/51/5 |

**The ladder was not rescuing speech.** Words emitted went **up**, 17,960 →
18,022, and the deletion rate went **down**, 0.02344 → 0.02316 on the shipped
arm. That was the fear the report before this one named, and on this sample it
does not happen.

## The cost is smaller than the noise

A first attempt at the noise floor used 12 windows and returned **exactly zero** —
two ladder draws, byte-identical on all five arms. That was a finding about the
sample, not about the decoder: the nondeterminism appears only where a segment
trips a threshold and the ladder actually fires, and none of those twelve do.
Quoting it as the noise floor would have made a difference look real by measuring
the noise where there is none.

Repeated on the **same 60 windows** the cost was measured on, a second ladder draw
against the first gives:

| arm | cost of dropping the ladder | the ladder's disagreement with itself |
|---|---:|---:|
| ours | +0.00111 | **+0.00238** |
| majority vote | −0.00022 | **+0.00000** |
| **rules_on** | +0.00061 | **+0.00094** |

**On both arms the cost is smaller than the noise.** The +0.00061 is not a
measured regression; it is inside the range the ladder produces by rerunning
itself on the same audio. Removing it costs nothing this instrument can see, and
buys determinism and roughly 2.3× the speed on CPU, 10× on GPU.

Two details worth keeping. The vote's noise is **exactly zero** — a majority of
three absorbs the whole of `ours` moving — while `rules_on` inherits some of it,
which is what you would expect from an arm that follows `ours` where the rules
say to. And only 3 or 4 windows in 60 differ between draws at all: the
nondeterminism is rare and large, not common and small.

**The control moved**, by −0.00006 and +0.00017. Scribe and Soniox should be
identical in both conditions. They are not, because the earlier parallel runs
raced on some windows and left the cache holding a different vendor draw than the
80-window results file recorded. It is an order of magnitude below the `ours`
delta, but this table's fifth decimal is not trustworthy.

## What is settled and what is not

Settled, on agreement-with-OpenCouncil: the ladder does not remove speech, and
dropping it costs less than the decoder's own draw-to-draw variation.

Not settled: fidelity to the audio. Every number here is agreement with our own
published text, which records product compatibility and decides nothing about
which decoder is closer to what was said. That is the listening test's job, and
it is still unrun.

`OURS_DECODE` in `eval/live_fusion/providers.py` still carries the ladder while
the serving stack does not, so the two stacks still run different decoders. That
stays deliberate until a listening test closes the question.
