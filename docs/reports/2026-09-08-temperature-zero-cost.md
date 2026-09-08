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

The cost is **+0.00061 WER on the arm we ship**, against determinism and roughly
2.3× the speed on CPU and 10× on GPU.

## Two things that keep this open

**The noise floor came out exactly zero, and that is a finding about the
experiment.** Two ladder draws over the same 12 windows produced byte-identical
output on every arm — 0/12/0 on all five. So the nondeterminism is **not
uniform**: it appears only when a segment trips a threshold and the ladder
actually fires. The window that produced the original 0.0972–0.1215 self-WER
tripped it; these twelve did not. The +0.00061 is therefore above the noise floor
*of a subset that contains no noisy windows*, which is weaker than it looks. A
noise floor measured on windows where the ladder fires is the missing number.

**The control moved**, by −0.00006 and +0.00017. Scribe and Soniox should be
identical in both conditions. They are not, because the earlier parallel runs
raced on some windows and left the cache holding a different vendor draw than the
80-window results file recorded. The drift is an order of magnitude below the
`ours` delta, but it means this table's third decimal is not trustworthy.

## What is settled and what is not

Settled: the ladder does not remove speech on these 60 windows, and the shipped
arm loses about 0.0006 WER by dropping it.

Not settled: whether that holds on windows where the ladder actually fires, and
whether any of it survives a listening test. The metric here is
agreement-with-OpenCouncil throughout. It records product compatibility and
decides nothing about which decoder is closer to the audio.

`OURS_DECODE` in `eval/live_fusion/providers.py` still carries the ladder while
the serving stack does not, so the two stacks still run different decoders. That
stays deliberate until a listening test closes the question.
