# Arm B at a 2-epoch budget — live run, 2026-08-22

Everything the run needs, so the pod is never the only copy.

| | |
|---|---|
| pod | `blzmsvsqmd2c0b`, RTX A4000, **$0.17/h** |
| ssh | `ssh -p 1706 root@193.183.22.58` |
| watchdog | terminates 2026-08-22 **19:30 EEST**, armed on the minipc |
| notifications | `curl -d "msg" https://ntfy.sh/MvtrfeSLN3jvt50N` |
| run root | `/workspace/oc/runs` on the pod |
| bundle | `/workspace/oc/bundle` (synced from `s3://qzw88vdwv2/oc-bundles/clean-pack-screen-736cedc61ce3a5fe/`) |

## The arms

All at **619 steps = 2 epochs** on 2,475 contiguous packs, effective batch 8, seeds
13/29/47, checkpoints every 155 steps. Steps are matched across arms so a data change
is the only change.

1. `cont_s{13,29,47}` — contiguous packs. **Running.**
2. `comb_s{13,29,47}` — contiguous + jittered spliced (6,983 packs). Runs **only** if
   `build_combined.py` writes `/workspace/oc/combined/manifest.jsonl`; that script
   refuses on any integrity failure rather than training on half the data.
3. `dec_s{13,29,47}` — contiguous packs, `LORA_SCOPE=decoder` (encoder fully frozen).
   Fallback if 2 fails, and worth running regardless: it is a one-variable comparison
   against arm 1, whose control is already on disk.

## Decision rule, declared before any number was seen

Ship a challenger over the published `artifact-adapter-fixed` **only** if the upper
bound of the paired meeting-clustered bootstrap CI against it, on the same GPU float16
stack, falls below zero. Otherwise the incumbent ships. Do not pick the lowest point
estimate.

Known bias, to be stated in the report: the 39 validation windows were this
incumbent's own validation set and have now driven two screens, the stopping
behaviour and a budget choice. The rule is deliberately conservative because of it.

## Order of operations

1. `stage_pod.sh <s3_key_id> <s3_secret>` — bundle, deps, packs.
2. `run_armb.sh` — arms 1 and 2.
3. `build_combined.py` — before arm 2 can run.
4. `run_decoder_only.sh` — arm 3.
5. Score validation-only for every adapter and checkpoint, plus the incumbent
   re-scored on this stack.
