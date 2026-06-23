"""Round 2: combine the winning axes + extra seeds (P4).

Runs after round 1 (loop.py). Reuses the same frozen manifest, the same model
recipe, and the same zero-shot reference (read from leaderboard.jsonl) so every
number is comparable across both rounds. Appends to leaderboard.jsonl +
results.tsv + loop.log. Time-boxed; does NOT write report.md (the human writes
the consolidated report afterwards).

Combos are built around the data-composition keeper from round 1 (no-edit
backbone at 1x), layering sampling / error-focus / lr / filter axes on top, then
reseeding the best combo.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import experiment as X
import loop as L

BB = "corr+backbone_1x"  # round-1 data-composition keeper


def combos():
    base = dict(L.BASE_CFG)
    base["composition"] = BB
    return [
        ("bb1x_capped", {**base, "sampling": "capped_oversample"}),
        ("bb1x_balanced", {**base, "sampling": "error_category_balanced"}),
        ("bb1x_acoustic", {**base, "error_focus": "acoustic"}),
        ("bb1x_capped_acoustic", {**base, "sampling": "capped_oversample", "error_focus": "acoustic"}),
        ("bb1x_lr5e-5", {**base, "lr": 5e-5}),
        ("bb1x_strict", {**base, "filters": "strict"}),
        ("bb1x_capped_strict", {**base, "sampling": "capped_oversample", "filters": "strict"}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=14.0)
    ap.add_argument("--val-corr-cap", type=int, default=70)
    ap.add_argument("--val-reg-cap", type=int, default=70)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    deadline = time.time() + args.budget_min * 60

    # zero-shot reference from round 1
    lb = [json.loads(x) for x in L.LEADER.open()]
    zero_shot = next(r for r in lb if r.get("label") == "zero_shot")

    L.log(f"== ROUND 2 (combinations) start, budget={args.budget_min}min ==")
    data = L.load_data(args.val_corr_cap, args.val_reg_cap)

    t0 = time.time()
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    proc = WhisperProcessor.from_pretrained("openai/whisper-base")
    base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    base.config.forced_decoder_ids = None
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none"))
    L.log(f"model loaded ({time.time()-t0:.0f}s)")

    runs = []
    grid = combos()
    done = 0
    for label, cfg in grid:
        if deadline - time.time() < 200:
            L.log(f"-- R2 budget nearly exhausted; stopping after {done} combos "
                  f"(skipped {len(grid)-done}) --")
            break
        L.log(f"-- [R2:{label}] cfg={cfg} (remaining {(deadline-time.time())/60:.1f}min) --")
        try:
            r = X.run_experiment(model, proc, data, cfg, L.log)
        except Exception as e:  # noqa: BLE001
            L.log(f"    CRASH {label}: {type(e).__name__} {str(e)[:80]}")
            done += 1
            continue
        r["label"] = f"r2_{label}"
        status, z_vc, z_vr = L.keep_decision(r, zero_shot)
        r["status"] = status
        r["delta_val_corr"] = round(r["val_corr"]["wer_norm"] - z_vc, 4)
        r["delta_val_reg"] = round(r["val_reg"]["wer_norm"] - z_vr, 4)
        L.append_jsonl(L.LEADER, r)
        with L.TSV.open("a") as f:
            f.write(f"r2_{label}\t{r['val_corr']['wer_norm']}\t{r['val_reg']['wer_norm']}\t"
                    f"{status}\tΔcorr={r['delta_val_corr']:+.4f} Δreg={r['delta_val_reg']:+.4f}\n")
        L.log(f"    => {status} Δval_corr={r['delta_val_corr']:+.4f} Δval_reg={r['delta_val_reg']:+.4f}")
        runs.append(r)
        done += 1

    # reseed best round-2 combo
    if runs:
        best = min(runs, key=lambda r: r["val_corr"]["wer_norm"])
        L.log(f"-- R2 best = {best['label']} (val_corr={best['val_corr']['wer_norm']}); reseeding --")
        for seed in (1, 2):
            if deadline - time.time() < 200:
                L.log("-- no time to reseed; done --")
                break
            cfg = {**best["cfg"], "seed": seed}
            try:
                r = X.run_experiment(model, proc, data, cfg, L.log)
                r["label"] = f"{best['label']}#s{seed}"
                status, *_ = L.keep_decision(r, zero_shot)
                r["status"] = status
                L.append_jsonl(L.LEADER, r)
                with L.TSV.open("a") as f:
                    f.write(f"{best['label']}#s{seed}\t{r['val_corr']['wer_norm']}\t"
                            f"{r['val_reg']['wer_norm']}\t{status}\treseed\n")
                L.log(f"    reseed s{seed} => val_corr={r['val_corr']['wer_norm']} val_reg={r['val_reg']['wer_norm']}")
            except Exception as e:  # noqa: BLE001
                L.log(f"    reseed CRASH: {type(e).__name__} {str(e)[:80]}")
    L.log("== ROUND 2 done ==")


if __name__ == "__main__":
    main()
