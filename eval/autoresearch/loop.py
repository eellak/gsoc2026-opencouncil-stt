"""Autonomous, wall-clock-bounded auto-research loop (Track 2).

karpathy/autoresearch adapted to ASR data-mixture research:
  - establish a ZERO-SHOT baseline (no fine-tune) first — the reference WER
  - then sweep one-axis configs against it, each under the SAME fixed recipe
  - keep/discard by the selection rule: a config "wins" only if it improves
    val_corr normalized WER WITHOUT regressing val_reg (ordinary speech)
  - log every run to leaderboard.jsonl + results.tsv; stop when the time budget
    is exhausted; reseed the finalist if time remains; write report.md.

Everything is logged; nothing about what was skipped is silent. Runs on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import experiment as X

ROOT = Path(__file__).resolve().parent.parent.parent
ASR = ROOT / "data" / "asr"
MANIFEST = ASR / "manifest.jsonl"
OUT = ROOT / "data" / "reports" / "finetune-research"
LOG = OUT / "loop.log"
LEADER = OUT / "leaderboard.jsonl"
TSV = OUT / "results.tsv"
REPORT = OUT / "report.md"

TOL = 0.01  # val_reg regression tolerance (absolute normalized WER)

BASE_CFG = dict(composition="corrections_only", sampling="uniform",
                error_focus="none", lr=1e-4, filters="none",
                steps=40, grad_accum=4, seed=0)


def variants():
    """Baseline FT first, then one-axis changes from BASE_CFG."""
    v = [("base_ft", dict(BASE_CFG))]
    v += [
        ("comp_backbone_1x", {**BASE_CFG, "composition": "corr+backbone_1x"}),
        ("comp_backbone_3x", {**BASE_CFG, "composition": "corr+backbone_3x"}),
        ("sample_cat_balanced", {**BASE_CFG, "sampling": "error_category_balanced"}),
        ("sample_capped_oversample", {**BASE_CFG, "sampling": "capped_oversample"}),
        ("focus_acoustic", {**BASE_CFG, "error_focus": "acoustic"}),
        ("lr_5e-5", {**BASE_CFG, "lr": 5e-5}),
        ("filters_strict", {**BASE_CFG, "filters": "strict"}),
    ]
    return v


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_data(val_corr_cap, val_reg_cap):
    rows = [json.loads(l) for l in MANIFEST.open()]
    by = {"train": [], "val_corr": [], "val_reg": [], "train_noedit": []}
    for r in rows:
        r["clip_path"] = str(ROOT / r["clip_path"])
        by[r["split"]].append(r)
    by["val_corr"] = by["val_corr"][:val_corr_cap]
    by["val_reg"] = by["val_reg"][:val_reg_cap]
    return by


def keep_decision(run, zero_shot):
    vc = run["val_corr"]["wer_norm"]
    vr = run["val_reg"]["wer_norm"]
    z_vc, z_vr = zero_shot["val_corr"]["wer_norm"], zero_shot["val_reg"]["wer_norm"]
    improved = vc < z_vc
    no_regress = vr <= z_vr + TOL
    return ("keep" if (improved and no_regress) else "discard"), z_vc, z_vr


def append_jsonl(path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=38.0)
    ap.add_argument("--val-corr-cap", type=int, default=70)
    ap.add_argument("--val-reg-cap", type=int, default=70)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    deadline = time.time() + args.budget_min * 60
    LOG.write_text(""); LEADER.write_text("")
    TSV.write_text("label\twer_corr_norm\twer_reg_norm\tstatus\tdescription\n")

    log(f"== auto-research loop start, budget={args.budget_min}min, deadline in {args.budget_min}min ==")
    data = load_data(args.val_corr_cap, args.val_reg_cap)
    log(f"data: train={len(data['train'])} val_corr={len(data['val_corr'])} "
        f"val_reg={len(data['val_reg'])} backbone={len(data['train_noedit'])}")

    # ---- load model + attach LoRA once ----
    t0 = time.time()
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    proc = WhisperProcessor.from_pretrained("openai/whisper-base")
    base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    base.config.forced_decoder_ids = None
    lc = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.0, bias="none")
    model = get_peft_model(base, lc)
    log(f"model loaded + LoRA attached ({time.time()-t0:.0f}s)")

    runs = []

    # ---- ZERO-SHOT baseline (reset LoRA -> identity -> == base model) ----
    log("-- ZERO-SHOT baseline (no fine-tune) --")
    X.reset_lora(model)
    zero_shot = {
        "label": "zero_shot", "cfg": None,
        "val_corr": X.evaluate(model, proc, data["val_corr"], log, "val_corr"),
        "val_reg": X.evaluate(model, proc, data["val_reg"], log, "val_reg"),
    }
    append_jsonl(LEADER, zero_shot)
    with TSV.open("a") as f:
        f.write(f"zero_shot\t{zero_shot['val_corr']['wer_norm']}\t"
                f"{zero_shot['val_reg']['wer_norm']}\tref\tno fine-tune baseline\n")
    runs.append(zero_shot)

    # ---- one-axis sweep, time-boxed ----
    grid = variants()
    done = 0
    for label, cfg in grid:
        remaining = deadline - time.time()
        if remaining < 200:  # not enough time for another run + report
            log(f"-- budget nearly exhausted ({remaining:.0f}s left); stopping sweep "
                f"after {done} configs (skipped {len(grid)-done}) --")
            break
        log(f"-- [{label}] cfg={cfg} (remaining {remaining/60:.1f}min) --")
        try:
            r = X.run_experiment(model, proc, data, cfg, log)
        except Exception as e:  # noqa: BLE001
            log(f"    CRASH {label}: {type(e).__name__} {str(e)[:80]}")
            with TSV.open("a") as f:
                f.write(f"{label}\t0.0\t0.0\tcrash\t{type(e).__name__}\n")
            done += 1
            continue
        r["label"] = label
        status, z_vc, z_vr = keep_decision(r, zero_shot)
        r["status"] = status
        r["delta_val_corr"] = round(r["val_corr"]["wer_norm"] - z_vc, 4)
        r["delta_val_reg"] = round(r["val_reg"]["wer_norm"] - z_vr, 4)
        append_jsonl(LEADER, r)
        with TSV.open("a") as f:
            f.write(f"{label}\t{r['val_corr']['wer_norm']}\t{r['val_reg']['wer_norm']}\t"
                    f"{status}\tΔcorr={r['delta_val_corr']:+.4f} Δreg={r['delta_val_reg']:+.4f}\n")
        log(f"    => {status} Δval_corr={r['delta_val_corr']:+.4f} Δval_reg={r['delta_val_reg']:+.4f}")
        runs.append(r)
        done += 1

    # ---- reseed the best 'keep' finalist if time remains (>=3 seeds total) ----
    keeps = [r for r in runs if r.get("status") == "keep"]
    finalist = min(keeps, key=lambda r: r["val_corr"]["wer_norm"]) if keeps else None
    finalist_seeds = []
    if finalist:
        log(f"-- finalist = {finalist['label']} (val_corr wer_norm={finalist['val_corr']['wer_norm']}) --")
        for seed in (1, 2):
            if deadline - time.time() < 220:
                log(f"-- no time to reseed seed={seed}; reporting with available seeds --")
                break
            cfg = {**finalist["cfg"], "seed": seed}
            log(f"-- reseed {finalist['label']} seed={seed} --")
            try:
                r = X.run_experiment(model, proc, data, cfg, log)
                r["label"] = f"{finalist['label']}#s{seed}"
                status, *_ = keep_decision(r, zero_shot)
                r["status"] = status
                append_jsonl(LEADER, r)
                finalist_seeds.append(r)
            except Exception as e:  # noqa: BLE001
                log(f"    reseed CRASH: {type(e).__name__} {str(e)[:80]}")

    write_report(runs, zero_shot, finalist, finalist_seeds, args)
    log("== loop done ==")


def write_report(runs, zero_shot, finalist, finalist_seeds, args):
    stats = json.loads((ASR / "dataset_stats.json").read_text())
    z_vc = zero_shot["val_corr"]["wer_norm"]; z_vr = zero_shot["val_reg"]["wer_norm"]
    sweep = [r for r in runs if r.get("cfg") is not None]
    sweep_sorted = sorted(sweep, key=lambda r: r["val_corr"]["wer_norm"])

    L = []
    L.append("# Tiny-Whisper CPU auto-research — findings (Track 2)\n")
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')}. Model: whisper-base (72.6M, "
             f"LoRA r=8 q/v, encoder frozen), CPU-only. Budget {args.budget_min:.0f} min._\n")
    L.append("## What this is\n")
    L.append("An automated fine-tuning experiment loop (after karpathy/autoresearch): "
             "each run varies **one data/training axis**, fine-tunes tiny Whisper on CPU, "
             "and measures validation WER. The goal is **which data choices move val WER** "
             "(transferable to a large-v3 GPU run), not a deployable model. 1.8 h of "
             "corrections-scale data will overfit — read deltas as *direction*, not magnitude.\n")

    L.append("## Data (frozen manifest)\n")
    L.append(f"- clips: **{stats['n_clips']}** — by split: `{stats['by_split']}`")
    L.append(f"- minutes by split: `{stats['minutes_by_split']}`")
    L.append(f"- val cities (held out, disjoint): `{stats['val_cities']}`")
    L.append(f"- build: {stats['params']['train_meetings']} train + "
             f"{stats['params']['val_meetings_per_city']}/city val meetings, "
             f"{stats['build_seconds']:.0f}s\n")

    L.append("## Baseline (zero-shot whisper-base, no fine-tune)\n")
    L.append("| set | wer_norm | wer_raw | cer |")
    L.append("|---|---|---|---|")
    L.append(f"| val_corr | {z_vc} | {zero_shot['val_corr']['wer_raw']} | {zero_shot['val_corr']['cer_norm']} |")
    L.append(f"| val_reg | {z_vr} | {zero_shot['val_reg']['wer_raw']} | {zero_shot['val_reg']['cer_norm']} |\n")

    L.append("## Sweep leaderboard (vs zero-shot)\n")
    L.append("Selection rule: **keep** only if val_corr wer_norm improves AND val_reg "
             f"does not regress beyond +{TOL}.\n")
    L.append("| rank | config | n_train | val_corr (Δ) | val_reg (Δ) | status |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(sweep_sorted, 1):
        L.append(f"| {i} | `{r['label']}` | {r['n_train']} | "
                 f"{r['val_corr']['wer_norm']} ({r['delta_val_corr']:+.4f}) | "
                 f"{r['val_reg']['wer_norm']} ({r['delta_val_reg']:+.4f}) | {r['status']} |")
    L.append("")

    if finalist:
        L.append(f"## Finalist: `{finalist['label']}`\n")
        seeds = [finalist] + finalist_seeds
        vcs = [s["val_corr"]["wer_norm"] for s in seeds]
        L.append(f"- seeds run: {len(seeds)} → val_corr wer_norm = "
                 f"{[round(v,4) for v in vcs]} (mean {sum(vcs)/len(vcs):.4f})")
        pc = finalist["val_corr"].get("per_category", {})
        if pc:
            L.append("\n### Finalist per-category val_corr wer_norm\n")
            L.append("| category | n | wer_norm |")
            L.append("|---|---|---|")
            for c, d in sorted(pc.items(), key=lambda kv: -kv[1]["n"]):
                L.append(f"| {c} | {d['n']} | {d['wer_norm']} |")
        L.append("")
    else:
        L.append("## Finalist\n\nNo config satisfied the selection rule "
                 "(improve val_corr without regressing val_reg). "
                 "Expected at this data scale; see caveats.\n")

    L.append("## What transfers to the large-v3 GPU run\n")
    keeps = [r for r in sweep if r.get("status") == "keep"]
    if keeps:
        best = min(keeps, key=lambda r: r["val_corr"]["wer_norm"])
        axes = {k: v for k, v in best["cfg"].items()
                if k in ("composition", "sampling", "error_focus", "lr", "filters")
                and v != BASE_CFG[k]}
        L.append(f"- Best-keeping axis change vs the FT baseline: `{axes or 'none (baseline FT itself won)'}`.")
        L.append("- Carry the **winning data-mixture axis** into the large-v3 LoRA run; "
                 "re-tune lr/batch there (those do **not** transfer from a tiny CPU model).")
    else:
        L.append("- No axis beat the regression guard at this scale. Signal is inconclusive; "
                 "the transferable next step is **more data (no-edit backbone + larger val)**, "
                 "not a tiny-model hyperparameter.")
    L.append("\n## Caveats / what was capped for the 1 h budget\n")
    L.append(f"- val drawn from same meetings as some corrections (same-meeting regression view); "
             "report meeting-level CIs in a longer run.")
    L.append(f"- single seed per axis (finalist reseeded to {1+len(finalist_seeds)} seeds); "
             "tiny model; ~few-hundred-step CPU runs; capped meeting set. "
             "See `loop.log` + `leaderboard.jsonl` for full traces.\n")

    REPORT.write_text("\n".join(L))
    log(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
