"""Design-A pipeline matrix for the error-division experiment.

The 2x2:  rows = {baseline ASR, fine-tuned ASR}, cols = {no LLM, + LLM fix-task}
  A = baseline ASR             B = baseline ASR + LLM fix
  C = fine-tuned ASR           D = fine-tuned ASR + LLM fix

Scored on the categorized held-out val set (val_manifest.jsonl, split
val_corr_cat). Per error_category we report micro-averaged WER (raw + Greek-
normalized) and normalized CER for A/B/C/D, with meeting-clustered bootstrap CIs
and paired clustered deltas (C-A, B-A, D-C, D-B). Every cell carries n_clips and
n_meetings; cells below the thresholds are flagged directional.

Design rules (Codex review, 2026-06-23):
  - identical decoding + normalization across all four stages (reuse experiment.py)
  - stages A/C are RAW per-item Whisper hypotheses; score from the same raw text
  - the LLM stage sees ONLY (city, ASR hypothesis) -> no reference/category leak
  - strict fail-loud extraction; a clip unparseable in any stage is dropped from
    ALL stages (paired alignment preserved) and logged, never silently kept
  - micro-average (summed edit-distance / summed reference length), not mean WER
  - bootstrap resamples MEETINGS, not clips

Pure scoring core (edit_distance / build_records / micro / per_category /
bootstrap_ci / paired_delta_ci / extract_fix / run_llm_stage / render_report) is
model-free and covered by test_matrix.py. main() wires the real model + LLM.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment import normalize_el  # identical Greek normalization across stages

STAGES = ("A", "B", "C", "D")
METRICS = ("word_raw", "word_norm", "char_norm")
N_CLIPS_MIN = 30
N_MEETINGS_MIN = 5

ROOT = Path(__file__).resolve().parent.parent.parent
_NUM_RE = re.compile(r"^\s*(\d+)\.\s?(.*)$")


# ---------------- edit distance (unit-cost Levenshtein = S+D+I) ----------------
def edit_distance(a, b) -> int:
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] if ca == cb else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _counts(ref: str, hyp: str, normfn, char: bool):
    r = normfn(ref)
    h = normfn(hyp)
    ru = list(r) if char else r.split()
    hu = list(h) if char else h.split()
    return edit_distance(ru, hu), len(ru)


def _clip_metrics(ref: str, hyp: str) -> dict:
    return {
        "word_raw": _counts(ref, hyp, lambda x: x.strip(), char=False),
        "word_norm": _counts(ref, hyp, normalize_el, char=False),
        "char_norm": _counts(ref, hyp, normalize_el, char=True),
    }


# ---------------- assemble per-clip records across the four stages -------------
def build_records(rows, hyps_a, hyps_b, hyps_c, hyps_d):
    """Return (records, dropped_uids). A uid missing/empty in ANY stage is dropped
    from ALL stages so paired comparisons stay aligned."""
    stage_hyps = {"A": hyps_a, "B": hyps_b, "C": hyps_c, "D": hyps_d}
    records, dropped = [], []
    for row in rows:
        uid = row["utterance_id"]
        if any(uid not in stage_hyps[s] or stage_hyps[s][uid] is None for s in STAGES):
            dropped.append(uid)
            continue
        ref = row["text"]
        records.append({
            "uid": uid,
            "meeting": (row["city"], row["meeting"]),
            "cats": list(row.get("error_categories") or []),
            "stages": {s: _clip_metrics(ref, stage_hyps[s][uid]) for s in STAGES},
        })
    return records, dropped


# ---------------- micro-averaged WER over a subset -----------------------------
def micro(records, stage: str, metric: str) -> float:
    e = sum(r["stages"][stage][metric][0] for r in records)
    n = sum(r["stages"][stage][metric][1] for r in records)
    return e / n if n else float("nan")


def _by_meeting(records):
    g = defaultdict(list)
    for r in records:
        g[r["meeting"]].append(r)
    return g


# ---------------- meeting-clustered bootstrap ----------------------------------
def bootstrap_ci(records, stage: str, metric: str, n_boot: int = 1000, seed: int = 0):
    g = _by_meeting(records)
    keys = list(g)
    if not keys:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        sample = [r for i in idx for r in g[keys[i]]]
        vals.append(micro(sample, stage, metric))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta_ci(records, s_from: str, s_to: str, metric: str,
                    n_boot: int = 1000, seed: int = 0):
    """Point estimate + CI for (micro[s_from] - micro[s_to]); meetings resampled
    jointly so the two stages share each bootstrap draw."""
    point = micro(records, s_from, metric) - micro(records, s_to, metric)
    g = _by_meeting(records)
    keys = list(g)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        sample = [r for i in idx for r in g[keys[i]]]
        vals.append(micro(sample, s_from, metric) - micro(sample, s_to, metric))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------- per-category breakdown ---------------------------------------
def per_category(records) -> dict:
    cats = sorted({c for r in records for c in r["cats"]})
    out = {}
    for c in cats:
        subset = [r for r in records if c in r["cats"]]
        n_clips = len(subset)
        n_meetings = len({r["meeting"] for r in subset})
        row = {
            "n_clips": n_clips,
            "n_meetings": n_meetings,
            "directional": n_clips < N_CLIPS_MIN or n_meetings < N_MEETINGS_MIN,
        }
        for s in STAGES:
            row[s] = {m: micro(subset, s, m) for m in METRICS}
        out[c] = row
    return out


# ---------------- strict, fail-loud LLM-output extraction ----------------------
def extract_fix(raw: str):
    """Return the single corrected line, or None if the output is not exactly one
    numbered line (ambiguous/garbled -> caller treats as a failure)."""
    lines = [m.group(2).rstrip() for m in (_NUM_RE.match(l) for l in raw.splitlines()) if m]
    return lines[0] if len(lines) == 1 else None


def run_llm_stage(rows, src_hyps, fix_fn):
    """Apply the LLM fix-task per utterance over src_hyps. fix_fn(city, hyp)->raw.
    Returns (hyps, failures); unparseable outputs are recorded, never guessed."""
    hyps, failures = {}, []
    for row in rows:
        uid = row["utterance_id"]
        if uid not in src_hyps or src_hyps[uid] is None:
            continue
        cand = extract_fix(fix_fn(row["city"], src_hyps[uid]))
        if cand is None:
            failures.append(uid)
        else:
            hyps[uid] = cand
    return hyps, failures


# ---------------- report -------------------------------------------------------
def _fmt(v):
    return "nan" if v != v else f"{v:.4f}"


def render_report(records, n_boot=1000, seed=0, meta=None) -> str:
    meta = meta or {}
    L = []
    L.append("# Error-division — Design A pipeline matrix\n")
    if meta.get("generated"):
        L.append(f"_Generated {meta['generated']}._\n")
    L.append("Stages: **A** baseline ASR · **B** baseline+LLM · **C** fine-tuned ASR · "
             "**D** fine-tuned+LLM. Metric: micro-averaged WER (summed edits / summed "
             "reference words); CI = 95% meeting-clustered bootstrap.\n")
    L.append(f"- scored clips: **{len(records)}** across "
             f"**{len({r['meeting'] for r in records})}** meetings")
    if meta.get("seeds") is not None:
        L.append(f"- fine-tune seeds (C/D): {meta['seeds']}")
    if meta.get("dropped"):
        L.append(f"- dropped (unparseable in some stage): {meta['dropped']}")
    if meta.get("llm_failures"):
        L.append(f"- LLM parse failures: {meta['llm_failures']}")
    L.append("")

    # aggregate
    L.append("## Aggregate (all categorized corrections)\n")
    L.append("| stage | WER_norm | 95% CI | WER_raw | CER_norm |")
    L.append("|---|---|---|---|---|")
    for s in STAGES:
        lo, hi = bootstrap_ci(records, s, "word_norm", n_boot, seed)
        L.append(f"| {s} | {_fmt(micro(records, s, 'word_norm'))} | "
                 f"[{_fmt(lo)}, {_fmt(hi)}] | {_fmt(micro(records, s, 'word_raw'))} | "
                 f"{_fmt(micro(records, s, 'char_norm'))} |")
    L.append("")

    # paired deltas — the reads the experiment cares about
    L.append("## Paired deltas (WER_norm, meeting-clustered)\n")
    L.append("Negative = the second stage is better. C-A = fine-tune effect (no LLM); "
             "B-A = LLM effect on baseline; D-C = LLM effect after fine-tune; "
             "D-B = fine-tune effect after LLM.\n")
    L.append("| delta | meaning | point | 95% CI |")
    L.append("|---|---|---|---|")
    for a, b, meaning in (("C", "A", "fine-tune, no LLM"), ("B", "A", "LLM on baseline"),
                          ("D", "C", "LLM after fine-tune"), ("D", "B", "fine-tune after LLM")):
        p, lo, hi = paired_delta_ci(records, a, b, "word_norm", n_boot, seed)
        L.append(f"| {a}-{b} | {meaning} | {_fmt(p)} | [{_fmt(lo)}, {_fmt(hi)}] |")
    L.append("")

    # per category
    L.append("## Per error category (WER_norm)\n")
    L.append("`directional` = n_clips < 30 or n_meetings < 5 — read as a hunch, not a result.\n")
    L.append("| category | n_clips | n_meetings | A | B | C | D | flag |")
    L.append("|---|---|---|---|---|---|---|---|")
    pc = per_category(records)
    for c, d in sorted(pc.items(), key=lambda kv: -kv[1]["n_clips"]):
        flag = "directional" if d["directional"] else "ok"
        L.append(f"| {c} | {d['n_clips']} | {d['n_meetings']} | "
                 f"{_fmt(d['A']['word_norm'])} | {_fmt(d['B']['word_norm'])} | "
                 f"{_fmt(d['C']['word_norm'])} | {_fmt(d['D']['word_norm'])} | {flag} |")
    L.append("")
    return "\n".join(L)


# ============================ real run (main) ==================================
def _default_fix_fn():
    """Lazily build the on-box fix-task callable (verbatim task-v2 prompt). Imported
    here so the pure test never touches the CLI / eval package."""
    sys.path.insert(0, str(ROOT))
    from eval.prompts import SYSTEM_PROMPT, build_user_prompt
    from eval.backends import generate

    def fix_fn(city, hyp):
        up = build_user_prompt(city, [hyp])
        return generate(SYSTEM_PROMPT, up, backend="claude", model="sonnet")

    return fix_fn


def _transcribe(model, proc, rows):
    """Raw per-item Whisper hypotheses, identical decoding to experiment.evaluate."""
    import torch
    import experiment as X
    model.eval()
    fe = proc.feature_extractor
    hyps = {}
    with torch.no_grad():
        for i in range(0, len(rows), X.EVAL_BATCH):
            batch = rows[i:i + X.EVAL_BATCH]
            feats = torch.cat([
                fe(X.load_audio(str(ROOT / r["clip_path"]) if not r["clip_path"].startswith("/") else r["clip_path"]),
                   sampling_rate=X.SR, return_tensors="pt").input_features
                for r in batch], dim=0)
            gen = model.generate(input_features=feats, **X.GEN)
            for r, h in zip(batch, proc.batch_decode(gen, skip_special_tokens=True)):
                hyps[r["utterance_id"]] = h
    return hyps


def main():
    import argparse
    import json
    import time

    import torch
    import experiment as X

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap val clips (debug)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ASR = ROOT / "data" / "asr"
    OUT = ROOT / "data" / "reports" / "error-division"
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    val = [json.loads(l) for l in (ASR / "val_manifest.jsonl").open()]
    val_corr = [r for r in val if r["split"] == "val_corr_cat"]
    if args.limit:
        val_corr = val_corr[: args.limit]
    train_rows = [json.loads(l) for l in (ASR / "manifest.jsonl").open()]
    train = [r for r in train_rows if r["split"] == "train"]
    backbone = [r for r in train_rows if r["split"] == "train_noedit"]
    for r in train + backbone:
        r["clip_path"] = str(ROOT / r["clip_path"])
    log(f"val_corr_cat={len(val_corr)} train={len(train)} backbone={len(backbone)}")

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    proc = WhisperProcessor.from_pretrained("openai/whisper-base")
    base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    base.config.forced_decoder_ids = None
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none"))

    fix_fn = _default_fix_fn()

    # ---- stage A: zero-shot baseline ASR ----
    X.reset_lora(model)
    log("stage A: zero-shot transcribe")
    hyp_a = _transcribe(model, proc, val_corr)
    log("stage B: LLM fix over A")
    hyp_b, fail_b = run_llm_stage(val_corr, hyp_a, fix_fn)

    # ---- stage C/D per seed (composition keeper recipe) ----
    cfg = dict(composition="corr+backbone_1x", sampling="uniform", error_focus="none",
               lr=1e-4, filters="none", steps=args.steps, grad_accum=4, seed=0)
    data = {"train": train, "train_noedit": backbone}
    per_seed = []
    for sd in args.seeds:
        X.reset_lora(model)
        rng = np.random.default_rng(sd)
        items = X.build_train(data["train"], data["train_noedit"], {**cfg, "seed": sd}, rng)
        log(f"seed {sd}: train C on {len(items)} items")
        X.train(model, proc, items, {**cfg, "seed": sd}, log)
        hyp_c = _transcribe(model, proc, val_corr)
        hyp_d, fail_d = run_llm_stage(val_corr, hyp_c, fix_fn)
        recs, dropped = build_records(val_corr, hyp_a, hyp_b, hyp_c, hyp_d)
        per_seed.append({"seed": sd, "records": recs, "dropped": dropped,
                         "fail": sorted(set(fail_b) | set(fail_d))})
        log(f"seed {sd}: scored {len(recs)} (dropped {len(dropped)}, llm_fail {len(set(fail_b)|set(fail_d))})")

    # report seed 0 in full; summarize C/D spread across seeds
    s0 = per_seed[0]
    md = render_report(s0["records"], n_boot=args.n_boot, seed=0, meta={
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "seeds": args.seeds, "dropped": s0["dropped"], "llm_failures": s0["fail"],
    })
    # cross-seed C/D aggregate spread
    if len(per_seed) > 1:
        cvals = [micro(p["records"], "C", "word_norm") for p in per_seed]
        dvals = [micro(p["records"], "D", "word_norm") for p in per_seed]
        md += ("\n## Seed spread (aggregate WER_norm)\n\n"
               f"- C across seeds {args.seeds}: {[round(v,4) for v in cvals]} "
               f"(mean {np.mean(cvals):.4f}, min {min(cvals):.4f}, max {max(cvals):.4f})\n"
               f"- D across seeds {args.seeds}: {[round(v,4) for v in dvals]} "
               f"(mean {np.mean(dvals):.4f}, min {min(dvals):.4f}, max {max(dvals):.4f})\n")
    (OUT / "matrix_categorized.md").write_text(md)
    # dump raw hyps for audit
    audit = {"stage_A": hyp_a, "stage_B": hyp_b,
             "seeds": {p["seed"]: {"dropped": p["dropped"], "fail": p["fail"]} for p in per_seed}}
    (OUT / "matrix_hyps.json").write_text(__import__("json").dumps(audit, ensure_ascii=False, indent=2))
    log(f"wrote {OUT/'matrix_categorized.md'}")


if __name__ == "__main__":
    main()
