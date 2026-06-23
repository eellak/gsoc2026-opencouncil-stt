"""One auto-research experiment = one fine-tune CONFIG, evaluated on val.

Analog of karpathy/autoresearch `train.py`, but the lever is the *data/training
config* (the transferable dimensions), not the architecture. Fixed recipe:
whisper-base + LoRA(r=8, q/v) on CPU, encoder frozen by LoRA scoping. The loop
holds the recipe constant and varies the config so val-WER deltas are
attributable to the data choice.

A config dict:
  composition : 'corrections_only' | 'corr+backbone_1x' | 'corr+backbone_3x'
  sampling    : 'uniform' | 'error_category_balanced' | 'capped_oversample'
  error_focus : 'none' | 'acoustic'
  lr          : float
  filters     : 'none' | 'strict'
  steps, grad_accum, seed

Metric: Greek-normalized WER on val_corr (optimise) with val_reg as the
regression guard (raw WER + CER also reported). Decoding params identical across
all runs (greedy, language=el).
"""
from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import defaultdict

import jiwer
import numpy as np
import soundfile as sf
import torch

SR = 16000
ACOUSTIC = {
    "substitution_phonetic", "homophone", "person_name", "place_name",
    "number_date", "acronym_abbreviation", "org_party_name",
}
# decoding held identical across baseline + every run
GEN = dict(language="el", task="transcribe", num_beams=1, max_new_tokens=128)
EVAL_BATCH = 16


# ---------- Greek text normalization (for normalized WER/CER) ----------
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_el(s: str) -> str:
    s = s.lower().strip()
    # strip diacritics (tonos/dialytika) -> normalized WER is accent-insensitive
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _wer(refs, hyps, normfn):
    r = [normfn(x) for x in refs]
    h = [normfn(x) for x in hyps]
    # guard empty refs
    r2, h2 = zip(*[(a, b) for a, b in zip(r, h) if a.strip()]) if any(x.strip() for x in r) else ([], [])
    if not r2:
        return float("nan")
    return jiwer.wer(list(r2), list(h2))


# ---------- audio ----------
def load_audio(clip_path: str) -> np.ndarray:
    a, sr = sf.read(clip_path, dtype="float32")
    if sr != SR:
        raise ValueError(f"bad sr {sr} in {clip_path}")
    return a


# ---------- LoRA reset (fresh start per config, no reload) ----------
def reset_lora(model) -> None:
    for n, p in model.named_parameters():
        if "lora_A" in n:
            torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        elif "lora_B" in n:
            torch.nn.init.zeros_(p)


# ---------- build a train set from a config ----------
def build_train(train_corr, backbone, cfg, rng):
    """Return a list of (clip_path, text) per the config's data choices."""
    corr = list(train_corr)

    # filters (applied to corrections)
    if cfg["filters"] == "strict":
        corr = [r for r in corr if 1.5 <= r["dur"] <= 20.0 and len(r["text"]) >= 8]

    # sampling / weighting over corrections (by error category)
    def cats(r):
        return r["error_categories"] or ["__none__"]

    if cfg["sampling"] == "uniform":
        pool = list(corr)
    elif cfg["sampling"] == "error_category_balanced":
        by = defaultdict(list)
        for r in corr:
            for c in cats(r):
                by[c].append(r)
        target = max(len(v) for v in by.values()) if by else 0
        pool = []
        for c, v in by.items():
            idx = rng.integers(0, len(v), size=target)
            pool += [v[i] for i in idx]
    elif cfg["sampling"] == "capped_oversample":
        # cap rare categories at the median count so they don't dominate
        by = defaultdict(list)
        for r in corr:
            for c in cats(r):
                by[c].append(r)
        counts = sorted(len(v) for v in by.values())
        cap = counts[len(counts) // 2] if counts else 0
        pool = []
        for c, v in by.items():
            if len(v) <= cap:
                pool += v
            else:
                idx = rng.choice(len(v), size=cap, replace=False)
                pool += [v[i] for i in idx]
    else:
        pool = list(corr)

    # error-type focus: oversample acoustic categories 2x
    if cfg["error_focus"] == "acoustic":
        extra = [r for r in pool if set(cats(r)) & ACOUSTIC]
        pool = pool + extra

    # data composition: add no-edit backbone at a ratio of the corrections volume
    if cfg["composition"] != "corrections_only" and backbone:
        ratio = 1 if cfg["composition"] == "corr+backbone_1x" else 3
        n = min(len(backbone), ratio * len(corr))
        idx = rng.choice(len(backbone), size=n, replace=False)
        pool = pool + [backbone[i] for i in idx]

    rng.shuffle(pool)
    return [(r["clip_path"], r["text"]) for r in pool]


# ---------- training ----------
def train(model, proc, items, cfg, log):
    tok = proc.tokenizer
    tok.set_prefix_tokens(language="el", task="transcribe")
    fe = proc.feature_extractor
    start_tok = model.config.decoder_start_token_id
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["lr"])

    steps, ga = cfg["steps"], cfg["grad_accum"]
    order = list(range(len(items)))
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(order)
    ptr = 0
    t0 = time.time()
    losses = []
    for step in range(steps):
        opt.zero_grad()
        acc = 0.0
        for _ in range(ga):
            if ptr >= len(order):
                rng.shuffle(order); ptr = 0
            clip, text = items[order[ptr]]; ptr += 1
            audio = load_audio(clip)
            feat = fe(audio, sampling_rate=SR, return_tensors="pt").input_features
            ids = tok(text).input_ids
            labels = torch.tensor([ids])
            if labels[0, 0] == start_tok:
                labels = labels[:, 1:]
            loss = model(input_features=feat, labels=labels).loss / ga
            loss.backward()
            acc += loss.item()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        losses.append(acc)
        if step == 0 or (step + 1) % 10 == 0:
            log(f"    step {step+1}/{steps} loss={acc:.3f} ({time.time()-t0:.0f}s)")
    return {"train_seconds": round(time.time() - t0, 1),
            "loss_first": round(losses[0], 3) if losses else None,
            "loss_last": round(losses[-1], 3) if losses else None}


# ---------- evaluation (batched, identical decoding) ----------
@torch.no_grad()
def evaluate(model, proc, rows, log, tag):
    model.eval()
    fe = proc.feature_extractor
    refs = [r["text"] for r in rows]
    cats = [r["error_categories"] for r in rows]
    hyps = []
    t0 = time.time()
    for i in range(0, len(rows), EVAL_BATCH):
        batch = rows[i:i + EVAL_BATCH]
        feats = torch.cat([
            fe(load_audio(r["clip_path"]), sampling_rate=SR, return_tensors="pt").input_features
            for r in batch], dim=0)
        gen = model.generate(input_features=feats, **GEN)
        hyps += proc.batch_decode(gen, skip_special_tokens=True)
    dt = time.time() - t0
    out = {
        "n": len(rows),
        "wer_raw": round(_wer(refs, hyps, lambda x: x.strip()), 4),
        "wer_norm": round(_wer(refs, hyps, normalize_el), 4),
        "cer_norm": round(jiwer.cer([normalize_el(x) for x in refs if x.strip()],
                                    [normalize_el(h) for r, h in zip(refs, hyps) if r.strip()]), 4),
        "eval_seconds": round(dt, 1),
    }
    # per-category normalized WER (val_corr)
    if any(cats):
        bycat = defaultdict(lambda: ([], []))
        for ref, hyp, cs in zip(refs, hyps, cats):
            for c in (cs or []):
                bycat[c][0].append(ref); bycat[c][1].append(hyp)
        out["per_category"] = {
            c: {"n": len(rr), "wer_norm": round(_wer(rr, hh, normalize_el), 4)}
            for c, (rr, hh) in sorted(bycat.items())
        }
    log(f"    eval[{tag}] n={out['n']} wer_norm={out['wer_norm']} "
        f"wer_raw={out['wer_raw']} cer={out['cer_norm']} ({dt:.0f}s)")
    return out


def run_experiment(model, proc, data, cfg, log):
    """Full experiment: reset adapter -> build train -> train -> eval val_corr+val_reg."""
    reset_lora(model)
    rng = np.random.default_rng(cfg["seed"])
    items = build_train(data["train"], data["train_noedit"], cfg, rng)
    log(f"  train items={len(items)} (corr={len(data['train'])} backbone_avail={len(data['train_noedit'])})")
    tr = train(model, proc, items, cfg, log)
    vc = evaluate(model, proc, data["val_corr"], log, "val_corr")
    vr = evaluate(model, proc, data["val_reg"], log, "val_reg")
    return {"cfg": cfg, "n_train": len(items), "train": tr, "val_corr": vc, "val_reg": vr}
