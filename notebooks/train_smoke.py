#!/usr/bin/env python3
"""CPU-only pipeline smoke test for the Whisper LoRA fine-tune.

Purpose: prove the WHOLE pipeline loads and runs end to end on a machine with no
usable GPU (the mini PC) using minimal data — fetch export, decode audio, cut
clips, processor, model + LoRA, ONE train step, ONE eval with generate, save the
adapter. It is NOT meant to produce good metrics.

Defaults to openai/whisper-tiny for fast per-minute confidence; pass
--model-id openai/whisper-large-v3 for a real-model acceptance run (slow on CPU).

Derived from notebooks/whisper_finetune_kaggle.ipynb. Reviewed with Codex
(2026-07-14): CPU float32 everywhere, bound by max_steps not epochs, decode audio
ourselves (the HF datasets Audio() interface changed to TorchCodec), jiwer instead
of evaluate.load so no metric download is needed, and fail-fast ordering so model
load fails before we spend time downloading hours of audio.
"""
import argparse, sys, os, json, time, hashlib, gc, shutil, collections, pathlib

EXPORT_URL_DEFAULT = "https://79-76-114-184.sslip.io/api/export"
MEETING_API = "https://opencouncil.gr/api/cities/{city}/meetings/{meeting}"
LANGUAGE, TASK = "greek", "transcribe"
SR = 16000
PAD_S = 0.2
MIN_DUR, MAX_DUR = 0.3, 30.0


def log(msg):
    print(f"[smoke {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg, code=1):
    print(f"[smoke FAIL] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def ok_span(s, e):
    d = (e or 0) - (s or 0)
    return MIN_DUR <= d <= MAX_DUR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="openai/whisper-tiny",
                    help="whisper model (default tiny; use openai/whisper-large-v3 for acceptance)")
    ap.add_argument("--export-url", default=EXPORT_URL_DEFAULT)
    ap.add_argument("--train-meetings", type=int, default=1)
    ap.add_argument("--val-meetings", type=int, default=1)
    ap.add_argument("--clips-per-meeting", type=int, default=4)
    ap.add_argument("--out-dir", default="/tmp/whisper-smoke-out")
    ap.add_argument("--audio-cache", default="/tmp/whisper-smoke-audio")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    # ---- STEP 1: preflight (imports, ffmpeg, writability) — fail before any network work ----
    log(f"preflight: python {sys.version.split()[0]}, model_id={args.model_id}")
    if not shutil.which("ffmpeg"):
        die("ffmpeg not found on PATH (librosa needs it to decode mp3)")
    try:
        import torch, numpy as np, requests, librosa, soundfile  # noqa: F401
        import jiwer
        from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                                   Seq2SeqTrainingArguments, Seq2SeqTrainer)
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
    except Exception as e:
        die(f"import failed: {type(e).__name__}: {e}")
    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.out_dir, exist_ok=True)
    probe = pathlib.Path(args.out_dir) / ".writable"
    probe.write_text("ok"); probe.unlink()
    log(f"preflight OK: torch {torch.__version__}, cuda={torch.cuda.is_available()} (expected False on mini PC)")

    # ---- STEP 2: fetch export + pick the exact meetings/clips ----
    log(f"fetching export: {args.export_url}")
    try:
        r = requests.get(args.export_url, timeout=600); r.raise_for_status()
        rows = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    except Exception as e:
        die(f"export fetch failed: {type(e).__name__}: {e}")
    log(f"export rows: {len(rows)}")
    usable = [x for x in rows if x.get("final_after_text") and x.get("audio_url")
              and ok_span(x.get("start"), x.get("end"))]
    if not usable:
        die("no usable rows (need final_after_text + audio_url + valid span)")
    by_mtg = collections.defaultdict(list)
    for x in usable:
        by_mtg[(x["city_id"], x["meeting_id"])].append(x)
    mids = [m for m, v in by_mtg.items() if len(v) >= 1]
    random.shuffle(mids)
    n_need = args.train_meetings + args.val_meetings
    if len(mids) < n_need:
        die(f"need {n_need} meetings, only {len(mids)} available")
    train_mids = mids[:args.train_meetings]
    val_mids = mids[args.train_meetings:n_need]

    def take(mid_list, dst):
        out = []
        for mid in mid_list:
            clips = by_mtg[mid][:args.clips_per_meeting]
            for c in clips:
                out.append({"url": c["audio_url"], "start": c["start"], "end": c["end"],
                            "text": (c["final_after_text"] or "").strip(), "dst": dst})
        return out
    tasks = take(train_mids, "train") + take(val_mids, "val")
    log(f"selected train_mtgs={train_mids} val_mtgs={val_mids} -> {len(tasks)} clips total")

    # ---- STEP 3: metrics (local jiwer, no download) ----
    def wer_cer(refs, preds):
        refs2 = [x if x.strip() else "." for x in refs]
        return jiwer.wer(refs2, preds) * 100, jiwer.cer(refs2, preds) * 100

    # ---- STEP 4: processor + model + LoRA (assert it actually attached) BEFORE audio download ----
    log(f"loading processor + model: {args.model_id}")
    processor = WhisperProcessor.from_pretrained(args.model_id, language=LANGUAGE, task=TASK)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id, dtype=torch.float32)
    except TypeError:  # older transformers uses torch_dtype
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = True  # smoke: no gradient checkpointing, keep cache for fast generate
    lconf = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lconf)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        die("LoRA attached 0 trainable params — target_modules did not match")
    n_lora = sum(1 for n, _ in model.named_modules() if "lora_" in n)
    if n_lora == 0:
        die("no lora_ submodules inserted")
    log(f"LoRA OK: trainable={n_trainable:,} params, {n_lora} lora modules")

    # ---- STEP 5+6: download selected audio, decode ONE meeting at a time, cut+feature-extract ----
    cache = pathlib.Path(args.audio_cache); cache.mkdir(parents=True, exist_ok=True)

    def dl(url):
        p = cache / (hashlib.md5(url.encode()).hexdigest() + ".mp3")
        if not p.exists():
            with requests.get(url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(p, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
        return str(p)

    def cut(y, start, end):
        a = max(0, int((start - PAD_S) * SR)); b = min(len(y), int((end + PAD_S) * SR))
        return y[a:b]

    buckets = collections.defaultdict(list)
    for t in tasks:
        buckets[t["url"]].append(t)
    recs = {"train": [], "val": []}
    for i, (url, ts) in enumerate(buckets.items(), 1):
        log(f"audio {i}/{len(buckets)}: downloading + decoding one meeting mp3 ...")
        try:
            path = dl(url)
            y = librosa.load(path, sr=SR, mono=True)[0]
        except Exception as e:
            die(f"audio load failed for {url}: {type(e).__name__}: {e}")
        for t in ts:
            clip = cut(y, t["start"], t["end"])
            if len(clip) < int(MIN_DUR * SR):
                continue
            feat = processor.feature_extractor(clip, sampling_rate=SR).input_features[0]
            labels = processor.tokenizer(t["text"]).input_ids
            recs[t["dst"]].append({"input_features": feat, "labels": labels})
        del y; gc.collect()
        try: os.remove(path)
        except Exception: pass
    log(f"built clips: train={len(recs['train'])} val={len(recs['val'])}")
    if not recs["train"] or not recs["val"]:
        die("empty train or val after decode — cannot smoke the train/eval path")

    ds_train = Dataset.from_list(recs["train"])
    ds_val = Dataset.from_list(recs["val"])

    # ---- collator: keep float32, mask pad labels, strip leading bos ----
    import torch as _t
    class Collator:
        def __init__(self, proc): self.proc = proc
        def __call__(self, feats):
            batch = self.proc.feature_extractor.pad(
                [{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
            batch["input_features"] = batch["input_features"].to(_t.float32)
            lab = self.proc.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
            labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.proc.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch
    collator = Collator(processor)

    def compute_metrics(pred):
        import numpy as _np
        lab = _np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        P = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        R = processor.tokenizer.batch_decode(lab, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        w, c = wer_cer([x.strip() for x in R], [x.strip() for x in P])
        return {"wer": w, "cer": c}

    # ---- STEP 7: train ONE step, eval, save adapter ----
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out_dir, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=1e-4, max_steps=1, warmup_steps=0,
        fp16=False, bf16=False, use_cpu=True, predict_with_generate=True,
        generation_max_length=32, generation_num_beams=1, eval_strategy="no", save_strategy="no",
        logging_steps=1, report_to=[], remove_unused_columns=False, label_names=["labels"],
        optim="adamw_torch", dataloader_pin_memory=False, dataloader_num_workers=0, seed=args.seed)
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
                             data_collator=collator, compute_metrics=compute_metrics,
                             processing_class=processor)

    log("BASELINE eval (generate on val) ...")
    base = trainer.evaluate(ds_val)
    log(f"BASELINE: wer={base.get('eval_wer'):.1f} cer={base.get('eval_cer'):.1f}")
    log("training ONE step ...")
    trainer.train()
    log("AFTER eval ...")
    after = trainer.evaluate(ds_val)
    log(f"AFTER: wer={after.get('eval_wer'):.1f} cer={after.get('eval_cer'):.1f}")

    adapter_dir = os.path.join(args.out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    weights = [f for f in os.listdir(adapter_dir) if f.startswith("adapter_model")]
    if not os.path.exists(cfg) or not weights:
        die(f"adapter not saved correctly (config={os.path.exists(cfg)} weights={weights})")
    log(f"adapter saved -> {adapter_dir} ({weights})")
    log("SMOKE PASSED: pipeline loaded and ran end to end (fetch -> decode -> LoRA -> train -> eval -> save)")


if __name__ == "__main__":
    main()
