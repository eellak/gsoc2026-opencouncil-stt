#!/usr/bin/env python3
"""Whisper-large-v3 LoRA fine-tune for RunPod (GPU), headless.

Ported from notebooks/whisper_finetune_kaggle.ipynb. Same recipe (large-v3 fp16,
LoRA r32/alpha64 on q/v, frozen encoder, val_corr + val_reg), self-contained (fetches the
export + downloads meeting audio itself — no local dataset needed).

IMPORTANT: this trains on ALL curated included utterances. The PII removal work is
for the FUTURE PUBLIC dataset ONLY — it does NOT filter the training set. Do not
add any PII gating here.

Config via env:
  SMOKE=1  (default)  GPU ACCEPTANCE run: 4 train + 2 val meetings, 1 epoch — just
                      proves the real GPU path (model load, fp16 LoRA, train, eval,
                      save) works and fits VRAM. ~15-30 min.
  SMOKE=0             FULL run: all included meetings, 2 epochs (sweep pick).
  WORK_DIR   (default /workspace/whisper-run)  outputs + clip/manifest cache
  MODEL_ID   (default openai/whisper-large-v3)

Deps (install on the pod first):
  pip install -U transformers datasets peft accelerate evaluate jiwer librosa soundfile
  apt-get install -y ffmpeg
"""
import os, sys, json, time, random, hashlib, pathlib, gc, collections

EXPORT_URL = os.environ.get("EXPORT_URL", "https://79-76-114-184.sslip.io/api/export")
MEETING_API = "https://opencouncil.gr/api/cities/{city}/meetings/{meeting}"
MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-large-v3")
LANGUAGE, TASK = "greek", "transcribe"
VAL_CITIES = {"orestiada", "argos"}
SMOKE = os.environ.get("SMOKE", "1") not in ("0", "false", "False", "")
SAMPLE_N = 60 if SMOKE else None
SMOKE_TRAIN_MEETINGS = 4 if SMOKE else None
SMOKE_VAL_MEETINGS = 2 if SMOKE else None
VAL_REG_PER_MEETING = 8
SR = 16000; PAD_S = 0.2; MIN_DUR, MAX_DUR = 0.3, 30.0; SEED = 13
LORA_R, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.05  # sweep pick
LR, TRAIN_BS, GRAD_ACC, EVAL_BS = 1e-4, 2, 4, 4
EPOCHS = 1 if SMOKE else 2  # sweep: epoch 4 overfit
WORK = pathlib.Path(os.environ.get("WORK_DIR", "/workspace/whisper-run"))
OUT_DIR = str(WORK / "adapter")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)


def log(m): print(f"[train {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    import numpy as np
    np.random.seed(SEED)
    import torch
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
        f"| SMOKE={SMOKE} model={MODEL_ID} epochs={EPOCHS}")
    if not torch.cuda.is_available():
        log("WARNING: no CUDA — this is meant for a GPU pod (will be very slow on CPU)")

    import requests

    # --- fetch export (included/corrected utterances) + denylist ---
    def fetch_jsonl(url):
        r = requests.get(url, timeout=600); r.raise_for_status()
        return [json.loads(l) for l in r.text.splitlines() if l.strip()]
    rows = fetch_jsonl(EXPORT_URL)
    log(f"included rows: {len(rows)} | per city: "
        f"{dict(collections.Counter(r['city_id'] for r in rows).most_common(6))}")
    try:
        _exj = requests.get("https://raw.githubusercontent.com/eellak/gsoc2026-"
                            "opencouncil-stt/main/data/exclusions/unreviewed_meetings.json",
                            timeout=60).json()
        _excl = {(m["city_id"], m["meeting_id"]) for m in _exj.get("meetings", [])}
        _b = len(rows)
        rows = [r for r in rows if (r["city_id"], r["meeting_id"]) not in _excl]
        log(f"denylist: dropped {_b - len(rows)} rows ({len(_excl)} excluded meetings)")
    except Exception as e:
        log(f"denylist filter skipped: {e}")

    def fetch_meeting(city, meeting):
        r = requests.get(MEETING_API.format(city=city, meeting=meeting),
                         headers={"User-Agent": "oc-ft/1.0", "Accept": "application/json"},
                         timeout=120)
        r.raise_for_status(); return r.json()

    # --- audio helpers (decode per meeting, free after) ---
    import librosa, soundfile as sf
    CACHE = pathlib.Path("/tmp/audio_cache"); CACHE.mkdir(parents=True, exist_ok=True)

    def dl(url):
        p = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".mp3")
        if not p.exists():
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(p, "wb") as f:
                    for c in r.iter_content(1 << 20):
                        f.write(c)
        return str(p)

    def cut(y, s, e):
        a = max(0, int((s - PAD_S) * SR)); b = min(len(y), int((e + PAD_S) * SR))
        return y[a:b]

    def ok_span(s, e):
        d = (e or 0) - (s or 0); return MIN_DUR <= d <= MAX_DUR

    # --- build clips + manifest with cache guard (restart doesn't re-decode) ---
    CLIPS = WORK / "clips"; CLIPS.mkdir(parents=True, exist_ok=True)
    MAN_PATH = WORK / "manifest.json"
    # DATA_DIR set -> train on the pre-built COMBINED parquet manifest (28.6h:
    # corrections + no-edit backbone), the curated set. Else self-fetch corrections.
    DATA_DIR = os.environ.get("DATA_DIR")
    if DATA_DIR:
        _pq = {f: (pathlib.Path(DATA_DIR) / f).stat().st_size
               for f in ("train.parquet", "validation.parquet")
               if (pathlib.Path(DATA_DIR) / f).exists()}
        _sig_str = json.dumps({"ver": 3, "sr": SR, "data_dir": DATA_DIR, "smoke": SMOKE,
                               "pad": PAD_S, "dur": [MIN_DUR, MAX_DUR], "parquet": _pq},
                              sort_keys=True)
    else:
        _sig_str = json.dumps({"ver": 2, "sr": SR, "val_cities": sorted(VAL_CITIES),
                               "smoke_train": SMOKE_TRAIN_MEETINGS,
                               "smoke_val": SMOKE_VAL_MEETINGS, "sample_n": SAMPLE_N,
                               "val_reg_per_mtg": VAL_REG_PER_MEETING,
                               "n_included": len(rows)}, sort_keys=True, ensure_ascii=False)

    man = None
    if MAN_PATH.exists():
        _c = json.load(open(MAN_PATH))
        _spot = [c["audio"] for s in ("train", "valc", "valr") for c in _c.get(s, [])[:5]]
        if _c.get("_sig") == _sig_str and all(pathlib.Path(a).exists() for a in _spot):
            man = {k: _c[k] for k in ("train", "valc", "valr")}
            log(f"CACHE HIT -> train={len(man['train'])} valc={len(man['valc'])} "
                f"valr={len(man['valr'])}")
        else:
            log("cache mismatch -> rebuilding")

    if man is None:
        if DATA_DIR:
            man = build_from_parquet(pathlib.Path(DATA_DIR), dl, ok_span, CLIPS,
                                     MAN_PATH, _sig_str, librosa, sf, log)
        else:
            man = build_manifest(rows, fetch_meeting, dl, cut, ok_span, CLIPS, MAN_PATH,
                                 _sig_str, librosa, sf, log)

    # --- HF datasets + Whisper preprocessing ---
    # NB: decode the wav clips OURSELVES with soundfile rather than datasets'
    # Audio() feature — recent `datasets` routes Audio decoding through torchcodec
    # ("please install 'torchcodec'"), a fragile extra dep. Our clips are already
    # cut to 16 kHz mono wav on disk, so a plain soundfile.read is enough.
    import soundfile as sf
    from datasets import Dataset
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)

    def to_ds(recs):
        if not recs:
            return None
        d = Dataset.from_list(recs)

        def prep(b):
            arr, sr = sf.read(b["audio"], dtype="float32")
            b["input_features"] = processor.feature_extractor(
                arr, sampling_rate=sr).input_features[0]
            b["labels"] = processor.tokenizer(b["text"]).input_ids
            return b
        return d.map(prep, remove_columns=["audio", "text"])

    ds_train, ds_valc, ds_valr = to_ds(man["train"]), to_ds(man["valc"]), to_ds(man["valr"])
    gc.collect()
    # fail-fast: never start a long run on an empty/None train or val set (Codex)
    if ds_train is None or ds_train.num_rows == 0:
        sys.exit("[train FATAL] no training clips built — check manifest/audio")
    if ds_valc is None or ds_valc.num_rows == 0:
        sys.exit("[train FATAL] no val_corr clips built — check manifest/audio")
    log(f"datasets: train={ds_train.num_rows} valc={ds_valc.num_rows} "
        f"valr={ds_valr.num_rows if ds_valr else 0}")

    # --- collator + metrics ---
    import unicodedata, re
    from dataclasses import dataclass

    @dataclass
    class Collator:
        processor: object

        def __call__(self, feats):
            batch = self.processor.feature_extractor.pad(
                [{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
            batch["input_features"] = batch["input_features"].to(torch.float16)
            lab = self.processor.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
            labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    collator = Collator(processor)
    import evaluate
    wer_m, cer_m = evaluate.load("wer"), evaluate.load("cer")
    _PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

    def _ws(s):
        return " ".join(s.split())

    def gnorm(s):
        s = unicodedata.normalize("NFD", s.lower())
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = unicodedata.normalize("NFC", s).replace("ς", "σ")
        return re.sub(r"\s+", " ", _PUNCT.sub(" ", s)).strip()

    _DBG = {"n": 0}

    def metrics(pred):
        lab = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        P = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        R = processor.tokenizer.batch_decode(lab, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        if _DBG["n"] < 2:
            for p, r in list(zip(P, R))[:3]:
                print("PRED:", repr(p)); print("REF :", repr(r))
            _DBG["n"] += 1
        return {"wer": 100 * wer_m.compute(predictions=[_ws(x) for x in P],
                                           references=[_ws(x) for x in R]),
                "wer_norm": 100 * wer_m.compute(predictions=[gnorm(x) for x in P],
                                                references=[gnorm(x) for x in R]),
                "cer": 100 * cer_m.compute(predictions=[x.strip() for x in P],
                                           references=[x.strip() for x in R])}

    # --- model + LoRA (freeze encoder) ---
    from transformers import (WhisperForConditionalGeneration,
                              Seq2SeqTrainingArguments, Seq2SeqTrainer)
    from peft import LoraConfig, get_peft_model
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    model.config.suppress_tokens = []
    model.generation_config.language, model.generation_config.task = LANGUAGE, TASK
    model.model.encoder.requires_grad_(False)
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                             lora_dropout=LORA_DROPOUT,
                                             target_modules=["q_proj", "v_proj"]))
    model.print_trainable_parameters()
    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR, per_device_train_batch_size=TRAIN_BS,
        gradient_accumulation_steps=GRAD_ACC, learning_rate=LR, warmup_ratio=0.1,
        num_train_epochs=EPOCHS, fp16=True, predict_with_generate=True,
        generation_max_length=225, eval_strategy="epoch", save_strategy="epoch",
        logging_steps=20, report_to=[], remove_unused_columns=False,
        label_names=["labels"], load_best_model_at_end=True, metric_for_best_model="wer",
        greater_is_better=False, seed=SEED, per_device_eval_batch_size=EVAL_BS,
        save_total_limit=2)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds_train,
                             eval_dataset=ds_valc, data_collator=collator,
                             compute_metrics=metrics, processing_class=processor)

    # --- baseline -> train (resume if a checkpoint exists) -> after ---
    import glob
    ckpts = sorted(glob.glob(os.path.join(OUT_DIR, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else 0)
    resume = ckpts[-1] if ckpts else None   # explicit path, not bool (Codex)
    if resume:
        log(f"RESUMING from {resume}")
    else:
        log(f"BASELINE val_corr: {trainer.evaluate(ds_valc)}")
        if ds_valr:
            log(f"BASELINE val_reg: {trainer.evaluate(ds_valr)}")
    trainer.train(resume_from_checkpoint=resume)
    log(f"AFTER val_corr: {trainer.evaluate(ds_valc)}")
    if ds_valr:
        log(f"AFTER val_reg (regression check): {trainer.evaluate(ds_valr)}")

    # --- save adapter ---
    processor.tokenizer.clean_up_tokenization_spaces = False
    model.save_pretrained(OUT_DIR); processor.save_pretrained(OUT_DIR)
    json.dump({"model": MODEL_ID, "lora_r": LORA_R, "lr": LR, "epochs": EPOCHS,
               "seed": SEED, "smoke": SMOKE, "n_train": ds_train.num_rows,
               "n_val_corr": ds_valc.num_rows,
               "n_val_reg": (ds_valr.num_rows if ds_valr else 0),
               "val_cities": sorted(VAL_CITIES)},
              open(OUT_DIR + "/run_meta.json", "w"), ensure_ascii=False, indent=2)
    log(f"ACCEPTANCE OK — adapter saved -> {OUT_DIR}")


def build_manifest(rows, fetch_meeting, dl, cut, ok_span, CLIPS, MAN_PATH,
                   sig_str, librosa, sf, log):
    """Build train/valc/valr clip manifests (cut clips to disk, cache the manifest)."""
    train_src = [r for r in rows if r["city_id"] not in VAL_CITIES
                 and r.get("final_after_text") and ok_span(r["start"], r["end"])]
    val_src = [r for r in rows if r["city_id"] in VAL_CITIES
               and r.get("final_after_text") and ok_span(r["start"], r["end"])]

    def cap_meetings(src, n):
        if not n:
            return src
        mids = list({(r["city_id"], r["meeting_id"]) for r in src})
        random.shuffle(mids); keep = set(mids[:n])
        return [r for r in src if (r["city_id"], r["meeting_id"]) in keep]

    train_src = cap_meetings(train_src, SMOKE_TRAIN_MEETINGS)
    val_src = cap_meetings(val_src, SMOKE_VAL_MEETINGS)
    if SAMPLE_N:
        random.shuffle(train_src); train_src = train_src[:SAMPLE_N]
    log(f"sources: train_src={len(train_src)} "
        f"({len({(r['city_id'], r['meeting_id']) for r in train_src})} mtgs) "
        f"val_src={len(val_src)} "
        f"({len({(r['city_id'], r['meeting_id']) for r in val_src})} mtgs)")

    # val_reg: no-edit utterances from the val meetings (regression slice)
    reg_src = []
    for city, mtg in sorted({(r["city_id"], r["meeting_id"]) for r in val_src}):
        try:
            mj = fetch_meeting(city, mtg)
        except Exception as e:
            log(f"skip {city} {mtg}: {e}"); continue
        au = (mj.get("meeting") or {}).get("audioUrl")
        if not au:
            continue
        ne = [u for seg in (mj.get("transcript") or []) for u in (seg.get("utterances") or [])
              if u.get("lastModifiedBy") is None
              and ok_span(u.get("startTimestamp"), u.get("endTimestamp"))
              and (u.get("text") or "").strip()]
        random.shuffle(ne)
        for u in ne[:VAL_REG_PER_MEETING]:
            reg_src.append({"city_id": city, "meeting_id": mtg, "audio_url": au,
                            "start": u["startTimestamp"], "end": u["endTimestamp"],
                            "final_after_text": u["text"]})

    def build(src, tag):
        by_mtg = collections.defaultdict(list)
        for r in src:
            by_mtg[(r["city_id"], r["meeting_id"], r.get("audio_url"))].append(r)
        out = []
        for (city, mtg, url), items in by_mtg.items():
            au = url or items[0].get("audio_url")
            if not au:
                continue
            try:
                y = librosa.load(dl(au), sr=SR, mono=True)[0]
            except Exception as e:
                log(f"audio fail {city}/{mtg}: {e}"); continue
            d = CLIPS / tag / city / mtg
            d.mkdir(parents=True, exist_ok=True)
            for i, r in enumerate(items):
                clip = cut(y, r["start"], r["end"])
                if len(clip) < int(MIN_DUR * SR):
                    continue
                p = d / f"{i}.wav"
                sf.write(str(p), clip, SR)
                out.append({"audio": str(p), "text": r["final_after_text"]})
            del y; gc.collect()
        log(f"built {tag}: {len(out)} clips")
        return out

    man = {"train": build(train_src, "train"),
           "valc": build(val_src, "valc"),
           "valr": build(reg_src, "valr")}
    save = dict(man); save["_sig"] = sig_str
    json.dump(save, open(MAN_PATH, "w"), ensure_ascii=False)
    return man


def build_from_parquet(data_dir, dl, ok_span, CLIPS, MAN_PATH, sig_str, librosa, sf, log):
    """Build clips from the pre-built COMBINED manifest (data/hf-dataset/public).

    Cuts on the boundary-corrected span (start_adj/end_adj) when present, else
    raw start/end. val split: correction rows -> valc, no_edit rows -> valr.
    Downloads each meeting's mp3 once (cached), cuts all its clips, frees it."""
    import numpy as np
    import pandas as pd
    tr = pd.read_parquet(data_dir / "train.parquet")
    va = pd.read_parquet(data_dir / "validation.parquet")

    def cap(df):
        if not SMOKE:
            return df
        mids = sorted({(c, m) for c, m in zip(df.city_id, df.meeting_id)})
        random.shuffle(mids); keep = set(mids[:4])
        return df[[(c, m) in keep for c, m in zip(df.city_id, df.meeting_id)]]

    def span(r):
        sa, ea = r.get("start_adj"), r.get("end_adj")
        good = lambda x: x is not None and not (isinstance(x, float) and np.isnan(x))
        return (float(sa), float(ea)) if good(sa) and good(ea) else (float(r["start"]), float(r["end"]))

    def build(df, tag):
        recs = df.to_dict("records")
        by_mtg = collections.defaultdict(list)
        for r in recs:
            by_mtg[(r["city_id"], r["meeting_id"], r["audio_url"])].append(r)
        out, n_mtg = [], 0
        for (city, mtg, au), items in sorted(by_mtg.items()):
            n_mtg += 1
            if not au or (isinstance(au, float) and np.isnan(au)):
                log(f"skip {city}/{mtg}: null audio_url"); continue
            try:
                mp3 = dl(au)
                y = librosa.load(mp3, sr=SR, mono=True)[0]
            except Exception as e:
                log(f"audio fail {city}/{mtg}: {str(e)[:60]}"); continue
            d = CLIPS / tag / city / mtg; d.mkdir(parents=True, exist_ok=True)
            for r in items:
                s, e = span(r)
                if not ok_span(s, e):
                    continue
                a = max(0, int((s - PAD_S) * SR)); b = min(len(y), int((e + PAD_S) * SR))
                clip = y[a:b]
                if len(clip) < int(MIN_DUR * SR):
                    continue
                p = d / f"{r['utterance_id']}.wav"
                sf.write(str(p), clip, SR)
                out.append({"audio": str(p), "text": r["text"]})
            del y; gc.collect()
            # free the meeting mp3 (each meeting processed once; keeps disk bounded
            # on an 80GB pod vs caching all ~367 full-meeting mp3s ~37GB) (Codex)
            try:
                os.remove(mp3)
            except OSError:
                pass
            if n_mtg % 20 == 0:
                log(f"  {tag}: {n_mtg}/{len(by_mtg)} meetings, {len(out)} clips")
        log(f"built {tag}: {len(out)} clips from {len(by_mtg)} meetings")
        return out

    tr, va = cap(tr), cap(va)
    log(f"parquet manifest: train={len(tr)} val={len(va)} "
        f"(valc={int((va['source'] == 'correction').sum())} "
        f"valr={int((va['source'] == 'no_edit').sum())})")
    man = {"train": build(tr, "train"),
           "valc": build(va[va["source"] == "correction"], "valc"),
           "valr": build(va[va["source"] == "no_edit"], "valr")}
    save = dict(man); save["_sig"] = sig_str
    json.dump(save, open(MAN_PATH, "w"), ensure_ascii=False)
    return man


if __name__ == "__main__":
    main()
