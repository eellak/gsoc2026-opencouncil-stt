#!/usr/bin/env python3
"""A/B: what did the Whisper label-prefix bug cost?

Paired LoRA fine-tunes of whisper-large-v3 on identical data, identical step
count, repeated over several seeds. The ONLY difference between the two arms is
the collator:

  fixed   — labels drop the leading <|startoftranscript|> (decoder_start_token_id),
            so shift_tokens_right rebuilds the canonical decoder input.
  legacy  — the pre-2026-07-31 code verbatim: strip only if labels[:, 0] equals
            tokenizer.bos_token_id, which for Whisper is <|endoftext|> (50257) and
            never matches <|startoftranscript|> (50258). The strip never fires, the
            decoder input carries a doubled SOT, and every target sits one learned
            position later than at inference.

Every published GPU number came from the legacy arm. This puts a number on the gap.

WHAT THIS DOES AND DOES NOT CLAIM. It measures the bug's effect under a scaled-down
recipe (fewer meetings, 300 steps) that is identical for both arms — not the exact
8h historical run. It is a controlled comparison of two training objectives, not a
reproduction of the published adapter. Uncertainty is reported as a paired cluster
bootstrap over held-out MEETINGS (utterances within a meeting are correlated; an
utterance-level interval over a couple of meetings would be badly overconfident).

Outputs:
  eval/ab_label_bug/results_ab.json    aggregates only — tracked in the repo
  $WORK_DIR/results_ab_detail.json     per-utterance text — PII, stays off the repo

Env:
  AB_TRAIN_MEETINGS (24)   training meetings, largest-first by corrected rows
  AB_VAL_MEETINGS   (16)   held-out val meetings (orestiada + argos), largest-first
  AB_MAX_STEPS      (300)  optimizer steps per arm
  AB_SEEDS   (13,23,37)    training seeds; each seed runs both arms
  AB_BOOTSTRAP    (10000)  bootstrap resamples
  AB_PREPARE_ONLY   (0)    build+cache clips and features, then exit
  WORK_DIR (/workspace/ab-run)   MODEL_ID (openai/whisper-large-v3)
"""
import os, sys, json, time, random, hashlib, pathlib, gc, collections, re, unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "notebooks"))
import train_runpod as tr   # noqa: E402  (Collator + manifest builder under test)

MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-large-v3")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
LANGUAGE, TASK = "greek", "transcribe"
DATA_SEED = 13                      # data selection + ordering: fixed across arms
SR = tr.SR
WORK = pathlib.Path(os.environ.get("WORK_DIR", "/workspace/ab-run"))
RESULTS = pathlib.Path(__file__).resolve().parent / "results_ab.json"
TRAIN_MEETINGS = int(os.environ.get("AB_TRAIN_MEETINGS", "24"))
VAL_MEETINGS = int(os.environ.get("AB_VAL_MEETINGS", "16"))
MAX_STEPS = int(os.environ.get("AB_MAX_STEPS", "300"))
SEEDS = [int(s) for s in os.environ.get("AB_SEEDS", "13,23,37").split(",") if s.strip()]
N_BOOT = int(os.environ.get("AB_BOOTSTRAP", "10000"))
# Same recipe as the published run, so the measured gap is the bug's, not a
# different hyperparameter's.
LORA_R, LORA_ALPHA, LORA_DROPOUT = tr.LORA_R, tr.LORA_ALPHA, tr.LORA_DROPOUT
LR, TRAIN_BS, GRAD_ACC, EVAL_BS = tr.LR, tr.TRAIN_BS, tr.GRAD_ACC, tr.EVAL_BS


def log(m): print(f"[ab {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- scoring
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def gnorm(s):
    """Greek-insensitive normalization — the same one the training script scores with."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFC", s).replace("ς", "σ")
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s)).strip()


def edits(ref_toks, hyp_toks):
    prev = list(range(len(hyp_toks) + 1))
    for i, r in enumerate(ref_toks, 1):
        cur = [i]
        for j, h in enumerate(hyp_toks, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def per_utt_counts(refs, hyps):
    """(word_errors, word_len, char_errors, char_len) per utterance, normalized."""
    out = []
    for r, h in zip(refs, hyps):
        rn, hn = gnorm(r), gnorm(h)
        rw, hw = rn.split(), hn.split()
        out.append((edits(rw, hw), len(rw), edits(list(rn), list(hn)), len(rn)))
    return out


def agg_wer(counts):
    e = sum(c[0] for c in counts); n = sum(c[1] for c in counts)
    return e / n if n else float("nan")


def agg_cer(counts):
    e = sum(c[2] for c in counts); n = sum(c[3] for c in counts)
    return e / n if n else float("nan")


def cluster_bootstrap(counts_a, counts_b, meetings, n_boot, seed=7):
    """Paired CI on WER(a) - WER(b), resampling MEETINGS, not utterances.

    Utterances from one meeting share a speaker, a room and an acoustic setup, so
    they are not independent draws. Resampling them individually would report a
    tight interval that does not survive a new meeting. Resampling whole meetings
    (jointly for both arms, since the references are identical) is the honest unit.
    """
    import numpy as np
    assert len(counts_a) == len(counts_b) == len(meetings) > 0
    assert all(a[1] == b[1] for a, b in zip(counts_a, counts_b)), \
        "arms scored against different references — the pairing is broken"
    groups = collections.defaultdict(list)
    for i, m in enumerate(meetings):
        groups[m].append(i)
    keys = sorted(groups)
    a = np.array([(c[0], c[1]) for c in counts_a], dtype=float)
    b = np.array([(c[0], c[1]) for c in counts_b], dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        den = a[idx, 1].sum()
        diffs[i] = (a[idx, 0].sum() - b[idx, 0].sum()) / den if den else np.nan
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    point = agg_wer(counts_a) - agg_wer(counts_b)
    # Per-meeting deltas: with few clusters these carry more information than the
    # interval does — a consistent sign across meetings is the real evidence.
    per_meeting = {}
    for m in keys:
        idx = groups[m]
        da = sum(counts_a[i][0] for i in idx); na = sum(counts_a[i][1] for i in idx)
        db = sum(counts_b[i][0] for i in idx)
        per_meeting[m] = (da - db) / na if na else float("nan")
    pos = sum(1 for v in per_meeting.values() if v > 0)
    return {"delta_wer": point, "ci95_cluster": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_clusters": len(keys),
            "bootstrap_wrong_sign_fraction": float(
                (diffs <= 0).mean() if point > 0 else (diffs >= 0).mean()),
            "meetings_a_worse": pos, "meetings_total": len(keys),
            "per_meeting_delta": per_meeting}


# ---------------------------------------------------------------- legacy arm
class LegacyCollator:
    """The pre-fix collator, verbatim. Its strip condition is never true.

    Kept as a literal copy of the old code rather than a hardcoded "don't strip",
    so the arm reproduces the historical runs rather than our story about them.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, feats):
        import torch
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


class TracingCollator:
    """Wrap a collator and record the example order it is fed.

    Both arms must see the same examples in the same order. `seed`+`data_seed`
    should guarantee that, but a library upgrade can silently change the sampler,
    and that failure would look exactly like a training effect.
    """

    def __init__(self, inner):
        self.inner = inner
        self.trace = []

    def __call__(self, feats):
        self.trace.append(tuple(int(f["ex_id"]) for f in feats))
        return self.inner(feats)

    def trace_hash(self):
        return hashlib.sha256(repr(self.trace).encode()).hexdigest()[:16]


def main():
    import numpy as np
    import torch
    import transformers, datasets as hfds, peft
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed)
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    import soundfile as sf
    import librosa, requests

    WORK.mkdir(parents=True, exist_ok=True)
    log(f"torch {torch.__version__} transformers {transformers.__version__} "
        f"peft {peft.__version__} datasets {hfds.__version__} "
        f"cuda={torch.cuda.is_available()} "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if not torch.cuda.is_available():
        sys.exit("[ab FATAL] needs a GPU — two large-v3 fine-tunes on CPU is not a plan")

    # ---------------- data: selected once, shared by every arm and seed --------
    random.seed(DATA_SEED); np.random.seed(DATA_SEED)
    r = requests.get(tr.EXPORT_URL, timeout=600); r.raise_for_status()
    export_sha = hashlib.sha256(r.content).hexdigest()[:16]
    rows = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    log(f"export: {len(rows)} rows sha={export_sha}")

    # Fail closed on the denylist: it exists because those meetings are unreviewed.
    # Silently proceeding without it would change the sample and the reason for it.
    dl_resp = requests.get("https://raw.githubusercontent.com/eellak/gsoc2026-"
                           "opencouncil-stt/main/data/exclusions/unreviewed_meetings.json",
                           timeout=60)
    dl_resp.raise_for_status()
    denylist_sha = hashlib.sha256(dl_resp.content).hexdigest()[:16]
    _excl = {(m["city_id"], m["meeting_id"]) for m in dl_resp.json().get("meetings", [])}
    _b = len(rows)
    rows = [r_ for r_ in rows if (r_["city_id"], r_["meeting_id"]) not in _excl]
    log(f"denylist sha={denylist_sha}: dropped {_b - len(rows)} rows")

    # Pick meetings largest-first rather than at random. A random draw over 256
    # meetings whose median is 13 corrected rows spends most of the audio-download
    # budget on almost-empty meetings; both arms get the identical sample either
    # way, and this one buys ~5x the training clips per meeting downloaded.
    def top_meetings(pool, n):
        c = collections.Counter((x["city_id"], x["meeting_id"]) for x in pool)
        return {m for m, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]}

    train_pool = [x for x in rows if x["city_id"] not in tr.VAL_CITIES]
    val_pool = [x for x in rows if x["city_id"] in tr.VAL_CITIES]
    keep = top_meetings(train_pool, TRAIN_MEETINGS) | top_meetings(val_pool, VAL_MEETINGS)
    rows = [x for x in rows if (x["city_id"], x["meeting_id"]) in keep]
    log(f"selected {len(keep)} meetings -> {len(rows)} corrected rows")
    # The production builder caps meetings by random shuffle; we already chose them.
    tr.SMOKE_TRAIN_MEETINGS = None
    tr.SMOKE_VAL_MEETINGS = None
    tr.SAMPLE_N = None

    CACHE = pathlib.Path("/tmp/audio_cache"); CACHE.mkdir(parents=True, exist_ok=True)

    def dl(url):
        p = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".mp3")
        if not p.exists():
            with requests.get(url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(p, "wb") as f:
                    for c in resp.iter_content(1 << 20):
                        f.write(c)
        return str(p)

    def fetch_meeting(city, meeting):
        resp = requests.get(tr.MEETING_API.format(city=city, meeting=meeting),
                            headers={"User-Agent": "oc-ab/1.0", "Accept": "application/json"},
                            timeout=120)
        resp.raise_for_status(); return resp.json()

    def cut(y, s, e):
        a = max(0, int((s - tr.PAD_S) * SR)); b = min(len(y), int((e + tr.PAD_S) * SR))
        return y[a:b]

    def ok_span(s, e):
        d = (e or 0) - (s or 0); return tr.MIN_DUR <= d <= tr.MAX_DUR

    CLIPS = WORK / "clips"; CLIPS.mkdir(parents=True, exist_ok=True)
    MAN_PATH = WORK / "manifest.json"
    sig = json.dumps({"ab": 3, "train_mtgs": TRAIN_MEETINGS, "val_mtgs": VAL_MEETINGS,
                      "export_sha": export_sha, "denylist_sha": denylist_sha,
                      "seed": DATA_SEED, "model": MODEL_ID, "rev": MODEL_REVISION,
                      "tf": transformers.__version__}, sort_keys=True)
    man = None
    if MAN_PATH.exists():
        _c = json.load(open(MAN_PATH))
        if _c.get("_sig") == sig and all(
                pathlib.Path(c["audio"]).exists()
                for s in ("train", "valc", "valr") for c in _c.get(s, [])[:5]):
            man = {k: _c[k] for k in ("train", "valc", "valr")}
            log("manifest CACHE HIT")
    if man is None:
        man = tr.build_manifest(rows, fetch_meeting, dl, cut, ok_span, CLIPS, MAN_PATH,
                                sig, librosa, sf, log)

    processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                 language=LANGUAGE, task=TASK)
    feat_cache = WORK / "featcache"; feat_cache.mkdir(parents=True, exist_ok=True)
    # The arrow cache is keyed by FILENAME when cache_file_name is explicit, so
    # datasets will happily reload features built from a different sample or a
    # different feature extractor. Fingerprint everything that shapes them.
    sig_tag = hashlib.sha256((sig + processor.feature_extractor.to_json_string()
                              + processor.tokenizer.name_or_path).encode()).hexdigest()[:10]

    def meeting_of(clip_path):
        """city/meeting from CLIPS/<tag>/<city>/<meeting>/<i>.wav — the cluster id."""
        p = pathlib.Path(clip_path).parts
        return f"{p[-3]}/{p[-2]}"

    def to_ds(recs, name):
        if not recs:
            return None
        d = Dataset.from_list([dict(r_, ex_id=i, meeting=meeting_of(r_["audio"]))
                               for i, r_ in enumerate(recs)])

        def prep(b):
            arr, srate = sf.read(b["audio"], dtype="float32")
            b["input_features"] = processor.feature_extractor(
                arr, sampling_rate=srate).input_features[0]
            b["labels"] = processor.tokenizer(b["text"]).input_ids
            return b
        return d.map(prep, remove_columns=["audio"], keep_in_memory=False,
                     writer_batch_size=200,
                     cache_file_name=str(feat_cache / f"{name}.{sig_tag}.arrow"))

    ds_train = to_ds(man["train"], "train")
    ds_valc = to_ds(man["valc"], "valc")
    ds_valr = to_ds(man["valr"], "valr")
    if ds_train is None or ds_valc is None:
        sys.exit("[ab FATAL] empty train or val_corr")

    # ---------------- preflight: check EVERY row, before renting time on it ----
    sot = 50258
    tok = processor.tokenizer
    max_pos = 448
    lens = []
    for split, ds in (("train", ds_train), ("valc", ds_valc), ("valr", ds_valr)):
        if ds is None:
            continue
        for i, ids in enumerate(ds["labels"]):
            if not ids or ids[0] != sot:
                sys.exit(f"[ab FATAL] {split}[{i}] labels do not start with {sot}: {ids[:6]}")
            # the legacy arm keeps one extra token — check the longer of the two
            if len(ids) > max_pos:
                sys.exit(f"[ab FATAL] {split}[{i}] label length {len(ids)} > {max_pos}; "
                         f"it would crash only once its shuffled batch came up")
            if split == "train":
                lens.append(len(ids))
    assert tok.bos_token_id != sot, (
        "bos_token_id equals the first label token — the legacy arm would strip "
        "correctly and there is no bug to measure on this build")
    assert getattr(tok, "padding_side", "right") == "right"
    train_meetings = {m for m in ds_train["meeting"]}
    val_meetings = {m for m in ds_valc["meeting"]}
    if train_meetings & val_meetings:
        sys.exit(f"[ab FATAL] train/val meeting overlap: {train_meetings & val_meetings}")
    eff_epochs = MAX_STEPS * TRAIN_BS * GRAD_ACC / ds_train.num_rows
    log(f"data: train={ds_train.num_rows} ({len(train_meetings)} mtgs) "
        f"valc={ds_valc.num_rows} ({len(val_meetings)} mtgs) "
        f"valr={ds_valr.num_rows if ds_valr else 0}")
    log(f"exposure: {MAX_STEPS} steps x {TRAIN_BS}x{GRAD_ACC} = "
        f"{MAX_STEPS * TRAIN_BS * GRAD_ACC} presentations = {eff_epochs:.2f} epochs | "
        f"median target len {sorted(lens)[len(lens) // 2]} tokens")

    if os.environ.get("AB_PREPARE_ONLY") in ("1", "true", "True"):
        log("AB_PREPARE_ONLY — clips and features cached, exiting before training")
        return

    refs_c = list(ds_valc["text"])
    refs_r = list(ds_valr["text"]) if ds_valr else []
    mtg_c = list(ds_valc["meeting"])
    mtg_r = list(ds_valr["meeting"]) if ds_valr else []

    # ---------------- shared decoding + diagnostics ---------------------------
    @torch.no_grad()
    def transcribe(model, ds, tag):
        """Greedy decode. Identical settings for the baseline and both arms."""
        model.eval()
        hyps, n_capped, t0 = [], 0, time.time()
        for i in range(0, ds.num_rows, EVAL_BS):
            chunk = ds[i:i + EVAL_BS]
            feats = torch.tensor(np.array(chunk["input_features"]),
                                 dtype=torch.float16, device=model.device)
            out = model.generate(input_features=feats, language=LANGUAGE, task=TASK,
                                 do_sample=False, num_beams=1, max_new_tokens=400,
                                 use_cache=True)
            n_capped += int((out.shape[1] >= 400) and 1)
            hyps += processor.tokenizer.batch_decode(out, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
        log(f"  [{tag}] decoded {len(hyps)} clips in {time.time() - t0:.0f}s "
            f"({n_capped} batches hit the length cap)")
        return hyps, n_capped

    @torch.no_grad()
    def sot_diagnostic(model, ds, n=64):
        """Does the model want to emit <|startoftranscript|> as its FIRST token?

        The legacy target sequence begins with SOT at decoder position 0, so that
        is where the learned habit lives: P(SOT | decoder_input=[SOT]). Reading it
        further along the canonical prompt would ask a different question and find
        nothing. It also cannot be read off generated text — large-v3's
        generation_config suppresses 50258 — so this goes to the raw logits, before
        any logits processor runs.
        """
        model.eval()
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        s = base.config.decoder_start_token_id
        lang_id = processor.tokenizer.convert_tokens_to_ids("<|el|>")
        p_sot, rank_sot, p_lang = [], [], []
        for i in range(0, min(n, ds.num_rows), EVAL_BS):
            chunk = ds[i:i + EVAL_BS]
            feats = torch.tensor(np.array(chunk["input_features"]),
                                 dtype=torch.float16, device=model.device)
            dec = torch.full((feats.shape[0], 1), s, dtype=torch.long, device=model.device)
            logits = model(input_features=feats, decoder_input_ids=dec,
                           use_cache=False).logits[:, 0, :].float()
            probs = logits.softmax(-1)
            own = logits[:, s]
            rank_sot += ((logits > own[:, None]).sum(dim=1) + 1).tolist()
            p_sot += probs[:, s].tolist()
            p_lang += probs[:, lang_id].tolist()
        return {"n": len(p_sot), "sot_token": s,
                "p_sot_at_pos0": float(np.mean(p_sot)),
                "median_rank_sot_at_pos0": float(np.median(rank_sot)),
                "rank1_rate": float(np.mean([r_ == 1 for r_ in rank_sot])),
                "p_language_token_at_pos0": float(np.mean(p_lang))}

    @torch.no_grad()
    def canonical_nll(model, ds, collator, n=128):
        """Teacher-forced NLL under the CORRECT label layout, for every arm.

        This is the inference-time question stated directly: how well does the
        model predict the real targets at the positions it will actually be asked
        about? It is far less noisy than WER, so it can see an effect a 300-step
        run leaves too small for the decoder to show.
        """
        model.eval()
        tot, ntok = 0.0, 0
        for i in range(0, min(n, ds.num_rows), EVAL_BS):
            feats = [ds[j] for j in range(i, min(i + EVAL_BS, ds.num_rows))]
            batch = collator(feats)
            out = model(input_features=batch["input_features"].to(model.device),
                        labels=batch["labels"].to(model.device), use_cache=False)
            k = int((batch["labels"] != -100).sum())
            tot += float(out.loss) * k; ntok += k
        return tot / ntok if ntok else float("nan")

    def fresh_model(seed):
        """Load base + inject LoRA, seeded so both arms start bit-identical.

        set_seed BEFORE get_peft_model: LoRA A is randomly initialized and the
        Trainer's own seeding happens too late to cover it.
        """
        set_seed(seed)
        m = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16)
        m.generation_config.language, m.generation_config.task = LANGUAGE, TASK
        m.generation_config.forced_decoder_ids = None
        m.model.encoder.requires_grad_(False)
        m.gradient_checkpointing_enable(); m.config.use_cache = False
        set_seed(seed)
        m = get_peft_model(m, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                         lora_dropout=LORA_DROPOUT,
                                         init_lora_weights=True,
                                         target_modules=["q_proj", "v_proj"]))
        return m.to("cuda")

    def adapter_hash(model):
        h = hashlib.sha256()
        for n_, p in sorted((n_, p) for n_, p in model.named_parameters() if p.requires_grad):
            h.update(n_.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    results = {"config": {
        "model": MODEL_ID, "revision": MODEL_REVISION, "seeds": SEEDS,
        "data_seed": DATA_SEED, "max_steps": MAX_STEPS,
        "train_meetings": len(train_meetings), "val_meetings": len(val_meetings),
        "lr": LR, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT, "bs": TRAIN_BS, "grad_acc": GRAD_ACC,
        "effective_epochs": eff_epochs,
        "export_sha": export_sha, "denylist_sha": denylist_sha,
        "n_train": ds_train.num_rows, "n_valc": ds_valc.num_rows,
        "n_valr": ds_valr.num_rows if ds_valr else 0,
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "peft": peft.__version__, "datasets": hfds.__version__},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "baseline": {}, "runs": [], "comparisons": {}}
    detail = {"refs_valc": refs_c, "refs_valr": refs_r,
              "meetings_valc": mtg_c, "meetings_valr": mtg_r, "runs": {}}
    counts = {}

    def flush():
        """Write after every stage — a crash in arm 5 must not lose arms 1-4."""
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        (WORK / "results_ab_detail.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=2))

    # ---------------- baseline (untrained) ------------------------------------
    # LoRA B is zero-initialized, so a freshly wrapped model is numerically the
    # base model: one decode serves as the baseline for every arm and seed.
    log("=== BASELINE (untrained base = zero-init adapter) ===")
    model = fresh_model(SEEDS[0])
    init_hashes = {SEEDS[0]: adapter_hash(model)}
    fixed_collator = tr.Collator(processor, model.get_base_model().config.decoder_start_token_id)
    base_c, cap_c = transcribe(model, ds_valc, "base/valc")
    base_r, cap_r = transcribe(model, ds_valr, "base/valr") if ds_valr else ([], 0)
    cnt_base_c = per_utt_counts(refs_c, base_c)
    cnt_base_r = per_utt_counts(refs_r, base_r) if refs_r else []
    counts["baseline"] = {"valc": cnt_base_c, "valr": cnt_base_r}
    results["baseline"] = {
        "init_adapter_sha": init_hashes[SEEDS[0]],
        "valc": {"wer": agg_wer(cnt_base_c), "cer": agg_cer(cnt_base_c), "n": len(base_c)},
        "valr": ({"wer": agg_wer(cnt_base_r), "cer": agg_cer(cnt_base_r), "n": len(base_r)}
                 if refs_r else None),
        "sot_diagnostic": sot_diagnostic(model, ds_valc),
        "canonical_nll_valc": canonical_nll(model, ds_valc, fixed_collator)}
    detail["runs"]["baseline"] = {"hyps_valc": base_c, "hyps_valr": base_r}
    log(f"BASELINE valc WER={agg_wer(cnt_base_c):.4f} CER={agg_cer(cnt_base_c):.4f}"
        + (f" | valr WER={agg_wer(cnt_base_r):.4f}" if refs_r else ""))
    log(f"BASELINE {results['baseline']['sot_diagnostic']} "
        f"nll={results['baseline']['canonical_nll_valc']:.4f}")
    flush()
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---------------- the arms, over seeds ------------------------------------
    for si, seed in enumerate(SEEDS):
        # Alternate which arm runs first, so any drift over the pod's lifetime
        # (thermal, neighbour load) cannot line up with one arm.
        arms = ("fixed", "legacy") if si % 2 == 0 else ("legacy", "fixed")
        for arm in arms:
            key = f"{arm}-s{seed}"
            log(f"=== ARM {arm} seed {seed} ===")
            model = fresh_model(seed)
            h = adapter_hash(model)
            if seed in init_hashes and h != init_hashes[seed]:
                sys.exit(f"[ab FATAL] {key} started from a different adapter "
                         f"({h} vs {init_hashes[seed]}) — two variables, not one")
            init_hashes[seed] = h
            sot_id = model.get_base_model().config.decoder_start_token_id
            inner = (tr.Collator(processor, sot_id) if arm == "fixed"
                     else LegacyCollator(processor))
            collator = TracingCollator(inner)
            # Prove the arm is what it claims before spending GPU hours on it.
            probe = inner([ds_train[0], ds_train[1]])
            first = probe["labels"][:, 0].tolist()
            if arm == "legacy" and not all(t == sot_id for t in first):
                sys.exit(f"[ab FATAL] legacy arm did not reproduce the bug: {first}")
            if arm == "fixed" and any(t == sot_id for t in first):
                sys.exit(f"[ab FATAL] fixed arm still carries <|sot|>: {first}")
            log(f"  collator check ok — first label ids {first} (sot={sot_id})")

            args = Seq2SeqTrainingArguments(
                output_dir=str(WORK / f"arm-{key}"),
                per_device_train_batch_size=TRAIN_BS,
                gradient_accumulation_steps=GRAD_ACC, learning_rate=LR,
                warmup_steps=int(0.1 * MAX_STEPS), max_steps=MAX_STEPS, fp16=True,
                lr_scheduler_type="linear", optim="adamw_torch",
                predict_with_generate=False, eval_strategy="no",
                save_strategy="no", logging_steps=25, report_to=[],
                remove_unused_columns=False, label_names=["labels"],
                seed=seed, data_seed=DATA_SEED, dataloader_num_workers=0,
                per_device_eval_batch_size=EVAL_BS)
            trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds_train,
                                     data_collator=collator, processing_class=processor)
            t0 = time.time()
            hist = trainer.train()
            train_elapsed = time.time() - t0        # before eval, or it measures eval
            nonfinite = [h_ for h_ in trainer.state.log_history
                         if not np.isfinite(h_.get("loss", 0.0))]
            log(f"  trained {MAX_STEPS} steps in {train_elapsed:.0f}s "
                f"loss={hist.training_loss:.4f} nonfinite={len(nonfinite)}")

            model.save_pretrained(str(WORK / f"adapter-{key}"))
            hyp_c, capc = transcribe(model, ds_valc, f"{key}/valc")
            hyp_r, capr = transcribe(model, ds_valr, f"{key}/valr") if ds_valr else ([], 0)
            cc = per_utt_counts(refs_c, hyp_c)
            crr = per_utt_counts(refs_r, hyp_r) if refs_r else []
            counts[key] = {"valc": cc, "valr": crr}
            results["runs"].append({
                "arm": arm, "seed": seed,
                # NOT comparable across arms: legacy has one extra supervised token
                # per utterance, so its mean loss is over a different denominator.
                "train_loss_not_cross_arm_comparable": float(hist.training_loss),
                "train_seconds": train_elapsed, "nonfinite_losses": len(nonfinite),
                "batch_order_sha": collator.trace_hash(),
                "init_adapter_sha": h,
                "valc": {"wer": agg_wer(cc), "cer": agg_cer(cc), "n": len(hyp_c)},
                "valr": ({"wer": agg_wer(crr), "cer": agg_cer(crr), "n": len(hyp_r)}
                         if refs_r else None),
                "sot_diagnostic": sot_diagnostic(model, ds_valc),
                "canonical_nll_valc": canonical_nll(model, ds_valc, fixed_collator),
                "length_cap_batches": capc})
            detail["runs"][key] = {"hyps_valc": hyp_c, "hyps_valr": hyp_r}
            log(f"  {key}: valc WER={agg_wer(cc):.4f} CER={agg_cer(cc):.4f}"
                + (f" | valr WER={agg_wer(crr):.4f}" if refs_r else "")
                + f" | nll={results['runs'][-1]['canonical_nll_valc']:.4f}")
            log(f"  {key} sot: {results['runs'][-1]['sot_diagnostic']}")
            flush()
            del model, trainer; gc.collect(); torch.cuda.empty_cache()

        # Per-seed comparison, written as soon as the pair completes.
        f_key, l_key = f"fixed-s{seed}", f"legacy-s{seed}"
        if f_key in counts and l_key in counts:
            results["comparisons"][f"seed{seed}_valc_legacy_minus_fixed"] = cluster_bootstrap(
                counts[l_key]["valc"], counts[f_key]["valc"], mtg_c, N_BOOT)
            if refs_r:
                results["comparisons"][f"seed{seed}_valr_legacy_minus_fixed"] = \
                    cluster_bootstrap(counts[l_key]["valr"], counts[f_key]["valr"],
                                      mtg_r, N_BOOT)
            results["comparisons"][f"seed{seed}_valc_fixed_minus_baseline"] = \
                cluster_bootstrap(counts[f_key]["valc"], cnt_base_c, mtg_c, N_BOOT)
            c = results["comparisons"][f"seed{seed}_valc_legacy_minus_fixed"]
            log(f"SEED {seed} valc legacy-fixed: {c['delta_wer']:+.4f} "
                f"cluster CI [{c['ci95_cluster'][0]:+.4f}, {c['ci95_cluster'][1]:+.4f}] "
                f"({c['meetings_a_worse']}/{c['meetings_total']} meetings worse)")
            flush()

    # ---------------- across-seed summary -------------------------------------
    deltas = [results["comparisons"][k]["delta_wer"] for k in results["comparisons"]
              if k.endswith("valc_legacy_minus_fixed")]
    if deltas:
        results["summary"] = {
            "valc_legacy_minus_fixed_per_seed": deltas,
            "mean_delta_wer": float(np.mean(deltas)),
            "min_delta_wer": float(np.min(deltas)), "max_delta_wer": float(np.max(deltas)),
            "all_same_sign": bool(all(d > 0 for d in deltas) or all(d < 0 for d in deltas)),
            "baseline_valc_wer": results["baseline"]["valc"]["wer"],
            "nll_by_run": {f"{r_['arm']}-s{r_['seed']}": r_["canonical_nll_valc"]
                           for r_ in results["runs"]}}
        log("=" * 70)
        log(f"COST OF THE BUG — val_corr WER, legacy minus fixed, per seed: "
            f"{[f'{d:+.4f}' for d in deltas]}")
        log(f"  mean {np.mean(deltas):+.4f} | consistent sign: "
            f"{results['summary']['all_same_sign']}")
        log(f"  baseline {results['baseline']['valc']['wer']:.4f}")
        log(f"  canonical NLL: {results['summary']['nll_by_run']}")
    flush()
    log(f"results -> {RESULTS}")


if __name__ == "__main__":
    main()
