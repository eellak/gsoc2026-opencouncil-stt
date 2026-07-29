"""Diagnose WHY the training-time eval and the controlled A/B disagree.

Training eval (2026-06-24) said: base 33.4 WER -> ours 26.7 (big win).
Controlled A/B (2026-07-24) said: base 0.158 -> ours 0.176 (regression).

Those two harnesses differ in FOUR ways at once. This script holds the sample fixed
and varies them one at a time, so the gap can be attributed:

  1. decoder stack   : faster-whisper/CT2 (controlled)  vs  HF transformers generate (training)
  2. search width    : beam=2 (controlled)              vs  greedy/beam=1 (training)
  3. scoring         : normalized WER (controlled)      vs  raw WER (training headline)
  4. tokenizer bug   : clean_up_tokenization_spaces False vs True (the June run used True)

Sample: ALL eligible held-out (argos+orestiada) human-corrected utterances, using the
training script's own span filter (0.3-30s) and +/-0.2s pad. Reference = final_after_text
(human), never another ASR's output.

NOTE: the June run's exact 191-clip val_corr set cannot be reproduced -- export.jsonl has
grown since (3,854 rows now). This is a larger, fresher sample, not a literal replication.

Outputs (written incrementally, safe to interrupt):
  eval/diagnose/hyps.json     - every hypothesis from every config
  eval/diagnose/results.json  - the WER table
Run:  .venv-eval/bin/python eval/diagnose/decompose_training_gap.py
"""
import os, sys, json, re, wave, random, subprocess, tempfile, time, unicodedata
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
OUT = ROOT / "eval/diagnose"
OUT.mkdir(parents=True, exist_ok=True)
AUDIO = ROOT / "data/asr/audio"
WORK = Path(tempfile.mkdtemp(prefix="diag_", dir="/tmp"))

# float32 CT2 models built by ab_general_utterances.py (reused if present)
CT2_BASE = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/231379b0-42e3-4cc6-a8d0-71d11d25d331/scratchpad/base_f32_ct2"
CT2_OURS = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/231379b0-42e3-4cc6-a8d0-71d11d25d331/scratchpad/ours_f32_ct2"
HF_BASE = "openai/whisper-large-v3"
HF_OURS = "/home/harold/oc-asr-serve/merged"

VAL_CITIES = {"argos", "orestiada"}
PAD_S, SR = 0.2, 16000          # matches train_runpod.py
MIN_DUR, MAX_DUR = 0.3, 30.0    # matches train_runpod.py ok_span
N_STAGE1 = int(os.environ.get("N_STAGE1", 150))   # faster-whisper passes
N_STAGE2 = int(os.environ.get("N_STAGE2", 50))    # HF generate passes (slow)
SEED = 13
random.seed(SEED)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---------------- scoring ----------------
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

def _ws(s):                      # training's "raw" WER preprocessing
    return " ".join((s or "").split())

def gnorm(s):                    # training's gnorm(), verbatim
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFC", s).replace("ς", "σ")
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s)).strip()

def edist(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]

def pooled_wer(pairs, prep):
    """pairs = [(ref, hyp)]; prep = _ws (raw) or gnorm (normalized)."""
    num = den = 0
    for ref, hyp in pairs:
        r, h = prep(ref).split(), prep(hyp).split()
        num += edist(r, h); den += len(r)
    return 100.0 * num / den if den else float("nan")

# ---------------- sample ----------------
sel = []
for line in open(ROOT / "data/asr/export.jsonl"):
    d = json.loads(line)
    if d.get("city_id") not in VAL_CITIES: continue
    bef, aft = d.get("initial_before_text", ""), d.get("final_after_text", "")
    if not aft or not aft.strip(): continue
    if bef.strip() == aft.strip(): continue            # actually-corrected only
    ap = AUDIO / f"{d['city_id']}__{d['meeting_id']}.mp3"
    if not ap.exists(): continue
    try: dur = float(d["end"]) - float(d["start"])
    except (TypeError, ValueError, KeyError): continue
    if not (MIN_DUR <= dur <= MAX_DUR): continue
    d["_audio"] = str(ap); sel.append(d)
random.shuffle(sel)
sel = sel[:N_STAGE1]
sub = sel[:N_STAGE2]
log(f"eligible corrected held-out clips sampled: stage1={len(sel)} stage2={len(sub)}")

def cut(d):
    out = WORK / f"{d['utterance_id']}.wav"
    if out.exists(): return str(out)
    s = max(0.0, float(d["start"]) - PAD_S); e = float(d["end"]) + PAD_S
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", str(s),
                    "-to", str(e), "-i", d["_audio"], "-ar", str(SR), "-ac", "1", str(out)],
                   check=True)
    return str(out)

log("cutting clips ...")
for d in sel: d["_wav"] = cut(d)
log("clips ready")

HYPS_F, RES_F = OUT / "hyps.json", OUT / "results.json"
hyps = json.load(open(HYPS_F)) if HYPS_F.exists() else {}

def save_hyps():
    json.dump(hyps, open(HYPS_F, "w"), ensure_ascii=False, indent=1)

# ---------------- stage 1: faster-whisper, beam=2 vs greedy ----------------
from faster_whisper import WhisperModel

for tag, path in (("base", CT2_BASE), ("ours", CT2_OURS)):
    if not Path(path, "model.bin").exists():
        log(f"!! missing CT2 model {path} -- skipping {tag} stage1"); continue
    m = None
    for beam in (2, 1):
        key = f"fw_{tag}_beam{beam}"
        if key in hyps and len(hyps[key]) >= len(sel):
            log(f"skip {key} (cached)"); continue
        if m is None:
            log(f"loading CT2 {tag} (float32)")
            m = WhisperModel(path, device="cpu", compute_type="float32", cpu_threads=8)
        log(f"=== {key} ===")
        hyps.setdefault(key, {})
        for i, d in enumerate(sel):
            if d["utterance_id"] in hyps[key]: continue
            segs, _ = m.transcribe(d["_wav"], language="el", beam_size=beam,
                                   condition_on_previous_text=False)
            hyps[key][d["utterance_id"]] = " ".join(s.text.strip() for s in segs).strip()
            if (i + 1) % 25 == 0: log(f"  {key}: {i+1}/{len(sel)}"); save_hyps()
        save_hyps()
    del m

# ---------------- stage 2: HF transformers generate (the training path) ----------------
def read_wav(p):
    import numpy as np
    with wave.open(p, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (a.astype("float32") / 32768.0)

try:
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    torch.set_num_threads(8)
    for tag, path in (("base", HF_BASE), ("ours", HF_OURS)):
        keys = [f"hf_{tag}_greedy_cleanup{v}" for v in ("True", "False")]
        if all(k in hyps and len(hyps[k]) >= len(sub) for k in keys):
            log(f"skip hf_{tag} (cached)"); continue
        log(f"loading HF {tag} float32 (slow) ...")
        proc = WhisperProcessor.from_pretrained(path)
        model = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float32).eval()
        for k in keys: hyps.setdefault(k, {})
        log(f"=== hf_{tag}_greedy ({len(sub)} clips) ===")
        for i, d in enumerate(sub):
            uid = d["utterance_id"]
            if all(uid in hyps[k] for k in keys): continue
            feats = proc(read_wav(d["_wav"]), sampling_rate=SR,
                         return_tensors="pt").input_features.to(torch.float32)
            with torch.no_grad():
                ids = model.generate(feats, language="el", task="transcribe",
                                     num_beams=1, do_sample=False, max_new_tokens=225)
            for v, k in zip((True, False), keys):
                hyps[k][uid] = proc.tokenizer.batch_decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=v)[0].strip()
            if (i + 1) % 5 == 0: log(f"  hf_{tag}: {i+1}/{len(sub)}"); save_hyps()
        save_hyps()
        del model, proc
except Exception as e:
    log(f"!! stage 2 (HF path) failed: {type(e).__name__}: {e}")
    log("   stage 1 results are still valid; see hyps.json")

# ---------------- report ----------------
refs = {d["utterance_id"]: d["final_after_text"] for d in sel}
scribe = {d["utterance_id"]: d.get("initial_before_text", "") for d in sel}
hyps["scribe_before"] = scribe
save_hyps()

rows = []
for key, hs in sorted(hyps.items()):
    ids = [u for u in hs if u in refs and hs[u] is not None]
    if not ids: continue
    pairs = [(refs[u], hs[u]) for u in ids]
    rows.append({"config": key, "n": len(ids),
                 "wer_raw": pooled_wer(pairs, _ws),
                 "wer_norm": pooled_wer(pairs, gnorm)})

print("\n==== decomposition: corrected held-out utterances, ref = human final_after_text ====")
print(f"{'config':<32} {'n':>4} {'WER raw':>9} {'WER norm':>9}")
for r in rows:
    print(f"{r['config']:<32} {r['n']:>4} {r['wer_raw']:>9.2f} {r['wer_norm']:>9.2f}")

# head-to-head ours vs base per config family, on the shared subset
h2h = {}
for fam in ("fw_beam2", "fw_beam1", "hf_greedy"):
    kb = {"fw_beam2": "fw_base_beam2", "fw_beam1": "fw_base_beam1",
          "hf_greedy": "hf_base_greedy_cleanupFalse"}[fam]
    ko = kb.replace("base", "ours")
    if kb not in hyps or ko not in hyps: continue
    ids = [u for u in hyps[kb] if u in hyps[ko] and u in refs]
    w = l = t = 0
    for u in ids:
        r = gnorm(refs[u]).split()
        if not r: continue
        wb = edist(r, gnorm(hyps[kb][u]).split()) / len(r)
        wo = edist(r, gnorm(hyps[ko][u]).split()) / len(r)
        if wo < wb - 1e-9: w += 1
        elif wo > wb + 1e-9: l += 1
        else: t += 1
    h2h[fam] = {"n": len(ids), "ours_better": w, "worse": l, "tie": t}
    print(f"\n{fam}: ours vs base -> better {w} | worse {l} | tie {t}  (n={len(ids)}, normalized)")

json.dump({"sample": {"stage1": len(sel), "stage2": len(sub), "seed": SEED},
           "table": rows, "head2head": h2h},
          open(RES_F, "w"), ensure_ascii=False, indent=2)
log(f"wrote {RES_F}")
for d in sel: Path(d["_wav"]).unlink(missing_ok=True)
try: WORK.rmdir()
except OSError: pass
log("done")
