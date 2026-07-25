# PORTABILITY NOTE (read before running):
# This ran on the mini-PC in a session scratchpad. To reuse on another machine, fix:
#   - SC              : output/work dir (was a session scratchpad; set to a writable dir)
#   - OURS_INT8       : the shipped CTranslate2 int8 model dir
#   - build("/home/harold/oc-asr-serve/merged", ...) : the merged base+adapter model.
#     For portability, build "ours" from the PUBLIC adapter instead: merge
#     openai/whisper-large-v3 + opencouncil/whisper-large-v3-el-council-lora, then
#     ct2-transformers-converter (see serve/oc-asr-serverless Dockerfile for the recipe).
# Needs the .venv-eval env (faster_whisper, ctranslate2, ffmpeg). CPU-only => float32,
# not float16 (float16 is GPU-only in CTranslate2).
"""Controlled same-stack A/B on GENERAL held-out utterances (argos+orestiada).
Does int8 quantization erode the fine-tune's gain? Compare, on ONE faster-whisper
stack: base float32, ours float32 (full precision), ours int8 (shipped).
Result (n=34): base 0.163 WER / ours 0.164 / ours-int8 0.164 -> tie; int8 == float32."""
import os, csv, json, re, subprocess, tempfile, random, unicodedata, sys, time
from pathlib import Path

ROOT = "/home/harold/opencouncil-fine-tuning"
SCRATCH = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/231379b0-42e3-4cc6-a8d0-71d11d25d331/scratchpad"
VENVBIN = f"{ROOT}/.venv-eval/bin"
AUDIO = Path(ROOT, "data/asr/audio")
OURS_INT8 = "/home/harold/oc-asr-serve/ct2"
OURS_F32 = f"{SCRATCH}/ours_f32_ct2"
BASE_F32 = f"{SCRATCH}/base_f32_ct2"
random.seed(13)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def build(src, out, quant):
    if Path(out, "model.bin").exists():
        log(f"exists: {out}"); return
    log(f"building {out} ({quant}) from {src} ...")
    subprocess.run([f"{VENVBIN}/ct2-transformers-converter", "--model", src, "--output_dir", out,
                    "--copy_files", "tokenizer.json", "preprocessor_config.json",
                    "--quantization", quant, "--force"], check=True)
    log(f"built {out}")

def norm(s):
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()

# ---- entity vocab from glossary ----
gl = json.load(open(f"{ROOT}/data/glossary/glossary.json"))
ents = set(gl.get("global", []))
for c in ("argos", "orestiada"):
    v = gl.get("per_city", {}).get(c)
    if isinstance(v, list): ents |= set(v)
ent_norm = sorted({norm(e) for e in ents if len(e.strip()) >= 4}, key=len, reverse=True)
ent_pat = {e: re.compile(r"(?<!\w)" + re.escape(e) + r"(?!\w)") for e in ent_norm}

def gold_entities(text):
    tn = norm(text)
    g = [e for e in ent_norm if ent_pat[e].search(tn)]
    return [e for e in g if not any(e != o and e in o and ent_pat[o].search(tn) for o in g)]

# ---- sample held-out utterances ----
rows = list(csv.DictReader(open(f"{ROOT}/data/asr/val_manifest.csv")))
cand = []
for r in rows:
    ap = AUDIO / f"{r['city_id']}__{r['meeting_id']}.mp3"
    if not ap.exists(): continue
    try: dur = float(r["utterance_end"]) - float(r["utterance_start"])
    except: continue
    nwords = len(r["text"].split())
    if 8.0 <= dur <= 25.0 and nwords >= 15:
        r["_dur"] = dur; r["_gold"] = gold_entities(r["text"]); r["_audio"] = str(ap)
        cand.append(r)
log(f"candidate utterances: {len(cand)}")
with_ent = [r for r in cand if r["_gold"]]
random.shuffle(with_ent); random.shuffle(cand)
# 34 entity-containing + 8 any, both cities represented
sample = with_ent[:34] + [r for r in cand if r not in with_ent][:8]
random.shuffle(sample)
log(f"sample size: {len(sample)} (entity-containing: {sum(1 for r in sample if r['_gold'])})")

def cut(audio, start, end):
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=SCRATCH); os.close(fd)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", str(start),
                    "-to", str(end), "-i", audio, "-ar", "16000", "-ac", "1", tmp], check=True)
    return tmp

# ---- build models ----
build("/home/harold/oc-asr-serve/merged", OURS_F32, "float32")
build("openai/whisper-large-v3", BASE_F32, "float32")

from faster_whisper import WhisperModel

def wtoks(s):
    s = norm(s)
    return re.findall(r"\w+", s)
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

MODELS = [("base_f32", BASE_F32, "float32"),
          ("ours_f32", OURS_F32, "float32"),
          ("ours_int8", OURS_INT8, "int8")]

# pre-cut clips once
log("cutting clips ...")
clips = []
for r in sample:
    try: clips.append((r, cut(r["_audio"], r["utterance_start"], r["utterance_end"])))
    except Exception as e: log(f"cut fail {r['utterance_id']}: {e}")
log(f"clips ready: {len(clips)}")

results = {}
hyps = {}
for name, path, comp in MODELS:
    log(f"=== model {name} ({comp}) ===")
    m = WhisperModel(path, device="cpu", compute_type=comp, cpu_threads=8)
    hyps[name] = {}
    for i, (r, wav) in enumerate(clips):
        segs, _ = m.transcribe(wav, language="el", beam_size=2, condition_on_previous_text=False)
        hyps[name][r["utterance_id"]] = " ".join(s.text.strip() for s in segs).strip()
        if (i + 1) % 10 == 0: log(f"  {name}: {i+1}/{len(clips)}")
    del m
log("transcription done")

# persist hyps immediately so scoring can't lose the expensive transcription work
json.dump({"clips": [{"id": r["utterance_id"], "city": r["city_id"], "ref": r["text"],
                       "gold": r["_gold"]} for r, _ in clips], "hyps": hyps},
          open(f"{SCRATCH}/step3_hyps.json", "w"), ensure_ascii=False)
log("saved step3_hyps.json")

# ---- score (own pooled WER, no jiwer) ----
def score(name):
    num = den = 0
    tg = tm = 0
    for r, _ in clips:
        ref = wtoks(r["text"]); hyp = wtoks(hyps[name][r["utterance_id"]])
        num += edist(ref, hyp); den += len(ref)
        g = r["_gold"]
        if g:
            hn = norm(hyps[name][r["utterance_id"]])
            tg += len(g); tm += sum(1 for e in g if re.search(r"(?<!\w)" + re.escape(e) + r"(?!\w)", hn))
    return num / den, (tm / tg if tg else 0), tm, tg

print("\n==== STEP 3 RESULTS (held-out, same faster-whisper stack, CPU) ====")
print(f"{'model':>10} | {'WER':>7} | {'entity-recall':>13}")
for name, _, _ in MODELS:
    w, er, tm, tg = score(name)
    results[name] = {"wer": w, "entity_recall": er, "ent_matched": tm, "ent_gold": tg}
    print(f"{name:>10} | {w:7.4f} | {er*100:6.1f}%  ({tm}/{tg})")
json.dump({"n_clips": len(clips), "results": results,
           "hyps_sample": {k: dict(list(v.items())[:3]) for k, v in hyps.items()}},
          open(f"{SCRATCH}/step3_results.json", "w"), ensure_ascii=False, indent=2)
# cleanup wavs
for _, wav in clips: Path(wav).unlink(missing_ok=True)
log("wrote step3_results.json")
