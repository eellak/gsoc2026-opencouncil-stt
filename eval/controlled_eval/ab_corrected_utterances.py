# PORTABILITY NOTE (read before running):
# Ran on the mini-PC in a session scratchpad. To reuse, fix SC (work dir) and the two
# ct2 model dirs (base_f32_ct2, ours_f32_ct2) - build them with ab_general_utterances.py
# first, or from the public adapter (see serve/oc-asr-serverless Dockerfile). Uses
# data/asr/export.jsonl (initial_before_text / final_after_text) + data/asr/audio/*.mp3.
# Needs .venv-eval (faster_whisper, ctranslate2, ffmpeg).
"""CAPSTONE (the decisive test): on the ACTUALLY-CORRECTED held-out utterances (the
fine-tune's target), does ours beat base whisper on the SAME stack, scored vs the
human reference? Result (n=50): base 0.158 WER, ours 0.176 (WORSE); head-to-head
ours better 6 / worse 20 / tie 24. The fine-tune regresses on its own target."""
import os, json, re, subprocess, tempfile, random, unicodedata, time
from pathlib import Path

ROOT = "/home/harold/opencouncil-fine-tuning"
SC = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/231379b0-42e3-4cc6-a8d0-71d11d25d331/scratchpad"
AUDIO = Path(ROOT, "data/asr/audio")
MODELS = [("base_f32", f"{SC}/base_f32_ct2", "float32"),
          ("ours_f32", f"{SC}/ours_f32_ct2", "float32")]
random.seed(13)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def norm(s):
    s = unicodedata.normalize("NFD", s or ""); return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
def wtoks(s): return re.findall(r"\w+", norm(s))
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

# corrected held-out utterances
sel = []
for line in open(f"{ROOT}/data/asr/export.jsonl"):
    d = json.loads(line)
    if d.get("city_id") not in ("argos", "orestiada"): continue
    bef, aft = d.get("initial_before_text", ""), d.get("final_after_text", "")
    if not aft or bef.strip() == aft.strip(): continue     # only actually-corrected
    ap = AUDIO / f"{d['city_id']}__{d['meeting_id']}.mp3"
    if not ap.exists(): continue
    try: dur = float(d["end"]) - float(d["start"])
    except: continue
    if not (4.0 <= dur <= 30.0) or len(aft.split()) < 6: continue
    d["_audio"] = str(ap); sel.append(d)
random.shuffle(sel); sel = sel[:50]
log(f"corrected held-out utterances sampled: {len(sel)}")

def cut(a, s, e):
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=SC); os.close(fd)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", str(s), "-to", str(e),
                    "-i", a, "-ar", "16000", "-ac", "1", tmp], check=True)
    return tmp
log("cutting clips ...")
clips = [(d, cut(d["_audio"], d["start"], d["end"])) for d in sel]

from faster_whisper import WhisperModel
hyps = {}
for name, path, comp in MODELS:
    log(f"=== {name} ===")
    m = WhisperModel(path, device="cpu", compute_type=comp, cpu_threads=8)
    hyps[name] = {}
    for i, (d, wav) in enumerate(clips):
        segs, _ = m.transcribe(wav, language="el", beam_size=2, condition_on_previous_text=False)
        hyps[name][d["utterance_id"]] = " ".join(s.text.strip() for s in segs).strip()
        if (i + 1) % 10 == 0: log(f"  {name}: {i+1}/{len(clips)}")
    del m
# also score the ORIGINAL scribe text (initial_before_text) as a reference point
hyps["scribe_before"] = {d["utterance_id"]: d.get("initial_before_text", "") for d, _ in clips}

json.dump({"n": len(clips), "hyps": hyps,
           "refs": {d["utterance_id"]: d["final_after_text"] for d, _ in clips}},
          open(f"{SC}/step3c_hyps.json", "w"), ensure_ascii=False)

def wer(name):
    num = den = 0; wins = losses = 0
    per = []
    for d, _ in clips:
        ref = wtoks(d["final_after_text"]); hyp = wtoks(hyps[name][d["utterance_id"]])
        e = edist(ref, hyp); num += e; den += len(ref)
    return num / den
print("\n==== CAPSTONE: corrected held-out utterances, vs human reference ====")
print(f"n={len(clips)} corrected utterances (argos+orestiada, same faster-whisper stack)\n")
res = {}
for name in ("scribe_before", "base_f32", "ours_f32"):
    res[name] = wer(name); print(f"  {name:>14}: WER={res[name]:.4f}")
# head-to-head ours vs base per-utterance
wins = losses = ties = 0
for d, _ in clips:
    ref = wtoks(d["final_after_text"])
    wb = edist(ref, wtoks(hyps["base_f32"][d["utterance_id"]])) / max(len(ref), 1)
    wo = edist(ref, wtoks(hyps["ours_f32"][d["utterance_id"]])) / max(len(ref), 1)
    if wo < wb - 1e-9: wins += 1
    elif wo > wb + 1e-9: losses += 1
    else: ties += 1
print(f"\nours vs base per-utterance: ours better {wins} | worse {losses} | tie {ties}")
json.dump({"n": len(clips), "wer": res, "head2head": {"ours_better": wins, "worse": losses, "tie": ties}},
          open(f"{SC}/step3c_results.json", "w"), ensure_ascii=False, indent=2)
for _, wav in clips: Path(wav).unlink(missing_ok=True)
log("done")
