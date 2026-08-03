# PORTABILITY NOTE (read before running):
# Same-stack extension of ab_corrected_utterances.py. Reuses that script's exact sample
# (same filters + random.seed(13) over data/asr/export.jsonl) so the base/ours hypotheses
# already computed there can be reused instead of re-transcribed. Set:
#   SC        : writable work dir
#   PREV_HYPS : step3c_hyps.json written by ab_corrected_utterances.py (optional; if
#               missing, base/ours are re-transcribed here)
#   MODELS    : the ct2 model dirs (base_f32_ct2 / ours_f32_ct2)
# Needs .venv-eval (faster_whisper>=1.0.2 for `hotwords`, ctranslate2, ffmpeg).
"""Does inference-time contextual biasing (per-meeting roster as faster-whisper
`hotwords`) beat the plain models on the ACTUALLY-CORRECTED held-out utterances?

Same stack, same normalization, same 50 clips as the capstone A/B. Configs:
  base            - base whisper, no biasing            (reused)
  ours            - the LoRA fine-tune, no biasing      (reused)
  base+hotwords   - base whisper, roster hotwords       (new)
  ours+hotwords   - fine-tune, roster hotwords          (new, optional via RUN_OURS_HW)
Hotwords come from data/improve_loop/rosters.json (per-meeting speaker roster fetched
from the OpenCouncil API) - never from the reference text, so this is not oracle.
"""
import os, json, re, subprocess, tempfile, random, unicodedata, time, sys
from pathlib import Path

ROOT = "/home/harold/opencouncil-fine-tuning"
SC = os.environ.get("SC", "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/512d0560-8032-4651-80e9-da43053cd4b0/scratchpad")
PREV = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/231379b0-42e3-4cc6-a8d0-71d11d25d331/scratchpad"
PREV_HYPS = f"{PREV}/step3c_hyps.json"
BASE_CT2 = f"{PREV}/base_f32_ct2"
OURS_CT2 = f"{PREV}/ours_f32_ct2"
AUDIO = Path(ROOT, "data/asr/audio")
# data/pii/rosters_full.json covers all 311 meetings (improve_loop/rosters.json only 73,
# which left 40/50 sample utterances without a roster).
ROSTERS = f"{ROOT}/data/pii/rosters_full.json"
RUN_OURS_HW = os.environ.get("RUN_OURS_HW", "1") == "1"
HOTWORD_TOKEN_BUDGET = 180          # whisper prompt window is 224 tokens; leave headroom

random.seed(13)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def norm(s):
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
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

# ---- identical sample to ab_corrected_utterances.py ----
sel = []
for line in open(f"{ROOT}/data/asr/export.jsonl"):
    d = json.loads(line)
    if d.get("city_id") not in ("argos", "orestiada"): continue
    bef, aft = d.get("initial_before_text", ""), d.get("final_after_text", "")
    if not aft or bef.strip() == aft.strip(): continue
    ap = AUDIO / f"{d['city_id']}__{d['meeting_id']}.mp3"
    if not ap.exists(): continue
    try: dur = float(d["end"]) - float(d["start"])
    except: continue
    if not (4.0 <= dur <= 30.0) or len(aft.split()) < 6: continue
    d["_audio"] = str(ap); sel.append(d)
random.shuffle(sel); sel = sel[:50]
log(f"corrected held-out utterances sampled: {len(sel)}")

prev_hyps = {}
if Path(PREV_HYPS).exists():
    p = json.load(open(PREV_HYPS))
    if set(p["refs"]) == {d["utterance_id"] for d in sel}:
        prev_hyps = p["hyps"]
        log(f"reusing cached hypotheses: {sorted(prev_hyps)}")
    else:
        log("WARNING: cached sample differs from the one built here - re-transcribing")

# ---- per-meeting hotwords from the roster ----
rosters = json.load(open(ROSTERS))
from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{BASE_CT2}/tokenizer.json")

def hotwords_for(city, meeting):
    terms = rosters.get(f"{city}/{meeting}") or []
    if not terms: return ""
    # full names (multi-word) first - they carry the surname spelling we care about -
    # then single tokens, deterministic order, truncated to the prompt budget.
    multi = sorted(t for t in terms if " " in t)
    single = sorted(t for t in terms if " " not in t)
    picked, used = [], 0
    for t in multi + single:
        n = len(tok.encode(" " + t).ids)
        if used + n > HOTWORD_TOKEN_BUDGET: continue
        picked.append(t); used += n
    return ", ".join(picked)

hw = {d["utterance_id"]: hotwords_for(d["city_id"], d["meeting_id"]) for d in sel}
n_hw = sum(1 for v in hw.values() if v)
log(f"utterances with a roster: {n_hw}/{len(sel)}; "
    f"median terms {sorted(len(v.split(', ')) for v in hw.values() if v)[max(n_hw//2-1,0)] if n_hw else 0}")

def cut(a, s, e):
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=SC); os.close(fd)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", str(s), "-to", str(e),
                    "-i", a, "-ar", "16000", "-ac", "1", tmp], check=True)
    return tmp
log("cutting clips ...")
clips = [(d, cut(d["_audio"], d["start"], d["end"])) for d in sel]

from faster_whisper import WhisperModel
hyps = {}
if prev_hyps:
    hyps["base"] = prev_hyps["base_f32"]
    hyps["ours"] = prev_hyps["ours_f32"]
    hyps["scribe_before"] = prev_hyps["scribe_before"]

RUNS = []
if not prev_hyps:
    RUNS += [("base", BASE_CT2, False), ("ours", OURS_CT2, False)]
RUNS += [("base+hotwords", BASE_CT2, True)]
if RUN_OURS_HW: RUNS += [("ours+hotwords", OURS_CT2, True)]

for name, path, use_hw in RUNS:
    log(f"=== {name} ===")
    m = WhisperModel(path, device="cpu", compute_type="float32", cpu_threads=8)
    hyps[name] = {}
    for i, (d, wav) in enumerate(clips):
        kw = {"hotwords": hw[d["utterance_id"]]} if use_hw and hw[d["utterance_id"]] else {}
        segs, _ = m.transcribe(wav, language="el", beam_size=2,
                               condition_on_previous_text=False, **kw)
        hyps[name][d["utterance_id"]] = " ".join(s.text.strip() for s in segs).strip()
        if (i + 1) % 10 == 0: log(f"  {name}: {i+1}/{len(clips)}")
    del m
if "scribe_before" not in hyps:
    hyps["scribe_before"] = {d["utterance_id"]: d.get("initial_before_text", "") for d, _ in clips}

refs = {d["utterance_id"]: d["final_after_text"] for d, _ in clips}
json.dump({"n": len(clips), "hyps": hyps, "refs": refs,
           "hotwords": hw}, open(f"{SC}/hotwords_hyps.json", "w"), ensure_ascii=False)

def wer(name):
    num = den = 0
    for d, _ in clips:
        ref = wtoks(refs[d["utterance_id"]]); hyp = wtoks(hyps[name][d["utterance_id"]])
        num += edist(ref, hyp); den += len(ref)
    return num / den

# Entity recall restricted to roster terms that actually occur in the reference.
# LIMITATION (seen on the 2026-07-25 run): this sample is name-poor - only 1 gold term
# across all 50 utterances (5 of 98 in the whole corrected held-out pool), so the recall
# column carries no signal. For the name hypothesis, sample from val_manifest.csv instead.
def roster_recall(name):
    tg = tm = 0
    for d, _ in clips:
        terms = rosters.get(f"{d['city_id']}/{d['meeting_id']}") or []
        rn = norm(refs[d["utterance_id"]]); hn = norm(hyps[name][d["utterance_id"]])
        gold = {norm(t) for t in terms if len(t) >= 4
                and re.search(r"(?<!\w)" + re.escape(norm(t)) + r"(?!\w)", rn)}
        tg += len(gold)
        tm += sum(1 for g in gold if re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", hn))
    return (tm / tg if tg else 0.0), tm, tg

print("\n==== HOTWORDS A/B: corrected held-out utterances, vs human reference ====")
print(f"n={len(clips)} (argos+orestiada, same faster-whisper stack, roster hotwords)\n")
order = [k for k in ("scribe_before", "base", "ours", "base+hotwords", "ours+hotwords") if k in hyps]
res = {}
print(f"{'config':>16} | {'WER':>7} | {'roster-recall':>13}")
for name in order:
    w = wer(name); r, tm, tg = roster_recall(name)
    res[name] = {"wer": w, "roster_recall": r, "matched": tm, "gold": tg}
    print(f"{name:>16} | {w:7.4f} | {r*100:6.1f}%  ({tm}/{tg})")

def h2h(a, b):
    win = loss = tie = 0
    for d, _ in clips:
        ref = wtoks(refs[d["utterance_id"]])
        wa = edist(ref, wtoks(hyps[a][d["utterance_id"]])) / max(len(ref), 1)
        wb = edist(ref, wtoks(hyps[b][d["utterance_id"]])) / max(len(ref), 1)
        if wa < wb - 1e-9: win += 1
        elif wa > wb + 1e-9: loss += 1
        else: tie += 1
    return {"a_better": win, "a_worse": loss, "tie": tie}

pairs = [("base+hotwords", "base")]
if "ours+hotwords" in hyps: pairs += [("ours+hotwords", "ours"), ("base+hotwords", "ours")]
h = {}
print()
for a, b in pairs:
    h[f"{a}_vs_{b}"] = h2h(a, b)
    print(f"{a} vs {b}: better {h[f'{a}_vs_{b}']['a_better']} | "
          f"worse {h[f'{a}_vs_{b}']['a_worse']} | tie {h[f'{a}_vs_{b}']['tie']}")

json.dump({"n": len(clips), "results": res, "head2head": h},
          open(f"{Path(__file__).parent}/results_hotwords.json", "w"), ensure_ascii=False, indent=2)
for _, wav in clips: Path(wav).unlink(missing_ok=True)
log("done")
