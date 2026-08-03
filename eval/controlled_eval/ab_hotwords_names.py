# PORTABILITY NOTE (read before running):
# Set SC (writable work dir) and the two ct2 model dirs (BASE_CT2 / OURS_CT2, built by
# ab_general_utterances.py). Needs .venv-eval (faster_whisper>=1.0.2, ctranslate2, ffmpeg).
"""The name test that ab_hotwords.py could not do.

The corrected held-out subset is name-poor (5 of 98 utterances contain a roster name),
so it cannot measure whether contextual biasing fixes names. This script samples from
the GENERAL held-out pool (data/asr/val_manifest.csv, argos+orestiada) restricted to
utterances whose reference actually contains a per-meeting roster name, and runs the
same-stack A/B over it:

  base | base+hotwords | ours | ours+hotwords      (hotwords = per-meeting roster)

Reported: pooled WER and name recall over the gold roster terms in each reference.

Roster terms include party names and common words ("Πρωτοβουλία"), which would inflate
recall without saying anything about names. They are filtered by document frequency over
the whole val set: a term appearing in more than DF_MAX of val utterances is dropped.
"""
import os, csv, json, re, subprocess, tempfile, random, unicodedata, time
from pathlib import Path
from collections import Counter

ROOT = "/home/harold/opencouncil-fine-tuning"
SC = os.environ.get("SC", "/tmp/scratch")
PREV = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/231379b0-42e3-4cc6-a8d0-71d11d25d331/scratchpad"
BASE_CT2 = os.environ.get("BASE_CT2", f"{PREV}/base_f32_ct2")
OURS_CT2 = os.environ.get("OURS_CT2", f"{PREV}/ours_f32_ct2")
AUDIO = Path(ROOT, "data/asr/audio")
ROSTERS = f"{ROOT}/data/pii/rosters_full.json"
N_SAMPLE = int(os.environ.get("N_SAMPLE", "70"))
HOTWORD_TOKEN_BUDGET = 180
DF_MAX = 0.01                       # roster term seen in >1% of val utterances = not a name

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
def has(term_norm, text_norm):
    return re.search(r"(?<!\w)" + re.escape(term_norm) + r"(?!\w)", text_norm) is not None

rosters = json.load(open(ROSTERS))
rows = list(csv.DictReader(open(f"{ROOT}/data/asr/val_manifest.csv")))
log(f"val_manifest rows: {len(rows)}")

# ---- document frequency of every roster term over the val set (drops common words) ----
all_terms = {norm(t) for k, ts in rosters.items() if k.split("/")[0] in ("argos", "orestiada")
             for t in ts if len(t) >= 5}
val_norm = [norm(r["text"]) for r in rows]
df = Counter()
for tn in all_terms:
    pat = re.compile(r"(?<!\w)" + re.escape(tn) + r"(?!\w)")
    df[tn] = sum(1 for t in val_norm if pat.search(t))
name_terms = {t for t in all_terms if 0 < df[t] <= DF_MAX * len(rows)}
dropped = sorted((t for t in all_terms if df[t] > DF_MAX * len(rows)), key=lambda t: -df[t])
log(f"roster terms: {len(all_terms)} | kept as names: {len(name_terms)} | "
    f"dropped as common: {len(dropped)} e.g. {dropped[:6]}")

# ---- sample held-out utterances whose reference contains a gold name ----
cand = []
for i, r in enumerate(rows):
    ap = AUDIO / f"{r['city_id']}__{r['meeting_id']}.mp3"
    if not ap.exists(): continue
    try: dur = float(r["utterance_end"]) - float(r["utterance_start"])
    except: continue
    if not (5.0 <= dur <= 25.0) or len(r["text"].split()) < 10: continue
    terms = [norm(t) for t in (rosters.get(f"{r['city_id']}/{r['meeting_id']}") or []) if len(t) >= 5]
    gold = sorted({t for t in terms if t in name_terms and has(t, val_norm[i])})
    if not gold: continue
    # keep only the longest form when one gold term is a substring of another
    gold = [g for g in gold if not any(g != o and g in o for o in gold)]
    r["_gold"] = gold; r["_audio"] = str(ap)
    cand.append(r)
log(f"held-out utterances containing a roster name: {len(cand)}")
random.shuffle(cand)
sample = cand[:N_SAMPLE]
log(f"sampled {len(sample)} | cities: {Counter(r['city_id'] for r in sample)} | "
    f"gold names total: {sum(len(r['_gold']) for r in sample)}")

from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{BASE_CT2}/tokenizer.json")
def hotwords_for(city, meeting):
    terms = rosters.get(f"{city}/{meeting}") or []
    if not terms: return ""
    multi = sorted(t for t in terms if " " in t)
    single = sorted(t for t in terms if " " not in t)
    picked, used = [], 0
    for t in multi + single:
        n = len(tok.encode(" " + t).ids)
        if used + n > HOTWORD_TOKEN_BUDGET: continue
        picked.append(t); used += n
    return ", ".join(picked)
hw = {r["utterance_id"]: hotwords_for(r["city_id"], r["meeting_id"]) for r in sample}
log(f"utterances with hotwords: {sum(1 for v in hw.values() if v)}/{len(sample)}")

def cut(a, s, e):
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=SC); os.close(fd)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", str(s), "-to", str(e),
                    "-i", a, "-ar", "16000", "-ac", "1", tmp], check=True)
    return tmp
log("cutting clips ...")
clips = []
for r in sample:
    try: clips.append((r, cut(r["_audio"], r["utterance_start"], r["utterance_end"])))
    except Exception as e: log(f"cut fail {r['utterance_id']}: {e}")
log(f"clips ready: {len(clips)}")

from faster_whisper import WhisperModel
RUNS = [("base", BASE_CT2, False), ("base+hotwords", BASE_CT2, True),
        ("ours", OURS_CT2, False), ("ours+hotwords", OURS_CT2, True)]
hyps = {}
for name, path, use_hw in RUNS:
    log(f"=== {name} ===")
    m = WhisperModel(path, device="cpu", compute_type="float32", cpu_threads=8)
    hyps[name] = {}
    for i, (r, wav) in enumerate(clips):
        kw = {"hotwords": hw[r["utterance_id"]]} if use_hw and hw[r["utterance_id"]] else {}
        segs, _ = m.transcribe(wav, language="el", beam_size=2,
                               condition_on_previous_text=False, **kw)
        hyps[name][r["utterance_id"]] = " ".join(s.text.strip() for s in segs).strip()
        if (i + 1) % 10 == 0: log(f"  {name}: {i+1}/{len(clips)}")
    del m
    json.dump({"hyps": hyps, "hotwords": hw,
               "clips": [{"id": r["utterance_id"], "city": r["city_id"], "meeting": r["meeting_id"],
                          "ref": r["text"], "gold": r["_gold"]} for r, _ in clips]},
              open(f"{SC}/names_hyps.json", "w"), ensure_ascii=False)   # flush after each model

def wer(name):
    num = den = 0
    for r, _ in clips:
        ref = wtoks(r["text"]); num += edist(ref, wtoks(hyps[name][r["utterance_id"]])); den += len(ref)
    return num / den
def name_recall(name):
    tg = tm = 0
    for r, _ in clips:
        hn = norm(hyps[name][r["utterance_id"]])
        tg += len(r["_gold"]); tm += sum(1 for g in r["_gold"] if has(g, hn))
    return (tm / tg if tg else 0.0), tm, tg

print("\n==== NAME-FOCUSED A/B: held-out utterances containing a roster name ====")
print(f"n={len(clips)} (argos+orestiada, same faster-whisper stack)\n")
print(f"{'config':>16} | {'WER':>7} | {'name recall':>12}")
res = {}
for name, _, _ in RUNS:
    w = wer(name); rc, tm, tg = name_recall(name)
    res[name] = {"wer": w, "name_recall": rc, "matched": tm, "gold": tg}
    print(f"{name:>16} | {w:7.4f} | {rc*100:6.1f}%  ({tm}/{tg})")

def h2h(a, b):
    win = loss = tie = 0
    for r, _ in clips:
        ref = wtoks(r["text"])
        wa = edist(ref, wtoks(hyps[a][r["utterance_id"]])) / max(len(ref), 1)
        wb = edist(ref, wtoks(hyps[b][r["utterance_id"]])) / max(len(ref), 1)
        if wa < wb - 1e-9: win += 1
        elif wa > wb + 1e-9: loss += 1
        else: tie += 1
    return {"a_better": win, "a_worse": loss, "tie": tie}
h = {}
print()
for a, b in [("base+hotwords", "base"), ("ours+hotwords", "ours"), ("base+hotwords", "ours")]:
    h[f"{a}_vs_{b}"] = h2h(a, b)
    print(f"{a} vs {b}: better {h[f'{a}_vs_{b}']['a_better']} | "
          f"worse {h[f'{a}_vs_{b}']['a_worse']} | tie {h[f'{a}_vs_{b}']['tie']}")

json.dump({"n": len(clips), "n_gold": sum(len(r["_gold"]) for r, _ in clips),
           "results": res, "head2head": h},
          open(f"{Path(__file__).parent}/results_hotwords_names.json", "w"),
          ensure_ascii=False, indent=2)
for _, wav in clips: Path(wav).unlink(missing_ok=True)
log("done")
