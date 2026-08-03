# Experiment C of the "lexical costume" test (2026-07-29), see
# docs/reports/2026-07-29-lexical-thesis-experiments.md.
# PORTABILITY NOTE: needs the cached name-focused sample from ab_hotwords_names.py
# (CACHE below, written by that script), the ct2 model dir BASE_CT2, .venv-eval
# (faster_whisper>=1.0.2), ffmpeg, and data/asr/{val_manifest.csv,audio}.
"""Oracle candidate-selection bound for roster biasing (plus a distractor control).

On the same 59 name-focused held-out clips as ab_hotwords_names.py (base name recall
27.2%, full-roster hotwords 36.0%), run base whisper with:

  oracle     - hotwords = ONLY the roster names actually present in the reference
               (canonical roster surface forms, not the reference's inflected forms)
  distractor - hotwords = the same NUMBER of roster names, sampled from names that are
               NOT in the reference (controls for "shorter list = stronger bias")

Reported: WER, gold-name recall, and false name insertions (supplied names that appear
in the hypothesis but not the reference). This is an oracle *candidate-selection*
bound - what perfect knowledge of which names are spoken would buy - not a ceiling on
all contextual-biasing methods.
"""
import csv, json, os, re, subprocess, tempfile, random, time, unicodedata
from pathlib import Path

ROOT = "/home/harold/opencouncil-fine-tuning"
SC = os.environ.get("SC", ".")
PREV = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/231379b0-42e3-4cc6-a8d0-71d11d25d331/scratchpad"
CACHE = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/512d0560-8032-4651-80e9-da43053cd4b0/scratchpad/names_hyps.json"
BASE_CT2 = os.environ.get("BASE_CT2", f"{PREV}/base_f32_ct2")
AUDIO = Path(ROOT, "data/asr/audio")
ROSTERS = f"{ROOT}/data/pii/rosters_full.json"

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

p = json.load(open(CACHE))
clips_meta = p["clips"]           # id / city / meeting / ref / gold (normalized terms)
cached_hyps = p["hyps"]           # base, base+hotwords, ours, ours+hotwords
log(f"cached name-focused sample: {len(clips_meta)} clips, "
    f"{sum(len(c['gold']) for c in clips_meta)} gold names")

rosters = json.load(open(ROSTERS))
by_id = {r["utterance_id"]: r for r in csv.DictReader(open(f"{ROOT}/data/asr/val_manifest.csv"))}

# canonical roster surface form for each normalized gold term; distractors are roster
# terms of the same meeting whose normalized form is NOT in the reference.
def surfaces(city, meeting):
    return {norm(t): t for t in (rosters.get(f"{city}/{meeting}") or []) if len(t) >= 5}

oracle_hw, distract_hw = {}, {}
for c in clips_meta:
    sf = surfaces(c["city"], c["meeting"])
    gold_surf = [sf[g] for g in c["gold"] if g in sf]
    absent = sorted(t for tn, t in sf.items()
                    if not has(tn, norm(c["ref"])) and " " in t)   # full names only
    random.shuffle(absent)
    oracle_hw[c["id"]] = ", ".join(gold_surf)
    distract_hw[c["id"]] = ", ".join(absent[:max(len(gold_surf), 1)])
log(f"oracle terms/clip median: {sorted(len(v.split(', ')) for v in oracle_hw.values())[len(oracle_hw)//2]}")

def cut(a, s, e):
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=SC); os.close(fd)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", str(s), "-to", str(e),
                    "-i", a, "-ar", "16000", "-ac", "1", tmp], check=True)
    return tmp
log("cutting clips ...")
clips = []
for c in clips_meta:
    r = by_id[c["id"]]
    wav = cut(str(AUDIO / f"{c['city']}__{c['meeting']}.mp3"),
              r["utterance_start"], r["utterance_end"])
    clips.append((c, wav))

from faster_whisper import WhisperModel
hyps = {"base": cached_hyps["base"], "base+roster": cached_hyps["base+hotwords"]}
m = WhisperModel(BASE_CT2, device="cpu", compute_type="float32", cpu_threads=8)
for name, hw in [("base+oracle", oracle_hw), ("base+distractor", distract_hw)]:
    log(f"=== {name} ===")
    hyps[name] = {}
    for i, (c, wav) in enumerate(clips):
        kw = {"hotwords": hw[c["id"]]} if hw[c["id"]] else {}
        segs, _ = m.transcribe(wav, language="el", beam_size=2,
                               condition_on_previous_text=False, **kw)
        hyps[name][c["id"]] = " ".join(s.text.strip() for s in segs).strip()
        if (i + 1) % 10 == 0: log(f"  {name}: {i+1}/{len(clips)}")
    json.dump(hyps, open(f"{SC}/oracle_hyps.json", "w"), ensure_ascii=False)
del m

def wer(name):
    num = den = 0
    for c, _ in clips:
        ref = wtoks(c["ref"]); num += edist(ref, wtoks(hyps[name][c["id"]])); den += len(ref)
    return num / den
def name_recall(name):
    tg = tm = 0
    for c, _ in clips:
        hn = norm(hyps[name][c["id"]])
        tg += len(c["gold"]); tm += sum(1 for g in c["gold"] if has(g, hn))
    return (tm / tg if tg else 0.0), tm, tg
def false_insertions(name, hw):
    """Supplied hotword names that appear in the hypothesis but not in the reference."""
    fi = 0
    for c, _ in clips:
        hn, rn = norm(hyps[name][c["id"]]), norm(c["ref"])
        for t in (hw.get(c["id"]) or "").split(", "):
            if not t: continue
            tn = norm(t)
            if has(tn, hn) and not has(tn, rn): fi += 1
    return fi

print("\n==== ORACLE / DISTRACTOR BIASING: name-focused held-out subset ====")
print(f"n={len(clips)} clips, {sum(len(c['gold']) for c, _ in clips)} gold names "
      f"(base + roster rows reused from the 2026-07-25 run)\n")
res = {}
for name, hw in [("base", {}), ("base+roster", {}), ("base+oracle", oracle_hw),
                 ("base+distractor", distract_hw)]:
    w = wer(name); rc, tm, tg = name_recall(name)
    fi = false_insertions(name, hw) if hw else None
    res[name] = {"wer": w, "name_recall": rc, "matched": tm, "gold": tg,
                 "false_name_insertions": fi}
    fis = f" | false-ins {fi}" if fi is not None else ""
    print(f"{name:>16} | WER {w:.4f} | recall {rc*100:5.1f}% ({tm}/{tg}){fis}")

def h2h(a, b):
    win = loss = tie = 0
    for c, _ in clips:
        ref = wtoks(c["ref"])
        wa = edist(ref, wtoks(hyps[a][c["id"]])) / max(len(ref), 1)
        wb = edist(ref, wtoks(hyps[b][c["id"]])) / max(len(ref), 1)
        if wa < wb - 1e-9: win += 1
        elif wa > wb + 1e-9: loss += 1
        else: tie += 1
    return {"a_better": win, "a_worse": loss, "tie": tie}
h = {}
print()
for a, b in [("base+oracle", "base"), ("base+oracle", "base+roster"),
             ("base+distractor", "base")]:
    h[f"{a}_vs_{b}"] = h2h(a, b)
    print(f"{a} vs {b}: better {h[f'{a}_vs_{b}']['a_better']} | "
          f"worse {h[f'{a}_vs_{b}']['a_worse']} | tie {h[f'{a}_vs_{b}']['tie']}")

json.dump({"n": len(clips), "results": res, "head2head": h},
          open(Path(__file__).parent / "results_oracle_hotwords.json", "w"),
          ensure_ascii=False, indent=2)
for _, wav in clips: Path(wav).unlink(missing_ok=True)
log("done")
