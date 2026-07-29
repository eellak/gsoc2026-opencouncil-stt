"""Does the fine-tune's advantage survive CONTINUOUS decoding, or is it an artifact
of evaluating on isolated cut clips?

The n=300 run showed ours beating base by -4.35pp, concentrated in SHORT utterances
(-8.98pp). But that eval cuts each utterance out on its own -- exactly the condition the
model was trained on. Production decodes continuous meeting audio, where no utterance is
presented in isolation. This script tests both conditions on the SAME utterances:

  condition ISO : cut the utterance out (+/-0.2s) and decode it alone      <- old setup
  condition CTX : decode a continuous 25-min window of the real meeting,
                  then extract each utterance's words by word-level timestamps

If ours wins in ISO but not in CTX, the gain is an eval artifact and must not be migrated.
If it wins in both, the gain is real under production-like conditions.

Also reports window-level WER (whole 25-min transcript vs assembled reference), which is
the closest thing to a production number -- see the caveat about that reference below.

Sample: 3 held-out meetings (2 orestiada + 1 argos), for each the 25-min contiguous window
containing the most human-corrected utterances.

Outputs (incremental, safe to interrupt):
  eval/diagnose/ctx_hyps.json / ctx_results.json
Run:  .venv-eval/bin/python eval/diagnose/full_meeting_context.py
"""
import os, csv, json, re, subprocess, tempfile, time, unicodedata
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
OUT = ROOT / "eval/diagnose"; OUT.mkdir(parents=True, exist_ok=True)
AUDIO = ROOT / "data/asr/audio"
WORK = Path(tempfile.mkdtemp(prefix="ctx_", dir="/tmp"))
SCR = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/231379b0-42e3-4cc6-a8d0-71d11d25d331/scratchpad"
MODELS = [("base", f"{SCR}/base_f32_ct2"), ("ours", f"{SCR}/ours_f32_ct2")]
MEETINGS = [("orestiada", "mar23_2026"), ("orestiada", "jan21_2026"), ("argos", "dec9_2025")]
WIN_S = float(os.environ.get("WIN_S", 1500))   # 25 min
PAD_S, SR = 0.2, 16000

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

_P = re.compile(r"[^\w\s]", re.UNICODE)
def gnorm(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFC", s).replace("ς", "σ")
    return re.sub(r"\s+", " ", _P.sub(" ", s)).strip()

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

def pooled(pairs):
    num = den = 0
    for ref, hyp in pairs:
        r, h = gnorm(ref).split(), gnorm(hyp).split()
        num += edist(r, h); den += len(r)
    return 100.0 * num / den if den else float("nan")

# ---------------- assemble reference timeline ----------------
corr = {}
for line in open(ROOT / "data/asr/export.jsonl"):
    d = json.loads(line)
    b, a = d.get("initial_before_text", ""), d.get("final_after_text", "")
    if a and b.strip() != a.strip():
        corr[d["utterance_id"]] = d

utts = {}
for r in csv.DictReader(open(ROOT / "data/asr/val_manifest.csv")):
    k = (r["city_id"], r["meeting_id"])
    utts.setdefault(k, []).append(r)

plan = []
for city, mtg in MEETINGS:
    rows = sorted(utts.get((city, mtg), []), key=lambda r: float(r["utterance_start"]))
    if not rows: log(f"!! no manifest rows for {city}__{mtg}"); continue
    cids = [r for r in rows if r["utterance_id"] in corr]
    if not cids: log(f"!! no corrections for {city}__{mtg}"); continue
    # window start maximising corrected utterances fully inside
    best, bs = -1, 0.0
    for r in cids:
        t0 = max(0.0, float(r["utterance_start"]) - 5.0)
        c = sum(1 for x in cids if float(x["utterance_start"]) >= t0
                and float(x["utterance_end"]) <= t0 + WIN_S)
        if c > best: best, bs = c, t0
    inside = [r for r in rows if float(r["utterance_start"]) >= bs
              and float(r["utterance_end"]) <= bs + WIN_S]
    ci = [r for r in inside if r["utterance_id"] in corr]
    ap = AUDIO / f"{city}__{mtg}.mp3"
    if not ap.exists(): log(f"!! missing audio {ap}"); continue
    plan.append({"city": city, "mtg": mtg, "audio": str(ap), "t0": bs,
                 "inside": inside, "corrected": ci})
    log(f"{city}__{mtg}: window {bs/60:.1f}-{(bs+WIN_S)/60:.1f} min | "
        f"{len(inside)} utts, {len(ci)} corrected")

def ref_of(r):
    d = corr.get(r["utterance_id"])
    return d["final_after_text"] if d else r["text"]

def ffmpeg_cut(src, s, e, dst):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", str(s),
                    "-to", str(e), "-i", src, "-ar", str(SR), "-ac", "1", str(dst)], check=True)
    return str(dst)

HF, RF = OUT / "ctx_hyps.json", OUT / "ctx_results.json"
H = json.load(open(HF)) if HF.exists() else {}
def save(): json.dump(H, open(HF, "w"), ensure_ascii=False, indent=1)

from faster_whisper import WhisperModel

for tag, path in MODELS:
    if not Path(path, "model.bin").exists():
        log(f"!! missing model {path}"); continue
    need = any(f"{c}__{m}__{tag}__{k}" not in H
               for c, m in [(p["city"], p["mtg"]) for p in plan] for k in ("ctx", "iso"))
    if not need: log(f"skip {tag} (cached)"); continue
    log(f"loading {tag} (float32)")
    model = WhisperModel(path, device="cpu", compute_type="float32", cpu_threads=8)

    for p in plan:
        base_key = f"{p['city']}__{p['mtg']}__{tag}"
        # ---- CTX: continuous window decode with word timestamps ----
        ck = f"{base_key}__ctx"
        if ck not in H:
            wav = ffmpeg_cut(p["audio"], p["t0"], p["t0"] + WIN_S, WORK / f"{base_key}_win.wav")
            log(f"=== CTX {base_key} ({WIN_S/60:.0f} min continuous) ===")
            t = time.time()
            segs, _ = model.transcribe(wav, language="el", beam_size=2,
                                       condition_on_previous_text=False, word_timestamps=True)
            words, full = [], []
            for s in segs:
                full.append(s.text.strip())
                for w in (s.words or []):
                    words.append({"w": w.word, "s": float(w.start), "e": float(w.end)})
            H[ck] = {"words": words, "text": " ".join(full).strip()}
            log(f"    done in {time.time()-t:.0f}s, {len(words)} words")
            Path(wav).unlink(missing_ok=True); save()
        # ---- ISO: same utterances, cut out individually ----
        ik = f"{base_key}__iso"
        H.setdefault(ik, {})
        todo = [r for r in p["corrected"] if r["utterance_id"] not in H[ik]]
        if todo:
            log(f"=== ISO {base_key} ({len(todo)} isolated clips) ===")
            for i, r in enumerate(todo):
                s = max(0.0, float(r["utterance_start"]) - PAD_S)
                e = float(r["utterance_end"]) + PAD_S
                wav = ffmpeg_cut(p["audio"], s, e, WORK / "iso.wav")
                sg, _ = model.transcribe(wav, language="el", beam_size=2,
                                         condition_on_previous_text=False)
                H[ik][r["utterance_id"]] = " ".join(x.text.strip() for x in sg).strip()
                if (i + 1) % 10 == 0: log(f"    iso {i+1}/{len(todo)}"); save()
            save()
    del model

# ---------------- score ----------------
def extract(words, t0, s, e):
    """hypothesis words whose midpoint falls inside the utterance span (window-relative)."""
    a, b = s - t0, e - t0
    return " ".join(w["w"] for w in words if a <= (w["s"] + w["e"]) / 2 <= b).strip()

table, per_meeting = [], []
for cond in ("iso", "ctx"):
    for tag, _ in MODELS:
        pairs_c, pairs_all = [], []
        for p in plan:
            k = f"{p['city']}__{p['mtg']}__{tag}__{cond}"
            if k not in H: continue
            for r in p["corrected"]:
                ref = ref_of(r)
                hyp = (H[k].get(r["utterance_id"], "") if cond == "iso"
                       else extract(H[k]["words"], p["t0"],
                                    float(r["utterance_start"]), float(r["utterance_end"])))
                pairs_c.append((ref, hyp))
            if cond == "ctx":
                for r in p["inside"]:
                    pairs_all.append((ref_of(r), extract(H[k]["words"], p["t0"],
                                     float(r["utterance_start"]), float(r["utterance_end"]))))
        if pairs_c:
            table.append({"cond": cond, "model": tag, "n_corrected": len(pairs_c),
                          "wer_corrected": pooled(pairs_c),
                          "n_all": len(pairs_all) or None,
                          "wer_all": pooled(pairs_all) if pairs_all else None})

# window-level WER: continuous transcript vs concatenated reference, restricted to the
# time span the reference actually covers (otherwise speech outside the annotated
# utterances counts as insertions and WER exceeds 100%).
win = []
for tag, _ in MODELS:
    pr = []
    for p in plan:
        k = f"{p['city']}__{p['mtg']}__{tag}__ctx"
        if k not in H or not p["inside"]: continue
        lo = min(float(r["utterance_start"]) for r in p["inside"])
        hi = max(float(r["utterance_end"]) for r in p["inside"])
        hyp = extract(H[k]["words"], p["t0"], lo, hi)
        pr.append((" ".join(ref_of(r) for r in p["inside"]), hyp))
    if pr: win.append({"model": tag, "wer_window": pooled(pr), "n_windows": len(pr)})

print("\n==== ISOLATED CLIPS vs CONTINUOUS CONTEXT (same utterances) ====")
print(f"{'cond':<5} {'model':<6} {'n_corr':>7} {'WER corrected':>14} {'n_all':>7} {'WER all':>9}")
for r in table:
    na = r["n_all"] if r["n_all"] else "-"
    wa = f"{r['wer_all']:.2f}" if r["wer_all"] is not None else "-"
    print(f"{r['cond']:<5} {r['model']:<6} {r['n_corrected']:>7} {r['wer_corrected']:>14.2f} {str(na):>7} {wa:>9}")
print("\n==== window-level (whole 25-min transcript, reference partly Scribe) ====")
for r in win: print(f"  {r['model']:<6} WER={r['wer_window']:.2f}  ({r['n_windows']} windows)")

# per-utterance head-to-head, per condition
h2h = {}
for cond in ("iso", "ctx"):
    w = l = t = 0
    for p in plan:
        kb, ko = (f"{p['city']}__{p['mtg']}__base__{cond}", f"{p['city']}__{p['mtg']}__ours__{cond}")
        if kb not in H or ko not in H: continue
        for r in p["corrected"]:
            ref = gnorm(ref_of(r)).split()
            if not ref: continue
            def hy(k):
                return (H[k].get(r["utterance_id"], "") if cond == "iso"
                        else extract(H[k]["words"], p["t0"],
                                     float(r["utterance_start"]), float(r["utterance_end"])))
            b = edist(ref, gnorm(hy(kb)).split()) / len(ref)
            o = edist(ref, gnorm(hy(ko)).split()) / len(ref)
            if o < b - 1e-9: w += 1
            elif o > b + 1e-9: l += 1
            else: t += 1
    h2h[cond] = {"ours_better": w, "worse": l, "tie": t}
    print(f"\n{cond}: ours vs base -> better {w} | worse {l} | tie {t}")

json.dump({"window_s": WIN_S, "meetings": [f"{p['city']}__{p['mtg']}" for p in plan],
           "table": table, "window_level": win, "head2head": h2h},
          open(RF, "w"), ensure_ascii=False, indent=2)
log(f"wrote {RF}")
for f in WORK.glob("*"): f.unlink(missing_ok=True)
try: WORK.rmdir()
except OSError: pass
log("done")
