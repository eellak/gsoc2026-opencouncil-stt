# Experiment B of the "lexical costume" test (2026-07-29), see
# docs/reports/2026-07-29-lexical-thesis-experiments.md.
# PORTABILITY NOTE: needs the cached corrected-sample hypotheses from the 2026-07-25
# hotwords run (CACHE below) and the on-box `claude` CLI (OAuth, via eval/backends.py).
"""Can an LLM post-editor with per-meeting context close the gap the fine-tune could not?

On the 50 actually-corrected held-out utterances (same sample as the capstone A/B and
ab_hotwords.py), take the ORIGINAL ASR text (scribe_before, WER 0.1552 vs the human
reference) and ask claude-sonnet to correct it. Three configs isolate what does the work:

  llm_plain    - LLM cleanup, no context
  llm_roster   - LLM + the same per-meeting roster string the biasing runs used
  llm_base     - LLM + roster, applied to base whisper output instead of scribe

Scored with the exact WER normalization of the controlled A/Bs. If the LLM recovers a
large share of the remaining gap, the corrections are (mostly) a text task.
"""
import json, os, re, sys, time, unicodedata
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
CACHE = "/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/512d0560-8032-4651-80e9-da43053cd4b0/scratchpad/hotwords_hyps.json"
SC = os.environ.get("SC", ".")
MODEL = os.environ.get("LLM_MODEL", "sonnet")

from eval.backends import generate

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

p = json.load(open(CACHE))
refs, hyps, hotwords = p["refs"], p["hyps"], p["hotwords"]
ids = sorted(refs)
log(f"cached corrected sample: {len(ids)} utterances")

SYSTEM = (
    "Είσαι επιμελητής απομαγνητοφωνήσεων ελληνικών δημοτικών συμβουλίων. Θα λάβεις "
    "κείμενο από αυτόματη αναγνώριση ομιλίας (ASR), το οποίο μπορεί να περιέχει "
    "παραφθαρμένες λέξεις, λάθος ονόματα ή λάθος καταλήξεις. Διόρθωσε ΜΟΝΟ ό,τι είναι "
    "πιθανό λάθος αναγνώρισης ώστε το κείμενο να αποδίδει αυτό που ειπώθηκε. Μην "
    "παραφράζεις, μην αφαιρείς και μην προσθέτεις περιεχόμενο, μην αλλάζεις τη σειρά. "
    "Αν δεν είσαι βέβαιος, άφησε τη λέξη όπως είναι. Επίστρεψε ΜΟΝΟ το διορθωμένο "
    "κείμενο, χωρίς εισαγωγικά, σχόλια ή εξηγήσεις."
)

def user_prompt(text, roster):
    ctx = f"Ομιλητές και πρόσωπα της συνεδρίασης (για σωστή απόδοση ονομάτων):\n{roster}\n\n" if roster else ""
    return f"{ctx}Κείμενο ASR προς διόρθωση:\n{text}"

CONFIGS = [
    ("llm_plain",  "scribe_before", False),
    ("llm_roster", "scribe_before", True),
    ("llm_base",   "base",          True),
]

out_path = Path(SC) / "llm_postedit_hyps.json"
out = json.load(open(out_path)) if out_path.exists() else {}
for name, src, use_roster in CONFIGS:
    out.setdefault(name, {})
    todo = [u for u in ids if u not in out[name]]
    log(f"=== {name} (src={src}, roster={use_roster}) - {len(todo)} to do ===")
    for i, u in enumerate(todo):
        roster = hotwords.get(u, "") if use_roster else ""
        try:
            txt = generate(SYSTEM, user_prompt(hyps[src][u], roster),
                           backend="claude", model=MODEL, timeout=240)
        except Exception as e:
            log(f"  FAIL {u}: {e}"); continue
        out[name][u] = txt.strip()
        if (i + 1) % 5 == 0:
            json.dump(out, open(out_path, "w"), ensure_ascii=False)
            log(f"  {name}: {i+1}/{len(todo)}")
    json.dump(out, open(out_path, "w"), ensure_ascii=False)

def wer(h):
    num = den = 0
    for u in ids:
        ref = wtoks(refs[u]); num += edist(ref, wtoks(h[u])); den += len(ref)
    return num / den
def h2h(ha, hb):
    win = loss = tie = 0
    for u in ids:
        ref = wtoks(refs[u])
        wa = edist(ref, wtoks(ha[u])) / max(len(ref), 1)
        wb = edist(ref, wtoks(hb[u])) / max(len(ref), 1)
        if wa < wb - 1e-9: win += 1
        elif wa > wb + 1e-9: loss += 1
        else: tie += 1
    return {"a_better": win, "a_worse": loss, "tie": tie}

print("\n==== LLM POST-EDIT: corrected held-out utterances, vs human reference ====")
print(f"n={len(ids)} (same sample + normalization as the capstone A/B; model={MODEL})\n")
res = {}
for name in ("scribe_before", "base", "ours"):
    res[name] = {"wer": wer(hyps[name])}
    print(f"{name:>14} | {res[name]['wer']:.4f}")
for name, src, _ in CONFIGS:
    done = {u: out[name].get(u, hyps[src][u]) for u in ids}   # missing -> unchanged src
    res[name] = {"wer": wer(done), "completed": len(out[name])}
    print(f"{name:>14} | {res[name]['wer']:.4f}  ({len(out[name])}/{len(ids)} edited)")

h = {}
for a, b in [("llm_plain", "scribe_before"), ("llm_roster", "scribe_before"),
             ("llm_roster", "llm_plain"), ("llm_base", "base"), ("llm_roster", "ours")]:
    da = {u: out[a].get(u, "") for u in ids} if a.startswith("llm") else hyps[a]
    db = {u: out[b].get(u, "") for u in ids} if b.startswith("llm") else hyps[b]
    if a.startswith("llm"):
        src = dict(CONFIGS)[a] if False else {n: s for n, s, _ in CONFIGS}[a]
        da = {u: out[a].get(u, hyps[src][u]) for u in ids}
    if b.startswith("llm"):
        src = {n: s for n, s, _ in CONFIGS}[b]
        db = {u: out[b].get(u, hyps[src][u]) for u in ids}
    h[f"{a}_vs_{b}"] = h2h(da, db)
    print(f"{a} vs {b}: better {h[f'{a}_vs_{b}']['a_better']} | "
          f"worse {h[f'{a}_vs_{b}']['a_worse']} | tie {h[f'{a}_vs_{b}']['tie']}")

# paired bootstrap CI on the WER delta (2000 resamples, per-utterance pairs)
import random as _rnd
_rnd.seed(13)
def boot_ci(ha, hb, B=2000):
    pairs = []
    for u in ids:
        ref = wtoks(refs[u])
        pairs.append((edist(ref, wtoks(ha[u])), edist(ref, wtoks(hb[u])), len(ref)))
    deltas = []
    for _ in range(B):
        s = [pairs[_rnd.randrange(len(pairs))] for _ in pairs]
        den = sum(p[2] for p in s) or 1
        deltas.append((sum(p[0] for p in s) - sum(p[1] for p in s)) / den)
    deltas.sort()
    return deltas[int(0.025 * B)], deltas[int(0.975 * B)]

cis = {}
print()
for a, b in [("llm_plain", "scribe_before"), ("llm_roster", "scribe_before"),
             ("llm_base", "base")]:
    src = {n: s for n, s, _ in CONFIGS}[a]
    da = {u: out[a].get(u, hyps[src][u]) for u in ids}
    lo, hi = boot_ci(da, hyps[b])
    cis[f"{a}_minus_{b}"] = [lo, hi]
    print(f"WER delta {a} - {b}: 95% CI [{lo:+.4f}, {hi:+.4f}]")

json.dump({"n": len(ids), "model": MODEL, "results": res, "head2head": h,
           "bootstrap_ci": cis},
          open(Path(__file__).parent / "results_llm_postedit.json", "w"),
          ensure_ascii=False, indent=2)
log("done")
