# Experiment A of the "lexical costume" test (2026-07-29), see
# docs/reports/2026-07-29-lexical-thesis-experiments.md.
"""What KIND of edits are the human corrections, actually?

The asr-v2 design note claims the correction dataset is "a lexical and textual signal
(names, domain terms, punctuation, casing) wearing an acoustic costume". This script
quantifies that claim over every included correction in data/asr/export.jsonl:

  1. utterance-level: is the whole edit invisible after WER normalization
     (punctuation / casing / diacritics only)?
  2. word-level (on the normalized token diff): classify each edit op as
     number | roster-name | glossary-term | spelling-tweak | other-lexical.

"other-lexical" is the residue that looks like genuine misrecognition of ordinary
words - the only class that is plainly acoustic.
"""
import json, re, unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")

def norm(s):
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
def wtoks(s): return re.findall(r"\w+", norm(s))

def cedist(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]

# ---- context term banks -------------------------------------------------------
# The global glossary is polluted with common Greek words (verified by sampling:
# "κανει", "πραγμα", "πολλοι" ...), which mis-attributes ordinary mishearings to
# "glossary-term". Use per-city glossaries + rosters only, and drop any term token
# whose document frequency over the corrected references exceeds DF_MAX - same idea
# as the name filter in ab_hotwords_names.py.
DF_MAX = 0.02
rosters = json.load(open(ROOT / "data/pii/rosters_full.json"))
glossary = json.load(open(ROOT / "data/glossary/glossary.json"))

def term_tokens(terms):
    out = set()
    for t in terms:
        for w in wtoks(t):
            if len(w) >= 4: out.add(w)
    return out

_refs = []
for _line in open(ROOT / "data/asr/export.jsonl"):
    _d = json.loads(_line)
    if _d.get("include_status") == "include":
        _refs.append(set(wtoks(_d.get("final_after_text", ""))))
def df(tok): return sum(1 for r in _refs if tok in r)
_cache = {}
def is_common(tok):
    if tok not in _cache: _cache[tok] = df(tok) > DF_MAX * len(_refs)
    return _cache[tok]
def filtered(toks): return {t for t in toks if not is_common(t)}

roster_toks = {k: filtered(term_tokens(v)) for k, v in rosters.items()}
gloss_city = {c: filtered(term_tokens(v)) for c, v in glossary.get("per_city", {}).items()}

def classify_op(b_toks, a_toks, rtoks, gtoks):
    toks = b_toks + a_toks
    if any(re.search(r"\d", t) for t in toks): return "number"
    if any(t in rtoks for t in toks): return "roster-name"
    if any(t in gtoks for t in toks): return "glossary-term"
    if b_toks and a_toks:
        bj, aj = " ".join(b_toks), " ".join(a_toks)
        d = cedist(bj, aj)
        if d <= max(2, int(0.34 * max(len(bj), len(aj)))): return "spelling-tweak"
    return "other-lexical"

# ---- pass over the corrections ------------------------------------------------
ops = Counter()          # edit ops per class
tokvol = Counter()       # changed-token volume per class (max of the two sides)
utt = Counter()          # utterance-level outcome
utt_class = Counter()    # for non-formatting utts: the mix of classes they contain
n = 0
for line in open(ROOT / "data/asr/export.jsonl"):
    d = json.loads(line)
    if d.get("include_status") != "include": continue
    bef, aft = d.get("initial_before_text", ""), d.get("final_after_text", "")
    if not aft or bef.strip() == aft.strip(): continue
    n += 1
    bt, at = wtoks(bef), wtoks(aft)
    if bt == at:
        utt["pure-formatting"] += 1
        continue
    key = f"{d['city_id']}/{d['meeting_id']}"
    rtoks = roster_toks.get(key, set())
    gtoks = gloss_city.get(d["city_id"], set())
    classes = set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, bt, at, autojunk=False).get_opcodes():
        if tag == "equal": continue
        c = classify_op(bt[i1:i2], at[j1:j2], rtoks, gtoks)
        ops[c] += 1
        tokvol[c] += max(i2 - i1, j2 - j1)
        classes.add(c)
    utt["word-level-change"] += 1
    if classes <= {"number", "roster-name", "glossary-term", "spelling-tweak"}:
        utt_class["all-ops-contextual/textual"] += 1
    elif "other-lexical" in classes and len(classes) == 1:
        utt_class["all-ops-other-lexical"] += 1
    else:
        utt_class["mixed"] += 1

res = {
    "n_corrections": n,
    "utterance_level": {
        "pure_formatting_pct": round(100 * utt["pure-formatting"] / n, 1),
        **dict(utt),
    },
    "word_level_ops": {k: {"ops": ops[k], "ops_pct": round(100 * ops[k] / sum(ops.values()), 1),
                           "token_volume": tokvol[k],
                           "tok_pct": round(100 * tokvol[k] / sum(tokvol.values()), 1)}
                       for k in sorted(ops, key=lambda k: -ops[k])},
    "non_formatting_utterances": dict(utt_class),
}
print(json.dumps(res, ensure_ascii=False, indent=2))
json.dump(res, open(Path(__file__).parent / "results_edit_taxonomy.json", "w"),
          ensure_ascii=False, indent=2)
