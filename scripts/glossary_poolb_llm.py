"""Pool B, stage 2: cheap-LLM evidence and proposal, on leak-free transcripts only.

Two jobs, both through the codex bridge with `gpt-5.6-luna` at low effort, the same
mechanism `eval/hparl2_punctuate.py` already uses:

  expand   For each acronym that survived scripts/glossary_distill.py, read a handful of
           concordance lines and return the expansion, the kind of body, and whether it
           exists in every Greek municipality. This is the C4 category evidence: the
           acronym regex proves a shape, not a meaning.

  propose  Read whole-meeting excerpts and propose institutional or legal terms that
           recur in every municipal council. Discovery only. Everything proposed still
           has to pass the same pre-registered C1-C5 filter, and a proposal that no
           transcript attests is dropped.

Source meetings are exactly those of scripts/glossary_poolb_mine.py: TRAIN fold and
absent from the benchmark. The LLM never sees a benchmark reference.

Transcript text goes to the model and to a local cache. Never to git.

Run:  <venv>/bin/python scripts/glossary_poolb_llm.py expand
      <venv>/bin/python scripts/glossary_poolb_llm.py propose
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.glossary_poolb_mine import meeting_text, source_meetings  # noqa: E402

CLIENT = "/home/harold/codex-bridge/codex_client.py"
LLM_MODEL = "gpt-5.6-luna"
WORK = Path.home() / ".cache/oc-public/glossary"
OUT_EXPAND = ROOT / "data/glossary/poolb_expansions.json"
OUT_PROPOSE = ROOT / "data/glossary/poolb_proposals.json"
CANDIDATES_B = ROOT / "data/glossary/candidates_pool_b.json"

SEED = 20260816
N_PROPOSE_MEETINGS = 24     # stratified over the 8 source cities
EXCERPT_CHARS = 9000
BATCH_ACRONYMS = 20

EXPAND_PROMPT = """Είσαι βοηθός ορολογίας για ελληνικά δημοτικά συμβούλια.

Για κάθε ακρωνύμιο σου δίνω μερικές γραμμές από πρακτικά όπου εμφανίζεται.

Για κάθε ένα επίστρεψε:
- "acronym": το ακρωνύμιο όπως το έδωσα
- "expansion": η πλήρης ονομασία στα ελληνικά, ή null αν δεν είσαι σίγουρος
- "kind": ένα από "org_national", "org_municipal", "org_utility", "legal",
  "financial", "other"
- "universal": true αν ο φορέας ή ο όρος υπάρχει σε ΚΑΘΕ ελληνικό δήμο,
  false αν είναι τοπικός ή αφορά συγκεκριμένη πόλη
- "confusable_with": κοινή ελληνική λέξη με την οποία μπορεί να μπερδευτεί
  ακουστικά, ή null
- "confidence": "high" | "medium" | "low"

Μην μαντεύεις. Αν δεν αναγνωρίζεις το ακρωνύμιο, βάλε expansion null και
confidence "low".

Επίστρεψε ΜΟΝΟ ένα JSON array.

ΣΤΟΙΧΕΙΑ:
"""

PROPOSE_PROMPT = """Διαβάζεις απόσπασμα από πρακτικά ελληνικού δημοτικού συμβουλίου.

Βρες θεσμικούς και νομικούς όρους που επαναλαμβάνονται σε ΚΑΘΕ ελληνικό δημοτικό
συμβούλιο, ανεξάρτητα από πόλη. Παραδείγματα κατηγορίας: τύποι αποφάσεων,
διαδικαστικά στάδια, ονομασίες νομοθετημάτων, είδη προϋπολογιστικών πράξεων.

ΜΗΝ προτείνεις:
- ονόματα προσώπων, τοπωνύμια, ονόματα παρατάξεων, ονόματα τοπικών φορέων
- καθημερινές λέξεις της ελληνικής
- ό,τι αφορά μόνο αυτή τη συνεδρίαση ή αυτή την πόλη

Για κάθε όρο επίστρεψε:
- "term": ο όρος στην ονομαστική, όπως θα τον έγραφε κανείς σε λεξικό
- "kind": "legal" | "procedural" | "financial"
- "evidence": η ακριβής φράση του αποσπάσματος όπου εμφανίζεται
- "why_not_common": σε μία πρόταση, γιατί δεν συγχέεται με κοινή λέξη

Το πολύ 12 όροι. Επίστρεψε ΜΟΝΟ ένα JSON array.

ΑΠΟΣΠΑΣΜΑ:
"""


def call_llm(prompt: str, timeout_wait: int = 900) -> list[dict]:
    p = subprocess.run(
        [sys.executable, CLIENT, "enqueue", "exec",
         "-c", f"model={LLM_MODEL}", "-c", "model_reasoning_effort=low", prompt],
        capture_output=True, text=True, timeout=180)
    try:
        job = json.loads(p.stdout)["job_id"]
    except Exception:
        raise RuntimeError(f"enqueue failed: {p.stdout[-300:]} {p.stderr[-300:]}")
    w = subprocess.run([sys.executable, CLIENT, "wait", job, str(timeout_wait)],
                       capture_output=True, text=True, timeout=timeout_wait + 180)
    res = json.loads(w.stdout)
    if res.get("status") != "completed":
        raise RuntimeError(f"job {job}: {res.get('status')}")
    return parse_json_array(res.get("output") or "")


def parse_json_array(text: str) -> list[dict]:
    m = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    blob = m[-1] if m else None
    if blob is None:
        i, j = text.find("["), text.rfind("]")
        if i < 0 or j <= i:
            return []
        blob = text[i:j + 1]
    try:
        out = json.loads(blob)
    except Exception:
        return []
    return out if isinstance(out, list) else []


def concordances(terms: set[str], per_term: int = 4) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = defaultdict(list)
    for city, meeting in source_meetings():
        for line in meeting_text(city, meeting).splitlines():
            toks = {t.strip("«»\"'()[]{}·,;:!?…—–-.") for t in line.split()}
            for t in terms & toks:
                if len(hits[t]) < per_term:
                    hits[t].append(line.strip()[:280])
        if all(len(hits[t]) >= per_term for t in terms):
            break
    return hits


def cmd_expand() -> None:
    cand = json.loads(CANDIDATES_B.read_text())
    terms = {c["term"] for c in cand["candidates"]}
    print(f"{len(terms)} surviving acronyms")
    ctx = concordances(terms)
    items = sorted(terms)
    out = []
    for i in range(0, len(items), BATCH_ACRONYMS):
        batch = items[i:i + BATCH_ACRONYMS]
        payload = json.dumps([{"acronym": a, "lines": ctx.get(a, [])} for a in batch],
                             ensure_ascii=False)
        print(f"  batch {i // BATCH_ACRONYMS + 1}: {len(batch)} acronyms", flush=True)
        out.extend(call_llm(EXPAND_PROMPT + payload))
    WORK.mkdir(parents=True, exist_ok=True)
    OUT_EXPAND.write_text(json.dumps(
        {"model": LLM_MODEL, "prompt_sha_note": "see scripts/glossary_poolb_llm.py",
         "n_acronyms": len(items), "expansions": out},
        ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT_EXPAND} ({len(out)} rows)")


def cmd_propose() -> None:
    pairs = source_meetings()
    by_city = defaultdict(list)
    for c, m in pairs:
        by_city[c].append((c, m))
    rng = random.Random(SEED)
    picked = []
    cities = sorted(by_city)
    while len(picked) < N_PROPOSE_MEETINGS:
        added = False
        for c in cities:
            pool = [p for p in by_city[c] if p not in picked]
            if not pool or len(picked) >= N_PROPOSE_MEETINGS:
                continue
            picked.append(rng.choice(pool))
            added = True
        if not added:
            break
    print(f"{len(picked)} meetings over {len({c for c, _ in picked})} cities")

    proposals = []
    for city, meeting in picked:
        text = meeting_text(city, meeting)
        start = max(0, (len(text) - EXCERPT_CHARS) // 2)
        excerpt = text[start:start + EXCERPT_CHARS]
        print(f"  {city}/{meeting} ({len(excerpt)} chars)", flush=True)
        try:
            rows = call_llm(PROPOSE_PROMPT + excerpt)
        except RuntimeError as e:
            print(f"    skipped: {e}", flush=True)
            continue
        for r in rows:
            if isinstance(r, dict) and r.get("term"):
                r["_city"] = city
                r["_meeting"] = meeting
                r.pop("evidence", None)   # do not persist verbatim transcript
                proposals.append(r)

    counts: dict[str, set] = defaultdict(set)
    cityset: dict[str, set] = defaultdict(set)
    kind: dict[str, str] = {}
    why: dict[str, str] = {}
    for r in proposals:
        t = " ".join(str(r["term"]).split()).strip()
        counts[t].add((r["_city"], r["_meeting"]))
        cityset[t].add(r["_city"])
        kind.setdefault(t, r.get("kind") or "unknown")
        why.setdefault(t, r.get("why_not_common") or "")
    rows = sorted(({"term": t, "kind": kind[t],
                    "n_meetings_proposed_from": len(counts[t]),
                    "n_cities_proposed_from": len(cityset[t]),
                    "cities": sorted(cityset[t]),
                    "why_not_common": why[t]} for t in counts),
                  key=lambda r: (-r["n_cities_proposed_from"],
                                 -r["n_meetings_proposed_from"], r["term"]))
    OUT_PROPOSE.write_text(json.dumps(
        {"model": LLM_MODEL, "seed": SEED,
         "source": {"rule": "TRAIN fold AND absent from the benchmark",
                    "meetings": [{"city_id": c, "meeting_id": m} for c, m in picked]},
         "n_raw_proposals": len(proposals), "n_distinct": len(rows),
         "proposals": rows}, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT_PROPOSE} ({len(rows)} distinct of {len(proposals)} raw)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "expand":
        cmd_expand()
    elif cmd == "propose":
        cmd_propose()
    else:
        raise SystemExit("usage: glossary_poolb_llm.py expand|propose")
