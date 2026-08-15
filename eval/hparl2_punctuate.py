"""Restore punctuation and casing on the kept HParl2 rows.

The two sides we have per row carry different halves of a good label:

  - the dataset transcript: accented, human-derived word sequence, but lowercase
    and with zero punctuation (0% of rows in any shard have a mark)
  - the Soniox output: punctuation, casing and sentence structure, but it is ASR
    and it gets words wrong

So the label we want is **the dataset's words with Soniox's punctuation**. Two
routes, both run here so they can be compared:

  A. `transfer()` — deterministic. Align the two token sequences; where a dataset
     token matches its Soniox token, adopt that token's leading capital and
     trailing punctuation. Never invents anything. This is the fallback and the
     honest baseline.

  B. `--llm` — the same job given to gpt-5.6-luna through the codex bridge, in
     batches, because A cannot judge *whether a segment is a complete sentence*.
     Most HParl segments are cut mid-sentence, so a blanket trailing period would
     be wrong; the model is told to leave those open.

Route B is only accepted per-row when its output has the **identical
greek_normalize token sequence** as the dataset transcript. Any row where the
model added, dropped or changed a word falls back to A and is counted. That guard
is the whole reason this is safe to run unattended.

Run:
    .venv-eval/bin/python eval/hparl2_punctuate.py                 # A only
    .venv-eval/bin/python eval/hparl2_punctuate.py --llm --batch 25
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.scoring import greek_normalize  # noqa: E402

WORK = Path.home() / ".cache/oc-public/hparl2"
CLIENT = "/home/harold/codex-bridge/codex_client.py"
LLM_MODEL = "gpt-5.6-luna"
LLM_CACHE = WORK / "punct_llm_cache.jsonl"

LEAD_PUNCT = "«\"'(-—"
TRAIL_PUNCT = ".,;:!?·»\"')…-—"


def _bare(tok: str) -> str:
    """Token stripped of edge punctuation, accents, case — for matching only."""
    t = tok.strip(LEAD_PUNCT + TRAIL_PUNCT)
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.replace("ς", "σ")


def transfer(ref: str, hyp: str) -> str:
    """Dataset words + Soniox punctuation/casing, on matched tokens only."""
    R, H = ref.split(), hyp.split()
    sm = difflib.SequenceMatcher(a=[_bare(t) for t in R], b=[_bare(t) for t in H],
                                 autojunk=False)
    out = list(R)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for i, j in zip(range(i1, i2), range(j1, j2)):
            src, dst = H[j], R[i]
            lead = ""
            k = 0
            while k < len(src) and src[k] in LEAD_PUNCT:
                lead += src[k]
                k += 1
            trail = ""
            k = len(src)
            while k > 0 and src[k - 1] in TRAIL_PUNCT:
                trail = src[k - 1] + trail
                k -= 1
            core = src[len(lead):len(src) - len(trail)]
            if core[:1].isupper():
                dst = dst[:1].upper() + dst[1:]
            out[i] = lead + dst + trail
    return " ".join(out)


# ---------- LLM route ----------
PROMPT = """Είσαι επιμελητής ελληνικών μεταγραφών. Για κάθε στοιχείο σου δίνω:
- "words": η ΣΩΣΤΗ ακολουθία λέξεων (τονισμένη, πεζά, χωρίς στίξη). Είναι το κείμενο
  αναφοράς ενός σώματος κοινοβουλευτικού λόγου.
- "asr": η έξοδος ενός ASR για τον ΙΔΙΟ ήχο. Έχει στίξη και κεφαλαία, αλλά κάνει
  λάθη σε λέξεις.

Δουλειά σου: επίστρεψε το "words" με στίξη και κεφαλαία, χρησιμοποιώντας το "asr"
ως οδηγό.

ΑΠΑΡΑΒΑΤΟΙ ΚΑΝΟΝΕΣ:
1. ΜΗΝ αλλάξεις, προσθέσεις ή αφαιρέσεις καμία λέξη του "words". Ίδιες λέξεις, ίδια
   σειρά, ίδιοι τόνοι. Αλλάζεις ΜΟΝΟ στίξη και πεζά/κεφαλαία.
2. Τα περισσότερα αποσπάσματα είναι ΚΟΜΜΕΝΑ στη μέση πρότασης. Αν το απόσπασμα δεν
   τελειώνει σε ολοκληρωμένη πρόταση, ΜΗΝ βάλεις τελεία ή άλλο τελικό σημείο —
   άφησέ το ανοιχτό. Ομοίως, αν αρχίζει στη μέση πρότασης, μην κεφαλαιοποιήσεις την
   πρώτη λέξη.
3. Όταν μια πρόταση ΤΕΛΕΙΩΝΕΙ ΜΕΣΑ στο απόσπασμα και αρχίζει καινούργια, βάλε
   ΤΕΛΕΙΑ, όχι κόμμα, και κεφαλαίο στην επόμενη λέξη. Κόμμα μόνο ανάμεσα σε προτάσεις
   της ίδιας περιόδου. (Ο κανόνας 2 αφορά μόνο το ΤΕΛΟΣ του αποσπάσματος.)
4. Κεφαλαία μόνο σε κύρια ονόματα, θεσμούς και αρχή πρότασης.
5. Μην προσθέσεις εισαγωγικά ή παύλες που δεν υπαγορεύονται από τη δομή.

Επίστρεψε ΜΟΝΟ ένα JSON array, ένα αντικείμενο ανά στοιχείο, με τα πεδία:
{"id": <το id>, "text": "<το αποτέλεσμα>", "complete": true|false}
όπου "complete" = αν το απόσπασμα είναι ολοκληρωμένη πρόταση.

ΣΤΟΙΧΕΙΑ:
"""


def call_llm(batch: list[dict], timeout_wait: int = 900) -> list[dict]:
    payload = json.dumps([{"id": b["row_id"], "words": b["ref"], "asr": b["soniox_text"]}
                          for b in batch], ensure_ascii=False, indent=None)
    p = subprocess.run(
        [sys.executable, CLIENT, "enqueue", "exec",
         "-c", f"model={LLM_MODEL}", "-c", "model_reasoning_effort=low",
         PROMPT + payload],
        capture_output=True, text=True, timeout=120)
    try:
        job = json.loads(p.stdout)["job_id"]
    except Exception:
        raise RuntimeError(f"enqueue failed: {p.stdout[-300:]} {p.stderr[-300:]}")
    w = subprocess.run([sys.executable, CLIENT, "wait", job, str(timeout_wait)],
                       capture_output=True, text=True, timeout=timeout_wait + 120)
    res = json.loads(w.stdout)
    if res.get("status") != "completed":
        raise RuntimeError(f"job {job}: {res.get('status')}")
    return parse_json_array(res.get("output") or "")


def parse_json_array(text: str) -> list[dict]:
    m = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    blob = m[-1] if m else None
    if blob is None:
        i, j = text.find("["), text.rfind("]")
        if i < 0 or j < i:
            raise RuntimeError("no JSON array in model output")
        blob = text[i:j + 1]
    return json.loads(blob)


def same_words(a: str, b: str) -> bool:
    return greek_normalize(a).split() == greek_normalize(b).split()


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(WORK / "filtered.jsonl"))
    ap.add_argument("--out", default=str(WORK / "punctuated.jsonl"))
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--llm-workers", type=int, default=3,
                    help="concurrent bridge jobs (the worker runs 3 at a time)")
    ap.add_argument("--min-align", type=float, default=0.95)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.jsonl).read_text(encoding="utf-8").splitlines()]
    kept = [r for r in rows if r.get("asr_ok") and r.get("align", 0) >= args.min_align]
    print(f"[punct] {len(kept)} kept rows (align >= {args.min_align})")

    for r in kept:
        r["text_transfer"] = transfer(r["ref"], r["soniox_text"])

    n_llm_ok = n_llm_bad = 0
    if args.llm:
        # Accepted rows are cached the moment their batch lands, so a crash or a
        # Ctrl-C costs one batch, not the whole run.
        cache: dict[str, dict] = {}
        if LLM_CACHE.exists():
            for line in LLM_CACHE.read_text(encoding="utf-8").splitlines():
                try:
                    o = json.loads(line)
                    cache[o["row_id"]] = o
                except Exception:
                    pass
        for r in kept:
            o = cache.get(r["row_id"])
            if o:
                r["text_llm"], r["complete"] = o["text"], o["complete"]
                n_llm_ok += 1

        todo = [r for r in kept if not r.get("text_llm")]
        batches = [todo[s:s + args.batch] for s in range(0, len(todo), args.batch)]
        print(f"[punct] {len(cache)} cached, {len(todo)} rows in {len(batches)} "
              f"batches -> {LLM_MODEL} x{args.llm_workers}", flush=True)

        lock = threading.Lock()
        done = 0
        t0 = time.time()
        cf = LLM_CACHE.open("a", encoding="utf-8")
        with ThreadPoolExecutor(max_workers=args.llm_workers) as ex:
            futs = {ex.submit(call_llm, b): b for b in batches}
            for fut in as_completed(futs):
                batch = futs[fut]
                try:
                    got = {str(o["id"]): o for o in fut.result()}
                except Exception as e:
                    with lock:
                        done += 1
                        print(f"  batch failed: {e}; those rows keep transfer()",
                              flush=True)
                    continue
                with lock:
                    done += 1
                    for r in batch:
                        o = got.get(r["row_id"])
                        if not o or not isinstance(o.get("text"), str):
                            continue
                        if same_words(o["text"], r["ref"]):
                            r["text_llm"] = o["text"]
                            r["complete"] = bool(o.get("complete"))
                            n_llm_ok += 1
                            cf.write(json.dumps(
                                {"row_id": r["row_id"], "text": r["text_llm"],
                                 "complete": r["complete"]}, ensure_ascii=False) + "\n")
                        else:
                            r["llm_rejected"] = o["text"]
                            n_llm_bad += 1
                    cf.flush()
                    el = time.time() - t0
                    print(f"[punct] batch {done}/{len(batches)} {el/60:.1f}m "
                          f"ok={n_llm_ok} guard_rejected={n_llm_bad}", flush=True)
        cf.close()

    with Path(args.out).open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({
                "row_id": r["row_id"], "align": r["align"], "dur": r["dur"],
                "ref": r["ref"], "soniox_text": r["soniox_text"],
                "text_transfer": r["text_transfer"],
                "text_llm": r.get("text_llm"),
                "llm_rejected": r.get("llm_rejected"),
                "complete": r.get("complete"),
                "final": r.get("text_llm") or r["text_transfer"],
                "mp3": r.get("mp3"),
            }, ensure_ascii=False) + "\n")

    n_punct = sum(1 for r in kept if re.search(r"[.,;·!?]", r["text_transfer"]))
    print(f"[punct] transfer() put a mark on {n_punct}/{len(kept)} rows "
          f"({n_punct / len(kept):.0%})")
    if args.llm:
        print(f"[punct] llm accepted {n_llm_ok}, word-guard rejected {n_llm_bad}")
        both = [r for r in kept if r.get("text_llm")]
        agree = sum(1 for r in both if r["text_llm"] == r["text_transfer"])
        if both:
            print(f"[punct] llm == transfer on {agree}/{len(both)} "
                  f"({agree / len(both):.0%})")
            comp = sum(1 for r in both if r.get("complete"))
            print(f"[punct] llm calls {comp}/{len(both)} "
                  f"({comp / len(both):.0%}) complete sentences")
    print(f"[punct] -> {args.out}")


if __name__ == "__main__":
    main()
