#!/usr/bin/env python3
"""F1 — an LLM arbiter over 2-of-3 token majorities. EXPLORATORY ONLY.

Frozen preregistration: `docs/specs/2026-08-17-llm-f1-arbiter-prereg.md`.
Governing plan: `docs/specs/2026-08-17-llm-composer-draft.md` revision 2, §0 binding.

WHAT THIS IS. Where the per-column vote of `exp-2026-08-16-composition-over-selection`
resolved a column by a 2-of-3 token majority (`exact_2_of_3`), ask an LLM whether the
lone dissenting token should win instead. The model sees the sentence with the decision
position BLANKED, two candidate tokens under neutral labels Α/Β, and an explicit
abstain option. It never sees provider names, system counts, or which token is the
majority.

WHAT THIS IS NOT. `exp-2026-08-17-confirmation-audit` established that F1's eligibility
class, its abstention success condition and the elimination of families F2/F3 are each
traceable to numbers computed with the autoresearch confirmation labels in the
denominator. So:

  * NO confirmation batch is frozen and NO confirmation is spent. The budget stays 5
    of 5. This module does not import or call `autoresearch`.
  * Every interval it prints is DESCRIPTIVE. Never confirmatory, never gate-valid,
    never multiplicity-controlled. F1 may not be described as having passed the ship
    gate whatever it measures.
  * The deliverable is the prospective-design and power planning output, not the WER.

ELIGIBILITY IS REFERENCE-BLIND, and that is the single easiest way to ruin this
experiment. `eligible_columns()` takes the column list ONLY — the reference is not in
its signature, and `test_llm_arbiter.py` asserts the eligible set is invariant when
the reference is replaced with garbage. Revision 1 of the plan kept saying "the 1,245
wrong majorities"; that set is knowable only from the reference.

`msa.py` IS NOT TOUCHED. `fusion_lab._cache_path()` keys the 18 MB alignment cache on
its sha256; the run asserts the cache filename is unchanged at the end.

Verbatim council speech (contexts, candidate tokens, raw model answers) goes to
$SC/llm_arbiter/ and NEVER into git. `results_llm_arbiter.json` carries counts and
class labels only.

    SC=~/.cache/oc-public .venv-eval/bin/python -m eval.controlled_eval.exp_llm_arbiter
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval.column_classes import (                     # noqa: E402
    column_class, split_merge_columns,
)
from eval.controlled_eval.fusion_lab import (                         # noqa: E402
    Idea, Window, _cache_path, evaluate, load_substrate, log, sc,
)

CLIENT = str(Path.home() / "codex-bridge/codex_client.py")

# ------------------------------------------------------------------ frozen knobs
PROMPT_VERSION = "f1b-arbiter-2026-08-17a"
LLM_MODEL = "gpt-5.6-luna"
LLM_EFFORT = "high"
SEED = 17
CTX = 20                      # tokens of masked context on each side
TERM_BUDGET = 24              # max terms shown per question
LABELS = ("Α", "Β")           # neutral, inherited from exp_fusion.py:128
ABSTAIN = "ΑΠΟΧΗ"
N_PASSES = 2                  # both candidate orders, every question
ALIGN_CACHE_EXPECTED = "align_65b1c4d64618a429.json"
EXCLUDED_BUCKETS = ("numeric", "function_word")

OUT = Path(__file__).with_name("results_llm_arbiter.json")

OUTCOMES = ("override", "confirm", "abstain_explicit", "order_disagree", "invalid")


def cache_dir() -> Path:
    d = sc() / "llm_arbiter"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ lexical buckets
# These are the frozen text heuristics used by the majority taxonomy. They live here
# because F1's question eligibility and its descriptive analysis must share one
# implementation; the runner only imports the public helpers below.
FUNCTION_WORDS = set("""
ο η το οι τα του της των τον την τους τις
στο στη στην στον στους στις στα στου στων
ενας μια μιας μιαν ενα εναν ενος
και κι να θα ας μα αλλα ομως δε δεν μη μην που πως οτι αν εαν οταν
γιατι διοτι επειδη ωστε αφου ενω καθως ειτε ουτε λοιπον μεν
ναι οχι μαλιστα
σε με απο για προς κατα μετα παρα περι υπερ υπο δια ως εως μεχρι χωρις
μεσα εξω πανω κατω πριν μπροστα πισω διπλα γυρω αντι
μου σου μας σας
εγω εσυ εμεις εσεις αυτος αυτη αυτο αυτοι αυτες αυτα αυτον αυτην αυτου αυτης αυτων
εκεινος εκεινη εκεινο εκεινοι εκεινα
ποιος ποια ποιο ποιοι ποιες οποιος οποια οποιο οποιοι οποιες
καθε ολα ολο ολη ολος ολοι ολες πολυ πιο ηδη ακομα ακομη μονο επισης
τωρα τοτε εδω εκει ετσι οπως οπου κατι τιποτα καποιος καποια καποιο
""".split())

NUMBER_WORDS = set("""
μηδεν ενα ενας μια δυο τρια τρεις τριων τεσσερα τεσσερις πεντε εξι επτα εφτα
οκτω οχτω εννεα εννια δεκα ενδεκα εντεκα δωδεκα δεκατρια δεκατεσσερα δεκαπεντε
δεκαεξι δεκαεπτα δεκαοκτω δεκαεννεα εικοσι τριαντα σαραντα πενηντα εξηντα
εβδομηντα ογδοντα ενενηντα εκατο εκατον διακοσια διακοσιες τριακοσια τριακοσιες
τετρακοσια πεντακοσια εξακοσια εφτακοσια επτακοσια οκτακοσια εννιακοσια
χιλια χιλιες χιλιαδες χιλιαδων εκατομμυριο εκατομμυρια δισεκατομμυρια
μισο μιση ημισυ τοις ποσοστο
""".split())

ORDINAL_STEMS = ("πρωτ", "δευτερ", "τριτ", "τεταρτ", "πεμπτ", "εκτ", "εβδομ",
                 "ογδο", "ενατ", "δεκατ", "εικοστ", "τριακοστ")
MONTHS = set("""ιανουαριου φεβρουαριου μαρτιου απριλιου μαιου ιουνιου ιουλιου
αυγουστου σεπτεμβριου οκτωβριου νοεμβριου δεκεμβριου ιανουαριο φεβρουαριο μαρτιο
απριλιο μαιο ιουνιο ιουλιο αυγουστο σεπτεμβριο οκτωβριο νοεμβριο δεκεμβριο""".split())


def is_numeric(tok: str) -> bool:
    if any(ch.isdigit() for ch in tok):
        return True
    if tok in NUMBER_WORDS or tok in MONTHS:
        return True
    return any(tok.startswith(s) and len(tok) - len(s) <= 3 for s in ORDINAL_STEMS)


def surface_suffix_neighbor(a: str, b: str) -> bool:
    """String shape, not linguistic truth: shared prefix and short tails."""
    if a == b:
        return False
    p = 0
    for x, y in zip(a, b):
        if x != y:
            break
        p += 1
    if p < 4:
        return False
    return (len(a) - p) <= 4 and (len(b) - p) <= 4


def bucket_of(maj: str, mino: str, meeting_terms: set[str]) -> str:
    """Descriptive linguistic bucket, using the frozen precedence order."""
    if maj in meeting_terms or mino in meeting_terms:
        return "name_entity"
    if is_numeric(maj) or is_numeric(mino):
        return "numeric"
    if surface_suffix_neighbor(maj, mino):
        return "morphology"
    if maj in FUNCTION_WORDS and mino in FUNCTION_WORDS:
        return "function_word"
    return "other_content"


# ------------------------------------------------------------------ eligibility
def eligible_columns(cols) -> list[int]:
    """Indices of the columns F1 may ask about. REFERENCE-BLIND BY SIGNATURE.

    The only input is the aligned column list: presence and string identity. No
    reference, no oracle, no taxonomy label, no outcome. All 6,645 `exact_2_of_3`
    columns over the 247 windows are eligible — including the 133 that are also
    `split_merge`, which the plan keeps and which are reported as a stratum.
    """
    return [i for i, col in enumerate(cols) if column_class(col) == "exact_2_of_3"]


def majority_minority(col) -> tuple[str, str]:
    """(token held by two systems, token held by one). Only valid for exact_2_of_3."""
    c = Counter(e for e in col if e is not None)
    (maj, nmaj), (mino, nmin) = c.most_common(2)
    assert nmaj == 2 and nmin == 1, f"not an exact_2_of_3 column: {col}"
    return maj, mino


def w_positions(decisions) -> dict[int, int]:
    """Column index -> index into the W token stream, for columns W emits."""
    out, k = {}, 0
    for d in decisions:
        if d["token"] is not None:
            out[d["col"]] = k
            k += 1
    return out


# ------------------------------------------------------------------ questions
def _flip(qid: str) -> bool:
    """Frozen per-question decision: does the MINORITY token take label Α in pass 1?

    A stable hash, so it cannot be tuned after seeing results, and so that position
    and majority status are decorrelated across the corpus instead of the majority
    always sitting first.
    """
    return int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 2 == 1


def masked_context(w_tokens: list[str], pos: int, ctx: int = CTX) -> str:
    lo, hi = max(0, pos - ctx), min(len(w_tokens), pos + ctx + 1)
    body = list(w_tokens[lo:hi])
    body[pos - lo] = "_____"
    return " ".join(body)


def _meeting_terms(ctx) -> set[str]:
    if ctx is None:
        return set()
    return {str(info["term"].get("canonical") or tid)
            for tid, info in ctx.present.items()}


def _surface_key(value: str) -> str:
    folded = unicodedata.normalize("NFD", str(value).casefold())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def _edit_distance(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for j, y in enumerate(b, 1):
        current = [j]
        for i, x in enumerate(a, 1):
            current.append(min(current[-1] + 1, previous[i] + 1,
                               previous[i - 1] + (x != y)))
        previous = current
    return previous[-1]


def _term_in_context(term: str, context: str) -> bool:
    term_key = _surface_key(term).strip()
    context_key = " ".join(_surface_key(context).split())
    if not term_key or not context_key:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(term_key) + r"(?!\w)",
                          context_key))


def _term_close_to_candidate(term: str, candidate: str) -> bool:
    term_key, candidate_key = _surface_key(term), _surface_key(candidate)
    if len(term_key) >= 4 and len(candidate_key) >= 4:
        if term_key[:4] == candidate_key[:4]:
            return True
    return _edit_distance(term_key, candidate_key) <= 2


def terms_for(ctxs, city, meeting, context="", candidates=(), budget=TERM_BUDGET) -> str:
    """Return the meeting's closed term list ranked for this question.

    Context matches outrank candidate-near terms, which outrank the remainder.
    Every tie is sorted by an accent-insensitive alphabetic key and then the original
    spelling, so repeated calls are deterministic.
    """
    # Keep the old fourth positional form harmless for callers that only requested a
    # budget; all F1 question construction uses the per-question form below.
    if isinstance(context, int) and candidates == ():
        budget, context = context, ""
    ctx = ctxs.get((city, meeting))
    if ctx is None:
        return ""
    names = sorted(_meeting_terms(ctx))
    candidate_list = tuple(str(c) for c in candidates)

    def rank(name: str) -> int:
        if _term_in_context(name, context):
            return 0
        if any(_term_close_to_candidate(name, candidate)
               for candidate in candidate_list):
            return 1
        return 2

    names.sort(key=lambda name: (rank(name), _surface_key(name), name))
    return ", ".join(names[:budget])


def build_questions(sub, ctxs) -> list[dict]:
    """One record per eligible column. No reference is read anywhere in here."""
    qs = []
    excluded = Counter()
    for w in sub.windows:
        pos = w_positions(w.decisions)
        sm = split_merge_columns(w.cols)
        meeting_terms = _meeting_terms(ctxs.get((w.city, w.meeting)))
        for i in eligible_columns(w.cols):
            maj, mino = majority_minority(w.cols[i])
            assert w.decisions[i]["token"] == maj, \
                f"{w.item_id}#{i}: W emitted {w.decisions[i]['token']!r}, not {maj!r}"
            bucket = bucket_of(maj, mino, meeting_terms)
            if bucket in EXCLUDED_BUCKETS:
                excluded[bucket] += 1
                continue
            qid = f"{w.item_id}#{i}"
            context = masked_context(w.w_tokens, pos[i])
            qs.append({
                "id": qid, "item_id": w.item_id, "col": i,
                "city": w.city, "meeting": w.meeting,
                "pos": pos[i], "majority": maj, "minority": mino,
                "context": context,
                "terms": terms_for(ctxs, w.city, w.meeting, context,
                                   candidates=(maj, mino)),
                "split_merge": i in sm,
                "flip": _flip(qid),
            })
    for bucket in EXCLUDED_BUCKETS:
        log(f"excluded bucket {bucket}: {excluded[bucket]} questions removed")
    return qs


# ------------------------------------------------------------------ the prompt
PREAMBLE = f"""Είσαι επιμελητής μεταγραφών ελληνικών δημοτικών συμβουλίων.

Τρία ανεξάρτητα συστήματα αυτόματης αναγνώρισης ομιλίας μετέγραψαν τον ΙΔΙΟ ήχο. Οι
μεταγραφές ευθυγραμμίστηκαν λέξη-προς-λέξη. Σου δίνω θέσεις όπου δεν συμφώνησαν. Η
επίμαχη θέση εμφανίζεται ως _____ μέσα στο γύρω κείμενο.

Για κάθε θέση σου δίνω ΔΥΟ υποψήφιες λέξεις με ουδέτερες ετικέτες "{LABELS[0]}" και
"{LABELS[1]}". Δεν σου λέω ποιο σύστημα πρότεινε ποια, ούτε πόσα συστήματα πρότειναν
την καθεμιά. Διάλεξε την ετικέτα της λέξης που ανήκει στη θέση _____.

Μαζί με την ετικέτα "{LABELS[0]}" ή "{LABELS[1]}" δίνεις και το πεδίο "conf": έναν
ακέραιο αριθμό από 0 έως 100 που δηλώνει πόσο σίγουρη είσαι ότι η επιλεγμένη λέξη
είναι πράγματι αυτή που ακούστηκε. Το "conf" είναι ΥΠΟΧΡΕΩΤΙΚΟ όταν διαλέγεις
"{LABELS[0]}" ή "{LABELS[1]}". Όταν απαντάς "{ABSTAIN}", μην γράφεις "conf".

Σε αυτές τις θέσεις, όταν δύο από τα τρία συστήματα συμφωνούν, η κοινή λέξη είναι
σωστή περίπου στο 80% των περιπτώσεων. Γι' αυτό δίνεις υψηλό "conf" μόνο όταν έχεις
πραγματικό λόγο να πιστεύεις ότι η μοναχική υποψήφια είναι η σωστή — όχι απλώς
επειδή είναι λίγο πιο εύλογη.

ΑΠΑΡΑΒΑΤΟΙ ΚΑΝΟΝΕΣ:
- Επιστρέφεις ΕΤΙΚΕΤΑ: "{LABELS[0]}", "{LABELS[1]}" ή "{ABSTAIN}". ΠΟΤΕ δεν γράφεις
  δική σου λέξη και ΠΟΤΕ δεν προτείνεις τρίτη εκδοχή.
- Το "{ABSTAIN}" είναι πλήρης, κανονική, σωστή απάντηση — όχι αποτυχία. Απάντησε
  "{ABSTAIN}" όταν το συγκείμενο δεν αρκεί για να ξεχωρίσεις τις δύο, όταν και οι δύο
  είναι εξίσου εύλογες, ή όταν υποψιάζεσαι ότι η πραγματικά ειπωμένη λέξη δεν είναι
  καμία από τις δύο.
- ΜΗΝ προτιμάς τη συντομότερη ή την πιο "ομαλή" ή πιο γραμματικά κομψή εκδοχή. Ο ήχος
  είναι αυθόρμητος προφορικός λόγος: περιέχει επαναλήψεις, ατελείς προτάσεις,
  ιδιωματισμούς και δύσκολα κύρια ονόματα. Η ρέουσα εκδοχή είναι συχνά η λάθος.
- Κάθε θέση κρίνεται εντελώς μόνη της. Το γύρω κείμενο είναι δεδομένο· δεν το αλλάζεις.
- Το πεδίο "οροι" (όταν υπάρχει) είναι κλειστή λίστα επωνύμων παρόντων, τοπωνυμίων και
  ακρωνυμίων αυτής της συνεδρίασης. Αν μια υποψήφια είναι όρος της λίστας ή προφανής
  παραμόρφωσή του, αυτό μετράει υπέρ του σωστού τύπου.

Επίστρεψε ΜΟΝΟ ένα JSON array, με ΑΚΡΙΒΩΣ μία εγγραφή για κάθε θέση που σου δόθηκε:
[{{"id": "<το id της θέσης>", "pick": "{LABELS[0]}", "conf": 0}}, ...]
Για "{LABELS[1]}" ισχύει το ίδιο. Για "{ABSTAIN}" το "conf" παραλείπεται.

ΘΕΣΕΙΣ:
"""


#: Pass 1 and pass 3 (the A/A replicate) use the SAME candidate mapping; pass 2 is the
#: exact swap. Pass 3 therefore measures stochastic disagreement, not order sensitivity.
def _minority_first(q: dict, pass_no: int) -> bool:
    return q["flip"] if pass_no in (1, 3) else (not q["flip"])


def _ab(q: dict, pass_no: int) -> tuple[str, str]:
    return ((q["minority"], q["majority"]) if _minority_first(q, pass_no)
            else (q["majority"], q["minority"]))


def render(q: dict, pass_no: int) -> dict:
    """The per-question payload as the model sees it, for one candidate order."""
    a, b = _ab(q, pass_no)
    r = {"id": q["id"], "κειμενο": q["context"], LABELS[0]: a, LABELS[1]: b}
    if q["terms"]:
        r["οροι"] = q["terms"]
    return r


def token_for(q: dict, pass_no: int, label: str) -> str | None:
    """Which TOKEN a label names, for this question in this pass."""
    a, b = _ab(q, pass_no)
    return {LABELS[0]: a, LABELS[1]: b}.get(label)


def plan_batches(questions: list[dict], batch_size: int) -> list[list[dict]]:
    """Batches computed once over the FULL question list, so they are identical on a
    resume and identical between the two passes.

    Amendment 1: revision 0 batched the outstanding set, which made a resumed run
    produce different wire requests (and therefore different cache keys) than the run
    it resumed, and made pass 1 and pass 2 differ in batch context as well as in
    candidate order.
    """
    order = sorted(questions, key=lambda q: hashlib.sha256(
        f"{PROMPT_VERSION}|{SEED}|{q['id']}".encode()).hexdigest())
    return [order[k:k + batch_size] for k in range(0, len(order), batch_size)]


def batch_wire(batch: list[dict], pass_no: int) -> str:
    """The exact text sent to the model for one batch."""
    return PREAMBLE + json.dumps([render(q, pass_no) for q in batch],
                                 ensure_ascii=False)


def cache_key(q: dict, pass_no: int, batch: list[dict]) -> str:
    """sha256 of the COMPLETE wire request, plus the question's identity.

    Includes batch membership and order, so answers obtained under a rejected pilot
    batch size can never be mixed into the production run.
    """
    blob = "|".join([
        q["id"], PROMPT_VERSION, str(SEED), LLM_MODEL, LLM_EFFORT, str(pass_no),
        str(len(batch)),
        hashlib.sha256(batch_wire(batch, pass_no).encode()).hexdigest()])
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


# ------------------------------------------------------------------ transport
def parse_json_array(text: str):
    """Inherited from exp_composition.py:200."""
    m = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    blob = m[-1] if m else None
    if blob is None:
        i, j = text.find("["), text.rfind("]")
        if i < 0 or j < 0:
            return []
        blob = text[i:j + 1]
    try:
        return json.loads(blob)
    except Exception:
        return []


#: The model occasionally answers with Latin lookalikes of the Greek labels. Mapping
#: them is a transcription of the SAME choice, not a repair of a wrong one; anything
#: outside this table is `invalid` and counted.
_LABEL_ALIAS = {"A": LABELS[0], "B": LABELS[1], "Α": LABELS[0], "Β": LABELS[1],
                ABSTAIN: ABSTAIN, "ΑΠΟΧΗ": ABSTAIN}


def norm_label(v) -> str | None:
    if not isinstance(v, str):
        return None
    return _LABEL_ALIAS.get(v.strip().upper())


def call_llm(wire: str, timeout_enqueue=180):
    """One bridge job. No --timeout is passed; `wait` gets no second argument."""
    p = subprocess.run(
        [sys.executable, CLIENT, "enqueue", "exec",
         "-c", f"model={LLM_MODEL}", "-c", f"model_reasoning_effort={LLM_EFFORT}",
         wire],
        capture_output=True, text=True, timeout=timeout_enqueue)
    try:
        job = json.loads(p.stdout)["job_id"]
    except Exception:
        raise RuntimeError(f"enqueue failed: {p.stdout[-300:]} {p.stderr[-300:]}")
    w = subprocess.run([sys.executable, CLIENT, "wait", job],
                       capture_output=True, text=True)
    try:
        res = json.loads(w.stdout)
    except Exception:
        raise RuntimeError(f"job {job}: unparseable wait output {w.stdout[-300:]}")
    if res.get("status") != "completed":
        raise RuntimeError(f"job {job}: {res.get('status')}")
    return job, parse_json_array(res.get("output") or "")


RETRY_LIMIT = 1                                  # frozen


def _valid_conf(value) -> bool:
    return type(value) is int and 0 <= value <= 100


def answers_of(got, batch: list[dict]) -> dict[str, dict | None]:
    """Model reply -> per-question ``pick``/``conf`` answer.

    An id appearing MORE THAN ONCE makes that question invalid for this pass: neither
    first-wins nor last-wins is defensible when the model contradicted itself. Ids not
    asked for are discarded. Anything outside {Α, Β, ΑΠΟΧΗ} after label normalisation
    is invalid. A non-abstain pick without an integer confidence in [0, 100] is also
    invalid; it is never converted to confidence zero.
    """
    if not isinstance(got, list):
        got = []
    asked = {q["id"] for q in batch}
    seen = Counter(g.get("id") for g in got if isinstance(g, dict))
    answers: dict[str, dict | None] = {}
    for g in got:
        if not isinstance(g, dict):
            continue
        qid = g.get("id")
        if qid not in asked or seen[qid] > 1:
            continue
        pick = norm_label(g.get("pick"))
        if pick is None:
            answers[qid] = None
        elif pick == ABSTAIN:
            answers[qid] = {"pick": ABSTAIN}
        elif _valid_conf(g.get("conf")):
            answers[qid] = {"pick": pick, "conf": g["conf"]}
        else:
            answers[qid] = None
    return {q["id"]: answers.get(q["id"]) for q in batch}


def run_pass(questions: list[dict], pass_no: int, batch_size: int,
             cache_file: Path, workers: int = 3) -> tuple[dict, dict]:
    """Ask one candidate order for every question. Returns (key -> answer, accounting).

    A TRANSPORT failure is not an answer. Caching it as a no-op makes the failure
    permanent and silently inflates the invalid count with questions the model was
    never asked (`exp_composition.py`, caught by CodeRabbit). Failed batches are left
    uncached and retried once.
    """
    cache: dict[str, dict] = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                # Old append-only records are harmless: the new prompt version makes
                # their keys unreachable, and missing confidence remains invalid.
                cache[r["k"]] = {"pick": r.get("pick", r.get("label")),
                                 "conf": r.get("conf")}

    batches = plan_batches(questions, batch_size)
    todo = [b for b in batches
            if any(cache_key(q, pass_no, b) not in cache for q in b)]
    log(f"pass {pass_no}: {len(batches) - len(todo)}/{len(batches)} batches cached, "
        f"{len(todo)} to ask (batch {batch_size})")

    acct = {"batches": len(batches), "batches_asked": len(todo), "batches_failed": 0,
            "questions_unanswered": 0, "jobs": [], "wall_s": 0.0,
            "retries": 0, "batch_size": batch_size, "pass": pass_no}
    if not todo:
        return cache, acct

    def run(batch):
        last = None
        for attempt in range(RETRY_LIMIT + 1):
            try:
                job, got = call_llm(batch_wire(batch, pass_no))
                return batch, job, got, attempt
            except Exception as e:                   # transport, not an answer
                last = e
        return batch, None, last, RETRY_LIMIT

    lock = threading.Lock()
    t0 = time.time()
    done = 0
    with open(cache_file, "a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        for batch, job, got, attempts in pool.map(run, todo):
            with lock:
                acct["retries"] += attempts
                if isinstance(got, Exception) or got is None:
                    log(f"  batch failed after retry, NOT cached: {str(got)[:120]}")
                    acct["batches_failed"] += 1
                    acct["questions_unanswered"] += len(batch)
                    done += len(batch)
                    continue
                acct["jobs"].append(job)
                answers = answers_of(got, batch)
                for q in batch:
                    k = cache_key(q, pass_no, batch)
                    answer = answers[q["id"]]
                    record = {
                        "k": k,
                        "pick": answer.get("pick") if answer else None,
                        "conf": answer.get("conf") if answer else None,
                    }
                    cache[k] = {"pick": record["pick"], "conf": record["conf"]}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                done += len(batch)
                if done % (batch_size * 10) < batch_size:
                    log(f"  pass {pass_no}: {done}/{sum(len(b) for b in todo)} "
                        f"({time.time() - t0:.0f}s)")
    acct["wall_s"] = time.time() - t0
    return cache, acct


# ------------------------------------------------------------------ resolution
def _cached_answer(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    pick = norm_label(value.get("pick"))
    if pick == ABSTAIN:
        return {"pick": ABSTAIN, "conf": None}
    if pick in LABELS and _valid_conf(value.get("conf")):
        return {"pick": pick, "conf": value["conf"]}
    return None


def resolve(questions, caches, batch_size: int, passes=(1, 2), *,
            conf_threshold: int = 0) -> dict[str, dict]:
    """Per question: the frozen five-way outcome partition, in precedence order.

      1 invalid            either pass missing / unparseable / illegal label
      2 abstain_explicit   either pass returned ΑΠΟΧΗ
      3 override           both passes resolve to the MINORITY token and both
                           confidence scores meet ``conf_threshold``
      4 confirm            both passes resolve to the MAJORITY token
      5 order_disagree     the two passes resolve to different tokens

    Exhaustive and mutually exclusive over the eligible set. Only `override` changes
    the transcript; the other four leave W's token in place AND ARE COUNTED. An
    `invalid` or an `order_disagree` is NEVER evidence that the model knowingly
    abstained, and must not be reported as such. With ``conf_threshold=0``, every
    valid non-abstain confidence passes the threshold, preserving the old partition.
    """
    by_q = {q["id"]: b for b in plan_batches(questions, batch_size) for q in b}
    out = {}
    for q in questions:
        b = by_q[q["id"]]
        answers = [_cached_answer(caches[p].get(cache_key(q, p, b)))
                   for p in passes]
        picks = [a["pick"] if a else None for a in answers]
        confs = [a["conf"] if a else None for a in answers]
        base = {"labels": picks, "conf": confs}
        if any(a is None for a in answers):
            out[q["id"]] = {"outcome": "invalid", **base}
            continue
        if any(a["pick"] == ABSTAIN for a in answers):
            out[q["id"]] = {"outcome": "abstain_explicit", **base}
            continue
        toks = [token_for(q, p, a["pick"]) for p, a in zip(passes, answers)]
        if toks[0] != toks[1]:
            out[q["id"]] = {"outcome": "order_disagree", **base}
            continue
        out[q["id"]] = {
            "outcome": ("override"
                        if toks[0] == q["minority"]
                        and all(c >= conf_threshold for c in confs)
                        else "confirm"),
            **base, "token": toks[0]}
    return out


class F1Arbiter(Idea):
    """Replace W's token with the minority token exactly where F1 overrode.

    No fitted parameter: leave-one-city-out is vacuous by construction and
    `fusion_lab.evaluate` says so in `fold_note`.
    """
    name = "F1_llm_arbiter"
    fitted = False

    def __init__(self, overrides: dict[str, dict[int, str]]):
        self.overrides = overrides
        #: Application invariant, Amendment 1: every override must alter its intended
        #: W index exactly once. A mapping failure or a collision hard-fails.
        self.applied = 0
        self.mapping_failures = 0
        self.collisions = 0

    def apply(self, w: Window, params) -> list[str]:
        ov = self.overrides.get(w.item_id)
        toks = list(w.w_tokens)
        if not ov:
            return toks
        pos = w_positions(w.decisions)
        touched: set[int] = set()
        for col, tok in ov.items():
            if col not in pos:
                self.mapping_failures += 1
                raise AssertionError(f"{w.item_id}#{col}: override maps to no W index")
            i = pos[col]
            if i in touched:
                self.collisions += 1
                raise AssertionError(f"{w.item_id}#{col}: two overrides on W index {i}")
            touched.add(i)
            toks[i] = tok
            self.applied += 1
        return toks


def pilot_sample(questions: list[dict], n: int = 120) -> list[dict]:
    """The 120 questions with the smallest sha256("f1-pilot|" + id). Reference-blind
    and deterministic: no reference, no outcome and no WER enters the draw."""
    return sorted(questions, key=lambda q: hashlib.sha256(
        ("f1-pilot|" + q["id"]).encode()).hexdigest())[:n]


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True)
                          .encode()).hexdigest()


if __name__ == "__main__":     # pragma: no cover - the runner lives in run_llm_arbiter
    log("this module is a library; run eval/controlled_eval/run_llm_arbiter.py")
