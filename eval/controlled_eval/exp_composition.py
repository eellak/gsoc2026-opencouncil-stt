#!/usr/bin/env python3
"""Composition over selection: word-level fusion of three ASR hypotheses (#22).

`exp-2026-08-16-roster-grounded-selection` closed whole-window selection: an LLM
asked "which transcript is best" cost a full WER point by preferring the shorter,
fluent text. And selection has a ceiling anyway - the trio's whole-window oracle on
this substrate is 0.1064 against the vote's 0.1201. This experiment stops choosing
and starts composing, so the output is a text none of the three systems produced and
the whole-window ceiling stops applying.

Arms, declared on wayfinder #22 before any number was computed:

  V       the existing whole-window consensus vote (baseline, recomputed here)
  W       exact 3-way MSA + hierarchical per-column vote. NO LLM. The critical arm.
  W+len   W plus the frozen length fail-safe: if W is shorter than V in tokens,
          emit V
  W+L     W plus an LLM arbitrating ONLY the columns where the vote had to
          tie-break, mechanically restricted to an index into that column's own
          candidates - it can never write a token no system proposed and it never
          selects a whole hypothesis
  W+L+D   plus speaker-grounded restoration: a run of columns W dropped, where only
          one system heard anything, is restored if pyannote's NON-exclusive
          diarization says the gap carries simultaneous speakers or a speaker who is
          not the one talking on either side

Also reported, and the reason the W arm is the critical one: the ALIGNMENT-
CONDITIONAL COLUMN ORACLE - the best text obtainable by choosing one entry per
column. That is a different ceiling from the 0.1064 whole-window oracle and it is
the real bound on the information three transcripts carry.

The 6 sealed temporal-holdout windows of `eval-freeze-2026-08` are filtered out
exactly as `exp_fusion_deletions.py` does. Speaker-attribution ACCURACY is not
measured and no claim about it may be made: there is no ground truth until #21.

Writes results_composition.json (aggregates only, never transcript text).

Env: SC (cache dir), N_BOOT (10000), WORKERS (cpu count), LLM (1 to call the model)
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import permutations
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import bench_data as B                         # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import rates, sdi         # noqa: E402
from eval.controlled_eval.exp_parakeet_voter import token_times          # noqa: E402
from eval.controlled_eval.exp_speaker_fusion import active_intervals     # noqa: E402
from eval.controlled_eval.msa import (                                   # noqa: E402
    align3, columns_cost, compose, oracle_columns, progressive_align,
)
from eval.controlled_eval.scoring import (cluster_bootstrap, head2head,  # noqa: E402
                                          wtoks)

RUN_ID = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
TRIO = ["scribe-v2-clean", "soniox", "oc-runpod-fixed-2026-08-10"]
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
OUT = Path(__file__).with_name("results_composition.json")

BAND_FLOOR = 40          # widened per window to cover the largest length difference
MIN_GAP_SEC = 0.30       # a restored run must have at least this much room in time
CLIENT = "/home/harold/codex-bridge/codex_client.py"
LLM_MODEL = "gpt-5.6-luna"
LLM_CTX = 8              # tokens of context shown either side of an arbitrated column
LLM_CACHE = Path.home() / ".cache/oc-public/composition/llm_cache.jsonl"


def log(m):
    print(m, flush=True)


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


# ------------------------------------------------------------------- per window
def align_window(payload):
    """Everything that is pure alignment, so it can run in a worker process."""
    wid, toks, pivot, ref = payload
    a, b, c = toks
    band = max(BAND_FLOOR, max(len(a), len(b), len(c)) - min(len(a), len(b), len(c)) + 20)
    cols = align3(a, b, c, band=band)
    w_tokens, decisions = compose(cols, pivot=pivot)
    orc = oracle_columns(cols, ref)
    prog = {}
    for order in permutations((0, 1, 2)):
        pcols = progressive_align([a, b, c], order)
        pw, _ = compose(pcols, pivot=pivot)
        prog["".join(map(str, order))] = {
            "cost": columns_cost(pcols),
            "n_cols": len(pcols),
            "voted": pw,
            "oracle": oracle_columns(pcols, ref),
        }
    return wid, {
        "cols": cols, "band": band, "cost": columns_cost(cols), "n_cols": len(cols),
        "w_tokens": w_tokens, "decisions": decisions, "oracle": orc,
        "progressive": prog,
    }


# ------------------------------------------------------------------ time anchors
def pyannote_window(wid: str) -> dict | None:
    p = sc() / "whisper_turbo" / f"{wid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["output"]


def pk_stream(pk: dict):
    toks, times, ends = [], [], []
    for w in pk["wordLevelTranscription"]:
        tt = wtoks(w["text"])
        if not tt:
            continue
        # a pyannote "word" is one token in practice; if it splits, spread it
        span = (float(w["end"]) - float(w["start"])) / len(tt)
        for n, t in enumerate(tt):
            toks.append(t)
            times.append(float(w["start"]) + n * span)
            ends.append(float(w["start"]) + (n + 1) * span)
    end = max(ends) if ends else 0.0
    for s in pk["diarization"]:
        end = max(end, float(s["end"]))
    return toks, times, end


def anchor_indices(tokens: list[str], pk_tokens: list[str]) -> set[int]:
    """Which positions of `tokens` `token_times` pinned to a real pyannote word.

    Needed because the headline time-discrepancy figure is uninformative without
    it: two hypotheses that BOTH anchor a token get the identical pyannote
    timestamp by construction, so the median discrepancy is 0.0 for a reason that
    says nothing about interpolation quality. The interesting distribution is the
    one restricted to pairs where at least one side was interpolated.
    """
    from difflib import SequenceMatcher

    from eval.controlled_eval.exp_parakeet_voter import _code
    enc: dict = {}
    a, b = _code(tokens, enc), _code(pk_tokens, enc)
    out = set()
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            out.add(blk.a + k)
    return out


def speakers_at(intervals, t: float):
    for a, b, sp in intervals:
        if a <= t < b:
            return sp
    return frozenset()


def speakers_over(intervals, a: float, b: float):
    """(union of speaker labels in [a,b), max simultaneous count)."""
    u, mx = set(), 0
    for x, y, sp in intervals:
        if x < b and a < y:
            u |= set(sp)
            mx = max(mx, len(sp))
    return u, mx


# ------------------------------------------------------------------------ LLM arm
PROMPT = """Είσαι επιμελητής μεταγραφών ελληνικών δημοτικών συμβουλίων.

Τρία ανεξάρτητα συστήματα ASR μετέγραψαν τον ΙΔΙΟ ήχο. Οι τρεις μεταγραφές
ευθυγραμμίστηκαν λέξη-προς-λέξη και ψηφίστηκαν ανά θέση. Σου δίνω ΜΟΝΟ τις θέσεις
όπου δεν προέκυψε πλειοψηφία, με το γύρω κείμενο που έχει ήδη αποφασιστεί.

Για κάθε θέση διάλεξε ΕΝΑΝ από τους αριθμημένους υποψηφίους — αυτόν που ταιριάζει
στα ελληνικά και στο συγκείμενο. Σου δίνω και μια ΚΛΕΙΣΤΗ ΛΙΣΤΑ όρων (επώνυμα
παρόντων, τοπωνύμια, ακρωνύμια) αυτής της συνεδρίασης: αν ένας υποψήφιος είναι
όρος της λίστας ή προφανής παραμόρφωσή του, αυτό μετράει υπέρ του σωστού τύπου.

ΑΠΑΡΑΒΑΤΟΙ ΚΑΝΟΝΕΣ:
- ΔΕΝ γράφεις λέξη. Επιστρέφεις ΑΡΙΘΜΟ υποψηφίου. Οτιδήποτε άλλο απορρίπτεται.
- ΔΕΝ διαλέγεις μεταγραφή. Κάθε θέση κρίνεται μόνη της.
- Το γύρω κείμενο είναι δεδομένο· δεν το αλλάζεις.
- Σε αμφιβολία διάλεξε τον υποψήφιο 0.

Επίστρεψε ΜΟΝΟ ένα JSON array: [{"id": <το id>, "pick": <αριθμός>}, ...]

ΘΕΣΕΙΣ:
"""


def parse_json_array(text: str):
    import re
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


def call_llm(batch: list[dict]):
    payload = json.dumps(batch, ensure_ascii=False)
    p = subprocess.run(
        [sys.executable, CLIENT, "enqueue", "exec",
         "-c", f"model={LLM_MODEL}", "-c", "model_reasoning_effort=low",
         PROMPT + payload],
        capture_output=True, text=True, timeout=180)
    try:
        job = json.loads(p.stdout)["job_id"]
    except Exception:
        raise RuntimeError(f"enqueue failed: {p.stdout[-300:]} {p.stderr[-300:]}")
    w = subprocess.run([sys.executable, CLIENT, "wait", job],
                       capture_output=True, text=True, timeout=2400)
    res = json.loads(w.stdout)
    if res.get("status") != "completed":
        raise RuntimeError(f"job {job}: {res.get('status')}")
    return parse_json_array(res.get("output") or "")


def llm_pass(questions: list[dict], batch_size: int = 24):
    """One cached decision per arbitrated column. Returns qid -> pick index."""
    LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, int] = {}
    if LLM_CACHE.exists():
        for line in LLM_CACHE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["id"]] = r.get("pick")
    todo = [q for q in questions if q["id"] not in cache]
    log(f"LLM: {len(cache)} cached, {len(todo)} to call")
    chunks = [todo[k:k + batch_size] for k in range(0, len(todo), batch_size)]

    def run(chunk):
        try:
            return chunk, call_llm(chunk)
        except Exception as e:
            return chunk, e

    lock = threading.Lock()
    done = failed_q = 0
    with open(LLM_CACHE, "a") as f, ThreadPoolExecutor(max_workers=3) as pool:
        for chunk, got in pool.map(run, chunks):
            # A TRANSPORT failure is not an answer. Caching it as a no-op makes the
            # failure permanent and silently inflates the "invalid or missing" count
            # with questions the model was never asked. Leave it uncached so the next
            # run asks again.
            if isinstance(got, Exception):
                log(f"  batch failed, NOT cached: {got}")
                failed_q += len(chunk)
                done += len(chunk)
                continue
            by = {g.get("id"): g for g in got if isinstance(g, dict)}
            with lock:
                for q in chunk:
                    g = by.get(q["id"]) or {}
                    pick = g.get("pick")
                    cache[q["id"]] = pick
                    f.write(json.dumps({"id": q["id"], "pick": pick}) + "\n")
                f.flush()
                done += len(chunk)
                log(f"  {done}/{len(todo)}")
    if failed_q:
        log(f"  {failed_q} questions left unanswered by failed batches")
    return cache


# ----------------------------------------------------------------------- main
def main():
    report = B.load_report(RUN_ID)
    providers = B.provider_ids(report)
    items = B.common_items(report, providers)
    sealed = {w["window_id"] for w in json.loads(
        (ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())["holdout_windows"]}
    before = len(items)
    items = [it for it in items if it["item_id"] not in sealed]
    log(f"{RUN_ID}: {before} common -> {len(items)} after removing "
        f"{before - len(items)} sealed holdout windows")
    assert before - len(items) == 6, \
        f"expected 6 sealed holdout windows inside this run, removed {before - len(items)}"
    assert len(items) == 247, f"expected 247 windows, got {len(items)}"
    if os.environ.get("LIMIT"):          # smoke only; never used for a reported number
        items = items[:int(os.environ["LIMIT"])]
        log(f"LIMIT set: {len(items)} windows - SMOKE RUN, not reportable")
    clusters = [it["cluster"] for it in items]

    toks = {it["item_id"]: [wtoks(it["hyp"][p]) for p in TRIO] for it in items}
    refs = {it["item_id"]: wtoks(it["ref"]) for it in items}
    v_pick = {it["item_id"]: B.consensus_pick(it, TRIO) for it in items}
    pivot = {w: TRIO.index(p) for w, p in v_pick.items()}

    # -------- alignment + composition, parallel over windows --------
    workers = int(os.environ.get("WORKERS", str(min(8, os.cpu_count() or 4))))
    payloads = [(it["item_id"], toks[it["item_id"]], pivot[it["item_id"]],
                 refs[it["item_id"]]) for it in items]
    log(f"aligning {len(payloads)} windows on {workers} workers")
    A: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (wid, r) in enumerate(ex.map(align_window, payloads, chunksize=1), 1):
            A[wid] = r
            if n % 25 == 0:
                log(f"  {n}/{len(payloads)}")

    # -------- arms --------
    texts: dict[str, dict[str, str]] = {}
    texts["V"] = {it["item_id"]: it["hyp"][v_pick[it["item_id"]]] for it in items}
    texts["W"] = {w: " ".join(A[w]["w_tokens"]) for w in A}
    texts["W+len"] = {}
    len_guard_fired = []
    for it in items:
        w = it["item_id"]
        if len(A[w]["w_tokens"]) < len(toks[w][pivot[w]]):
            texts["W+len"][w] = texts["V"][w]
            len_guard_fired.append(w)
        else:
            texts["W+len"][w] = texts["W"][w]

    # -------- alignment quality, measured explicitly --------
    times: dict[str, list[list[float]]] = {}
    anchor_frac: dict[str, list[float]] = {p: [] for p in TRIO}
    w_times: dict[str, list[float]] = {}
    pk_missing = []
    audio_end: dict[str, float] = {}
    diar: dict[str, list] = {}
    anchors: dict[str, dict[int, set]] = {}
    for it in items:
        w = it["item_id"]
        pk = pyannote_window(w)
        if pk is None:
            pk_missing.append(w)
            continue
        pt, ptm, end = pk_stream(pk)
        audio_end[w] = end
        diar[w] = active_intervals(pk["diarization"])
        tt = []
        for n, p in enumerate(TRIO):
            t, na = token_times(toks[w][n], pt, ptm, end)
            tt.append(t)
            anchor_frac[p].append(na / max(len(toks[w][n]), 1))
            anchors.setdefault(w, {})[n] = anchor_indices(toks[w][n], pt)
        times[w] = tt
        t, na = token_times(A[w]["w_tokens"], pt, ptm, end)
        w_times[w] = t

    # cross-system time discrepancy on columns where two systems agree on the token:
    # the same word, timed through two independent hypotheses, should land in the
    # same place. This is the number that says whether the timestamps hold.
    disc, disc_interp = [], []
    for it in items:
        w = it["item_id"]
        if w not in times:
            continue
        idx = [0, 0, 0]
        for col in A[w]["cols"]:
            here = []
            for n in range(3):
                if col[n] is not None:
                    here.append((n, idx[n]))
                    idx[n] += 1
            for x in range(len(here)):
                for y in range(x + 1, len(here)):
                    (na, ia), (nb, ib) = here[x], here[y]
                    if col[na] != col[nb]:
                        continue
                    d = abs(times[w][na][ia] - times[w][nb][ib])
                    disc.append(d)
                    if not (ia in anchors[w][na] and ib in anchors[w][nb]):
                        disc_interp.append(d)
    disc.sort()
    disc_interp.sort()

    def pct(v, q):
        return v[min(len(v) - 1, int(q * len(v)))] if v else None

    alignment_quality = {
        "n_matched_token_pairs": len(disc),
        "abs_time_discrepancy_sec": {
            "p50": pct(disc, 0.50), "p90": pct(disc, 0.90),
            "p95": pct(disc, 0.95), "p99": pct(disc, 0.99),
            "mean": statistics.fmean(disc) if disc else None,
            "share_under_0_5s": sum(1 for d in disc if d < 0.5) / len(disc) if disc else None,
            "share_under_1s": sum(1 for d in disc if d < 1.0) / len(disc) if disc else None,
        },
        "abs_time_discrepancy_sec_interpolated_only": {
            "n_pairs": len(disc_interp),
            "share_of_all_pairs": len(disc_interp) / len(disc) if disc else None,
            "p50": pct(disc_interp, 0.50), "p90": pct(disc_interp, 0.90),
            "p95": pct(disc_interp, 0.95), "p99": pct(disc_interp, 0.99),
            "mean": statistics.fmean(disc_interp) if disc_interp else None,
            "share_under_0_5s": (sum(1 for d in disc_interp if d < 0.5) / len(disc_interp)
                                 if disc_interp else None),
            "share_under_1s": (sum(1 for d in disc_interp if d < 1.0) / len(disc_interp)
                               if disc_interp else None),
            "note": "at least one side interpolated. Pairs anchored on BOTH sides get "
                    "the same pyannote timestamp by construction and say nothing.",
        },
        "anchor_fraction_median": {p: statistics.median(v) if v else None
                                   for p, v in anchor_frac.items()},
        "anchor_fraction_p10": {p: pct(sorted(v), 0.10) for p, v in anchor_frac.items()},
        "windows_without_pyannote": len(pk_missing),
    }

    # -------- LLM arm --------
    llm_acct = Counter()
    if os.environ.get("LLM") == "1":
        from eval.controlled_eval.roster_lexicon import (
            admitted_mined, build_meeting_context, load_city_terms, load_rosters)
        city_terms = load_city_terms()
        rosters = load_rosters()
        mined_by_city, _ = admitted_mined()
        per_city_freq = {c: Counter() for c in city_terms}
        total = Counter()
        # over the UNSEALED items only: the sealed holdout windows' reference text
        # must not reach any part of this pipeline, not even a frequency table
        for x in report["items"]:
            if x["itemId"] in sealed:
                continue
            for tok in wtoks(x["referenceText"]):
                total[tok] += 1
                if x["cityId"] in per_city_freq:
                    per_city_freq[x["cityId"]][tok] += 1
        questions, qmap = [], {}
        for it in items:
            w = it["item_id"]
            city = it["city_id"]
            ctx, _acct = build_meeting_context(
                city, it["meeting_id"], city_terms[city], mined_by_city.get(city, []),
                rosters, total - per_city_freq.get(city, Counter()))
            terms = sorted({i["term"]["canonical"] for i in ctx.present.values()})
            out_tokens = A[w]["w_tokens"]
            pos = 0
            for d in A[w]["decisions"]:
                if d["token"] is not None:
                    pos += 1
                if not d["reason"].startswith("tie_"):
                    continue
                col = A[w]["cols"][d["col"]]
                cands, seen = [], set()
                for e in col:
                    if e is not None and e not in seen:
                        seen.add(e)
                        cands.append(e)
                # candidate 0 is always what W already chose: "in doubt, pick 0"
                cands.sort(key=lambda e: (e != d["token"], e))
                if len(cands) < 2:
                    continue
                qid = f"{w}#{d['col']}"
                left = out_tokens[max(0, pos - 1 - LLM_CTX):max(0, pos - 1)]
                right = out_tokens[pos:pos + LLM_CTX]
                questions.append({"id": qid, "left": " ".join(left),
                                  "right": " ".join(right),
                                  "cands": cands, "terms": terms[:60]})
                qmap[qid] = (w, d["col"], cands)
        log(f"LLM arbitration points: {len(questions)}")
        picks = llm_pass(questions)
        override: dict[str, dict[int, str]] = {}
        for qid, (w, col, cands) in qmap.items():
            p = picks.get(qid)
            if not isinstance(p, int) or not (0 <= p < len(cands)):
                llm_acct["pick_invalid_or_missing"] += 1
                continue
            llm_acct["pick_ok"] += 1
            if cands[p] != cands[0]:
                llm_acct["pick_changed"] += 1
                override.setdefault(w, {})[col] = cands[p]
        texts["W+L"] = {}
        for it in items:
            w = it["item_id"]
            ov = override.get(w, {})
            out = []
            for d in A[w]["decisions"]:
                t = ov.get(d["col"], d["token"])
                if t is not None:
                    out.append(t)
            texts["W+L"][w] = " ".join(out)
            A[w]["w_l_tokens"] = out
        llm_acct["arbitration_points"] = len(questions)

    # -------- D arm: speaker-grounded restoration of dropped runs --------
    d_acct = Counter()
    restored: dict[str, list[tuple[int, list[str]]]] = {}
    for it in items:
        w = it["item_id"]
        decisions = A[w]["decisions"]
        cols = A[w]["cols"]
        kept_times = w_times.get(w)
        ivs = diar.get(w)
        # rebuild the emitted sequence with dropped runs marked
        seq, run = [], []
        for d in decisions:
            col = cols[d["col"]]
            if d["token"] is None:
                if any(e is not None for e in col):
                    tok = next(e for e in col if e is not None)
                    run.append(tok)
                continue
            if run:
                seq.append(("drop", run))
                run = []
            seq.append(("keep", d["token"]))
        if run:
            seq.append(("drop", run))
        restored[w] = []
        if kept_times is None or ivs is None:
            d_acct["no_pyannote"] += 1
            continue
        ki = 0
        end = audio_end.get(w, 120.0)
        for kind, payload in seq:
            if kind == "keep":
                ki += 1
                continue
            d_acct["dropped_runs"] += 1
            t0 = kept_times[ki - 1] if ki > 0 else 0.0
            t1 = kept_times[ki] if ki < len(kept_times) else end
            if t1 - t0 < MIN_GAP_SEC:
                d_acct["gap_too_short"] += 1
                continue
            union, mx = speakers_over(ivs, t0, t1)
            flank = set(speakers_at(ivs, t0)) | set(speakers_at(ivs, max(t1 - 1e-6, t0)))
            if mx >= 2:
                d_acct["restored_overlap"] += 1
            elif union and not (union & flank):
                d_acct["restored_other_speaker"] += 1
            else:
                d_acct["not_restored"] += 1
                continue
            d_acct["restored_tokens"] += len(payload)
            restored[w].append((ki, payload))

    # The LLM arm changes token IDENTITY inside kept columns, never which columns
    # are kept, so the same insertion points apply to W and to W+L unchanged.
    for base in [b for b in ("W", "W+L") if b in texts]:
        seqs = {w: (A[w].get("w_l_tokens") if base == "W+L" else A[w]["w_tokens"])
                for w in A}
        out_arm = {}
        for it in items:
            w = it["item_id"]
            base_toks = list(seqs[w])
            for ki, payload in reversed(restored.get(w, [])):
                base_toks[ki:ki] = payload
            out_arm[w] = " ".join(base_toks)
        texts[f"{base}+D"] = out_arm

    # -------- scoring --------
    per_arm: dict[str, list] = {}
    for p in providers:
        per_arm[f"single:{p}"] = [sdi(it["ref"], it["hyp"][p]) for it in items]
    for arm, tx in texts.items():
        per_arm[arm] = [sdi(it["ref"], tx[it["item_id"]]) for it in items]
    per_arm["oracle_window_trio"] = [
        min((sdi(it["ref"], it["hyp"][p]) for p in TRIO),
            key=lambda c: c[0] + c[1] + c[2]) for it in items]
    per_arm["oracle_window_all"] = [
        min((sdi(it["ref"], it["hyp"][p]) for p in providers),
            key=lambda c: c[0] + c[1] + c[2]) for it in items]
    per_arm["oracle_column"] = [
        sdi(it["ref"], " ".join(A[it["item_id"]]["oracle"])) for it in items]

    prog_arms = {}
    for order in permutations((0, 1, 2)):
        k = "".join(map(str, order))
        per_arm[f"prog{k}:W"] = [
            sdi(it["ref"], " ".join(A[it["item_id"]]["progressive"][k]["voted"]))
            for it in items]
        per_arm[f"prog{k}:oracle"] = [
            sdi(it["ref"], " ".join(A[it["item_id"]]["progressive"][k]["oracle"]))
            for it in items]
        prog_arms[k] = {
            "mean_cost": statistics.fmean(A[w]["progressive"][k]["cost"] for w in A),
            "exact_cost": statistics.fmean(A[w]["cost"] for w in A),
        }

    def counts(arm, idx=None):
        if idx is None:
            return [(r[0] + r[1] + r[2], r[3]) for r in per_arm[arm]]
        return [(r[idx], r[3]) for r in per_arm[arm]]

    def contrast(a, b):
        out = {}
        for metric, idx in (("wer", None), ("sub_rate", 0), ("del_rate", 1),
                            ("ins_rate", 2)):
            out[metric] = cluster_bootstrap(counts(a, idx), counts(b, idx),
                                            clusters, n_boot=N_BOOT)
        out["head2head_wer"] = head2head(counts(a), counts(b))
        return out

    def loo_by(a, b, keyf):
        groups: dict = {}
        for i, it in enumerate(items):
            groups.setdefault(keyf(it), []).append(i)
        ca, cb = counts(a), counts(b)
        allidx = list(range(len(items)))

        def delta(idx):
            den = sum(ca[i][1] for i in idx)
            if den == 0:                 # dropping the only group leaves nothing
                return None
            return (sum(ca[i][0] for i in idx) - sum(cb[i][0] for i in idx)) / den
        full = delta(allidx)
        ds = []
        for k, idx in groups.items():
            s = set(idx)
            d = delta([i for i in allidx if i not in s])
            if d is not None:
                ds.append((d, str(k)))
        ds.sort()
        if not ds:
            return {"full": full, "n_groups": len(groups), "sign_flips": 0,
                    "note": "single group; leave-one-out undefined"}
        return {"full": full, "min": ds[0][0], "min_group": ds[0][1],
                "max": ds[-1][0], "max_group": ds[-1][1],
                "n_groups": len(ds),
                "sign_flips": sum(1 for d, _ in ds if (d > 0) != (full > 0))}

    arms = [a for a in ("V", "W", "W+len", "W+L", "W+D", "W+L+D") if a in per_arm]
    res = {
        "run_id": RUN_ID, "n_items": len(items), "n_meetings": len(set(clusters)),
        "trio": TRIO, "n_boot": N_BOOT,
        "scorer": "eval/controlled_eval/scoring.py (not the benchmark app's)",
        "alignment": {
            "method": "exact 3-way sum-of-pairs DP, banded, band verified per window",
            "band_floor": BAND_FLOOR,
            "mean_columns": statistics.fmean(A[w]["n_cols"] for w in A),
            "mean_cost": statistics.fmean(A[w]["cost"] for w in A),
            "max_band_used": max(A[w]["band"] for w in A),
            "progressive_cost_penalty": prog_arms,
            "quality": alignment_quality,
        },
        "vote_reasons": dict(Counter(
            d["reason"] for w in A for d in A[w]["decisions"])),
        "len_guard": {"windows_fired": len(len_guard_fired)},
        "llm": {"model": LLM_MODEL, "reasoning_effort": "low",
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "accounting": dict(llm_acct)},
        "D": {"min_gap_sec": MIN_GAP_SEC, "accounting": dict(d_acct)},
        "arms": {k: rates(v) for k, v in per_arm.items()},
        "contrasts": {}, "loo": {}, "gates": {}, "oracle_recovery": {},
        "by_mining_disjoint": {},
    }

    v = res["arms"]["V"]
    # The preregistration says reproducing V exactly is what makes this the declared
    # substrate. Assert it rather than eyeballing the log.
    if not os.environ.get("LIMIT"):
        for metric, frozen in (("wer", 0.1201), ("del_rate", 0.0247),
                               ("ins_rate", 0.0443), ("sub_rate", 0.0512)):
            assert abs(v[metric] - frozen) < 5e-5, \
                f"baseline V {metric} is {v[metric]}, preregistered {frozen}"
    o_win = res["arms"]["oracle_window_trio"]["wer"]
    o_col = res["arms"]["oracle_column"]["wer"]
    o_all = res["arms"]["oracle_window_all"]["wer"]
    for a in arms:
        if a == "V":
            continue
        res["contrasts"][f"{a} vs V"] = contrast(a, "V")
        res["loo"][f"{a} vs V"] = {
            "window": loo_by(a, "V", lambda it: it["item_id"]),
            "meeting": loo_by(a, "V", lambda it: it["cluster"]),
            "city": loo_by(a, "V", lambda it: it["city_id"]),
        }
        m = res["arms"][a]
        res["gates"][a] = {
            "ins_rate_gate": {"V": v["ins_rate"], "arm": m["ins_rate"],
                              "pass": m["ins_rate"] <= v["ins_rate"]},
            "del_rate_gate": {"V": v["del_rate"], "arm": m["del_rate"],
                              "pass": m["del_rate"] <= v["del_rate"]},
            "wer_ci_excludes_zero":
                res["contrasts"][f"{a} vs V"]["wer"]["excludes_zero"],
            "loo_sign_stable": all(
                res["loo"][f"{a} vs V"][k]["sign_flips"] == 0
                for k in ("window", "meeting", "city")),
        }
        res["oracle_recovery"][a] = {
            "vs_oracle_window_trio": ((v["wer"] - m["wer"]) / (v["wer"] - o_win)
                                      if v["wer"] != o_win else None),
            "vs_oracle_column": ((v["wer"] - m["wer"]) / (v["wer"] - o_col)
                                 if v["wer"] != o_col else None),
            "vs_oracle_window_all": ((v["wer"] - m["wer"]) / (v["wer"] - o_all)
                                     if v["wer"] != o_all else None),
        }
    if "W+L" in per_arm:
        res["contrasts"]["W+L vs W"] = contrast("W+L", "W")
    for base in [b for b in ("W", "W+L") if f"{b}+D" in per_arm]:
        res["contrasts"][f"{base}+D vs {base}"] = contrast(f"{base}+D", base)
    res["contrasts"]["oracle_column vs oracle_window_trio"] = contrast(
        "oracle_column", "oracle_window_trio")

    # the six progressive orderings, as the honest price of the alignment choice
    res["alignment"]["ordering_sensitivity"] = {
        "exact": {"W": res["arms"]["W"]["wer"], "oracle": o_col},
        "progressive": {k: {"W": res["arms"][f"prog{k}:W"]["wer"],
                            "oracle": res["arms"][f"prog{k}:oracle"]["wer"]}
                        for k in prog_arms},
    }

    # -------- windows disjoint from BOTH the mining fold and training --------
    fold = json.loads((ROOT / "data/glossary/glossary.build-manifest.json").read_text())
    mined_pairs = {(p["city_id"], p["meeting_id"])
                   for p in fold["source_meetings"]["pairs"]}
    dis = [i for i, it in enumerate(items)
           if (it["city_id"], it["meeting_id"]) not in mined_pairs
           and not it["in_training"]]
    res["by_mining_disjoint"] = {
        "n_windows": len(dis),
        "n_meetings": len({items[i]["cluster"] for i in dis}),
        "arms": {a: rates([per_arm[a][i] for i in dis])
                 for a in arms + ["oracle_window_trio", "oracle_column"]},
        "note": "exploratory, underpowered, no clustered CIs: disjoint from BOTH the "
                "glossary mining fold and the fine-tune training manifest",
    }

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")
    for k, val in res["arms"].items():
        log(f"  {k:26s} wer={val['wer']:.4f}  del={val['del_rate']:.4f}  "
            f"ins={val['ins_rate']:.4f}  sub={val['sub_rate']:.4f}")
    for k, val in res["contrasts"].items():
        log(f"  {k}: dWER {val['wer']['delta']:+.5f} {val['wer']['ci95']}")


if __name__ == "__main__":
    main()
