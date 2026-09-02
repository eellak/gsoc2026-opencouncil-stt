"""Arm R: retrieval-biased hotwords on the 39 frozen validation windows.

Spec: docs/specs/2026-08-26-training-round-2.md section 3.4.
Record: exp-2026-08-27-retrieval-hotwords (OPEN).

Forked from `hotwords_arm.py` (arm B, CLOSED negative) so the two are directly
comparable: same model, same frozen CONTROL, same window-only seed, same scorer, same
bootstrap. The only difference is where the prompt comes from.

  arm B  static, hash-ordered, 160 tokens, ~20 surnames, on every window
  arm R  retrieved per window from the unbiased pass-1 output, K <= 3, 24 tokens,
         and no second decode at all when nothing is retrieved

Pass 1 is READ from `$SC/decode-ablation/eval-A.json` rather than re-derived, because
`exp-2026-08-16-adapter-confidence` withdrew the bit-exact reproducibility claim for
this local CPU decode. That file is also the control the contrast is scored against,
so R - A is paired by construction and empty-retrieval windows enter as exact zeroes.

Modes:
  plan     build every retrieval without loading the model (coverage, budgets)
  decode   run the conditional second pass; resumable
  score    R - A on the eight frozen gates
  decoy    the separate R+D negative-control pass

Endpoints are agreement-with-OpenCouncil. They advance an engineering screen; they do
not authorise shipping without fidelity adjudication of the changed spans.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from eval.controlled_eval.retrieval_hotwords import (  # noqa: E402
    RetrievalConfig, fit_budget, retrieve)
from scripts.serving_stack.hotwords_arm import (  # noqa: E402
    load_tokenizer, name_stats, pooled_names)
from scripts.serving_stack.roster_hotwords import (  # noqa: E402
    _load_person_terms, _load_rosters, _surnames_from_roster,
    _surnames_from_terms, count_tokens)


def _load_decode_ablation():
    spec = importlib.util.spec_from_file_location(
        "decode_ablation", ROOT / "notebooks" / "decode_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DA = _load_decode_ablation()

ARM = "R-retrieval"

# --------------------------------------------------------------------- frozen
# Written into the spec on 2026-08-27 before any decode. Editing these means
# editing the spec table and the ledger caveat in the same change.
CFG = RetrievalConfig(
    # STRICT. The earlier LOOSE choice compared canonicals only; once the recorded
    # inflections are in the lexicon it inverts. LOOSE merges beta/upsilon, so a
    # genuine misspelling of an inflected name (`Μαβρίδη` for the alias `μαυριδη`)
    # collapses to distance 0 and is skipped as already-correct -- exactly the error
    # class this arm exists to catch. STRICT keeps them one edit apart.
    loose=False,
    max_distance=1,        # distance 2 is the pre-declared secondary
    min_skeleton_len=4,    # below 4, Greek function words collide
    k_max=3,
    token_budget=24,       # arm B used 160; short must be structural, not incidental
    include_exact=False,
)
NAME_GAIN_MIN = 18                  # half of arm B's +35 on the frozen 246 denominator
NONINFERIORITY_MARGIN = 0.0010      # the standing serving-stack margin
INSERTION_MARGIN = 0.0005
DELETION_PARTIAL_MARGIN = 0.0028    # a quarter of arm B's +0.0112
NAME_DOMINATION_MAX = 0.25


def log(m):
    print(m, flush=True)


def out_dir() -> Path:
    d = DA.sc() / "retrieval-hotwords-2026-08"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dest_path(which: str, decoy: bool = False) -> Path:
    # Primary and decoy never share a result file: they are different policies and a
    # mixed file would let one be scored as the other.
    return out_dir() / f"{which}-{ARM}{'-decoy' if decoy else ''}.json"


def block_key(row: dict) -> str:
    """Cluster unit. (city, meeting) and never meeting alone: slugs repeat across
    cities (exp_composition_postjune.py:212-216)."""
    return f"{row['city']}/{row['meeting_id']}"


# ------------------------------------------------------------------- lexicon
def meeting_lexicon(city: str, meeting: str) -> list[str]:
    """The roster-present `person_surname` universe of arm B, canonical -> forms.

    Deliberately NOT `roster_lexicon.py`: that admits error-mined terms whose mining
    fold overlaps the benchmark, which would change coverage and application at the
    same time. A broader lexicon is a separate, explicitly exploratory arm.
    """
    entries = _load_rosters().get(f"{city}/{meeting}") or []
    if not entries:
        return {}
    terms = _load_person_terms(city)
    if not terms:
        return {n: [n] for n in sorted(set(_surnames_from_roster(entries)))}
    names = set(_surnames_from_terms(terms, entries))
    # Carry each surname's recorded inflections as extra surface forms. Without them a
    # plain genitive is one edit from the nominative and the prompt fires on a name
    # the model already wrote correctly -- found by `plan` before any decode ran.
    return {t["canonical"]: [t["canonical"]] + list(t.get("aliases") or [])
            for t in terms if t["canonical"] in names}


def pass1_texts(which: str) -> dict[str, str]:
    """The cached unbiased control decode. Refuses an incomplete file."""
    p = DA.out_dir() / f"{which}-A.json"
    if not p.exists():
        raise SystemExit(f"pass 1 is missing: {p}. Arm R does not re-derive it.")
    state = json.loads(p.read_text())
    texts = {}
    for r in DA.rows(which):
        wid = r["window_id"]
        if wid not in state["windows"]:
            raise SystemExit(f"pass 1 is incomplete: {wid} missing from {p}. "
                             "No complete-case subsets.")
        texts[wid] = state["windows"][wid]["text"]
    return texts


def plan_window(row: dict, text: str, tok, decoy_from: dict | None = None) -> dict:
    """Retrieval for one window, without the model. Pure and cheap."""
    lex = meeting_lexicon(row["city"], row["meeting_id"])
    got = fit_budget(retrieve(text, lex, CFG), CFG.token_budget, tok)
    surfaces = [r.surface for r in got]
    decoy = None
    if decoy_from is not None and surfaces:
        decoy = decoy_from.get(row["window_id"])
        if decoy:
            surfaces = surfaces + [decoy]
    hw = " ".join(surfaces) if surfaces else None
    return {
        "hotwords": hw,
        "hotwords_tokens": count_tokens(tok, hw) if hw else 0,
        "n_retrieved": len(got),
        "n_lexicon": len(lex),
        "decoy": decoy,
        "retrieved": [
            {"surface": r.surface, "skeleton": r.skeleton, "distance": r.distance,
             "rank": r.rank, "matched_token": r.matched_token} for r in got
        ],
    }


def build_decoys(which: str) -> dict[str, str]:
    """One decoy surname per window, deterministically drawn from a DIFFERENT city.

    Deterministic so the decoy pass is reproducible, and cross-city so the entry is
    genuinely absent from the meeting's own roster. An emission is still only
    `prompt-induced decoy emission` until the audio is adjudicated: a councillor from
    another municipality can be mentioned.
    """
    rows = DA.rows(which)
    pools = {}
    for r in rows:
        pools.setdefault(r["city"], set()).update(
            meeting_lexicon(r["city"], r["meeting_id"]))
    out = {}
    for r in rows:
        others = sorted(
            n for c, names in sorted(pools.items()) if c != r["city"] for n in names)
        own = set(meeting_lexicon(r["city"], r["meeting_id"]))
        others = [n for n in others if n not in own]
        if others:
            out[r["window_id"]] = others[
                int(DA.seed_for(ARM, r["window_id"])) % len(others)]
    return out


def plan(which: str, decoy: bool = False) -> dict:
    tok = load_tokenizer()
    decoys = build_decoys(which) if decoy else None
    texts = pass1_texts(which)
    rows = {}
    for r in DA.rows(which):
        rows[r["window_id"]] = plan_window(r, texts[r["window_id"]], tok, decoys)
    fired = [w for w, p in rows.items() if p["hotwords"]]
    toks = [p["hotwords_tokens"] for p in rows.values() if p["hotwords"]]
    summary = {
        "windows": len(rows),
        "pass2_windows": len(fired),
        "pass2_share": round(len(fired) / len(rows), 3) if rows else None,
        "tokens_max": max(toks) if toks else 0,
        "tokens_mean": round(sum(toks) / len(toks), 2) if toks else 0,
        "retrieved_total": sum(p["n_retrieved"] for p in rows.values()),
        "no_lexicon_windows": sum(1 for p in rows.values() if p["n_lexicon"] == 0),
    }
    return {"summary": summary, "rows": rows}


# -------------------------------------------------------------------- decode
def decode(which: str, limit: int | None = None, decoy: bool = False) -> Path:
    import ctranslate2
    from faster_whisper import WhisperModel

    from eval.chunking_decode import verify_model

    model_hash = verify_model()   # arm B never did this
    tok = load_tokenizer()
    dest = dest_path(which, decoy)
    planned = plan(which, decoy=decoy)
    state = (json.loads(dest.read_text()) if dest.exists() else
             {"arm": ARM, "set": which, "decoy": decoy,
              "config": dict(DA.CONTROL), "retrieval_config": vars(CFG),
              "model_sha256_16": model_hash, "windows": {}})
    if state.get("retrieval_config") != vars(CFG) or state.get("decoy") != decoy:
        raise SystemExit(f"{dest} was written under a different retrieval config "
                         "or mode; refusing to mix policies in one file")
    if state.get("model_sha256_16") != model_hash:
        raise SystemExit(f"{dest} was decoded with model {state['model_sha256_16']}, "
                         f"now {model_hash}")

    todo = [r for r in DA.rows(which)[:limit]
            if r["window_id"] not in state["windows"]]
    fires = sum(1 for r in todo if planned["rows"][r["window_id"]]["hotwords"])
    log(f"arm {ARM}{' decoy' if decoy else ''} [{which}]: {len(todo)} windows, "
        f"{fires} need a second pass ({len(todo) - fires} carried from pass 1)")

    texts = pass1_texts(which)
    model = None
    diag = DA.Diagnostics()
    fw_log = logging.getLogger("faster_whisper")
    prev_level = fw_log.level
    try:
        for i, r in enumerate(todo, 1):
            wid = r["window_id"]
            p = planned["rows"][wid]
            base = {"window_id": wid, **p, "pass1_text": texts[wid],
                    "seed": DA.seed_for(ARM, wid)}
            if not p["hotwords"]:
                state["windows"][wid] = {
                    **base, "text": texts[wid], "pass2_ran": False}
                dest.write_text(json.dumps(state, ensure_ascii=False, indent=1))
                log(f"  {i}/{len(todo)} {wid} no retrieval, pass-1 text kept")
                continue

            if model is None:
                model = WhisperModel(DA.MODEL_DIR, device=DA.DEVICE,
                                     compute_type=DA.COMPUTE, cpu_threads=DA.THREADS)
                fw_log.setLevel(logging.DEBUG)
                fw_log.addHandler(diag)

            wav = DA.sc() / "bench_windows" / f"{wid}.wav"
            if not wav.exists():
                raise SystemExit(f"missing audio for {wid}")
            kw = dict(DA.CONTROL)
            assert kw.get("prefix") is None and kw.get("initial_prompt") is None
            kw["hotwords"] = p["hotwords"]

            diag.reset()
            ctranslate2.set_random_seed(DA.seed_for(ARM, wid))
            t0 = time.time()
            segments, info = model.transcribe(str(wav), **kw)
            segs = list(segments)
            elapsed = time.time() - t0

            resolved = DA.opts_to_dict(info.transcription_options)
            assert resolved.get("prefix") is None, "prefix leaked into the decode"
            assert resolved.get("initial_prompt") is None, "initial_prompt leaked"
            assert resolved.get("hotwords") == p["hotwords"], \
                "hotwords did not reach the decoder unchanged"

            state["windows"][wid] = {
                **base,
                "text": "".join(s.text for s in segs).strip(),
                "pass2_ran": True,
                "n_segments": len(segs),
                "wall_seconds": round(elapsed, 1),
                **diag.snapshot(),
            }
            state.setdefault("resolved_options", resolved)
            dest.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            log(f"  {i}/{len(todo)} {wid} {elapsed:.0f}s "
                f"hw={p['hotwords_tokens']}tok/{p['n_retrieved']} {p['hotwords']!r}")
    finally:
        fw_log.removeHandler(diag)
        fw_log.setLevel(prev_level)
    log(f"arm {ARM} [{which}] -> {dest}")
    return dest


# --------------------------------------------------------------------- score
def per_window_r(which: str, decoy: bool = False) -> tuple[dict, dict]:
    p = dest_path(which, decoy)
    if not p.exists():
        raise SystemExit(f"arm {ARM} [{which}] not decoded: {p} missing")
    state = json.loads(p.read_text())
    scores, texts = {}, {}
    for r in DA.rows(which):
        wid = r["window_id"]
        if wid not in state["windows"]:
            raise SystemExit(f"arm {ARM} [{which}] is incomplete: {wid} missing. "
                             "No complete-case subsets.")
        texts[wid] = state["windows"][wid]["text"]
        ref = ftoks(DA.reference_text(wid))
        scores[wid] = (*sdi(ref, ftoks(texts[wid])), len(ref))
    return scores, texts


def name_domination(b_rows: dict, a_rows: dict, wids: list[str],
                    blocks: dict) -> dict:
    """Does one window or one meeting supply more than 25 % of the name gain?

    Anchored to the name-mention gain rather than to WER, because dWER here is near
    zero by design and '25 % of a quantity indistinguishable from zero' is undefined.
    """
    per_w = {w: b_rows[w]["matched"] - a_rows[w]["matched"] for w in wids}
    total = sum(per_w.values())
    if total <= 0:
        return {"name_gain": total, "max_share": None, "max_unit": None,
                "dominated": None}
    per_m: dict[str, int] = {}
    for w in wids:
        per_m[blocks[w]] = per_m.get(blocks[w], 0) + per_w[w]
    units = {**{w: v for w, v in per_w.items() if v > 0},
             **{m: v for m, v in per_m.items() if v > 0}}
    unit = max(units, key=lambda k: (units[k], k))
    share = units[unit] / total
    return {"name_gain": total, "max_share": share, "max_unit": unit,
            "dominated": bool(share > NAME_DOMINATION_MAX)}


def evaluate_gates(m: dict) -> dict:
    """The frozen advance rule of spec section 3.4. Pure; see the test file."""
    g5 = m["name_gain"] >= NAME_GAIN_MIN
    gates = {
        "1_wer_point": m["wer_delta"] < 0,
        "2_wer_upper": m["wer_hi"] < NONINFERIORITY_MARGIN,
        "3_deletion_upper": m["del_hi"] <= 0,
        "4_insertion": m["ins_delta"] < INSERTION_MARGIN,
        "5_name_gain": g5,
        "6_name_precision": m["precision_delta"] >= 0 and m["oor_delta"] <= 0,
        # Undefined without a gain to dominate: reported n/a, never passed.
        "7_loo_reversal": (not m["loo_reversal"]) if g5 else None,
        "8_name_domination": (m["name_dom_share"] is not None
                              and m["name_dom_share"] <= NAME_DOMINATION_MAX) if g5 else None,
    }
    if all(v is True for v in gates.values()):
        verdict = "ADVANCE"
    elif (all(gates[k] is True for k in
              ("1_wer_point", "2_wer_upper", "4_insertion", "5_name_gain",
               "6_name_precision"))
          and m["del_point"] <= 0 and m["del_hi"] <= DELETION_PARTIAL_MARGIN):
        verdict = "PARTIAL"
    else:
        verdict = "STOP"
    return {"gates": gates, "verdict": verdict}


def score(which: str, decoy: bool = False) -> dict:
    rows = DA.rows(which)
    wids = [r["window_id"] for r in rows]
    blocks = {r["window_id"]: block_key(r) for r in rows}
    block_list = [blocks[w] for w in wids]

    ctrl = DA.per_window(which, "A")
    ctrl_texts = pass1_texts(which)
    arm, arm_texts = per_window_r(which, decoy)

    names_a = name_stats(which, ctrl_texts)
    names_r = name_stats(which, arm_texts)
    pooled_a, pooled_r = pooled_names(names_a), pooled_names(names_r)

    # Argument order matters and is the repo's convention: paired_ci(a, b) is
    # rate(a) - rate(b), so the arm comes FIRST to make the contrast R - A. Reversing
    # it silently flips the sign of every gate.
    cis = {k: DA.paired_ci(arm, ctrl, wids, block_list, DA.PICK[k])
           for k in ("wer", "del_rate", "ins_rate")}
    dom = name_domination(names_r, names_a, wids, blocks)
    loo = DA.influence(arm, ctrl, wids, DA.PICK["wer"])

    measured = {
        "wer_delta": cis["wer"]["delta"],
        "wer_hi": cis["wer"]["ci95"][1],
        "del_point": cis["del_rate"]["delta"],
        "del_hi": cis["del_rate"]["ci95"][1],
        "ins_delta": cis["ins_rate"]["delta"],
        "name_gain": pooled_r["matched"] - pooled_a["matched"],
        "precision_delta": (pooled_r["mention_precision"] or 0)
                           - (pooled_a["mention_precision"] or 0),
        "oor_delta": pooled_r["oor_name_insertions"] - pooled_a["oor_name_insertions"],
        "loo_reversal": bool(loo.get("sign_reversed_by")),
        "name_dom_share": dom["max_share"],
    }
    state = json.loads(dest_path(which, decoy).read_text())
    fired = sum(1 for w in state["windows"].values() if w.get("pass2_ran"))
    return {
        "arm": ARM, "set": which, "decoy": decoy,
        "model_sha256_16": state.get("model_sha256_16"),
        "retrieval_config": vars(CFG),
        "windows": len(wids), "pass2_windows": fired,
        "control": DA.rates(ctrl), "arm_rates": DA.rates(arm),
        "cis": cis, "names_control": pooled_a, "names_arm": pooled_r,
        "name_domination": dom, "leave_one_out": loo,
        "measured": measured, **evaluate_gates(measured),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["plan", "decode", "score", "decoy-decode",
                                     "decoy-score"])
    ap.add_argument("--set", default="eval")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if a.mode == "plan":
        out = plan(a.set)
        log(json.dumps(out["summary"], indent=2))
        for wid, p in out["rows"].items():
            if p["hotwords"]:
                log(f"  {wid}: {p['hotwords_tokens']}tok "
                    + ", ".join(f"{r['matched_token']}->{r['surface']}(d{r['distance']})"
                                for r in p["retrieved"]))
    elif a.mode == "decode":
        decode(a.set, a.limit)
    elif a.mode == "decoy-decode":
        decode(a.set, a.limit, decoy=True)
    else:
        res = score(a.set, decoy=a.mode.startswith("decoy"))
        p = out_dir() / f"results-{a.set}-{ARM}{'-decoy' if a.mode.startswith('decoy') else ''}.json"
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        log(json.dumps({k: res[k] for k in
                        ("windows", "pass2_windows", "measured", "gates", "verdict")},
                       indent=2, ensure_ascii=False))
        log(f"-> {p}")


if __name__ == "__main__":
    main()
