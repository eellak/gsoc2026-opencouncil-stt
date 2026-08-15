#!/usr/bin/env python3
"""Arm B of the serving-stack plan: per-meeting roster hotwords biasing.

Spec: docs/specs/2026-08-12-serving-stack-plan.md (arm B). Thin variant of the
frozen decode harness `notebooks/decode_ablation.py`, which is imported (not
copied) so the CONTROL config, the window-only seeding, the diagnostics handler
and the resumable state format are shared by construction. The ONLY difference
from the control decode is the `hotwords=<string>` transcribe kwarg, built by
`roster_hotwords.build_hotwords_detail` from `data/pii/rosters_full.json`.

Preregistered runtime constraints (asserted, not assumed):
- `initial_prompt` exactly as the control (the control passes none -> None here);
- `prefix=None` (faster-whisper ignores hotwords when a prefix is set);
- hotwords budget <= 160 tokenizer tokens, checked BEFORE the decode;
- uncovered meetings are exact no-ops: the kwarg is not passed at all;
- seeding is the control's own window-only seed (common random numbers), so an
  uncovered window must come out byte-identical to eval-A.

    SC=~/.cache/oc-public .venv-eval/bin/python scripts/serving_stack/hotwords_arm.py plan
    SC=~/.cache/oc-public .venv-eval/bin/python scripts/serving_stack/hotwords_arm.py decode
    SC=~/.cache/oc-public .venv-eval/bin/python scripts/serving_stack/hotwords_arm.py score

Outputs (never in the repo): $SC/serving-stack/eval-B-hotwords.json (decode state,
same schema as decode-ablation state files) and $SC/serving-stack/results-B-eval.json.
Control for scoring is the existing $SC/decode-ablation/eval-A.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval.eval_freeze import ftoks  # noqa: E402
from eval.controlled_eval.exp_same_stack import sdi  # noqa: E402
from scripts.serving_stack.roster_hotwords import (  # noqa: E402
    HOTWORD_TOKEN_BUDGET, build_hotwords_detail, count_tokens, rnorm)


def _load_decode_ablation():
    """Import the frozen harness by path (notebooks/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "decode_ablation", ROOT / "notebooks/decode_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DA = _load_decode_ablation()

ARM = "B-hotwords"
TERMS_DIR = ROOT / "research/ds_wer/terms"
NONINFERIORITY_MARGIN = 0.0010   # common local gate, from the serving-stack spec


def log(m):
    print(m, flush=True)


def out_dir() -> Path:
    d = DA.sc() / "serving-stack"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dest_path(which: str, budget: int = HOTWORD_TOKEN_BUDGET) -> Path:
    """Primary arm at the default budget; the exploratory secondary mode
    (identical policy, different budget, e.g. 200) gets its own file."""
    suffix = "" if budget == HOTWORD_TOKEN_BUDGET else f"-{budget}"
    return out_dir() / f"{which}-B-hotwords{suffix}.json"


def load_tokenizer():
    """The hotword budget is counted with the ACTUAL model tokenizer. The ct2
    dir ships tokenizer.json, so this loads without touching the model weights."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(DA.MODEL_DIR) / "tokenizer.json"))


def transcribe_kwargs(hotwords: str | None, control: dict | None = None,
                      tokenizer=None,
                      budget: int = HOTWORD_TOKEN_BUDGET) -> dict:
    """CONTROL + (optionally) hotwords, with the preregistered assertions.

    Raises AssertionError if the control config carries a prefix or an
    initial_prompt (the control passes neither; faster-whisper silently ignores
    hotwords when prefix is set), or if the hotwords string is over budget.
    """
    kw = dict(DA.CONTROL if control is None else control)
    assert kw.get("prefix") is None, \
        "prefix must be None: faster-whisper ignores hotwords when prefix is set"
    assert kw.get("initial_prompt") is None, \
        "initial_prompt must be exactly the control's (the control passes none)"
    if hotwords is not None:
        assert hotwords.strip(), "empty hotwords must be None (exact no-op)"
        if tokenizer is not None:
            n = count_tokens(tokenizer, hotwords)
            assert n <= budget, \
                f"hotwords over budget before decode: {n} > {budget}"
        kw["hotwords"] = hotwords
    return kw


# ------------------------------------------------------------------------ decode
def decode(which: str, limit: int | None,
           budget: int = HOTWORD_TOKEN_BUDGET) -> Path:
    import ctranslate2
    from faster_whisper import WhisperModel

    tok = load_tokenizer()
    dest = dest_path(which, budget)
    state = (json.loads(dest.read_text()) if dest.exists() else
             {"arm": ARM, "set": which, "config": dict(DA.CONTROL),
              "hotword_budget": budget,
              "exploratory": budget != HOTWORD_TOKEN_BUDGET, "windows": {}})
    assert state["hotword_budget"] == budget, \
        f"{dest} was decoded with budget {state['hotword_budget']}, not {budget}"
    todo = [r for r in DA.rows(which)[:limit]
            if r["window_id"] not in state["windows"]]
    log(f"arm {ARM} [{which}]: {len(todo)} windows to decode "
        f"({len(state['windows'])} already done)")
    if not todo:
        return dest

    model = WhisperModel(DA.MODEL_DIR, device=DA.DEVICE, compute_type=DA.COMPUTE,
                         cpu_threads=DA.THREADS)
    diag = DA.Diagnostics()
    fw_log = logging.getLogger("faster_whisper")
    prev_level = fw_log.level
    fw_log.setLevel(logging.DEBUG)
    fw_log.addHandler(diag)
    try:
        for i, r in enumerate(todo, 1):
            wid = r["window_id"]
            wav = DA.sc() / "bench_windows" / f"{wid}.wav"
            if not wav.exists():
                raise SystemExit(f"missing audio for {wid} - the manifest says "
                                 f"it should be there")
            hw = build_hotwords_detail(r["city"], r["meeting_id"], tok,
                                       budget=budget)
            kw = transcribe_kwargs(hw["hotwords"], tokenizer=tok, budget=budget)

            diag.reset()
            # the control's own window-only seed (common random numbers)
            ctranslate2.set_random_seed(DA.seed_for(ARM, wid))
            t0 = time.time()
            segments, info = model.transcribe(str(wav), **kw)
            segs = list(segments)
            elapsed = time.time() - t0

            resolved = DA.opts_to_dict(info.transcription_options)
            assert resolved.get("prefix") is None, "prefix leaked into the decode"
            assert resolved.get("initial_prompt") is None, \
                "initial_prompt differs from the control"
            assert resolved.get("hotwords") == hw["hotwords"], \
                "hotwords did not reach the decoder unchanged"

            state["windows"][wid] = {
                "text": "".join(s.text for s in segs).strip(),
                "n_segments": len(segs),
                "decoded_seconds": round(sum(s.end - s.start for s in segs), 2),
                "audio_seconds": round(info.duration, 2),
                "temperatures_used": sorted({s.temperature for s in segs
                                             if s.temperature is not None}),
                "seed": DA.seed_for(ARM, wid),
                "wall_seconds": round(elapsed, 1),
                "hotwords": hw["hotwords"],
                "hotwords_tokens": hw["tokens"],
                "hotwords_kept": len(hw["kept"]),
                "hotwords_surnames": hw["n_surnames"],
                "hotwords_source": hw["source"],
                "hotwords_dropped": hw["dropped"],
                "roster_entries": hw["n_roster_entries"],
                **diag.snapshot(),
            }
            state.setdefault("resolved_options", resolved)
            dest.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            log(f"  {ARM} {i}/{len(todo)} {wid} {elapsed:.0f}s segs={len(segs)} "
                f"hw={hw['tokens']}tok/{len(hw['kept'])}kept"
                f"{'/'+str(len(hw['dropped']))+'dropped' if hw['dropped'] else ''}"
                f"{' NO-ROSTER' if hw['hotwords'] is None else ''}")
    finally:
        fw_log.removeHandler(diag)
        fw_log.setLevel(prev_level)
    log(f"arm {ARM} [{which}] -> {dest}")
    return dest


def plan(which: str, budget: int = HOTWORD_TOKEN_BUDGET) -> None:
    """Build every hotwords string without loading the model: coverage + budgets."""
    tok = load_tokenizer()
    covered = 0
    for r in DA.rows(which):
        hw = build_hotwords_detail(r["city"], r["meeting_id"], tok,
                                   budget=budget)
        covered += hw["hotwords"] is not None
        log(f"{r['window_id']}: roster={hw['n_roster_entries']} "
            f"persons={hw['n_surnames']} kept={len(hw['kept'])} "
            f"dropped={len(hw['dropped'])} tokens={hw['tokens']}/{budget}"
            f"{'  UNCOVERED (exact no-op)' if hw['hotwords'] is None else ''}")
    log(f"coverage: {covered}/{len(DA.rows(which))} windows (budget {budget})")


# ------------------------------------------------------------------- name metrics
def meeting_term_ids(city_terms: list[dict], roster_entries: list[str]) -> set[str]:
    """Term ids present in this meeting's roster (name_lexicon_audit's rule):
    canonical surname appears as a word in a roster entry, or a covered full name
    appears verbatim among the entries (both sides rnorm)."""
    roster_norm = [rnorm(e) for e in roster_entries]
    roster_words = set()
    for e in roster_norm:
        roster_words.update(re.findall(r"\w+", e))
    present = set()
    for t in city_terms:
        if rnorm(t["canonical"]) in roster_words or any(
                rnorm(cov) in roster_norm for cov in t.get("covers", [])):
            present.add(t["id"])
    return present


def name_stats(which: str, texts: dict[str, str]) -> dict:
    """Per-window mention counts against the DS-WER person_surname terms.

    Returns per-window rows {N, matched, hyp_mentions, oor_insertions} where
    matched = reference mentions aligned MATCH (same term id), precision counts
    every hypothesis mention, and oor = erroneous hypothesis name mentions whose
    term is NOT in the meeting roster.
    """
    from ds_wer import MATCH, TermList, align, tag

    rosters = json.loads((ROOT / "data/pii/rosters_full.json").read_text())
    terms, tls = {}, {}
    for city in sorted({r["city"] for r in DA.rows(which)}):
        data = json.loads((TERMS_DIR / f"{city}.json").read_text())
        terms[city] = [t for t in data["terms"] if t["klass"] == "person_surname"]
        tls[city] = TermList(terms[city])

    present_cache: dict[str, set[str]] = {}
    out = {}
    for r in DA.rows(which):
        wid, city, meeting = r["window_id"], r["city"], r["meeting_id"]
        key = f"{city}/{meeting}"
        if key not in present_cache:
            present_cache[key] = meeting_term_ids(
                terms[city], rosters.get(key) or [])
        present = present_cache[key]

        ref_tag = tag(ftoks(DA.reference_text(wid)), tls[city])
        hyp_tag = tag(ftoks(texts[wid]), tls[city])
        ops = align([s for _, s in ref_tag], [s for _, s in hyp_tag])

        n = sum(1 for is_t, _ in ref_tag if is_t)
        h = sum(1 for is_t, _ in hyp_tag if is_t)
        matched = oor = 0
        for op, i, j in ops:
            hyp_is_term = j is not None and hyp_tag[j][0]
            if not hyp_is_term:
                continue
            if op == MATCH and ref_tag[i][0]:
                matched += 1          # same symbol => same term id
            elif hyp_tag[j][1] not in present:
                oor += 1              # erroneous name mention, not on the roster
        out[wid] = {"N": n, "matched": matched, "hyp_mentions": h,
                    "oor_insertions": oor}
    return out


def pooled_names(rows: dict) -> dict:
    n = sum(r["N"] for r in rows.values())
    m = sum(r["matched"] for r in rows.values())
    h = sum(r["hyp_mentions"] for r in rows.values())
    oor = sum(r["oor_insertions"] for r in rows.values())
    return {"gold_mentions": n, "matched": m, "hyp_mentions": h,
            "mention_recall": m / n if n else None,
            "mention_precision": m / h if h else None,
            "oor_name_insertions": oor}


# ------------------------------------------------------------------------- score
PICK = {"wer": lambda t: t[0] + t[1] + t[2], "sub_rate": lambda t: t[0],
        "del_rate": lambda t: t[1], "ins_rate": lambda t: t[2]}


def per_window_b(which: str, budget: int = HOTWORD_TOKEN_BUDGET) -> tuple[dict, dict]:
    """(scores, texts) for the B decode; refuses incomplete decodes."""
    p = dest_path(which, budget)
    if not p.exists():
        raise SystemExit(f"arm {ARM} [{which}] not decoded: {p} missing")
    state = json.loads(p.read_text())
    scores, texts = {}, {}
    for r in DA.rows(which):
        wid = r["window_id"]
        if wid not in state["windows"]:
            raise SystemExit(f"arm {ARM} [{which}] is incomplete: {wid} missing. "
                             f"No complete-case subsets.")
        texts[wid] = state["windows"][wid]["text"]
        ref = ftoks(DA.reference_text(wid))
        scores[wid] = (*sdi(ref, ftoks(texts[wid])), len(ref))
    return scores, texts


def domination(b: dict, a: dict, wids: list[str]) -> dict:
    """Does one window supply >50% of the net WER gain (spec's domination check)?"""
    deltas = {w: PICK["wer"](b[w]) - PICK["wer"](a[w]) for w in wids}
    total = sum(deltas.values())
    if total >= 0:
        return {"net_error_delta": total, "max_gain_share": None,
                "max_gain_window": None, "dominated": None}
    gains = {w: d for w, d in deltas.items() if d < 0}
    w_max = min(gains, key=gains.get)
    share = gains[w_max] / total
    return {"net_error_delta": total, "max_gain_share": share,
            "max_gain_window": w_max, "dominated": bool(share > 0.5)}


def score(which: str, budget: int = HOTWORD_TOKEN_BUDGET) -> dict:
    wids = [r["window_id"] for r in DA.rows(which)]
    blocks = [r["meeting_id"] for r in DA.rows(which)]

    ctrl = DA.per_window(which, "A")     # $SC/decode-ablation/eval-A.json
    ctrl_texts = {w: json.loads(
        (DA.out_dir() / f"{which}-A.json").read_text())["windows"][w]["text"]
        for w in wids}
    b, b_texts = per_window_b(which, budget)

    st = json.loads(dest_path(which, budget).read_text())
    uncovered = [w for w in wids if st["windows"][w]["hotwords"] is None]
    noop_violations = [w for w in uncovered if b_texts[w] != ctrl_texts[w]]

    vs = {}
    for metric, pick in PICK.items():
        ci = DA.paired_ci(b, ctrl, wids, blocks, pick)
        vs[metric] = {"delta": ci["delta"], "ci95": ci["ci95"]}

    names_ctrl = name_stats(which, ctrl_texts)
    names_b = name_stats(which, b_texts)
    pn_ctrl, pn_b = pooled_names(names_ctrl), pooled_names(names_b)

    wer_ci = vs["wer"]
    gate = {
        "point_delta": wer_ci["delta"],
        "upper_bound": wer_ci["ci95"][1],
        "upper_bound_note": ("two-sided 97.5th percentile used as the upper "
                             "bound; conservative vs the spec's one-sided 95%"),
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "wer_screen_passed": bool(wer_ci["delta"] < 0
                                  and wer_ci["ci95"][1] < NONINFERIORITY_MARGIN),
        "domination": domination(b, ctrl, wids),
        "leave_one_out": DA.influence(b, ctrl, wids, PICK["wer"]),
        "noop_violations": noop_violations,
    }

    result = {
        "arm": ARM, "set": which, "hotword_budget": budget,
        "exploratory": budget != HOTWORD_TOKEN_BUDGET,
        "n_windows": len(wids),
        "n_meetings": len(set(blocks)),
        "control": "decode-ablation eval-A",
        "bootstrap": {"replicates": DA.BOOTSTRAP_REPLICATES,
                      "seed": DA.BOOTSTRAP_SEED, "blocks": "meeting_id"},
        "totals": {"control": DA.rates(ctrl), "B": DA.rates(b)},
        "vs_control": vs,
        "coverage": {"covered": len(wids) - len(uncovered),
                     "uncovered_windows": uncovered},
        "names": {"terms": "research/ds_wer/terms person_surname",
                  "control": pn_ctrl, "B": pn_b,
                  "recall_delta": (pn_b["mention_recall"] or 0)
                                  - (pn_ctrl["mention_recall"] or 0),
                  "precision_delta": (pn_b["mention_precision"] or 0)
                                     - (pn_ctrl["mention_precision"] or 0),
                  "oor_insertions_delta": pn_b["oor_name_insertions"]
                                          - pn_ctrl["oor_name_insertions"],
                  "per_window": {"control": names_ctrl, "B": names_b}},
        "gate": gate,
        "diagnostics": {k: sum(w[k] for w in st["windows"].values())
                        for k in ("no_speech_skips", "compression_trips",
                                  "logprob_trips", "n_segments")},
        "wall_seconds": round(sum(w["wall_seconds"]
                                  for w in st["windows"].values()), 1),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "decode", "score"):
        p = sub.add_parser(name)
        p.add_argument("--set", dest="which", default="eval",
                       choices=("eval", "holdout"))
        p.add_argument("--budget", type=int, default=HOTWORD_TOKEN_BUDGET,
                       help="hotword token budget; non-default (e.g. 200) is the "
                            "exploratory secondary mode, identical policy")
        if name == "decode":
            p.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.cmd == "plan":
        plan(a.which, a.budget)
    elif a.cmd == "decode":
        decode(a.which, a.limit, a.budget)
    else:
        res = score(a.which, a.budget)
        suffix = "" if a.budget == HOTWORD_TOKEN_BUDGET else f"-{a.budget}"
        dest = out_dir() / f"results-B-{a.which}{suffix}.json"
        dest.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        t_c, t_b = res["totals"]["control"], res["totals"]["B"]
        log(f"control  WER {t_c['wer']:.4f}  del {t_c['del_rate']:.4f}  "
            f"ins {t_c['ins_rate']:.4f}")
        log(f"B        WER {t_b['wer']:.4f}  del {t_b['del_rate']:.4f}  "
            f"ins {t_b['ins_rate']:.4f}")
        g = res["gate"]
        log(f"dWER {g['point_delta']:+.4f}  upper {g['upper_bound']:+.4f}  "
            f"wer_screen {'PASS' if g['wer_screen_passed'] else 'FAIL'}")
        nm = res["names"]
        log(f"names: recall {nm['control']['mention_recall']:.3f} -> "
            f"{nm['B']['mention_recall']:.3f}  precision "
            f"{nm['control']['mention_precision']:.3f} -> "
            f"{nm['B']['mention_precision']:.3f}  oor "
            f"{nm['control']['oor_name_insertions']} -> "
            f"{nm['B']['oor_name_insertions']}")
        log(f"-> {dest}")


if __name__ == "__main__":
    main()
