#!/usr/bin/env python3
"""Can the LLM post-editor be deployed? Two things stand between it and production.

The 2026-07-29 run (`exp_llm_postedit.py`) showed post-editing Scribe output with a
per-meeting roster takes the corrected subset from 0.155 to 0.119 WER, the best result
this project has measured. It also showed the failure mode: in 2 of 150 outputs the
model wrote commentary instead of a transcript, and when it goes off-script it destroys
the utterance (+63 edits against a 17-word reference). Two open questions:

  1. OUTPUT-VALIDITY GATE. Can a reference-free check catch those without throwing away
     good edits? A gate that also rejects real corrections is not a fix.

  2. OVER-EDITING. That subset is 100% utterances a human had to correct, i.e. where the
     ASR was wrong. In production the editor runs over everything, most of which is
     already right. What does it cost on text that needs no help?

Question 2 is measured by feeding the editor the HUMAN-CORRECTED text and scoring its
output against that same text. Input already equals the reference, so a well-behaved
editor scores 0.000 and anything above that is damage it caused — no ASR, no audio and
no contaminated reference involved.

Arms (all use the same frozen prompt as exp_llm_postedit.py's llm_roster):
  A_scribe   post-edit(initial_before_text + roster)  scored vs final_after_text
  B_clean    post-edit(final_after_text  + roster)    scored vs final_after_text  -> ideal 0
  B_clean_train  same as B, on corrected utterances from the TRAINING cities, to put a
                 usable n behind a rare failure rate (nothing is trained here, so
                 held-out has no meaning for a post-editor; it only matters for
                 comparability with the earlier ASR runs)

Writes aggregates to results_postedit_gate.json (tracked) and per-utterance text to
$SC/postedit_gate_detail.json (NOT tracked — utterance text is PII).

Env: N_HELD (98) N_TRAIN (150) LLM_MODEL (sonnet) SC (scratchpad dir) DRY (1 = no calls)
"""
import collections, json, os, random, re, sys, time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
from eval.backends import generate                                    # noqa: E402
from eval.controlled_eval.scoring import (norm, wtoks, edist, counts,  # noqa: E402
                                          wer, cluster_bootstrap, head2head)

AUDIO = ROOT / "data/asr/audio"
EXPORT = ROOT / "data/asr/export.jsonl"
ROSTERS = ROOT / "data/pii/rosters_full.json"
HELD_CITIES = ("argos", "orestiada")
SC = Path(os.environ.get("SC", "/tmp"))
OUT = Path(__file__).with_name("results_postedit_gate.json")
N_HELD = int(os.environ.get("N_HELD", "98"))
N_TRAIN = int(os.environ.get("N_TRAIN", "150"))
MODEL = os.environ.get("LLM_MODEL", "sonnet")
DRY = os.environ.get("DRY") == "1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- the prompt
# Frozen, character-for-character, from exp_llm_postedit.py. Changing it would make
# these numbers incomparable with the 0.119 result they exist to qualify.
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
    ctx = (f"Ονόματα και όροι που εμφανίζονται σε αυτή τη συνεδρίαση:\n{roster}\n\n"
           if roster else "")
    return f"{ctx}Κείμενο ASR:\n{text}"


# ---------------------------------------------------------------- the gate
# Reference-free by construction: at serving time there is no human transcript to
# compare against, so the gate may only look at the input and the output.
META = re.compile(r"^\s*(?:```|>|[-*]\s|\"|«|Το διορθωμένο|Διορθωμένο|Σημείωση|"
                  r"Note:|Here is|Δεν μπορώ|Δεν υπάρχ)", re.IGNORECASE)
MIN_RATIO, MAX_RATIO = 0.6, 1.6
MAX_REL_EDIT = 0.35


def gate(src, out):
    """Return (accepted, reason). A rejected edit falls back to the ASR text.

    Thresholds are deliberately loose. The gate exists to catch an editor that stopped
    transcribing and started talking, not to second-guess its word choices — every
    rejection of a genuine correction is a lost improvement, so it must be rare.
    """
    o = (out or "").strip()
    if not o:
        return False, "empty"
    if "\n\n" in o.strip() or o.count("\n") > 1:
        return False, "multi_paragraph"
    if META.match(o):
        return False, "meta_marker"
    s_toks, o_toks = wtoks(src), wtoks(o)
    if not o_toks:
        return False, "no_tokens"
    ratio = len(o_toks) / max(1, len(s_toks))
    if not (MIN_RATIO <= ratio <= MAX_RATIO):
        return False, "length_ratio"
    rel = edist(s_toks, o_toks) / max(1, len(s_toks))
    if rel > MAX_REL_EDIT:
        return False, "rewrote_too_much"
    return True, "ok"


def repair(out):
    """What exp_llm_postedit.py did: keep the text before the first blank line.

    Included to check whether silently repairing beats rejecting. Repairing keeps some
    salvageable outputs but also lets through anything whose first paragraph is already
    wrong, so it is not obviously the safer default.
    """
    return (out or "").split("\n\n")[0].strip()


# ---------------------------------------------------------------- sample
def load_sample():
    random.seed(13)
    rosters = json.load(open(ROSTERS))
    held, train = [], []
    for line in open(EXPORT):
        d = json.loads(line)
        bef, aft = d.get("initial_before_text", ""), d.get("final_after_text", "")
        if not aft or bef.strip() == aft.strip():
            continue
        try:
            dur = float(d["end"]) - float(d["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (4.0 <= dur <= 30.0) or len(aft.split()) < 6:
            continue
        d["_roster"] = rosters.get(f"{d['city_id']}/{d['meeting_id']}") or []
        if d["city_id"] in HELD_CITIES:
            # same audio-existence filter as ab_hotwords.py, so this sample is a
            # superset of the 50 utterances the 0.119 figure was measured on
            if not (AUDIO / f"{d['city_id']}__{d['meeting_id']}.mp3").exists():
                continue
            held.append(d)
        else:
            train.append(d)
    random.shuffle(held); random.shuffle(train)
    return held[:N_HELD], train[:N_TRAIN]


def roster_str(terms, budget=180):
    out, n = [], 0
    for t in terms:
        n += len(t.split()) + 1
        if n > budget:
            break
        out.append(t)
    return ", ".join(out)


def run_arm(rows, field, tag, detail):
    """Post-edit `field` of every row; return raw outputs keyed by utterance_id."""
    outs = {}
    t0 = time.time()
    for i, d in enumerate(rows, 1):
        src = d[field]
        if DRY:
            outs[d["utterance_id"]] = src
            continue
        try:
            outs[d["utterance_id"]] = generate(SYSTEM, user_prompt(src, roster_str(d["_roster"])),
                                               backend="claude", model=MODEL)
        except Exception as e:
            log(f"  {tag} [{i}] call failed: {str(e)[:80]}")
            outs[d["utterance_id"]] = ""          # an empty output is a gate reject
        if i % 25 == 0:
            log(f"  {tag}: {i}/{len(rows)} ({time.time() - t0:.0f}s)")
    detail[tag] = {d["utterance_id"]: {"src": d[field], "ref": d["final_after_text"],
                                       "out": outs[d["utterance_id"]]} for d in rows}
    return outs


def score_arm(rows, field, outs, tag):
    """Score raw / gated / repaired variants of one arm against final_after_text."""
    refs = [d["final_after_text"] for d in rows]
    srcs = [d[field] for d in rows]
    raw = [outs[d["utterance_id"]] or "" for d in rows]
    decisions = [gate(s, o) for s, o in zip(srcs, raw)]
    gated = [o if ok else s for s, o, (ok, _) in zip(srcs, raw, decisions)]
    repaired = [repair(o) for o in raw]
    clusters = [f"{d['city_id']}/{d['meeting_id']}" for d in rows]

    c_src, c_raw = counts(refs, srcs), counts(refs, raw)
    c_gate, c_rep = counts(refs, gated), counts(refs, repaired)
    rejected = [i for i, (ok, _) in enumerate(decisions) if not ok]
    # Was each rejection right? Only answerable with the reference, so this is
    # analysis of the gate, never an input to it.
    saved = sum(1 for i in rejected if c_raw[i][0] > c_src[i][0])
    cost = sum(1 for i in rejected if c_raw[i][0] < c_src[i][0])
    res = {
        "n": len(rows),
        "wer_source": wer(c_src), "wer_raw": wer(c_raw),
        "wer_gated": wer(c_gate), "wer_repaired": wer(c_rep),
        "gate": {
            "rejected": len(rejected),
            "reject_rate": len(rejected) / len(rows),
            "reasons": dict(collections.Counter(
                r for ok, r in decisions if not ok).most_common()),
            "rejections_that_saved_errors": saved,
            "rejections_that_lost_a_good_edit": cost,
            "errors_avoided": sum(c_raw[i][0] - c_src[i][0] for i in rejected),
        },
        "head2head_raw_vs_source": head2head(c_raw, c_src),
        "head2head_gated_vs_source": head2head(c_gate, c_src),
        "utterances_made_worse_by_raw": sum(1 for a, b in zip(c_raw, c_src) if a[0] > b[0]),
        "utterances_made_worse_by_gated": sum(1 for a, b in zip(c_gate, c_src) if a[0] > b[0]),
    }
    if len(set(clusters)) > 1:
        res["ci_gated_minus_source"] = cluster_bootstrap(c_gate, c_src, clusters)
        res["ci_raw_minus_source"] = cluster_bootstrap(c_raw, c_src, clusters)
    log(f"{tag}: source {res['wer_source']:.4f} | raw {res['wer_raw']:.4f} | "
        f"gated {res['wer_gated']:.4f} | repaired {res['wer_repaired']:.4f} | "
        f"rejected {len(rejected)}/{len(rows)}")
    return res


def main():
    held, train = load_sample()
    log(f"sample: held-out {len(held)} | training-city {len(train)} | model {MODEL}"
        + (" | DRY RUN" if DRY else ""))
    detail, results = {}, {"model": MODEL, "gate": {
        "min_ratio": MIN_RATIO, "max_ratio": MAX_RATIO, "max_rel_edit": MAX_REL_EDIT},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    log("=== A_scribe: post-edit the ASR text (the headline config) ===")
    outs = run_arm(held, "initial_before_text", "A_scribe", detail)
    results["A_scribe"] = score_arm(held, "initial_before_text", outs, "A_scribe")

    log("=== B_clean: post-edit text that is ALREADY CORRECT (held-out) ===")
    outs = run_arm(held, "final_after_text", "B_clean_held", detail)
    results["B_clean_held"] = score_arm(held, "final_after_text", outs, "B_clean_held")

    log("=== B_clean_train: same, on training-city utterances (bigger n) ===")
    outs = run_arm(train, "final_after_text", "B_clean_train", detail)
    results["B_clean_train"] = score_arm(train, "final_after_text", outs, "B_clean_train")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    SC.mkdir(parents=True, exist_ok=True)
    (SC / "postedit_gate_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2))
    log(f"aggregates -> {OUT}")
    log(f"per-utterance text (PII) -> {SC / 'postedit_gate_detail.json'} (keep off the repo)")


if __name__ == "__main__":
    main()
