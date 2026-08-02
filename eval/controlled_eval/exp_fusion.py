#!/usr/bin/env python3
"""If an LLM reading several ASR transcripts beats one ASR, WHAT did the work?

`exp_fusion_headroom.py` established that the systems fail on different windows, and
that a reference-free consensus vote over three of them already beats the best single
provider (0.1319 -> 0.1211, clustered CI [-0.0142, -0.0074]). The obvious next step is
to let an LLM read the hypotheses and write one merged transcript instead of just
picking a winner.

The trap is that such a result is uninterpretable on its own. An LLM handed a SINGLE
transcript also improves it (measured: 0.155 -> 0.114 on corrected utterances), so
"fusion beat Scribe" is equally consistent with the combination doing nothing and the
editor doing everything. The experiment is built around that ambiguity, not around the
headline.

  A_scribe    the best single provider, verbatim                        free
  B_consensus reference-free vote over the trio, no LLM                 free
  C_edit_one  the LLM, given ONE transcript                             LLM, no fusion
  D_fusion    the same LLM, same prompt, given THREE transcripts        the treatment
  F_ctx       D plus the meeting roster                                 opt-in, separate

THE RESULT IS D vs C. Nothing else identifies the effect. C and D share one system
prompt, one output format, one failure policy and one model; the only thing that differs
between them is how many transcripts arrive in the user message. D vs A confounds
combination with editing. D vs B answers a different question (does the LLM beat a free
selector), worth reporting, not the claim.

Three things this file does deliberately, each because the first draft got it wrong:

  * The primary analysis is UNGATED, with the SAME fallback in both arms. An output
    validity gate that falls back to Scribe in C and to the consensus pick in D hands D
    a better floor, so D could win on the fallback alone with the LLM contributing
    nothing. The gated operational pipelines are reported as a secondary analysis, where
    the different fallbacks are the point rather than a confound.
  * C and D are INTERLEAVED per window, so a model or service change mid-run cannot line
    up with an arm.
  * Hypothesis order is a stable hash of the meeting id, fixed before the run, so no
    provider sits in the first slot more often than the others.

Known limitation, stated up front: the trio was chosen by looking at this benchmark's
references (`exp_fusion_headroom.py` tried every subset). The split-half check there
suggests the SELECTION PROCEDURE generalizes, but the specific trio is not confirmed on
untouched audio, so D's absolute number is exploratory. The D-vs-C contrast is what
survives this, since both arms are scored on the same windows with the same LLM.

Writes results_fusion.json (tracked, aggregates only) and per-window text to
$SC/fusion_detail.json, which is NOT tracked: hypotheses and references are verbatim
council speech, the category of PII the 2026-07-21 purge removed from git history.

Env: SC (scratchpad) LLM_MODEL (sonnet) ARMS (C,D | add F) N_ITEMS (0 = all) DRY (1)
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))
from eval.backends import generate                                     # noqa: E402
from eval.controlled_eval import bench_data as B                       # noqa: E402
from eval.controlled_eval.exp_postedit_gate import gate                # noqa: E402
from eval.controlled_eval.scoring import (cluster_bootstrap, counts,   # noqa: E402
                                          head2head, wer)

OUT = Path(__file__).with_name("results_fusion.json")
ROSTERS = ROOT / "data/pii/rosters_full.json"
SC = Path(os.environ.get("SC", "/tmp"))
MODEL = os.environ.get("LLM_MODEL", "sonnet")
DRY = os.environ.get("DRY") == "1"
N_ITEMS = int(os.environ.get("N_ITEMS", "0"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "C,D").split(",") if a.strip()]

# Scribe and Soniox are the two best systems on the benchmark's PUBLISHED leaderboard,
# which predates this experiment. The third slot is the one this project selected, and
# selected on these references — see the module docstring.
TRIO = ("scribe-v2-clean", "soniox", "oc-minipc-finetune")
PRIMARY = "scribe-v2-clean"
LABELS = ("Α", "Β", "Γ")          # neutral labels: the model never sees provider names


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------ the prompt
# ONE system prompt for C and D. It has to read naturally at one version and at three,
# because any wording difference between the arms becomes a rival explanation for the
# effect. The constraints on what may change are lifted from the frozen post-editor
# prompt (exp_postedit_gate.py) so this run stays comparable with that one.
SYSTEM = (
    "Είσαι επιμελητής απομαγνητοφωνήσεων ελληνικών δημοτικών συμβουλίων. Θα λάβεις μία "
    "ή περισσότερες εκδοχές του ΙΔΙΟΥ αποσπάσματος ήχου, από συστήματα αυτόματης "
    "αναγνώρισης ομιλίας (ASR). Καμία εκδοχή δεν είναι εξ ορισμού σωστή: μπορεί να "
    "περιέχουν παραφθαρμένες λέξεις, λάθος ονόματα ή λάθος καταλήξεις. Παρήγαγε ΜΙΑ "
    "τελική εκδοχή του αποσπάσματος, που να αποδίδει όσο γίνεται πιστότερα αυτό που "
    "ειπώθηκε. Όπου οι εκδοχές συμφωνούν, κράτα το κοινό κείμενο αυτούσιο. Όπου "
    "διαφωνούν, διάλεξε ή συνδύασε. Μην παραφράζεις, μην αφαιρείς και μην προσθέτεις "
    "περιεχόμενο, μην αλλάζεις τη σειρά. Αν δεν είσαι βέβαιος, άφησε τη λέξη όπως "
    "είναι. Επίστρεψε ΜΟΝΟ το τελικό κείμενο, χωρίς εισαγωγικά, σχόλια ή εξηγήσεις."
)


def roster_str(terms, budget=180):
    out, n = [], 0
    for t in terms:
        n += len(t.split()) + 1
        if n > budget:
            break
        out.append(t)
    return ", ".join(out)


def user_prompt(texts, roster=""):
    """Identical structure at one version and at three. Only the list length differs."""
    ctx = (f"Ονόματα και όροι που εμφανίζονται σε αυτή τη συνεδρίαση:\n{roster}\n\n"
           if roster else "")
    body = "\n\n".join(f"Εκδοχή {lab}:\n{t}" for lab, t in zip(LABELS, texts))
    return f"{ctx}{body}"


def order_for(item):
    """Provider order for this window: a stable hash of the meeting id.

    Fixed before the run and reproducible, so it cannot be tuned after seeing results,
    and no provider occupies the first slot more often than the others.
    """
    h = int(hashlib.sha256(item["meeting_id"].encode()).hexdigest()[:8], 16)
    k = h % len(TRIO)
    t = list(TRIO)
    return t[k:] + t[:k]


# --------------------------------------------------------------------------- running
def call(user, tag, i):
    """Return raw text, or "" if the call failed. Never raises: a lost call must be
    handled by the same failure policy in every arm, not by aborting one of them."""
    if DRY:
        return ""
    try:
        return generate(SYSTEM, user, backend="claude", model=MODEL)
    except Exception as e:
        log(f"  {tag} [{i}] call failed: {str(e)[:90]}")
        return ""


def run_interleaved(items, arms):
    """Run every requested arm on window i before moving to window i+1.

    Model versions and service quality drift over a run of hundreds of calls. Running
    arm C to completion and then arm D would let that drift align exactly with the
    contrast the experiment exists to measure.
    """
    raw = {tag: [] for tag in arms}
    t0 = time.time()
    for i, it in enumerate(items):
        for tag, build in arms.items():
            raw[tag].append(call(build(it), tag, i))
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(items)} windows, {len(arms)} arms "
                f"({time.time() - t0:.0f}s)")
    return raw


# --------------------------------------------------------------------------- scoring
def failure_fallback(raw, fallback):
    """The SAME policy in every arm: an empty output falls back to the same text.

    This is not the validity gate. It covers API failures and empty responses only, so
    that a dropped call does not silently score as a zero-word transcript.
    """
    return [r.strip() if (r or "").strip() else f for r, f in zip(raw, fallback)]


def contrast(name, c_a, c_b, label_b, items):
    ci = cluster_bootstrap(c_a, c_b, [it["cluster"] for it in items])
    log(f"{name}: {wer(c_a):.4f} vs {label_b} {wer(c_b):.4f} | {ci['delta']:+.4f} "
        f"CI[{ci['ci95'][0]:+.4f},{ci['ci95'][1]:+.4f}]"
        f"{'  SIGNIFICANT' if ci['excludes_zero'] else ''}")
    return {"wer": wer(c_a), "wer_baseline": wer(c_b), "vs": label_b,
            "ci_minus_baseline": ci, "head2head": head2head(c_a, c_b)}


def gate_stats(decisions, c_raw, c_base):
    rej = [i for i, (ok, _) in enumerate(decisions) if not ok]
    return {"rejected": len(rej), "reject_rate": len(rej) / max(1, len(decisions)),
            "reasons": dict(collections.Counter(
                r for ok, r in decisions if not ok).most_common()),
            "rejections_that_saved_errors": sum(1 for i in rej
                                                if c_raw[i][0] > c_base[i][0]),
            "rejections_that_lost_a_good_edit": sum(1 for i in rej
                                                    if c_raw[i][0] < c_base[i][0])}


def robustness(c_a, c_b, items):
    """Per-city effect and leave-one-city-out on the primary contrast.

    Ten cities is a thin basis for any claim about council audio in general. This does
    not fix that; it shows whether one city is carrying the whole result.
    """
    cities = sorted({it["city_id"] for it in items})
    per_city, loco = {}, {}
    for c in cities:
        inn = [i for i, it in enumerate(items) if it["city_id"] == c]
        out = [i for i, it in enumerate(items) if it["city_id"] != c]
        if inn:
            per_city[c] = {"n": len(inn),
                           "delta": wer([c_a[i] for i in inn]) - wer([c_b[i] for i in inn])}
        if len(out) > 20:
            loco[c] = wer([c_a[i] for i in out]) - wer([c_b[i] for i in out])
    return {"per_city_delta": per_city, "leave_one_city_out_delta": loco,
            "worst_loco": max(loco.values()) if loco else None}


def main():
    report = B.load_report()
    items = B.common_items(report, TRIO)
    if N_ITEMS:
        items = items[:N_ITEMS]
    rosters = json.load(open(ROSTERS)) if ROSTERS.exists() else {}
    for it in items:
        it["_roster"] = roster_str(rosters.get(f"{it['city_id']}/{it['meeting_id']}") or [])

    refs = [it["ref"] for it in items]
    scribe = [it["hyp"][PRIMARY] for it in items]
    cons_pick = [B.consensus_pick(it, TRIO) for it in items]
    cons = [it["hyp"][p] for it, p in zip(items, cons_pick)]
    log(f"{len(items)} windows | {len({it['cluster'] for it in items})} meetings | "
        f"{len({it['city_id'] for it in items})} cities | arms {'+'.join(ARMS)} | "
        f"model {MODEL}" + (" | DRY" if DRY else ""))

    builders = {}
    if "C" in ARMS:
        builders["C_edit_one"] = lambda it: user_prompt([it["hyp"][PRIMARY]])
    if "D" in ARMS:
        builders["D_fusion"] = lambda it: user_prompt(
            [it["hyp"][p] for p in order_for(it)])
    if "F" in ARMS:
        builders["F_ctx"] = lambda it: user_prompt(
            [it["hyp"][p] for p in order_for(it)], it["_roster"])

    raw = run_interleaved(items, builders) if builders else {}

    c_scribe, c_cons = counts(refs, scribe), counts(refs, cons)
    results = {
        "model": MODEL, "trio": list(TRIO), "primary_provider": PRIMARY,
        "n_items": len(items), "n_meetings": len({it["cluster"] for it in items}),
        "arms": list(builders), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trio_selected_on_this_benchmark": True,
        "consensus_picks": dict(collections.Counter(cons_pick)),
        "A_scribe": {"wer": wer(c_scribe)},
        "B_consensus": contrast("B_consensus", c_cons, c_scribe, "A_scribe", items),
    }
    log(f"A_scribe: {wer(c_scribe):.4f}")

    # ---- primary: ungated, identical failure fallback (Scribe) in every LLM arm
    ung = {tag: counts(refs, failure_fallback(r, scribe)) for tag, r in raw.items()}
    results["ungated"] = {tag: {"wer": wer(c),
                                "empty_or_failed": sum(1 for x in raw[tag]
                                                       if not (x or "").strip())}
                          for tag, c in ung.items()}
    for tag, c in ung.items():
        log(f"{tag} (ungated): {wer(c):.4f}")

    if "C_edit_one" in ung and "D_fusion" in ung:
        results["PRIMARY_D_vs_C"] = contrast(
            "PRIMARY  D_fusion vs C_edit_one (ungated, same fallback)",
            ung["D_fusion"], ung["C_edit_one"], "C_edit_one", items)
        results["PRIMARY_D_vs_C"]["robustness"] = robustness(
            ung["D_fusion"], ung["C_edit_one"], items)
        w = results["PRIMARY_D_vs_C"]["robustness"]["worst_loco"]
        log(f"  leave-one-city-out worst delta: {w:+.4f}"
            if w is not None else "  (no LOCO)")
    if "F_ctx" in ung and "D_fusion" in ung:
        results["F_vs_D"] = contrast("F_ctx vs D_fusion (ungated)",
                                     ung["F_ctx"], ung["D_fusion"], "D_fusion", items)

    # ---- secondary: the operational pipelines, each with its own natural fallback
    OPERATIONAL = {"C_edit_one": (scribe, c_scribe, "A_scribe"),
                   "D_fusion": (cons, c_cons, "B_consensus"),
                   "F_ctx": (cons, c_cons, "B_consensus")}
    results["operational_gated"] = {}
    for tag, r in raw.items():
        base_txt, c_base, base_name = OPERATIONAL[tag]
        dec = [gate(s, o) for s, o in zip(base_txt, r)]
        gated = [o if ok else s for s, o, (ok, _) in zip(base_txt, r, dec)]
        c_g = counts(refs, gated)
        results["operational_gated"][tag] = contrast(
            f"{tag} (gated, falls back to {base_name})", c_g, c_base, base_name, items)
        results["operational_gated"][tag]["gate"] = gate_stats(
            dec, counts(refs, r), c_base)
        results["operational_gated"][tag]["note"] = (
            "Different arms fall back to different text on purpose here. This is the "
            "deployable pipeline, NOT the identifying contrast.")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    SC.mkdir(parents=True, exist_ok=True)
    (SC / "fusion_detail.json").write_text(json.dumps(
        {"items": [{"item_id": it["item_id"], "ref": it["ref"], "hyp": it["hyp"],
                    "consensus": p, "order": order_for(it)}
                   for it, p in zip(items, cons_pick)],
         "raw": raw}, ensure_ascii=False, indent=2))
    log(f"aggregates -> {OUT}")
    log(f"per-window text (PII) -> {SC / 'fusion_detail.json'} (keep off the repo)")


if __name__ == "__main__":
    main()
