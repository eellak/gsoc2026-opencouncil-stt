#!/usr/bin/env python3
"""What ARE the errors inside 2-of-3 token majorities? A taxonomy, not an arm.

`exp-2026-08-16-char-vote-homophones` decomposed the 5.30-point gap between W
(0.1005) and the alignment-conditional column oracle (0.0475). The single largest
class is the `exact_2_of_3` column `[x, x, y]` where the two agreeing systems are
jointly wrong: a hindsight replay there closes 25.0% of the gap, twice everything the
unresolved columns hold. Three arms that tried to override such majorities all made
WER worse. Nobody had ever looked at WHAT the errors are. This script looks.

IT IS DESCRIPTIVE. Every outcome, gain and replay here is computed WITH the reference
text. None of it is an achievable gain, no threshold is fitted, nothing is promoted.

CORRECTED 2026-08-17 (exp-2026-08-17-confirmation-audit). This docstring used to end
"and the autoresearch confirmation partition is untouched." That was WRONG in the sense
that matters. `load_substrate()` below has no city filter: all 247 windows and all 10
cities enter every count here, so 27,665 of the 74,917 reference tokens belong to the
four sealed confirmation cities. The harness API was never called - that much is true -
but the confirmation labels ARE in these numbers, and a hypothesis selected by reading
them cannot afterwards be confirmed on that partition.
See docs/reports/2026-08-17-confirmation-audit.md.

`msa.py` IS NOT TOUCHED. `fusion_lab._cache_path()` keys the 9 MB alignment cache on
the sha256 of `msa.py`, so adding an attribution helper there silently invalidates it
for every other run. The per-column attribution this script needs is therefore
recomputed locally from the same DP's forward/backward tables, and `msa.oracle_select`
is used unchanged for the oracle's own per-column choice.

THREE AXES, kept separate (Codex job 95b03e7c required this: the first draft folded
"what kind of token is this" and "what relation holds between the two tokens" into one
first-match partition, which discards one of the two dimensions).

  OUTCOME - what went wrong, read off the OPTIMAL-SUPPORT SET of the oracle DP rather
  than off one backtrace, because on a column whose candidates are all wrong the
  backtrace is a tie-break:
    correct     the majority token can exactly match a reference token on some
                optimal path
    selection   it cannot, but the MINORITY token can: the right word was in the
                column and lost the vote. Recoverable by a better voting rule.
    coverage    every optimal path makes the majority a SUBSTITUTION for a reference
                word neither system proposed. No voting rule can fix this.
    spurious    every optimal path makes the majority an INSERTION.
    ambiguous   both explanations are optimal. Substituting x for z and deleting z
                while inserting x cost exactly the same, so the label would otherwise
                be decided by tie-break order, not by the data.

  RELATION - the string/phonemic relation between the majority token and the target
  (the reference word it should have been), a partition.

  TYPE FLAGS - cross-cutting properties of either side: entity, numeric, protocol,
  function word, plus the pair role (function<->function, function<->content, ...).

RECOVERABILITY IS PRICED THREE WAYS, never merged:
  g_oracle  regret inside the oracle lattice: can the majority token participate in
            SOME globally optimal oracle path? Other columns stay free.
  g_W       the decision-relevant one: with every other column frozen to what W
            emitted, does changing this one column alone reduce the edit distance?
  replay    the joint hindsight effect of changing a whole named set of columns,
            scored by the frozen scorer.

Writes results_majority_taxonomy.json (counts and class labels only). The verbatim
column pairs go to $SC/majority_taxonomy/ and never into git.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.controlled_eval.column_classes import (                      # noqa: E402
    column_class, split_merge_columns,
)
from eval.controlled_eval.exp_fusion_deletions import rates, sdi       # noqa: E402
from eval.controlled_eval.fusion_lab import load_substrate, log, sc    # noqa: E402
from eval.controlled_eval.greek_phonetics import phon                  # noqa: E402
from eval.controlled_eval.msa import (                                 # noqa: E402
    oracle_columns, oracle_select,
)
from eval.controlled_eval.scoring import edist, norm                   # noqa: E402

OUT = Path(__file__).with_name("results_majority_taxonomy.json")

# --------------------------------------------------------------------- lexicons
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

PROTOCOL_STEMS = ("αρθρ", "παραγραφ", "εδαφι", "περιπτωσ", "κεφαλαι")
PROTOCOL_WORDS = set("""
φεκ αδα αδαμ καε αφμ ααδε κυα πδ νδ εγκυκλιος εγκυκλιο πρωτοκολλο πρωτοκολλου
κωδικος κωδικο κωδικοι αριθμος αριθμο αριθμου αριθμ
""".split())

RELATIONS = ("alignment_artifact", "no_target", "phonemic_key_equivalent",
             "surface_suffix_neighbor", "unrelated_substitution")

# the reader-facing partition the ticket asked for, derived from relation + flags
CLASSES = ("alignment_artifact", "numeric_identifier", "protocol_legal",
           "named_entity", "orthography_homophone", "morphology_suffix_neighbor",
           "function_word_pair", "different_content_word", "no_target")


def load_term_lexicons():
    """Frozen per-city term aliases: (union set, per-city sets, form -> klass)."""
    tdir = ROOT / "research/ds_wer/terms"
    cities = [p.stem for p in sorted(tdir.glob("*.json")) if ".v2" not in p.name]
    per_city: dict[str, set[str]] = {}
    klass: dict[str, str] = {}
    for c in cities:
        d = json.loads((tdir / f"{c}.json").read_text())
        s: set[str] = set()
        for t in d["terms"]:
            for f in list(t.get("aliases", [])) + [t.get("canonical", "")]:
                nf = norm(f)
                if not nf:
                    continue
                s.add(nf)
                klass.setdefault(nf, t["klass"])
                for part in nf.split():
                    if len(part) >= 4:
                        s.add(part)
                        klass.setdefault(part, t["klass"])
        per_city[c] = s
    union: set[str] = set()
    for s in per_city.values():
        union |= s
    return union, per_city, klass


def load_meeting_contexts(sub):
    """Per (city, meeting), the RosterContext the frozen name-repair rule would use.

    This is the roster-GATED list of `roster_lexicon.build_meeting_context`: a person
    surname is admitted only when the meeting's roster puts that person in the room.
    A term can therefore sit in the frozen city file and be invisible here, which is
    one candidate mechanical answer to the SCHOINA question.
    """
    try:
        from eval.controlled_eval import roster_lexicon as RL
        city_terms = RL.load_city_terms()
        rosters = RL.load_rosters()
        mined, _ = RL.admitted_mined()
    except Exception as exc:                       # pragma: no cover - optional inputs
        log(f"roster lexicon unavailable: {exc}")
        return {}
    out: dict[tuple[str, str], object] = {}
    for w in sub.windows:
        key = (w.city, w.meeting)
        if key in out or w.city not in city_terms:
            continue
        try:
            ctx, _ = RL.build_meeting_context(w.city, w.meeting, city_terms[w.city],
                                              mined.get(w.city, []), rosters,
                                              Counter())
            out[key] = ctx
        except Exception as exc:                   # pragma: no cover
            log(f"roster context failed for {key}: {exc}")
    return out


# ---------------------------------------------------------------- type flags
def is_numeric(tok: str) -> bool:
    if any(ch.isdigit() for ch in tok):
        return True
    if tok in NUMBER_WORDS or tok in MONTHS:
        return True
    return any(tok.startswith(s) and len(tok) - len(s) <= 3 for s in ORDINAL_STEMS)


def is_protocol(tok: str) -> bool:
    if tok in PROTOCOL_WORDS:
        return True
    return any(tok.startswith(s) and len(tok) - len(s) <= 3 for s in PROTOCOL_STEMS)


def surface_suffix_neighbor(a: str, b: str) -> bool:
    """String SHAPE, not linguistic truth: a long shared prefix and two short tails.

    Deliberately not called `same_lemma`. In Greek it will both over-merge
    derivationally related words and miss real inflection with stem alternation.
    """
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


def relation_of(maj: str, target: str | None, artifact: bool) -> str:
    if artifact:
        return "alignment_artifact"
    if target is None:
        return "no_target"
    if phon(maj) == phon(target):
        return "phonemic_key_equivalent"
    if surface_suffix_neighbor(maj, target):
        return "surface_suffix_neighbor"
    return "unrelated_substitution"


def class_of(rel: str, flags: dict) -> str:
    """The reader-facing partition, derived from the relation and the type flags."""
    if rel == "alignment_artifact":
        return "alignment_artifact"
    if flags["any_numeric"]:
        return "numeric_identifier"
    if flags["any_protocol"]:
        return "protocol_legal"
    if flags["any_entity"]:
        return "named_entity"
    if rel == "no_target":
        return "no_target"
    if rel == "phonemic_key_equivalent":
        return "orthography_homophone"
    if rel == "surface_suffix_neighbor":
        return "morphology_suffix_neighbor"
    if flags["pair_role"] == "function_function":
        return "function_word_pair"
    return "different_content_word"


# ---------------------------------------------------- oracle-lattice support sets
def _cand_list(col):
    seen, cl = set(), []
    if any(e is None for e in col):
        cl.append(None)
    for e in col:
        if e is not None and e not in seen:
            seen.add(e)
            cl.append(e)
    return cl


def oracle_tables(cols, ref):
    """Forward/backward tables of the `msa.oracle_select` DP, plus its optimal cost.

    The recurrence is copied from `msa.oracle_select` deliberately rather than
    imported: `fusion_lab` hashes `msa.py` into its alignment cache key, so that file
    stays byte-identical. `test_majority_taxonomy.py` asserts the two agree.
    """
    n, m = len(cols), len(ref)
    cands = [_cand_list(c) for c in cols]
    has_eps = [bool(cl) and cl[0] is None for cl in cands]

    F = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        F[0][j] = j
    for i in range(1, n + 1):
        cl, Fi, Fp = cands[i - 1], F[i], F[i - 1]
        Fi[0] = Fp[0] + (0 if has_eps[i - 1] else 1)
        for j in range(1, m + 1):
            best = Fi[j - 1] + 1
            rj = ref[j - 1]
            for e in cl:
                v = Fp[j] if e is None else min(Fp[j - 1] + (0 if e == rj else 1),
                                                Fp[j] + 1)
                if v < best:
                    best = v
            Fi[j] = best

    B = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        B[n][j] = m - j
    for i in range(n - 1, -1, -1):
        cl, Bi, Bn = cands[i], B[i], B[i + 1]
        Bi[m] = Bn[m] + (0 if has_eps[i] else 1)
        for j in range(m - 1, -1, -1):
            best = Bi[j + 1] + 1
            rj = ref[j]
            for e in cl:
                v = Bn[j] if e is None else min(Bn[j + 1] + (0 if e == rj else 1),
                                                Bn[j] + 1)
                if v < best:
                    best = v
            Bi[j] = best
    return F, B, F[n][m]


def support(F, B, ref, i, e, star):
    """Which explanations of candidate `e` at column `i` lie on an OPTIMAL path."""
    m = len(ref)
    Fi, Bn = F[i], B[i + 1]
    exact, subs, ins = [], [], False
    for j in range(m + 1):
        if Fi[j] + 1 + Bn[j] == star:
            ins = True
        if j < m:
            c = Fi[j] + (0 if e == ref[j] else 1) + Bn[j + 1]
            if c == star:
                (exact if e == ref[j] else subs).append(j)
    return {"exact": exact, "sub": subs, "ins": ins,
            "any": bool(exact or subs or ins)}


# --------------------------------------------- single-column recoverability on W
def w_tables(hyp, ref):
    n, m = len(hyp), len(ref)
    F = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        F[0][j] = j
    for i in range(1, n + 1):
        F[i][0] = i
        for j in range(1, m + 1):
            F[i][j] = min(F[i - 1][j] + 1, F[i][j - 1] + 1,
                          F[i - 1][j - 1] + (0 if hyp[i - 1] == ref[j - 1] else 1))
    B = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        B[n][j] = m - j
    for i in range(n - 1, -1, -1):
        B[i][m] = n - i
        for j in range(m - 1, -1, -1):
            B[i][j] = min(B[i + 1][j] + 1, B[i][j + 1] + 1,
                          B[i + 1][j + 1] + (0 if hyp[i] == ref[j] else 1))
    return F, B, F[n][m]


def w_cost_if(F, B, ref, p, e):
    """Edit distance of W with the token at position p replaced by `e` (None = drop)."""
    m = len(ref)
    Fi, Bn = F[p], B[p + 1]
    if e is None:
        return min(Fi[j] + Bn[j] for j in range(m + 1))
    best = min(Fi[j] + 1 + Bn[j] for j in range(m + 1))
    for j in range(m):
        v = Fi[j] + (0 if e == ref[j] else 1) + Bn[j + 1]
        if v < best:
            best = v
    return best


# -------------------------------------------------------------------------- main
def main():
    sub = load_substrate()
    log(json.dumps(sub.meta, indent=1))
    terms_union, terms_city, term_klass = load_term_lexicons()
    contexts = load_meeting_contexts(sub)
    log(f"term lexicon: {len(terms_union)} forms, {len(contexts)} meeting contexts")

    from serving_stack.name_repair import rnorm, select as nr_select

    rows = []
    oracle_choice: dict[str, dict[int, str | None]] = defaultdict(dict)
    n_maj_total = 0
    inv = Counter()
    for w in sub.windows:
        oa = oracle_select(w.cols, w.ref)
        sm = split_merge_columns(w.cols)
        F, B, star = oracle_tables(w.cols, w.ref)
        WF, WB, wed = w_tables(w.w_tokens, w.ref)
        pos, p = {}, 0                       # position in w_tokens of each column
        for d in w.decisions:
            if d["token"] is not None:
                pos[d["col"]] = p
                p += 1
        city_terms = terms_city.get(w.city, set())
        ctx = contexts.get((w.city, w.meeting))
        raw_hyp = " ".join(w.w_tokens)
        for i, col in enumerate(w.cols):
            if column_class(col) != "exact_2_of_3":
                continue
            n_maj_total += 1
            maj = w.decisions[i]["token"]
            minority = next(t for t in col if t is not None and t != maj)
            oracle_choice[w.item_id][i] = oa[i]

            sup_maj = support(F, B, w.ref, i, maj, star)
            sup_min = support(F, B, w.ref, i, minority, star)
            if not (sup_maj["any"] or sup_min["any"]):
                inv["no_supported_candidate"] += 1
            g_oracle = 0 if sup_maj["any"] else 1

            pp = pos[i]
            if w_cost_if(WF, WB, w.ref, pp, maj) != wed:
                inv["w_cost_mismatch"] += 1
            alt_e = w_cost_if(WF, WB, w.ref, pp, minority)
            drop_e = w_cost_if(WF, WB, w.ref, pp, None)
            g_w = wed - min(wed, alt_e)
            g_w_drop = wed - min(wed, drop_e)

            if sup_maj["exact"]:
                continue                                     # majority is right
            if sup_min["exact"]:
                outcome, target = "selection", minority
            elif sup_maj["sub"] and sup_maj["ins"]:
                outcome = "ambiguous"
                target = w.ref[sup_maj["sub"][0]]
            elif sup_maj["sub"]:
                outcome = "coverage"
                target = w.ref[sup_maj["sub"][0]]
            elif sup_maj["ins"]:
                outcome, target = "spurious", None
            else:
                outcome, target = "unsupported", None
                inv["majority_unsupported"] += 1

            targets = {w.ref[j] for j in sup_maj["sub"]} if sup_maj["sub"] else set()
            artifact = i in sm
            rel = relation_of(maj, target, artifact)
            sides = [t for t in (maj, target) if t]
            ent_forms = [t for t in sides if t in terms_union]
            flags = {
                "any_numeric": any(is_numeric(t) for t in sides),
                "any_protocol": any(is_protocol(t) for t in sides),
                "any_entity": bool(ent_forms),
                "pair_role": ("function_function"
                              if target and maj in FUNCTION_WORDS
                              and target in FUNCTION_WORDS
                              else "function_content"
                              if target and (maj in FUNCTION_WORDS)
                              != (target in FUNCTION_WORDS)
                              else "content_content" if target else "no_target"),
            }
            klass = class_of(rel, flags)

            # ---- the name-repair funnel, on the TARGET (what was actually said)
            funnel = None
            if target and target in terms_union:
                decision = None
                if ctx is not None:
                    decision = nr_select(rnorm(maj), ctx, raw_hyp)["decision"]
                funnel = {
                    "target_in_own_city_file": target in city_terms,
                    "target_in_any_city_file": True,
                    "target_admitted_to_meeting_list": bool(ctx) and
                    rnorm(target) in ctx.valid_aliases,
                    "repair_decision_on_majority": decision,
                    "majority_len": len(rnorm(maj)),
                    "term_klass": term_klass.get(target),
                }

            rows.append({
                "item": w.item_id, "meeting": w.meeting, "city": w.city, "col": i,
                "outcome": outcome, "relation": rel, "class": klass,
                "g_oracle": g_oracle, "g_W": g_w, "g_W_drop": g_w_drop,
                "target_ambiguous": len(targets) > 1,
                "w_differs_from_oracle_choice": oa[i] != maj,
                "chardist": edist(list(maj), list(target)) if target else None,
                "maj_len": len(maj), "target_len": len(target) if target else None,
                "loose_homophone": bool(target and phon(maj, loose=True)
                                        == phon(target, loose=True)),
                "flags": flags,
                "maj_is_entity": maj in terms_union,
                "target_is_entity": bool(target and target in terms_union),
                "entity_other_city_only": bool(ent_forms) and not any(
                    t in city_terms for t in ent_forms),
                "funnel": funnel,
                "_maj": maj, "_target": target, "_minority": minority,
            })

    log(f"{n_maj_total} exact_2_of_3 columns, {len(rows)} wrong, invariants={dict(inv)}")

    # ------------------------------------------------------------------ replays
    def score(pick_cols):
        """Frozen-scorer replay: put the oracle's entry into `pick_cols` per window."""
        rws = []
        for w in sub.windows:
            keep = pick_cols.get(w.item_id, {})
            out = []
            for d in w.decisions:
                tok = keep[d["col"]] if d["col"] in keep else d["token"]
                if tok is not None:
                    out.append(tok)
            rws.append(sdi(" ".join(w.ref), " ".join(out)))
        r = rates(rws)
        r["edits"] = sum(a + b + c for a, b, c, _ in rws)
        return r

    def picks(keep):
        out: dict[str, dict[int, str | None]] = defaultdict(dict)
        n = 0
        for r in rows:
            if keep(r):
                out[r["item"]][r["col"]] = oracle_choice[r["item"]][r["col"]]
                n += 1
        return out, n

    base = score({})
    full_oracle = [sdi(" ".join(w.ref), " ".join(oracle_columns(w.cols, w.ref)))
                   for w in sub.windows]
    orc = rates(full_oracle)
    orc["edits"] = sum(a + b + c for a, b, c, _ in full_oracle)
    gap_edits = base["edits"] - orc["edits"]

    def replay(keep):
        pk, n = picks(keep)
        r = score(pk)
        d_edits = base["edits"] - r["edits"]
        return {"n_columns": n, "wer": r["wer"], "delta_wer": r["wer"] - base["wer"],
                "edits_saved": d_edits,
                "standalone_closure_of_gap": d_edits / gap_edits if gap_edits else None,
                "yield_per_column": d_edits / n if n else None,
                "del_rate": r["del_rate"], "ins_rate": r["ins_rate"],
                "sub_rate": r["sub_rate"]}

    replays = {
        "all_wrong_majorities": replay(lambda r: True),
        "recoverable_g_W_ge_1": replay(lambda r: r["g_W"] >= 1),
        "g_W_zero": replay(lambda r: r["g_W"] < 1),
    }
    for k in CLASSES:
        if any(r["class"] == k for r in rows):
            replays[f"class:{k}"] = replay(lambda r, k=k: r["class"] == k)
            replays[f"drop_class:{k}"] = replay(lambda r, k=k: r["class"] != k)
    for o in sorted({r["outcome"] for r in rows}):
        replays[f"outcome:{o}"] = replay(lambda r, o=o: r["outcome"] == o)

    # ------------------------------------------------------------------ summary
    def dom(subset, key):
        c = Counter(r[key] for r in subset)
        if not c:
            return None
        _, n = c.most_common(1)[0]
        return {"n_groups": len(c), "top_share": n / len(subset)}

    def block(subset):
        return {
            "n": len(subset),
            "share_of_wrong_majorities": len(subset) / len(rows) if rows else None,
            "incidence_per_1000_ref_tokens": 1000 * len(subset) / sub.meta["ref_tokens"],
            "outcomes": dict(Counter(r["outcome"] for r in subset)),
            "relations": dict(Counter(r["relation"] for r in subset)),
            "recoverable_g_W_ge_1": sum(1 for r in subset if r["g_W"] >= 1),
            "no_gain_g_W_0": sum(1 for r in subset if r["g_W"] < 1),
            "g_oracle_ge_1": sum(1 for r in subset if r["g_oracle"] >= 1),
            "chardist_1": sum(1 for r in subset if r["chardist"] == 1),
            "chardist_2": sum(1 for r in subset if r["chardist"] == 2),
            "chardist_gt2": sum(1 for r in subset
                                if r["chardist"] is not None and r["chardist"] > 2),
            "loose_homophone": sum(1 for r in subset if r["loose_homophone"]),
            "pair_roles": dict(Counter(r["flags"]["pair_role"] for r in subset)),
            "target_ambiguous": sum(1 for r in subset if r["target_ambiguous"]),
            "meeting_domination": dom(subset, "meeting"),
            "window_domination": dom(subset, "item"),
            "city_domination": dom(subset, "city"),
        }

    ents = [r for r in rows if r["class"] == "named_entity"]
    # the funnel population is EVERY row whose target is a frozen term, including the
    # few the partition assigns to numeric_identifier / protocol_legal first
    with_funnel = [r for r in rows if r["funnel"]]
    admitted = [r for r in with_funnel
                if r["funnel"]["target_admitted_to_meeting_list"]]
    nocand = [r for r in admitted
              if r["funnel"]["repair_decision_on_majority"] == "no_candidate"]
    res = {
        "substrate": sub.meta,
        "hindsight_warning": (
            "Every outcome, gain and replay below is computed WITH the reference text. "
            "None of it is an achievable gain and none may be quoted as one."),
        "invariants": dict(inv),
        "n_exact_2_of_3_columns": n_maj_total,
        "n_wrong": len(rows),
        "n_w_differs_from_oracle_choice": sum(
            1 for r in rows if r["w_differs_from_oracle_choice"]),
        "baseline_W": base, "column_oracle": orc, "gap_edits": gap_edits,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "outcome_recoverability": {
            o: {"g_W_ge_1": sum(1 for r in rows if r["outcome"] == o and r["g_W"] >= 1),
                "g_W_0": sum(1 for r in rows if r["outcome"] == o and r["g_W"] < 1),
                "g_oracle_ge_1": sum(1 for r in rows
                                     if r["outcome"] == o and r["g_oracle"] >= 1)}
            for o in sorted({r["outcome"] for r in rows})},
        "relations": {k: block([r for r in rows if r["relation"] == k])
                      for k in RELATIONS},
        "classes": {k: block([r for r in rows if r["class"] == k]) for k in CLASSES},
        "class_by_outcome": {k: dict(Counter(r["outcome"] for r in rows
                                             if r["class"] == k)) for k in CLASSES},
        "entities": {
            "n": len(ents),
            "majority_side_is_term": sum(1 for r in ents if r["maj_is_entity"]),
            "target_side_is_term": sum(1 for r in ents if r["target_is_entity"]),
            "other_city_only": sum(1 for r in ents if r["entity_other_city_only"]),
            "by_outcome": dict(Counter(r["outcome"] for r in ents)),
            "term_klass": dict(Counter(r["funnel"]["term_klass"] for r in with_funnel)),
            "funnel": {
                "target_is_a_frozen_term": len(with_funnel),
                "target_in_own_city_file": sum(
                    1 for r in with_funnel if r["funnel"]["target_in_own_city_file"]),
                "target_admitted_to_meeting_list": len(admitted),
                "repair_decision_on_majority": dict(Counter(
                    r["funnel"]["repair_decision_on_majority"] for r in with_funnel)),
                "repair_decision_when_admitted": dict(Counter(
                    r["funnel"]["repair_decision_on_majority"] for r in admitted)),
                "admitted_majority_len_lt_6": sum(
                    1 for r in admitted if r["funnel"]["majority_len"] < 6),
                "admitted_chardist": dict(Counter(
                    r["chardist"] for r in admitted)),
                "recoverable_g_W_ge_1": sum(1 for r in with_funnel if r["g_W"] >= 1),
                "admitted_recoverable_g_W_ge_1": sum(
                    1 for r in admitted if r["g_W"] >= 1),
                "admitted_fired": sum(
                    1 for r in admitted
                    if r["funnel"]["repair_decision_on_majority"] == "fire"),
                "admitted_fired_recoverable": sum(
                    1 for r in admitted if r["g_W"] >= 1
                    and r["funnel"]["repair_decision_on_majority"] == "fire"),
                "no_candidate_reason": {
                    "majority_shorter_than_6_chars": sum(
                        1 for r in nocand if r["funnel"]["majority_len"] < 6),
                    "len_6_to_9_and_chardist_gt_1": sum(
                        1 for r in nocand
                        if 6 <= r["funnel"]["majority_len"] <= 9
                        and (r["chardist"] or 99) > 1),
                    "len_ge_10_and_chardist_gt_2": sum(
                        1 for r in nocand
                        if r["funnel"]["majority_len"] >= 10
                        and (r["chardist"] or 99) > 2),
                    "n": len(nocand),
                },
                "class_of_funnel_rows": dict(Counter(r["class"] for r in with_funnel)),
                "distinct_target_terms": len({r["_target"] for r in with_funnel}),
                "top_term_share": (Counter(r["_target"] for r in with_funnel)
                                   .most_common(1)[0][1] / len(with_funnel)
                                   if with_funnel else None),
                "n_meetings": len({r["meeting"] for r in with_funnel}),
                "n_cities": len({r["city"] for r in with_funnel}),
                "caveat": (
                    "nr_select runs on W's normalized token stream, so the "
                    "capital-mid signal of has_signal() is structurally unavailable "
                    "and seen_freq is empty, which makes abstain_common unreachable. "
                    "Both make this funnel OPTIMISTIC about firing."),
            },
        },
        "replays": replays,
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {OUT}")

    ex = sc() / "majority_taxonomy"
    ex.mkdir(parents=True, exist_ok=True)
    (ex / "examples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    log(f"-> {ex / 'examples.json'} ({len(rows)} rows, verbatim, not in git)")

    for k in CLASSES:
        b = res["classes"][k]
        if b["n"]:
            log(f"  {k:26s} n={b['n']:5d} {b['share_of_wrong_majorities']:6.2%} "
                f"g_W>=1 {b['recoverable_g_W_ge_1']:5d}  "
                f"closure {replays[f'class:{k}']['standalone_closure_of_gap']:6.2%}")


if __name__ == "__main__":
    main()
