"""Routing report + SUMMARY from the A/B results (runbook Step 6).

Reads data/reports/fix-task-eval/ab_results.jsonl (and optionally
ab_results_segment.jsonl) and writes:
  per_category.csv / .md   — baseline vs glossary, per category
  by_ebclass.md            — task vs user residual breakdown
  routing.md               — per-category routing recommendation
  examples.md              — example wins/losses
  SUMMARY.md               — the pull-back-to-Mac summary

Metric selection:
  formatting categories (accent_tonos, punctuation_capitalization, final_sigma)
  are accent/punct/case fixes that the Greek normaliser deliberately erases, so
  they are scored on SURFACE exact match. All other categories are scored on
  edit-application (was the targeted fix applied) + normalised exact match.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "data" / "reports" / "fix-task-eval"

FORMATTING = {"accent_tonos", "punctuation_capitalization", "final_sigma"}
# categories whose residual failures are most plausibly acoustic/phonetic
PHONETIC_LEANING = {"homophone", "named_entity", "acronym_abbreviation"}

CONTEXT_CATS = {"named_entity", "acronym_abbreviation", "word_boundary", "number_date"}


def _load(path: Path) -> list[dict]:
    from eval.rescore import enrich
    if not path.exists():
        return []
    by_uid = {}
    for l in path.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if "error" in r:
            continue
        by_uid[r["utterance_id"]] = enrich(r)  # keep last good per utterance
    return list(by_uid.values())


def _surface_exact(out: str, gold: str) -> bool:
    return (out or "").strip() == (gold or "").strip()


def _arm_metrics(records: list[dict], arm: str, category: str) -> dict:
    exact_norm = [r[arm]["normalized_exact"] for r in records]
    edit = [r[arm]["edit_application"] for r in records]
    over = [r[arm]["overcorrection"] for r in records]
    surf = [r[arm]["surface_fidelity"] for r in records]
    surf_exact = [_surface_exact(r[arm]["output"], r["gold_final"]) for r in records]
    parse_ok = [r[arm].get("parse_ok", True) for r in records]
    fix_rate = mean(surf_exact) if category in FORMATTING else mean(edit)
    return {
        "n": len(records),
        "fix_rate": fix_rate,
        "norm_exact": mean(exact_norm),
        "edit_application": mean(edit),
        "surface_exact": mean(surf_exact),
        "surface_fidelity": mean(surf),
        "overcorrection": mean(over),
        "parse_ok": mean(parse_ok),
    }


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def _route(cat: str, base: dict, gloss: dict) -> tuple[str, str]:
    best = max(base["fix_rate"], gloss["fix_rate"])
    lift = gloss["fix_rate"] - base["fix_rate"]
    if best >= 0.70:
        rec = "llm_post_correction"
        why = "prompt already fixes it reliably (text context sufficient)"
    elif best < 0.40 and cat in PHONETIC_LEANING:
        rec = "asr_finetune"
        why = "both prompts fail; residual looks acoustic/OOV — keep for ASR finetune"
    elif cat in FORMATTING:
        rec = "rule_based"
        why = "deterministic accent/punct/case cleanup — cheaper as a normaliser"
    else:
        rec = "review"
        why = "neither prompt reliable and not clearly acoustic — needs audit"
    if lift >= 0.05:
        why += f"; glossary helps (+{lift:.2f})"
    elif lift <= -0.05:
        why += f"; glossary hurts ({lift:.2f})"
    return rec, why


def _fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def main() -> None:
    recs = _load(REPORTS / "ab_results.jsonl")
    seg = _load(REPORTS / "ab_results_segment.jsonl")
    if not recs:
        print("no results yet")
        return

    cats = sorted({r["category"] for r in recs})

    # --- per-category table ---
    rows_md = ["| category | n | metric | baseline | glossary | lift | base over | gloss over |",
               "|---|---|---|---|---|---|---|---|"]
    per_cat = {}
    for cat in cats:
        rr = [r for r in recs if r["category"] == cat]
        b = _arm_metrics(rr, "baseline", cat)
        g = _arm_metrics(rr, "glossary", cat)
        per_cat[cat] = {"baseline": b, "glossary": g}
        metric = "surface-exact" if cat in FORMATTING else "edit-applied"
        rows_md.append(
            f"| {cat} | {b['n']} | {metric} | {_fmt_pct(b['fix_rate'])} | "
            f"{_fmt_pct(g['fix_rate'])} | {g['fix_rate']-b['fix_rate']:+.3f} | "
            f"{b['overcorrection']:.3f} | {g['overcorrection']:.3f} |"
        )
    (REPORTS / "per_category.md").write_text("\n".join(rows_md) + "\n")

    # --- by edited_by class (overall fix rate per arm) ---
    eb_md = ["| ebclass | n | base fix | gloss fix | base norm-exact | gloss norm-exact |",
             "|---|---|---|---|---|---|"]
    for eb in sorted({r["ebclass"] for r in recs}):
        rr = [r for r in recs if r["ebclass"] == eb]
        # use edit_application as the cross-category fix proxy
        b = mean([r["baseline"]["edit_application"] for r in rr])
        g = mean([r["glossary"]["edit_application"] for r in rr])
        bn = mean([r["baseline"]["normalized_exact"] for r in rr])
        gn = mean([r["glossary"]["normalized_exact"] for r in rr])
        eb_md.append(f"| {eb} | {len(rr)} | {_fmt_pct(b)} | {_fmt_pct(g)} | "
                     f"{_fmt_pct(bn)} | {_fmt_pct(gn)} |")
    (REPORTS / "by_ebclass.md").write_text("\n".join(eb_md) + "\n")

    # --- routing ---
    route_md = ["| category | best fix | glossary lift | route | rationale |",
                "|---|---|---|---|---|"]
    routing = {}
    for cat in cats:
        b, g = per_cat[cat]["baseline"], per_cat[cat]["glossary"]
        rec, why = _route(cat, b, g)
        routing[cat] = rec
        route_md.append(f"| {cat} | {_fmt_pct(max(b['fix_rate'], g['fix_rate']))} | "
                        f"{g['fix_rate']-b['fix_rate']:+.3f} | **{rec}** | {why} |")
    (REPORTS / "routing.md").write_text("\n".join(route_md) + "\n")

    # --- examples: glossary wins and losses ---
    wins, losses = [], []
    for r in recs:
        d = r["glossary"]["edit_application"] - r["baseline"]["edit_application"]
        if d > 0 and r["glossary_terms"]:
            wins.append((d, r))
        elif d < 0:
            losses.append((d, r))
    wins.sort(key=lambda x: -x[0])
    losses.sort(key=lambda x: x[0])
    ex = ["# Example glossary wins\n"]
    for d, r in wins[:15]:
        ex.append(f"- [{r['category']}] IN: `{r['input_raw'][:80]}`\n"
                  f"  - GOLD: `{r['gold_final'][:80]}`\n"
                  f"  - BASE: `{r['baseline']['output'][:80]}`\n"
                  f"  - GLOSS: `{r['glossary']['output'][:80]}` (terms: {r['glossary_terms'][:6]})")
    ex.append("\n# Example glossary losses\n")
    for d, r in losses[:15]:
        ex.append(f"- [{r['category']}] IN: `{r['input_raw'][:80]}`\n"
                  f"  - GOLD: `{r['gold_final'][:80]}`\n"
                  f"  - BASE: `{r['baseline']['output'][:80]}`\n"
                  f"  - GLOSS: `{r['glossary']['output'][:80]}` (terms: {r['glossary_terms'][:6]})")
    (REPORTS / "examples.md").write_text("\n".join(ex) + "\n")

    # --- overall + segment gap ---
    overall_b = mean([r["baseline"]["edit_application"] for r in recs])
    overall_g = mean([r["glossary"]["edit_application"] for r in recs])
    seg_md = ""
    if seg:
        # match by utterance_id to compare per-utterance vs segment
        seg_by = {r["utterance_id"]: r for r in seg}
        common = [r for r in recs if r["utterance_id"] in seg_by]
        seg_md = "\n## Per-utterance vs segment gap (context categories)\n\n"
        seg_md += "| category | n | per-utt base | seg base | per-utt gloss | seg gloss |\n|---|---|---|---|---|---|\n"
        for cat in sorted({r["category"] for r in common}):
            cu = [r for r in common if r["category"] == cat]
            ub = mean([r["baseline"]["edit_application"] for r in cu])
            ug = mean([r["glossary"]["edit_application"] for r in cu])
            sb = mean([seg_by[r["utterance_id"]]["baseline"]["edit_application"] for r in cu])
            sg = mean([seg_by[r["utterance_id"]]["glossary"]["edit_application"] for r in cu])
            seg_md += f"| {cat} | {len(cu)} | {_fmt_pct(ub)} | {_fmt_pct(sb)} | {_fmt_pct(ug)} | {_fmt_pct(sg)} |\n"

    summary = f"""# Fix-task prompt eval — SUMMARY

Text-only A/B of the OpenCouncil fix-task prompt: **baseline** (verbatim task-v2)
vs **glossary-augmented** (baseline + a per-utterance retrieved glossary block
mined from the training meeting split). Per-utterance cheap pass over a
stratified held-out sample of {len(recs)} corrections, {len(cats)} categories.

**Inference:** on-box `claude -p` (sonnet), tools disabled, system prompt
overridden with the verbatim task prompt. No ANTHROPIC_API_KEY — OAuth via the
Claude Code CLI.

## Headline

- Overall edit-application: baseline {_fmt_pct(overall_b)} → glossary {_fmt_pct(overall_g)} ({overall_g-overall_b:+.3f}).
- Per-category table: [per_category.md](per_category.md)
- Routing recommendation: [routing.md](routing.md)
- task vs user residual: [by_ebclass.md](by_ebclass.md)
- Example wins/losses: [examples.md](examples.md)

## Did the glossary help, per category?

{chr(10).join(f"- **{c}**: baseline {_fmt_pct(per_cat[c]['baseline']['fix_rate'])} → glossary {_fmt_pct(per_cat[c]['glossary']['fix_rate'])} ({per_cat[c]['glossary']['fix_rate']-per_cat[c]['baseline']['fix_rate']:+.3f}); route → {routing[c]}" for c in cats)}

## Prompt-reliable vs ASR-finetune

- **Prompt-reliable (route out of ASR finetune):** {", ".join(c for c in cats if routing[c]=='llm_post_correction') or '—'}
- **Rule-based cleanup:** {", ".join(c for c in cats if routing[c]=='rule_based') or '—'}
- **Keep for ASR finetune (acoustic/OOV):** {", ".join(c for c in cats if routing[c]=='asr_finetune') or '—'}
- **Needs audit/review:** {", ".join(c for c in cats if routing[c]=='review') or '—'}
{seg_md}
## Caveats / open questions

- **No roster/agenda in the CSV.** Production injects a party roster + agenda
  titles; the corrections CSV has neither, so BOTH arms omit them. The A/B
  isolates the glossary lever, but absolute fix rates understate production.
- **Glossary retrieval is fuzzy from the input only** (no oracle). Precision is
  imperfect — some common capitalised words leak into the glossary and into the
  injected block; this can add mild distractor noise to the glossary arm.
- **Categories are a text-only heuristic** (no audio, no NER): person/place/org
  are merged into `named_entity`; verb/noun/article into `morph_grammar`. Treat
  category routing as triage, not ground truth.
- **Formatting categories** (accent/punctuation) are scored on surface exact
  match because the Greek normaliser erases exactly what they fix.
- **`{'segment pass present' if seg else 'segment pass NOT run'}`** — see the per-utterance-vs-segment section{' above' if seg else ' (proxy segment pass pending; per-utterance only so far)'}.
"""
    (REPORTS / "SUMMARY.md").write_text(summary)
    print("wrote report files to", REPORTS)
    print(f"overall edit-application: base {overall_b:.3f} gloss {overall_g:.3f}")


if __name__ == "__main__":
    main()
