"""Self-contained HTML report for the fix-task eval.

Reads:
  data/eval/dataset_stats.json
  eval/timeline.json
  data/reports/fix-task-eval/ab_results.jsonl   (+ _segment optional)
Writes:
  data/reports/fix-task-eval/report.html

Regenerable at any time — shows live progress (timeline + how many rows done)
and the latest A/B results. Safe to run mid-experiment.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
REPORTS = ROOT / "data" / "reports" / "fix-task-eval"

FORMATTING = {"accent_tonos", "punctuation_capitalization", "final_sigma"}
PHONETIC_LEANING = {"homophone", "named_entity", "acronym_abbreviation"}


def _mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def _load_jsonl(p: Path):
    from eval.rescore import enrich
    if not p.exists():
        return []
    by_uid = {}
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if "error" not in r:
            by_uid[r["utterance_id"]] = enrich(r)  # keep last good per utterance
    return list(by_uid.values())


def _surface_exact(o, g):
    return (o or "").strip() == (g or "").strip()


def _fix_rate(rr, arm, cat):
    if cat in FORMATTING:
        return _mean([_surface_exact(r[arm]["output"], r["gold_final"]) for r in rr])
    return _mean([r[arm]["edit_application"] for r in rr])


def _route(cat, bf, gf):
    best = max(bf, gf)
    lift = gf - bf
    if best >= 0.70:
        return "llm_post_correction"
    if best < 0.40 and cat in PHONETIC_LEANING:
        return "asr_finetune"
    if cat in FORMATTING:
        return "rule_based"
    return "review"


ROUTE_COLOR = {
    "llm_post_correction": "#1a7f37",
    "rule_based": "#0969da",
    "asr_finetune": "#bc4c00",
    "review": "#6e7781",
}


def _bar(pct, color="#0969da"):
    w = max(0, min(100, pct * 100))
    return (f'<div class="bar"><div class="fill" style="width:{w:.1f}%;'
            f'background:{color}"></div><span>{pct*100:.1f}%</span></div>')


def main() -> None:
    stats = json.loads((EVAL / "dataset_stats.json").read_text()) if (EVAL / "dataset_stats.json").exists() else {}
    timeline = json.loads((ROOT / "eval" / "timeline.json").read_text()) if (ROOT / "eval" / "timeline.json").exists() else []
    recs = _load_jsonl(REPORTS / "ab_results.jsonl")
    seg = _load_jsonl(REPORTS / "ab_results_segment.jsonl")
    n_target = stats.get("split", {}).get("n_eval_meeting_chains")

    cats = sorted({r["category"] for r in recs})
    overall_b = _mean([r["baseline"]["edit_application"] for r in recs]) if recs else 0
    overall_g = _mean([r["glossary"]["edit_application"] for r in recs]) if recs else 0

    # ---- timeline html ----
    kind_color = {"milestone": "#1a7f37", "bugfix": "#bc4c00", "incident": "#cf222e",
                  "validation": "#0969da", "note": "#6e7781"}
    tl = []
    for e in timeline:
        c = kind_color.get(e.get("kind", "note"), "#6e7781")
        tl.append(
            f'<div class="tl-item"><span class="dot" style="background:{c}"></span>'
            f'<div><span class="tl-time">{html.escape(e["time"])}</span> '
            f'<span class="tag" style="background:{c}">{html.escape(e.get("phase",""))} · {html.escape(e.get("kind",""))}</span>'
            f'<div class="tl-text">{html.escape(e["event"])}</div></div></div>'
        )
    timeline_html = "\n".join(tl)

    # ---- per-category table ----
    rows_html = []
    routing = {}
    for cat in cats:
        rr = [r for r in recs if r["category"] == cat]
        bf, gf = _fix_rate(rr, "baseline", cat), _fix_rate(rr, "glossary", cat)
        over_b = _mean([r["baseline"]["overcorrection"] for r in rr])
        over_g = _mean([r["glossary"]["overcorrection"] for r in rr])
        rt = _route(cat, bf, gf)
        routing[cat] = rt
        lift = gf - bf
        lift_c = "#1a7f37" if lift > 0.005 else ("#cf222e" if lift < -0.005 else "#6e7781")
        metric = "surface-exact" if cat in FORMATTING else "edit-applied"
        rows_html.append(
            f"<tr><td><b>{html.escape(cat)}</b><br><span class='muted'>{metric}, n={len(rr)}</span></td>"
            f"<td>{_bar(bf,'#6e7781')}</td><td>{_bar(gf,'#0969da')}</td>"
            f"<td style='color:{lift_c};font-weight:600'>{lift:+.3f}</td>"
            f"<td class='muted'>{over_b:.3f} → {over_g:.3f}</td>"
            f"<td><span class='route' style='background:{ROUTE_COLOR[rt]}'>{rt}</span></td></tr>"
        )
    cat_table = "\n".join(rows_html)

    # ---- by ebclass ----
    eb_html = []
    for eb in sorted({r["ebclass"] for r in recs}):
        rr = [r for r in recs if r["ebclass"] == eb]
        b = _mean([r["baseline"]["edit_application"] for r in rr])
        g = _mean([r["glossary"]["edit_application"] for r in rr])
        eb_html.append(f"<tr><td><b>{eb}</b> <span class='muted'>n={len(rr)}</span></td>"
                       f"<td>{_bar(b,'#6e7781')}</td><td>{_bar(g,'#0969da')}</td></tr>")
    eb_table = "\n".join(eb_html)

    # ---- segment gap ----
    seg_html = ""
    if seg:
        seg_by = {r["utterance_id"]: r for r in seg}
        common = [r for r in recs if r["utterance_id"] in seg_by]
        srows = []
        for cat in sorted({r["category"] for r in common}):
            cu = [r for r in common if r["category"] == cat]
            ub = _mean([r["baseline"]["edit_application"] for r in cu])
            sb = _mean([seg_by[r["utterance_id"]]["baseline"]["edit_application"] for r in cu])
            ug = _mean([r["glossary"]["edit_application"] for r in cu])
            sg = _mean([seg_by[r["utterance_id"]]["glossary"]["edit_application"] for r in cu])
            srows.append(f"<tr><td><b>{cat}</b> <span class='muted'>n={len(cu)}</span></td>"
                         f"<td>{ub*100:.1f}%</td><td>{sb*100:.1f}%</td>"
                         f"<td>{ug*100:.1f}%</td><td>{sg*100:.1f}%</td></tr>")
        seg_html = (
            "<h2>Per-utterance vs segment (context categories)</h2>"
            "<table><thead><tr><th>category</th><th>per-utt base</th><th>seg base</th>"
            "<th>per-utt gloss</th><th>seg gloss</th></tr></thead><tbody>"
            + "\n".join(srows) + "</tbody></table>")

    # ---- examples ----
    wins, losses = [], []
    for r in recs:
        d = r["glossary"]["edit_application"] - r["baseline"]["edit_application"]
        if d > 0 and r.get("glossary_terms"):
            wins.append((d, r))
        elif d < 0:
            losses.append((d, r))
    wins.sort(key=lambda x: -x[0]); losses.sort(key=lambda x: x[0])

    def ex_html(items, n=8):
        out = []
        for _, r in items[:n]:
            out.append(
                f"<div class='ex'><span class='tag' style='background:#6e7781'>{html.escape(r['category'])}</span>"
                f"<div><code class='in'>{html.escape(r['input_raw'][:110])}</code></div>"
                f"<div>gold: <code>{html.escape(r['gold_final'][:110])}</code></div>"
                f"<div>base: <code>{html.escape(r['baseline']['output'][:110])}</code></div>"
                f"<div>gloss: <code>{html.escape(r['glossary']['output'][:110])}</code></div>"
                f"<div class='muted'>terms: {html.escape(', '.join(r.get('glossary_terms', [])[:6]))}</div></div>")
        return "\n".join(out) or "<p class='muted'>none yet</p>"

    # ---- progress ----
    done_pct = (len(recs) / 1000.0) if recs else 0
    gen_time = time.strftime("%Y-%m-%d %H:%M:%S")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fix-task prompt eval — report</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--mut:#8b949e}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 20px 80px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:34px 0 12px;border-bottom:1px solid var(--bd);padding-bottom:6px}}
.sub{{color:var(--mut);margin:0 0 20px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
.kpi{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 18px;min-width:150px}}
.kpi .v{{font-size:26px;font-weight:700}} .kpi .l{{color:var(--mut);font-size:13px}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);border-radius:10px;overflow:hidden}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--bd);vertical-align:middle}}
th{{color:var(--mut);font-weight:600;font-size:13px}}
.muted{{color:var(--mut);font-size:12.5px}}
.bar{{position:relative;background:#21262d;border-radius:5px;height:20px;min-width:120px}}
.bar .fill{{height:100%;border-radius:5px}} .bar span{{position:absolute;right:6px;top:0;font-size:12px;line-height:20px}}
.route{{color:#fff;padding:2px 8px;border-radius:20px;font-size:12px;white-space:nowrap}}
.tag{{color:#fff;padding:1px 7px;border-radius:20px;font-size:11.5px}}
.tl-item{{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid var(--bd)}}
.dot{{width:11px;height:11px;border-radius:50%;margin-top:6px;flex:0 0 auto}}
.tl-time{{color:var(--mut);font-size:12.5px;margin-right:6px}}
.tl-text{{margin-top:3px}}
.ex{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:10px 12px;margin:8px 0}}
.ex code{{background:#21262d;padding:1px 5px;border-radius:4px;font-size:12.5px;word-break:break-word}}
.ex code.in{{color:#ffa657}}
.prog{{background:#21262d;border-radius:6px;height:10px;overflow:hidden;margin-top:6px}}
.prog .f{{height:100%;background:#1a7f37}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} @media(max-width:780px){{.two{{grid-template-columns:1fr}}}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
</style></head><body><div class="wrap">

<h1>Fix-task prompt eval — text-only A/B</h1>
<p class="sub">Baseline (verbatim task-v2 prompt) vs glossary-augmented, on held-out Greek council corrections.
On-box <code>claude -p</code> (sonnet), OAuth. Generated {gen_time}.</p>

<div class="cards">
  <div class="kpi"><div class="v">{stats.get('n_chains',0):,}</div><div class="l">chains ({stats.get('n_rows',0):,} edits)</div></div>
  <div class="kpi"><div class="v">{stats.get('n_meetings',0)}</div><div class="l">meetings · {stats.get('n_cities',0)} cities</div></div>
  <div class="kpi"><div class="v">{stats.get('glossary_global_terms',0):,}</div><div class="l">global glossary terms</div></div>
  <div class="kpi"><div class="v">{stats.get('glossary_per_city_total_terms',0):,}</div><div class="l">per-city terms</div></div>
  <div class="kpi"><div class="v">{len(recs)}/1000</div><div class="l">A/B rows scored<div class="prog"><div class="f" style="width:{done_pct*100:.0f}%"></div></div></div></div>
</div>

<div class="cards">
  <div class="kpi"><div class="v">{overall_b*100:.1f}%</div><div class="l">overall edit-applied · baseline</div></div>
  <div class="kpi"><div class="v">{overall_g*100:.1f}%</div><div class="l">overall edit-applied · glossary</div></div>
  <div class="kpi"><div class="v" style="color:{'#3fb950' if overall_g>=overall_b else '#f85149'}">{(overall_g-overall_b)*100:+.1f} pp</div><div class="l">glossary lift (overall)</div></div>
</div>

<h2>Timeline</h2>
<div class="timeline">{timeline_html}</div>

<h2>Per-category results (baseline vs glossary)</h2>
<p class="muted">Formatting categories scored on surface-exact match (the Greek normaliser erases accents/punctuation, which is exactly what they fix); all others on edit-application of the targeted diff span. Overcorrection = harm rate.</p>
<table><thead><tr><th>category</th><th>baseline</th><th>glossary</th><th>lift</th><th>overcorr.</th><th>route</th></tr></thead>
<tbody>{cat_table or '<tr><td colspan=6 class=muted>no results yet</td></tr>'}</tbody></table>

<div class="two">
<div><h2>By edit provenance</h2>
<table><thead><tr><th>class</th><th>base</th><th>glossary</th></tr></thead><tbody>{eb_table or '<tr><td colspan=3 class=muted>—</td></tr>'}</tbody></table>
<p class="muted">edit-application by who edited: task_only (reproduce the task), user_only / task_then_user (residual the task missed).</p></div>
<div><h2>Routing summary</h2>
<table><thead><tr><th>route</th><th>categories</th></tr></thead><tbody>
{''.join(f"<tr><td><span class='route' style='background:{ROUTE_COLOR[rt]}'>{rt}</span></td><td>{', '.join(c for c in cats if routing[c]==rt) or '—'}</td></tr>" for rt in ['llm_post_correction','rule_based','asr_finetune','review'])}
</tbody></table></div>
</div>

{seg_html}

<h2>Example glossary wins</h2>
{ex_html(wins)}
<h2>Example glossary losses</h2>
{ex_html(losses)}

<h2>Caveats</h2>
<ul class="muted">
<li><b>No roster/agenda in the CSV</b> — production injects a party roster + agenda; both A/B arms omit them, so the A/B isolates the glossary lever but absolute fix rates understate production.</li>
<li><b>Glossary retrieval is fuzzy from the input only</b> (no oracle); some common capitalised words leak in as mild distractor noise.</li>
<li><b>Categories are a text-only heuristic</b> (no audio/NER): person/place/org merged to <code>named_entity</code>, verb/noun/article to <code>morph_grammar</code>. Triage, not ground truth.</li>
<li><b>Segment pass</b> uses a consecutive-same-meeting proxy (no speaker column / cached meeting JSON).</li>
</ul>

</div></body></html>"""
    out = REPORTS / "report.html"
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(recs)} rows, overall base {overall_b:.3f} gloss {overall_g:.3f})")


if __name__ == "__main__":
    main()
