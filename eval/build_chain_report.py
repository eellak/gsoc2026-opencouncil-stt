"""Generate a self-contained public HTML report of the full fix-task experiment
chain. Aggregate metrics + curated narrative only — NO personal data (rosters,
raw transcripts) and no secrets. Reads eval/timeline.json for the timeline.

Output: docs/reports/fix-task-experiment-report.html  (curated committed snapshot;
copy it to the gh-pages branch as index.html to publish).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMELINE = ROOT / "eval" / "timeline.json"
OUT = ROOT / "docs" / "reports" / "fix-task-experiment-report.html"

GENERATED = "2026-06-21"

# ---- curated result tables (from the committed JSON artifacts) ----------------
AB = [  # 1000-row stratified A/B, sonnet, by category: glossary delta (pp)
    ("named_entity", "+9.9"), ("acronym_abbreviation", "-6.5"), ("number_date", "-6.0"),
    ("accent / morph", "-5.9"), ("HIR overall", "+1.9 (worse)"), ("WER overall", "0.153 -> 0.151"),
]
LOOP_VALID = [  # the three independent held-out measurements of the loop winner
    ("Naive loop (dev)", "-16.7pp / -26%", "overfit — vanished on held-out"),
    ("Naive loop, codex held-out (n=140)", "+2.1pp WORSE", "McNemar p=0.61, CI contains 0"),
    ("Naive loop, sonnet transfer (n=140)", "+1.4pp (n.s.)", "p=0.77, WER -0.07"),
    ("Anti-overfit pool, held-out (n=245)", "+2.0pp (n.s.)", "p=0.44, CI [-0.02,+0.06]"),
]
FUZZY = [  # deterministic fuzzy, clean-control (gold_final_retention)
    ("Fuzzy v2 (global glossary)", "0.11", "—", "0.43 (57% corrupted)"),
    ("Fuzzy strict (global)", "0.15", "—", "0.86 (14% corrupted)"),
]
GATE = [  # candidate-set / gate progression (GreekBERT gate)
    ("Global glossary (5894)", "0.34", "10.0", "0.855"),
    ("Per-meeting proxy (~116)", "0.39", "7.6", "0.908"),
    ("REAL per-meeting roster (~146)", "0.42", "1.2", "0.955"),
]
NER = [  # gate model comparison (permissive, gated)
    ("GLiNER (multilingual)", "0.24", "0.20", "0.83", "false-positives common words"),
    ("GreekBERT-NER", "0.34", "0.26", "0.86", "no common-word false positives"),
]


def _rows(rows):
    return "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)


# redact illustrative personal-name examples from the public report
REDACT = [
    ("'Κύριε Σουμπάκη'->person 0.83, 'Κολυμβητική Ομοσπονδία'->org 0.95",
     "a misspelled surname is tagged person, an org name is tagged organization"),
]


def _timeline_html():
    items = json.loads(TIMELINE.read_text())
    kinds = {"milestone": "#2563eb", "validation": "#16a34a", "bugfix": "#d97706",
             "incident": "#dc2626", "note": "#6b7280"}
    out = []
    for e in items:
        color = kinds.get(e.get("kind", "note"), "#6b7280")
        ev = e["event"]
        for src, repl in REDACT:
            ev = ev.replace(src, repl)
        out.append(
            f'<div class="tl"><span class="dot" style="background:{color}"></span>'
            f'<span class="t">{html.escape(e["time"])}</span> '
            f'<span class="ph">{html.escape(e.get("phase",""))}</span> '
            f'<span class="k" style="color:{color}">{html.escape(e.get("kind",""))}</span>'
            f'<div class="ev">{html.escape(ev)}</div></div>')
    return "\n".join(out)


HTML = """<!doctype html><html lang="el"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenCouncil fix-task: experiment report</title>
<style>
:root{{--fg:#1f2937;--mut:#6b7280;--bd:#e5e7eb;--bg:#fff;--accent:#2563eb}}
*{{box-sizing:border-box}}
body{{font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);
max-width:920px;margin:0 auto;padding:24px;background:var(--bg)}}
h1{{font-size:1.7rem;margin:.2em 0}}h2{{font-size:1.3rem;margin-top:1.8em;border-bottom:2px solid var(--bd);padding-bottom:.2em}}
h3{{font-size:1.05rem;margin-top:1.3em}}
.sub{{color:var(--mut);margin-top:0}}
code{{background:#f3f4f6;padding:.1em .35em;border-radius:4px;font-size:.9em}}
table{{border-collapse:collapse;width:100%;margin:.8em 0;font-size:.92rem}}
th,td{{border:1px solid var(--bd);padding:.45em .6em;text-align:left}}
th{{background:#f9fafb}}
.tldr{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:.6em 1.1em;margin:1em 0}}
.win{{color:#16a34a;font-weight:600}}.bad{{color:#dc2626;font-weight:600}}
.tl{{border-left:2px solid var(--bd);padding:.1em 0 .9em 1.1em;position:relative;margin-left:6px}}
.dot{{position:absolute;left:-7px;top:.45em;width:11px;height:11px;border-radius:50%}}
.tl .t{{font-variant-numeric:tabular-nums;color:var(--mut);font-size:.82rem}}
.tl .ph{{font-weight:600;font-size:.82rem}}.tl .k{{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em}}
.tl .ev{{font-size:.9rem;margin-top:.15em}}
footer{{margin-top:3em;color:var(--mut);font-size:.85rem;border-top:1px solid var(--bd);padding-top:1em}}
</style></head><body>

<h1>OpenCouncil Greek ASR — fix-task experiment report</h1>
<p class="sub">Text-only evaluation of the transcript-correction prompt, an autoresearch
improvement loop, and deterministic name correction. Generated {generated}. GSoC 2026.</p>

<div class="tldr">
<strong>TL;DR.</strong> On the hardest residual errors (the ones the production fix-task already
missed once), <strong>prompt engineering is not the lever</strong>: a glossary injected into the
prompt helps only named entities and is net-negative overall; an automated prompt-tuning loop
yields at most a <strong>~2pp, statistically non-significant</strong> reduction in the human-intervention
rate (HIR). Deterministic name correction by fuzzy-matching <em>corrupts clean text</em> unless it is
gated by NER <em>and</em> restricted to the small per-meeting roster — and even then it is a
conservative first-pass, not a replacement for the context-aware LLM. The evidence points to
<strong>ASR fine-tuning</strong> as the real lever.
</div>

<h2>Metrics</h2>
<ul>
<li><strong>HIR (Human Intervention Rate)</strong> — fraction of utterances that would still need a
human edit (= 1 − normalized-exact match to the human-approved final). The primary product metric.</li>
<li><strong>WER</strong> — token-level error rate of the corrected output vs the human final.</li>
<li><strong>edit_application / overcorrection</strong> — fraction of the needed fix applied / harm introduced.</li>
<li><strong>precision / recall</strong> (name correction) — useful fixes / all interventions; entity errors fixed.</li>
<li><strong>gold_final_retention</strong> — fraction of already-clean (human-approved) text left untouched
(a deployment guard; target ≥99%).</li>
</ul>

<h2>Phase 1 — Glossary in the prompt (1000-row A/B, sonnet)</h2>
<p>A per-city + global glossary block injected into the prompt. Verdict: helps named entities,
hurts everything else via retrieval-noise overcorrection, net-negative on HIR.</p>
<table><tr><th>Category / metric</th><th>Glossary effect</th></tr>{ab}</table>
<p><strong>Conclusion:</strong> use the glossary <em>scoped</em> (precise per-utterance retrieval +
entity-only instruction + anti-overcorrection guardrail), never as a global dump.</p>

<h2>Phase 2 — Autoresearch improvement loop</h2>
<p>An LLM-researcher proposes revised system prompts; we keep the best by HIR. The first
(naive) loop overfit the small dev set the proposer could see. We rebuilt it per an adversarial
review: bigger data, a stratified <code>propose/select</code> split with the proposer blind to the
select half, a frozen candidate pool (no iterative holdout reuse), a fixed budget, and a single
held-out test used once.</p>
<table><tr><th>Measurement</th><th>HIR change</th><th>Verdict</th></tr>{loop}</table>
<p><strong>Conclusion:</strong> prompt tuning gives at most a marginal, non-significant HIR
improvement on these residuals. The winning prompt (anti-overcorrection + grounded examples) is a
<span class="win">safe mild upgrade</span> — never worse, slightly lower WER — but not a real lever.</p>

<h2>Phase 3 — Deterministic name correction</h2>
<h3>3a. Fuzzy-match without gating (refuted)</h3>
<p>Snap tokens to the nearest glossary name by edit distance. The clean-control set
(<code>gold_final_retention</code>) immediately exposed it: it corrupts already-correct text.</p>
<table><tr><th>Config</th><th>precision</th><th>—</th><th>clean retention</th></tr>{fuzzy}</table>

<h3>3b. NER gate + candidate set (the hypothesis)</h3>
<p>Gate the corrector on NER entity spans, and shrink the candidate set from the global glossary
to the actual per-meeting roster (which OpenCouncil already has). Both precision and clean
retention rise monotonically as the candidate set gets smaller and cleaner.</p>
<table><tr><th>Candidate set (GreekBERT-gated)</th><th>precision</th><th>collateral/100</th><th>clean retention</th></tr>{gate}</table>
<h3>3c. Gate model: GreekBERT vs GLiNER</h3>
<table><tr><th>Gate</th><th>precision</th><th>recall</th><th>retention</th><th>note</th></tr>{ner}</table>
<p><strong>Conclusion:</strong> the per-meeting roster (not the global glossary) is the right
candidate set, and <strong>GreekBERT-NER</strong> is the better gate. Best deterministic config:
precision 0.42, collateral 1.2/100, retention 0.955 — a viable conservative first-pass, but still
short of standalone deployability (~99% retention) and modest recall. The LLM, which knows when a
name is already correct from context, remains better for standalone correction.</p>

<h2>Recommendations</h2>
<ul>
<li><strong>ASR fine-tuning</strong> is the real lever for the residual errors — not more prompt engineering.</li>
<li>Adopt the loop's winning prompt (anti-overcorrection + grounded examples) as a safe baseline upgrade.</li>
<li>If a glossary is used in the prompt, keep it <strong>scoped + entity-only</strong>.</li>
<li>For names, prefer <strong>NER-gate (GreekBERT) + per-meeting roster</strong> as a cheap first-pass alongside the LLM.</li>
<li>Gate any deployment on a <strong>clean-control</strong> (no-edit-traffic retention ≥99%).</li>
</ul>

<h2>Timeline</h2>
{timeline}

<footer>
Reproducible harness: <code>eval/</code> in the project repo (scoring, chains, glossary, improve_loop,
fuzzy_correct, ner_gate_eval, roster_gate_eval). Aggregate metrics only; no personal data or
transcripts are included in this report.
</footer>
</body></html>"""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = HTML.format(
        generated=GENERATED,
        ab=_rows(AB), loop=_rows(LOOP_VALID), fuzzy=_rows(FUZZY),
        gate=_rows(GATE), ner=_rows(NER), timeline=_timeline_html(),
    )
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
