"""Build a local review page for the HParl2 alignment filter.

Takes the jsonl written by eval/hparl2_filter.py, picks a stratified sample across
the alignment range (so you audit the gate, not just the easy rows), copies the MP3s
next to a self-contained HTML page and writes it under
~/.cache/oc-public/hparl2/review/.

Nothing goes in git: the page, the audio and the text all live in the cache dir.

Run:
    .venv-eval/bin/python scripts/hparl2_review_page.py --n 40
    cd ~/.cache/oc-public/hparl2/review && python3 -m http.server 8123
    # then open http://localhost:8123/

Keys in the page: space = play/pause, k = keep, x = reject, j/n = next, arrow keys.
Your verdicts persist in localStorage; "Export" downloads them as JSON.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

WORK = Path.home() / ".cache/oc-public/hparl2"
OUT = WORK / "review"


def stratify(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Sample across alignment bands so the gate itself gets audited."""
    import random
    rnd = random.Random(seed)
    bands = [
        ("perfect  1.00", lambda r: r["align"] >= 0.999),
        ("kept  .95-1.0", lambda r: 0.95 <= r["align"] < 0.999),
        ("near  .85-.95", lambda r: 0.85 <= r["align"] < 0.95),
        ("weak  .50-.85", lambda r: 0.50 <= r["align"] < 0.85),
        ("bad     <.50 ", lambda r: r["align"] < 0.50),
    ]
    pools = [(label, [r for r in rows if pred(r)]) for label, pred in bands]
    pools = [(label, pool) for label, pool in pools if pool]
    per = -(-n // max(1, len(pools)))  # ceil, so empty bands don't shrink the sample
    out: list[dict] = []
    for label, pool in pools:
        rnd.shuffle(pool)
        for r in pool[:per]:
            out.append({**r, "band": label})
    rnd.shuffle(out)
    return out[:n]


def render(items: list[dict], stats: dict) -> str:
    data = json.dumps(items, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="el"><head><meta charset="utf-8">
<title>HParl2 alignment review</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #f6f6f4; color: #1a1a1a; }}
 header {{ position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd;
          padding: 10px 16px; display: flex; gap: 18px; align-items: baseline; flex-wrap: wrap; }}
 header b {{ font-size: 16px; }}
 .muted {{ color: #666; font-size: 13px; }}
 main {{ max-width: 900px; margin: 0 auto; padding: 16px; }}
 .card {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 8px;
          padding: 14px 16px; margin-bottom: 14px; }}
 .card.cur {{ border-color: #2b6cb0; box-shadow: 0 0 0 2px #2b6cb022; }}
 .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
 .pill {{ font-size: 12px; padding: 2px 8px; border-radius: 99px; background: #eee; }}
 .keep {{ background: #d8f3dc; color: #14532d; }}
 .drop {{ background: #fde2e2; color: #7f1d1d; }}
 .txt {{ margin: 8px 0 0; }}
 .lbl {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: .04em; }}
 ins {{ background: #d8f3dc; text-decoration: none; }}
 del {{ background: #fde2e2; text-decoration: line-through; }}
 tag {{ background: #fff3bf; padding: 0 3px; border-radius: 3px; font-size: 12px; }}
 audio {{ width: 100%; margin-top: 8px; }}
 button {{ font: inherit; padding: 4px 12px; border-radius: 6px; border: 1px solid #ccc;
           background: #fff; cursor: pointer; }}
 button.on {{ background: #2b6cb0; color: #fff; border-color: #2b6cb0; }}
</style></head><body>
<header>
 <b>HParl2 — έλεγχος ευθυγράμμισης</b>
 <span class="muted">{stats['n_total']} rows filtered · keep@0.95 = {stats['keep_pct']} ·
 pooled WER {stats['wer']}</span>
 <span class="muted" id="progress"></span>
 <button id="export">Export verdicts</button>
 <span class="muted">space = play · k = σωστό · x = λάθος · j = επόμενο</span>
</header>
<main id="list"></main>
<script>
const ITEMS = {data};
const KEY = "hparl2-verdicts";
const V = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let cur = 0;

function esc(s) {{ const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }}
function showTags(s) {{ return esc(s).replace(/&lt;[^&]*?&gt;/g, m => "<tag>" + m + "</tag>"); }}

function diff(a, b) {{
  // token-level LCS diff, ref = a, hyp = b
  const A = a.split(/\\s+/).filter(Boolean), B = b.split(/\\s+/).filter(Boolean);
  const m = A.length, n = B.length;
  const L = Array.from({{length: m + 1}}, () => new Int32Array(n + 1));
  for (let i = m - 1; i >= 0; i--) for (let j = n - 1; j >= 0; j--)
    L[i][j] = A[i].toLowerCase() === B[j].toLowerCase()
      ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  let i = 0, j = 0, out = "";
  while (i < m && j < n) {{
    if (A[i].toLowerCase() === B[j].toLowerCase()) {{ out += esc(A[i]) + " "; i++; j++; }}
    else if (L[i + 1][j] >= L[i][j + 1]) {{ out += "<del>" + esc(A[i]) + "</del> "; i++; }}
    else {{ out += "<ins>" + esc(B[j]) + "</ins> "; j++; }}
  }}
  while (i < m) out += "<del>" + esc(A[i++]) + "</del> ";
  while (j < n) out += "<ins>" + esc(B[j++]) + "</ins> ";
  return out;
}}

function progress() {{
  const done = Object.keys(V).length;
  const ok = Object.values(V).filter(x => x === "ok").length;
  document.getElementById("progress").textContent =
    `αξιολογημένα ${{done}}/${{ITEMS.length}} · συμφωνώ ${{ok}}`;
}}

function draw() {{
  const list = document.getElementById("list");
  list.innerHTML = "";
  ITEMS.forEach((it, idx) => {{
    const d = document.createElement("div");
    d.className = "card" + (idx === cur ? " cur" : "");
    d.id = "c" + idx;
    const v = V[it.row_id];
    d.innerHTML = `
      <div class="row">
        <span class="pill ${{it.keep ? "keep" : "drop"}}">${{it.keep ? "KEEP" : "DROP"}}</span>
        <span class="pill">align ${{it.align.toFixed(2)}}</span>
        <span class="pill">${{esc(it.band)}}</span>
        <span class="muted">${{it.dur}}s · S${{it.sub}} D${{it.del}} I${{it.ins}} ·
          ${{esc(it.row_id)}}</span>
        <span style="flex:1"></span>
        <button class="okb ${{v === "ok" ? "on" : ""}}" data-i="${{idx}}">σωστό (k)</button>
        <button class="nob ${{v === "no" ? "on" : ""}}" data-i="${{idx}}">λάθος (x)</button>
      </div>
      <audio controls preload="none" src="clips/${{encodeURIComponent(it.row_id)}}.mp3"></audio>
      <p class="txt"><span class="lbl">dataset</span><br>${{showTags(it.transcription_raw)}}</p>
      <p class="txt"><span class="lbl">soniox</span><br>${{esc(it.soniox_text)}}</p>
      <p class="txt"><span class="lbl">diff (del = μόνο στο dataset, ins = μόνο στο soniox)</span><br>
        ${{diff(it.ref, it.soniox_text)}}</p>
      ${{it.text_llm ? `
      <p class="txt"><span class="lbl">στίξη — μηχανική μεταφορά</span><br>${{esc(it.text_transfer)}}</p>
      <p class="txt"><span class="lbl">στίξη — luna ${{it.complete ? "(ολοκληρωμένη πρόταση)" : "(κομμένη)"}}</span><br>
        <b>${{esc(it.text_llm)}}</b></p>` : ""}}`;
    list.appendChild(d);
  }});
  list.querySelectorAll(".okb").forEach(b => b.onclick = () => mark(+b.dataset.i, "ok"));
  list.querySelectorAll(".nob").forEach(b => b.onclick = () => mark(+b.dataset.i, "no"));
  progress();
}}

function mark(i, v) {{
  const id = ITEMS[i].row_id;
  if (V[id] === v) delete V[id]; else V[id] = v;
  localStorage.setItem(KEY, JSON.stringify(V));
  cur = Math.min(ITEMS.length - 1, i + 1);
  draw();
  document.getElementById("c" + cur).scrollIntoView({{block: "center", behavior: "smooth"}});
}}

document.addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT") return;
  const a = document.querySelector("#c" + cur + " audio");
  if (e.key === " ") {{ e.preventDefault(); a && (a.paused ? a.play() : a.pause()); }}
  else if (e.key === "k") mark(cur, "ok");
  else if (e.key === "x") mark(cur, "no");
  else if (e.key === "j" || e.key === "ArrowDown") {{ cur = Math.min(ITEMS.length - 1, cur + 1); draw();
    document.getElementById("c" + cur).scrollIntoView({{block: "center"}}); }}
  else if (e.key === "ArrowUp") {{ cur = Math.max(0, cur - 1); draw();
    document.getElementById("c" + cur).scrollIntoView({{block: "center"}}); }}
}});

document.getElementById("export").onclick = () => {{
  // export EVERY verdict in localStorage, not just this page's items — earlier
  // review rounds put row_ids here that this page no longer lists.
  const byId = Object.fromEntries(ITEMS.map(it => [it.row_id, it]));
  const ids = new Set([...Object.keys(V), ...Object.keys(byId)]);
  const out = [...ids].map(id => ({{row_id: id,
                                   align: byId[id] ? byId[id].align : null,
                                   keep: byId[id] ? byId[id].keep : null,
                                   verdict: V[id] || null}}));
  const b = new Blob([JSON.stringify(out, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "hparl2-verdicts.json"; a.click();
}};

draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(WORK / "filtered.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only-kept", action="store_true",
                    help="review only rows that passed the filter (phase 2: punctuation)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.jsonl).read_text(encoding="utf-8").splitlines()]
    ok = [r for r in rows if r.get("asr_ok")]
    if not ok:
        raise SystemExit("no transcribed rows in " + args.jsonl)

    if args.only_kept:
        ok = [r for r in ok if r.get('keep')]
    items = stratify(ok, args.n, args.seed)
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    for it in items:
        src = Path(it["mp3"])
        dst = OUT / "clips" / f"{it['row_id']}.mp3"
        if not dst.exists():
            shutil.copy2(src, dst)

    nref = sum(r["n_ref"] for r in ok)
    stats = {
        "n_total": len(ok),
        "keep_pct": f"{sum(1 for r in ok if r['keep']) / len(ok):.0%}",
        "wer": f"{sum(r['sub'] + r['del'] + r['ins'] for r in ok) / nref:.3f}",
    }
    punct = {}
    pj = WORK / "punctuated.jsonl"
    if pj.exists():
        for line in pj.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            punct[r["row_id"]] = r

    slim = []
    for it in items:
        d = {k: it[k] for k in ("row_id", "align", "keep", "dur", "sub", "del", "ins",
                                "band", "transcription_raw", "ref", "soniox_text")}
        p = punct.get(it["row_id"])
        if p:
            d.update({"text_transfer": p.get("text_transfer"),
                      "text_llm": p.get("text_llm"),
                      "complete": p.get("complete")})
        slim.append(d)
    (OUT / "index.html").write_text(render(slim, stats), encoding="utf-8")
    print(f"[review] {len(items)} items -> {OUT/'index.html'}")
    print(f"[review] serve:  cd {OUT} && python3 -m http.server 8123")


if __name__ == "__main__":
    main()
