"""Blind calibration page for the HParl2 alignment gate.

The first calibration said `align >= 0.95` throws away usable audio at roughly a 1:1
rate with what it keeps — but the page showed the alignment score, the KEEP/DROP pill
and the diff, so the judgement was not independent of the thing being judged.

This page shows **the audio and the candidate label, and nothing else**. No score, no
verdict, no band, no Soniox text, no diff. One question per clip:

    Λέει ο ήχος αυτό ακριβώς το κείμενο;   ΝΑΙ / ΟΧΙ / ΔΕΝ ΞΕΡΩ

Sampling is stratified over the bands the decision actually turns on, plus **hidden
controls** at both extremes: rows above 0.99 should come back almost all ΝΑΙ and rows
below 0.40 almost all ΟΧΙ. If they do not, the round is not measuring what it claims
and the answers must not be used to move the gate.

The key map (row_id -> band, align) is written OUTSIDE the served directory, so it is
neither in the page nor reachable over the local HTTP server. Verdicts live under their own localStorage key so they cannot mix with the
earlier, non-blind round.

Run:
    .venv-eval/bin/python scripts/hparl2_blind_page.py
    cd ~/.cache/oc-public/hparl2/blind && python3 -m http.server 8124
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

WORK = Path.home() / ".cache/oc-public/hparl2"
OUT = WORK / "blind"
KEY_PATH = WORK / "blind-key.json"   # deliberately NOT under OUT: that dir is served

# (label, lo, hi, n) — the two decision bands, plus controls at both ends.
STRATA = [
    ("mid-high", 0.80, 0.95, 25),
    ("mid-low", 0.50, 0.80, 25),
    ("control-good", 0.99, 1.01, 6),
    ("control-bad", 0.0, 0.40, 6),
]


def render(items: list[dict]) -> str:
    data = json.dumps([{"row_id": i["row_id"], "text": i["text"], "dur": i["dur"]}
                       for i in items], ensure_ascii=False)
    return f"""<!doctype html>
<html lang="el"><head><meta charset="utf-8">
<title>Τυφλός έλεγχος — λέει ο ήχος αυτό το κείμενο;</title>
<style>
 body {{ font: 16px/1.6 system-ui, sans-serif; margin: 0; background: #f6f6f4; color: #1a1a1a; }}
 header {{ position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd;
          padding: 10px 16px; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; z-index: 2; }}
 .muted {{ color: #666; font-size: 13px; }}
 main {{ max-width: 760px; margin: 0 auto; padding: 16px; }}
 .card {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 8px;
          padding: 16px; margin-bottom: 14px; }}
 .card.cur {{ border-color: #2b6cb0; box-shadow: 0 0 0 2px #2b6cb022; }}
 .txt {{ font-size: 18px; margin: 10px 0 4px; }}
 audio {{ width: 100%; margin-top: 6px; }}
 .btns {{ display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }}
 button {{ font: inherit; padding: 5px 14px; border-radius: 6px; border: 1px solid #ccc;
           background: #fff; cursor: pointer; }}
 button.on.yes {{ background: #2f855a; color: #fff; border-color: #2f855a; }}
 button.on.no {{ background: #c53030; color: #fff; border-color: #c53030; }}
 button.on.idk {{ background: #718096; color: #fff; border-color: #718096; }}
 .n {{ font-size: 12px; color: #999; }}
</style></head><body>
<header>
 <b>Λέει ο ήχος αυτό ακριβώς το κείμενο;</b>
 <span class="muted" id="progress"></span>
 <button id="export">Export</button>
 <span class="muted">space = play · 1 ναι · 2 όχι · 3 δεν ξέρω · j επόμενο</span>
</header>
<main>
 <p class="muted">Άκου το απόσπασμα και κρίνε <b>μόνο</b> αν οι λέξεις που ακούγονται
 είναι αυτές που γράφει το κείμενο. Αγνόησε στίξη, κεφαλαία και το ότι η φράση μπορεί
 να είναι κομμένη στη μέση — έτσι είναι φτιαγμένο το σώμα. «Δεν ξέρω» αν ο ήχος δεν
 ακούγεται καθαρά.</p>
 <div id="list"></div>
</main>
<script>
const ITEMS = {data};
const KEY = "hparl2-blind-v1";
const V = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let cur = 0;

function esc(s) {{ const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }}

function progress() {{
  const done = Object.keys(V).length;
  document.getElementById("progress").textContent = `${{done}}/${{ITEMS.length}} κριμένα`;
}}

function draw() {{
  const list = document.getElementById("list");
  list.innerHTML = "";
  ITEMS.forEach((it, idx) => {{
    const v = V[it.row_id];
    const d = document.createElement("div");
    d.className = "card" + (idx === cur ? " cur" : "");
    d.id = "c" + idx;
    d.innerHTML = `
      <div class="n">${{idx + 1}} / ${{ITEMS.length}} · ${{it.dur}}s</div>
      <audio controls preload="none" src="clips/${{encodeURIComponent(it.row_id)}}.mp3"></audio>
      <p class="txt">${{esc(it.text)}}</p>
      <div class="btns">
        <button class="yes ${{v === "yes" ? "on" : ""}}" data-i="${{idx}}" data-v="yes">ναι (1)</button>
        <button class="no ${{v === "no" ? "on" : ""}}" data-i="${{idx}}" data-v="no">όχι (2)</button>
        <button class="idk ${{v === "idk" ? "on" : ""}}" data-i="${{idx}}" data-v="idk">δεν ξέρω (3)</button>
      </div>`;
    list.appendChild(d);
  }});
  list.querySelectorAll("button[data-v]").forEach(b =>
    b.onclick = () => mark(+b.dataset.i, b.dataset.v));
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
  const a = document.querySelector("#c" + cur + " audio");
  if (e.key === " ") {{ e.preventDefault(); a && (a.paused ? a.play() : a.pause()); }}
  else if (e.key === "1") mark(cur, "yes");
  else if (e.key === "2") mark(cur, "no");
  else if (e.key === "3") mark(cur, "idk");
  else if (e.key === "j" || e.key === "ArrowDown") {{
    cur = Math.min(ITEMS.length - 1, cur + 1); draw();
    document.getElementById("c" + cur).scrollIntoView({{block: "center"}}); }}
  else if (e.key === "ArrowUp") {{
    cur = Math.max(0, cur - 1); draw();
    document.getElementById("c" + cur).scrollIntoView({{block: "center"}}); }}
}});

document.getElementById("export").onclick = () => {{
  const out = ITEMS.map(it => ({{row_id: it.row_id, verdict: V[it.row_id] || null}}));
  const b = new Blob([JSON.stringify(out, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "hparl2-blind-verdicts.json"; a.click();
}};

draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(WORK / "filtered.jsonl"))
    ap.add_argument("--seed", type=int, default=815)
    ap.add_argument("--exclude", default=str(WORK / "verdicts/hparl2-verdicts.json"),
                    help="rows already judged in the non-blind round, to keep out")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.jsonl).read_text(encoding="utf-8").splitlines()]
    ok = [r for r in rows if r.get("asr_ok")]

    seen: set[str] = set()
    ex = Path(args.exclude)
    if ex.exists():
        for o in json.loads(ex.read_text()):
            if o.get("verdict"):
                seen.add(o["row_id"])

    rnd = random.Random(args.seed)
    picked, key = [], []
    for label, lo, hi, n in STRATA:
        pool = [r for r in ok
                if lo <= r["align"] < hi and r["row_id"] not in seen
                and Path(r["mp3"]).exists()]
        rnd.shuffle(pool)
        take = pool[:n]
        if len(take) < n:
            print(f"[blind] WARNING {label}: only {len(take)} of {n} available")
        for r in take:
            picked.append({"row_id": r["row_id"], "text": r["ref"], "dur": r["dur"],
                           "mp3": r["mp3"]})
            key.append({"row_id": r["row_id"], "band": label,
                        "align": r["align"], "keep": r["keep"], "dur": r["dur"]})

    rnd.shuffle(picked)   # bands must not be guessable from position
    order = {p["row_id"]: i for i, p in enumerate(picked)}
    key.sort(key=lambda k: order[k["row_id"]])

    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    for p in picked:
        dst = OUT / "clips" / f"{p['row_id']}.mp3"
        if not dst.exists():
            shutil.copy2(p["mp3"], dst)

    (OUT / "index.html").write_text(render(picked), encoding="utf-8")
    KEY_PATH.write_text(
        json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for k in key:
        counts[k["band"]] = counts.get(k["band"], 0) + 1
    print(f"[blind] {len(picked)} clips: {counts}")
    print(f"[blind] page  -> {OUT/'index.html'}")
    print(f"[blind] key   -> {KEY_PATH}  (outside the served dir)")
    print(f"[blind] serve :  cd {OUT} && python3 -m http.server 8124")


if __name__ == "__main__":
    main()
