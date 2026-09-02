"""Build the private, model-blind page for the training/reference listening audit.

Council audio and labels remain below ~/.cache and are never written to git.  The
served page contains no model hypotheses, arm names, WER, strata, or selection scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


DEFAULT_RESIDUAL = Path.home() / ".cache/oc-public/training-residual-audit-2026-08"
DEFAULT_DENSE = Path.home() / ".cache/oc-public/dense-eval-recovery-cx9ra8iluhr225"
DEFAULT_OUT = Path.home() / ".cache/oc-public/training-listening-audit-2026-08/served"


def dense_window_ids(diagnosis: dict, limit: int = 2) -> list[str]:
    """Select the diagnostic queue by aggregate extra insertions across seeds."""
    totals: Counter[str] = Counter()
    for seed in diagnosis["seeds"]:
        for row in seed["all_windows"]:
            totals[row["window_id"]] += row["insertion_delta"]
    return [key for key, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def review_id(prefix: str, key: str) -> str:
    return hashlib.sha256(f"{prefix}|{key}".encode()).hexdigest()[:16]


def link_audio(source: Path, clips: Path, item_id: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower() or ".audio"
    name = f"{item_id}{suffix}"
    target = clips / name
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"audio link collision: {target}")
    elif target.exists():
        raise RuntimeError(f"refusing to replace non-symlink: {target}")
    else:
        target.symlink_to(source.resolve())
    return f"clips/{name}"


def build_items(residual_root: Path, dense_root: Path, out: Path) -> tuple[list[dict], list[dict]]:
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    review_rows = [json.loads(line) for line in (residual_root / "review.jsonl").read_text().splitlines()]
    residual_key = {
        row["review_id"]: row for row in json.loads((residual_root / "hidden-key.json").read_text())
    }
    public, hidden = [], []
    for row in review_rows:
        item_id = row["review_id"]
        public.append({
            "review_id": item_id,
            "kind": "training_clip",
            "audio": link_audio(Path(row["audio"]), clips, item_id),
            "label": row["training_label"],
        })
        hidden.append({"review_id": item_id, "kind": "training_clip", **residual_key[item_id]})

    validation_dir = dense_root / "corrected-input/eval/validation"
    manifest = json.loads((validation_dir / "manifest.json").read_text())["eval_windows"]
    by_id = {row["window_id"]: row for row in manifest}
    diagnosis = json.loads((dense_root / "dense-insertion-private.json").read_text())
    for window_id in dense_window_ids(diagnosis):
        row = by_id[window_id]
        item_id = review_id("dense-reference-review", window_id)
        public.append({
            "review_id": item_id,
            "kind": "validation_window",
            "audio": link_audio(validation_dir / row["audio_local"], clips, item_id),
            "label": " ".join(row["ref_tokens"]),
            "duration_sec": row["duration_sec"],
        })
        hidden.append({
            "review_id": item_id,
            "kind": "validation_window",
            "window_id": window_id,
            "selection_reason": "top aggregate dense-minus-isolated insertion delta",
        })
    return public, hidden


def render(items: list[dict]) -> str:
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="el"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Τυφλός έλεγχος training data</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f5f5f2;color:#181818}
header{position:sticky;top:0;z-index:2;padding:10px 16px;background:white;border-bottom:1px solid #ddd;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
main{max-width:900px;margin:auto;padding:16px}.card{background:white;border:1px solid #ddd;border-radius:9px;padding:16px;margin:14px 0}.label{font-size:18px;white-space:pre-wrap}.muted{color:#666;font-size:13px}.q{margin-top:14px}.buttons{display:flex;gap:7px;flex-wrap:wrap}button{font:inherit;padding:6px 11px;border:1px solid #bbb;border-radius:6px;background:white;cursor:pointer}button.on{background:#245b9e;color:white;border-color:#245b9e}audio{width:100%}textarea{width:100%;min-height:58px;font:inherit;box-sizing:border-box}.window{border-color:#8b5a00}.saved{color:#287a3c}
</style></head><body>
<header><b>Τυφλός έλεγχος ήχου/αναφοράς</b><span id="progress" class="muted"></span><span id="saved" class="saved"></span><button id="export">Export</button></header>
<main><p>Άκου τον ήχο και κρίνε μόνο την αναφορά που βλέπεις. Αγνόησε στίξη και κεφαλαία. Δεν εμφανίζεται κανένα model output ή metric.</p><div id="items"></div></main>
<script>
const ITEMS=__ITEMS__;const KEY='oc-training-listening-v1';const V=JSON.parse(localStorage.getItem(KEY)||'{}');let timer=null;
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function answered(it){const v=V[it.review_id]||{};return it.kind==='training_clip'?(v.label_faithful&&v.boundaries_usable):v.reference_completeness}
function progress(){const n=ITEMS.filter(answered).length;document.getElementById('progress').textContent=n+'/'+ITEMS.length+' ολοκληρωμένα'}
async function save(){localStorage.setItem(KEY,JSON.stringify(V));try{const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(V)});if(r.ok)document.getElementById('saved').textContent='αποθηκεύτηκε'}catch(e){document.getElementById('saved').textContent='μόνο τοπικά'}}
function schedule(){document.getElementById('saved').textContent='αποθήκευση…';clearTimeout(timer);timer=setTimeout(save,350)}
function mark(id,field,value){V[id]=V[id]||{};V[id][field]=value;V[id].t=Date.now();draw();schedule()}
function buttons(it,field,choices){const v=(V[it.review_id]||{})[field];return '<div class="buttons">'+choices.map(x=>'<button class="'+(v===x[0]?'on':'')+'" data-id="'+it.review_id+'" data-field="'+field+'" data-value="'+x[0]+'">'+x[1]+'</button>').join('')+'</div>'}
function draw(){let train=0,win=0;const root=document.getElementById('items');root.innerHTML=ITEMS.map(it=>{const isWin=it.kind==='validation_window';const num=isWin?++win:++train;let qs='';if(isWin){qs='<div class="q"><b>Η δημοσιευμένη αναφορά παραλείπει λόγια που ακούγονται καθαρά;</b>'+buttons(it,'reference_completeness',[['complete','όχι / πλήρης'],['minor_omission','ναι, λίγα'],['material_omission','ναι, αρκετά'],['unsure','δεν ξέρω']])+'</div><div class="q muted">Αν μπορείς, γράψε στο σημείωμα περίπου timestamps ή τι λείπει.</div>'}else{qs='<div class="q"><b>Είναι πιστή η ετικέτα σε όσα ακούγονται;</b>'+buttons(it,'label_faithful',[['yes','ναι'],['no','όχι'],['unsure','δεν ξέρω']])+'</div><div class="q"><b>Είναι usable τα όρια του clip για training;</b>'+buttons(it,'boundaries_usable',[['yes','ναι'],['no','όχι'],['unsure','δεν ξέρω']])+'</div>'}const note=(V[it.review_id]||{}).notes||'';return '<section class="card '+(isWin?'window':'')+'"><div class="muted">'+(isWin?'Μεγάλο validation window ':'Training clip ')+num+(it.duration_sec?' · '+Math.round(it.duration_sec)+'s':'')+'</div><audio controls preload="none" src="'+it.audio+'"></audio><p class="label">'+esc(it.label)+'</p>'+qs+'<div class="q"><textarea data-note="'+it.review_id+'" placeholder="προαιρετικές σημειώσεις">'+esc(note)+'</textarea></div></section>'}).join('');root.querySelectorAll('button[data-field]').forEach(b=>b.onclick=()=>mark(b.dataset.id,b.dataset.field,b.dataset.value));root.querySelectorAll('textarea[data-note]').forEach(t=>t.onchange=()=>{V[t.dataset.note]=V[t.dataset.note]||{};V[t.dataset.note].notes=t.value;V[t.dataset.note].t=Date.now();schedule()});progress()}
document.getElementById('export').onclick=()=>{const b=new Blob([JSON.stringify(V,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='training-listening-answers.json';a.click()};window.addEventListener('beforeunload',()=>navigator.sendBeacon('/save',new Blob([JSON.stringify(V)],{type:'application/json'})));draw();
</script></body></html>"""
    return template.replace("__ITEMS__", data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-root", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items, hidden = build_items(args.residual_root, args.dense_root, args.out)
    (args.out / "index.html").write_text(render(items), encoding="utf-8")
    key_path = args.out.parent / "hidden-key.json"
    key_path.write_text(json.dumps(hidden, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "served": str(args.out),
        "hidden_key": str(key_path),
        "training_clips": sum(row["kind"] == "training_clip" for row in items),
        "validation_windows": sum(row["kind"] == "validation_window" for row in items),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
