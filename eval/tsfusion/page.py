#!/usr/bin/env python3
"""Render the diagnostic page: speaker turns as readable text, not a spreadsheet.

The bundle is opened both from `file://` and over http, so the data is EMBEDDED in
a script tag rather than fetched: a `file://` page is a null origin and `fetch` of a
sibling file is blocked in every current browser. The audio stays a sibling
`<audio src="page.mp3">`, which `file://` does allow, so the page does not carry
2.4 MB of base64.

Everything derived is computed in `derive.py`. This file only lays it out.
"""
from __future__ import annotations

import json
from pathlib import Path

from eval.tsfusion import derive
from eval.tsfusion.speakers import AMBIGUOUS_SHARE as SP_SHARE

SC = Path.home() / ".cache/oc-public"

PALETTE = ["#3f6fa8", "#4f7d46", "#9a6a2c", "#7a5296", "#2f7d7d", "#9c4a5e"]

CSS = """
:root{
 --bg:#fbfaf8; --fg:#22201d; --mut:#6b665f; --line:#e3ded7; --card:#ffffff;
 --soft:#f4f1ec; --acc:#2f5a8c; --acc-soft:#e8f0f9; --focus:#1f6feb;
 --dis:#b07a1c; --dis-bg:#fdf5e4; --ref:#a53b2c; --ref-bg:#fbeeec;
 --omit:#1d6f66; --omit-bg:#e7f4f1; --drift:#4a6fa5; --drift-bg:#eaf0f9;
 --karaoke:#ffe9a8; --karaoke-fg:#22201d;
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){
 :root:not([data-theme="light"]){
  --bg:#15181c; --fg:#e6e3de; --mut:#9d988f; --line:#2c3138; --card:#1b1f24;
  --soft:#20252b; --acc:#8fb6e2; --acc-soft:#1d2937; --focus:#79b8ff;
  --dis:#e0b25c; --dis-bg:#2d2718; --ref:#e8907f; --ref-bg:#33211e;
  --omit:#6fc9bb; --omit-bg:#17302d; --drift:#9bb8e0; --drift-bg:#1b2536;
  --karaoke:#4a4020; --karaoke-fg:#fff5d8;
 }
}
:root[data-theme="dark"]{
 --bg:#15181c; --fg:#e6e3de; --mut:#9d988f; --line:#2c3138; --card:#1b1f24;
 --soft:#20252b; --acc:#8fb6e2; --acc-soft:#1d2937; --focus:#79b8ff;
 --dis:#e0b25c; --dis-bg:#2d2718; --ref:#e8907f; --ref-bg:#33211e;
 --omit:#6fc9bb; --omit-bg:#17302d; --drift:#9bb8e0; --drift-bg:#1b2536;
 --karaoke:#4a4020; --karaoke-fg:#fff5d8;
}
*{box-sizing:border-box}
html,body{overflow-x:hidden}
body{margin:0;background:var(--bg);color:var(--fg);
 font:16px/1.6 -apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;
 -webkit-text-size-adjust:100%}
.wrap{max-width:820px;margin:0 auto;padding:20px 16px 140px}
h1{font-size:22px;line-height:1.3;margin:0 0 6px;font-weight:600}
h2{font-size:17px;margin:32px 0 8px;font-weight:600}
h3{font-size:14px;margin:16px 0 6px;font-weight:600;color:var(--mut)}
p{margin:8px 0}
.sub{color:var(--mut);font-size:13.5px}
.small{font-size:13.5px;color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:14px 16px;margin:14px 0}
.num{font-variant-numeric:tabular-nums}
.big{font-size:26px;font-weight:600;line-height:1.1}
.cats{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.cat{border:1px solid var(--line);border-radius:8px;padding:10px 12px;
 background:var(--soft)}
.cat .n{font-size:20px;font-weight:600}
.ex{font-size:13px;color:var(--mut);margin-top:6px}
.ex b{font-weight:600;color:var(--fg)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%}
th{text-align:left;font-weight:600;color:var(--mut);padding:6px 8px;
 border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
code,.mono{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:12.5px}

#player{position:sticky;top:0;z-index:20;background:var(--bg);
 padding:10px 0 8px;border-bottom:1px solid var(--line)}
audio{width:100%;height:36px}
#lane{position:relative;height:34px;margin:8px 0 6px;border:1px solid var(--line);
 border-radius:6px;background:var(--card);overflow:hidden;cursor:pointer}
.turnbar{position:absolute;top:5px;height:14px;border-radius:3px;opacity:.85}
.multbar{position:absolute;bottom:3px;height:6px;background:#c2452f;opacity:.55}
#cursor{position:absolute;top:0;bottom:0;width:2px;background:#c2452f;z-index:3}
#seam{position:absolute;top:0;bottom:0;width:2px;background:var(--acc);
 opacity:.6;z-index:2}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
button{font:inherit;font-size:13px;padding:5px 11px;border:1px solid var(--line);
 background:var(--card);border-radius:999px;cursor:pointer;color:var(--fg)}
button.on{background:var(--acc);color:var(--bg);border-color:var(--acc)}
:focus-visible{outline:3px solid var(--focus);outline-offset:2px;border-radius:3px}

.turn{border:1px solid var(--line);border-radius:10px;background:var(--card);
 padding:12px 14px;margin:14px 0}
.turn[hidden]{display:none}
.turnhead{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
 margin-bottom:8px;font-size:13.5px;color:var(--mut)}
.dot{width:10px;height:10px;border-radius:3px;display:inline-block}
.play{border-radius:999px;padding:3px 12px}
.flow{font-size:17px;line-height:2.0;overflow-wrap:break-word}
.w{cursor:pointer;border-radius:3px;padding:0 1px}
.w.noint{color:var(--mut);text-decoration:underline dotted var(--mut) 1px;
 text-underline-offset:4px}
.w.m-disagree{box-shadow:inset 0 -2px 0 var(--dis)}
/* Three error shapes, each with its own outline as well as its own colour:
   substitution underlined, insertion boxed with dots, suspected omission of
   the published text boxed with dashes and greyed. */
.w.m-sub{background:var(--ref-bg);text-decoration:underline solid var(--ref) 2px;
 text-underline-offset:3px}
.w.m-ins{background:var(--ref-bg);border:1px dotted var(--ref);border-radius:4px;
 padding:0 3px}
.w.m-suspect_ref{background:var(--omit-bg);border:1px dashed var(--omit);
 border-radius:4px;padding:0 3px;color:var(--mut)}
.w.m-conflict{background:var(--drift-bg)}
.w.interp{border-bottom:1px dotted var(--mut)}
.w.interp::after{content:"~";font-size:10px;vertical-align:super;
 color:var(--mut);margin-left:1px}
.f-dis .flow .w:not(.mk),.f-drift .flow .w:not(.m-conflict),
.f-omit .flow .w:not(.m-suspect_ref),.f-wrong .flow .w:not(.x-w),
.f-loss .flow .w:not(.x-loss),.f-scribe .flow .w:not(.x-scribe),
.f-soniox .flow .w:not(.x-soniox),.f-whisper .flow .w:not(.x-whisper)
{color:var(--mut);opacity:.5}
.pop{position:absolute;z-index:60;width:min(420px,calc(100vw - 24px));
 background:var(--card);border:1px solid var(--line);
 border-left:3px solid var(--acc);border-radius:10px;padding:12px 14px 10px;
 font-size:13.5px;line-height:1.55;box-shadow:0 8px 28px rgba(0,0,0,.28)}
.pop dl{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:0}
.pop dt{color:var(--mut)}
.pop dd{margin:0}
.pop .x{position:absolute;top:3px;right:6px;border:0;background:none;
 padding:2px 8px;font-size:17px;line-height:1;color:var(--mut)}
.sys th,.sys td{white-space:nowrap}
sup.fl{font-size:10px;line-height:0;color:var(--mut);margin-left:1px;
 letter-spacing:1px}
.w.m-overlap,.w.m-straddle{box-shadow:inset 0 -2px 0 var(--drift)}
.w.m-unresolved{border-bottom:1px dashed var(--mut)}
#ledger table{font-size:13px}
#ledger td{vertical-align:top}
#ledger tr.lrow{cursor:pointer}
#ledger tr.lrow:hover td{background:var(--soft)}
#ledger th[data-sort]{cursor:pointer;user-select:none}
#ledger th[data-sort].on{color:var(--fg);text-decoration:underline}
.ctx{color:var(--mut)}
.tag{display:inline-block;font-size:11px;padding:0 6px;border-radius:999px;
 border:1px solid var(--line);background:var(--soft);color:var(--mut)}
.tag.S{border-color:var(--ref);color:var(--ref)}
.tag.D{border-color:var(--mut);border-style:dashed}
.tag.I{border-color:var(--omit);color:var(--omit)}
.ctrl{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0}
.ctrl label{font-size:13px;color:var(--mut)}
select{font:inherit;font-size:13px;padding:4px 8px;border:1px solid var(--line);
 border-radius:6px;background:var(--card);color:var(--fg)}
#clocklane{position:relative;height:62px;margin:10px 0 4px;
 border:1px solid var(--line);border-radius:6px;background:var(--card);
 overflow:hidden}
#clocklane svg{position:absolute;inset:0;width:100%;height:100%}
#clocklane .turnbar{top:26px;height:10px}
.sys tr.wrow td{font-weight:600}
.w.live{background:var(--karaoke);color:var(--karaoke-fg);opacity:1}
.w.hit{outline:2px solid var(--acc);outline-offset:1px}
.gap{color:var(--ref);cursor:pointer;font-weight:600;padding:0 2px}
.miss{color:var(--mut);cursor:pointer;border-bottom:1px dashed var(--ref);
 padding:0 2px;font-size:14px}
.panel{display:block;margin:8px 0;padding:10px 12px;border:1px solid var(--line);
 border-left:3px solid var(--acc);border-radius:8px;background:var(--soft);
 font-size:13.5px;line-height:1.55}
.panel dl{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:0}
.panel dt{color:var(--mut)}
.panel dd{margin:0}
.band{border-left:3px solid var(--drift);background:var(--drift-bg);
 border-radius:0 8px 8px 0;padding:10px 12px;margin:12px 0}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:13px;color:var(--mut);
 margin-top:8px}
.legend span.k{padding:0 3px;border-radius:3px}
details{border:1px solid var(--line);border-radius:10px;background:var(--card);
 padding:10px 14px;margin:12px 0}
summary{cursor:pointer;font-weight:600;font-size:14px}
@media (max-width:520px){
 .flow{font-size:16px;line-height:1.95}
 .wrap{padding:14px 12px 140px}
}
@media (prefers-reduced-motion:reduce){
 *{scroll-behavior:auto !important}
}
"""

JS = r"""
(function(){
"use strict";
var D = JSON.parse(document.getElementById("payload").textContent);
var DET = D.det, DUR = D.duration;
var au = document.getElementById("au");
var lane = document.getElementById("lane");
var cursor = document.getElementById("cursor");

/* ---------------------------------------------------------------- lane */
function pct(t){ return (100*t/DUR) + "%"; }
D.lane.turns.forEach(function(t){
  var d = document.createElement("div");
  d.className = "turnbar";
  d.style.left = pct(t[0]);
  d.style.width = pct(Math.max(t[1]-t[0], 0.15));
  d.style.background = t[2];
  lane.appendChild(d);
});
D.lane.mult.forEach(function(t){
  var d = document.createElement("div");
  d.className = "multbar";
  d.style.left = pct(t[0]); d.style.width = pct(Math.max(t[1]-t[0],0.05));
  lane.appendChild(d);
});
var seam = document.createElement("div");
seam.id = "seam"; seam.style.left = pct(D.lane.seam);
lane.appendChild(seam);
lane.addEventListener("click", function(e){
  var r = lane.getBoundingClientRect();
  seek((e.clientX - r.left) / r.width * DUR);
});
lane.addEventListener("keydown", function(e){
  var step = e.shiftKey ? 10 : 1, t = null;
  if (e.key === "ArrowRight") t = au.currentTime + step;
  else if (e.key === "ArrowLeft") t = au.currentTime - step;
  else if (e.key === "Home") t = 0;
  else if (e.key === "End") t = DUR - 0.1;
  if (t === null) return;
  e.preventDefault();
  e.stopPropagation();
  seek(t);
});

function seek(t){
  au.currentTime = Math.max(0, Math.min(DUR - 0.05, t));
  au.play().catch(function(){});
}

/* ------------------------------------------------------------ karaoke */
var timed = [];
Object.keys(DET).forEach(function(k){
  var d = DET[k];
  if (d.s === null) return;
  timed.push([d.s, (d.e === null || d.e <= d.s) ? d.s + 0.3 : d.e, k]);
});
timed.sort(function(a,b){ return a[0]-b[0]; });

var live = [], selfScroll = false, lastTop = window.scrollY;
var follow = true;
var fb = document.getElementById("follow");
var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function setFollow(v){
  follow = v;
  fb.classList.toggle("on", v);
  fb.setAttribute("aria-pressed", v ? "true" : "false");
  fb.textContent = v ? "Παρακολούθηση: ναι" : "Παρακολούθηση: όχι";
}
fb.addEventListener("click", function(){ setFollow(!follow); });
window.addEventListener("scroll", function(){
  /* A real user gesture turns the follow off. Our own scrollIntoView does not. */
  if (!selfScroll && follow && Math.abs(window.scrollY - lastTop) > 8)
    setFollow(false);
  lastTop = window.scrollY;
}, {passive:true});

function active(t){
  var out = [], lo = 0, hi = timed.length - 1, k = timed.length;
  while (lo <= hi){                       /* first index with start > t */
    var mid = (lo + hi) >> 1;
    if (timed[mid][0] > t){ k = mid; hi = mid - 1; } else lo = mid + 1;
  }
  for (var i = k - 1; i >= 0 && i > k - 60; i--){
    if (timed[i][0] <= t && t < timed[i][1]) out.push(timed[i][2]);
  }
  return out;
}

au.addEventListener("timeupdate", function(){
  var t = au.currentTime;
  cursor.style.left = pct(t);
  lane.setAttribute("aria-valuenow", t.toFixed(1));
  var now = active(t);
  if (now.length === live.length && now.every(function(x,i){return x===live[i];}))
    return;
  live.forEach(function(k){
    var el = document.getElementById("w" + k);
    if (el) el.classList.remove("live");
  });
  live = now;
  var first = null;
  live.forEach(function(k){
    var el = document.getElementById("w" + k);
    if (el){ el.classList.add("live"); if (!first) first = el; }
  });
  if (first && follow && !au.paused){
    var r = first.getBoundingClientRect();
    if (r.top < 90 || r.bottom > window.innerHeight - 90){
      selfScroll = true;
      first.scrollIntoView({block:"center",
        behavior: reduced ? "auto" : "smooth"});
      setTimeout(function(){ selfScroll = false; lastTop = window.scrollY; }, 700);
    }
  }
});

/* ------------------------------------------------------------- popover */
var GR = {unanimous:"ομοφωνία", majority:"πλειοψηφία", epsilon:"κανένα",
          tie_pivot:"ισοψηφία, αποφάσισε ο πυλώνας"};
var OPS = {equal:"ίδια", sub:"διαφορετική", insert:"δεν υπάρχει στο δημοσιευμένο",
           "delete":"λείπει"};
var SNAME = {scribe:"Scribe", soniox:"Soniox", whisper:"Το μοντέλο μας"};
function esc(s){
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function row(k, v){ return "<dt>" + k + "</dt><dd>" + v + "</dd>"; }
function tok(x){ return x ? esc(x) : "<i>τίποτα</i>"; }
function panelHTML(d){
  var h = "<dl>";
  h += row("Scribe", tok(d.sc));
  h += row("Soniox", tok(d.so));
  h += row("Το μοντέλο μας", tok(d.wh));
  h += row("Διάλεξε το W", (d.w ? "<b>" + esc(d.w) + "</b>" : "<i>τίποτα</i>")
        + " (" + (GR[d.wr] || esc(d.wr || "")) + ")");
  h += row("Δημοσιευμένο", tok(d.rf)
        + (d.rop ? " (" + (OPS[d.rop] || esc(d.rop)) + ")" : ""));
  if (d.s !== null){
    h += row("Θέση", d.s.toFixed(2) + " s"
      + (d.e !== null ? ", ως " + d.e.toFixed(2) + " s" : "")
      + (d.ip ? " (παρεμβολή ανάμεσα σε δύο άγκυρες)" : ""));
  } else {
    h += row("Θέση", "<i>χωρίς θέση</i>");
  }
  h += row("Αβεβαιότητα θέσης", d.un !== null && d.un !== undefined
        ? "±" + d.un.toFixed(2) + " s" : "άγνωστη");
  h += row("Πώς βρέθηκε", esc(d.tm || "") + (d.ip ? ", παρεμβολή" : ", άγκυρα"));
  if (d.cg !== null && d.cg !== undefined)
    h += row("Σύγκρουση χρόνου", d.cg.toFixed(2) + " s απόσταση");
  Object.keys(d.src || {}).forEach(function(k){
    var v = d.src[k];
    var txt = "χωρίς διάστημα";
    if (v.s !== null && v.e !== null)
      txt = v.s.toFixed(2) + " ως " + v.e.toFixed(2) + " s";
    else if (v.s !== null) txt = "από " + v.s.toFixed(2) + " s";
    if (v.c !== null && v.c !== undefined)
      txt += ", εμπιστοσύνη " + v.c.toFixed(2);
    if (v.p) txt += ", " + esc(v.p);
    h += row("Χρόνος " + (SNAME[k] || esc(k)), txt);
  });
  if (d.sp) h += row("Ομιλητής", esc(d.sp)
        + (d.cov !== null && d.cov !== undefined && d.cov < 1
           ? ", καλύπτει το " + Math.round(100*d.cov) + "% της λέξης" : ""));
  if (d.om) h += row("Σημείωση", "εισαγωγή σε στήλη που δύο συστήματα κατέλαβαν, "
                                 + "ύποπτη παράλειψη της δημοσίευσης");
  if (d.fl && d.fl.length){
    h += row("Γιατί σημαδεύτηκε", d.fl.map(esc).join("<br>"));
  }
  h += "</dl>";
  return h;
}

var pop = null;
function closeAll(){
  if (pop){ pop.remove(); pop = null; }
  var h = document.querySelectorAll(".hit");
  for (var j = 0; j < h.length; j++) h[j].classList.remove("hit");
}
function placePop(el){
  var r = el.getBoundingClientRect();
  var sx = window.scrollX, sy = window.scrollY;
  var vw = document.documentElement.clientWidth;
  var w = pop.offsetWidth, h = pop.offsetHeight;
  var left = r.left + sx + r.width / 2 - w / 2;
  left = Math.max(sx + 8, Math.min(left, sx + vw - w - 8));
  var top;
  if (r.bottom + 10 + h <= window.innerHeight - 8) top = sy + r.bottom + 10;
  else if (r.top - 10 - h >= 8) top = sy + r.top - 10 - h;
  else top = sy + Math.max(8, Math.min(r.bottom + 10,
                                       window.innerHeight - h - 8));
  pop.style.left = left + "px";
  pop.style.top = top + "px";
}
function openPop(el){
  var same = el.classList.contains("hit");
  closeAll();
  if (same) return;
  el.classList.add("hit");
  pop = document.createElement("div");
  pop.className = "pop";
  pop.setAttribute("role", "dialog");
  pop.innerHTML = '<button class="x" aria-label="κλείσιμο">×</button>'
                + panelHTML(DET[el.dataset.i]);
  document.body.appendChild(pop);
  placePop(el);
  pop.querySelector(".x").addEventListener("click", function(e){
    e.stopPropagation(); closeAll();
  });
}

document.addEventListener("click", function(e){
  if (pop && pop.contains(e.target)) return;
  var el = e.target.closest("[data-i]");
  if (!el){ closeAll(); return; }
  var d = DET[el.dataset.i];
  if (!d) return;
  if (d.s !== null) seek(d.s - 0.25);
  openPop(el);
});
document.addEventListener("keydown", function(e){
  var t = e.target;
  var tag = t && t.tagName ? t.tagName.toLowerCase() : "";
  var typing = tag === "input" || tag === "textarea" || tag === "select"
            || (t && t.isContentEditable);
  if (e.key === "Enter" && t && t.dataset && t.dataset.i){
    e.preventDefault(); t.click(); return;
  }
  if (e.key === " " || e.key === "Spacebar"){
    if (typing || tag === "button" || tag === "a") return;
    e.preventDefault();
    if (au.paused) au.play().catch(function(){}); else au.pause();
  }
  if (e.key === "Escape") closeAll();
});
window.addEventListener("resize", closeAll);

/* -------------------------------------------------------------- chips */
var chips = document.querySelectorAll("[data-filter]");
function applyFilter(name){
  closeAll();
  for (var i = 0; i < chips.length; i++)
    chips[i].classList.toggle("on", chips[i].dataset.filter === name);
  document.body.className = "f-" + name;
  var turns = document.querySelectorAll(".turn");
  for (var j = 0; j < turns.length; j++)
    turns[j].hidden = !(name === "all" || turns[j].dataset[name] === "1");
}
for (var c = 0; c < chips.length; c++){
  (function(b){
    b.addEventListener("click", function(){ applyFilter(b.dataset.filter); });
  })(chips[c]);
}
applyFilter("all");

/* ------------------------------------------------------------- ledger */
var LED = D.led || [];
var lbody = document.getElementById("lbody");
var lsys = document.getElementById("lsys"), ltype = document.getElementById("ltype");
var lspk = document.getElementById("lspk"), lright = document.getElementById("lright");
var lcount = document.getElementById("lcount");
var SYSL = {scribe:"Scribe", soniox:"Soniox", whisper:"δικό μας", W:"W"};
var TYL = {S:"αντικατάσταση", D:"έλλειψη", I:"εισαγωγή"};
var sortKey = "t", sortDir = 1;

(D.speakers || []).forEach(function(sp){
  var o = document.createElement("option");
  o.value = sp; o.textContent = sp; lspk.appendChild(o);
});

function keep(e){
  if (lsys.value !== "*" && e.sy !== lsys.value) return false;
  if (ltype.value !== "*" && e.ty !== ltype.value) return false;
  if (lspk.value !== "*" && e.sp !== lspk.value) return false;
  var v = lright.value;
  if (v === "none" && e.ok.length) return false;
  if (v === "any"){
    var others = e.ok.filter(function(x){ return x !== e.sy; });
    if (!others.length) return false;
  }
  if (v !== "*" && v !== "none" && v !== "any" && e.ok.indexOf(v) < 0) return false;
  return true;
}
function cell(x){ return x ? esc(x) : '<span class="ctx">-</span>'; }
function renderLedger(){
  var rows = LED.filter(keep);
  rows.sort(function(a,b){
    var x = a[sortKey], y = b[sortKey];
    if (x === null || x === undefined) x = sortKey === "t" ? 1e9 : "";
    if (y === null || y === undefined) y = sortKey === "t" ? 1e9 : "";
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  });
  var h = [];
  rows.forEach(function(e){
    var said = e.ty === "D" ? '<i>τίποτα</i>'
             : esc(e.hy || (e.sy === "W" ? e.w : "") || "");
    var okl = e.ok.length ? e.ok.map(function(x){ return SYSL[x] || x; }).join(", ")
                          : '<span class="ctx">κανένα</span>';
    h.push('<tr class="lrow" data-col="' + (e.col === null ? "" : e.col)
      + '" data-t="' + (e.t === null ? "" : e.t) + '">'
      + '<td class="num">' + (e.t === null ? "-" : e.t.toFixed(2))
        + (e.b ? ' <span title="χρόνος δανεισμένος από τους γείτονες, '
                 + esc(e.me || "") + '">~</span>' : "") + '</td>'
      + '<td>' + cell(e.sp) + '</td>'
      + '<td><span class="tag ' + e.ty + '">' + TYL[e.ty] + '</span>'
        + (e.sy === "*" ? "" : "") + '</td>'
      + '<td><span class="ctx">' + esc(e.l || "") + '</span> <b>'
        + (e.rf ? esc(e.rf) : "<i>τίποτα</i>") + '</b> <span class="ctx">'
        + esc(e.r || "") + '</span></td>'
      + '<td>' + said + '</td>'
      + '<td>' + cell(e.sc) + '</td><td>' + cell(e.so) + '</td>'
      + '<td>' + cell(e.wh) + '</td>'
      + '<td>' + (e.rs ? esc(GR[e.rs] || e.rs) : '<span class="ctx">-</span>')
        + '</td>'
      + '<td>' + okl + '</td></tr>');
  });
  lbody.innerHTML = h.join("");
  lcount.textContent = rows.length + " από " + LED.length + " λάθη";
}
[lsys, ltype, lspk, lright].forEach(function(el){
  el.addEventListener("change", renderLedger);
});
var heads = document.querySelectorAll("#ledger th[data-sort]");
for (var q = 0; q < heads.length; q++){
  (function(th){
    th.addEventListener("click", function(){
      var k = th.dataset.sort;
      if (k === "ref") k = "rf";
      if (k === "speaker") k = "sp";
      if (k === "type") k = "ty";
      sortDir = (k === sortKey) ? -sortDir : 1;
      sortKey = k;
      for (var z = 0; z < heads.length; z++)
        heads[z].classList.toggle("on", heads[z] === th);
      renderLedger();
    });
  })(heads[q]);
}
lbody.addEventListener("click", function(e){
  var tr = e.target.closest("tr.lrow");
  if (!tr) return;
  var t = tr.dataset.t === "" ? null : parseFloat(tr.dataset.t);
  if (t !== null) seek(Math.max(0, t - 0.25));
  var col = tr.dataset.col;
  if (col === "") return;
  var el = document.getElementById("w" + col);
  if (!el) return;
  var card = el.closest(".turn");
  if (card && card.hidden){ applyFilter("all"); }
  closeAll();
  el.classList.add("hit");
  selfScroll = true;
  el.scrollIntoView({block:"center", behavior: reduced ? "auto" : "smooth"});
  setTimeout(function(){ selfScroll = false; lastTop = window.scrollY; }, 700);
});
renderLedger();

/* --------------------------------------------------------- clock lane */
var cl = document.getElementById("clocklane");
if (cl){
  D.lane.turns.forEach(function(t){
    var d = document.createElement("div");
    d.className = "turnbar";
    d.style.left = pct(t[0]);
    d.style.width = pct(Math.max(t[1]-t[0], 0.15));
    d.style.background = t[2];
    cl.appendChild(d);
  });
  var NS = "http://www.w3.org/2000/svg";
  var svg = document.createElementNS(NS, "svg");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("viewBox", "0 0 1000 62");
  function ticks(list, y0, y1, colour){
    list.forEach(function(x){
      var ln = document.createElementNS(NS, "line");
      var px = 1000 * x / DUR;
      ln.setAttribute("x1", px); ln.setAttribute("x2", px);
      ln.setAttribute("y1", y0); ln.setAttribute("y2", y1);
      ln.setAttribute("stroke", colour);
      ln.setAttribute("stroke-width", "0.6");
      svg.appendChild(ln);
    });
  }
  ticks(D.ticks.soniox || [], 2, 24, "#3f6fa8");
  ticks(D.ticks.whisper || [], 38, 60, "#9a6a2c");
  cl.appendChild(svg);
  cl.addEventListener("click", function(e){
    var r = cl.getBoundingClientRect();
    seek((e.clientX - r.left) / r.width * DUR);
  });
}

/* per turn play button */
var plays = document.querySelectorAll("[data-play]");
for (var p = 0; p < plays.length; p++){
  (function(b){
    b.addEventListener("click", function(e){
      e.stopPropagation();
      var v = b.dataset.play.split(",");
      seek(parseFloat(v[0]));
      var stop = parseFloat(v[1]);
      var h = function(){
        if (au.currentTime >= stop){
          au.pause(); au.removeEventListener("timeupdate", h);
        }
      };
      au.addEventListener("timeupdate", h);
    });
  })(plays[p]);
}
})();
"""

GREEK_REASON = {"unanimous": "ομοφωνία", "majority": "πλειοψηφία",
                "epsilon": "καμία πρόταση", "tie_pivot": "ισοψηφία"}
GREEK_STATE = {"named": "ένας ομιλητής", "ambiguous": "δύο ομιλητές μοιράζονται",
               "overlap": "επικάλυψη", "non_speech": "χωρίς ομιλία",
               "unresolved": "αβέβαιη θέση", "no_diarization": "χωρίς διαρισμό"}


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def embed(obj) -> str:
    """JSON safe inside a script tag and safe for a JS string literal."""
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def num(x, fmt="{:.2f}") -> str:
    """A number, or a dash. Half this page's fields are legitimately empty."""
    return "-" if x is None else fmt.format(x)


def secs(x) -> str:
    return "" if x is None else f"{x:.2f}"


def clock(x) -> str:
    if x is None:
        return "-"
    m, s = divmod(max(0.0, x), 60)
    return f"{int(m)}:{s:04.1f}"


# ----------------------------------------------------------------- sections
def summary_section(view, data) -> list:
    s = view["summary"]
    c = s["census"]["counts"]
    items = s["census"]["items"]
    out = []
    A = out.append

    def example(bucket, fallback):
        if not items[bucket]:
            return fallback
        it = items[bucket][0]
        if it["kind"] == "sub":
            return (f'το W είπε <b>{esc(it["w"])}</b> και το δημοσιευμένο κείμενο '
                    f'γράφει <b>{esc(it["ref"])}</b>')
        if it["kind"] == "insert":
            return (f'το W είπε <b>{esc(it["w"])}</b> και το δημοσιευμένο κείμενο '
                    f'δεν έχει τίποτα εκεί')
        return (f'το δημοσιευμένο κείμενο έχει <b>{esc(it["ref"])}</b> και το W '
                f'δεν έγραψε λέξη εκεί')

    A('<div class="card">')
    A('<h2 style="margin-top:0">Τι λέει αυτή η σελίδα, με απλά λόγια</h2>')
    A(f'<p>Στα {data["manifest"]["page_duration"]:.0f} δευτερόλεπτα ήχου το '
      f'δημοσιευμένο κείμενο έχει <b class="num">{s["ref_tokens"]}</b> λέξεις. '
      f'Η σύνθεση W έγραψε <b class="num">{s["w_tokens"]}</b> λέξεις και '
      f'<b class="num">{s["correct"]}</b> από αυτές ταιριάζουν ακριβώς με το '
      f'δημοσιευμένο κείμενο, δηλαδή '
      f'{100.0 * s["correct"] / s["ref_tokens"]:.1f} τοις εκατό.</p>')
    A(f'<p>Χρεώνονται <b class="num">{s["distance"]}</b> λάθη συνολικά. '
      f'Δεν είναι όλα λάθη του ίδιου είδους, και δεν είναι όλα δικά μας. '
      f'Ο διαχωρισμός παρακάτω βγαίνει από τα ίδια τα δεδομένα της σελίδας.</p>')
    A('<div class="cats">')
    A(f'<div class="cat"><div class="n num">{c["convention"]}</div>'
      'σύμβαση του δημοσιευμένου κειμένου, που κανένα σύστημα ομιλίας δεν μπορεί '
      'να παραγάγει'
      f'<div class="ex">{example("convention", "καμία περίπτωση")}</div></div>')
    A(f'<div class="cat"><div class="n num">{c["ref_omission"]}</div>'
      'λόγος που ακούστηκε και λείπει από το δημοσιευμένο κείμενο, με δύο '
      'τουλάχιστον συστήματα να τον προτείνουν χωριστά'
      f'<div class="ex">{example("ref_omission", "καμία περίπτωση")}</div></div>')
    A(f'<div class="cat"><div class="n num">{c["ours"]}</div>'
      'πραγματικό λάθος της επιλογής μας'
      f'<div class="ex">{example("ours", "καμία περίπτωση")}</div></div>')
    A('</div>')
    A(f'<p class="small">Οι τρεις κατηγορίες αθροίζουν ακριβώς στα '
      f'{c["total"]} χρεωμένα λάθη. Η πρώτη αναγνωρίζεται ως εξής: το '
      f'δημοσιευμένο κείμενο έχει λέξη το πολύ '
      f'{derive.ABBREVIATION_MAX_LEN} χαρακτήρων που κανένα από τα τρία '
      f'συστήματα δεν έγραψε πουθενά σε αυτή τη σελίδα, δηλαδή συντομογραφία ή '
      f'αρχικό. Οι λέξεις εδώ είναι ήδη κανονικοποιημένες, χωρίς τόνους και '
      f'σημεία στίξης, οπότε άλλες συμβάσεις της δημοσίευσης είναι αόρατες και '
      f'ο αριθμός είναι κάτω όριο. Η δεύτερη κατηγορία είναι ένδειξη, όχι '
      f'απόδειξη: τρία συστήματα μπορούν να κάνουν και το ίδιο λάθος μαζί.</p>')
    A(f'<p class="small">Το WER του W απέναντι στο δημοσιευμένο κείμενο είναι '
      f'<b class="num">{s["wer"]:.4f}</b>. Αν αφαιρεθούν οι '
      f'{s["insertions_suspect"]} εισαγωγές που δύο συστήματα πρότειναν μόνα '
      f'τους, γίνεται <b class="num">{s["wer_without_suspects"]:.4f}</b>. '
      f'Ο δεύτερος αριθμός δεν αντικαθιστά τον πρώτο. Μπαίνει δίπλα του, ώστε '
      f'να φαίνεται πόσο από το νούμερο εξαρτάται από την υπόθεση ότι το '
      f'δημοσιευμένο κείμενο είναι πλήρες.</p>')
    A(f'<p class="small">Οι {s["insertions"]} εισαγωγές του W χωρίζονται σε '
      f'{s["insertions_suspect"]} ύποπτες παραλείψεις της δημοσίευσης και '
      f'{s["insertions"] - s["insertions_suspect"]} γνήσιες εισαγωγές.</p>')
    A('</div>')
    out += system_panel(view)
    return out


def system_panel(view) -> list:
    """What each system gets wrong on its own, next to what the vote gets wrong."""
    out = []
    A = out.append
    tbl = view["per_system"]
    n = tbl[0]["ref_tokens"]
    A('<div class="card">')
    A('<h2 style="margin-top:0">Τι λάθη κάνει το κάθε σύστημα χωριστά</h2>')
    A(f'<p class="small">Κάθε γραμμή είναι μια δική της στοίχιση απέναντι στις '
      f'{n} λέξεις του δημοσιευμένου κειμένου, όχι προβολή της μιας πάνω στην '
      f'άλλη. Το W δεν είναι σύστημα, είναι η ψηφοφορία των τριών.</p>')
    A('<div class="scroll"><table class="sys">')
    A('<tr><th>σύστημα</th><th>αντικαταστάσεις</th><th>ελλείψεις</th>'
      '<th>εισαγωγές</th><th>(Α+Ε)/Ν</th><th>WER</th>'
      '<th>WER χωρίς ύποπτες εισαγωγές</th></tr>')
    for r in tbl:
        cls = ' class="wrow"' if r["key"] == "W" else ""
        A(f'<tr{cls}><td>{esc(r["name"])}</td>'
          f'<td class="num">{r["S"]}</td>'
          f'<td class="num">{r["D"]}</td>'
          f'<td class="num">{r["I"]}</td>'
          f'<td class="num">{r["sd_rate"]:.4f}</td>'
          f'<td class="num">{r["wer"]:.4f}</td>'
          f'<td class="num">{r["wer_excl_suspect"]:.4f} '
          f'<span class="small">({r["suspect_insertions"]})</span></td></tr>')
    A('</table></div>')
    A('<p class="small">Η στήλη (Α+Ε)/Ν είναι το ποσοστό λέξεων του '
      'δημοσιευμένου κειμένου που χάθηκαν ή γράφτηκαν λάθος. Δεν κατεβαίνει '
      'γράφοντας λιγότερα, οπότε είναι η στήλη που δείχνει ποιος πραγματικά '
      'ακούει. Οι εισαγωγές δεν μπαίνουν σε αυτήν εξ ορισμού.</p>')
    A('<p class="small">Η τελευταία στήλη είναι το ίδιο WER αφού αφαιρεθούν οι '
      'εισαγωγές που έπεσαν σε στήλη την οποία κατέλαβαν τουλάχιστον δύο από τα '
      'τρία συστήματα (ο αριθμός τους σε παρένθεση). Είναι υπόθεση για το '
      'δημοσιευμένο κείμενο, όχι διόρθωση του συστήματος, και μπαίνει δίπλα '
      'στο κανονικό WER, ποτέ στη θέση του.</p>')
    A(f'<p class="small">Σε <b class="num">{view["summary"]["selection_loss"]}'
      '</b> λέξεις το W έγραψε λάθος ενώ κάποιο από τα τρία συστήματα είχε '
      'ακριβώς τη δημοσιευμένη λέξη. Το φίλτρο «Απώλειες επιλογής» δείχνει '
      'μόνο αυτές.</p>')
    A('</div>')
    return out


def player_section(view, data) -> list:
    m = data["manifest"]
    colors = {s: PALETTE[i % len(PALETTE)]
              for i, s in enumerate(data["diar"]["speakers"])}
    out = []
    A = out.append
    A('<div id="player">')
    A('<audio id="au" controls preload="metadata" src="page.mp3"></audio>')
    A(f'<div id="lane" role="slider" tabindex="0" aria-label="χρονογραμμή" '
      f'aria-valuemin="0" aria-valuemax="{m["page_duration"]:.0f}" '
      f'aria-valuenow="0"><div id="cursor"></div></div>')
    A('<div class="chips">')
    A('<button data-filter="all" class="on">Όλα</button>')
    A('<button data-filter="dis">Διαφωνίες</button>')
    A('<button data-filter="wrong">Λάθη του W</button>')
    A('<button data-filter="loss">Απώλειες επιλογής</button>')
    A('<button data-filter="scribe">Λάθη Scribe</button>')
    A('<button data-filter="soniox">Λάθη Soniox</button>')
    A('<button data-filter="whisper">Λάθη δικού μας</button>')
    A('<button data-filter="drift">Ζώνες παρέκκλισης</button>')
    A('<button data-filter="omit">Ύποπτες παραλείψεις</button>')
    A('<button id="follow" class="on" aria-pressed="true">'
      'Παρακολούθηση: ναι</button>')
    A('</div>')
    A('<p class="small">«Απώλεια επιλογής» είναι η λέξη όπου το W έγραψε κάτι '
      'άλλο από το δημοσιευμένο κείμενο ενώ ένα από τα τρία συστήματα το είχε '
      'σωστά. Είναι η μόνη κατηγορία που μια καλύτερη ψηφοφορία θα διόρθωνε '
      'χωρίς νέο μοντέλο.</p>')
    A('<div class="legend">')
    for s in data["diar"]["speakers"]:
        A(f'<span><span class="dot" style="background:{colors[s]}"></span> '
          f'{esc(s)}</span>')
    A('<span><span class="k" style="box-shadow:inset 0 -2px 0 var(--dis)">'
      'λέξη</span> διαφωνία συστημάτων</span>')
    A('<span><span class="k" style="background:var(--ref-bg);'
      'text-decoration:underline solid var(--ref) 2px">λέξη</span> '
      'αντικατάσταση, το δημοσιευμένο γράφει άλλη λέξη</span>')
    A('<span><span class="k" style="background:var(--ref-bg);'
      'border:1px dotted var(--ref);border-radius:4px">λέξη</span> εισαγωγή, '
      'δεν υπάρχει στο δημοσιευμένο</span>')
    A('<span><span class="k" style="background:var(--omit-bg);'
      'border:1px dashed var(--omit);border-radius:4px;color:var(--mut)">λέξη'
      '</span> ύποπτη παράλειψη της δημοσίευσης</span>')
    A('<span><span class="k" style="background:var(--drift-bg)">λέξη</span> '
      'σύγκρουση χρονοσήμανσης</span>')
    A('<span><span class="k" style="color:var(--ref);font-weight:600">◦</span> '
      'το W άφησε κενό ενώ κάποιο σύστημα πρότεινε λέξη</span>')
    A('<span><span class="k" style="color:var(--mut);'
      'border-bottom:1px dashed var(--ref)">λέξη</span> λείπει από το W ενώ την '
      'έχει το δημοσιευμένο</span>')
    for key, (badge, text) in FLAG.items():
        A(f'<span><span class="k mono">{badge}</span> {text}'
          f' ({view["summary"]["marked"].get(key, 0)})</span>')
    A('</div>')
    A('</div>')
    A(f'<p class="small">Πατήστε το πλήκτρο διαστήματος για αναπαραγωγή και '
      f'παύση. Πατήστε οποιαδήποτε λέξη για να πάτε εκεί στον ήχο. Η μπάρα '
      f'δείχνει τον αποκλειστικό διαρισμό, με κόκκινο εκεί όπου ο κανονικός '
      f'διαρισμός βλέπει πάνω από έναν ενεργό ομιλητή, και με μπλε τη ραφή των '
      f'δύο παραθύρων στο '
      f'{clock(data["seam_page"][0])}.</p>')
    return out


MARK_TITLE = {"m-sub": "το δημοσιευμένο κείμενο γράφει άλλη λέξη",
              "m-ins": "δεν υπάρχει στο δημοσιευμένο κείμενο",
              "m-suspect_ref": "εισαγωγή σε στήλη που δύο συστήματα κατέλαβαν",
              "m-disagree": "τα συστήματα διαφωνούν",
              "m-conflict": "σύγκρουση χρονοσήμανσης"}

# Every warning the page is allowed to paint, its badge, and the sentence the
# reader gets on hover. A warning with no entry here cannot be rendered.
FLAG = {
    "overlap": ("◑", "άλλος ομιλητής μιλά ταυτόχρονα"),
    "straddle": ("⇄", "η λέξη πέφτει πάνω στην αλλαγή ομιλητή"),
    "unresolved": ("?", "η θέση της λέξης δεν είναι αξιόπιστη"),
    "wide": ("±", "πολύ φαρδύ διάστημα αβεβαιότητας"),
    "conflict": ("≠", "δύο συστήματα τη βάζουν σε ασύνδετα σημεία"),
    "interpolated": ("~", "θέση με παρεμβολή ανάμεσα σε δύο άγκυρες"),
}


def flag_text(w: dict) -> str:
    """One sentence per warning, with the number that produced it."""
    k = w["k"]
    base = FLAG[k][1]
    if k == "overlap":
        return (f'{base}: {esc(w.get("speaker") or "άλλος")} για '
                f'{w["seconds"]:.2f} s, {100 * w["fraction"]:.0f}% της λέξης')
    if k == "straddle":
        cov = w.get("coverage")
        extra = f', κάλυψη {100 * cov:.0f}%' if cov is not None else ""
        return (f'{base}: {esc(w.get("speaker") or "?")} και '
                f'{esc(w.get("runner_up") or "?")}{extra}')
    if k == "unresolved":
        return f'{base}: {esc(w.get("reason") or "χωρίς αιτία")}'
    if k == "wide":
        return (f'{base}: ±{w["seconds"]:.2f} s, πάνω από το όριο του '
                f'{derive.WIDE_UNCERTAINTY_SECONDS:.1f} s')
    if k == "conflict":
        sec = w.get("seconds")
        return base + (f': απόσταση {sec:.2f} s' if sec is not None else "")
    return f'{base} ({esc(w.get("method") or "")})'


def word_classes(r) -> list:
    """Every visual mark this chip carries. Never colour on its own."""
    cls = ["w"]
    if r["marks"]:
        cls.append("mk")
    if "disagree" in r["marks"]:
        cls.append("m-disagree")
    if "conflict" in r["marks"]:
        cls.append("m-conflict")
    if r["ref_omission_suspect"]:
        cls.append("m-suspect_ref")
    elif r["ref_op"] == "sub":
        cls.append("m-sub")
    elif r["ref_op"] == "insert":
        cls.append("m-ins")
    for w in r["warnings"]:
        cls.append("m-" + w["k"])
    for sysname in r["sys_wrong"]:
        cls.append("x-" + sysname)
    if r["w_wrong"]:
        cls.append("x-w")
    if r["selection_loss"]:
        cls.append("x-loss")
    return cls


def word_html(r, det_ids: set) -> str:
    cls = word_classes(r)
    det_ids.add(r["i"])
    reasons = [MARK_TITLE[c] for c in cls if c in MARK_TITLE]
    reasons += [flag_text(w) for w in r["warnings"]]
    badges = "".join(FLAG[w["k"]][0] for w in r["warnings"])
    title = "; ".join(reasons)
    return (f'<span class="{" ".join(cls)}" id="w{r["i"]}" data-i="{r["i"]}" '
            f'tabindex="0" role="button"'
            + (f' title="{esc(title)}"' if title else "")
            + f'>{esc(r["w"])}'
            + (f'<sup class="fl">{badges}</sup>' if badges else "")
            + '</span>')


def turn_section(view, data) -> list:
    by_i = {r["i"]: r for r in view["rows"]}
    dels = view["deletions_at"]
    colors = {s: PALETTE[i % len(PALETTE)]
              for i, s in enumerate(data["diar"]["speakers"])}
    out, det_ids = [], set()
    A = out.append
    A('<h2>Οι τοποθετήσεις, όπως διαβάζονται</h2>')
    A('<p class="small">Κάθε κάρτα είναι ένας συνεχόμενος λόγος ενός ομιλητή. Το '
      'κείμενο είναι οι λέξεις που διάλεξε το W. Τα σημάδια δείχνουν πού κάτι '
      'πήγε στραβά, και κάθε σημαδεμένη λέξη ανοίγει τι πρότεινε το κάθε '
      'σύστημα.</p>')
    last_i = view["rows"][-1]["i"] if view["rows"] else None
    for t in view["turns"]:
        rows = [by_i[i] for i in t["rows"]]
        # deletions are keyed by the column they follow, so the bucket of the
        # very last column has no next row to hang off and is emitted here
        tail = (dels.get(last_i, []) if rows and rows[-1]["i"] == last_i else [])
        tail = [d for d in tail if d.get("system") == "W"]
        has_dis = any(not r["agree"] or r["marks"] for r in rows)
        has_drift = any(r["time_conflict"] for r in rows)
        has_omit = any(r["ref_omission_suspect"] for r in rows)
        flags = {"dis": has_dis, "drift": has_drift, "omit": has_omit,
                 "wrong": any(r["w_wrong"] for r in rows),
                 "loss": any(r["selection_loss"] for r in rows)}
        for sysname in derive.SYSTEMS:
            flags[sysname] = any(sysname in r["sys_wrong"] for r in rows)
        name = t["speaker"] or GREEK_STATE.get(t["state"], t["state"])
        col = colors.get(t["speaker"], "#8a8a8a")
        attrs = " ".join(f'data-{k}="{1 if v else 0}"'
                         for k, v in flags.items())
        A(f'<section class="turn" {attrs}>')
        A('<div class="turnhead">')
        A(f'<span class="dot" style="background:{col}"></span>')
        A(f'<b>{esc(name)}</b>')
        if t["state"] != "named":
            A(f'<span>({esc(GREEK_STATE.get(t["state"], t["state"]))})</span>')
        A(f'<span class="num">{clock(t["page_start"])} ως '
          f'{clock(t["page_end"])}</span>')
        if t["page_start"] is not None and t["page_end"] is not None:
            A(f'<button class="play" data-play="{t["page_start"]:.3f},'
              f'{t["page_end"]:.3f}">Ακρόαση</button>')
        n_sub = sum(1 for r in rows if r["ref_op"] == "sub")
        n_ins = sum(1 for r in rows if r["ref_op"] == "insert")
        n_del = sum(1 for i in t["rows"]
                    for d in dels.get(i - 1, [])
                    if d.get("system") == "W") + len(tail)
        A(f'<span class="small">{len(rows)} λέξεις, '
          f'{n_sub} αντικαταστάσεις, {n_del} ελλείψεις, '
          f'{n_ins} εισαγωγές</span>')
        A('</div>')
        A('<div class="flow">')
        parts = []
        for r in rows:
            for d in dels.get(r["i"] - 1, []):
                if d.get("system") != "W":
                    continue
                parts.append(
                    f'<span class="miss" title="λείπει από το W">'
                    f'{esc(d["word"])}</span>')
            if r["w"]:
                parts.append(word_html(r, det_ids))
            elif "dropped" in r["marks"]:
                det_ids.add(r["i"])
                parts.append(f'<span class="gap" id="w{r["i"]}" '
                             f'data-i="{r["i"]}" tabindex="0" role="button" '
                             f'title="το W δεν έγραψε λέξη εδώ">◦</span>')
        for d in tail:
            parts.append(f'<span class="miss" title="λείπει από το W">'
                         f'{esc(d["word"])}</span>')
        A(" ".join(parts))
        A('</div>')
        A('</section>')
    return out, det_ids


def ledger_section(view) -> list:
    """The shell of the error ledger. Its rows are built in JS so they sort."""
    out = []
    A = out.append
    led = view["ledger"]
    n = {k: sum(1 for e in led if e["system"] == k)
         for k in derive.LEDGER_SYSTEMS}
    A('<h2 id="ledger-h">Κατάλογος λαθών, ένα ένα</h2>')
    A(f'<p class="small">Κάθε χρεωμένο λάθος κάθε συστήματος, με τα συμφραζόμενά '
      f'του. Σύνολα: W {n["W"]}, Scribe {n["scribe"]}, Soniox {n["soniox"]}, '
      f'το μοντέλο μας {n["whisper"]}. Οι αριθμοί συμφωνούν με τον πίνακα στην '
      f'κορυφή, γιατί βγαίνουν από την ίδια στοίχιση ανά σύστημα. Πατήστε '
      f'γραμμή για να ακούσετε το σημείο και να πάτε στη λέξη μέσα στο '
      f'κείμενο.</p>')
    A('<div id="ledger">')
    A('<div class="ctrl">')
    A('<label for="lsys">Σύστημα</label><select id="lsys">')
    A('<option value="W">W (σύνθεση)</option>')
    A('<option value="scribe">Scribe v2</option>')
    A('<option value="soniox">Soniox</option>')
    A('<option value="whisper">Το μοντέλο μας</option>')
    A('<option value="*">Όλα τα συστήματα</option>')
    A('</select>')
    A('<label for="ltype">Είδος</label><select id="ltype">'
      '<option value="*">Όλα</option>'
      '<option value="S">Αντικαταστάσεις</option>'
      '<option value="D">Ελλείψεις</option>'
      '<option value="I">Εισαγωγές</option></select>')
    A('<label for="lspk">Ομιλητής</label><select id="lspk">'
      '<option value="*">Όλοι</option></select>')
    A('<label for="lright">Ποιος το είχε σωστά</label><select id="lright">'
      '<option value="*">Αδιάφορο</option>'
      '<option value="none">Κανένα σύστημα</option>'
      '<option value="any">Κάποιο άλλο σύστημα</option>'
      '<option value="scribe">Το Scribe</option>'
      '<option value="soniox">Το Soniox</option>'
      '<option value="whisper">Το μοντέλο μας</option>'
      '<option value="W">Το W</option></select>')
    A('<span class="small" id="lcount"></span>')
    A('</div>')
    A('<div class="scroll"><table><thead><tr>'
      '<th data-sort="t" class="on">t</th>'
      '<th data-sort="speaker">ομιλητής</th>'
      '<th data-sort="type">είδος</th>'
      '<th data-sort="ref">δημοσιευμένο κείμενο</th>'
      '<th>είπε</th><th>scribe</th><th>soniox</th><th>δικό μας</th>'
      '<th>ψήφος</th><th>το είχαν σωστά</th></tr></thead>'
      '<tbody id="lbody"></tbody></table></div>')
    A('</div>')
    A('<p class="small">Η έλλειψη δεν έχει δικό της χρόνο: μια λέξη που κανένα '
      'σύστημα δεν έγραψε δεν έχει χρονοσήμανση πουθενά. Ο χρόνος που βλέπετε '
      'στις γραμμές με το σύμβολο ~ είναι δανεισμένος από τους γείτονες με τη '
      'σταθερή ιεραρχία του <code>refalign.place_deletions</code> '
      '(εγκιβωτισμός ανάμεσα σε δύο άγκυρες, αλλιώς προέκταση), και η στήλη '
      '«είδος» το λέει σε κάθε τέτοια γραμμή. Είναι θέση για ακρόαση, όχι '
      'μέτρηση.</p>')
    return out


def clock_section(view, data) -> list:
    """Which of the two timed systems should be the page's clock."""
    c = view["clocks"]
    sx, wh = c["systems"]["soniox"], c["systems"]["whisper"]
    out = []
    A = out.append
    A('<h2 id="clocks">Ποιο ρολόι να δεθεί με τον διαρισμό</h2>')
    A(f'<p class="small">Σήμερα κάθε σύστημα κουβαλά το δικό του ρολόι και η '
      f'σελίδα τα ενώνει ανά στήλη. Η καλύτερη πρακτική είναι το αντίθετο: ένα '
      f'σύστημα δίνει το χρονολόγιο, ο διαρισμός δένεται πάνω σε αυτό, και τα '
      f'άλλα κείμενα συγκρίνονται με αυτό. Δύο υποψήφιοι υπάρχουν εδώ, γιατί '
      f'μόνο δύο έχουν χρόνους ανά λέξη.</p>')
    A('<div class="scroll"><table class="sys"><tr><th>μέτρηση</th>'
      '<th>Soniox (stt-rt-v4)</th><th>Το μοντέλο μας (word_timestamps)</th>'
      '</tr>')

    def line(label, key, fmt, better=None):
        a, b = sx[key], wh[key]
        f = (lambda x: "-" if x is None else fmt.format(x))
        return (f'<tr><td>{label}</td><td class="num">{f(a)}</td>'
                f'<td class="num">{f(b)}</td></tr>')

    A(line("λέξεις με χρόνο", "n_words", "{:d}"))
    A(line("κάλυψη των στηλών", "coverage", "{:.1%}"))
    A(line("διάμεση διάρκεια λέξης", "median_duration", "{:.2f} s"))
    A(line("λέξεις που πατούν πάνω σε όριο σειράς", "straddling", "{:d}"))
    A(line("ποσοστό τους", "straddling_rate", "{:.1%}"))
    A(line("αλλαγές ομιλητή που πέφτουν μέσα σε λέξη",
           "changes_inside_a_word", "{:d}"))
    A(line("διάμεση απόσταση άκρου λέξης ως το κοντινότερο όριο σειράς",
           "median_word_edge_to_turn_edge", "{:.2f} s"))
    A(line("διάμεση απόσταση ορίου σειράς ως το κοντινότερο άκρο λέξης",
           "median_turn_edge_to_word_edge", "{:.2f} s"))
    A('</table></div>')
    A(f'<p class="small">Στις {c["both_timed"]} στήλες όπου και τα δύο δίνουν '
      f'χρόνο, οι αρχές τους απέχουν διάμεσα '
      f'{num(c["median_start_gap"], "{:.2f} s")} και διαφωνούν πάνω από '
      f'{c["threshold"]:.2f} s σε <b class="num">'
      f'{c["disagree_over_threshold"]}</b> από αυτές '
      f'({num(c["disagree_rate"], "{:.1%}")}).</p>')
    A('<div id="clocklane"></div>')
    A('<p class="small">Πάνω από τη μπάρα του διαρισμού είναι οι αρχές των '
      'λέξεων του Soniox, κάτω από αυτήν του δικού μας. Όπου οι δύο σειρές '
      'γραμμών δεν πέφτουν μαζί, τα δύο ρολόγια διαφωνούν.</p>')
    fav = {"soniox": "το Soniox", "whisper": "το δικό μας",
           "mixed": "κανένα καθαρά", None: "κανένα"}[c["favours"]]
    A(f'<p class="small">Τα νούμερα δείχνουν προς <b>{fav}</b> ως άγκυρα: '
      f'το Soniox καλύπτει {num(sx["coverage"], "{:.1%}")} των στηλών απέναντι '
      f'σε {num(wh["coverage"], "{:.1%}")} και τα όρια των σειρών πέφτουν κατά '
      f'μέσο όρο {num(sx["median_turn_edge_to_word_edge"], "{:.2f} s")} από '
      f'κάποιο άκρο λέξης του, απέναντι σε '
      f'{num(wh["median_turn_edge_to_word_edge"], "{:.2f} s")}. '
      f'Το δικό μας πατά λιγότερο πάνω στα όρια '
      f'({num(wh["straddling_rate"], "{:.1%}")} απέναντι σε '
      f'{num(sx["straddling_rate"], "{:.1%}")}) και '
      f'{wh["changes_inside_a_word"]} από τις {c["n_speaker_changes"]} αλλαγές '
      f'ομιλητή πέφτουν μέσα σε λέξη του, ενώ στο Soniox πέφτουν '
      f'{sx["changes_inside_a_word"]}. Οι δύο μετρήσεις δεν λένε το ίδιο '
      f'πράγμα: η μία μετρά κάλυψη, η άλλη ακρίβεια στα όρια.</p>')
    A(f'<p class="small">Το δείγμα είναι ΕΝΑ κομμάτι '
      f'{data["manifest"]["page_duration"]:.0f} δευτερολέπτων: '
      f'{c["n_columns"]} στήλες, {c["n_turns"]} σειρές διαρισμού, '
      f'{c["n_speaker_changes"]} αλλαγές ομιλητή. Δεν βγαίνει απόφαση '
      f'χρονολογίου από τόσο, μόνο ένδειξη.</p>')
    A('<p class="small"><b>Παγίδα προέλευσης.</b> Οι χρόνοι του Soniox εδώ '
      'είναι από το <code>stt-rt-v4</code>, ΟΧΙ από το '
      '<code>stt-async-v5</code> του οποίου το κείμενο ψηφίζει μέσα στο W. '
      'Το ένα δεν αντικαθιστά το άλλο: αν πάρετε χρόνους από το ένα μοντέλο '
      'και κείμενο από το άλλο, το W που μετρήθηκε παύει να είναι το W. '
      'Το πάνελ λέει ποιο ΡΟΛΟΙ να δεθεί με τον διαρισμό, τίποτε άλλο.</p>')
    return out


def zones_section(view) -> list:
    out = []
    A = out.append
    A('<h2 id="zones">Ζώνες παρέκκλισης χρόνου</h2>')
    A('<p class="small">Εδώ οι χρονοσημάνσεις δύο συστημάτων για την ίδια στήλη '
      'δεν τέμνονται. Σχεδόν πάντα σημαίνει ότι η στοίχιση έβαλε στην ίδια στήλη '
      'δύο διαφορετικές εκφορές της ίδιας λέξης, οπότε ο ένας από τους δύο '
      'χρόνους ανήκει αλλού. Η σελίδα τοποθετεί τη λέξη στο νωρίτερο διάστημα '
      'που παρατήρησε κάποιο σύστημα, και δείχνει τη σύγκρουση χωριστά.</p>')
    if not view["zones"]:
        A('<p class="small">Καμία ζώνη.</p>')
        return out
    by_i = {r["i"]: r for r in view["rows"]}
    for z in view["zones"]:
        A('<div class="band">')
        words = " ".join(z["words"]) or "χωρίς λέξη"
        gap_txt = ("άγνωστη απόσταση" if z["max_gap"] is None
                   else f'απόσταση έως {z["max_gap"]:.2f} δευτερόλεπτα')
        A(f'<p style="margin-top:0"><b>Ζώνη {z["id"] + 1}</b>, '
          f'{z["n"]} στήλ' + ("η" if z["n"] == 1 else "ες") +
          f' γύρω στο {clock(z["page_start"])}: '
          f'τα συστήματα {esc(", ".join(z["systems"]))} τοποθετούν τις ίδιες '
          f'λέξεις σε σημεία με {gap_txt}, '
          f'άρα ένα από τα δύο διαστήματα ανήκει σε άλλη εκφορά.</p>')
        A('<div class="scroll"><table><tr><th>#</th><th>λέξη</th>'
          '<th>άγκυρα</th><th>διάρκεια</th><th>αβεβαιότητα</th>'
          '<th>απόσταση</th><th>πηγές</th></tr>')
        for i in range(z["first"], z["last"] + 1):
            r = by_i.get(i)
            if not r:
                continue
            a = r["anchor"]
            dur = "" if a["duration"] is None else f'{a["duration"]:.2f} s'
            src = ", ".join(f'{k} {v[0]:.2f} ως {v[1]:.2f}'
                            for k, v in sorted(r["source_intervals"].items()))
            A(f'<tr><td class="num">{r["i"]}</td>'
              f'<td>{esc(r["w"] or "")}</td>'
              f'<td class="num">{secs(a["page_start"])} s</td>'
              f'<td class="num">{dur}</td>'
              f'<td class="num">±{secs(r["time_uncertainty"])} s</td>'
              f'<td class="num">{secs(r["conflict_gap"])} s</td>'
              f'<td class="small">{esc(src)}</td></tr>')
        A('</table></div>')
        A('</div>')
    return out


def method_section(view, data) -> list:
    m = data["manifest"]
    s = view["summary"]
    ps = data["per_system"]
    out = []
    A = out.append
    A('<h2>Πώς μετρήθηκαν τα παραπάνω</h2>')

    A('<details><summary>Πού στέκεται κάθε λέξη στον χρόνο</summary>')
    A(f'<p class="small">Άγκυρα είναι μια στήλη που κάποιο σύστημα χρονομέτρησε '
      f'πραγματικά και για την οποία κανένα άλλο δεν διαφωνεί '
      f'(<code>observed</code> χωρίς σύγκρουση). Οι άγκυρες κρατούν τον δικό '
      f'τους χρόνο. Κάθε ομάδα στηλών ανάμεσα σε δύο άγκυρες μοιράζεται ισομερώς '
      f'το διάστημα που αφήνουν οι δύο, και σημειώνεται με <code>~</code>. '
      f'Έτσι η σειρά που βλέπετε δεν γυρίζει ποτέ πίσω.</p>')
    A(f'<p class="small">Παλιότερα η σελίδα τοποθετούσε κάθε στήλη στο ΜΕΣΟ του '
      f'διαστήματος αβεβαιότητάς της. Όταν δύο συστήματα συγκρούονταν, το '
      f'διάστημα άνοιγε πολύ και το μέσο του έπεφτε μετά το μέσο της επόμενης '
      f'στήλης, οπότε οι λέξεις φαίνονταν να πηγαίνουν πίσω στον χρόνο '
      f'(η στήλη 87 έδειχνε 32.62 s και η 88 έδειχνε 31.89 s).</p>')
    A(f'<p class="small">Σε αυτή τη σελίδα '
      f'{len(view["rows"]) - s["interpolated"]} από {len(view["rows"])} στήλες '
      f'είναι άγκυρες και {s["interpolated"]} πήραν θέση με παρεμβολή. '
      f'{s["clamped"]} άγκυρες χρειάστηκε να σπρωχτούν μπροστά επειδή δύο '
      f'αδιαμφισβήτητες άγκυρες ήρθαν ανάποδα. Η διάρκεια της λέξης και η '
      f'αβεβαιότητα της θέσης είναι δύο διαφορετικά μεγέθη και εμφανίζονται '
      f'χωριστά στο αναδυόμενο πλαίσιο κάθε λέξης, μαζί με τον χρόνο που '
      f'δίνει το κάθε σύστημα χωριστά.</p>')
    A('</details>')

    mk = s["marked"]
    A('<details open><summary>Τι σημαίνει κάθε σημάδι, και πόσες λέξεις '
      'σημαδεύει</summary>')
    A('<p class="small">Κανένα σημάδι δεν μπαίνει χωρίς αιτία που να διαβάζεται. '
      'Περνώντας τον δείκτη πάνω από τη λέξη, ή ανοίγοντάς την, βλέπετε την '
      'πρόταση που το εξηγεί, με τον αριθμό που το προκάλεσε.</p>')
    A('<div class="scroll"><table><tr><th>σημάδι</th><th>σημαίνει</th>'
      '<th>κατώφλι</th><th>λέξεις</th></tr>')
    thresholds = {
        "overlap": (f'{derive.MIN_OVERLAP_SECONDS:.2f} s ΚΑΙ '
                    f'{derive.MIN_OVERLAP_FRACTION * 100:.0f}% της λέξης'),
        "straddle": f'δεύτερη σειρά με {SP_SHARE * 100:.0f}% της λέξης',
        "unresolved": "το διάστημα κρίθηκε αναξιόπιστο από το timing",
        "wide": f'±{derive.WIDE_UNCERTAINTY_SECONDS:.1f} s ή περισσότερο',
        "conflict": "τα δύο διαστήματα δεν τέμνονται καθόλου",
        "interpolated": "η στήλη δεν είναι άγκυρα",
    }
    for key, (badge, text) in FLAG.items():
        A(f'<tr><td class="mono">{badge}</td><td>{text}</td>'
          f'<td class="small">{thresholds[key]}</td>'
          f'<td class="num">{mk.get(key, 0)}</td></tr>')
    A('</table></div>')
    A(f'<p class="small">Το κατώφλι της αβεβαιότητας δεν είναι διακοσμητικό: η '
      f'αβεβαιότητα θέσης σε αυτή τη σελίδα έχει διάμεσο 0.36 s και ενενηκοστό '
      f'εκατοστημόριο 0.72 s, οπότε ένα σημάδι σε κάθε μη μηδενική τιμή θα '
      f'σημάδευε και τις 835 στήλες. Στο '
      f'{derive.WIDE_UNCERTAINTY_SECONDS:.1f} s σημαδεύονται {mk["wide"]}, '
      f'δηλαδή τα διαστήματα που είναι φαρδύτερα από όσο διαρκεί μια λέξη και '
      f'άρα δεν μπορούν να την τοποθετήσουν μέσα στη διάρκειά της.</p>')
    A(f'<p class="small">Οι κάρτες δεν κόβονται πια σε κάθε αλλαγή κατάστασης. '
      f'Πριν, μία λέξη με επικάλυψη ή διφορούμενο ομιλητή έσπαγε την ομιλία σε '
      f'δική της κάρτα, και μια παράδοση λόγου γινόταν έντεκα κάρτες του ενός '
      f'λόγου. Τώρα η κάρτα ακολουθεί τον ομιλητή και η κατάσταση μπαίνει πάνω '
      f'στη λέξη: {s["n_turns"]} κάρτες.</p>')
    A('</details>')

    A('<details><summary>Επικάλυψη ομιλητών, με αυστηρότερο κανόνα</summary>')
    A(f'<p class="small">Ο διαρισμός χαρακτήριζε στήλη ως επικάλυψη μόλις ο '
      f'κανονικός διαρισμός είχε πάνω από μία ενεργή σειρά, ακόμη κι όταν η '
      f'δεύτερη σειρά ακουμπούσε τη λέξη για λίγα χιλιοστά. Εδώ ζητείται ο '
      f'δεύτερος ομιλητής να καλύπτει τουλάχιστον '
      f'{derive.MIN_OVERLAP_SECONDS:.2f} δευτερόλεπτα ΚΑΙ τουλάχιστον '
      f'{derive.MIN_OVERLAP_FRACTION * 100:.0f} τοις εκατό του διαστήματος της '
      f'λέξης, με τις σειρές του να ενώνονται πρώτα ώστε να μη μετρηθεί δύο '
      f'φορές το ίδιο κομμάτι.</p>')
    A(f'<p class="small">Από τις <b class="num">{s["overlap_before"]}</b> στήλες '
      f'που χαρακτηρίζονταν επικάλυψη, επιβεβαιώνονται '
      f'<b class="num">{s["overlap_after"]}</b>. Οι υπόλοιπες κρατούν τη μέτρηση '
      f'της αστοχίας τους και δεν χάνεται τίποτα. Το απόλυτο όριο των '
      f'{derive.MIN_OVERLAP_SECONDS:.2f} δευτερολέπτων δεν μπορεί να περαστεί '
      f'από λέξη συντομότερη από τόσο, και αυτό είναι πραγματική αδυναμία του '
      f'κανόνα: με μόνο το ποσοστιαίο κριτήριο θα επιβεβαιώνονταν '
      f'{s["overlap_fraction_only"]}. Και οι δύο αριθμοί δίνονται εδώ.</p>')
    A(f'<p class="small">Ξεχωριστά από όλα αυτά, το πεδίο '
      f'<code>overlap_fraction</code> του πακέτου ΔΕΝ είναι μέτρο επικάλυψης, '
      f'παρά το όνομά του: είναι το ποσοστό της λέξης που καλύπτει ο ομιλητής '
      f'στον οποίο ανατέθηκε. Είναι κάτω από 1.0 σε '
      f'{s["partial_coverage"]} στήλες, δηλαδή ένα κομμάτι της λέξης πέφτει σε '
      f'σιωπή ή σε ξένη σειρά. Η σελίδα δεν το σημαδεύει, το δείχνει μόνο μέσα '
      f'στο πλαίσιο της λέξης, γιατί από μόνο του δεν λέει ότι κάτι πήγε '
      f'στραβά.</p>')
    A('</details>')

    A('<details><summary>Πώς διαβάζεται ο πίνακας ανά σύστημα</summary>')
    A('<p class="small">Ο πίνακας είναι στην κορυφή της σελίδας. Κάθε σύστημα '
      'στοιχίζεται μόνο του απέναντι στο δημοσιευμένο κείμενο, χωρίς προβολή '
      'της μιας στοίχισης πάνω στην άλλη. Οι επαναλαμβανόμενες λέξεις δίνουν '
      'πολλές ισοδύναμα βέλτιστες στοιχίσεις, οπότε ο διαχωρισμός σε '
      'αντικαταστάσεις, ελλείψεις και εισαγωγές δεν είναι μοναδικός. Το '
      'σύνολο των λαθών είναι.</p>')
    A('<p class="small">Διφορούμενες πράξεις, όπου μια άλλη βέλτιστη στοίχιση '
      'διαφωνεί: '
      + ", ".join(f'{r["name"]} {ps[r["key"]]["counts"]["ambiguous"]}'
                  for r in view["per_system"]) + '.</p>')
    A('</details>')

    A('<details><summary>Προέλευση των δεδομένων</summary>')
    ac = m["audio_check"]
    lags = sorted({c["lag_ms"] for c in ac["checks"]})
    pos = [c["peak_corr"] for c in ac["checks"] if c["peak_corr"] > 0]
    corr_txt = "χωρίς μέτρηση" if not pos else f"{min(pos):.3f}"
    A(f'<p class="small">Δήμος {esc(m["city"])}, συνεδρίαση {esc(m["meeting"])}. '
      f'Απόλυτος χρόνος {m["page_start_abs"]:.3f} ως {m["page_end_abs"]:.3f} '
      f'δευτερόλεπτα. Παρήχθη {esc(m["generated_at"][:19])}Z.</p>')
    A(f'<p class="small">Ο ήχος κόπηκε συνεχόμενα από την πηγή. Ανάμεσα στα δύο '
      f'παράθυρα του benchmark υπάρχει κενό '
      f'{(m["seam_abs"][1] - m["seam_abs"][0]) * 1000:.0f} ms χωρίς κείμενο '
      f'αναφοράς. Έλεγχος με ετεροσυσχέτιση σε {len(ac["checks"])} δείγματα: '
      f'υστέρηση {esc(", ".join(str(x) for x in lags))} ms, ελάχιστη συσχέτιση '
      f'{corr_txt}.</p>')
    A(f'<p class="small">Χρονοσήμανση Soniox: {esc(m["soniox_timestamp_model"])}. '
      f'Χρονοσήμανση Whisper: {esc(m["whisper_timestamp_decode"])}. Και οι δύο '
      f'είναι άλλες αποκωδικοποιήσεις από αυτές που βαθμολογήθηκαν, οπότε οι '
      f'χρόνοι μεταφέρονται μόνο σε λέξεις που ταυτίζονται σε κάθε βέλτιστη '
      f'στοίχιση.</p>')
    A(f'<p class="small">Στοίχιση: παγωμένο '
      f'<code>eval/controlled_eval/msa.py</code>, sha256 '
      f'<code>{esc(m["msa_sha256_16"])}</code>. Κανονικοποίηση: '
      f'{esc(m["scoring_normalisation"])}. Διαρισμός: pyannote precision-2, '
      f'exclusive, με {m["thresholds"]["diarization_pad_s"]:.0f} s συμφραζόμενα '
      f'εκατέρωθεν που κόπηκαν πριν την προβολή.</p>')
    A('</details>')

    A('<details><summary>Τι δεν είναι αυτή η σελίδα</summary>')
    A('<p class="small">Είναι όργανο παρατήρησης, όχι μέτρηση. Δεν υπάρχει πύλη, '
      'δεν υπάρχει διάστημα εμπιστοσύνης, δεν βγαίνει συμπέρασμα για το ποιο '
      'σύστημα είναι καλύτερο. Δύο παράθυρα δεν αρκούν για τίποτα από αυτά.</p>')
    A('<p class="small">Το δημοσιευμένο κείμενο δεν είναι η αλήθεια του ήχου. '
      'Είναι το κείμενο του OpenCouncil. Ό,τι μετριέται εδώ είναι συμφωνία με '
      'αυτό, όχι πιστότητα σε αυτό που ακούγεται. Σε ελεγμένο δείγμα, 23.7 τοις '
      'εκατό από όσα χρεώνονται ως εισαγωγές του μοντέλου μας ήταν λέξεις που '
      'ένας άνθρωπος όντως άκουσε.</p>')
    A('</details>')
    return out


# --------------------------------------------------------------------- main
def render(data: dict) -> str:
    view = derive.build_view(data)
    m = data["manifest"]

    body = []
    body.append('<div class="wrap">')
    body.append('<h1>Τρία συστήματα, μία σύνθεση και ο ήχος, σε '
                f'{m["page_duration"]:.0f} δευτερόλεπτα</h1>')
    body.append(f'<p class="sub">Δήμος {esc(m["city"])}, συνεδρίαση '
                f'{esc(m["meeting"])}.</p>')
    body += summary_section(view, data)
    body += player_section(view, data)
    turns_html, det_ids = turn_section(view, data)
    body += turns_html
    body += ledger_section(view)
    body += clock_section(view, data)
    body += zones_section(view)
    body += method_section(view, data)

    det = {}
    for r in view["rows"]:
        if r["i"] not in det_ids:
            continue
        a = r["anchor"]
        det[str(r["i"])] = {
            "sc": r["systems"]["scribe"], "so": r["systems"]["soniox"],
            "wh": r["systems"]["whisper"], "w": r["w"], "wr": r["w_reason"],
            "rf": r["ref_word"], "rop": r["ref_op"],
            "s": r["pos"]["t"], "e": r["pos"]["end"],
            "ip": r["pos"]["interpolated"],
            "un": r["time_uncertainty"], "tm": r["time_method"],
            "cg": r["conflict_gap"], "om": r["ref_omission_suspect"],
            "sp": r.get("card_speaker") or r.get("speaker"),
            "cov": r["speaker_coverage"],
            "fl": [flag_text(w) for w in r["warnings"]],
            "an": (round(a["page_start"], 3)
                   if a["page_start"] is not None else None),
            "src": {k: {"s": (round(v["start"], 3)
                              if v["start"] is not None else None),
                        "e": (round(v["end"], 3)
                              if v["end"] is not None else None),
                        "c": v["conf"], "p": v["provenance"]}
                    for k, v in r["source_detail"].items()},
        }

    colors = {s: PALETTE[i % len(PALETTE)]
              for i, s in enumerate(data["diar"]["speakers"])}
    reg = sorted(data["diar"]["regular"], key=lambda t: t["page_start"])
    mult = []
    for i in range(len(reg)):
        for j in range(i + 1, len(reg)):
            a = max(reg[i]["page_start"], reg[j]["page_start"])
            b = min(reg[i]["page_end"], reg[j]["page_end"])
            if b > a:
                mult.append([round(a, 3), round(b, 3)])
    led = [{"sy": e["system"], "ty": e["type"], "t": e["t"],
            "sp": e["speaker"], "rf": e["ref_word"], "hy": e["hyp_word"],
            "l": e["left"], "r": e["right"],
            "sc": e["tokens"].get("scribe"), "so": e["tokens"].get("soniox"),
            "wh": e["tokens"].get("whisper"), "w": e["w"],
            "rs": e["reason"], "ok": e["right_systems"],
            "b": e["time_borrowed"], "me": e["time_method"],
            "col": e["anchor_column"], "am": e["ambiguous"]}
           for e in view["ledger"]]
    ticks = {k: sorted(round(v["start"], 2) for r in view["rows"]
                       for v in [(r["source_detail"] or {}).get(k)]
                       if v and v.get("start") is not None)
             for k in ("soniox", "whisper")}
    payload = {
        "duration": m["page_duration"],
        "det": det,
        "led": led,
        "speakers": sorted({e["speaker"] for e in view["ledger"]
                            if e["speaker"]}),
        "ticks": ticks,
        "lane": {
            "turns": [[round(t["page_start"], 3), round(t["page_end"], 3),
                       colors.get(t["speaker"], "#8a8a8a")]
                      for t in data["diar"]["exclusive"]],
            "mult": mult,
            "seam": round(data["seam_page"][0], 3),
        },
    }
    body.append('<script type="application/json" id="payload">'
                + embed(payload) + '</script>')
    body.append(f'<script>{JS}</script>')
    body.append('</div>')

    return (
        '<!doctype html>\n<html lang="el">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Διαγνωστικό τριών συστημάτων, 299 δευτερόλεπτα</title>\n'
        f'<style>{CSS}</style>\n'
        '</head>\n<body>\n'
        + "\n".join(body)
        + '\n</body>\n</html>\n'
    )


def main():
    out = SC / "tsfusion-2026-08"
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    html = render(data)
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"{out / 'index.html'}  {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
