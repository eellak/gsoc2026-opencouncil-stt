/* Gold-set verification tool - issue #21.
   Vanilla JS on purpose: it is served by eval/controlled_eval/audit_server.py,
   which already gives byte-range audio seeking, merge-on-save and atomic writes.
   No build step, no node_modules, works offline, one command to start. */
"use strict";
const SPKCOL = ["--a","--b","--c","--d","--e","--f"];
const LET = ["A","B","C","D","E","F"];
const PROTOCOL = "gold-set-2026-08-16";
const SCHEMA = 1;

let CELLS = [], idx = 0, A = {}, ctx = null, buf = null, wave = null;
let audio = new Audio(), playEnd = null, rate = 1, undoStack = [], selStart = null, selEnd = null;
let heardEnd = 0, tickT = Date.now(), phaseNow = "a";

const $ = s => document.querySelector(s);
const now = () => Date.now();
const cur = () => CELLS[idx];
const fmt = t => (t < 0 ? "-" : "") + Math.abs(t).toFixed(2) + "s";

/* ---------- persistence ---------------------------------------------- */
function blank(c) {
  return { v: SCHEMA, protocol: PROTOCOL, cell: c.id, t: now(),
    mode: c.calib ? "blank_first" : "hybrid", status: "unseen",
    a: null, blank: null, b: null, c: null, time: { a_ms: 0, b_ms: 0, c_ms: 0, plays: 0 } };
}
function ans() { if (!A[cur().id]) A[cur().id] = blank(cur()); return A[cur().id]; }
function touch() { ans().t = now(); localStorage.setItem("gold-set-v1", JSON.stringify(A)); queueSave(); }
let saveT = null, saving = false;
function queueSave() { clearTimeout(saveT); saveT = setTimeout(push, 1500); }
function push() {
  if (saving) { queueSave(); return; }
  saving = true;
  fetch("/save", { method: "POST", body: JSON.stringify(A) })
    .then(r => { $("#sync").textContent = r.ok ? "saved" : "save error"; $("#sync").className = "pill " + (r.ok ? "ok" : "warn"); })
    .catch(() => { $("#sync").textContent = "offline (local only)"; $("#sync").className = "pill warn"; })
    .finally(() => { saving = false; });
}
addEventListener("beforeunload", () => navigator.sendBeacon("/save", JSON.stringify(A)));
setInterval(push, 60000);
setInterval(() => {
  const a = ans(); const d = now() - tickT; tickT = now();
  if (document.visibilityState === "visible" && d < 20000) a.time[phaseNow + "_ms"] += d;
  renderClock();
}, 5000);

/* ---------- audio ----------------------------------------------------- */
function loadAudio(c) {
  audio.pause(); audio.src = c.clip; audio.playbackRate = rate; playEnd = null; heardEnd = 0;
  buf = null; wave = null; draw();
  fetch(c.clip).then(r => r.arrayBuffer()).then(b => {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
    return ctx.decodeAudioData(b);
  }).then(d => { buf = d; wave = peaks(d, 1200); draw(); }).catch(() => {});
}
function peaks(d, n) {
  const ch = d.getChannelData(0), step = Math.floor(ch.length / n), out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    let lo = 1, hi = -1;
    for (let j = i * step, e = Math.min(ch.length, j + step); j < e; j++) { const v = ch[j]; if (v < lo) lo = v; if (v > hi) hi = v; }
    out[i * 2] = lo; out[i * 2 + 1] = hi;
  }
  return out;
}
function play(s, e) {
  const a = ans(); a.time.plays++;
  audio.currentTime = Math.max(0, s); playEnd = e; audio.playbackRate = rate; audio.play();
}
audio.addEventListener("timeupdate", () => {
  if (audio.currentTime > heardEnd) heardEnd = audio.currentTime;
  if (playEnd !== null && audio.currentTime >= playEnd) { audio.pause(); playEnd = null; }
  draw();
});
audio.addEventListener("ended", () => { heardEnd = cur().dur; draw(); renderA(); });
document.addEventListener("visibilitychange", () => { if (document.hidden) audio.pause(); });

/* ---------- waveform -------------------------------------------------- */
function draw() {
  const cv = $("#wave"); if (!cv) return;
  const c = cur(), W = cv.clientWidth, LANE = 16, nl = laneCount();
  const H = 66 + LANE * Math.max(nl, 1) + 10;
  cv.style.height = H + "px";
  cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
  const g = cv.getContext("2d"); g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  g.clearRect(0, 0, W, H);
  const x = t => t / c.dur * W;
  g.fillStyle = getCss("--core"); g.fillRect(x(c.lead), 0, x(c.core) , 66);
  if (wave) {
    g.fillStyle = "#5b6478";
    const n = wave.length / 2;
    for (let i = 0; i < n; i++) {
      const px = i / n * W, lo = wave[i * 2], hi = wave[i * 2 + 1];
      g.fillRect(px, 33 - hi * 30, Math.max(1, W / n), Math.max(1, (hi - lo) * 30));
    }
  }
  if (selStart !== null && selEnd !== null) {
    g.fillStyle = "rgba(255,210,74,.18)"; g.fillRect(x(Math.min(selStart, selEnd)), 0, x(Math.abs(selEnd - selStart)), 66);
  }
  const a = A[c.id];
  if (a && a.b && phaseNow !== "a") {
    const lanes = laneMap();
    a.b.blocks.forEach(b => {
      const y = 70 + lanes[b.spk] * LANE;
      g.fillStyle = getCss(SPKCOL[LET.indexOf(b.spk) % 6]);
      g.globalAlpha = b.t_src === "human" ? 1 : .55;
      g.fillRect(x(b.s), y, Math.max(2, x(b.e - b.s)), LANE - 4);
      g.globalAlpha = 1;
      g.fillStyle = "#0d0f14"; g.font = "10px system-ui";
      g.fillText(b.spk, x(b.s) + 3, y + 10);
    });
  }
  g.strokeStyle = "#fff"; g.beginPath(); g.moveTo(x(audio.currentTime), 0); g.lineTo(x(audio.currentTime), 66); g.stroke();
  g.strokeStyle = "#ffd24a"; g.setLineDash([3, 3]);
  [c.lead, c.lead + c.core].forEach(t => { g.beginPath(); g.moveTo(x(t), 0); g.lineTo(x(t), H); g.stroke(); });
  g.setLineDash([]);
}
function getCss(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function laneMap() {
  const a = A[cur().id], m = {}; let i = 0;
  ((a && a.b) ? a.b.blocks : []).forEach(b => { if (!(b.spk in m)) m[b.spk] = i++; });
  return m;
}
function laneCount() { return Object.keys(laneMap()).length; }

/* ---------- waveform interaction ------------------------------------- */
function wireWave() {
  const cv = $("#wave");
  let drag = null;
  const tAt = ev => {
    const r = cv.getBoundingClientRect();
    return Math.max(0, Math.min(cur().dur, (ev.clientX - r.left) / r.width * cur().dur));
  };
  cv.addEventListener("mousedown", ev => {
    const t = tAt(ev), r = cv.getBoundingClientRect(), y = ev.clientY - r.top;
    const a = A[cur().id];
    if (y > 70 && a && a.b && phaseNow !== "a") {
      const lanes = laneMap(), li = Math.floor((y - 70) / 16);
      const hit = a.b.blocks.find(b => lanes[b.spk] === li && t > b.s - .35 && t < b.e + .35);
      if (hit) {
        drag = { b: hit, edge: Math.abs(t - hit.s) < Math.abs(t - hit.e) ? "s" : "e" };
        snapshot(); return;
      }
    }
    selStart = t; selEnd = t; drag = { sel: true };
  });
  addEventListener("mousemove", ev => {
    if (!drag) return;
    const t = tAt(ev);
    if (drag.sel) { selEnd = t; }
    else {
      if (drag.edge === "s") drag.b.s = Math.min(t, drag.b.e - .1); else drag.b.e = Math.max(t, drag.b.s + .1);
      drag.b.t_src = "human"; autoOverlap();
    }
    draw();
  });
  addEventListener("mouseup", () => {
    if (drag && !drag.sel) { renderB(); touch(); }
    if (drag && drag.sel && Math.abs(selEnd - selStart) < .05) {
      const t = selStart; selStart = selEnd = null; play(t, cur().dur);
    }
    drag = null; draw();
  });
}

/* ---------- pass A ---------------------------------------------------- */
function renderA() {
  const c = cur(), a = ans();
  const heardAll = heardEnd >= c.dur - 1.2;
  $("#a-heard").textContent = heardAll ? "ακούστηκε ολόκληρο" : `ακούστηκε ${heardEnd.toFixed(0)}/${c.dur.toFixed(0)}s`;
  $("#a-heard").className = "pill " + (heardAll ? "ok" : "warn");
  // Restore ONLY from a submitted answer. Before submit the DOM is the source of
  // truth: renderA also runs on the selects' own onchange and on audio 'ended', so
  // writing here unconditionally wiped the pick the instant it was made and left
  // the submit button disabled forever. go() clears the selects on cell change.
  if (a.a) {
    $("#a-ov").value = a.a.overlap || "";
    $("#a-voices").value = a.a.max_voices || "";
  }
  $("#a-submit").disabled = !(heardAll && $("#a-ov").value && $("#a-voices").value);
  $("#pa").classList.toggle("on", phaseNow === "a");
}
function submitA() {
  const a = ans();
  a.a = { overlap: $("#a-ov").value, max_voices: +$("#a-voices").value, done_at: now(),
          heard_sec: +heardEnd.toFixed(2) };
  a.status = "a_done";
  phaseNow = cur().calib && !a.blank ? "blank" : "b";
  if (phaseNow === "b" && !a.b) initB();
  touch(); render();
}

/* ---------- blank calibration ---------------------------------------- */
function submitBlank() {
  const a = ans();
  a.blank = { text: $("#blank-text").value, done_at: now() };
  phaseNow = "b"; if (!a.b) initB(); touch(); render();
}

/* ---------- pass B ---------------------------------------------------- */
function initB() {
  const c = cur(), a = ans(), spkMap = {}; let n = 0;
  const blocks = c.turns.filter(t => t.e > c.lead - 1.5 && t.s < c.lead + c.core + 1.5).map((t, i) => {
    if (!(t.spk in spkMap)) spkMap[t.spk] = LET[n++ % 6];
    return { id: "p" + i, s: +t.s.toFixed(3), e: +t.e.toFixed(3), spk: spkMap[t.spk],
             text: t.text, t_src: "prefill", text_unc: false, spk_unc: false, ov_with: [] };
  });
  a.b = { blocks, excluded: null, prefill_source: c.prefill_source, done_at: null };
  autoOverlap();
}
function snapshot() {
  const a = ans(); if (!a.b) return;
  undoStack.push(JSON.stringify(a.b.blocks)); if (undoStack.length > 60) undoStack.shift();
}
function undo() {
  const a = ans(); if (!a.b || !undoStack.length) return;
  a.b.blocks = JSON.parse(undoStack.pop()); autoOverlap(); renderB(); draw(); touch();
}
function autoOverlap() {
  const a = ans(); if (!a.b) return;
  a.b.blocks.forEach(b => {
    b.ov_with = a.b.blocks.filter(o => o.id !== b.id && o.spk !== b.spk &&
      Math.min(o.e, b.e) - Math.max(o.s, b.s) > 0.15).map(o => o.id);
  });
}
function addSpeaker() {
  const a = ans(); if (!a.b) return;
  snapshot();
  let s, e;
  if (selStart !== null && selEnd !== null && Math.abs(selEnd - selStart) > .1) {
    s = Math.min(selStart, selEnd); e = Math.max(selStart, selEnd);
  } else { s = audio.currentTime; e = Math.min(cur().dur, s + 2); }
  const used = new Set(a.b.blocks.map(b => b.spk));
  const free = LET.find(l => !used.has(l)) || "F";
  a.b.blocks.push({ id: "h" + now().toString(36), s: +s.toFixed(3), e: +e.toFixed(3),
                    spk: free, text: "", t_src: "human", text_unc: false, spk_unc: false, ov_with: [] });
  a.b.blocks.sort((x, y) => x.s - y.s);
  autoOverlap(); renderB(); draw(); touch();
}
function splitAt(b, pos) {
  const a = ans(); snapshot();
  const t = b.text, i = a.b.blocks.indexOf(b);
  const cut = (b.s + b.e) / 2;
  const nb = { id: "h" + now().toString(36), s: cut, e: b.e, spk: b.spk, text: t.slice(pos).trim(),
               t_src: "human", text_unc: false, spk_unc: false, ov_with: [] };
  b.text = t.slice(0, pos).trim(); b.e = cut;
  a.b.blocks.splice(i + 1, 0, nb); autoOverlap(); renderB(); draw(); touch();
}
function renderB() {
  const c = cur(), a = ans(), host = $("#blocks");
  $("#pb").classList.toggle("on", phaseNow === "b");
  if (!a.b) { host.innerHTML = ""; return; }
  host.innerHTML = "";
  a.b.blocks.forEach(b => {
    const el = document.createElement("div"); el.className = "blk";
    const inCore = b.e > c.lead && b.s < c.lead + c.core;
    el.innerHTML =
      `<div class="tm">${fmt(b.s - c.lead)}<br>${fmt(b.e - c.lead)}` +
      `<br><span class="sm">${inCore ? "core" : "context"}</span></div>` +
      `<div><select class="spk-sel">${LET.map(l => `<option ${l === b.spk ? "selected" : ""}>${l}</option>`).join("")}</select>` +
      `<div class="sm" style="margin-top:4px">${b.t_src === "human" ? "χρόνοι: εσύ" : "χρόνοι: prefill"}</div>` +
      (b.ov_with.length ? `<div class="ovtag" style="margin-top:4px">ταυτόχρονα</div>` : "") + `</div>` +
      `<div><textarea>${esc(b.text)}</textarea>` +
      `<div class="row"><label><input type="checkbox" class="tu" ${b.text_unc ? "checked" : ""}> κείμενο αβέβαιο</label>` +
      `<label><input type="checkbox" class="su" ${b.spk_unc ? "checked" : ""}> ομιλητής αβέβαιος</label></div></div>` +
      `<div><button class="pl">▶</button><br><button class="sp2" title="χώρισε στον κέρσορα">✂</button>` +
      `<br><button class="danger del">✕</button></div>`;
    el.querySelector(".pl").onclick = () => play(Math.max(0, b.s - .4), Math.min(c.dur, b.e + .4));
    el.querySelector(".del").onclick = () => { snapshot(); a.b.blocks.splice(a.b.blocks.indexOf(b), 1); autoOverlap(); renderB(); draw(); touch(); };
    el.querySelector(".sp2").onclick = () => { const ta = el.querySelector("textarea"); splitAt(b, ta.selectionStart || 0); };
    el.querySelector(".spk-sel").onchange = ev => { snapshot(); b.spk = ev.target.value; autoOverlap(); renderB(); draw(); touch(); };
    const ta = el.querySelector("textarea");
    ta.oninput = () => { b.text = ta.value; touch(); };
    ta.onfocus = () => { window.__block = b; };
    el.querySelector(".tu").onchange = ev => { b.text_unc = ev.target.checked; touch(); };
    el.querySelector(".su").onchange = ev => { b.spk_unc = ev.target.checked; touch(); };
    host.appendChild(el);
  });
}
function esc(s) { return (s || "").replace(/[&<>]/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m])); }
function submitB() {
  const a = ans();
  const bad = a.b.blocks.filter(b => b.e > cur().lead && b.s < cur().lead + cur().core && !b.text.trim() && !b.text_unc);
  if (bad.length && !confirm(`${bad.length} τμήμα(τα) στον πυρήνα είναι κενά. Συνέχεια;`)) return;
  a.b.done_at = now(); a.status = "b_done"; phaseNow = "c"; if (!a.c) initC();
  touch(); render();
}
function excludeCell() {
  const r = prompt("Λόγος εξαίρεσης: unintelligible / overlap_impossible / music_noise / not_speech");
  if (!r) return;
  const a = ans(); a.b = a.b || { blocks: [] }; a.b.excluded = r; a.status = "excluded";
  touch(); next();
}

/* ---------- pass C: omission audit ------------------------------------ */
function initC() {
  const c = cur(), a = ans(), flags = [];
  const words = a.b.blocks.filter(b => b.text.trim()).map(b => [b.s, b.e]);
  const covered = t => words.some(w => t >= w[0] - .3 && t <= w[1] + .3);
  c.speech.forEach(sp => {                      // pyannote speech with no words of ours
    let s = null;
    for (let t = sp[0]; t <= sp[1]; t += .1) {
      if (!covered(t)) { if (s === null) s = t; }
      else if (s !== null) { if (t - s > .8) flags.push(mk(s, t, "nowords")); s = null; }
    }
    if (s !== null && sp[1] - s > .8) flags.push(mk(s, sp[1], "nowords"));
  });
  (c.alt || []).forEach(u => {                  // published transcript says speech here
    if (u.e > c.lead && u.s < c.lead + c.core) flags.push(mk(u.s, u.e, "altonly", u.text));
  });
  a.c = { flags: flags.filter(f => f.e > c.lead - .5 && f.s < c.lead + c.core + .5).slice(0, 12), done_at: null };
  function mk(s, e, kind, text) {
    return { id: kind + Math.round(s * 100), s: +s.toFixed(2), e: +e.toFixed(2), kind,
             hint: text || "", verdict: null };
  }
}
function renderC() {
  const a = ans(), host = $("#flags");
  $("#pc").classList.toggle("on", phaseNow === "c");
  if (!a.c) { host.innerHTML = ""; return; }
  if (!a.c.flags.length) { host.innerHTML = `<p class="sm">Καμία ένδειξη χαμένης ομιλίας.</p>`; return; }
  host.innerHTML = "";
  a.c.flags.forEach(f => {
    const el = document.createElement("div"); el.className = "flag" + (f.verdict ? " done" : "");
    el.innerHTML = `<div class="row"><button class="pl">▶ ${fmt(f.s - cur().lead)}–${fmt(f.e - cur().lead)}</button>` +
      `<span class="sm">${f.kind === "nowords" ? "ήχος με ομιλία, χωρίς λέξεις στο κείμενό σου" : "το δημοσιευμένο κείμενο έχει εδώ λόγια"}</span></div>` +
      (f.hint ? `<div class="sm" style="margin-top:4px">«${esc(f.hint)}»</div>` : "") +
      `<div class="row"><button data-v="missing">χάθηκε ομιλία</button>` +
      `<button data-v="present">ήδη γραμμένο</button>` +
      `<button data-v="nonspeech">δεν είναι ομιλία</button>` +
      `<span class="sm">${f.verdict || ""}</span></div>`;
    el.querySelector(".pl").onclick = () => play(Math.max(0, f.s - .5), Math.min(cur().dur, f.e + .5));
    el.querySelectorAll("[data-v]").forEach(b => b.onclick = () => { f.verdict = b.dataset.v; renderC(); touch(); });
    host.appendChild(el);
  });
}
function submitC() {
  const a = ans();
  const left = a.c.flags.filter(f => !f.verdict).length;
  if (left && !confirm(`${left} ενδείξεις χωρίς απάντηση. Ολοκλήρωση;`)) return;
  a.c.done_at = now(); a.status = "complete"; touch(); next();
}

/* ---------- navigation ------------------------------------------------ */
function go(i) {
  if (i < 0 || i >= CELLS.length) return;
  audio.pause(); idx = i; undoStack = []; selStart = selEnd = null;
  const a = ans();
  phaseNow = a.status === "unseen" ? "a" : a.status === "a_done" ? (cur().calib && !a.blank ? "blank" : "b")
    : a.status === "b_done" ? "c" : "c";
  if (phaseNow === "b" && !a.b) initB();
  if (phaseNow === "c" && !a.c) initC();
  if (!a.a) { $("#a-ov").value = ""; $("#a-voices").value = ""; }  // fresh cell starts empty
  loadAudio(cur()); render(); location.hash = cur().id;
}
function next() { const n = CELLS.findIndex((c, i) => i > idx && !["complete", "excluded"].includes((A[c.id] || {}).status)); go(n < 0 ? Math.min(idx + 1, CELLS.length - 1) : n); }

/* ---------- render ---------------------------------------------------- */
function renderClock() {
  const spent = Object.values(A).reduce((s, a) => s + (a.time ? a.time.a_ms + a.time.b_ms + a.time.c_ms : 0), 0);
  const done = Object.values(A).filter(a => ["complete", "excluded"].includes(a.status)).length;
  $("#clock").textContent = `${(spent / 60000).toFixed(0)} λεπτά ενεργά · ${done}/${CELLS.length} κελιά`;
  $("#clock").className = "pill" + (spent > 150 * 60000 ? " warn" : "");
  if (done) $("#rate").textContent = `${(spent / 60000 / done).toFixed(1)} λεπτά/κελί`;
}
function render() {
  const c = cur(), a = ans();
  $("#title").textContent = `${idx + 1}/${CELLS.length} · ${c.id}`;
  $("#tier").textContent = c.tier + (c.calib ? " · τυφλό" : "") + (c.warmup ? " · ζέσταμα" : "");
  renderA(); renderB(); renderC(); draw(); renderClock();
  $("#pblank").classList.toggle("on", phaseNow === "blank");
  $("#pblank").style.display = c.calib ? "" : "none";
  $("#blank-text").value = (a.blank && a.blank.text) || "";
  $("#b-locked").style.display = phaseNow === "a" ? "" : "none";
}

/* ---------- keyboard -------------------------------------------------- */
addEventListener("keydown", ev => {
  const typing = /^(TEXTAREA|INPUT|SELECT)$/.test(document.activeElement.tagName);
  if (ev.key === " " && (!typing || ev.ctrlKey)) {
    ev.preventDefault(); if (audio.paused) { playEnd = null; audio.play(); } else audio.pause(); return;
  }
  if (ev.key === "Enter" && ev.ctrlKey && typing && window.__block) { ev.preventDefault(); splitAt(window.__block, document.activeElement.selectionStart); return; }
  if (typing) return;
  const d = ev.shiftKey ? 3 : 1;
  if (ev.key === "ArrowLeft") { ev.preventDefault(); audio.currentTime = Math.max(0, audio.currentTime - d); }
  if (ev.key === "ArrowRight") { ev.preventDefault(); audio.currentTime = Math.min(cur().dur, audio.currentTime + d); }
  if (ev.key === "Enter") { ev.preventDefault(); play(cur().lead, cur().lead + cur().core); }
  if (ev.key === "o" && phaseNow === "b") { ev.preventDefault(); addSpeaker(); }
  if (ev.key === "z" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); undo(); }
  if (ev.key === "r") { rate = rate === 1 ? .75 : rate === .75 ? .5 : 1; audio.playbackRate = rate; $("#rate2").textContent = rate + "x"; }
});

/* ---------- boot ------------------------------------------------------ */
fetch("cells.json").then(r => r.json()).then(d => {
  CELLS = d.cells;
  try { A = JSON.parse(localStorage.getItem("gold-set-v1") || "{}"); } catch (e) { A = {}; }
  fetch("answers.json").then(r => r.ok ? r.json() : {}).then(srv => {
    Object.entries(srv || {}).forEach(([k, v]) => { if (!A[k] || (v.t || 0) > (A[k].t || 0)) A[k] = v; });
  }).catch(() => {}).finally(() => {
    wireWave();
    $("#a-ov").onchange = renderA; $("#a-voices").onchange = renderA;
    $("#a-submit").onclick = submitA;
    $("#blank-submit").onclick = submitBlank;
    $("#b-submit").onclick = submitB; $("#b-add").onclick = addSpeaker;
    $("#b-undo").onclick = undo; $("#b-exclude").onclick = excludeCell;
    $("#c-submit").onclick = submitC;
    $("#prev").onclick = () => go(idx - 1); $("#next").onclick = () => go(idx + 1);
    $("#export").onclick = () => {
      const b = new Blob([JSON.stringify(A, null, 1)], { type: "application/json" });
      const u = URL.createObjectURL(b), a2 = document.createElement("a");
      a2.href = u; a2.download = "gold-set-answers.json"; a2.click();
    };
    const h = location.hash.slice(1);
    const i = CELLS.findIndex(c => c.id === h);
    go(i >= 0 ? i : 0);
  });
});
