// Runs deletion_audit.html's own script under node with a stub DOM and a fetch I
// control, to pin the four ways an answer or a comment could be lost. A browser
// would be better; this at least tests the real code rather than a description
// of it.  node deletion_audit_jstest.js
const fs = require("fs"), path = require("path"), vm = require("vm");

const CIDS = ["aaaa000000000000", "bbbb000000000000", "cccc000000000000",
              "dddd000000000000", "eeee000000000000"];
const fails = [];
const check = (n, c, d) => { console.log((c ? "  ok   " : "  FAIL ") + n + (c ? "" : "  [" + d + "]")); if (!c) fails.push(n); };

function el(tag) {
  const e = {
    tagName: (tag || "div").toUpperCase(), children: [], style: {}, dataset: {},
    className: "", _text: "", value: "", attrs: {},
    get textContent() { return this._text; },
    set textContent(v) { this._text = String(v); this.children.length = 0; },
    set innerHTML(v) { this._text = ""; this.children.length = 0; },
    append(...n) { for (const x of n) this.children.push(x); },
    appendChild(n) { this.children.push(n); return n; },
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k]; },
    addEventListener() {}, focus() {}, blur() {}, load() {},
    play() { return Promise.resolve(); },
    closest() { return null },
  };
  return e;
}

function harness(state, fetchImpl, mutate) {
  const src = fs.readFileSync(path.join(__dirname, "deletion_audit.html"), "utf8");
  const js = src.slice(src.indexOf("<script>") + 8, src.lastIndexOf("</script>"))
    .replace("__DATA__", JSON.stringify(CIDS.map((c, i) =>
      ({ cid: c, L: "L" + i, R: "R" + i, lost: ["w" + i] }))))
    .replace("__STATE__", JSON.stringify(state))
    .replace("__TOKEN__", "T")
    + "\n;globalThis.__probe = () => ({ ST, dirty: Array.from(dirty), cur, busy: busy.size });"
    + "\n;globalThis.__api = { answer, goto, scheduleNote, flushNotes };";
  const js2 = mutate ? mutate(js) : js;
  const byId = {};
  for (const id of ["au", "ctx", "opts", "note", "prog", "status", "done", "prev", "next", "rep"])
    byId[id] = el(id === "note" ? "textarea" : "div");
  const doc = {
    getElementById: (i) => byId[i], querySelector: () => el(),
    createElement: el, createTextNode: (t) => ({ nodeValue: t }),
    addEventListener() {}, activeElement: null,
    visibilityState: "visible",
  };
  const ctx = {
    document: doc, window: { addEventListener() {} }, navigator: { sendBeacon() { return true } },
    fetch: fetchImpl, setTimeout, clearTimeout, console, Promise, JSON, Object, Number,
    String, Array, Map, Set, Blob: function () {}, Error,
  };
  ctx.globalThis = ctx;
  vm.runInNewContext(js2, ctx);
  return ctx;
}

const ok = (body) => Promise.resolve({ ok: true, json: () => Promise.resolve(Object.assign({ ok: true, answered: 1 }, body)) });

(async () => {
  console.log("\nA. answering saves then advances");
  {
    const sent = [];
    const c = harness({}, (u, o) => { sent.push(JSON.parse(o.body)); return ok(); });
    await c.__api.answer(2);
    const p = c.__probe();
    check("posted the answer", sent.length === 1 && sent[0].a === 2 && sent[0].cid === CIDS[0], JSON.stringify(sent));
    check("token attached", sent[0].token === "T");
    check("advanced to clip 2", p.cur === 1, "cur=" + p.cur);
    check("recorded locally", p.ST[CIDS[0]].a === 2);
  }

  console.log("\nB. navigating while a save is in flight must not skip a clip");
  {
    let release;
    const gate = new Promise((r) => { release = r; });
    const c = harness({}, async () => { await gate; return ok(); });
    const pending = c.__api.answer(1);        // on clip 0, not awaited
    c.__api.goto(1);                          // he moves on before it lands
    check("navigation was immediate", c.__probe().cur === 1);
    release();
    await pending;
    const p = c.__probe();
    check("answer landed on the clip it was given for", p.ST[CIDS[0]].a === 1);
    check("did NOT advance past where he navigated", p.cur === 1, "cur=" + p.cur);
    check("busy released", p.busy === 0);

    // The check above is only worth anything if it fails without the guard.
    let rel2; const gate2 = new Promise((r) => { rel2 = r; });
    const m = harness({}, async () => { await gate2; return ok(); },
      (src) => src.replace("if (cur === idx && cur < N - 1) cur++;", "if (cur < N - 1) cur++;"));
    const pend2 = m.__api.answer(1);
    m.__api.goto(1);
    rel2();
    await pend2;
    check("MUTATION: without the guard the same scenario skips a clip",
          m.__probe().cur === 2, "mutant cur=" + m.__probe().cur + " (test is vacuous)");
  }

  console.log("\nC. a comment that fails to save stays dirty and blocks the answer");
  {
    let mode = "fail";
    const c = harness({}, async () => mode === "fail"
      ? { ok: false, status: 500, json: () => Promise.resolve({}) } : ok());
    c.__api.scheduleNote(CIDS[0], "ακούγεται καθαρά");
    await c.__api.flushNotes().catch(() => {});
    let p = c.__probe();
    check("comment kept as dirty after a failed write", p.dirty.length === 1 && p.dirty[0][1] === "ακούγεται καθαρά", JSON.stringify(p.dirty));
    check("comment NOT recorded as saved", !(p.ST[CIDS[0]] && p.ST[CIDS[0]].note));
    await c.__api.answer(3);
    p = c.__probe();
    check("did not advance while the comment is unsaved", p.cur === 0, "cur=" + p.cur);
    check("no answer recorded either", !(p.ST[CIDS[0]] && p.ST[CIDS[0]].a));
    mode = "ok";
    await c.__api.answer(3);
    p = c.__probe();
    check("recovers once the server is back: comment saved", p.ST[CIDS[0]].note === "ακούγεται καθαρά");
    check("recovers: answer saved and advanced", p.ST[CIDS[0]].a === 3 && p.cur === 1);
    check("nothing left dirty", p.dirty.length === 0);
  }

  console.log("\nD. editing, then reverting, while the first write is in flight");
  {
    let release, sent = [];
    const gate = new Promise((r) => { release = r; });
    let first = true;
    const c = harness({ [CIDS[0]]: { note: "old" } }, async (u, o) => {
      sent.push(JSON.parse(o.body).note);
      if (first) { first = false; await gate; }
      return ok();
    });
    c.__api.scheduleNote(CIDS[0], "new");
    const p1 = c.__api.flushNotes();
    await new Promise((r) => setTimeout(r, 5));
    c.__api.scheduleNote(CIDS[0], "old");     // reverted before "new" landed
    release();
    await p1;
    await c.__api.flushNotes();
    const p = c.__probe();
    check("the reverted text is what ends up saved", p.ST[CIDS[0]].note === "old", JSON.stringify(sent) + " ST=" + JSON.stringify(p.ST[CIDS[0]]));
    check("it was actually re-sent, not just assumed", sent.join(",") === "new,old", sent.join(","));
    check("nothing left dirty", p.dirty.length === 0, JSON.stringify(p.dirty));
  }

  console.log("\n" + (fails.length ? fails.length + " FAILED: " + fails.join(", ") : "ALL PASS"));
  process.exit(fails.length ? 1 : 0);
})();
