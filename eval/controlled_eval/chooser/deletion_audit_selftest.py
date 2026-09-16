#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-test for the deletion audit. Synthetic rows in a temp dir; real HTTP.

Touches no real sample, no real clip, no real verdict file. The two properties it
exists to protect: the browser is never sent the answer, and a recorded verdict
survives a restart.

    python3 deletion_audit_selftest.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
PORT = int(os.environ.get("SELFTEST_PORT", "8879"))
BASE = f"http://127.0.0.1:{PORT}"
FAILS = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def req(path, data=None, ctype="application/json", headers=None):
    r = urllib.request.Request(BASE + path,
                               data=json.dumps(data).encode() if data is not None else None,
                               method="POST" if data is not None else "GET")
    if data is not None:
        r.add_header("Content-Type", ctype)
    for k, v in (headers or {}).items():
        r.add_header(k, v)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def build(root):
    d = root / "chooser-2026-08-25"
    clips = root / "deletion_clips"
    d.mkdir(parents=True); clips.mkdir(parents=True)
    rows = []
    for i in range(6):
        clip = f"win_SECRETCITY_meet{i}_999_{i}.mp3"
        (clips / clip).write_bytes(b"ID3" + bytes(400))
        rows.append({"city": "SECRETCITY", "meeting": f"meet{i}", "win": f"w{i}",
                     "cat": "SECRETCAT", "rule": "SECRETRULE",
                     "kind": "λέξεις" if i % 2 else "επανάληψη",
                     "lost": ["ναι"] if i < 2 else ["λοιπον"],
                     "oracle": ["ναι"] if i % 2 == 0 else [],
                     "L": "" if i == 3 else f"αριστερα{i}", "R": f"δεξια{i}",
                     "spans": [], "pos": 0.1 * i, "clip": clip, "clip_start": 1.0})
    rows.append({"city": "SECRETCITY", "meeting": "m9", "win": "w9", "cat": "C",
                 "rule": "R", "kind": "λέξεις", "lost": ["x"], "oracle": [],
                 "L": "a", "R": "b", "spans": [], "pos": 0.5, "clip": None})
    (d / "listen_sample.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (d / "listen_sample_small.json").write_text(
        json.dumps(rows[:2], ensure_ascii=False), encoding="utf-8")
    return d, clips


def start(root, extra=None):
    env = dict(os.environ)
    env.update({"SC": str(root), "PORT": str(PORT), "HOST": "127.0.0.1",
                "PYTHONUNBUFFERED": "1"})
    env.update(extra or {})
    p = subprocess.Popen([sys.executable, str(HERE / "deletion_audit_server.py")],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    for _ in range(80):
        if p.poll() is not None:
            return p, p.stdout.read()
        try:
            urllib.request.urlopen(BASE + "/", timeout=1).read()
            return p, None
        except Exception:
            time.sleep(0.1)
    p.kill()
    return p, "did not come up"


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def token_of(page):
    i = page.index('const TOKEN = "') + len('const TOKEN = "')
    return page[i:page.index('"', i)]


def cids_of(page):
    i = page.index("const DECK = ")
    j = page.index("\n", i)
    return [o["cid"] for o in json.loads(page[i + len("const DECK = "):j].rstrip(";"))]


def main():
    root = Path(tempfile.mkdtemp(prefix="audit-selftest-"))
    try:
        d, clips = build(root)
        vfile = d / "deletion_audit_verdicts.json"

        print("\n1. server starts, page renders")
        p, err = start(root)
        check("server up", err is None, str(err))
        if err is not None:
            return
        code, body, hdr = req("/")
        page = body.decode()
        check("GET / is 200 html", code == 200 and "text/html" in hdr.get("Content-Type", ""))
        check("no-store on the page", hdr.get("Cache-Control") == "no-store")

        print("\n2. BLINDNESS -- the page must not carry the answer")
        for leak in ("SECRETCAT", "SECRETRULE", "SECRETCITY", "oracle",
                     "win_SECRETCITY", "meet0", '"kind"'):
            check(f"page omits {leak!r}", leak not in page)
        check("page carries the deleted words", "λοιπον" in page)
        check("page carries the context", "αριστερα1" in page)

        cids = cids_of(page)
        tok = token_of(page)
        check("deck has 6 clips (the clip-less row is dropped)", len(cids) == 6, str(len(cids)))

        print("\n3. clips")
        code, body, hdr = req(f"/clip/{cids[0]}.mp3")
        check("clip 200 + audio/mpeg", code == 200 and hdr.get("Content-Type") == "audio/mpeg")
        check("clip bytes served", body.startswith(b"ID3"))
        check("unknown cid 404", req("/clip/" + "0" * 16 + ".mp3")[0] == 404)
        check("traversal shape 404", req("/clip/../deletion_audit.html")[0] == 404)

        print("\n4. /save validation")
        check("no token -> 403", req("/save", {"cid": cids[0], "a": 1})[0] == 403)
        check("bad token -> 403", req("/save", {"token": "x", "cid": cids[0], "a": 1})[0] == 403)
        check("bad cid -> 400", req("/save", {"token": tok, "cid": "z" * 16, "a": 1})[0] == 400)
        check("a=0 -> 400", req("/save", {"token": tok, "cid": cids[0], "a": 0})[0] == 400)
        check("a=5 -> 400", req("/save", {"token": tok, "cid": cids[0], "a": 5})[0] == 400)
        check("a=true -> 400", req("/save", {"token": tok, "cid": cids[0], "a": True})[0] == 400)
        check("empty patch -> 400", req("/save", {"token": tok, "cid": cids[0]})[0] == 400)
        check("huge note -> 400",
              req("/save", {"token": tok, "cid": cids[0], "note": "x" * 5000})[0] in (400, 413))
        check("non-json ctype -> 415",
              req("/save", {"token": tok, "cid": cids[0], "a": 1}, ctype="text/plain")[0] == 415)
        check("bad Host -> 403", req("/", headers={"Host": "evil.example"})[0] == 403)
        check("bad Origin -> 403",
              req("/save", {"token": tok, "cid": cids[0], "a": 1},
                  headers={"Origin": "http://evil.example"})[0] == 403)
        check("nothing written by rejected saves", not vfile.exists())

        print("\n5. partial updates: note and answer are independent")
        check("note alone -> 200",
              req("/save", {"token": tok, "cid": cids[1], "note": "ακούγεται καθαρά"})[0] == 200)
        code, body, _ = req("/save", {"token": tok, "cid": cids[1], "a": 2})
        check("answer after note -> 200", code == 200)
        rec = json.loads(vfile.read_text(encoding="utf-8"))["verdicts"]
        k1 = [c for c in rec if rec[c].get("a") == 2][0]
        check("note survived the answer", rec[k1].get("note") == "ακούγεται καθαρά")
        check("answer survived the note", rec[k1].get("a") == 2)
        check("answered count reported", json.loads(body)["answered"] == 1)
        req("/save", {"token": tok, "cid": cids[3], "a": 1})
        req("/save", {"token": tok, "cid": cids[4], "a": 4})
        stop(p)

        print("\n6. RESUME -- verdicts survive a restart")
        saved = json.loads(vfile.read_text(encoding="utf-8"))
        check("file records the seed", saved.get("seed") == 20260909)
        check("file records the order", len(saved.get("order", [])) == 6)
        check("file records a fingerprint", bool(saved.get("fingerprint")))
        p, err = start(root)
        check("restart ok", err is None, str(err))
        page2 = req("/")[1].decode()
        i = page2.index("const ST = ")
        st = json.loads(page2[i + len("const ST = "):page2.index("\n", i)].rstrip(";"))
        check("3 answers reloaded", sum(1 for e in st.values() if "a" in e) == 3, str(st))
        check("comment reloaded", any(e.get("note") == "ακούγεται καθαρά" for e in st.values()))
        check("reloaded state still hides oracle", "oracle" not in page2)
        stop(p)

        print("\n7. a changed sample must not silently merge into old answers")
        rows = json.loads((d / "listen_sample.json").read_text(encoding="utf-8"))
        rows[0]["L"] = "εντελως αλλο συμφραζομενο"
        (d / "listen_sample.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        p2, err2 = start(root)
        check("refuses to start on a fingerprint mismatch",
              p2.poll() is not None and "fingerprint" in (err2 or ""), str(err2)[:120])
        stop(p2)
        rows[0]["L"] = "αριστερα0"
        (d / "listen_sample.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

        print("\n8. host guard")
        p3, err3 = start(root, {"HOST": "0.0.0.0"})
        check("refuses to bind 0.0.0.0", p3.poll() is not None and "άρνηση" in (err3 or ""))
        stop(p3)
        p4, err4 = start(root, {"HOST": "192.168.1.50"})
        check("refuses a LAN address", p4.poll() is not None and "άρνηση" in (err4 or ""))
        stop(p4)

        print("\n9. report joins back to the oracle and separates the subgroups")
        env = dict(os.environ, SC=str(root))
        r = subprocess.run([sys.executable, str(HERE / "deletion_audit_report.py")],
                           env=env, capture_output=True, text=True)
        check("report runs", r.returncode == 0, r.stderr[-400:])
        o = r.stdout
        check("report warns about stratification", "NOT the" in o and "base" in o)
        check("report shows both subgroups", "A. the 36-clip pilot" in o and
              "contains ναι or δεν" in o)
        rj = subprocess.run([sys.executable, str(HERE / "deletion_audit_report.py"), "--json"],
                            env=env, capture_output=True, text=True)
        check("report --json parses", rj.returncode == 0 and json.loads(rj.stdout)["all"]["n"] == 6,
              rj.stderr[-300:])
        j = json.loads(rj.stdout)["all"]
        # answers given: one 2, one 1, one 4 -> 2 determinate, 1 harmful, 3 unanswered
        check("determinate = 2", j["determinate"] == 2, str(j))
        check("harmful = 1", j["harmful"] == 1, str(j))
        check("unjudgeable = 1", j["unjudgeable"] == 1, str(j))
        check("unanswered = 3", j["unanswered"] == 3, str(j))
        check("concordance counted", j["agree"] + j["disagree"] == 2)

        print("\n10. one writer per verdict file")
        p5, err5 = start(root)
        check("first server up", err5 is None, str(err5))
        env2 = dict(os.environ, SC=str(root), PORT=str(PORT + 1), HOST="127.0.0.1")
        r2 = subprocess.run([sys.executable, str(HERE / "deletion_audit_server.py")],
                            env=env2, capture_output=True, text=True, timeout=30)
        check("second server on the same file refuses",
              r2.returncode != 0 and "κρατά ήδη" in (r2.stdout + r2.stderr),
              (r2.stdout + r2.stderr)[-160:])
        stop(p5)

        print("\n11. verdicts may not land inside a git repo")
        (root / ".git").mkdir()
        p6, err6 = start(root)
        check("refuses a verdict path inside a repo",
              p6.poll() is not None and "git repo" in (err6 or ""), str(err6)[:160])
        stop(p6)
        (root / ".git").rmdir()

        print("\n12. the report refuses what it cannot trust")
        good = json.loads(vfile.read_text(encoding="utf-8"))
        env = dict(os.environ, SC=str(root))

        bad = json.loads(json.dumps(good))
        bad["verdicts"][bad["order"][0]] = {"a": 9}
        vfile.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        r = subprocess.run([sys.executable, str(HERE / "deletion_audit_report.py")],
                           env=env, capture_output=True, text=True)
        check("refuses an out-of-range answer",
              r.returncode != 0 and "not 1-4" in (r.stdout + r.stderr), r.stdout[-160:])

        bad = json.loads(json.dumps(good))
        bad["order"] = bad["order"] + [bad["order"][0]]
        vfile.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        r = subprocess.run([sys.executable, str(HERE / "deletion_audit_report.py")],
                           env=env, capture_output=True, text=True)
        check("refuses a duplicated order",
              r.returncode != 0 and "duplicates" in (r.stdout + r.stderr), r.stdout[-160:])

        vfile.write_text(json.dumps(good, ensure_ascii=False), encoding="utf-8")
        rows = json.loads((d / "listen_sample.json").read_text(encoding="utf-8"))
        rows[0]["lost"] = ["εντελως", "αλλες", "λεξεις"]
        (d / "listen_sample.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        r = subprocess.run([sys.executable, str(HERE / "deletion_audit_report.py")],
                           env=env, capture_output=True, text=True)
        check("refuses a fingerprint mismatch",
              r.returncode != 0 and "different question" in (r.stdout + r.stderr),
              r.stdout[-160:])
        rows[0]["lost"] = ["ναι"]
        (d / "listen_sample.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        r = subprocess.run([sys.executable, str(HERE / "deletion_audit_report.py")],
                           env=env, capture_output=True, text=True)
        check("accepts it again once restored", r.returncode == 0, r.stdout[-160:])

        print("\n13. the page's javascript parses")
        p7, err7 = start(root)
        if err7 is None:
            page = req("/")[1].decode()
            js = page[page.index("<script>") + 8:page.rindex("</script>")]
            jsf = root / "page.js"
            jsf.write_text(js, encoding="utf-8")
            n = shutil.which("node")
            if n:
                r = subprocess.run([n, "--check", str(jsf)], capture_output=True, text=True)
                check("node --check passes", r.returncode == 0, r.stderr[-300:])
                r = subprocess.run([n, str(HERE / "deletion_audit_jstest.js")],
                                   capture_output=True, text=True)
                check("page state machine holds (save/advance/comment races)",
                      r.returncode == 0, r.stdout[-500:])
            else:
                print("  skip  node not installed")
        stop(p7)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
