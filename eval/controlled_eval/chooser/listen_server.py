#!/usr/bin/env python3
"""Local listening page for the policy's deletions. Binds to 127.0.0.1 only.

WHY LOCAL. The benchmark's TEXT is published without auth, so quoting it costs
nothing. The AUDIO is a different act: re-hosting council speech under our own URL
is a new copy of the same PII category the 2026-07-21 purge removed from git. It
stays on this machine.

WHAT THE JUDGE SEES. One 18 s clip, the vote's context either side, and the words
the policy deleted. Nothing else. Whether the deletion was right is hidden until
after the judgement is recorded, and the sample is 50/50 right-and-wrong so the
base rate carries no information either.

    python3 listen_server.py     ->  http://127.0.0.1:877
"""
import json, os, re, http.server, socketserver, urllib.parse
from pathlib import Path

HERE = Path(__file__).parent
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
CLIPS = SC / "deletion_clips"
PORT = int(os.environ.get("PORT", "8770"))
# Bind to the tailnet address, never 0.0.0.0. A tailnet is device-authenticated and
# encrypted, so the laptop reaches it and nothing else does. Binding to 0.0.0.0 to
# "make it work" would put council audio on every network this machine joins.
HOST = os.environ.get("HOST", "100.123.255.50")
VERDICTS = HERE / "listen_verdicts.json"

rows = [o for o in json.loads((HERE / "listen_sample.json").read_text()) if o.get("clip")]
DATA = json.dumps([{ "i": n, "clip": o["clip"], "L": o["L"], "R": o["R"],
                     "lost": o["lost"], "kind": o["kind"], "city": o["city"],
                     "cat": o["cat"], "rule": o["rule"],
                     "wrong": int(o["oracle"] != [])}
                   for n, o in enumerate(rows)], ensure_ascii=False)

PAGE = (HERE / "listen.html").read_text().replace("__DATA__", DATA)


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                      # never log; the paths carry meeting ids

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p in ("/", "/index.html"):
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b); return
        m = re.fullmatch(r"/clip/([A-Za-z0-9_.-]+\.mp3)", p)
        if m:
            f = CLIPS / m.group(1)
            if not f.is_file() or CLIPS.resolve() not in f.resolve().parents:
                self.send_error(404); return
            b = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b); return
        self.send_error(404)

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/save":
            self.send_error(404); return
        n = int(self.headers.get("Content-Length", 0))
        VERDICTS.write_text(self.rfile.read(n).decode())
        self.send_response(204); self.end_headers()


if __name__ == "__main__":
    if HOST in ("0.0.0.0", "::"):
        raise SystemExit("άρνηση: 0.0.0.0 βγάζει τον ήχο στο LAN")
    print(f"{len(rows)} κλιπ · http://{HOST}:{PORT}", flush=True)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((HOST, PORT), H) as s:
        s.serve_forever()
