# -*- coding: utf-8 -*-
"""Cut one short clip per sampled deletion, so a human can hear what was dropped.

POSITION IS APPROXIMATE. The island's place inside the window is known in COLUMNS,
not seconds, so the clip is centred on (column position / total columns) x window
duration. That is a crude proxy: speakers do not talk at a constant rate. The clip
is 18 s wide to absorb the error, and the page says so. Anyone treating the clip
boundary as a timestamp is misreading it.

Audio stays under $SC. Never in git, never uploaded.
"""
import json, os, random, subprocess, sys
from pathlib import Path
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
OUT = SC / "deletion_clips"
REPORT = Path(os.environ["REPORT"])
WIDTH = 18.0

d = json.loads(REPORT.read_text())
meta = {it["itemId"]: it for it in d["items"]}
rows = json.loads(Path("deletions.json").read_text())

# stratified sample: every wrong deletion, plus correct ones as blind controls
wrong = [o for o in rows if o["oracle"] != []]
right = [o for o in rows if o["oracle"] == []]
rng = random.Random(20260826)
rng.shuffle(right)
sample = wrong + right[:len(wrong)]        # 50/50, so the judge cannot infer from base rate
rng.shuffle(sample)
print(f"δείγμα {len(sample)} = {len(wrong)} λάθος + {len(wrong)} σωστές (ελεγκτικές)")

def url(city, meeting):
    p = SC / "meetings" / f"{city}__{meeting}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["meeting"].get("audioUrl")

OUT.mkdir(parents=True, exist_ok=True)
ok = miss = fail = 0
for n, o in enumerate(sample):
    it = meta.get(o["win"])
    u = url(o["city"], o["meeting"])
    if not it or not u:
        o["clip"] = None; miss += 1; continue
    start = it["startSec"] + o["pos"] * it["durationSec"] - WIDTH / 2
    start = max(0.0, start)
    dst = OUT / f"{o['win']}_{int(o['pos']*1000)}.mp3"
    o["clip"] = dst.name
    o["clip_start"] = round(start, 2)
    if dst.exists() and dst.stat().st_size > 8000:
        ok += 1; continue
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}",
                        "-t", f"{WIDTH:.1f}", "-i", u, "-ac", "1", "-ar", "16000",
                        "-b:a", "32k", str(dst)], capture_output=True, text=True)
    good = p.returncode == 0 and dst.exists() and dst.stat().st_size > 8000
    if good: ok += 1
    else:
        fail += 1; o["clip"] = None
        if fail <= 3: print(f"  FAIL {o['win']}: {p.stderr.strip()[:160]}")
    if (n + 1) % 25 == 0:
        print(f"  {n+1}/{len(sample)}  ok={ok} fail={fail} miss={miss}", flush=True)

Path("listen_sample.json").write_text(json.dumps(sample, ensure_ascii=False))
tot = sum(f.stat().st_size for f in OUT.glob("*.mp3"))
print(f"\nκλιπ: {ok} · χωρίς ήχο: {miss} · απέτυχαν: {fail} · {tot/1e6:.1f} MB σε {OUT}")
