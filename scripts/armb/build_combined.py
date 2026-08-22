"""Build the combined contiguous+spliced pack manifest, or refuse and say why.

Every failure mode we have actually hit gets an explicit check here: a truncated
rsync, an audio file that exists but is a 44-byte header, a manifest key the trainer
reads only in a log line, a sample rate that silently differs, and a duration that
disagrees with what the manifest claims. A combined arm that trains on 4,000 packs
instead of 6,983 would look fine in the logs and be a different experiment.
"""
import json, sys
from pathlib import Path
import soundfile as sf

CONT = Path("/workspace/oc/bundle/packs/manifest.jsonl")
SPL  = Path("/workspace/oc/spliced/manifest.jsonl")
OUT  = Path("/workspace/oc/combined")
REQUIRED = {"audio", "dur_sec", "n_utterances", "text_p", "text_pn"}
SR = 16000
EXPECT_SPLICED = 4508
EXPECT_CONT = 2476

def load(p):
    if not p.is_file():
        sys.exit(f"FATAL: missing manifest {p}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def check(rows, root, tag, expect):
    if len(rows) != expect:
        sys.exit(f"FATAL: {tag} has {len(rows)} rows, expected {expect} — transfer incomplete")
    out, checked = [], 0
    for i, r in enumerate(rows):
        absent = REQUIRED - set(r)
        if absent:
            sys.exit(f"FATAL: {tag} row {i} missing {sorted(absent)}")
        a = Path(r["audio"])
        if not a.is_absolute():
            a = root / a
        if not a.is_file():
            sys.exit(f"FATAL: {tag} row {i} audio missing: {a}")
        size = a.stat().st_size
        if size <= 44:
            sys.exit(f"FATAL: {tag} row {i} audio is an empty wav header: {a}")
        # decode-check a 5% sample: full decode of 7k files costs minutes we do not have,
        # but a header-only check would pass a truncated file.
        if i % 20 == 0:
            info = sf.info(str(a))
            if info.samplerate != SR:
                sys.exit(f"FATAL: {tag} row {i} sample rate {info.samplerate} != {SR}: {a}")
            claimed = float(r["dur_sec"])
            if abs(info.duration - claimed) > 0.25:
                sys.exit(f"FATAL: {tag} row {i} duration {info.duration:.2f}s vs manifest {claimed:.2f}s: {a}")
            checked += 1
        if not (r["text_pn"] or "").strip():
            sys.exit(f"FATAL: {tag} row {i} has empty text_pn")
        r = dict(r); r["audio"] = str(a); r["arm_source"] = tag
        out.append(r)
    print(f"  {tag}: {len(out)} rows ok, {checked} decoded, "
          f"{sum(x['dur_sec'] for x in out)/3600:.2f} h")
    return out

print("verifying contiguous...")
cont = check(load(CONT), CONT.parent, "contiguous", EXPECT_CONT)
print("verifying spliced...")
spl = check(load(SPL), SPL.parent, "spliced", EXPECT_SPLICED)

ids = [r.get("pack_id") for r in cont + spl]
if len(set(ids)) != len(ids):
    sys.exit(f"FATAL: duplicate pack_id across arms ({len(ids)-len(set(ids))} collisions)")

OUT.mkdir(parents=True, exist_ok=True)
rows = cont + spl
with open(OUT / "manifest.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
hours = sum(r["dur_sec"] for r in rows) / 3600
print(f"OK: combined manifest {len(rows)} packs, {hours:.2f} h -> {OUT/'manifest.jsonl'}")
print(f"    2 epochs at effective batch 8 = {round(len(rows)*2/8)} steps (we run 619, matched to contiguous)")
