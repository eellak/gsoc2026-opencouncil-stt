"""Decode FLEURS el_gr on the same stack as the validation windows.

The point of this script is the one hole the project could not close from its own
data: every number we have is on Greek council speech, our own domain, against our
own references. FLEURS el_gr is read speech by other speakers, with references we
did not write. It is a different question - "is this a good Greek ASR model" rather
than "is this a good model for us" - and it deserves its own caveats, not a headline.

Comparability, not tuning, is what is bought here: same machine, same CT2 float16
build, same frozen CONTROL decode config, same `ftoks` normalisation and `sdi`
counter as `score_one.py`. CONTROL was frozen for ~25 s council windows, so it is
almost certainly not the best config for 10-20 s read sentences. That is deliberate.
Choosing a decode config per benchmark is how this project has produced false wins
before, and a config that flatters no arm in particular still ranks them fairly.
"""
import argparse, hashlib, json, sys
from pathlib import Path

# In-repo this file is two levels down; on the pod it is dropped next to the driver
# and the repo comes in via PYTHONPATH instead. Guard rather than assume the depth.
_p = Path(__file__).resolve().parents
if len(_p) > 2:
    sys.path.insert(0, str(_p[2]))
from eval.controlled_eval.dense_screen_eval import build_ct2, CONTROL, MODEL_ID
from eval.controlled_eval.eval_freeze import ftoks
from eval.controlled_eval.exp_same_stack import sdi

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", type=Path, default=None,
                help="LoRA adapter to merge; omit for base whisper-large-v3")
ap.add_argument("--tag", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--work", type=Path, default=Path("/workspace/oc/ct2tmp"))
ap.add_argument("--audio-dir", type=Path, required=True)
ap.add_argument("--manifest", type=Path, required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

import ctranslate2
from faster_whisper import WhisperModel

rows = [json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
if a.limit:
    rows = rows[: a.limit]

if a.adapter is None:
    # Base model: convert once, straight from the hub, no PEFT merge.
    ct2 = a.work / "base" / "ct2"
    if not (ct2 / "model.bin").is_file():
        import subprocess
        ct2.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ct2-transformers-converter", "--model", MODEL_ID,
                        "--output_dir", str(ct2), "--copy_files", "tokenizer.json",
                        "preprocessor_config.json", "--quantization", "float16"],
                       check=True)
else:
    ct2 = build_ct2(a.adapter, a.work / a.tag)

model = WhisperModel(str(ct2), device="cuda", compute_type="float16")

per, hyp = {}, {}
S = D = I = R = 0
for i, row in enumerate(rows, 1):
    key = row["id"]
    # Same seeding discipline as score_one.py: CONTROL samples above temperature 0,
    # so an unseeded decode is a different experiment run to run.
    ctranslate2.set_random_seed(
        int(hashlib.sha256(f"fleurs-el|{key}".encode()).hexdigest()[:8], 16))
    segs, _ = model.transcribe(str(a.audio_dir / row["audio"]), **CONTROL)
    text = [s.text for s in segs]
    hyp[key] = text
    ref = ftoks(row["ref"])
    got = [t for s in text for t in ftoks(s)]
    s, d, ins = sdi(ref, got)
    per[key] = {"s": s, "d": d, "i": ins, "ref": len(ref)}
    S += s; D += d; I += ins; R += len(ref)
    if i % 25 == 0:
        print(f"  {a.tag}: {i}/{len(rows)}", flush=True)

# The first run of this script keyed per_utt by the FLEURS sentence id, which is not
# unique: 650 rows collapsed to 333 records and the aggregate counters summed rows whose
# audio had been overwritten. Refuse to emit a number unless every row survived.
if len(per) != len(rows):
    raise SystemExit(f"FATAL: {len(rows)} rows in, {len(per)} results out - manifest keys "
                     f"are not unique; rebuild the manifest before trusting any number")

result = {"tag": a.tag, "benchmark": "fleurs_el_gr",
          "adapter": str(a.adapter) if a.adapter else "base:" + MODEL_ID,
          "n_utts": len(rows), "ref_tokens": R, "sub": S, "del": D, "ins": I,
          "wer": (S + D + I) / R if R else None,
          "sub_rate": S / R if R else None, "del_rate": D / R if R else None,
          "ins_rate": I / R if R else None,
          "decode_config": "CONTROL (frozen, dense_screen_eval)",
          "per_utt": per, "hyp": hyp}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, ensure_ascii=False))
print(json.dumps({k: result[k] for k in ("tag", "n_utts", "ref_tokens", "wer",
                                         "del_rate", "ins_rate", "sub_rate")}, indent=1))
