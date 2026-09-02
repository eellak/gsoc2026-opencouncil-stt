"""Score one adapter on the 39 validation windows, nothing else.

Same helpers as dense_screen_eval so the numbers land on the same scale: the same
ftoks normalisation, the same sdi counter, the same CT2 float16 build. What it drops
is the 300 training rows, which were 88% of the cost and are not what we are asking.
Runs happily alongside a training job: CT2 float16 large-v3 needs ~4 GB and the A4000
has ~11 GB free while training.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/oc/bundle/code")
from eval.controlled_eval.dense_screen_eval import build_ct2, load_rows, CONTROL
from eval.controlled_eval.eval_freeze import ftoks
from eval.controlled_eval.exp_same_stack import sdi

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", type=Path, required=True)
ap.add_argument("--validation", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ct2", type=Path, default=Path("/workspace/oc/ct2tmp"))
a = ap.parse_args()

import ctranslate2, hashlib
from faster_whisper import WhisperModel
rows = load_rows(a.validation, "eval_windows", "audio_local")
work = a.ct2 / a.tag
ct2 = build_ct2(a.adapter, work)
model = WhisperModel(str(ct2), device="cuda", compute_type="float16")

hyp, per = {}, {}
S = D = I = R = 0
for i, row in enumerate(rows, 1):
    # CONTROL is the frozen decode config of the screen, and the per-window seed is
    # part of it: sampling at temperature > 0 makes an unseeded decode a different
    # experiment. Importing both is the only way this number sits on the same scale.
    key = row["window_id"]
    ctranslate2.set_random_seed(
        int(hashlib.sha256(f"dense-screen|{key}".encode()).hexdigest()[:8], 16))
    segs, _ = model.transcribe(row["audio_local"], **CONTROL)
    text = [s.text for s in segs]
    hyp[key] = {"segments": text}
    ref = row["ref_tokens"] if isinstance(row.get("ref_tokens"), list) else ftoks(row["ref_tokens"])
    got = [t for s in text for t in ftoks(s)]
    s, d, ins = sdi(ref, got)
    per[key] = {"s": s, "d": d, "i": ins, "ref": len(ref)}
    S += s; D += d; I += ins; R += len(ref)
    if i % 10 == 0:
        print(f"  {a.tag}: {i}/{len(rows)}", flush=True)

result = {"tag": a.tag, "adapter": str(a.adapter), "n_windows": len(rows),
          "ref_tokens": R, "sub": S, "del": D, "ins": I,
          "wer": (S + D + I) / R, "sub_rate": S / R, "del_rate": D / R, "ins_rate": I / R,
          "per_window": per, "hyp": hyp}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, ensure_ascii=False))
print(json.dumps({k: result[k] for k in ("tag", "n_windows", "ref_tokens", "wer",
                                         "del_rate", "ins_rate", "sub_rate")}, indent=1))
import shutil; shutil.rmtree(work, ignore_errors=True)
