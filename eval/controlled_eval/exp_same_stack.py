"""Is the fine-tune's deletion gap real, or an artifact of two different decoders?

The benchmark says the fine-tune deletes 1.5 more words per hundred than base whisper,
and the mentor's diagnosis is built on that number. But the two rows were never produced
by the same machinery: `hf-openai-whisper-large-v3` came from the HuggingFace inference
API, and `oc-minipc-finetune` from our own faster-whisper server. Long-form decoding is
exactly where those two stacks differ — one slides a fixed 30 s window with overlap, the
other advances on the model's own predicted timestamps and can skip audio when a
timestamp comes back wrong. A deletion gap is the signature of both a model that omits
speech and a stack that loses a chunk.

So this re-decodes the same windows with both models through **one** stack, changing only
the weights. If the gap survives, the mechanism stands and it is worth building the
30-second-window arm. If it collapses, the diagnosis was measuring our server.

The other reason to run it first: it costs one GPU-hour and no training.

Three phases, run separately.

  cut     local, CPU. Slices each benchmark window out of the cached meeting mp3.
  decode  on a GPU. Base and fine-tune, identical settings, greedy, Greek.
  score   local, CPU. Substitution/deletion/insertion split plus a paired,
          meeting-clustered interval on the deletion-rate difference.

  SC=~/.cache/oc-public .venv-eval/bin/python -m eval.controlled_eval.exp_same_stack cut
  python -m eval.controlled_eval.exp_same_stack decode --clips ... --out ... \
         --adapter /workspace/prod_adapter
  SC=~/.cache/oc-public .venv-eval/bin/python -m eval.controlled_eval.exp_same_stack \
         score --hyps <dir>

Window audio and hypothesis text stay under $SC. Neither goes in git.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
AUDIO = ROOT / "data/asr/audio"
MODEL_ID = "openai/whisper-large-v3"


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------- phase: cut
def phase_cut(args) -> None:
    sys.path.insert(0, str(ROOT))
    from eval.controlled_eval import bench_data as B

    out = Path(args.out or (sc() / "bench_windows"))
    out.mkdir(parents=True, exist_ok=True)
    report = B.load_report()

    plan, missing = [], 0
    for it in report["items"]:
        src = AUDIO / f"{it['cityId']}__{it['meetingId']}.mp3"
        if not src.exists():
            missing += 1
            continue
        plan.append({"item_id": it["itemId"], "city_id": it["cityId"],
                     "meeting_id": it["meetingId"], "start": it["startSec"],
                     "dur": it["durationSec"], "src": str(src)})
    log(f"{len(plan)} windows to cut, {missing} skipped for missing local audio")

    for i, w in enumerate(plan, 1):
        dst = out / f"{w['item_id']}.wav"
        if dst.exists():
            continue
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{w['start']:.3f}",
                        "-t", f"{w['dur']:.3f}", "-i", w["src"], "-ac", "1",
                        "-ar", "16000", str(dst)], check=True)
        if i % 25 == 0:
            log(f"  {i}/{len(plan)}")
    (out / "windows.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    log(f"-> {out} ({len(list(out.glob('*.wav')))} wav)")


# ------------------------------------------------------------------- phase: decode
def phase_decode(args) -> None:
    """Both systems through the HuggingFace chunked long-form pipeline.

    Chunked rather than sequential on purpose: it advances on a fixed clock, so a bad
    predicted timestamp cannot make it skip audio. That removes the very mechanism
    suspected of manufacturing the deletions, which is what makes the comparison a test
    of the weights instead of a test of the server.
    """
    import torch
    from transformers import (AutomaticSpeechRecognitionPipeline,
                              WhisperForConditionalGeneration, WhisperProcessor)

    clips = sorted(Path(args.clips).glob("*.wav"))
    if not clips:
        raise SystemExit(f"no clips in {args.clips}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"{len(clips)} windows")

    todo = [("base", None)]
    if args.adapter:
        todo.append((args.tag, args.adapter))

    proc = WhisperProcessor.from_pretrained(MODEL_ID, language="greek", task="transcribe")
    for tag, adapter in todo:
        dest = out_dir / f"{tag}.json"
        if dest.exists():
            log(f"skip {tag}: already decoded")
            continue
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16).to("cuda")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.float16)
            model = model.merge_and_unload()
        model.eval()
        model.generation_config.forced_decoder_ids = None

        pipe = AutomaticSpeechRecognitionPipeline(
            model=model, tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            chunk_length_s=30, stride_length_s=5, device=0, torch_dtype=torch.float16)

        hyps = {}
        for i, p in enumerate(clips, 1):
            r = pipe(str(p), batch_size=8, return_timestamps=False,
                     generate_kwargs={"language": "greek", "task": "transcribe",
                                      "num_beams": 1, "do_sample": False})
            hyps[p.stem] = (r["text"] or "").strip()
            if i % 20 == 0:
                log(f"  {tag}: {i}/{len(clips)}")
        dest.write_text(json.dumps(hyps, ensure_ascii=False, indent=1))
        log(f"{tag}: {len(hyps)} windows -> {dest}")

        del pipe, model
        torch.cuda.empty_cache()


# -------------------------------------------------------------------- phase: score
def sdi(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Global alignment, split into substitutions, deletions, insertions.

    Ties are broken toward substitution. Where a reference word sits opposite some other
    word, that is a wrong word, not a missing one; the earliest-cheapest backtrace calls
    it a deletion and inflates exactly the number this experiment turns on.
    """
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def phase_score(args) -> None:
    sys.path.insert(0, str(ROOT))
    from eval.controlled_eval import bench_data as B
    from eval.controlled_eval.scoring import wtoks

    hyp_dir = Path(args.hyps)
    systems = {p.stem: json.loads(p.read_text()) for p in sorted(hyp_dir.glob("*.json"))}
    if len(systems) < 2:
        raise SystemExit(f"need at least two hypothesis files in {hyp_dir}")
    log(f"systems: {list(systems)}")

    report = B.load_report()
    by_id = {it["itemId"]: it for it in report["items"]}
    ids = sorted(set.intersection(*[set(h) for h in systems.values()]) & set(by_id))
    log(f"{len(ids)} windows scored by every system")

    per = {name: {} for name in systems}
    for iid in ids:
        ref = wtoks(by_id[iid]["referenceText"])
        for name, h in systems.items():
            per[name][iid] = (*sdi(ref, wtoks(h[iid])), len(ref))

    def totals(name):
        v = per[name].values()
        s, d, i, n = (sum(x[k] for x in v) for k in range(4))
        return {"sub": s, "del": d, "ins": i, "ref_words": n,
                "sub_rate": s / n, "del_rate": d / n, "ins_rate": i / n,
                "wer": (s + d + i) / n}

    tab = {name: totals(name) for name in systems}
    for name, t in tab.items():
        log(f"{name:>12}  WER {t['wer']:.4f}  sub {t['sub_rate']:.4f} "
            f"del {t['del_rate']:.4f} ins {t['ins_rate']:.4f}")

    result = {"n_windows": len(ids), "systems": tab, "contrasts": {}}
    others = [n for n in systems if n != args.base]
    if args.base in systems:
        meetings = {iid: (by_id[iid]["cityId"], by_id[iid]["meetingId"]) for iid in ids}
        by_meet = {}
        for iid in ids:
            by_meet.setdefault(meetings[iid], []).append(iid)
        keys = list(by_meet)
        rnd = random.Random(7)

        for other in others:
            def delta(sample_ids):
                out = {}
                for name in (args.base, other):
                    s = d = i = n = 0
                    for iid in sample_ids:
                        a, b, c, r = per[name][iid]
                        s, d, i, n = s + a, d + b, i + c, n + r
                    out[name] = (s / n, d / n, i / n, (s + d + i) / n)
                a, b = out[other], out[args.base]
                return [x - y for x, y in zip(a, b)]

            point = delta(ids)
            boots = []
            for _ in range(args.boot):
                pick = [rnd.choice(keys) for _ in keys]
                boots.append(delta([i for k in pick for i in by_meet[k]]))
            lo = [sorted(b[k] for b in boots)[int(0.05 * args.boot)] for k in range(4)]
            hi = [sorted(b[k] for b in boots)[int(0.95 * args.boot)] for k in range(4)]
            names = ["sub", "del", "ins", "wer"]
            result["contrasts"][f"{other}-{args.base}"] = {
                n: {"delta": point[k], "lo": lo[k], "hi": hi[k]}
                for k, n in enumerate(names)}
            log(f"\n{other} minus {args.base}, {len(keys)} meetings, "
                f"{args.boot} replicates:")
            for k, n in enumerate(names):
                log(f"  {n:>4}  {point[k]*100:+.2f} pts  "
                    f"[{lo[k]*100:+.2f}, {hi[k]*100:+.2f}]")

    Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    log(f"\n-> {args.json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)

    c = sub.add_parser("cut")
    c.add_argument("--out")
    c.set_defaults(fn=phase_cut)

    d = sub.add_parser("decode")
    d.add_argument("--clips", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--adapter")
    d.add_argument("--tag", default="finetune")
    d.set_defaults(fn=phase_decode)

    s = sub.add_parser("score")
    s.add_argument("--hyps", required=True)
    s.add_argument("--base", default="base")
    s.add_argument("--boot", type=int, default=10000)
    s.add_argument("--json", default="eval/controlled_eval/results_same_stack.json")
    s.set_defaults(fn=phase_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
