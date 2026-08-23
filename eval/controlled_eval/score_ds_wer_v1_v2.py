"""DS-WER for adapter v1 and v2 on one decoding stack, with no GPU.

The published DS-WER table (2026-08-12) took its whisper-family hypotheses from a stack
that no longer exists. The 2026-08-22 epoch-matched run decoded all six clean-pack
adapters and the incumbent on ONE A4000 with the frozen CONTROL config and the per-window
seed, and those hypotheses are on disk. Scoring them here puts v1 and v2 on the same
stack, which is the only way this comparison is allowed to be made.

References, term lists and scorer are the frozen ones the published table used, so the
commercial rows are re-scored here rather than copied forward.
"""
import json, os, sys, random
sys.path.insert(0, '/home/harold/opencouncil-fine-tuning')
from scripts.ds_wer import TermList, aggregate, ds_wer
from eval.controlled_eval import bench_data as B

SC = '/tmp/claude-1000/-home-harold-opencouncil-fine-tuning/2e880ed1-a91a-4395-b4e0-697593a6bf22/scratchpad/sc'
ROOT = '/home/harold/opencouncil-fine-tuning'
RUN = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"
COMMERCIAL = {"Gladia": "gladia-prod", "Scribe v2": "scribe-v2-clean",
              "Soniox": "soniox", "base whisper-large-v3": "hf-openai-whisper-large-v3",
              "v1 as decoded 2026-08-12": "oc-runpod-fixed-2026-08-10"}
LOCAL = {"v1 (incumbent)": "incumbent", "v2 seed 47": "cont_s47",
         "v2 seed 29": "cont_s29", "v2 seed 13": "cont_s13"}

man = json.load(open(f'{ROOT}/research/eval-freeze-2026-08/manifest.json'))
wins = man['eval_windows']
terms = {os.path.splitext(p)[0]: TermList.load(f'{ROOT}/research/ds_wer/terms/{p}')
         for p in os.listdir(f'{ROOT}/research/ds_wer/terms')}

report = B.load_report(RUN)
items = {it['itemId']: it for it in report['items']}
missing = [w['window_id'] for w in wins if w['window_id'] not in items]
assert not missing, f"windows absent from the benchmark run: {missing}"

# The reference must be the one the published table used, or the comparison is not
# against the same target.
refs = {w['window_id']: items[w['window_id']]['referenceText'] for w in wins}

hyps = {}
for label, tag in LOCAL.items():
    d = json.load(open(f'{SC}/{tag}.json'))
    assert set(d['hyp']) == {w['window_id'] for w in wins}, f"{tag}: window set differs"
    hyps[label] = {k: ' '.join(v['segments']) for k, v in d['hyp'].items()}
for label, pid in COMMERCIAL.items():
    hyps[label] = {}
    for w in wins:
        pp = items[w['window_id']]['perProvider'].get(pid) or {}
        t = pp.get('hypothesisText') or ''
        assert t.strip(), f"{label}: no hypothesis for {w['window_id']}"
        hyps[label][w['window_id']] = t

city = {w['window_id']: w['city'] for w in wins}
meeting = {w['window_id']: (w['city'], w['meeting_id']) for w in wins}
per = {lab: {wid: ds_wer(refs[wid], hyps[lab][wid], terms[city[wid]]) for wid in refs}
       for lab in hyps}

# N depends only on reference and term list, so every system must agree on it.
for wid in refs:
    ns = {per[lab][wid]['N'] for lab in per}
    assert len(ns) == 1, f"{wid}: systems disagree on denominator {ns}"
nby = {wid: per[list(per)[0]][wid]['N'] for wid in refs}
rollcall = [w for w, _ in sorted(nby.items(), key=lambda kv: (-kv[1], kv[0]))][:2]

def score(lab, keep):
    return aggregate([per[lab][w] for w in keep])

allw = list(refs); noroll = [w for w in allw if w not in rollcall]
print(f"{sum(nby.values())} term occurrences over {len(allw)} windows; "
      f"roll-call windows excluded in the sensitivity analysis: {rollcall}\n")
print(f"{'system':28s} {'DS-WER':>8s} {'no roll call':>13s}")
order = ["Soniox", "Scribe v2", "v1 (incumbent)", "v2 seed 13", "v2 seed 29",
         "v2 seed 47", "v1 as decoded 2026-08-12", "base whisper-large-v3", "Gladia"]
out = {}
for lab in order:
    a, b = score(lab, allw), score(lab, noroll)
    out[lab] = {"primary": a['ds_wer'], "no_rollcall": b['ds_wer']}
    print(f"{lab:28s} {a['ds_wer']:8.4f} {b['ds_wer']:13.4f}")

# paired meeting-clustered bootstrap, v2 seed 47 minus v1, on one stack
Ms = sorted(set(meeting.values()))
byM = {m: [w for w in allw if meeting[w] == m] for m in Ms}
def delta(sel, A="v2 seed 47", Bl="v1 (incumbent)"):
    ea = sum(per[A][w]['errors'] for m in sel for w in byM[m])
    eb = sum(per[Bl][w]['errors'] for m in sel for w in byM[m])
    n = sum(per[A][w]['N'] for m in sel for w in byM[m])
    return (ea - eb) / n if n else 0.0
rng = random.Random(20260823)
pt = delta(Ms)
boots = sorted(delta([rng.choice(Ms) for _ in Ms]) for _ in range(10000))
print(f"\nv2 seed47 minus v1, paired meeting-clustered bootstrap over {len(Ms)} meetings:")
print(f"  {pt:+.4f}   95% CI [{boots[250]:+.4f}, {boots[9750]:+.4f}]")
json.dump({"primary": out, "delta_v2s47_minus_v1": {"point": pt,
           "ci95": [boots[250], boots[9750]]}, "rollcall_excluded": rollcall,
           "term_occurrences": sum(nby.values())},
          open(f'{SC}/dswer_v1v2.json', 'w'), ensure_ascii=False, indent=1)
print(f"\nwrote {SC}/dswer_v1v2.json")
