# -*- coding: utf-8 -*-
"""Generate the four conformance fixtures of spec 2026-08-26-fusion-integration §4.4.

Outputs (to $SC/chooser-2026-08-25/, NEVER git — they hold transcript text):
  fixture_inputs_391.json        per window: hyps (3 norm-token lists), pivot, ref
  fixture_W_391.json             per window: fused tokens + [S,D,I,N] for the vote
  fixture_rules_off_391.json     R1+R2, guard off  (must equal frozen 0.11154)
  fixture_rules_on_391.json      R1+R2, guard on   (NEW number, measured here)
  fixture_policy_islands_391.json per island: category, route, rule, candidates,
                                 render-row sha (LLM islands carry route="llm";
                                 their frozen decisions were never stored, so the
                                 conformance test mocks the LLM and checks routing)
  MANIFEST.json                  sha256 of the bench report + every fixture — the
                                 ONLY file that goes into the tasks repo.

Guard definition (spec §4.1, operationalised 2026-09-02 — the pricing script was
never rescued, so this is the frozen definition from here on):
  An island is guarded iff (a) after removing every token in {ναι, δεν} the three
  candidate spans are identical, (b) at least one span contains such a token, and
  (c) the arm's island output contains none. Then the output is replaced by the
  span with the most critical tokens (ties: the longest, then scribe>soniox>ours).

Env: REPORT (bench report json), FREEZE (freeze.json), SC (cache root).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/harold/opencouncil-fine-tuning")
from eval.controlled_eval.atlas_substrate import load, TRIO, sc          # noqa
from eval.controlled_eval.exp_fusion_deletions import sdi                # noqa
from reconstruct import window_parts, RULES                              # noqa

CRIT = ("ναι", "δεν")
SCHEMA = "oc-fusion-fixture/1"


def apply_guard(spans, out):
    core = [[t for t in sp if t not in CRIT] for sp in spans]
    if not (core[0] == core[1] == core[2]):
        return out, False
    n_crit = [sum(t in CRIT for t in sp) for sp in spans]
    if not any(n_crit):
        return out, False
    if any(t in CRIT for t in out):
        return out, False
    k = max(range(3), key=lambda i: (n_crit[i], len(spans[i]), -i))
    return list(spans[k]), True


def build(parsed_w, arm, pol=None):
    isl, tail, _, _ = parsed_w
    out = []
    n_guard = 0
    for i in isl:
        out += i["before"]
        if arm == "W":
            out += i["vote"]
        elif arm in ("rules_off", "rules_on"):
            piece = list(i["r12"])
            if arm == "rules_on":
                piece, fired = apply_guard([list(x) for x in i["spans"]], piece)
                n_guard += fired
            out += piece
        else:
            raise ValueError(arm)
    return out + tail, n_guard


def row_sha(i):
    row = {"L": i["L"], "R": i["R"], "spans": i["spans"], "r12": i["r12"]}
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def main():
    report = Path(os.environ["REPORT"])
    fr = json.loads(Path(os.environ["FREEZE"]).read_text())
    pol = fr["policy"]
    sub = load(report)
    wins = list(sub["windows"])
    print(f"{len(wins)} windows", flush=True)
    parsed = {w["id"]: window_parts(w) for w in wins}

    outdir = sc() / "chooser-2026-08-25"
    outdir.mkdir(parents=True, exist_ok=True)

    fx_inputs = {"schema": SCHEMA, "kind": "inputs", "trio": TRIO, "windows": []}
    arms = {"W": {"schema": SCHEMA, "kind": "W", "windows": []},
            "rules_off": {"schema": SCHEMA, "kind": "rules_off", "windows": []},
            "rules_on": {"schema": SCHEMA, "kind": "rules_on", "windows": []}}
    fx_pol = {"schema": SCHEMA, "kind": "policy_islands",
              "policy_sha16": hashlib.sha256(json.dumps(
                  pol, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16],
              "windows": []}

    tot = {a: [0, 0, 0, 0] for a in arms}
    n_guard_total = 0
    n_islands = 0
    for w in wins:
        ref = " ".join(w["ref"])
        fx_inputs["windows"].append({
            "id": w["id"], "city": w["city"], "meeting": w["meeting"],
            "pivot": w["pivot"], "ref": w["ref"], "hyps": w["hyps"]})
        for arm in arms:
            toks, ng = build(parsed[w["id"]], arm)
            if arm == "rules_on":
                n_guard_total += ng
            s, d, i_, n = sdi(ref, " ".join(toks))
            arms[arm]["windows"].append(
                {"id": w["id"], "tokens": toks, "sidn": [s, d, i_, n]})
            for j, v in enumerate((s, d, i_, n)):
                tot[arm][j] += v
        isl_rows = []
        for k, i in enumerate(parsed[w["id"]][0]):
            p = pol.get(i["cat"])
            route = (p["mode"] if p else "vote")
            isl_rows.append({
                "k": k, "s": i["s"], "e": i["e"], "cat": i["cat"],
                "route": route,
                "rule": (p.get("rule") if p and p["mode"] == "rule" else None),
                "candidates": {"scribe": i["spans"][0], "soniox": i["spans"][1],
                               "ours": i["spans"][2], "vote": i["vote"],
                               "r12": i["r12"]},
                "row_sha16": row_sha(i)})
            n_islands += 1
        fx_pol["windows"].append({"id": w["id"], "islands": isl_rows})

    def wer(t):
        return (t[0] + t[1] + t[2]) / t[3]

    frozen = json.loads((HERE / "wer_result_ALL.json").read_text())["arms"]
    assert tot["W"] == frozen["W"], (tot["W"], frozen["W"])
    assert tot["rules_off"] == frozen["R1+R2"], (tot["rules_off"], frozen["R1+R2"])
    print(f"W         {wer(tot['W']):.5f}  (frozen ✓)")
    print(f"rules_off {wer(tot['rules_off']):.5f}  (frozen ✓)")
    print(f"rules_on  {wer(tot['rules_on']):.5f}  sidn {tot['rules_on']}  "
          f"guard fired on {n_guard_total} islands")
    print(f"policy islands: {n_islands}")

    files = {"fixture_inputs_391.json": fx_inputs,
             "fixture_W_391.json": arms["W"],
             "fixture_rules_off_391.json": arms["rules_off"],
             "fixture_rules_on_391.json": arms["rules_on"],
             "fixture_policy_islands_391.json": fx_pol}
    manifest = {"schema": "oc-fusion-fixture-manifest/1",
                "report": {"name": report.name,
                           "sha256": hashlib.sha256(report.read_bytes()).hexdigest()},
                "guard": {"critical_tokens": list(CRIT),
                          "islands_fired": n_guard_total,
                          "rules_on_sidn": tot["rules_on"]},
                "totals": {a: tot[a] for a in tot},
                "n_windows": len(wins), "n_islands": n_islands,
                "fixtures": {}}
    for name, obj in files.items():
        p = outdir / name
        blob = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
        p.write_bytes(blob)
        manifest["fixtures"][name] = {
            "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        print(f"wrote {p} ({len(blob)/1e6:.1f} MB)")
    (outdir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"wrote {outdir/'MANIFEST.json'}")


if __name__ == "__main__":
    main()
