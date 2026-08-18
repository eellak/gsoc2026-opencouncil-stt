"""Where inside a decode window do errors live?

FROZEN BEFORE LOOKING AT ANY OUTPUT:
  BIN_SECONDS = 5.0        six bins across Whisper's 30 s window
  WINDOW_SECONDS = 30.0    the audio context length the encoder accepts
  Reference tokens are attributed to a time by their aligned hypothesis word;
  a deleted reference token takes the time of the next aligned hypothesis word,
  or the previous one when it is the tail.

Two profiles are computed side by side, and the contrast between them IS the point:

  PHASE   offset inside the reconstructed 30 s inference window. If Whisper's
          window boundary is what hurts, error rate rises near 30 s in EVERY window.
  ABS     offset from the start of the whole evaluation clip. If the clip's own
          edges are what hurt, error rate rises only at the first and last bins.

A U in ABS with a flat PHASE means the 30-second story is wrong.

faster-whisper with vad_filter=False seeks sequentially: it decodes [seek, seek+30],
then advances seek to the end of the last segment it completed. The reconstruction
below replays exactly that from the emitted segment boundaries.
"""
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.controlled_eval.scoring import wtoks  # frozen normalizer

BIN_SECONDS = 5.0
WINDOW_SECONDS = 30.0
N_BINS = int(WINDOW_SECONDS / BIN_SECONDS)

CACHE = Path.home() / ".cache/oc-public/conf-substrate-2026-08"


def ops(ref, hyp):
    """Same DP and same tie-breaking as exp_same_stack.sdi, but emitting the path."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        row, prev = D[i], D[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1,
                         prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append(("S" if ref[i - 1] != hyp[j - 1] else "C", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out.append(("D", i - 1, None))
            i -= 1
        else:
            out.append(("I", None, j - 1))
            j -= 1
    out.reverse()
    return out


def inference_windows(segments):
    """Replay faster-whisper's sequential seek from the emitted segments."""
    bounds, seek, idx = [], 0.0, 0
    segs = sorted(segments, key=lambda s: s["start"])
    while idx < len(segs):
        stop = seek + WINDOW_SECONDS
        taken = [s for s in segs[idx:] if s["start"] < stop]
        if not taken:
            seek = segs[idx]["start"]
            continue
        bounds.append((seek, stop))
        idx += len(taken)
        nxt = max(s["end"] for s in taken)
        seek = nxt if nxt > seek else stop
    return bounds


def timed_hyp(window):
    """Frozen-normalizer tokens, each carrying the time of the word it came from."""
    toks, times = [], []
    for seg in window["segments"]:
        for w in seg.get("words") or []:
            for t in wtoks(w["w"]):
                toks.append(t)
                times.append((w["s"], w["e"]))
    return toks, times


def phase_of(t, bounds):
    for lo, hi in bounds:
        if lo <= t < hi:
            return t - lo
    return None


def main():
    decode = json.loads((CACHE / "decode-rw.json").read_text())
    sub = json.loads((CACHE / "substrate-rw.json").read_text())
    phase = [defaultdict(int) for _ in range(N_BINS)]
    absol = defaultdict(lambda: defaultdict(int))
    n_win = 0
    for wid, win in decode["windows"].items():
        meta = sub["windows"].get(wid)
        if meta is None:
            continue
        ref = list(meta["ref"])
        hyp, times = timed_hyp(win)
        if not hyp or not ref:
            continue
        n_win += 1
        bounds = inference_windows(win["segments"])
        path = ops(ref, hyp)
        # time for every reference token: its own word, else the next emitted word
        ref_time = {}
        pending = []
        for op, ri, hj in path:
            if op in ("C", "S"):
                for p in pending:
                    ref_time[p] = times[hj][0]
                pending = []
                ref_time[ri] = times[hj][0]
            elif op == "D":
                pending.append(ri)
        if pending and times:
            for p in pending:
                ref_time[p] = times[-1][1]
        for op, ri, hj in path:
            t = ref_time.get(ri) if ri is not None else (times[hj][0] if hj is not None else None)
            if t is None:
                continue
            ph = phase_of(t, bounds)
            ab = int(t // BIN_SECONDS)
            if ph is not None:
                b = min(int(ph // BIN_SECONDS), N_BINS - 1)
                phase[b][op] += 1
                if op != "I":
                    phase[b]["ref"] += 1
            absol[ab][op] += 1
            if op != "I":
                absol[ab]["ref"] += 1
    print(f"windows: {n_win}\n")
    print("PHASE — θέση μέσα στο ανακατασκευασμένο παράθυρο 30 s")
    print(f"{'δευτ.':>10} {'ref':>7} {'S':>6} {'D':>6} {'I':>6} {'(S+D)/N':>9} {'D/N':>8}")
    for b in range(N_BINS):
        r = phase[b]
        n = r["ref"] or 1
        print(f"{b*5:>3}-{b*5+5:<6} {r['ref']:>7} {r['S']:>6} {r['D']:>6} {r['I']:>6} "
              f"{(r['S']+r['D'])/n:>9.4f} {r['D']/n:>8.4f}")
    print("\nABS — θέση από την αρχή ολόκληρου του παραθύρου αξιολόγησης")
    print(f"{'δευτ.':>10} {'ref':>7} {'S':>6} {'D':>6} {'I':>6} {'(S+D)/N':>9} {'D/N':>8}")
    for b in sorted(absol)[:24]:
        r = absol[b]
        n = r["ref"] or 1
        print(f"{b*5:>3}-{b*5+5:<6} {r['ref']:>7} {r['S']:>6} {r['D']:>6} {r['I']:>6} "
              f"{(r['S']+r['D'])/n:>9.4f} {r['D']/n:>8.4f}")


if __name__ == "__main__":
    main()


def phase_excluding_first_window():
    """The decisive control.

    The first inference window's 0-5 s IS the clip's own 0-5 s, so a phase effect at
    the head could be nothing but the clip edge wearing a disguise. Drop every clip's
    FIRST inference window and every clip's LAST one, and whatever survives belongs to
    the 30 s boundary and to nothing else.
    """
    decode = json.loads((CACHE / "decode-rw.json").read_text())
    sub = json.loads((CACHE / "substrate-rw.json").read_text())
    phase = [defaultdict(int) for _ in range(N_BINS)]
    n_win = kept = 0
    for wid, win in decode["windows"].items():
        meta = sub["windows"].get(wid)
        if meta is None:
            continue
        ref = list(meta["ref"])
        hyp, times = timed_hyp(win)
        if not hyp or not ref:
            continue
        bounds = inference_windows(win["segments"])
        if len(bounds) < 3:
            continue
        n_win += 1
        inner = bounds[1:-1]
        kept += len(inner)
        path = ops(ref, hyp)
        ref_time, pending = {}, []
        for op, ri, hj in path:
            if op in ("C", "S"):
                for p in pending:
                    ref_time[p] = times[hj][0]
                pending = []
                ref_time[ri] = times[hj][0]
            elif op == "D":
                pending.append(ri)
        if pending and times:
            for p in pending:
                ref_time[p] = times[-1][1]
        for op, ri, hj in path:
            t = ref_time.get(ri) if ri is not None else (times[hj][0] if hj is not None else None)
            if t is None:
                continue
            ph = phase_of(t, inner)
            if ph is None:
                continue
            b = min(int(ph // BIN_SECONDS), N_BINS - 1)
            phase[b][op] += 1
            if op != "I":
                phase[b]["ref"] += 1
    print(f"\nPHASE ΧΩΡΙΣ το πρώτο και το τελευταίο παράθυρο κάθε κλιπ "
          f"({n_win} κλιπ, {kept} εσωτερικά παράθυρα)")
    print(f"{'δευτ.':>10} {'ref':>7} {'S':>6} {'D':>6} {'I':>6} {'(S+D)/N':>9} {'D/N':>8}")
    for b in range(N_BINS):
        r = phase[b]
        n = r["ref"] or 1
        print(f"{b*5:>3}-{b*5+5:<6} {r['ref']:>7} {r['S']:>6} {r['D']:>6} {r['I']:>6} "
              f"{(r['S']+r['D'])/n:>9.4f} {r['D']/n:>8.4f}")
