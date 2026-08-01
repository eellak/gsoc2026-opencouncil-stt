"""Shared scorer for the controlled evals.

Lifted verbatim from `ab_hotwords.py` / `ab_corrected_utterances.py`, which had been
carrying the same four functions by copy-paste. Numbers produced here are directly
comparable with those runs; do not "improve" the normalization without re-running
everything that quotes a WER.

Normalization is deliberately shallow: NFD, strip combining marks, lowercase, and
split on `\\w+`. It does NOT fold final sigma, so `ς` and `σ` are distinct tokens.
"""
import collections
import re
import unicodedata


def norm(s):
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def wtoks(s):
    return re.findall(r"\w+", norm(s))


def edist(a, b):
    """Levenshtein distance over any two sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


def counts(refs, hyps):
    """Per-utterance (word_errors, ref_words) pairs — the unit corpus WER sums over."""
    out = []
    for r, h in zip(refs, hyps):
        rt, ht = wtoks(r), wtoks(h)
        out.append((edist(rt, ht), len(rt)))
    return out


def wer(counts_):
    e = sum(c[0] for c in counts_)
    n = sum(c[1] for c in counts_)
    return e / n if n else float("nan")


def cluster_bootstrap(counts_a, counts_b, clusters, n_boot=10000, seed=7):
    """Paired CI on WER(a) - WER(b), resampling clusters (meetings), not utterances.

    Utterances from one meeting share a speaker, a room and a topic, so they are not
    independent draws. Resampling them individually reports confidence the data does
    not have.
    """
    import numpy as np
    assert len(counts_a) == len(counts_b) == len(clusters) > 0
    assert all(a[1] == b[1] for a, b in zip(counts_a, counts_b)), \
        "arms scored against different references — the pairing is broken"
    groups = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = sorted(groups)
    a = np.array(counts_a, dtype=float)
    b = np.array(counts_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        den = a[idx, 1].sum()
        diffs[i] = (a[idx, 0].sum() - b[idx, 0].sum()) / den if den else np.nan
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    point = wer(counts_a) - wer(counts_b)
    return {"delta": point, "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_clusters": len(keys)}


def head2head(counts_a, counts_b):
    """win/tie/loss for a vs b, per utterance (a better = fewer errors)."""
    w = t = l = 0
    for x, y in zip(counts_a, counts_b):
        if x[0] < y[0]:
            w += 1
        elif x[0] == y[0]:
            t += 1
        else:
            l += 1
    return {"a_better": w, "tie": t, "b_better": l}
