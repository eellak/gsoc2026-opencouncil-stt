"""Chain reconstruction + classification.

Spec: docs/specs/fix-task-eval-harness.md §1, runbook Step 1.

Per utterance_id, sort edits by timestamp and reconstruct a chain:
  input_raw   = first before_text  (raw STT)
  gold_final  = last  after_text   (human-final reference)
The user edit's input is the task's output, so user edits in a task>user chain
are the residual errors the fix-task left uncorrected.

chain_type heuristic (first pass; refine on a sample):
  pure_correction  — orthographic/lexical/entity fixes only (high similarity)
  mixed            — correction + added/removed content (medium similarity)
  semantic_rewrite — meaning changed (low similarity)
"""
from __future__ import annotations

from rapidfuzz import fuzz

from eval.scoring import greek_normalize

# similarity thresholds on the Greek-normalised char form
PURE_MIN = 0.85
MIXED_MIN = 0.50


def classify_chain(input_raw: str, gold_final: str) -> str:
    ni, ng = greek_normalize(input_raw), greek_normalize(gold_final)
    if ni == ng:
        # only accent/punct/sigma/spacing differed -> formatting-level fix
        return "pure_correction"
    sim = fuzz.ratio(ni, ng) / 100.0
    if sim >= PURE_MIN:
        return "pure_correction"
    if sim >= MIXED_MIN:
        return "mixed"
    return "semantic_rewrite"


def reconstruct_chains(df) -> list[dict]:
    """Reconstruct chains from a corrections DataFrame.

    Expects columns: utterance_id, edit_timestamp, before_text, after_text,
    edited_by, meeting_id, city_id (extra columns are ignored).
    """
    # sort once globally (stable) so each group is already timestamp-ordered
    df = df.sort_values(["utterance_id", "edit_timestamp"], kind="stable")
    chains: list[dict] = []
    for uid, grp in df.groupby("utterance_id", sort=False):
        befores = grp["before_text"].astype(str).tolist()
        afters = grp["after_text"].astype(str).tolist()
        seq = grp["edited_by"].astype(str).tolist()

        input_raw = befores[0]
        gold_final = afters[-1]

        # link integrity: each before == previous after
        links_ok = all(
            greek_normalize(befores[k]) == greek_normalize(afters[k - 1])
            for k in range(1, len(befores))
        )

        has_task = "task" in seq
        has_user = "user" in seq
        task_then_user = any(
            seq[i] == "task" and "user" in seq[i + 1 :] for i in range(len(seq))
        )

        chains.append(
            {
                "utterance_id": uid,
                "meeting_id": grp["meeting_id"].iloc[0],
                "city_id": grp["city_id"].iloc[0],
                "input_raw": input_raw,
                "gold_final": gold_final,
                "edited_by_seq": seq,
                "has_task": has_task,
                "has_user": has_user,
                "task_then_user": task_then_user,
                "chain_type": classify_chain(input_raw, gold_final),
                "n_edits": len(seq),
                "links_ok": links_ok,
            }
        )
    return chains
