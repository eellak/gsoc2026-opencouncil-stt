"""Add unique no-edit hours to the training pool, as a strict superset.

The mixture-ratio experiment (spec `docs/specs/mixture-ratio-preregistration.md`)
needs arm C to reach 80% no-edit at the SAME unique audio duration and the SAME
number of optimizer steps as arm A. The honest way to do that is more unique no-edit
audio, not repeating the clips we already have — the review that shaped this spec was
blunt that repetition changes effective epochs per clip and shrinks acoustic coverage
at the same time as it changes the ratio, which makes the result uninterpretable.

Three properties this script exists to guarantee:

1. **Strict superset.** The per-city selection reuses the export's seed and shuffle,
   so raising the cap appends rows and never reorders the ones already chosen. Arm A's
   backbone is a prefix of arm C's.
2. **Frozen validation.** New rows that would land in validation under the existing
   assignment (a held-out city, or one of the 7 seeded val speakers) are dropped
   rather than added. Both arms are scored against a byte-identical val set.
3. **Preserved city mix.** The extra hours are drawn per city in proportion to that
   city's current train no-edit hours, so arm C is not quietly a Chania/Athens model.

Run order:

    expand_backbone.py select      # write the extra rows, append to rows.parquet
    build.py boundary --shard K/N  # aligns only the new rows (resume skips the rest)
    expand_backbone.py finalize    # write train_expanded.parquet

`select` backs rows.parquet up to rows.parquet.pre-expand once, and refuses to run
twice against an already-expanded file.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
from pathlib import Path

from eval.hf_export.build import (BACKBONE_DUR, OUT, ROWS_PARQUET, TRUST_FRAC,
                                  TRUST_TSV, _iso_to_date, load_speaker_map, log)
from eval.hf_export.sources import span_key
from eval.hf_export.split import VAL_CITIES

BACKUP = OUT / "rows.parquet.pre-expand"
EXTRA = OUT / "backbone_extra_rows.parquet"
SPLIT_JSON = OUT / "split_assignments.json"
TRAIN_EXPANDED = OUT / "public" / "train_expanded.parquet"
BASE_TRAIN = OUT / "public" / "train.parquet"

# Same seed the published export used; changing it breaks the superset property.
EXPORT_SEED = 20260703
BASE_CAP = 2200                 # BACKBONE_CAP_TRAIN at the time of the export


def trusted_meetings() -> set:
    out = set()
    with TRUST_TSV.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                frac = float(r["frac_user"])
            except (TypeError, ValueError):
                frac = 0.0
            if frac >= TRUST_FRAC or r["humanReview"].strip().lower() == "true":
                out.add((r["city"], r["meeting"]))
    return out


def candidates(spk_by_utt: dict) -> dict[str, list[dict]]:
    """Per-city no-edit candidates in the export's own order, before the cap."""
    from eval.autoresearch.prepare_asr import fetch_meeting_json, noedit_utts
    lo, hi = BACKBONE_DUR
    by_city: dict[str, list[dict]] = collections.defaultdict(list)
    for (city, meeting) in sorted(trusted_meetings()):
        d = fetch_meeting_json(city, meeting)
        if d is None:
            continue
        mt = d.get("meeting") or {}
        audio_url = mt.get("audioUrl")
        if not audio_url:
            continue
        mdate = _iso_to_date(mt.get("dateTime"))
        for uid, s, e, txt in noedit_utts(d):
            if not (lo <= (e - s) <= hi):
                continue
            if not spk_by_utt.get(uid):
                continue
            by_city[city].append({
                "utterance_id": uid, "city_id": city, "meeting_id": meeting,
                "meeting_date": mdate, "speaker_id": spk_by_utt[uid],
                "audio_url": audio_url, "start": float(s), "end": float(e),
                "duration_s": float(e - s),
                "initial_before_text": txt, "final_after_text": txt,
                "error_categories": [], "has_overlap": False,
                "source": "no_edit", "split": "train", "reviewer_notes": None,
            })
    return by_city


def cmd_select(args) -> None:
    import numpy as np
    import pandas as pd

    rows = pd.read_parquet(ROWS_PARQUET)
    prev = pd.read_parquet(EXTRA) if EXTRA.exists() else None
    if BACKUP.exists() and prev is None:
        raise SystemExit(f"{BACKUP.name} exists but {EXTRA.name} does not — "
                         f"rows.parquet is in an unknown state, restore the backup.")

    base_train = pd.read_parquet(BASE_TRAIN)
    base_train["dur"] = base_train.end - base_train.start
    ne = base_train[base_train.source == "no_edit"]
    # A second pass tops up an existing expansion; the target is always the TOTAL, so
    # rows already picked count toward it and are never re-picked (span dedupe below).
    cur_h = (ne.groupby("city_id").dur.sum() / 3600).to_dict()
    if prev is not None:
        add = (prev.assign(dur=prev.end - prev.start)
               .groupby("city_id").dur.sum() / 3600).to_dict()
        for c, h in add.items():
            cur_h[c] = cur_h.get(c, 0.0) + h
        log(f"topping up an existing expansion of {len(prev)} rows, "
            f"{prev.duration_s.sum()/3600:.2f}h")
    total_cur = sum(cur_h.values())
    want_extra = args.target_hours - total_cur
    if not args.city_hours and want_extra <= 0:
        raise SystemExit(f"already at {total_cur:.2f}h no-edit train hours")
    log(f"train no-edit now {total_cur:.2f}h, target {args.target_hours:.2f}h, "
        f"need {want_extra:.2f}h more, proportional to the current city mix")

    if args.city_hours:
        # Explicit per-city TOTAL no-edit targets. Needed because a flat proportional
        # expansion cannot hold the city mixture constant once the correction share
        # changes: cities that are correction-heavy (chania, athens) need more no-edit
        # than their current share implies, or arm C quietly under-weights them.
        want = json.loads(Path(args.city_hours).read_text())
        per_city = {c: max(0.0, h - cur_h.get(c, 0.0)) for c, h in want.items()}
        want_extra = sum(per_city.values())
        log(f"explicit per-city targets, need {want_extra:.2f}h more in "
            f"{sum(1 for v in per_city.values() if v > 0.01)} cities")
    else:
        per_city = None

    split = json.loads(SPLIT_JSON.read_text())
    val_speakers = set(split["val_speakers"])
    taken_spans = {span_key(r) for r in rows.to_dict("records")}
    taken_utts = set(rows.utterance_id)

    spk = load_speaker_map()
    by_city = candidates(spk)
    rng = np.random.default_rng(EXPORT_SEED)

    picked: list[dict] = []
    for city in sorted(by_city):
        cand = by_city[city]
        idx = np.arange(len(cand))
        rng.shuffle(idx)                       # same draw the export made
        if city in VAL_CITIES:
            continue
        if per_city is not None:
            need_h = per_city.get(city, 0.0)
            if need_h <= 0.01:
                continue
        elif city not in cur_h:
            continue
        else:
            need_h = cur_h[city] / total_cur * want_extra
        got = 0.0
        n_skip_val = n_skip_dup = 0
        for i in idx[BASE_CAP:]:               # everything past the published cap
            if got >= need_h * 3600:
                break
            r = cand[i]
            if r["speaker_id"] in val_speakers:
                n_skip_val += 1
                continue
            if r["utterance_id"] in taken_utts or span_key(r) in taken_spans:
                n_skip_dup += 1
                continue
            taken_utts.add(r["utterance_id"])
            taken_spans.add(span_key(r))
            picked.append(r)
            got += r["duration_s"]
        log(f"  {city:14s} +{got/3600:5.2f}h / {need_h:5.2f}h wanted "
            f"({sum(1 for p in picked if p['city_id'] == city)} rows, "
            f"skipped {n_skip_val} val-speaker, {n_skip_dup} dup)")

    extra = pd.DataFrame(picked)
    got_h = extra.duration_s.sum() / 3600
    log(f"selected {len(extra)} extra no-edit rows, {got_h:.2f}h")
    if prev is not None:
        extra_all = pd.concat([prev, extra], ignore_index=True)
        log(f"expansion total now {len(extra_all)} rows, "
            f"{extra_all.duration_s.sum()/3600:.2f}h")
    else:
        extra_all = extra
    extra_all.to_parquet(EXTRA, index=False)

    if not BACKUP.exists():
        shutil.copy2(ROWS_PARQUET, BACKUP)
    combined = pd.concat([rows, extra[rows.columns]], ignore_index=True)
    combined.to_parquet(ROWS_PARQUET, index=False)
    log(f"rows.parquet {len(rows)} -> {len(combined)} (backup {BACKUP.name}); "
        f"run `build.py boundary` next, it will align only the {len(extra)} new rows")


def cmd_finalize(args) -> None:
    import numpy as np
    import pandas as pd

    from eval.hf_export.build import _boundary_files, _iter_boundary_lines, _row_sig

    extra = pd.read_parquet(EXTRA)
    expected = {r["utterance_id"]: _row_sig(r) for r in extra.to_dict("records")}
    bnd = {}
    for p in _boundary_files():
        for d in _iter_boundary_lines(p):
            if expected.get(d["utterance_id"]) == d.get("row_sig"):
                bnd[d["utterance_id"]] = d
    missing = set(expected) - set(bnd)
    if missing and not args.allow_pending:
        raise SystemExit(f"{len(missing)} of {len(expected)} extra rows are not "
                         f"aligned yet — finish `build.py boundary` or pass "
                         f"--allow-pending to drop them")
    if missing:
        log(f"dropping {len(missing)} unaligned rows (--allow-pending)")
        extra = extra[[u not in missing for u in extra.utterance_id]]

    failed = [u for u, d in bnd.items() if d["boundary_status"] == "align_failed"]
    log(f"align_failed: {len(failed)} dropped")
    extra = extra[[u not in set(failed) for u in extra.utterance_id]]

    base = pd.read_parquet(BASE_TRAIN)
    out = extra.rename(columns={"initial_before_text": "before_text",
                                "final_after_text": "text"}).copy()
    out["start_adj"] = [bnd[u]["start_adj"] for u in out.utterance_id]
    out["end_adj"] = [bnd[u]["end_adj"] for u in out.utterance_id]
    out["boundary_status"] = [bnd[u]["boundary_status"] for u in out.utterance_id]
    out = out[[c for c in base.columns]]
    combined = pd.concat([base, out], ignore_index=True)
    combined.to_parquet(TRAIN_EXPANDED, index=False)

    d = combined.end - combined.start
    by = combined.assign(dur=d).groupby("source").dur.sum() / 3600
    log(f"train_expanded.parquet: {len(combined)} rows "
        f"(was {len(base)}), corrections {by.get('correction', 0):.2f}h, "
        f"no_edit {by.get('no_edit', 0):.2f}h")
    log(f"wrote {TRAIN_EXPANDED}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--target-hours", type=float, default=17.98,
                   help="total train no-edit hours to reach (default: the 80/20 arm)")
    p.add_argument("--city-hours", default="",
                   help="JSON {city: total no-edit hours}; overrides --target-hours "
                        "and the proportional split")
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("finalize")
    p.add_argument("--allow-pending", action="store_true")
    p.set_defaults(fn=cmd_finalize)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
