"""Combined-dataset publish/leakage verification (Codex hard checks).

Run after `build finalize`. Asserts speaker-disjointness, no span collision,
temporal/no-pending/no-notes invariants, val window, and reports source x split.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "data" / "hf-dataset" / "public"


def main() -> int:
    t = pd.read_parquet(PUB / "train.parquet")
    v = pd.read_parquet(PUB / "validation.parquet")

    # 1) speaker identity is the authority: no person_id in both splits
    assert not (set(t.speaker_id.dropna()) & set(v.speaker_id.dropna())), "speaker leakage"
    # 2) no held-out city rows in train
    assert t[t.city_id.isin({"orestiada", "argos"})].empty, "held-out city in train"
    # 3) temporal test withheld
    assert (t.meeting_date < "2026-06-01").all() and (v.meeting_date < "2026-06-01").all()
    # 4) unique + disjoint utterance ids
    assert t.utterance_id.is_unique and v.utterance_id.is_unique
    assert not (set(t.utterance_id) & set(v.utterance_id)), "utterance overlap across splits"
    # 5) no span collision across the two files (canonical clip identity)
    def spans(df):
        return set(zip(df.audio_url, df.start.round(2), df.end.round(2)))
    assert not (spans(t) & spans(v)), "same audio span in both splits"
    # 6) publish hygiene
    for df_ in (t, v):
        assert "reviewer_notes" not in df_.columns
        assert (df_.boundary_status != "pending").all(), "pending boundary in published set"
        # no row is align_failed (all sources gated out of the release)
        assert df_[df_.boundary_status == "align_failed"].empty
        assert (df_.boundary_status != "pending").all()
        # corrected spans present for every published row
        assert df_["start_adj"].notna().all()

    stats = json.load(open(ROOT / "data" / "hf-dataset" / "stats.json"))
    share = stats["by_split"]["validation"]["pct_hours"]
    assert 18 <= share <= 22, f"val share {share}% outside 18-22"

    def mix(df):
        return dict(df.source.value_counts())
    print(f"OK  train {len(t)} rows {mix(t)} | val {len(v)} rows {mix(v)} | "
          f"val {share}% of hours")
    print(f"total: {stats['total_rows']} rows, {stats['total_hours']} h; "
          f"by_source: { {k: v['rows'] for k, v in stats['by_source'].items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
