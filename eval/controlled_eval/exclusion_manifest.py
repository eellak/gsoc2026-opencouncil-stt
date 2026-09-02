"""The exclusion manifest: which meetings may never supply a listening clip.

Spec: docs/specs/2026-08-26-training-round-2.md section 4.2.
Record: exp-2026-08-26-training-round-2.

Nothing is cut for the listening batch until this manifest is written and hashed, and
the hash goes in the ledger record. A clip taken from an evaluation meeting is training
on the test set, and that is not reversible.

Five sources, four id shapes, one canonical key.

  held_out_391     the 391 post-june held-out windows       (city, meeting)
  validation_39    the 39 frozen validation windows         (city, meeting)
  confirm_cities   the five confirmation cities             city only
  gold_set         the gold set                             packed `gc_<city>_<meeting>_<offset>`
  calibration_260  the 260-window benchmark run             (city, meeting)

`calibration_260` is here because of the user decision of 2026-08-27: the 137 of those
windows that survive the other four exclusions are the only material carrying five
independent witnesses, so they are used to measure the pollution rate of single-witness
mining -- and for that reason they must never also become training data.

AMENDED 2026-08-28. That exclusion collapsed the listening queue to two municipalities
-- chania and athens, 523 clips -- because `calibration_260` alone blocked 1,305 of the
2,661 candidates. A queue of two cities cannot separate "the new data helps" from "the
new data matches Chania and Athens", which is the question step 3 exists to ask. Harold
released two cities from THIS SOURCE ONLY, and the price was measured before the
decision rather than after:

  released   chalandri, vrilissia  -> 33 of the source's 203 meetings
  bought     721 clips / 2.16 h / 4 cities, from 523 / 1.56 h / 2
  paid       38 of the 137 five-witness calibration windows (99 remain), and
             36 of the 247 windows of the `composition-rt-2026-08` fusion substrate
             (211 remain), which is what a round-3 adapter may no longer be scored on

zografou was NOT released: it buys 66 clips for 19 calibration and 18 substrate
windows, the worst trade of the three. There was no free subset -- every meeting that
`calibration_260` blocks alone holds at least one calibration window, so any widening
spends evaluation material.

The release drops keys from `calibration_260` only. A released meeting that another
source also names stays excluded, because the manifest is the union; two of the 33 are
in that position.

The canonical key is `(city, meeting)`, never meeting alone. Meeting slugs repeat
across cities and collapse 117 real meetings into 66
(`exp_composition_postjune.py:212-216`); a meeting-only key silently over-excludes.

A note for whoever edits `SOURCES`. `held_out_391` reads the post-june *report*. It
does not read `~/.cache/oc-public/bench_windows`, which holds the 260 wavs of the
2026-06-10 / 2026-08-10 run and matches **0** of the 391. An earlier draft of the spec
named that directory, and a builder written from it would have excluded the wrong
benchmark. `test_bench_windows_is_not_the_391_source` exists to keep that from
happening again.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SC = Path.home() / ".cache" / "oc-public"
ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "data" / "asr" / "audio"  # <city>__<meeting>.mp3

# The five confirmation cities, restated from chooser/autoprompt.py:40. Held as a
# literal because that module pulls in the whole chooser harness on import.
CONFIRM_CITIES = ("argos", "samothraki", "sparta", "xylokastro", "argithea")

# The cities released from `calibration_260` on 2026-08-28, and from that source ONLY.
# Widening this tuple spends evaluation windows that cannot be bought back once an
# adapter has trained on them; `test_the_release_price_in_evaluation_windows_is_asserted`
# holds the measured price so a wider release has to restate it.
CALIBRATION_RELEASED_CITIES = ("chalandri", "vrilissia")

_GOLD_CELL_RE = re.compile(r"^gc_(?P<city>[^_]+)_(?P<meeting>.+)_(?P<offset>\d+)$")

# The trailing offset is all digits and so is the year that ends every meeting slug,
# which makes `gc_argos_apr30_2_2026` and `gc_argos_apr30_2_2026_1264428` look alike to
# the regex above. The discriminator is structural: a meeting slug always ends in a
# four-digit year (`jul29_2026`, `apr30_2_2026`), so a parse whose meeting does not is
# a malformed id rather than an unusual one. Failing loudly is the right behaviour for
# an exclusion list -- a silently mis-parsed gold cell would leave a gold meeting
# eligible for clip cutting.
_MEETING_YEAR_RE = re.compile(r"_\d{4}$")


@dataclass(frozen=True)
class Source:
    path: str
    kind: str  # "meeting" | "city"
    note: str


SOURCES: dict[str, Source] = {
    "held_out_391": Source(
        path=str(SC / "bench_2026-08-22-post-june-held-out-test-clean-pack-cont-.json"),
        kind="meeting",
        note="the 391 post-june held-out windows; NOT the bench_windows wav directory",
    ),
    "validation_39": Source(
        path=str(ROOT / "research" / "eval-freeze-2026-08" / "manifest.json"),
        kind="meeting",
        note="the 39 frozen validation windows, built by eval_freeze.py:84-98",
    ),
    "confirm_cities": Source(
        path="eval/controlled_eval/chooser/autoprompt.py:40",
        kind="city",
        note="exp-2026-08-26-category-chooser confirmation cities; city-level, no meetings",
    ),
    "gold_set": Source(
        path=str(SC / "gold-set-2026-08" / "manifest.json"),
        kind="meeting",
        note="scored + warmup cells, ids packed as gc_<city>_<meeting>_<offset>",
    ),
    "calibration_260": Source(
        path=str(SC / "bench_2026-08-10-corrected-adapter-label-prefix-fix-vs-ju.json"),
        kind="meeting",
        note=("calibration-only per the user decision of 2026-08-27, less the cities "
              "released 2026-08-28: " + ", ".join(CALIBRATION_RELEASED_CITIES)),
    ),
}


@dataclass(frozen=True)
class ResolvedSource:
    path: str
    kind: str
    note: str
    keys: frozenset  # (city, meeting) pairs, or bare city strings when kind == "city"


@dataclass(frozen=True)
class Manifest:
    meeting_keys: frozenset
    city_keys: frozenset
    sources: dict = field(default_factory=dict)
    # Section 4.2 asks for source audio hashes as well as ids, because ids alone miss
    # a renamed or duplicated meeting. Empty unless the manifest was built with
    # `with_audio=True`; `audio_missing` names what could not be hashed rather than
    # letting an absent file look like a meeting with no audio.
    audio_sha256: dict = field(default_factory=dict)
    audio_missing: frozenset = frozenset()
    # The cities released from `calibration_260` on 2026-08-28. Carried in the hashed
    # payload rather than in the code alone: the ledger vouches for a sha256, and a
    # release list that lives only in a constant would let the manifest stay valid
    # while the decision behind it changed.
    released_cities: frozenset = frozenset()
    # Set by `load_manifest` to the verified digest of the file it read, and left None
    # for a manifest built from sources. A loaded manifest cannot rebuild `sources` or
    # the audio hashes, so re-hashing it would produce a DIFFERENT and meaningless
    # digest; `sha256` refuses rather than return one, and this field is what a caller
    # records in the ledger.
    loaded_sha256: str | None = None

    def is_excluded(self, city: str, meeting: str) -> bool:
        return city in self.city_keys or (city, meeting) in self.meeting_keys

    def to_json(self) -> str:
        """Canonical, sorted, stable across runs. This string is what gets hashed."""
        return json.dumps(
            {
                "meeting_keys": sorted([c, m] for c, m in self.meeting_keys),
                "city_keys": sorted(self.city_keys),
                "sources": {
                    name: {
                        "path": s.path,
                        "kind": s.kind,
                        "note": s.note,
                        "n_keys": len(s.keys),
                    }
                    for name, s in sorted(self.sources.items())
                },
                "audio_sha256": {
                    f"{c}__{m}": h for (c, m), h in sorted(self.audio_sha256.items())
                },
                "audio_missing": sorted(f"{c}__{m}" for c, m in self.audio_missing),
                "calibration_released_cities": sorted(self.released_cities),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )

    @property
    def sha256(self) -> str:
        if self.loaded_sha256:
            raise ValueError(
                "this manifest was read from disk, so its sources and audio hashes are "
                "not reconstructed and re-hashing it would give a digest that matches "
                "no file. Use `loaded_sha256` for the file it came from, or "
                "`build_manifest()` to hash a freshly built one.")
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def parse_gold_cell_id(cell_id: str) -> tuple[str, str]:
    """`gc_<city>_<meeting>_<offset_ms>` -> (city, meeting).

    The meeting slug itself carries underscores and digits (`apr30_2_2026`), so the
    offset is taken as the trailing all-digit segment and the meeting is everything
    between the city and it. Anything else is an error rather than a guess: a
    mis-parsed gold cell would silently fail to exclude a gold meeting.
    """
    m = _GOLD_CELL_RE.match(cell_id or "")
    if not m:
        raise ValueError(f"not a gold cell id: {cell_id!r}")
    meeting = m.group("meeting")
    if not _MEETING_YEAR_RE.search(meeting):
        raise ValueError(
            f"gold cell id {cell_id!r} parses to meeting {meeting!r}, which does not "
            "end in a four-digit year -- the trailing segment is probably the year "
            "rather than an offset, so this id is missing its offset"
        )
    return m.group("city"), meeting


def audio_path_for(city: str, meeting: str, audio_dir: Path = AUDIO_DIR) -> Path:
    return Path(audio_dir) / f"{city}__{meeting}.mp3"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_audio(keys, audio_dir: Path = AUDIO_DIR) -> tuple[dict, frozenset]:
    """sha256 of every local source recording, plus the keys with no local file.

    Missing files are reported rather than skipped: a meeting whose audio is absent
    cannot be checked for renaming, and the caller needs to know which ones those are
    before trusting the manifest as a rename guard.
    """
    hashes: dict = {}
    missing: set = set()
    for city, meeting in sorted(keys):
        p = audio_path_for(city, meeting, audio_dir)
        if p.exists():
            hashes[(city, meeting)] = _sha256_file(p)
        else:
            missing.add((city, meeting))
    return hashes, frozenset(missing)


def _keys_from_bench_report(path: Path) -> frozenset:
    items = json.loads(Path(path).read_text())["items"]
    return frozenset((it["cityId"], it["meetingId"]) for it in items)


def _keys_from_eval_freeze(path: Path) -> frozenset:
    windows = json.loads(Path(path).read_text())["eval_windows"]
    return frozenset((w["city"], w["meeting_id"]) for w in windows)


def _keys_from_gold_set(path: Path) -> frozenset:
    payload = json.loads(Path(path).read_text())
    cells = list(payload.get("warmup_cell_ids") or []) + list(payload.get("scored_cell_ids") or [])
    if not cells:
        raise ValueError(f"gold-set manifest {path} carries no cell ids")
    return frozenset(parse_gold_cell_id(c) for c in cells)


def _keys_from_calibration_260(path: Path) -> frozenset:
    """The 2026-08-10 benchmark meetings, less the cities released on 2026-08-28.

    A separate reader rather than a filter over the shared one: `held_out_391` reads
    the same shape and must NOT be filtered, and a shared reader with a city argument
    is one default away from releasing the held-out set too.
    """
    keys = _keys_from_bench_report(path)
    return frozenset(k for k in keys if k[0] not in CALIBRATION_RELEASED_CITIES)


_READERS = {
    "held_out_391": _keys_from_bench_report,
    "validation_39": _keys_from_eval_freeze,
    "gold_set": _keys_from_gold_set,
    "calibration_260": _keys_from_calibration_260,
}


def build_manifest(with_audio: bool = False, audio_dir: Path = AUDIO_DIR) -> Manifest:
    """Read all five sources, normalise, and return the hashable manifest.

    `with_audio` is off by default so tests and quick checks stay fast; the
    operational entry point turns it on, because section 4.2 wants the audio hashes in
    the file whose sha256 goes in the ledger.
    """
    resolved: dict[str, ResolvedSource] = {}
    meeting_keys: set = set()
    city_keys: set = set()

    for name, src in SOURCES.items():
        if src.kind == "city":
            keys = frozenset(CONFIRM_CITIES)
            city_keys |= set(keys)
        else:
            keys = _READERS[name](Path(src.path))
            meeting_keys |= set(keys)
        if not keys:
            raise ValueError(f"exclusion source {name!r} resolved to nothing")
        resolved[name] = ResolvedSource(src.path, src.kind, src.note, keys)

    hashes: dict = {}
    missing: frozenset = frozenset()
    if with_audio:
        hashes, missing = hash_audio(meeting_keys, audio_dir)

    return Manifest(
        meeting_keys=frozenset(meeting_keys),
        city_keys=frozenset(city_keys),
        sources=resolved,
        audio_sha256=hashes,
        audio_missing=missing,
        released_cities=frozenset(CALIBRATION_RELEASED_CITIES),
    )


MANIFEST_PATH = SC / "exclusion-manifest-2026-08" / "manifest.json"

# The manifest of record for exp-2026-08-26-training-round-2. Anything that cuts a clip
# pins this: a manifest that has drifted is not the manifest the record vouches for, and
# the cost of finding out later is a contaminated benchmark.
#
# Rebuilt 2026-08-28 for the chalandri + vrilissia release; the 2026-08-27 file, whose
# sha256 was 11534d6f29ec03111586ffabc878b94c0b0be1a59f63159d6f4bf68735a641e3, is kept
# beside it as manifest-2026-08-27-superseded.json so the earlier queue stays auditable.
FROZEN_SHA256 = "a45de4c6c670124d0edfdec6daed39264ceb61cb6f114aed6fcb4ba6904d9a2c"


def load_manifest(path: Path = MANIFEST_PATH, expect_sha: str = FROZEN_SHA256,
                  cross_check: bool = True) -> Manifest:
    """Read the frozen manifest, verify its hash, and check it against live sources.

    The file's bytes ARE `to_json()`, so the recorded sha256 is the sha256 of the file
    and is checked directly rather than by round-tripping (a round trip would have to
    reconstruct `sources[*].keys`, which the file does not carry).

    `cross_check` rebuilds the key sets from the five live sources -- without hashing
    audio, which is slow -- and requires them to be identical. A source that has since
    grown a meeting is caught here rather than in a benchmark six weeks later. The
    audio hashes in the file are NOT re-verified on load; `main` is what writes them.
    """
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if expect_sha and got != expect_sha:
        raise ValueError(
            f"exclusion manifest {path} hashes to {got}, not the frozen "
            f"{expect_sha}. Do not cut clips against it: rebuild it, work out what "
            "moved, and record the new hash in the ledger before continuing."
        )
    d = json.loads(raw.decode("utf-8"))
    man = Manifest(
        meeting_keys=frozenset((c, m) for c, m in d["meeting_keys"]),
        city_keys=frozenset(d["city_keys"]),
        audio_sha256={
            tuple(k.split("__", 1)): v for k, v in d.get("audio_sha256", {}).items()},
        audio_missing=frozenset(
            tuple(k.split("__", 1)) for k in d.get("audio_missing", [])),
        released_cities=frozenset(d.get("calibration_released_cities", [])),
        loaded_sha256=got,
    )
    if man.released_cities != frozenset(CALIBRATION_RELEASED_CITIES):
        raise ValueError(
            f"{path} was written with calibration_260 released for "
            f"{sorted(man.released_cities)}, but the code now releases "
            f"{sorted(CALIBRATION_RELEASED_CITIES)}. One of the two moved without the "
            "other; rebuild the manifest and restate the price in evaluation windows "
            "before cutting anything."
        )
    if cross_check:
        live = build_manifest(with_audio=False)
        if live.meeting_keys != man.meeting_keys or live.city_keys != man.city_keys:
            added = (live.meeting_keys | live.city_keys) - (
                man.meeting_keys | man.city_keys)
            gone = (man.meeting_keys | man.city_keys) - (
                live.meeting_keys | live.city_keys)
            raise ValueError(
                f"the live sources no longer agree with {path}: "
                f"{len(added)} new keys, {len(gone)} vanished. Rebuild and re-record."
            )
    return man


if __name__ == "__main__":  # pragma: no cover - operational entry point
    man = build_manifest(with_audio=True)
    out = SC / "exclusion-manifest-2026-08" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(man.to_json(), encoding="utf-8")
    print(f"{len(man.meeting_keys)} meetings, {len(man.city_keys)} whole cities")
    for name, s in sorted(man.sources.items()):
        print(f"  {name:16s} {s.kind:8s} {len(s.keys):5d}  {s.path}")
    print(f"audio hashed {len(man.audio_sha256)}, no local file {len(man.audio_missing)}")
    print(f"sha256 {man.sha256}")
    print(f"written {out}")
