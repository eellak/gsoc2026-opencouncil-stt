#!/usr/bin/env python3
"""Build and verify a portable, content-addressed RunPod training bundle.

The bundle stays under ~/.cache/oc-public because it contains council audio and
transcripts.  Its pack manifest uses relative audio paths, so the same directory
works at /workspace, on a RunPod network volume, or after a direct upload.

Build:
  python3 scripts/pod_bundle.py build [--archive]
Verify locally or on the pod:
  python3 scripts/pod_bundle.py verify BUNDLE_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKS = ROOT / "data/hf-dataset/packs"
DEFAULT_DATA = ROOT / "data/hf-dataset/public"
DEFAULT_OUT = Path.home() / ".cache/oc-public/pod-bundles"
DEFAULT_EVAL_MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
DEFAULT_TRAIN_SAMPLE = Path.home() / ".cache/oc-public/training-wer-2026-08-n300/sample.json"
SCHEMA = "oc-pod-bundle-v1"
CODE_FILES = (
    "notebooks/train_runpod.py",
    "notebooks/run_dense_screen.sh",
    "notebooks/requirements-runpod.txt",
    "scripts/pod_bundle.py",
    "eval/__init__.py",
    "eval/controlled_eval/__init__.py",
    "eval/controlled_eval/bench_data.py",
    "eval/controlled_eval/dense_screen_eval.py",
    "eval/controlled_eval/eval_freeze.py",
    "eval/controlled_eval/exp_same_stack.py",
    "eval/controlled_eval/scoring.py",
)


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def resolve_audio(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def read_pack_rows(manifest: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for line_no, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"pack_id", "audio", "text_p", "text_pn", "dur_sec"} - row.keys()
        if missing:
            raise ValueError(f"{manifest}:{line_no}: missing {sorted(missing)}")
        pack_id = str(row["pack_id"])
        if pack_id in seen:
            raise ValueError(f"duplicate pack_id: {pack_id}")
        seen.add(pack_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"empty pack manifest: {manifest}")
    return rows


def file_inventory(root: Path) -> list[dict]:
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "bundle.json"):
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    return entries


def inventory_id(entries: list[dict]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def add_eval_inputs(temp: Path, eval_manifest: Path | None,
                    training_sample: Path | None) -> dict:
    counts = {"validation_rows": 0, "training_rows": 0}
    if eval_manifest:
        state = json.loads(eval_manifest.read_text())
        rows = state["eval_windows"]
        report_path = (Path.home() / ".cache/oc-public" /
                       f"bench_{state['source_run']}.json")
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text())
        from eval.controlled_eval.eval_freeze import ftoks
        references = {
            item["itemId"]: ftoks(item["referenceText"])
            for item in report["items"]
        }
        portable = []
        for row in rows:
            # The freeze records only an availability boolean; the immutable cache
            # contract names clips by window_id.
            source = Path.home() / ".cache/oc-public/bench_windows" / f"{row['window_id']}.wav"
            if not source.is_file():
                raise FileNotFoundError(source)
            target = temp / "eval/validation/audio" / f"{row['window_id']}{source.suffix}"
            link_or_copy(source, target)
            item = dict(row)
            item["audio_local"] = f"audio/{target.name}"
            reference = references.get(row["window_id"])
            if reference is None:
                raise ValueError(f"missing benchmark reference: {row['window_id']}")
            if len(reference) != row["ref_tokens"]:
                raise ValueError(
                    f"reference count mismatch for {row['window_id']}: "
                    f"freeze={row['ref_tokens']} report={len(reference)}")
            # Keep transcript-bearing material only in the private bundle/cache, and
            # store normalized tokens rather than raw council text.
            item["ref_tokens"] = reference
            portable.append(item)
        target_manifest = temp / "eval/validation/manifest.json"
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        target_manifest.write_text(json.dumps({"eval_windows": portable}, ensure_ascii=False))
        counts["validation_rows"] = len(rows)
    if training_sample:
        state = json.loads(training_sample.read_text())
        portable = []
        for row in state["rows"]:
            source = Path(row["clip"])
            if not source.is_file():
                raise FileNotFoundError(source)
            target = temp / "eval/training/audio" / f"{row['key']}{source.suffix}"
            link_or_copy(source, target)
            item = dict(row)
            item["clip"] = f"audio/{target.name}"
            portable.append(item)
        target_manifest = temp / "eval/training/sample.json"
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        target_manifest.write_text(json.dumps({
            "selection": state["selection"], "rows": portable,
        }, ensure_ascii=False))
        counts["training_rows"] = len(portable)
    return counts


def build(packs: Path, data: Path, output_root: Path, name: str, archive: bool,
          eval_manifest: Path | None = None, training_sample: Path | None = None) -> Path:
    source_manifest = packs / "manifest.jsonl"
    rows = read_pack_rows(source_manifest)
    for filename in ("train.parquet", "validation.parquet"):
        if not (data / filename).is_file():
            raise FileNotFoundError(data / filename)

    output_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=output_root))
    try:
        portable_rows = []
        for row in rows:
            source_audio = resolve_audio(source_manifest, str(row["audio"]))
            if not source_audio.is_file():
                raise FileNotFoundError(source_audio)
            filename = f"{row['pack_id']}{source_audio.suffix.lower()}"
            target_audio = temp / "packs/audio" / filename
            link_or_copy(source_audio, target_audio)
            portable = dict(row)
            portable["audio"] = f"audio/{filename}"
            portable_rows.append(portable)

        manifest = temp / "packs/manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w") as handle:
            for row in portable_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        for filename in ("train.parquet", "validation.parquet"):
            link_or_copy(data / filename, temp / "data/public" / filename)
        for relative in CODE_FILES:
            link_or_copy(ROOT / relative, temp / "code" / relative)
        eval_counts = add_eval_inputs(temp, eval_manifest, training_sample)

        entries = file_inventory(temp)
        bundle_id = inventory_id(entries)
        metadata = {
            "schema": SCHEMA,
            "bundle_id": bundle_id,
            "pack_rows": len(rows),
            **eval_counts,
            "files": entries,
        }
        (temp / "bundle.json").write_text(json.dumps(metadata, indent=2) + "\n")
        destination = output_root / f"{name}-{bundle_id}"
        if destination.exists():
            shutil.rmtree(temp)
        else:
            temp.rename(destination)

        verify(destination)
        if archive:
            archive_path = output_root / f"{destination.name}.tar.zst"
            if not archive_path.exists():
                subprocess.run([
                    "tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0",
                    "--numeric-owner", "-I", "zstd -T0 -3", "-cf", str(archive_path),
                    "-C", str(output_root), destination.name,
                ], check=True)
            checksum = sha256(archive_path)
            archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
                f"{checksum}  {archive_path.name}\n"
            )
        return destination
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def verify(root: Path) -> dict:
    metadata_path = root / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"unsupported bundle schema: {metadata.get('schema')!r}")
    entries = metadata.get("files")
    if not isinstance(entries, list) or metadata.get("bundle_id") != inventory_id(entries):
        raise ValueError("bundle inventory ID mismatch")
    expected = {entry["path"] for entry in entries}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "bundle.json"
    }
    if actual != expected:
        raise ValueError(f"file set mismatch: missing={sorted(expected-actual)[:5]} "
                         f"extra={sorted(actual-expected)[:5]}")
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe inventory path: {relative}")
        path = root / relative
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {relative}")

    pack_manifest = root / "packs/manifest.jsonl"
    rows = read_pack_rows(pack_manifest)
    missing_audio = [row["audio"] for row in rows
                     if not resolve_audio(pack_manifest, row["audio"]).is_file()]
    if missing_audio:
        raise ValueError(f"{len(missing_audio)} pack audio files missing")
    if len(rows) != metadata.get("pack_rows"):
        raise ValueError("pack row count mismatch")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--packs", type=Path, default=DEFAULT_PACKS)
    build_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    build_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    build_parser.add_argument("--name", default="dense-screen")
    build_parser.add_argument("--archive", action="store_true")
    build_parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    build_parser.add_argument("--training-sample", type=Path, default=DEFAULT_TRAIN_SAMPLE)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.packs.resolve(), args.data.resolve(),
                       args.output_root.expanduser().resolve(), args.name, args.archive,
                       args.eval_manifest.expanduser().resolve(),
                       args.training_sample.expanduser().resolve())
        print(result)
    else:
        result = verify(args.bundle.expanduser().resolve())
        print(json.dumps({"bundle_id": result["bundle_id"],
                          "pack_rows": result["pack_rows"], "status": "ok"}))


if __name__ == "__main__":
    main()
