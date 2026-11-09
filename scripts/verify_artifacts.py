"""Verify the supplied validation and checkpoint manifests without downloading anything."""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

MANIFESTS = ("baselines/validation/manifest.json", "baselines/manifest.json")
CHECKPOINT_FILES = {"3b": "3b/final_l3_mix.pt", "50b": "50b/final_l3_mix.pt"}


def verify_manifest(manifest: Path, entries: Iterable[str] | None = None) -> int:
    """Validate a manifest and verify all files, or only the named entries."""
    records = json.loads(manifest.read_text())
    if not isinstance(records, dict) or not records:
        raise ValueError(f"Manifest must be a nonempty filename mapping: {manifest}")
    if isinstance(entries, str):
        raise ValueError("entries must be an iterable of filenames, not one string")
    selected = set(records if entries is None else entries)
    if not all(isinstance(name, str) for name in selected):
        raise ValueError("Selected manifest entries must be filenames")
    unknown = selected - records.keys()
    if unknown:
        raise ValueError(f"Unknown manifest entries in {manifest}: {sorted(unknown)}")
    base = manifest.parent.resolve()
    for name, record in records.items():
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != name or "\\" in name or name == "."):
            raise ValueError(f"Unsafe relative artifact path in {manifest}: {name!r}")
        path = base / name
        if not path.resolve().is_relative_to(base):
            raise ValueError(f"Artifact resolves outside its manifest directory: {name}")
        if (not isinstance(record, dict) or type(record.get("bytes")) is not int
                or record["bytes"] < 0 or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", record["sha256"])):
            raise ValueError(f"Invalid byte count or SHA256 entry: {manifest}: {name}")
        if name not in selected:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != record["bytes"]:
            raise ValueError(f"Byte count mismatch for {path}: expected {record['bytes']}, got {actual_bytes}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != record["sha256"].lower():
            raise ValueError(f"SHA256 mismatch for {path}: expected {record['sha256']}, got {actual_sha}")
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Assignment folder containing baselines/")
    parser.add_argument("--checkpoints", choices=("none", "3b", "50b", "all"), default="none",
                        help="Which downloaded checkpoints to verify; validation is always checked")
    args = parser.parse_args()
    try:
        count = verify_manifest(args.root / MANIFESTS[0])
        print(f"PASS: {MANIFESTS[0]} ({count} files)", flush=True)
        if args.checkpoints != "none":
            entries = (CHECKPOINT_FILES.values() if args.checkpoints == "all"
                       else [CHECKPOINT_FILES[args.checkpoints]])
            count = verify_manifest(args.root / MANIFESTS[1], entries)
            print(f"PASS: {MANIFESTS[1]} ({count} files)", flush=True)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
