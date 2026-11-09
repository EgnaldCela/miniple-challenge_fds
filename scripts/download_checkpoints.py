"""Download selected release checkpoints and verify their recorded sizes and SHA256 hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

from verify_artifacts import CHECKPOINT_FILES, verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def release_url(repository: str, tag: str, checkpoint: str) -> str:
    if not isinstance(repository, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must have the form OWNER/REPO")
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("Release tag must be a nonempty string")
    if checkpoint not in CHECKPOINT_FILES:
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    return (f"https://github.com/{repository}/releases/download/{quote(tag, safe='')}/"
            f"miniple-{checkpoint}.pt")


def download_checkpoint(root: Path, checkpoint: str, url: str) -> Path:
    manifest = root / "baselines/manifest.json"
    # Validate all manifest metadata and paths before opening a network connection.
    verify_manifest(manifest, entries=[])
    name = CHECKPOINT_FILES[checkpoint]
    records = json.loads(manifest.read_text())
    if name not in records:
        raise ValueError(f"Checkpoint missing from manifest: {name}")
    record = records[name]
    destination = manifest.parent / name
    if destination.exists():
        verify_manifest(manifest, entries=[name])
        print(f"Already verified: {destination}", flush=True)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".part",
                                         delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            total = 0
            next_report = 64 * 1024 * 1024
            print(f"Downloading {checkpoint} checkpoint ({record['bytes'] / 1e9:.2f} GB)", flush=True)
            with urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > record["bytes"]:
                        raise ValueError(f"Download exceeds expected size for {name}")
                    output.write(chunk)
                    digest.update(chunk)
                    if total >= next_report:
                        print(f"  {total / record['bytes']:.0%}", flush=True)
                        next_report += 64 * 1024 * 1024
            if total != record["bytes"]:
                raise ValueError(f"Byte count mismatch for {name}: expected {record['bytes']}, got {total}")
            if digest.hexdigest() != record["sha256"].lower():
                raise ValueError(f"SHA256 mismatch for {name}")
        # Both paths share a filesystem; linking publishes atomically without overwriting.
        os.link(temporary, destination)
        print(f"Downloaded and verified: {destination}", flush=True)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=("3b", "50b", "all"), required=True,
                        help="Download only the reference needed for your track, or both")
    parser.add_argument("--repository", help="GitHub OWNER/REPO (default: distribution.json)")
    parser.add_argument("--tag", help="Release tag (default: distribution.json)")
    args = parser.parse_args()
    try:
        config = json.loads((ROOT / "distribution.json").read_text())
        repository = args.repository or config["repository"]
        tag = args.tag or config["release"]
        verify_manifest(ROOT / "baselines/validation/manifest.json")
        selected = CHECKPOINT_FILES if args.checkpoint == "all" else [args.checkpoint]
        for checkpoint in selected:
            download_checkpoint(ROOT, checkpoint, release_url(repository, tag, checkpoint))
    except HTTPError as exc:
        parser.exit(1, f"FAIL: HTTP {exc.code}. Check the repository, release tag and asset access. "
                    "For private releases, see docs/DISTRIBUTION.md.\n")
    except (OSError, ValueError, TypeError, KeyError, URLError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
