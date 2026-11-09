"""Published token counts for training alongside append-only data preparation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


def progress_path(path: Path) -> Path:
    return path.with_name(path.name + ".ready.json")


def publish_tokens(path: Path, tokens: int, target: int, *, complete=False, error=None):
    record = {"tokens": tokens, "target_tokens": target, "complete": complete,
              "error": error, "writer_pid": os.getpid(), "updated_unix": time.time()}
    final = progress_path(path)
    temporary = final.with_suffix(final.suffix + ".tmp")
    temporary.write_text(json.dumps(record) + "\n")
    os.replace(temporary, final)


def read_progress(path: Path) -> dict:
    return json.loads(progress_path(path).read_text())


def wait_for_tokens(path: Path, needed: int) -> int:
    """Return a committed readable length; never expose an unwritten file suffix."""
    announced = False
    while True:
        record = read_progress(path)
        if record.get("error"):
            raise RuntimeError(f"Data preparation failed for {path}: {record['error']}")
        ready = int(record["tokens"])
        if ready >= needed:
            return ready
        if record["complete"] or needed > int(record["target_tokens"]):
            raise ValueError(f"{path} cannot supply {needed:,} tokens; target={record['target_tokens']:,}")
        try:
            os.kill(int(record["writer_pid"]), 0)
        except ProcessLookupError as exc:
            raise RuntimeError(f"Data preparation process exited before completing {path}") from exc
        if not announced:
            print(f"Waiting for prepared tokens: {path.name}: {ready:,}/{needed:,}", flush=True)
            announced = True
        time.sleep(2)
