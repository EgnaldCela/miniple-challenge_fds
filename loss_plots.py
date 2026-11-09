"""Save training and full-vocabulary validation loss curves from metrics.jsonl."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_loss_plots(out_dir: Path) -> None:
    path = out_dir / "metrics.jsonl"
    if not path.exists():
        return
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A watcher may see the final line while the trainer appends it.
            continue
    training = [r for r in records if r.get("event") == "train"]
    validation = [r for r in records if r.get("event") == "validation"]
    if not training and not validation:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for width in sorted({r["candidate_max"] for r in training}):
        rows = [r for r in training if r["candidate_max"] == width]
        label = "Full vocabulary" if width == 32768 else f"{width:,} candidates"
        axes[0].plot([r["tokens"] / 1e9 for r in rows], [r["loss"] for r in rows],
                     linewidth=1, marker="." if len(rows) == 1 else None, label=label)
    if training:
        axes[0].legend(fontsize=8)
    if validation:
        axes[1].plot([r["tokens"] / 1e9 for r in validation],
                     [r["loss"] for r in validation], color="tab:red", marker=".", linewidth=1)
    else:
        axes[1].text(.5, .5, "First validation pending", transform=axes[1].transAxes,
                     ha="center", va="center")
    current = max(r["tokens"] for r in training + validation)
    start = next((r for r in records if r.get("event") == "start"), {})
    schedule = start.get("schedule", {})
    for ax, title in zip(axes, ("Training CE (candidate-dependent)", "Full-vocabulary L2 validation CE")):
        for key, label in (("mix_start_scheduled_tokens", "Data mix"),
                           ("stage1_scheduled_tokens", "Cooldown / full softmax")):
            boundary = schedule.get(key, 0)
            if 0 < boundary <= current:
                ax.axvline(boundary / 1e9, color="gray", linestyle="--", alpha=.6)
                ax.text(boundary / 1e9, .98, label, rotation=90, va="top", fontsize=7,
                        transform=ax.get_xaxis_transform())
        ax.set(title=title, xlabel="Predicted tokens (billions)", ylabel="Cross entropy")
        ax.grid(alpha=.25)
    fig.suptitle(out_dir.name)
    for extension in ("png", "svg"):
        final = out_dir / f"losses.{extension}"
        temporary = out_dir / f"losses.{extension}.tmp"
        fig.savefig(temporary, format=extension, dpi=150)
        os.replace(temporary, final)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dirs", type=Path, nargs="+")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    modified = {}
    while True:
        for out_dir in args.out_dirs:
            path = out_dir / "metrics.jsonl"
            stamp = path.stat().st_mtime_ns if path.exists() else None
            if stamp is not None and stamp != modified.get(path):
                save_loss_plots(out_dir)
                modified[path] = stamp
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
