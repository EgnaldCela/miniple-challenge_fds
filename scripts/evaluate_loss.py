"""Score a trusted MiniPLE checkpoint on the fixed course validation loss protocol."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import MiniPLE, MiniPLEConfig
from train import MemmapTokenLoader, choose_device, choose_dtype, evaluate, load_metadata, sha256_file
from validation_reference import TOKENIZER_SHA256, VALIDATION_SHA256

SEQUENCES = 128
SEQ_LEN = 2048


def validation_files(data_dir: Path) -> tuple[Path, str]:
    """Check frozen tokenizer and validation metadata without requiring training files."""
    meta = load_metadata(data_dir)
    if meta.get("format_version") != 2 or meta.get("vocab_size") != 32768:
        raise ValueError("Expected course format_version=2 and vocab_size=32768")
    tok = meta.get("tokenizer", {})
    if tok.get("sha256") != TOKENIZER_SHA256 or sha256_file(data_dir / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("Tokenizer does not match the frozen course tokenizer")
    path = data_dir / "val_l2.bin"
    tokens = int(meta.get("files", {}).get("val_l2.bin", 0))
    if tokens < SEQUENCES * SEQ_LEN + 1 or path.stat().st_size != tokens * 2:
        raise ValueError("Validation token count is missing, incomplete, or too short")
    val_sha = sha256_file(path)
    if val_sha != VALIDATION_SHA256:
        raise ValueError("Validation does not match the frozen course stream")
    return path, val_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("baselines/validation"))
    parser.add_argument("--out", type=Path, default=Path("runs/validation.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    val_path, val_sha = validation_files(args.data_dir)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    if device.type == "cpu" and dtype != torch.float32:
        parser.error("CPU evaluation supports --dtype auto or fp32")
    torch.set_float32_matmul_precision("high")
    # Only load instructor or self-produced checkpoints: full training states use pickle.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = MiniPLEConfig.from_checkpoint(state["model_config"])
    if cfg.vocab_size != 32768 or cfg.max_seq_len != SEQ_LEN:
        raise ValueError("Course scoring requires vocabulary 32768 and context 2048")
    cfg.attn_backend = "flash" if device.type == "cuda" else "sdpa"
    cfg.loss_backend = "liger" if device.type == "cuda" else "chunked"
    cfg.gradient_checkpointing = False
    model = MiniPLE(cfg)
    model.load_state_dict(state["model"], strict=True)
    completed_steps = int(state["step"]) + 1
    tokens_per_step = int(state["schedule"]["tokens_per_step"])
    del state
    model.to(device).eval()
    batch_size = min(args.batch_size, SEQUENCES)
    loader = MemmapTokenLoader(val_path, batch_size, SEQ_LEN,
                               device, seed=1387)
    loss = evaluate(model, loader, SEQUENCES, device, dtype)
    if not math.isfinite(loss):
        raise ValueError("Non-finite validation loss")
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "step": completed_steps, "training_tokens": completed_steps * tokens_per_step,
        "loss": loss, "perplexity": math.exp(loss), "full_vocabulary": True,
        "sequences": SEQUENCES, "sequence_length": SEQ_LEN,
        "validation_tokens": SEQUENCES * SEQ_LEN, "validation_sha256": val_sha,
        "tokenizer_sha256": TOKENIZER_SHA256, "batch_size": batch_size,
        "device": str(device), "dtype": str(dtype), "torch_version": torch.__version__,
        "attention_backend": cfg.attn_backend, "loss_backend": cfg.loss_backend,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
