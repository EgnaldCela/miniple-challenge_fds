"""Generate text from a MiniPLE v2 checkpoint on CUDA."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import KVGenerator
from model import MiniPLE, MiniPLEConfig
from tokenizer_utils import load_fixed_nanochat_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Independent continuations of the same prompt; larger batches improve total tokens/s")
    parser.add_argument("--temperature", type=float, default=1.0, help="0 selects greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus probability threshold after top-k filtering; 1 disables it")
    parser.add_argument("--top-k", type=int, default=20, help="0 disables top-k filtering")
    parser.add_argument("--prefill-chunk-size", type=int, default=512,
                        help="Reduce to 128 for large batches to bound prompt-processing memory")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.batch_size < 1 or args.prefill_chunk_size < 1:
        parser.error("Token count, batch size, and prefill chunk size must be positive")
    if not torch.cuda.is_available():
        parser.error("Generation requires CUDA and FlashAttention")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = MiniPLEConfig.from_checkpoint(state["model_config"])
    cfg.attn_backend = "flash"
    cfg.gradient_checkpointing = False
    model = MiniPLE(cfg)
    model.load_state_dict(state["model"], strict=True)
    del state
    model.to(device="cuda", dtype=torch.bfloat16).eval()
    tokenizer, _ = load_fixed_nanochat_tokenizer(args.data_dir)
    bos = tokenizer.token_to_id("<|bos|>")
    ids = [bos] + tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if len(ids) + args.max_new_tokens > cfg.max_seq_len:
        parser.error(f"Prompt plus continuation exceeds the {cfg.max_seq_len}-token context")
    inputs = torch.tensor([ids] * args.batch_size, device="cuda", dtype=torch.long)
    generator = KVGenerator(model, args.batch_size, temperature=args.temperature, top_k=args.top_k,
                            top_p=args.top_p,
                            compile_decode=args.compile, cuda_graph=args.cuda_graph,
                            prefill_chunk_size=args.prefill_chunk_size)
    generator.prepare()
    torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = generator.generate(inputs, args.max_new_tokens)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    continuations = []
    for row in tokens.cpu().tolist():
        if bos in row:
            row = row[:row.index(bos)]
        continuations.append(tokenizer.decode(row, skip_special_tokens=False))
    print(json.dumps({"prompt": args.prompt, "continuations": continuations,
                      "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                      "seconds_including_prefill": seconds,
                      "computed_tokens": tokens.numel(),
                      "total_tokens_per_second": tokens.numel() / seconds,
                      "tokens_per_second_per_sequence": args.max_new_tokens / seconds}, indent=2))


if __name__ == "__main__":
    main()
