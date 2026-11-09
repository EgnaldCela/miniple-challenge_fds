"""Zero-shot HellaSwag, PIQA, ARC-Easy, and ARC-Challenge evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.model import TemplateLM
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import MiniPLE, MiniPLEConfig
from tokenizer_utils import load_fixed_nanochat_tokenizer


# Match the dataset snapshots used for this project's earlier scores.
DATASETS = {
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76"),
    "piqa": ("baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1"),
    "arc_easy": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"),
    "arc_challenge": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"),
}


class MiniPLEHarness(TemplateLM):
    def __init__(self, checkpoint: Path, data_dir: Path, batch_size: int):
        super().__init__()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer, _ = load_fixed_nanochat_tokenizer(data_dir)
        self.bos = self.tokenizer.token_to_id("<|bos|>")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = MiniPLEConfig.from_checkpoint(state["model_config"])
        cfg.attn_backend = "flash" if self.device.type == "cuda" else "sdpa"
        cfg.gradient_checkpointing = False
        self.model = MiniPLE(cfg)
        self.model.load_state_dict(state["model"], strict=True)
        self.step = int(state["step"]) + 1
        self.max_length = cfg.max_seq_len
        del state
        self.model.to(self.device).eval()
        self.truncated = 0

    @property
    def eot_token_id(self):
        return self.bos

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        ids = self.tokenizer.encode(string, add_special_tokens=False).ids
        if add_special_tokens is not False and (not ids or ids[0] != self.bos):
            ids = [self.bos] + ids
        return ids

    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        encoded = []
        for index, (_, context, continuation) in enumerate(requests):
            if not context or not continuation or len(continuation) > self.max_length:
                raise ValueError(f"Each choice needs context and a continuation within {self.max_length} tokens")
            tokens = context + continuation
            if len(tokens) > self.max_length + 1:
                self.truncated += 1
            tokens = tokens[-(self.max_length + 1):]
            encoded.append((index, tokens[:-1], continuation))
        encoded.sort(key=lambda row: len(row[1]), reverse=True)
        results = [None] * len(encoded)
        for start in tqdm(range(0, len(encoded), self.batch_size),
                          desc="Scoring choices", disable=disable_tqdm):
            batch = encoded[start:start + self.batch_size]
            inputs = torch.full((len(batch), max(len(row[1]) for row in batch)),
                                self.bos, dtype=torch.long, device=self.device)
            for row, (_, ids, _) in enumerate(batch):
                inputs[row, :len(ids)] = torch.tensor(ids, device=self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                hidden = self.model._decoder(inputs)
                selected = torch.cat([
                    hidden[row, len(ids) - len(cont):len(ids)]
                    for row, (_, ids, cont) in enumerate(batch)
                ])
                targets = torch.tensor([token for _, _, cont in batch for token in cont],
                                       device=self.device)
                logps, matches = [], []
                for offset in range(0, len(targets), 1024):
                    logits = F.linear(selected[offset:offset + 1024],
                                      self.model.token_embedding.weight).float()
                    target = targets[offset:offset + 1024]
                    logps.append(-F.cross_entropy(logits, target, reduction="none"))
                    matches.append(logits.argmax(-1).eq(target))
            logps = torch.cat(logps).cpu()
            matches = torch.cat(matches).cpu()
            offset = 0
            for index, _, continuation in batch:
                end = offset + len(continuation)
                results[index] = (float(logps[offset:end].sum()),
                                  bool(matches[offset:end].all()))
                self.cache_hook.add_partial("loglikelihood", requests[index][0], results[index])
                offset = end
        return results

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError("Only multiple-choice likelihood tasks are supported")

    def generate_until(self, requests, **kwargs):
        raise NotImplementedError("Only multiple-choice likelihood tasks are supported")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/evaluation.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Quick check only; omit for reported scores")
    parser.add_argument("--tasks", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    torch.set_float32_matmul_precision("high")
    model = MiniPLEHarness(args.checkpoint, args.data_dir, args.batch_size)
    manager = TaskManager()
    configs = []
    for task in args.tasks:
        config = load_yaml(manager.task_index[task].yaml_path, resolve_func=True)
        config["dataset_kwargs"] = {"revision": DATASETS[task][1]}
        configs.append(config)
    result = simple_evaluate(
        model=model, tasks=configs, task_manager=manager, num_fewshot=0,
        batch_size=args.batch_size, limit=args.limit,
        random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "step": model.step,
        "zero_shot": True,
        "limit": args.limit,
        "dataset_revisions": {DATASETS[t][0]: DATASETS[t][1] for t in args.tasks},
        "truncated_requests": model.truncated,
        "results": result["results"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
