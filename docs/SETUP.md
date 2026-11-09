# Setup and first experiment

All commands below run from the repository root, the directory containing `train.py` and `scripts/`. The locally prepared package is named `student_assignment`; a Git clone can have a different directory name. Use your allocated GPU number; examples use GPU 0. Training and generation require the course Linux/NVIDIA environment. CPU checks can run on a laptop, but CPU timing is not a 3090 result.

## 1. Get the source and inspect it before allocating GPU time

See [distribution and downloads](DISTRIBUTION.md) for the configured GitHub repository, release, publication status, and access instructions. The repository includes the frozen validation file and tokenizer; it does not include the pretrained checkpoint files or training corpus.

```bash
python3 scripts/check_starter.py
python3 scripts/verify_artifacts.py --checkpoints none
```

This standard-library check validates syntax and 3B defaults and exercises the launcher with inert stand-ins. It neither downloads data nor starts training. Read `AGENTS.md`, [the track briefs](TRACKS.md), and [the protocol](PROTOCOL.md).

**No pretrained checkpoint is needed for the default scratch run.** Each optional reference checkpoint is about 1.4 GB; obtain only the one needed for reference evaluation or adaptation. Allow roughly 6 GB for the prepared default training corpus, additional space for dataset/download caches and the environment, and several roughly 1.4 GB output checkpoints per experiment. The frozen validation/tokenizer reference is included in the source checkout. **Training data is not included:** students download and tokenize it with the supplied preparation code.

## 2. Install on Linux x86_64 with Python 3.12

```bash
bash scripts/install.sh
source .venv/bin/activate
python -m pip freeze > environment.txt
nvidia-smi
```

The installer uses PyTorch 2.8.0/CUDA 12.8, FlashAttention 2.8.3.post1, and Liger 0.8.4. A working NVIDIA driver is required. The FlashAttention wheel is specific to the documented Python/PyTorch platform. If it is unavailable, use a compatible CUDA 12.8 toolkit and compiler to build the same version:

```bash
python -m pip install packaging ninja psutil einops
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 NVCC_THREADS=2 \
  python -m pip install --no-build-isolation --no-binary=flash-attn flash-attn==2.8.3.post1
```

For CPU-only development, use a compatible PyTorch installation and `python -m pip install -r requirements.txt`. CPU fallback supports small correctness checks; the production launcher still requires CUDA. Existing custom MiniPLE weights are not directly loadable by Transformers or TRL without an adapter/conversion.

## 3. Select and verify reference assets

For scratch training, verify the bundled validation/tokenizer without downloading model weights:

```bash
python3 scripts/verify_artifacts.py --checkpoints none
python -m unittest discover -s tests -v
```

The artifact verifier **always** checks the frozen validation/tokenizer reference. It checks only the checkpoint selection requested with `--checkpoints none|3b|50b|all`; the default is `none`. Missing unselected checkpoints are expected. These checks do not require training data or use a GPU. The unit tests exercise frozen loss evaluation on a tiny CPU model, including batch-size invariance and rejection of modified data. They do not run the full pretrained model.

For 3B reference evaluation, once the public release is available:

```bash
python3 scripts/download_checkpoints.py --checkpoint 3b
python3 scripts/verify_artifacts.py --checkpoints 3b
```

For E/F adaptation from the 50B-token model:

```bash
python3 scripts/download_checkpoints.py --checkpoint 50b
python3 scripts/verify_artifacts.py --checkpoints 50b
```

Use `--checkpoint all` and `--checkpoints all` only when both references are needed. Downloads use `distribution.json` and write to `baselines/3b/final_l3_mix.pt` or `baselines/50b/final_l3_mix.pt`. Both are **full training checkpoints**, preserving their original optimizer/RNG state and hashes. They have not been converted into weights-only exports. See [distribution](DISTRIBUTION.md) for repository/tag overrides and private-release access.

## 4. Reconstruct training data

Run preparation separately so you can inspect the result before allocating a long training job:

```bash
python prepare_data.py --out-dir data/ultrafineweb_3b \
  --tokenizer-json baselines/validation/tokenizer.json \
  --validation-reference baselines/validation/val_l2.bin
bash scripts/debug.sh
```

Preparation uses CPU, disk, and network access; it does not train or require a GPU. It streams the pinned L2/L3 source revisions, tokenizes documents, writes document-disjoint packed files and metadata, and checks that regenerated validation exactly matches the frozen reference. It can take substantial time. Preserve the resulting files and metadata for both control and candidate. The debug command validates prepared-file sizes, configuration, and CUDA availability.

The default `scripts/train.sh` also performs this preparation automatically if the selected `DATA_DIR` has no metadata, using the bundled tokenizer/reference when present. Unlike the separate preparation command, it proceeds to training afterward.

The fresh training corpus is **not byte-identical** to the historical corpus because the historical run reused specific older slices. Run an unchanged baseline on your reconstructed corpus before attributing gains to architecture or optimization. Historical plots/scores remain useful references; candidate and control must use the same reconstructed bytes. The frozen validation protocol stays unchanged. Use a new `--out-dir` for a changed preparation; existing metadata is protected against overwrite.

## 5. Train the reference from scratch

```bash
GPU_ID=0 OUT_DIR=runs/my_baseline_3b bash scripts/train.sh
```

Defaults: random initialization, seed 1337, 20 sequences per microbatch, four accumulation steps, 2048-token context, Muon plus AdamW, BF16, FlashAttention, Liger, compiled execution, and the 3B schedule. The historical run took about 11 hours after setup; hardware load, compilation, and storage affect this. First-use compilation takes time. Use a persistent session such as the course job scheduler or `tmux` for a long run.

The launcher refuses to overwrite an existing run and waits if the selected GPU has a compute process. It does not reserve the GPU against another simultaneous launcher; follow the allocation system. An output-directory lock only protects that one directory.

Inspect and plot a run:

```bash
tail -n 20 runs/my_baseline_3b/train.log
python loss_plots.py runs/my_baseline_3b
# Optional, ongoing refresh in another terminal:
python loss_plots.py --watch runs/my_baseline_3b
```

Outputs include `metrics.jsonl`, `status`, `train.log`, `latest.pt`, milestone checkpoints, and `final_l3_mix.pt`. The plotter writes `losses.png` and `losses.svg`. Start/validation/completion events and token counters are in the JSONL log.

To change optimizer, architecture, batch geometry, or validation behavior, edit the corresponding student copy and record the diff. `scripts/train.sh` accepts only `--resume` as a positional option; arbitrary `train.py` flags are not forwarded as a general experiment interface. Read its actual command before modifying it. `STAGE1_TOKENS`, `DECAY_TOKENS`, `MIX_START_TOKENS`, and `WARMUP_TOKENS` are explicit environment overrides, but prepared metadata and complete mixture cycles must match.

## 6. Resume your own interrupted run

```bash
GPU_ID=0 OUT_DIR=runs/my_baseline_3b bash scripts/train.sh \
  --resume runs/my_baseline_3b/latest.pt
```

Keep the original data, model, schedule, optimizer settings, output directory, and batch geometry. The loader/RNG/optimizer state is restored. The checkpoint can lag behind the log: preserve the original log, reconcile records after the saved step, and account for replayed compute before making plots or timing claims. The starter does not automatically repair duplicate log segments after a crash.

Do not use `--resume` on a completed reference checkpoint to start SFT, change the dataset, or extend a zero-LR schedule. The reconstructed data differs from the historical data and has a different metadata fingerprint. E/F must load model weights into a deliberate new training stage with a fresh schedule and optimizer.

## 7. Evaluate a checkpoint

Download and verify the selected reference as described in section 3, or use your own saved checkpoint. Frozen validation loss (batch 20 matches the original monitoring geometry; use a smaller batch if required and record it):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_loss.py \
  --checkpoint baselines/3b/final_l3_mix.pt \
  --data-dir baselines/validation --batch-size 20 \
  --out runs/reference_3b_validation.json
```

Use `baselines/50b/final_l3_mix.pt` for the other reference, or your own checkpoint. The evaluator defaults to `baselines/validation`, so it works before training data is prepared. It reads the model configuration and weights, validates frozen inputs, and reports the checkpoint hash and loss. For changed architectures, implement a compatible next-token evaluation path and verify it against this reference protocol. Its reported `training_tokens = completed_steps × tokens_per_step` assumes the original constant-batch format; use your explicit exposure counter for variable-batch or resumed/replayed experiments.

Full benchmark suite:

```bash
python -m pip install -r requirements-eval.txt
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --checkpoint runs/my_baseline_3b/final_l3_mix.pt \
  --data-dir baselines/validation --batch-size 8 \
  --out runs/my_baseline_3b/evaluation.json
```

`--limit 2` is useful for a functional pilot only. Omit it for reported scores. This evaluator scores multiple-choice likelihoods; it does not implement IFEval, chat evaluation, or arbitrary Italian tasks. Extend it or add a separate evaluator for those tracks without changing the reference scoring.

## 8. Generate a base-model continuation

This example requires the optional 50B-token checkpoint; download and verify it using section 3 first.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 \
  python scripts/generate.py \
  --checkpoint baselines/50b/final_l3_mix.pt \
  --data-dir baselines/validation \
  --prompt "The most important discovery was" --max-new-tokens 128 \
  --temperature 1.0 --top-p 0.95 --top-k 20
```

The supplied models are base language models, not chat assistants. The generator uses a CUDA KV-cache runtime specific to the existing architecture; architecture changes may require matching inference changes. Prompt plus generation must fit the 2048-token context. Use fixed prompts, sampling settings, and seeds for comparisons.

Only load instructor-provided or self-produced `.pt` files: these full training checkpoints contain Python optimizer/RNG state and are loaded with pickle support.
