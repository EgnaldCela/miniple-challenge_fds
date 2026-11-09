# Read the implementation before modifying it

## The learning problem

For a packed row of tokens `x[0:T]`, the target row is the same stream shifted by one position. A causal model must predict `y[t]` using only the available prefix through `x[t]`. The reference objective is mean next-token negative log probability. Validation computes that objective over the entire vocabulary. Perplexity is the exponential of the mean loss, not a separate objective.

Start by tracing one microbatch through the loader, model, loss, backward pass, optimizer, and metric counter. Then trace one complete optimizer update: four microbatches accumulate gradients before clipping and stepping. Check which tensors are in BF16 and which statistics/parameters remain FP32.

## File map

| File | What to study or modify |
|---|---|
| `scripts/train.sh` | Exact executable recipe, environment overrides, GPU waiting, output lock, resume invocation |
| `train.py` | Argument defaults; memmap loader; deterministic mixtures; WSD LR; parameter groups; validation; counters; checkpoints |
| `model.py` | Configuration, PLE, causal local/global attention, sinks integration, FFN, tied embeddings, loss paths |
| `attention_sinks.py` | Sink normalization/gradients for FlashAttention and portable attention |
| `muon.py` | Matrix update, momentum, orthogonalization, AdamW groups, state restoration |
| `sampled_softmax.py` | Candidate construction, target mapping, candidate schedule, RNG state |
| `prepare_data.py` | Pinned sources, document splits, tokenization, packing, file sizes/metadata |
| `data_progress.py` | Committed token prefixes and progressive-preparation checks |
| `tokenizer_utils.py` | Frozen vocabulary/revision, byte-level round trip, delimiter |
| `loss_plots.py` | Metric reading and separate plots for candidate regimes/full-vocabulary validation |
| `scripts/evaluate_loss.py` | Frozen loss scorer with tokenizer and validation hashes |
| `scripts/evaluate.py` | Fixed zero-shot multiple-choice likelihood scoring |
| `inference.py`, `scripts/generate.py` | Architecture-specific KV cache and CUDA generation |

## How changes propagate

An architecture change usually reaches configuration serialization, forward/loss paths, parameter grouping, checkpoint loading, benchmark scoring, and generation. The current benchmark harness calls the decoder and tied output embedding directly, so replacing the head requires an adapter. The cached generation implementation also assumes this architecture. Keep configuration versions explicit; incompatible checkpoints should fail clearly.

The trainer checks vocabulary/context, prepared-file sizes, tokenizer hash, mixture declarations, batch/schedule compatibility, and resumed optimizer/data fingerprints. These are reproducibility checks. For a new experiment, make a coherent new configuration and dataset manifest; do not delete checks until an invalid combination runs.

`DecayMixtureLoader` chooses a source per physical microbatch. A new microbatch/accumulation geometry can therefore change source grouping within optimizer updates even at the same effective batch. Preserve fixed logical batches if the experiment is meant to isolate kernel/batching speed, or report this as an ordering change.

The current validation helper calls `model(x, y)` without sampled candidates. If you add MTP or routing penalties, provide an explicit ordinary-next-token evaluation path. Otherwise the frozen scorer could accidentally measure a different objective. Write a small test comparing your reported CE to a direct next-token calculation.

## Checks that matter

| Change | A small meaningful check before a long run |
|---|---|
| Attention, n-grams, packing | Change future tokens and verify earlier logits do not change; check target shift and document boundaries |
| Fused loss/kernel/precision | Compare loss and gradients with a small unfused reference; use justified tolerances |
| MTP | Verify head/target offsets, masks, loss weights and gradient routing; compare explicit KL and recovered gradients where claimed |
| MoE | Verify routing, expert coverage, no unintended dropped targets, balance statistics and all-parameter accounting |
| Accumulation or batch size | Compare a small full-batch update with equivalent accumulated microbatches |
| Resume | Compare uninterrupted steps with save/reload/continue, including data and RNG state |
| SFT | Inspect tokenized role boundaries and masks; hand-check which labels contribute loss and EOS/end-of-turn behavior |
| Italian tokenizer | Test text round trips, token coverage, embedding mapping and fair likelihood normalization |

Use tiny inputs and small models for correctness. A test that only repeats the implementation's own formula is weak evidence. After correctness, profile representative shapes on the allocated 3090. A change that helps a tiny CPU test may hurt the real GPU workload.

## A practical experiment loop

1. Read the relevant code and one primary reference; state a falsifiable hypothesis.
2. Save the baseline source/config and choose a fresh output directory.
3. Make the smallest change that tests the idea; record all parameters and counters.
4. Run a correctness check, then a short explicitly labelled pilot.
5. Inspect loss, gradients, throughput, memory, and data accounting. Stop a clearly broken run early.
6. Run the budgeted controlled experiment, keep its raw outputs, and evaluate the saved model.
7. Decide whether the evidence supports the hypothesis; report limitations and the next discriminating experiment.

The provided starter implements pretraining. SFT, Italian adaptation, MoEs, n-gram features, and MTP are student research implementations, not hidden flags already supplied in the code.
