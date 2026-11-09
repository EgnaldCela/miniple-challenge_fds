# Study guide and primary references

Start with the [assignment](../README.md) and [code guide](CODE_GUIDE.md). This guide prepares you to choose and defend an experiment; the assignment's comparison rules determine whether results are comparable. Sources checked on **2026-11-07**. Pin the exact code and dataset revisions used in your work because online repositories change.

## 1. Understand the baseline before changing it

Read the local model, training loop, data loader, sampled-softmax implementation, and evaluator in the order given in the code guide. Draw the path from a raw document to token IDs, predictions, loss, gradients, and an optimizer update.

[nanoGPT](https://github.com/karpathy/nanoGPT) is useful background for a compact decoder-only model and training loop. Its README now marks it as deprecated; use it to study the mechanics, while this assignment's supplied code remains your starting point.

Before coding, explain:

- Why is the target shifted by one token, and what information can a causal prediction use?
- How do microbatch size, sequence length, and gradient accumulation determine tokens per update?
- Why can sampled-softmax training loss differ from full-vocabulary validation loss?
- Which tensors share parameters, and how do you count shared weights only once?
- What distinguishes restarting from weights from resuming optimizer, scheduler, random-number, and data-loader state?

First milestone: run the supplied smoke check, inspect a batch and its shifted targets, and identify the exact checkpoint and log files produced. Then reproduce the baseline before interpreting an architectural improvement.

## 2. Architecture, optimization, and speed on a 3090

The [Modded-NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) is a source of hypotheses about optimizers, initialization, attention, scheduling, and implementation efficiency. Its main benchmark uses **eight H100 GPUs**, a particular FineWeb validation target, and a changing recipe. Published times do not predict performance on this assignment's single RTX 3090.

Study one recorded change and its code rather than copying an entire current recipe. The current README includes a hashed bigram/trigram table described as having **65 billion parameters**, which is outside this assignment's comparable-size setting. Hardware-specific precision and kernels also need separate compatibility checks. [Source](https://github.com/KellerJordan/modded-nanogpt)

Before coding, explain:

- Is your hypothesis about learning per token, execution time per update, or both?
- Which parameter, memory, and compute costs does the change add or remove?
- What part of a measured run includes compilation, evaluation, and checkpoint writes?
- Can you first measure the affected operation and then confirm the end-to-end benefit?

First experiment: isolate one change, retain the common evaluation protocol, and plot loss against both consumed tokens and elapsed time. A faster kernel alone does not establish faster time to the target loss.

## 3. Multi-token prediction: alignment and gradients matter

Read [Multi-Token Prediction Done Right, v1](https://arxiv.org/abs/2610.38331v1) and its [official Soft-MTP implementation](https://github.com/JaydenTeoh/Soft-MTP). Soft-MTP uses straight-through gradient routing to recover a self-distillation signal from aligned next-token and multi-token cross-entropy computations sharing a linear output projection.

Two details matter for this assignment. First, Soft-MTP adds no parameters **relative to the corresponding hard-MTP model**; MTP heads still add parameters relative to a next-token-only model. The official example grows from 167M to 178M parameters with three MTP heads. Second, the repository scales the routed gradient for loss weights and the number of supervised positions, and provides gradient-equivalence tests. [Implementation and checks](https://github.com/JaydenTeoh/Soft-MTP)

Before coding, explain:

- Which future token does each head predict, and which tail or masked positions must be excluded?
- How do you prevent future inputs from leaking into the ordinary next-token prediction?
- Does your reduction preserve the intended auxiliary-loss weighting when valid lengths differ?
- Have you included every auxiliary head in the training parameter count?

Start with a small full-softmax correctness experiment comparing next-token-only, hard MTP, and Soft-MTP. Our sampled-softmax integration is not established by the paper: matched candidate support, sampling corrections, alignment, and gradients would need explicit verification. Check the intended gradient identity against a tiny explicit KL calculation before a long run.

Use ordinary full-vocabulary **next-token** validation cross-entropy for comparison. Log auxiliary losses separately. Count consumed input-token exposures once; report the number of auxiliary prediction targets separately rather than treating them as fewer consumed tokens.

## 4. Mixture of experts under a total parameter budget

[Switch Transformers](https://arxiv.org/abs/2101.03961) explains routing tokens to selected experts and the challenges of balance, stability, and sparse computation. Its large-model results motivate investigation; they do not guarantee that small experts will run efficiently on one 3090.

Before coding, explain:

- How will you keep **all experts, routers, embeddings, and other weights** inside the assignment's total parameter allowance?
- How many parameters are active per token, and what routing or dispatch overhead remains?
- How will you detect overloaded experts, collapsed routing, or discarded tokens?
- Will a dense control of comparable total size test the hypothesis fairly?

First experiment: replace a small number of feed-forward blocks, report expert utilization and memory, and measure actual throughput. Keep routing auxiliary losses separate from the common next-token loss.

## 5. N-grams without future leakage or hidden capacity

[N-Grammer](https://arxiv.org/abs/2207.06366) augments Transformers with n-grams constructed from discrete latent representations. It offers a more specific starting point than adding an unrestricted lookup table.

Before coding, explain:

- Which past tokens or causal hidden states form each n-gram?
- How are sequence starts and document boundaries handled?
- How many trainable table entries exist, including entries rarely accessed?
- What do collisions, table width, and lookup bandwidth trade off?

First experiment: use a small causal bigram feature and a matched parameter budget. A test that changes only future input tokens should leave earlier predictions unchanged. Include the complete table in the total parameter count, even when each token touches only a few rows.

## 6. Data mixture and data quality

Read [SmolLM2: Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737) for small-model data curation and [RefinedWeb](https://arxiv.org/abs/2306.01116) for filtering and deduplication. These motivate controlled data experiments; neither supplies a universally optimal mixture for MiniPLE.

Before coding, explain:

- What distribution should the finished model serve, and does your evaluation represent it?
- Are mixing weights defined by documents, bytes, or consumed tokens?
- How will you keep validation documents and benchmark answers out of training?
- What are the actual source revisions, filtering steps, repetition rates, and consumed-token counts?

First experiment: change one mixture proportion while holding architecture, token budget, and evaluation fixed. Use a common validation set and separately report losses by source. A lower training loss on an easier mixture is not evidence of better generalization.

## 7. Instruction tuning from the supplied base model

Study chat serialization, assistant-only masks, packing, and EOS handling in the [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer). [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) is a candidate English conversation dataset designed for small models. Inspect and pin the selected data rather than assuming every example suits this model.

TRL is a conceptual and implementation reference: the custom MiniPLE checkpoint is not automatically a compatible Hugging Face model. You must adapt the supplied training path or provide a correct integration.

Before coding, explain:

- Which tokens receive loss, including assistant delimiters and end-of-turn markers?
- Do training and generation use exactly the same conversation format?
- How will you detect truncation that removes the answer or includes only part of a conversation?
- How will you compare the base and tuned model without cherry-picking prompts?

First experiment: inspect token IDs, labels, and masks for a few conversations, overfit a tiny diagnostic batch, then train on a disjoint subset. Use [IFEval's verifiable instruction approach](https://arxiv.org/abs/2311.07911) and [official evaluation code](https://github.com/google-research/google-research/tree/master/instruction_following_eval) where appropriate, alongside fixed qualitative prompts and retention of base-model capabilities. A lower SFT loss alone does not demonstrate instruction following.

## 8. An Italian model: scratch training or adaptation

[ItaEval and TweetyIta](https://aclanthology.org/2024.clicit-1.6/) studies Italian evaluation and tokenization-based adaptation; its [official suite](https://github.com/RiTA-nlp/ita-eval) is a starting point for evaluation. [Evalita-LLM](https://arxiv.org/abs/2502.02289) emphasizes native Italian tasks and multiple prompts. [ITA-Bench](https://github.com/sapienzanlp/ita-bench) supplies another collection of Italian tasks.

For data exploration, [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) provides the **`ita_Latn`** configuration. Select a bounded, pinned subset; the complete language subset is far larger than a small student experiment requires. For adaptation, study learning-rate rewarming, decay, and replay in [Simple and Scalable Strategies to Continually Pre-train LLMs](https://arxiv.org/abs/2403.08763).

Before coding, explain:

- Are you training from scratch or adapting the supplied English checkpoint? What is the appropriate control?
- How efficiently does the existing tokenizer represent Italian, and is replacing it worth losing direct token-level comparability?
- How will you measure Italian improvement and English forgetting separately?
- Are evaluation tasks native Italian or translated, and are their prompts appropriate for a base or instruction model?

First experiment: freeze document-disjoint Italian validation data and a small downstream evaluation suite before selecting the training mixture. Keep the original tokenizer for the simplest adaptation baseline. If you change tokenization, compare identical raw held-out documents using a consistently implemented byte-normalized likelihood and downstream tasks; raw per-token perplexities across different tokenizers are not comparable.

## 9. Turn reading into an experiment

Write one paragraph stating the hypothesis, the control, the changed variable, the expected benefit, and a result that would refute the hypothesis. Record the source and revision of any borrowed method. Complete a short correctness check before allocating a full training run, and preserve unsuccessful experiments in the report: a well-supported negative finding is useful research.
