"""BF16 generation with FlashAttention KV caches and compiled CUDA-graph decoding."""
from __future__ import annotations

import torch

from attention_sinks import FLASH_ATTN_AVAILABLE
from model import MiniPLE


class KVGenerator:
    """Generate equal-length batches on one CUDA GPU without prefix recomputation.

    The caller supplies an eval-mode BF16/FP16 model. A generator owns its cache
    and can be reused for batches of the same size and sampling settings.
    """

    def __init__(self, model: MiniPLE, batch_size: int = 1, *,
                 temperature: float = 1.0, top_k: int = 20, top_p: float = 0.95,
                 compile_decode: bool = True, cuda_graph: bool = True,
                 prefill_chunk_size: int = 512):
        weight = model.token_embedding.weight
        if not FLASH_ATTN_AVAILABLE or not weight.is_cuda:
            raise ValueError("KVGenerator requires CUDA and the pinned FlashAttention installation")
        if weight.dtype not in {torch.bfloat16, torch.float16} or model.training:
            raise ValueError("Use model.to(device='cuda', dtype=torch.bfloat16).eval()")
        if (batch_size < 1 or prefill_chunk_size < 1 or temperature < 0
                or not 0 <= top_k <= model.cfg.vocab_size or not 0 < top_p <= 1):
            raise ValueError("Invalid batch_size, prefill_chunk_size, temperature, top_k, or top_p")
        if model.cfg.head_dim % 16:
            raise ValueError("Cached FlashAttention RoPE requires head_dim divisible by 16")
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.prefill_chunk_size = min(prefill_chunk_size, model.cfg.max_seq_len)
        self.compile_decode = compile_decode
        self.use_cuda_graph = cuda_graph
        self.lengths = torch.zeros(batch_size, device=weight.device, dtype=torch.int32)
        shape = (batch_size, model.cfg.max_seq_len, model.cfg.n_kv_heads, model.cfg.head_dim)
        self.cache = [(torch.empty(shape, device=weight.device, dtype=weight.dtype),
                       torch.empty(shape, device=weight.device, dtype=weight.dtype))
                      for _ in model.blocks]
        self.token = torch.zeros((batch_size, 1), device=weight.device, dtype=torch.long)
        self.decode = (torch.compile(model.forward_cached, fullgraph=True, dynamic=False,
                                     mode="max-autotune-no-cudagraphs")
                       if compile_decode else model.forward_cached)
        self.graph = None
        self.prepared = False

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits.float() / self.temperature
        indices = None
        if self.top_k:
            logits, indices = logits.topk(self.top_k)
        elif self.top_p < 1:
            logits, indices = logits.sort(dim=-1, descending=True)
        probs = logits.softmax(dim=-1)
        if self.top_p < 1:
            remove = probs.cumsum(dim=-1) > self.top_p
            # Retain the first token that crosses the nucleus threshold.
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            probs = probs.masked_fill(remove, 0)
        choice = torch.multinomial(probs, 1)
        return choice if indices is None else indices.gather(-1, choice)

    @torch.inference_mode()
    def _decode_step(self):
        logits = self.decode(self.token, self.cache, self.lengths)
        self.token.copy_(self._sample(logits))

    @torch.inference_mode()
    def prepare(self):
        """Compile and capture once, before timing generation or serving requests."""
        if self.prepared:
            return
        # Warm up on a side stream, as required by CUDA graph capture. Lengths
        # live on the GPU, so replay can advance the cache without host reads.
        stream = torch.cuda.Stream(device=self.token.device)
        stream.wait_stream(torch.cuda.current_stream(self.token.device))
        with torch.cuda.stream(stream):
            if self.compile_decode:
                prompt = torch.zeros((self.batch_size, self.prefill_chunk_size),
                                     device=self.token.device, dtype=torch.long)
                self.lengths.zero_()
                self.decode(prompt, self.cache, self.lengths)
            for _ in range(3):
                self.lengths.zero_()
                self._decode_step()
        torch.cuda.current_stream(self.token.device).wait_stream(stream)
        if self.use_cuda_graph:
            self.lengths.zero_()
            stream.wait_stream(torch.cuda.current_stream(self.token.device))
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self._decode_step()
            torch.cuda.current_stream(self.token.device).wait_stream(stream)
        self.lengths.zero_()
        self.prepared = True

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Return [batch, max_new_tokens] continuation IDs; stop/trim EOS in the caller.

        Prefill accepts one equally sized prompt per batch row. Padding and
        different prompt lengths are not supported by this minimal generator.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != self.batch_size or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [configured batch_size, nonempty prompt]")
        if input_ids.device != self.token.device or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long on the model's CUDA device")
        if max_new_tokens < 1 or input_ids.shape[1] + max_new_tokens > self.model.cfg.max_seq_len:
            raise ValueError("Prompt plus generated tokens must fit max_seq_len")
        self.prepare()
        self.lengths.zero_()
        # Bound temporary activations separately from the persistent KV cache.
        for offset in range(0, input_ids.shape[1], self.prefill_chunk_size):
            chunk = input_ids[:, offset:offset + self.prefill_chunk_size].contiguous()
            # Compile the fixed full-chunk shape once; arbitrary prompt tails
            # stay eager to avoid a new compilation for every prompt length.
            forward = (self.decode if chunk.shape[1] == self.prefill_chunk_size
                       else self.model.forward_cached)
            logits = forward(chunk, self.cache, self.lengths)
        self.token.copy_(self._sample(logits))
        output = torch.empty((self.batch_size, max_new_tokens),
                             device=self.token.device, dtype=torch.long)
        output[:, :1].copy_(self.token)
        for index in range(1, max_new_tokens):
            if self.graph is None:
                self._decode_step()
            else:
                self.graph.replay()
            output[:, index:index + 1].copy_(self.token)
        return output
