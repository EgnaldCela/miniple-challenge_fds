"""Zero-value attention sinks for FlashAttention-2 and portable PyTorch SDPA.

For each head, exp(sink_logit) contributes only to the softmax denominator.
The FlashAttention adapter augments both the forward and backward normalizer;
rescaling its public output with a detached log-sum-exp would miss Q/K gradients.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

try:
    # Internal operators from the pinned FlashAttention 2.8.3.post1 installation.
    from flash_attn.flash_attn_interface import (
        _wrapped_flash_attn_forward,
        _wrapped_flash_attn_backward,
        flash_attn_with_kvcache,
    )
except ImportError:
    _wrapped_flash_attn_forward = _wrapped_flash_attn_backward = None
    flash_attn_with_kvcache = None

FLASH_ATTN_AVAILABLE = _wrapped_flash_attn_forward is not None


class _FlashAttentionWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sink_logits, dropout_p, window_left):
        head_dim = q.shape[-1]
        padding = (-head_dim) % 8
        if padding:
            q, k, v = (F.pad(x, (0, padding)) for x in (q, k, v))
        scale = head_dim ** -0.5
        out, lse, _, rng_state = _wrapped_flash_attn_forward(
            q, k, v, dropout_p, scale, True, window_left, 0, 0.0, None, False,
        )
        # LSE is [batch, query_heads, time]; the sink is independent of query.
        total_lse = torch.logaddexp(lse, sink_logits.float()[None, :, None])
        mass = torch.exp(lse - total_lse).transpose(1, 2).unsqueeze(-1)
        out = (out.float() * mass).to(q.dtype)
        ctx.save_for_backward(q, k, v, out, total_lse, sink_logits, rng_state)
        ctx.dropout_p, ctx.scale, ctx.window_left = dropout_p, scale, window_left
        return out[..., :head_dim]

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, total_lse, sink_logits, rng_state = ctx.saved_tensors
        head_dim = grad_out.shape[-1]
        padding = q.shape[-1] - head_dim
        if padding:
            grad_out = F.pad(grad_out, (0, padding))
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        # Using the sink-normalized output AND LSE yields the full softmax
        # Jacobian for real tokens, including its Q/K normalization derivative.
        _wrapped_flash_attn_backward(
            grad_out, q, k, v, out, total_lse, dq, dk, dv,
            ctx.dropout_p, ctx.scale, True, ctx.window_left, 0, 0.0, None, False,
            rng_state=rng_state,
        )
        sink_prob = torch.exp(sink_logits.float()[None, :, None] - total_lse)
        output_grad = (grad_out.float() * out.float()).sum(-1).transpose(1, 2)
        dsink = -(sink_prob * output_grad).sum((0, 2)).to(sink_logits.dtype)
        return dq[..., :head_dim], dk[..., :head_dim], dv[..., :head_dim], dsink, None, None


def flash_attention_with_sink(q, k, v, sink_logits, dropout_p=0.0, window_left=-1):
    """Causal GQA with [B, T, H, D] inputs and no quadratic attention allocation."""
    if _wrapped_flash_attn_forward is None:
        raise RuntimeError("Attention sinks require the pinned FlashAttention installation; run scripts/install.sh")
    return _FlashAttentionWithSink.apply(q, k, v, sink_logits, dropout_p, window_left)


@torch.library.custom_op("miniple::flash_attention_cached", mutates_args=("key_cache", "value_cache"))
def _flash_attention_cached(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor, lengths: torch.Tensor,
    rotary_cos: Optional[torch.Tensor], rotary_sin: Optional[torch.Tensor],
    window_left: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if flash_attn_with_kvcache is None:
        raise RuntimeError("Cached decoding requires FlashAttention; run scripts/install.sh")
    return flash_attn_with_kvcache(
        q, key_cache, value_cache, k, v,
        rotary_cos=rotary_cos, rotary_sin=rotary_sin, cache_seqlens=lengths,
        causal=True, window_size=(window_left, 0), rotary_interleaved=True,
        return_softmax_lse=True,
    )


@_flash_attention_cached.register_fake
def _flash_attention_cached_fake(q, k, v, key_cache, value_cache, lengths,
                                 rotary_cos, rotary_sin, window_left):
    lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]),
                      device=q.device, dtype=torch.float32)
    return torch.empty_like(q), lse


def cached_attention_with_sink(q, k, v, key_cache, value_cache, lengths,
                               sink_logits, window_left=-1, rotary_cos=None, rotary_sin=None):
    """Inference-only cache update, fused RoPE/attention, and sink normalization."""
    out, lse = _flash_attention_cached(
        q, k, v, key_cache, value_cache, lengths, rotary_cos, rotary_sin, window_left,
    )
    total_lse = torch.logaddexp(lse, sink_logits.float()[None, :, None])
    mass = torch.exp(lse - total_lse).transpose(1, 2).unsqueeze(-1)
    return (out.float() * mass).to(q.dtype)


def sdpa_attention_with_sink(q, k, v, sink_logits, mask=None, dropout_p=0.0):
    """Portable [B, H, T, D] attention with a zero-valued virtual sink entry."""
    queries, keys = q.shape[-2], k.shape[-2]
    if mask is None:
        mask = torch.ones((queries, keys), device=q.device, dtype=torch.bool).tril()
    bias = torch.zeros(mask.shape, device=q.device, dtype=sink_logits.dtype)
    bias = bias.masked_fill(~mask, float("-inf"))
    # A zero key has score zero; its additive bias is the unscaled sink logit.
    # This column is always visible, including outside the local token window.
    bias = torch.cat((
        bias.expand(q.shape[1], queries, keys),
        sink_logits[:, None, None].expand(-1, queries, 1),
    ), dim=-1)
    return F.scaled_dot_product_attention(
        q, F.pad(k, (0, 0, 0, 1)), F.pad(v, (0, 0, 0, 1)),
        attn_mask=bias, dropout_p=dropout_p,
    )
