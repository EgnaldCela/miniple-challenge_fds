"""Single-device Muon + fused auxiliary AdamW, using Moonlight update scaling.

Algorithm references (implemented locally; no extra package required):
https://kellerjordan.github.io/posts/muon/
https://arxiv.org/abs/2502.16982
"""
from __future__ import annotations

import math

import torch
from torch import nn


def orthogonalize(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Quintic Newton-Schulz with independent normalization per matrix."""
    x = matrix.float()
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    if x.is_cuda:
        x = x.to(torch.bfloat16)
    for _ in range(steps):
        gram = x @ x.mT
        polynomial = -4.7750 * gram + 2.0315 * (gram @ gram)
        x = 3.4445 * x + polynomial @ x
    return x.mT if transpose else x


def update_bucket(weights, gradients, buffers, lr, beta, weight_decay, ns_steps):
    """One compiled region for momentum, batched NS, decay and weight updates.

    Lists contain disjoint logical matrix views of a single shape. Parameters
    and FP32 momentum remain in their original storage for v1 checkpoints.
    """
    torch._foreach_lerp_(buffers, gradients, 1.0 - beta)
    directions = torch.stack(torch._foreach_lerp(gradients, buffers, beta))
    updates = orthogonalize(directions, ns_steps)
    scale = 0.2 * math.sqrt(max(weights[0].shape))
    torch._foreach_mul_(weights, 1.0 - lr * weight_decay)
    deltas = updates.float() * (-lr * scale)
    torch._foreach_add_(weights, list(deltas.unbind(0)))
    return updates


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer interface for GradScaler, accumulation, and checkpointing."""

    def __init__(self, model: nn.Module, *, lr: float, muon_lr: float,
                 weight_decay: float, betas: tuple[float, float],
                 momentum: float = 0.95, ns_steps: int = 5,
                 compile_ns: bool = False):
        if lr <= 0 or muon_lr <= 0 or weight_decay < 0:
            raise ValueError("Learning rates must be positive and weight decay nonnegative")
        if not 0 <= momentum < 1 or ns_steps < 1:
            raise ValueError("Muon requires 0 <= momentum < 1 and ns_steps >= 1")
        excluded = {
            id(p) for module in model.modules()
            if isinstance(module, (nn.Embedding, nn.RMSNorm))
            for p in module.parameters(recurse=False)
        }
        hidden, auxiliary, names, aux_names, splits = [], [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dtype != torch.float32:
                raise ValueError("MuonAdamW requires FP32 master parameters; use autocast")
            if id(p) in excluded or p.ndim < 2:
                auxiliary.append(p)
                aux_names.append(name)
            else:
                if p.ndim != 2:
                    raise ValueError(f"Muon expects a hidden matrix: {name}")
                hidden.append(p)
                names.append(name)
                if name.endswith(".attn.qkv_proj.weight"):
                    q = model.cfg.n_heads * model.cfg.head_dim
                    kv = model.cfg.n_kv_heads * model.cfg.head_dim
                    sizes = [q, kv, kv]
                else:
                    sizes = [p.shape[0]]
                if sum(sizes) != p.shape[0]:
                    raise ValueError(f"Invalid Muon projection split: {name}")
                splits.append(sizes)
        assigned = [id(p) for p in hidden + auxiliary]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if len(assigned) != len(set(assigned)) or set(assigned) != expected:
            raise ValueError("Optimizer groups must cover every parameter exactly once")
        if not hidden or not auxiliary:
            raise ValueError("MuonAdamW requires hidden and auxiliary parameter groups")
        groups = [
            dict(params=hidden, param_names=names, splits=splits,
                 shapes=[list(p.shape) for p in hidden], lr=muon_lr,
                 initial_lr=muon_lr, weight_decay=weight_decay,
                 momentum=momentum, ns_steps=ns_steps, nesterov=True,
                 ns_coefficients=(3.4445, -4.7750, 2.0315), ns_eps=1e-7,
                 scaling="moonlight_0.2_sqrt_max", kind="muon"),
            dict(params=auxiliary, param_names=aux_names,
                 shapes=[list(p.shape) for p in auxiliary], lr=lr,
                 initial_lr=lr, weight_decay=0.0, betas=betas, eps=1e-8,
                 kind="adamw"),
        ]
        super().__init__(groups, {})
        self._init_adamw()
        self._update_bucket = (
            torch.compile(update_bucket, fullgraph=True, dynamic=False)
            if compile_ns else update_bucket
        )
        # Preserve each parameter's state layout; only the computation is batched.
        buckets = {}
        for p, sizes in zip(hidden, splits):
            offset = 0
            for rows in sizes:
                key = (p.device, rows, p.shape[1])
                buckets.setdefault(key, []).append((p, offset, rows))
                offset += rows
        self._buckets = list(buckets.values())
        self._lr = torch.zeros((), device=hidden[0].device, dtype=torch.float32)
        self.record_diagnostics = False
        self._diagnostics = {}

    def _init_adamw(self):
        # Share the auxiliary group dict so the outer LR schedule reaches AdamW.
        group = self.param_groups[1]
        group["fused"] = all(p.is_cuda for p in group["params"])
        self.adamw = torch.optim.AdamW(
            [group], fused=group["fused"]
        )
        self.param_groups[1] = self.adamw.param_groups[0]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        # A tensor LR changes in place without specializing compiled graphs to
        # each Python float during warmup and decay.
        self._lr.fill_(group["lr"])
        update_squares, weight_squares, count = [], [], 0
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise ValueError("Muon requires dense gradients")
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)
        for bucket in self._buckets:
            active = [(p, start, rows) for p, start, rows in bucket if p.grad is not None]
            if not active:
                continue
            weights = [p.narrow(0, start, rows) for p, start, rows in active]
            gradients = [p.grad.narrow(0, start, rows) for p, start, rows in active]
            buffers = [self.state[p]["momentum_buffer"].narrow(0, start, rows)
                       for p, start, rows in active]
            # Optional unused parameters must not create new compiled list-length
            # specializations or receive momentum/weight-decay updates.
            kernel = self._update_bucket if len(active) == len(bucket) else update_bucket
            updates = kernel(weights, gradients, buffers, self._lr,
                             group["momentum"], group["weight_decay"], group["ns_steps"])
            if self.record_diagnostics:
                scale = 0.2 * math.sqrt(max(weights[0].shape))
                update_squares.append(updates.float().square().sum() * scale**2)
                weight_squares.extend(w.square().sum() for w in weights)
                count += sum(w.numel() for w in weights)
        # The outer optimizer is unscaled/checked by GradScaler before this call.
        self.adamw.step()
        self._diagnostics = {}
        if count:
            self._diagnostics = {
                "muon_update_rms": (torch.stack(update_squares).sum() / count).sqrt(),
                "muon_weight_rms": (torch.stack(weight_squares).sum() / count).sqrt(),
            }
        return loss

    def diagnostics(self) -> dict[str, float]:
        # Called only at existing logging points, not every update.
        return {name: float(value) for name, value in self._diagnostics.items()}

    def state_dict(self):
        state = super().state_dict()
        state["muon_version"] = 1
        state["muon_execution"] = "batched_shape_v2"
        state["adamw"] = self.adamw.state_dict()
        return state

    def load_state_dict(self, state_dict):
        if state_dict.get("muon_version") != 1 or "adamw" not in state_dict:
            raise ValueError("Expected a MuonAdamW v1 checkpoint")
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("Muon checkpoint parameter-group mismatch")
        keys = ("kind", "param_names", "shapes", "splits", "initial_lr", "weight_decay",
                "momentum", "ns_steps", "nesterov", "scaling", "betas", "eps",
                "ns_coefficients", "ns_eps")
        for saved, current in zip(groups, self.param_groups):
            for key in keys:
                if saved.get(key) != current.get(key):
                    raise ValueError(f"Muon checkpoint mismatch for {key}")
        aux = state_dict["adamw"]
        if len(aux["param_groups"]) != 1:
            raise ValueError("Auxiliary AdamW checkpoint group mismatch")
        for key in keys + ("lr",):
            if aux["param_groups"][0].get(key) != groups[1].get(key):
                raise ValueError(f"Auxiliary AdamW checkpoint mismatch for {key}")
        super().load_state_dict({"state": state_dict["state"], "param_groups": groups})
        self._init_adamw()
        self.adamw.load_state_dict(aux)
        self.param_groups[1] = self.adamw.param_groups[0]
        self.param_groups[1]["fused"] = all(p.is_cuda for p in self.param_groups[1]["params"])
