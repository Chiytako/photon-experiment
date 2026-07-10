"""Shared LLaMA-style transformer building blocks used by both the vanilla
baseline model and PHOTON's hierarchical encoder/decoder stacks.

Components: RMSNorm, rotary position embeddings, causal self-attention
(via scaled_dot_product_attention) with optional incremental KV cache,
SwiGLU feed-forward, and a residual pre-norm TransformerBlock/Stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


def precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0,
                     device=None, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               position_offset: int = 0) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim). cos/sin: (max_seq_len, head_dim)."""
    T = x.shape[-2]
    cos_t = cos[position_offset:position_offset + T].to(x.dtype)
    sin_t = sin[position_offset:position_offset + T].to(x.dtype)
    cos_t = cos_t.unsqueeze(0).unsqueeze(0)
    sin_t = sin_t.unsqueeze(0).unsqueeze(0)
    return x * cos_t + _rotate_half(x) * sin_t


class KVCache:
    """Growable KV cache for a single attention layer. Preallocates capacity
    and grows by doubling, writing new steps via slice assignment, so a
    T-step decode costs O(T) copies total instead of the O(T^2) of
    per-step torch.cat. Only used under no_grad (generation)."""

    def __init__(self, capacity_hint: int = 64):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self._len = 0
        self._capacity_hint = capacity_hint

    def _ensure(self, template: torch.Tensor, needed: int):
        B, H, _, D = template.shape
        if self.k is None:
            cap = max(needed, self._capacity_hint)
            self.k = template.new_empty(B, H, cap, D)
            self.v = template.new_empty(B, H, cap, D)
        elif needed > self.k.shape[2]:
            cap = max(needed, 2 * self.k.shape[2])
            for name in ("k", "v"):
                old = getattr(self, name)
                new = template.new_empty(B, H, cap, D)
                new[:, :, :self._len] = old[:, :, :self._len]
                setattr(self, name, new)

    def append(self, k: torch.Tensor, v: torch.Tensor):
        T = k.shape[2]
        self._ensure(k, self._len + T)
        self.k[:, :, self._len:self._len + T] = k
        self.v[:, :, self._len:self._len + T] = v
        self._len += T
        return self.k[:, :, :self._len], self.v[:, :, :self._len]

    @property
    def length(self) -> int:
        return self._len

    def logical_bytes(self) -> int:
        """Bytes of live K+V entries (logical length, not allocated capacity);
        used for the paper-protocol KV-memory accounting in benchmark.py."""
        if self.k is None:
            return 0
        B, H, _, D = self.k.shape
        return 2 * B * H * self._len * D * self.k.element_size()


class GradKVCache:
    """Concatenation-based KV cache, autograd-safe (no in-place writes).
    Used by PHOTON's differentiable within-chunk latent recursion during
    training; the O(len^2) copy growth of torch.cat is irrelevant at the
    bounded window sizes involved (<= R_l + C_l)."""

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def append(self, k: torch.Tensor, v: torch.Tensor):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


def sample_token(logits: torch.Tensor, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
    """Temperature + top-k sampling shared by BaselineLM and PhotonLM.
    logits: (N, vocab). Returns (N, 1) sampled token ids."""
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: Optional[int] = None,
                 rope_base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        inner_dim = self.n_heads * self.head_dim
        self.wq = nn.Linear(dim, inner_dim, bias=False)
        self.wk = nn.Linear(dim, inner_dim, bias=False)
        self.wv = nn.Linear(dim, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        cos, sin = precompute_rope(max_seq_len, self.head_dim, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _maybe_extend_rope(self, needed_len: int, device, dtype):
        if needed_len > self.max_seq_len:
            self.max_seq_len = needed_len
            cos, sin = precompute_rope(needed_len, self.head_dim, self.rope_base, device=device, dtype=dtype)
            self.rope_cos, self.rope_sin = cos, sin

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, D). If kv_cache is provided, does incremental decoding:
        x is the new chunk of tokens, cache holds past K/V, and full causal
        attention (over cache+new) is computed with position offset = cache length.
        If attn_mask is given (bool, True=keep) it overrides the default causal
        assumption (used with kv_cache=None for custom masks); when kv_cache is
        None and attn_mask is None, standard causal masking (is_causal=True) is used.
        """
        B, T, _ = x.shape
        position_offset = kv_cache.length if kv_cache is not None else 0
        self._maybe_extend_rope(position_offset + T, x.device, self.rope_cos.dtype)

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin, position_offset)
        k = apply_rope(k, self.rope_cos, self.rope_sin, position_offset)

        if kv_cache is not None:
            k, v = kv_cache.append(k, v)
            # new queries (T) attend causally over the full cached+new keys
            Tk = k.shape[2]
            if T == 1:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            else:
                mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device)
                mask = torch.tril(mask, diagonal=Tk - T)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif attn_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_hidden: int,
                 head_dim: Optional[int] = None, norm_eps: float = 1e-5,
                 rope_base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        self.attn_norm = RMSNorm(dim, norm_eps)
        self.attn = CausalSelfAttention(dim, n_heads, head_dim, rope_base, max_seq_len)
        self.mlp_norm = RMSNorm(dim, norm_eps)
        self.mlp = SwiGLUMLP(dim, mlp_hidden)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache, attn_mask=attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TransformerStack(nn.Module):
    """A stack of causal TransformerBlocks operating on a sequence of vectors."""

    def __init__(self, n_layers: int, dim: int, n_heads: int, mlp_hidden: int,
                 head_dim: Optional[int] = None, norm_eps: float = 1e-5,
                 rope_base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_hidden, head_dim, norm_eps, rope_base, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(dim, norm_eps)

    def forward(self, x: torch.Tensor, kv_caches: Optional[list] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, kv_cache=cache, attn_mask=attn_mask)
        return self.norm_out(x)

    def new_kv_caches(self) -> list:
        return [KVCache() for _ in self.layers]
