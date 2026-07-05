"""Vanilla decoder-only LLaMA-style Transformer LM, used as the efficiency
baseline against PHOTON."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_blocks import TransformerStack


@dataclass
class BaselineConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_hidden: int = 1536
    max_seq_len: int = 2048
    tie_embeddings: bool = True


class BaselineLM(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.stack = TransformerStack(
            n_layers=cfg.n_layers, dim=cfg.dim, n_heads=cfg.n_heads,
            mlp_hidden=cfg.mlp_hidden, max_seq_len=cfg.max_seq_len,
        )
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        x = self.tok_emb(idx)
        x = self.stack(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = None):
        """idx: (B, T) prompt tokens. Uses incremental KV cache for O(1) per-step decode."""
        kv_caches = self.stack.new_kv_caches()
        x = self.tok_emb(idx)
        h = x
        for i, layer in enumerate(self.stack.layers):
            h = layer(h, kv_cache=kv_caches[i])
        h = self.stack.norm_out(h)
        logits = self.lm_head(h[:, -1:, :])

        generated = idx
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=1)

            h = self.tok_emb(next_tok)
            for i, layer in enumerate(self.stack.layers):
                h = layer(h, kv_cache=kv_caches[i])
            h = self.stack.norm_out(h)
            logits = self.lm_head(h)
        return generated
