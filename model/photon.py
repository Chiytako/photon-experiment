"""PHOTON: Parallel Hierarchical Operation for TOp-down Networks.

Faithful re-implementation of Ichikawa et al. (Fujitsu), arXiv:2512.20687
("PHOTON: Hierarchical Autoregressive Modeling for Lightspeed and
Memory-Efficient Language Generation"). No official code was released at the
time of writing; this follows the paper's equations (Sec. 2, Appendix A).

Architecture (L levels above the raw token sequence, default L=2):

Bottom-up encoder (per level l, causal; Sec. 2.1.1):
  A^l = ContextChunker_l(X^{l-1})   # concat C_l vectors -> linear -> 1 vector
  X^l = ContextEncoder_l(A^l)       # causal transformer over the M_l-length seq

Top-down decoder (per level l, from L down to 1; Sec. 2.1.2) -- RECURSIVE:
  X-hat^L := X^L                       # only the top level sees encoder states
  U^l_{g-1} = Converter_l(X-hat^l_{g-1})           # 1 latent -> R_l prefix vecs
  X-hat^{l-1}_{I_g, j} = Decoder_l([U^l_{g-1}; X-hat^{l-1}_{I_g, <j}])
                                                   # within-chunk LATENT recursion
so that  X-hat^0 = D^1 o ... o D^L o E^L o ... o E^1 (X^0).

Two properties of this design that differ from a standard causal LM:
  * Each level-(l-1) chunk g is conditioned only on the PREVIOUS level-l latent
    (shift at level l), so token positions inside one meta-context (C_{<=L}
    consecutive tokens) never see each other: given the top-level history,
    the tokens of a meta-context are conditionally INDEPENDENT. Sampled tokens
    do not feed back into the latent trajectory; the trajectory is a
    deterministic function of the top-level stream.
  * Every decoder attends within a bounded local window of R_l + C_l - 1
    positions, independent of T. Only the encoder streams grow with T.

X-hat^0 -> lm_head -> logits, trained with cross-entropy against the token ids
at the same (unshifted) positions: X-hat^0_j is the model's reconstruction of
embedding(t_j) built from strictly-prior information, so its logits predict
t_j itself (the causal shift is baked into the converter conditioning).

Optional recursive reconstruction loss (paper Eq. 1): cosine distance between
each decoder reconstruction X-hat^{l-1} and the encoder's true X^{l-1},
position-averaged, summed over levels, weight alpha (paper's main results use
alpha=0; Appendix B.1 finds alpha ~= 0.3 best for zero-shot quality).
Deviation noted in README: we detach the encoder targets (the paper does not
specify stop-gradient placement).

Generation (paper Sec. 2.3 / Appendix A): one meta-context (C_{<=L} tokens)
per top-level step, decoded by the same recursive cascade as training.
  * HierGen (Def. A.2): after sampling a meta-context, re-encode its tokens
    bottom-up (all encoder KV streams advance).
  * RecGen (Def. A.3): skip re-encoding; summarize the decoder-side
    reconstructions X-hat^{L-1} with the top chunker and advance ONLY the
    top-level encoder stream. Global KV drops to the top level alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_blocks import TransformerStack, GradKVCache, sample_token


@dataclass
class LevelConfig:
    chunk_size: int          # C_l: how many level-(l-1) vectors form one level-l vector
    dim: int                 # D_l
    prefix_len: int          # R_l: decoder conditioning prefix length
    enc_layers: int
    enc_heads: int
    enc_mlp_hidden: int
    dec_layers: int
    dec_heads: int
    dec_mlp_hidden: int


@dataclass
class PhotonConfig:
    vocab_size: int = 32000
    d0: int = 512
    levels: List[LevelConfig] = field(default_factory=lambda: [
        LevelConfig(chunk_size=4, dim=512, prefix_len=4, enc_layers=2, enc_heads=8,
                    enc_mlp_hidden=1536, dec_layers=2, dec_heads=8, dec_mlp_hidden=1536),
        LevelConfig(chunk_size=4, dim=512, prefix_len=4, enc_layers=2, enc_heads=8,
                    enc_mlp_hidden=1536, dec_layers=2, dec_heads=8, dec_mlp_hidden=1536),
    ])
    max_seq_len: int = 2048
    recon_loss_weight: float = 0.0  # alpha; paper's main results use 0.0
    tie_embeddings: bool = True

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def total_downsample(self) -> int:
        p = 1
        for lv in self.levels:
            p *= lv.chunk_size
        return p


class ContextChunker(nn.Module):
    """Concatenate C_l consecutive vectors and linearly project to D_l."""

    def __init__(self, in_dim: int, chunk_size: int, out_dim: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.proj = nn.Linear(in_dim * chunk_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M_prev, in_dim) -> (B, M_prev / C, out_dim)
        B, M, D = x.shape
        x = x.reshape(B, M // self.chunk_size, self.chunk_size * D)
        return self.proj(x)


class ContextConverter(nn.Module):
    """Expand one coarse latent vector into R_l fine-grained conditioning
    vectors via a strided transposed 1D convolution (kernel=stride=R_l), so
    each output frame is produced from exactly one input latent."""

    def __init__(self, in_dim: int, out_dim: int, prefix_len: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.out_dim = out_dim
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=prefix_len, stride=prefix_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, in_dim) -> (B, M, prefix_len, out_dim)
        B, M, D = x.shape
        x = x.transpose(1, 2)  # (B, D, M)
        y = self.conv(x)       # (B, out_dim, M * prefix_len)
        y = y.transpose(1, 2).reshape(B, M, self.prefix_len, self.out_dim)
        return y


def _shift_with_start(x: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """x: (B, M, D). Returns sequence where position 0 = start, position i = x[i-1]."""
    B, M, D = x.shape
    start_tok = start.view(1, 1, D).expand(B, 1, D)
    return torch.cat([start_tok, x[:, :-1, :]], dim=1)


class PhotonLM(nn.Module):
    def __init__(self, cfg: PhotonConfig):
        super().__init__()
        assert cfg.num_levels >= 1
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d0)

        dims = [cfg.d0] + [lv.dim for lv in cfg.levels]
        self.chunkers = nn.ModuleList([
            ContextChunker(dims[i], cfg.levels[i].chunk_size, dims[i + 1])
            for i in range(cfg.num_levels)
        ])
        self.encoders = nn.ModuleList([
            TransformerStack(cfg.levels[i].enc_layers, dims[i + 1], cfg.levels[i].enc_heads,
                              cfg.levels[i].enc_mlp_hidden, max_seq_len=cfg.max_seq_len)
            for i in range(cfg.num_levels)
        ])
        self.converters = nn.ModuleList([
            ContextConverter(dims[i + 1], dims[i], cfg.levels[i].prefix_len)
            for i in range(cfg.num_levels)
        ])
        # local decoder window: [U (R_l); X-hat_{<j} (C_l - 1)] -> R_l + C_l - 1
        self.decoders = nn.ModuleList([
            TransformerStack(cfg.levels[i].dec_layers, dims[i], cfg.levels[i].dec_heads,
                              cfg.levels[i].dec_mlp_hidden,
                              max_seq_len=cfg.levels[i].prefix_len + cfg.levels[i].chunk_size)
            for i in range(cfg.num_levels)
        ])
        # learned starting latents X-hat^l_0 (paper Sec. 2.1.2), one per level l=1..L,
        # used as the conditioning "previous latent" of the very first chunk.
        self.start_vecs = nn.ParameterList([
            nn.Parameter(torch.zeros(cfg.levels[i].dim)) for i in range(cfg.num_levels)
        ])

        self.lm_head = nn.Linear(cfg.d0, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)
        # start_vecs must NOT be zeros: an exactly-zero start latent makes the
        # first chunk's converter prefix exactly zero, the zero activations
        # propagate through the recursive cascade, and RMSNorm's backward at
        # zero input scales gradients by rsqrt(eps) ~ 316 per crossing --
        # compounding to inf and silently zeroing every step via grad clipping.
        for p in self.start_vecs:
            nn.init.normal_(p, mean=0.0, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.ConvTranspose1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    # ---------------------------------------------------------------
    # Shared decoding primitives (used by forward, generation, tests,
    # and diagnostics -- there is exactly one implementation of the
    # top-down math).
    # ---------------------------------------------------------------
    def _decode_chunks(self, i: int, U: torch.Tensor) -> torch.Tensor:
        """Within-chunk latent recursion (paper Sec. 2.1.2).

        U: (N, R_i, D_i) converter prefixes, one row per chunk. Decodes the
        C_i chunk latents sequentially: X-hat_j is the last-position output of
        a causal pass over [U; X-hat_1..X-hat_{j-1}] (the paper's mask M_{R,j}
        of size (R+j-1) x (R+j-1)). The recursion is over the decoder's own
        outputs -- no teacher forcing with true states, and sampled tokens
        never enter this trajectory.

        Implemented incrementally with an autograd-safe KV cache, so each of
        the R+C-1 window positions is processed exactly once (identical math
        to re-running the growing prefix, ~3x less decoder compute)."""
        C = self.cfg.levels[i].chunk_size
        caches = [GradKVCache() for _ in self.decoders[i].layers]
        h = self.decoders[i](U, kv_caches=caches)   # prefill; last output = X-hat_1
        outs = [h[:, -1:, :]]
        for _ in range(C - 1):
            h = self.decoders[i](outs[-1], kv_caches=caches)
            outs.append(h[:, -1:, :])
        return torch.cat(outs, dim=1)               # (N, C_i, D_i)

    def _decode_level(self, i: int, source: torch.Tensor) -> torch.Tensor:
        """One full top-down level over a whole latent stream (training-style,
        parallel across chunks). source: (B, M, D_{i+1}) = X-hat^{i+1}
        (or X^{L} at the top). Returns X-hat^{i}: (B, M*C_i, D_i)."""
        shifted = _shift_with_start(source, self.start_vecs[i])
        U = self.converters[i](shifted)          # (B, M, R_i, D_i)
        B, M, R, D = U.shape
        xh = self._decode_chunks(i, U.reshape(B * M, R, D))
        return xh.reshape(B, M * self.cfg.levels[i].chunk_size, D)

    # ---------------------------------------------------------------
    # Training forward: recursive top-down cascade, parallel across
    # chunks; within-chunk recursion is C_l cheap bounded-window steps.
    # ---------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                return_parts: bool = False):
        cfg = self.cfg
        B, T = idx.shape
        total_c = cfg.total_downsample
        assert T % total_c == 0, f"seq len {T} must be divisible by total downsample {total_c}"

        x = self.embed(idx)  # (B, T, D0)

        # --- bottom-up encoder ---
        enc_states = [x]
        for i in range(cfg.num_levels):
            a = self.chunkers[i](enc_states[-1])
            xi = self.encoders[i](a)
            enc_states.append(xi)
        # enc_states[i] = X^i for i=0..L (X^0 = token embeddings)

        # --- recursive top-down decoder: X-hat^L := X^L, then descend on
        # reconstructions only (paper: X-hat^0 = D^1 o ... o D^L (X^L)) ---
        recon_loss = x.new_zeros(())
        x_hat_prev = enc_states[-1]
        for i in reversed(range(cfg.num_levels)):
            x_hat = self._decode_level(i, x_hat_prev)
            if targets is not None:
                # paper Eq. 1: position-averaged cosine distance, summed over
                # levels. Targets detached (see module docstring).
                cos = F.cosine_similarity(x_hat, enc_states[i].detach(), dim=-1)
                recon_loss = recon_loss + (1.0 - cos).mean()
            x_hat_prev = x_hat

        logits = self.lm_head(x_hat_prev)  # (B, T, V) from X-hat^0

        loss = None
        parts = None
        if targets is not None:
            token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                         targets.reshape(-1), ignore_index=-1)
            loss = token_loss
            if cfg.recon_loss_weight > 0:
                loss = loss + cfg.recon_loss_weight * recon_loss
            parts = {"token_loss": token_loss, "recon_loss": recon_loss}
        if return_parts:
            return logits, loss, parts
        return logits, loss

    # ---------------------------------------------------------------
    # Generation: one meta-context (C_{<=L} tokens) per top-level step,
    # decoded by the same recursive cascade as training.
    # ---------------------------------------------------------------
    def _align_prompt(self, idx: torch.Tensor) -> torch.Tensor:
        total_c = self.cfg.total_downsample
        excess = idx.shape[1] % total_c
        if excess > 0:
            # left-truncate down to a multiple of total_c (keeps most recent context)
            idx = idx[:, excess:]
        assert idx.shape[1] >= total_c, "prompt too short for chunking configuration"
        return idx

    def _decode_meta_context(self, carry: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Decode all latents of the next meta-context top-down.

        carry[L]: (B, D_L) latest top-level encoder latent X^L_g.
        carry[l] (1 <= l < L): (B, D_l) last X-hat^l of the previous
        meta-context (the cross-boundary conditioning latent).

        Returns {level: (B, n_l, D_l)} with n_l = prod_{k>l} C_k new latents;
        new[0] holds the total_c token-level reconstructions.

        Chunk g at level l is conditioned on latent g-1 at level l+1, exactly
        as in the training shift: the conditioning stream for level l is
        [carry[l+1], new[l+1][0], ..., new[l+1][n-2]]."""
        L = self.cfg.num_levels
        new: Dict[int, torch.Tensor] = {}
        upper: Optional[torch.Tensor] = None
        for i in reversed(range(L)):
            if upper is None:
                cond = carry[i + 1].unsqueeze(1)                      # (B, 1, D_{i+1})
            else:
                cond = torch.cat([carry[i + 1].unsqueeze(1), upper[:, :-1, :]], dim=1)
            U = self.converters[i](cond)                              # (B, n, R, D_i)
            B, n, R, D = U.shape
            xh = self._decode_chunks(i, U.reshape(B * n, R, D))
            xh = xh.reshape(B, n * self.cfg.levels[i].chunk_size, D)
            new[i] = xh
            upper = xh
        return new

    @torch.no_grad()
    def _generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float,
                  top_k: Optional[int], recgen: bool) -> torch.Tensor:
        cfg = self.cfg
        L = cfg.num_levels
        total_c = cfg.total_downsample
        idx = self._align_prompt(idx)
        B = idx.shape[0]

        # --- one-time hierarchical prefill: bottom-up over the prompt,
        # building the encoder KV streams ---
        enc_kv = [self.encoders[i].new_kv_caches() for i in range(L)]
        enc_states = [self.embed(idx)]
        cur = enc_states[0]
        for i in range(L):
            a = self.chunkers[i](cur)
            cur = self.encoders[i](a, kv_caches=enc_kv[i])
            enc_states.append(cur)

        # carry[L] = latest top latent; carry[l] (1<=l<L) = last X-hat^l of the
        # prompt, from the training-style cascade run down to level 1 (level-0
        # reconstructions are never a conditioning source, so we skip them).
        carry: Dict[int, torch.Tensor] = {L: enc_states[L][:, -1, :]}
        x_hat_prev = enc_states[L]
        for i in reversed(range(1, L)):
            x_hat_prev = self._decode_level(i, x_hat_prev)
            carry[i] = x_hat_prev[:, -1, :]

        out_chunks = [idx]
        generated = 0
        while generated < max_new_tokens:
            new = self._decode_meta_context(carry)
            logits = self.lm_head(new[0])                             # (B, total_c, V)
            toks = sample_token(logits.reshape(B * total_c, -1),
                                temperature, top_k).view(B, total_c)
            n_take = min(total_c, max_new_tokens - generated)
            out_chunks.append(toks[:, :n_take])
            generated += n_take
            if n_take < total_c:
                break  # partial final meta-context: no state advance needed

            if recgen:
                # Def. A.3: summary from decoder-side reconstructions; only
                # the top-level encoder stream advances.
                a_top = self.chunkers[L - 1](new[L - 1])              # (B, 1, D_L)
                x_top = self.encoders[L - 1](a_top, kv_caches=enc_kv[L - 1])
                carry[L] = x_top[:, -1, :]
            else:
                # Def. A.2 (HierGen): re-encode sampled tokens bottom-up;
                # every encoder KV stream advances.
                cur = self.embed(toks)
                for i in range(L):
                    a = self.chunkers[i](cur)
                    cur = self.encoders[i](a, kv_caches=enc_kv[i])
                carry[L] = cur[:, -1, :]
            for l in range(1, L):
                carry[l] = new[l][:, -1, :]
        return torch.cat(out_chunks, dim=1)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """HierGen (paper Def. A.2)."""
        return self._generate(idx, max_new_tokens, temperature, top_k, recgen=False)

    @torch.no_grad()
    def generate_recgen(self, idx: torch.Tensor, max_new_tokens: int,
                         temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """RecGen (paper Def. A.3). Works for any number of levels."""
        return self._generate(idx, max_new_tokens, temperature, top_k, recgen=True)
