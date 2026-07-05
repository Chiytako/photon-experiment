"""PHOTON: Parallel Hierarchical Operation for TOp-down Networks.

Re-implementation from Fujitsu et al., arXiv:2512.20687 ("PHOTON: Hierarchical
Autoregressive Modeling for Lightspeed and Memory-Efficient Language
Generation"). No official code was released at the time of writing; this is a
from-scratch reimplementation based on the paper's described architecture.

Design (L levels above the raw token sequence, default L=2):

Bottom-up encoder (per level l, causal):
  A^l = ContextChunker_l(X^{l-1})   # concat C_l vectors -> linear -> 1 vector
  X^l = ContextEncoder_l(A^l)       # causal transformer over the M_l-length seq

Top-down decoder (per level l, from L down to 1):
  U^l = ContextConverter_l(shift(X^l))     # 1 latent -> R_l prefix vectors
                                            # (ConvTranspose1d, kernel=stride=R_l)
  own_shift^{l-1}_chunk = [ U^l[-1] , X^{l-1}_chunk[:-1] ]   # local shift-by-1
  dec_in = concat[ U^l , own_shift^{l-1}_chunk ]             # length R_l + C_l
  X_hat^{l-1}_chunk = ContextDecoder_l(dec_in)[-C_l:]        # local causal attn

X_hat^0 -> lm_head -> logits, trained with cross-entropy directly against the
token ids at the same (unshifted) positions, since the causal shift is already
baked into the local decoder's input construction (position i's output only
had access to strictly-prior information).

Only the TOP level's ContextEncoder does full (compressed-length M_L) causal
attention; every decoder does BOUNDED local attention over a fixed window
R_l + C_l, independent of the global sequence length T. This is the source of
PHOTON's claimed efficiency gains.

An optional recursive reconstruction loss (cosine distance between each
decoder's X_hat^{l-1} and the encoder's true X^{l-1}, weight alpha) trains the
decoders to approximate the encoder so that "RecGen" inference can later skip
bottom-up re-encoding. The paper's main results use alpha=0; we default to 0
too but support turning it on.

Generation (HierGen): sequential over level-1 chunks of C_1 tokens. Within a
chunk, tokens are sampled one at a time via ContextDecoder_1's small
(R_1+C_1)-window KV cache. After a chunk completes, its tokens are re-encoded
bottom-up (ContextChunker_1 -> ContextEncoder_1, incrementally via KV cache)
to refresh the true X^1 state; every C_2 level-1 chunks, X^2 is similarly
refreshed. This bottom-up re-encoding after each chunk is what "Hier" in
HierGen refers to (contrasted with "RecGen", which skips it and reuses the
decoders' own reconstructions -- not implemented here, since alpha=0 by
default makes that substitution unreliable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_blocks import TransformerStack, KVCache


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
    d0: int = 256
    levels: List[LevelConfig] = field(default_factory=lambda: [
        LevelConfig(chunk_size=4, dim=384, prefix_len=4, enc_layers=2, enc_heads=6,
                    enc_mlp_hidden=1024, dec_layers=2, dec_heads=6, dec_mlp_hidden=1024),
        LevelConfig(chunk_size=4, dim=448, prefix_len=4, enc_layers=2, enc_heads=7,
                    enc_mlp_hidden=1152, dec_layers=2, dec_heads=7, dec_mlp_hidden=1152),
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

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) single latent -> (B, prefix_len, out_dim); used in generation.
        return self.forward(x.unsqueeze(1)).squeeze(1)


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
        self.decoders = nn.ModuleList([
            TransformerStack(cfg.levels[i].dec_layers, dims[i], cfg.levels[i].dec_heads,
                              cfg.levels[i].dec_mlp_hidden,
                              max_seq_len=cfg.levels[i].prefix_len + cfg.levels[i].chunk_size)
            for i in range(cfg.num_levels)
        ])
        self.start_vecs = nn.ParameterList([
            nn.Parameter(torch.zeros(cfg.levels[i].dim)) for i in range(cfg.num_levels)
        ])

        self.lm_head = nn.Linear(cfg.d0, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

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
    # Training forward: fully parallel, teacher-forced at every level.
    # ---------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
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

        # --- top-down decoder ---
        recon_loss = x.new_zeros(())
        n_recon = 0
        x_hat_prev = None  # will hold X_hat^{l} while descending
        for i in reversed(range(cfg.num_levels)):
            C = cfg.levels[i].chunk_size
            R = cfg.levels[i].prefix_len
            own_dim = enc_states[i].shape[-1]

            source_l = enc_states[i + 1]  # true encoder X^{l}; used for both HierGen & training
            shifted = _shift_with_start(source_l, self.start_vecs[i])
            U = self.converters[i](shifted)  # (B, M_l, R, own_dim)

            own = enc_states[i]  # (B, M_{l-1}, own_dim) == (B, M_l * C, own_dim)
            Bsz, Mlm1, _ = own.shape
            Ml = Mlm1 // C
            own_chunks = own.reshape(Bsz, Ml, C, own_dim)
            own_shift = torch.cat([U[:, :, -1:, :], own_chunks[:, :, :-1, :]], dim=2)  # (B,Ml,C,own_dim)

            dec_in = torch.cat([U, own_shift], dim=2)  # (B, Ml, R+C, own_dim)
            dec_in_flat = dec_in.reshape(Bsz * Ml, R + C, own_dim)
            dec_out = self.decoders[i](dec_in_flat)
            x_hat_chunk = dec_out[:, R:, :]  # (B*Ml, C, own_dim)
            x_hat = x_hat_chunk.reshape(Bsz, Ml, C, own_dim).reshape(Bsz, Mlm1, own_dim)

            if cfg.recon_loss_weight > 0:
                target_states = enc_states[i].detach()
                cos = F.cosine_similarity(x_hat, target_states, dim=-1)
                recon_loss = recon_loss + (1.0 - cos).mean()
                n_recon += 1

            x_hat_prev = x_hat

        x0_hat = x_hat_prev  # (B, T, D0)
        logits = self.lm_head(x0_hat)

        loss = None
        if targets is not None:
            token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = token_loss
            if cfg.recon_loss_weight > 0 and n_recon > 0:
                loss = loss + cfg.recon_loss_weight * (recon_loss / n_recon)
        return logits, loss

    # ---------------------------------------------------------------
    # HierGen inference: sequential chunk-by-chunk generation with KV
    # caching at the encoder levels and small local decoder windows.
    # ---------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None):
        cfg = self.cfg
        device = idx.device
        B = idx.shape[0]
        total_c = cfg.total_downsample
        prompt_len = idx.shape[1]
        excess = prompt_len % total_c
        if excess > 0:
            # left-truncate down to a multiple of total_c (keeps most recent context)
            idx = idx[:, excess:]
            prompt_len = idx.shape[1]
        assert prompt_len >= total_c, "prompt too short for chunking configuration"

        tokens = idx.clone()
        x0_hist = self.embed(tokens)  # (B, T, D0) running embedding history

        # --- prefill encoder hierarchy over the prompt, building KV caches ---
        enc_kv = [self.encoders[i].new_kv_caches() for i in range(cfg.num_levels)]
        enc_hist = [x0_hist]
        cur = x0_hist
        for i in range(cfg.num_levels):
            a = self.chunkers[i](cur)
            xi = self.encoders[i](a, kv_caches=enc_kv[i])
            enc_hist.append(xi)
            cur = xi
        # enc_hist[i]: (B, M_i, D_i) full history so far, one entry per level

        counters = [0] * cfg.num_levels  # counts new level-(i-1) items accumulated toward next level-i chunk

        C0 = cfg.levels[0].chunk_size
        generated_total = 0
        while generated_total < max_new_tokens:
            n_new = min(C0, max_new_tokens - generated_total)
            # Build the R_0 prefix from the most recent (shifted) level-1 latent.
            if enc_hist[1].shape[1] > 0:
                latest_x1 = enc_hist[1][:, -1, :]
            else:
                latest_x1 = self.start_vecs[0].unsqueeze(0).expand(idx.shape[0], -1)
            new_tokens_chunk = self._sample_chunk_tokens(latest_x1, n_new, temperature, top_k)
            tokens = torch.cat([tokens, new_tokens_chunk], dim=1)
            new_emb = self.embed(new_tokens_chunk)
            x0_hist = torch.cat([x0_hist, new_emb], dim=1)
            generated_total += n_new

            if n_new == C0:
                # refresh encoder hierarchy bottom-up incrementally (HierGen re-encoding)
                a0 = self.chunkers[0](x0_hist[:, -C0:, :])  # (B,1,D1)
                x1_new = self.encoders[0](a0, kv_caches=enc_kv[0])
                enc_hist[1] = torch.cat([enc_hist[1], x1_new], dim=1)
                counters[0] += 1
                cur_new = x1_new
                for i in range(1, cfg.num_levels):
                    C_i = cfg.levels[i].chunk_size
                    if counters[i - 1] % C_i == 0:
                        recent = enc_hist[i][:, -C_i:, :]
                        a_i = self.chunkers[i](recent)
                        x_next = self.encoders[i](a_i, kv_caches=enc_kv[i])
                        enc_hist[i + 1] = torch.cat([enc_hist[i + 1], x_next], dim=1)
                        counters[i] += 1
                    else:
                        break
        return tokens

    def _sample_chunk_tokens(self, latest_x1, n_new, temperature, top_k):
        """Generate up to C_0 new tokens using the bounded local decoder-0 window,
        conditioned on `latest_x1`: the latent of the most recent COMPLETED
        level-1 chunk. In HierGen this is the true encoder state X^1 (latest
        entry of enc_hist[1], matching the training forward exactly); in RecGen
        it is the level-1 decoder-side reconstruction X-hat^1.
        """
        cfg = self.cfg
        U0 = self.converters[0].forward_one(latest_x1)  # (B, R_0, D0)

        C0 = cfg.levels[0].chunk_size
        R0 = cfg.levels[0].prefix_len
        kv = self.decoders[0].new_kv_caches()
        self.decoders[0](U0, kv_caches=kv)  # prefill prefix positions 0..R0-1 (output unused)
        # own_shift[0] duplicates the prefix's last vector (mirrors training's
        # own_shift construction, which has no true "previous token" at the
        # start of a chunk); this is position R0, whose output predicts token 0.
        dup = U0[:, -1:, :]
        h = self.decoders[0](dup, kv_caches=kv)
        logits = self.lm_head(h)

        new_tokens = []
        for step in range(n_new):
            step_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
                step_logits = step_logits.masked_fill(step_logits < v[:, [-1]], -float('inf'))
            probs = F.softmax(step_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            new_tokens.append(next_tok)
            if step < n_new - 1:
                emb = self.embed(next_tok)  # (B,1,D0); becomes own_shift[step+1]
                h = self.decoders[0](emb, kv_caches=kv)
                logits = self.lm_head(h)
        return torch.cat(new_tokens, dim=1)

    # ---------------------------------------------------------------
    # RecGen inference: like HierGen, but after prefill the level-0
    # encoder is never called again. The X^1 latent stream is instead
    # continued by the level-1 DECODER's recursive reconstructions
    # (X-hat^1), and the top-level X^2 state is refreshed from those
    # reconstructions. Only the top-level encoder KV grows with T:
    # global KV storage drops from O(T/C1)+O(T/C1C2) to O(T/C1C2).
    #
    # Design notes (the paper describes RecGen only at a high level; the
    # following choices are ours, mirroring the training dataflow):
    # - X-hat^1_k is produced by decoders[1] exactly as in training:
    #   within each level-2 chunk, a local KV window over
    #   [U^1 (from converters[1] on the latest X^2); previous X-hat^1s],
    #   with the training-time "dup" trick at the chunk start.
    # - Consequence: within a level-2 chunk (C1*C2 tokens), the latent
    #   trajectory is predicted open-loop by the top-down pathway -- the
    #   sampled tokens do not feed back into it (they only condition the
    #   token-level decoder within their own chunk). This is the price of
    #   skipping re-encoding; the reconstruction loss (alpha>0) trains
    #   X-hat^1 to track the true X^1, making the approximation viable.
    # - X^2 is refreshed every C1*C2 tokens by chunkers[1]+encoders[1]
    #   applied to the C2 accumulated X-hat^1 latents (encoder-side input
    #   is the reconstruction, not the true X^1 -- again the RecGen trade).
    # ---------------------------------------------------------------
    @torch.no_grad()
    def generate_recgen(self, idx: torch.Tensor, max_new_tokens: int,
                         temperature: float = 1.0, top_k: Optional[int] = None):
        cfg = self.cfg
        assert cfg.num_levels == 2, "RecGen implemented for L=2 hierarchies"
        B = idx.shape[0]
        total_c = cfg.total_downsample
        prompt_len = idx.shape[1]
        excess = prompt_len % total_c
        if excess > 0:
            idx = idx[:, excess:]  # left-truncate, keeps most recent context
            prompt_len = idx.shape[1]
        assert prompt_len >= total_c, "prompt too short for chunking configuration"

        C0, C1 = cfg.levels[0].chunk_size, cfg.levels[1].chunk_size
        R1 = cfg.levels[1].prefix_len

        tokens = idx.clone()
        x0 = self.embed(tokens)

        # --- prefill: one full bottom-up pass over the prompt. Only the
        # top-level encoder KV is retained for decoding. ---
        a1 = self.chunkers[0](x0)
        x1 = self.encoders[0](a1)                      # true X^1 over prompt (no KV kept)
        enc1_kv = self.encoders[1].new_kv_caches()     # top-level KV: the only growing cache
        a2 = self.chunkers[1](x1)
        x2 = self.encoders[1](a2, kv_caches=enc1_kv)

        latest_x1_est = x1[:, -1, :]   # last true X^1 latent (chunk index k-1)
        latest_x2 = x2[:, -1, :]

        dec1 = self.decoders[1]
        dec1_kv = None                  # reset at each level-2 chunk boundary
        xhat1_buffer: list = []         # X-hat^1 latents accumulated toward the next X^2

        chunk_idx = 0                   # level-1 chunks generated so far (mod C1 matters)
        generated_total = 0
        while generated_total < max_new_tokens:
            n_new = min(C0, max_new_tokens - generated_total)
            new_tokens_chunk = self._sample_chunk_tokens(latest_x1_est, n_new, temperature, top_k)
            tokens = torch.cat([tokens, new_tokens_chunk], dim=1)
            generated_total += n_new
            if n_new < C0:
                break  # partial final chunk: no state advance needed

            # --- advance the X^1 stream via the level-1 decoder (no re-encoding) ---
            if chunk_idx % C1 == 0:
                # level-2 chunk boundary: fresh local window conditioned on latest X^2
                U1 = self.converters[1].forward_one(latest_x2)  # (B, R1, D1)
                dec1_kv = dec1.new_kv_caches()
                dec1(U1, kv_caches=dec1_kv)              # prefill prefix (output unused)
                h = dec1(U1[:, -1:, :], kv_caches=dec1_kv)  # training's "dup" position
            else:
                h = dec1(latest_x1_est.unsqueeze(1), kv_caches=dec1_kv)
            latest_x1_est = h[:, -1, :]                  # X-hat^1 for the chunk just completed
            xhat1_buffer.append(latest_x1_est)
            chunk_idx += 1

            # --- refresh top-level X^2 from reconstructions every C1 chunks ---
            if chunk_idx % C1 == 0:
                stack = torch.stack(xhat1_buffer, dim=1)  # (B, C1, D1)
                xhat1_buffer = []
                a2_new = self.chunkers[1](stack)
                x2_new = self.encoders[1](a2_new, kv_caches=enc1_kv)
                latest_x2 = x2_new[:, -1, :]
        return tokens
