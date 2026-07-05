"""RecGen quality diagnostics for a trained PHOTON checkpoint.

1. Reconstruction fidelity: cosine similarity between the decoder-side
   reconstructions X-hat^l and the true encoder states X^l on validation
   text (the quantity the alpha loss optimizes; RecGen substitutes X-hat^1
   for X^1, so this predicts how viable RecGen is for a given checkpoint).

2. Forced-decoding continuation loss: feed real validation text through the
   token-by-token generation state machinery -- (a) HierGen-style updates
   (bottom-up re-encoding; provably equivalent to the parallel forward) vs
   (b) RecGen-style updates (decoder-side latent stream) -- and compare NLL
   on the same ground-truth continuation tokens. The gap is RecGen's quality
   cost, measured without sampling noise.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import load_model
from model.photon import _shift_with_start

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@torch.no_grad()
def recon_cosine(model, idx):
    """Replicates the training forward, returning per-level mean cosine
    similarity between X-hat^l and X^l (levels indexed bottom-up, l=0 is
    token embeddings)."""
    cfg = model.cfg
    x = model.embed(idx)
    enc_states = [x]
    for i in range(cfg.num_levels):
        a = model.chunkers[i](enc_states[-1])
        enc_states.append(model.encoders[i](a))
    sims = {}
    for i in reversed(range(cfg.num_levels)):
        C, R = cfg.levels[i].chunk_size, cfg.levels[i].prefix_len
        own_dim = enc_states[i].shape[-1]
        shifted = _shift_with_start(enc_states[i + 1], model.start_vecs[i])
        U = model.converters[i](shifted)
        own = enc_states[i]
        B, Mlm1, _ = own.shape
        Ml = Mlm1 // C
        own_chunks = own.reshape(B, Ml, C, own_dim)
        own_shift = torch.cat([U[:, :, -1:, :], own_chunks[:, :, :-1, :]], dim=2)
        dec_in = torch.cat([U, own_shift], dim=2).reshape(B * Ml, R + C, own_dim)
        dec_out = model.decoders[i](dec_in)
        x_hat = dec_out[:, R:, :].reshape(B, Ml, C, own_dim).reshape(B, Mlm1, own_dim)
        sims[f"level{i}"] = F.cosine_similarity(x_hat, enc_states[i], dim=-1).mean().item()
    return sims


@torch.no_grad()
def forced_chunk_nll(model, latest_x1, true_chunk_tokens):
    """NLL of one chunk of ground-truth tokens under the local decoder,
    conditioned on `latest_x1` (true X^1 for HierGen, X-hat^1 for RecGen)."""
    C0 = model.cfg.levels[0].chunk_size
    U0 = model.converters[0].forward_one(latest_x1)
    kv = model.decoders[0].new_kv_caches()
    model.decoders[0](U0, kv_caches=kv)
    h = model.decoders[0](U0[:, -1:, :], kv_caches=kv)  # predicts token 0
    nll = 0.0
    for j in range(C0):
        logits = model.lm_head(h)[:, -1, :]
        nll += F.cross_entropy(logits, true_chunk_tokens[:, j], reduction="sum").item()
        if j < C0 - 1:
            emb = model.embed(true_chunk_tokens[:, j:j + 1])
            h = model.decoders[0](emb, kv_caches=kv)
    return nll


@torch.no_grad()
def forced_continuation_loss(model, prompt, continuation, mode):
    """Average NLL/token of `continuation` with chunk-by-chunk state updates
    done HierGen-style (true re-encoding) or RecGen-style (decoder latents)."""
    cfg = model.cfg
    assert cfg.num_levels == 2
    C0, C1 = cfg.levels[0].chunk_size, cfg.levels[1].chunk_size
    B = prompt.shape[0]
    assert prompt.shape[1] % cfg.total_downsample == 0
    assert continuation.shape[1] % (C0 * C1) == 0

    x0 = model.embed(prompt)
    a1 = model.chunkers[0](x0)
    enc0_kv = model.encoders[0].new_kv_caches()
    x1 = model.encoders[0](a1, kv_caches=enc0_kv)
    enc1_kv = model.encoders[1].new_kv_caches()
    a2 = model.chunkers[1](x1)
    x2 = model.encoders[1](a2, kv_caches=enc1_kv)
    latest_x1 = x1[:, -1, :]
    latest_x2 = x2[:, -1, :]

    x1_recent = []       # HierGen: true new X^1 latents awaiting X^2 refresh
    xhat1_buffer = []    # RecGen: reconstructed latents awaiting X^2 refresh
    dec1_kv = None

    total_nll = 0.0
    n_tokens = 0
    n_chunks = continuation.shape[1] // C0
    for k in range(n_chunks):
        chunk = continuation[:, k * C0:(k + 1) * C0]
        total_nll += forced_chunk_nll(model, latest_x1, chunk)
        n_tokens += B * C0

        if mode == "hiergen":
            emb = model.embed(chunk)
            a0 = model.chunkers[0](emb)
            x1_new = model.encoders[0](a0, kv_caches=enc0_kv)
            latest_x1 = x1_new[:, -1, :]
            x1_recent.append(latest_x1)
            if len(x1_recent) == C1:
                stack = torch.stack(x1_recent, dim=1)
                x1_recent = []
                a2_new = model.chunkers[1](stack)
                x2_new = model.encoders[1](a2_new, kv_caches=enc1_kv)
                latest_x2 = x2_new[:, -1, :]
        else:  # recgen
            if k % C1 == 0:
                U1 = model.converters[1].forward_one(latest_x2)
                dec1_kv = model.decoders[1].new_kv_caches()
                model.decoders[1](U1, kv_caches=dec1_kv)
                h = model.decoders[1](U1[:, -1:, :], kv_caches=dec1_kv)
            else:
                h = model.decoders[1](latest_x1.unsqueeze(1), kv_caches=dec1_kv)
            latest_x1 = h[:, -1, :]
            xhat1_buffer.append(latest_x1)
            if len(xhat1_buffer) == C1:
                stack = torch.stack(xhat1_buffer, dim=1)
                xhat1_buffer = []
                a2_new = model.chunkers[1](stack)
                x2_new = model.encoders[1](a2_new, kv_caches=enc1_kv)
                latest_x2 = x2_new[:, -1, :]
    return total_nll / n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_bin", default=os.path.join(DATA_DIR, "val.bin"))
    ap.add_argument("--n_windows", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--prompt_frac", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda"
    model, arch, cfg, _ = load_model(args.ckpt, device)
    assert arch == "photon"
    total_c = cfg.total_downsample

    val = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(val) - args.window - 1, size=args.n_windows)
    batch = np.stack([val[s:s + args.window].astype(np.int64) for s in starts])
    batch = torch.from_numpy(batch).to(device)

    sims = recon_cosine(model, batch)
    print(f"reconstruction cosine similarity (X-hat vs X): {sims}")

    p_len = int(args.window * args.prompt_frac) // total_c * total_c
    c_len = (args.window - p_len) // total_c * total_c
    prompt, cont = batch[:, :p_len], batch[:, p_len:p_len + c_len]
    hier = forced_continuation_loss(model, prompt, cont, "hiergen")
    rec = forced_continuation_loss(model, prompt, cont, "recgen")
    print(f"forced continuation NLL/token  hiergen={hier:.4f} (ppl {math.exp(hier):.2f})  "
          f"recgen={rec:.4f} (ppl {math.exp(rec):.2f})  gap={rec - hier:+.4f}")


if __name__ == "__main__":
    main()
