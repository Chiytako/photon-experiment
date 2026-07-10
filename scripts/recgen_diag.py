"""RecGen quality diagnostics for a trained PHOTON checkpoint.

1. Reconstruction fidelity: cosine similarity between the decoder-side
   reconstructions X-hat^l and the true encoder states X^l on validation
   text (the quantity the alpha loss optimizes; RecGen substitutes the
   X-hat summaries for re-encoding, so this predicts RecGen viability).

2. Forced-decoding continuation loss: teacher-force real validation text
   through the meta-context state machinery -- (a) HierGen-style updates
   (bottom-up re-encoding of the TRUE tokens) vs (b) RecGen-style updates
   (top stream advanced from reconstruction summaries) -- and compare
   NLL/token on the same ground-truth continuation. The gap is RecGen's
   quality cost, measured without sampling noise.

Uses PhotonLM's own decode primitives (_decode_level / _decode_meta_context),
so it always measures exactly what the model does.
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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@torch.no_grad()
def recon_cosine(model, idx):
    """Per-level mean cosine similarity between the recursive cascade's
    X-hat^l and the true encoder X^l (l=0 is token embeddings), computed
    with the model's real forward-path helpers."""
    cfg = model.cfg
    enc_states = [model.embed(idx)]
    for i in range(cfg.num_levels):
        a = model.chunkers[i](enc_states[-1])
        enc_states.append(model.encoders[i](a))
    sims = {}
    x_hat_prev = enc_states[-1]
    for i in reversed(range(cfg.num_levels)):
        x_hat_prev = model._decode_level(i, x_hat_prev)
        sims[f"level{i}"] = F.cosine_similarity(x_hat_prev, enc_states[i], dim=-1).mean().item()
    return sims


@torch.no_grad()
def forced_continuation_loss(model, prompt, continuation, mode):
    """Average NLL/token of `continuation`, advancing the latent streams one
    meta-context at a time exactly like PhotonLM._generate, but teacher-forced
    on the ground-truth tokens instead of sampling. Works for any L."""
    cfg = model.cfg
    L = cfg.num_levels
    total_c = cfg.total_downsample
    B = prompt.shape[0]
    assert prompt.shape[1] % total_c == 0
    assert continuation.shape[1] % total_c == 0

    # prefill (same construction as PhotonLM._generate)
    enc_kv = [model.encoders[i].new_kv_caches() for i in range(L)]
    enc_states = [model.embed(prompt)]
    cur = enc_states[0]
    for i in range(L):
        a = model.chunkers[i](cur)
        cur = model.encoders[i](a, kv_caches=enc_kv[i])
        enc_states.append(cur)
    carry = {L: enc_states[L][:, -1, :]}
    x_hat_prev = enc_states[L]
    for i in reversed(range(1, L)):
        x_hat_prev = model._decode_level(i, x_hat_prev)
        carry[i] = x_hat_prev[:, -1, :]

    total_nll = 0.0
    n_tokens = 0
    n_mc = continuation.shape[1] // total_c
    for g in range(n_mc):
        true_mc = continuation[:, g * total_c:(g + 1) * total_c]
        new = model._decode_meta_context(carry)
        logits = model.lm_head(new[0])  # (B, total_c, V)
        total_nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     true_mc.reshape(-1), reduction="sum").item()
        n_tokens += true_mc.numel()

        if mode == "hiergen":
            cur = model.embed(true_mc)
            for i in range(L):
                a = model.chunkers[i](cur)
                cur = model.encoders[i](a, kv_caches=enc_kv[i])
            carry[L] = cur[:, -1, :]
        else:  # recgen
            a_top = model.chunkers[L - 1](new[L - 1])
            x_top = model.encoders[L - 1](a_top, kv_caches=enc_kv[L - 1])
            carry[L] = x_top[:, -1, :]
        for l in range(1, L):
            carry[l] = new[l][:, -1, :]
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
    starts = rng.integers(0, len(val) - args.window, size=args.n_windows)
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

    import json
    run_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                            f"recgen_diag_{run_name}.json")
    with open(out_path, "w") as f:
        json.dump({"ckpt": args.ckpt, "recon_cosine": sims,
                   "hiergen_nll": hier, "recgen_nll": rec,
                   "hiergen_ppl": math.exp(hier), "recgen_ppl": math.exp(rec)}, f, indent=2)
    print(f"saved diagnostics to {out_path}")


if __name__ == "__main__":
    main()
