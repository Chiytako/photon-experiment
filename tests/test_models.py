"""Correctness tests for BaselineLM and PhotonLM: strict causality (no
gradient leakage from present/future positions), overfit-to-near-zero-loss
on a tiny fixed batch, and -- for PHOTON specifically -- self-consistency
between HierGen incremental generation and the parallel teacher-forced
forward pass (the strongest available check given no official reference
implementation exists to compare against)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from model.baseline import BaselineLM, BaselineConfig
from model.photon import PhotonLM, PhotonConfig, LevelConfig, _shift_with_start

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_baseline_causality():
    torch.manual_seed(0)
    cfg = BaselineConfig(vocab_size=100, dim=64, n_layers=2, n_heads=4, mlp_hidden=128,
                          max_seq_len=32, tie_embeddings=False)
    m = BaselineLM(cfg).to(DEVICE)
    idx = torch.randint(0, 100, (1, 16), device=DEVICE)
    emb = m.tok_emb(idx).detach().clone().requires_grad_(True)
    logits = m.lm_head(m.stack(emb))
    loss = logits[0, 5].sum()
    grad = torch.autograd.grad(loss, emb)[0][0]
    assert grad[10].abs().sum().item() == 0.0
    assert grad[2].abs().sum().item() > 0.0


def test_baseline_overfit():
    # NOTE: targets must be shifted by one position relative to inputs.
    # Baseline's causal self-attention lets position j attend to token j
    # itself, so target=input (unshifted) would let the model trivially
    # "predict" each token from itself -- not a genuine LM objective.
    torch.manual_seed(0)
    cfg = BaselineConfig(vocab_size=50, dim=64, n_layers=2, n_heads=4, mlp_hidden=128, max_seq_len=32)
    m = BaselineLM(cfg).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    seq = torch.randint(0, 50, (4, 33), device=DEVICE)
    idx, targets = seq[:, :-1], seq[:, 1:]
    for _ in range(300):
        _, loss = m(idx, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1.0, f"baseline failed to overfit, final loss={loss.item()}"


def _small_photon_cfg(vocab=50, d0=32):
    return PhotonConfig(
        vocab_size=vocab, d0=d0,
        levels=[
            LevelConfig(chunk_size=4, dim=48, prefix_len=4, enc_layers=1, enc_heads=4,
                        enc_mlp_hidden=96, dec_layers=1, dec_heads=4, dec_mlp_hidden=96),
            LevelConfig(chunk_size=4, dim=64, prefix_len=4, enc_layers=1, enc_heads=4,
                        enc_mlp_hidden=128, dec_layers=1, dec_heads=4, dec_mlp_hidden=128),
        ],
        max_seq_len=256,
    )


def test_photon_causality():
    torch.manual_seed(0)
    cfg = _small_photon_cfg()
    m = PhotonLM(cfg).to(DEVICE)
    T = 64
    idx = torch.randint(0, cfg.vocab_size, (1, T), device=DEVICE)
    emb_leaf = m.embed(idx).detach().clone().requires_grad_(True)

    enc_states = [emb_leaf]
    for i in range(cfg.num_levels):
        a = m.chunkers[i](enc_states[-1])
        enc_states.append(m.encoders[i](a))
    x_hat_prev = None
    for i in reversed(range(cfg.num_levels)):
        C, R = cfg.levels[i].chunk_size, cfg.levels[i].prefix_len
        own_dim = enc_states[i].shape[-1]
        shifted = _shift_with_start(enc_states[i + 1], m.start_vecs[i])
        U = m.converters[i](shifted)
        own = enc_states[i]
        B, Mlm1, _ = own.shape
        Ml = Mlm1 // C
        own_chunks = own.reshape(B, Ml, C, own_dim)
        own_shift = torch.cat([U[:, :, -1:, :], own_chunks[:, :, :-1, :]], dim=2)
        dec_in = torch.cat([U, own_shift], dim=2).reshape(B * Ml, R + C, own_dim)
        dec_out = m.decoders[i](dec_in)
        x_hat = dec_out[:, R:, :].reshape(B, Ml, C, own_dim).reshape(B, Mlm1, own_dim)
        x_hat_prev = x_hat
    logits = m.lm_head(x_hat_prev)

    j = 30
    loss = logits[0, j].sum()
    grad = torch.autograd.grad(loss, emb_leaf)[0][0]
    assert grad[j].abs().sum().item() == 0.0, "position j must not see its own true input"
    assert grad[j + 1].abs().sum().item() == 0.0, "no leakage from future positions"
    assert grad[j + 10].abs().sum().item() == 0.0
    assert grad[j - 1].abs().sum().item() > 0.0, "must depend on past positions"
    assert grad[0].abs().sum().item() > 0.0


def test_photon_overfit():
    torch.manual_seed(0)
    cfg = _small_photon_cfg()
    m = PhotonLM(cfg).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    idx = torch.randint(0, cfg.vocab_size, (4, 32), device=DEVICE)
    for _ in range(300):
        _, loss = m(idx, idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"PHOTON failed to overfit, final loss={loss.item()}"


def _hiergen_self_consistency(seed, levels, d0, T_prompt, max_new, vocab=60):
    torch.manual_seed(seed)
    cfg = PhotonConfig(vocab_size=vocab, d0=d0, levels=levels, max_seq_len=512)
    m = PhotonLM(cfg).to(DEVICE)
    m.eval()
    total_c = cfg.total_downsample
    assert T_prompt % total_c == 0 and max_new % total_c == 0
    prompt = torch.randint(0, vocab, (3, T_prompt), device=DEVICE)
    out = m.generate(prompt, max_new_tokens=max_new, temperature=1e-6, top_k=1)
    logits, _ = m(out, out)
    argmax_all = logits.argmax(-1)
    gen = out[:, T_prompt:]
    teacher = argmax_all[:, T_prompt:]
    match = (gen == teacher).float().mean().item()
    assert match > 0.999, f"HierGen/teacher-forced mismatch: {match}"


def test_hiergen_matches_teacher_forcing_2level():
    _hiergen_self_consistency(
        2, [LevelConfig(3, 40, 5, 1, 4, 80, 1, 4, 80), LevelConfig(5, 56, 3, 1, 4, 112, 1, 4, 112)],
        24, 30, 30,
    )


def test_hiergen_matches_teacher_forcing_3level():
    _hiergen_self_consistency(
        3, [LevelConfig(2, 32, 2, 1, 4, 64, 1, 4, 64),
            LevelConfig(2, 40, 2, 1, 4, 80, 1, 4, 80),
            LevelConfig(2, 48, 2, 1, 4, 96, 1, 4, 96)],
        24, 32, 32,
    )


def test_hiergen_matches_teacher_forcing_uneven_chunks():
    _hiergen_self_consistency(
        4, [LevelConfig(6, 64, 6, 1, 4, 128, 1, 4, 128), LevelConfig(4, 80, 4, 1, 4, 160, 1, 4, 160)],
        32, 48, 48,
    )


def test_recgen_first_chunk_matches_hiergen():
    """Both modes condition the first generated chunk on the same true X^1
    from prefill, so greedy decoding must agree exactly on the first C_0
    tokens; later chunks may diverge (RecGen substitutes decoder-side
    reconstructions for re-encoded latents)."""
    torch.manual_seed(7)
    cfg = _small_photon_cfg()
    m = PhotonLM(cfg).to(DEVICE)
    m.eval()
    total_c = cfg.total_downsample
    prompt = torch.randint(0, cfg.vocab_size, (3, 2 * total_c), device=DEVICE)
    C0 = cfg.levels[0].chunk_size
    out_hier = m.generate(prompt, max_new_tokens=C0, temperature=1e-6, top_k=1)
    out_rec = m.generate_recgen(prompt, max_new_tokens=C0, temperature=1e-6, top_k=1)
    assert torch.equal(out_hier, out_rec), "first chunk must be identical across modes"


def test_recgen_runs_long():
    """RecGen mechanics across several level-2 chunk boundaries (X^2 refresh
    from reconstructions, dec1 KV resets) with batch > 1."""
    torch.manual_seed(8)
    cfg = _small_photon_cfg()
    m = PhotonLM(cfg).to(DEVICE)
    m.eval()
    total_c = cfg.total_downsample
    prompt = torch.randint(0, cfg.vocab_size, (2, 2 * total_c), device=DEVICE)
    n_new = 3 * total_c + 5  # several X^2 refreshes plus a partial final chunk
    out = m.generate_recgen(prompt, max_new_tokens=n_new, temperature=0.8, top_k=20)
    assert out.shape == (2, 2 * total_c + n_new)
    assert out.min().item() >= 0 and out.max().item() < cfg.vocab_size


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
