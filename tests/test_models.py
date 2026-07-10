"""Correctness tests for BaselineLM and PhotonLM.

All PHOTON tests exercise the REAL production code paths (PhotonLM.forward /
generate / generate_recgen) -- no re-implemented decode math.

PHOTON's causality structure (paper Sec. 2.1.2) is coarser than a vanilla
causal LM: each level-(l-1) chunk is conditioned on the PREVIOUS level-l
latent, so given the top-level history, all token positions inside one
meta-context (total_downsample consecutive tokens) are conditionally
independent -- position j must have ZERO gradient not only from future tokens
but from EVERY token in its own meta-context (including j-1), while depending
on tokens from earlier meta-contexts. The causality test asserts exactly this.

Self-consistency: greedy HierGen output must reproduce the teacher-forced
forward pass argmax exactly (both run the same recursive cascade); RecGen must
match HierGen on the first meta-context (identical prefill state) and run
mechanically for any number of levels.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.baseline import BaselineLM, BaselineConfig
from model.photon import PhotonLM, PhotonConfig, LevelConfig

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
    """Gradient test through the REAL PhotonLM.forward.

    total_downsample=16, T=64 (4 meta-contexts). Probe j=37, which lies in
    meta-context 2 (positions 32..47) but NOT in its first level-1 chunk.
    Correct behaviour of the recursive cascade:
      * zero gradient from EVERY token of the same meta-context (32..47),
        including j itself and j-1 (within-meta-context conditional
        independence given the top-level history);
      * zero gradient from all future meta-contexts (48..);
      * nonzero gradient from earlier meta-contexts (0..31), which reach j
        only through the top-level encoder stream."""
    torch.manual_seed(0)
    cfg = _small_photon_cfg()
    m = PhotonLM(cfg).to(DEVICE)
    total_c = cfg.total_downsample
    assert total_c == 16
    T = 64
    idx = torch.randint(0, cfg.vocab_size, (1, T), device=DEVICE)

    captured = {}

    def keep_embedding(mod, inp, out):
        out.retain_grad()
        captured["emb"] = out
        return out

    handle = m.embed.register_forward_hook(keep_embedding)
    try:
        logits, _ = m(idx)
    finally:
        handle.remove()

    j = 37
    logits[0, j].sum().backward()
    grad = captured["emb"].grad[0]  # (T, D0)
    mc_start = (j // total_c) * total_c  # 32

    same_mc = grad[mc_start:mc_start + total_c].abs().sum().item()
    future = grad[mc_start + total_c:].abs().sum().item()
    prev_mc = grad[mc_start - total_c:mc_start].abs().sum().item()
    first_mc = grad[:total_c].abs().sum().item()
    assert same_mc == 0.0, "leakage from tokens inside the same meta-context"
    assert future == 0.0, "leakage from future tokens"
    assert prev_mc > 0.0, "must depend on the previous meta-context"
    assert first_mc > 0.0, "must depend on distant past context"


def test_photon_overfit():
    # B=1: within one meta-context, tokens are conditionally independent given
    # the top-level history, so a batch of DIFFERENT sequences cannot be
    # memorized at the first meta-context (the model has no distinguishing
    # input there). A single fixed sequence is fully memorizable.
    torch.manual_seed(0)
    cfg = _small_photon_cfg()
    m = PhotonLM(cfg).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    idx = torch.randint(0, cfg.vocab_size, (1, 64), device=DEVICE)
    for _ in range(500):
        _, loss = m(idx, idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5, f"PHOTON failed to overfit, final loss={loss.item()}"


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


def test_recgen_first_metacontext_matches_hiergen():
    """Both modes decode the first new meta-context from the identical prefill
    state (same carries, same top latent), so greedy decoding must agree
    exactly on the first total_downsample tokens; later meta-contexts may
    diverge (RecGen advances the top stream from reconstruction summaries)."""
    torch.manual_seed(7)
    cfg = _small_photon_cfg()
    m = PhotonLM(cfg).to(DEVICE)
    m.eval()
    total_c = cfg.total_downsample
    prompt = torch.randint(0, cfg.vocab_size, (3, 2 * total_c), device=DEVICE)
    out_hier = m.generate(prompt, max_new_tokens=total_c, temperature=1e-6, top_k=1)
    out_rec = m.generate_recgen(prompt, max_new_tokens=total_c, temperature=1e-6, top_k=1)
    assert torch.equal(out_hier, out_rec), "first meta-context must be identical across modes"


def test_recgen_runs_long():
    """RecGen mechanics across several top-level stream updates plus a partial
    final meta-context, with batch > 1."""
    torch.manual_seed(8)
    cfg = _small_photon_cfg()
    m = PhotonLM(cfg).to(DEVICE)
    m.eval()
    total_c = cfg.total_downsample
    prompt = torch.randint(0, cfg.vocab_size, (2, 2 * total_c), device=DEVICE)
    n_new = 3 * total_c + 5  # several stream updates plus a partial final meta-context
    out = m.generate_recgen(prompt, max_new_tokens=n_new, temperature=0.8, top_k=20)
    assert out.shape == (2, 2 * total_c + n_new)
    assert out.min().item() >= 0 and out.max().item() < cfg.vocab_size


def test_recgen_3level():
    """RecGen is defined for any L (paper Def. A.3); the old implementation
    hardcoded L=2. Verify a 3-level hierarchy decodes mechanically."""
    torch.manual_seed(9)
    cfg = PhotonConfig(
        vocab_size=60, d0=24,
        levels=[LevelConfig(2, 32, 2, 1, 4, 64, 1, 4, 64),
                LevelConfig(2, 40, 2, 1, 4, 80, 1, 4, 80),
                LevelConfig(2, 48, 2, 1, 4, 96, 1, 4, 96)],
        max_seq_len=256,
    )
    m = PhotonLM(cfg).to(DEVICE)
    m.eval()
    total_c = cfg.total_downsample  # 8
    prompt = torch.randint(0, cfg.vocab_size, (2, 2 * total_c), device=DEVICE)
    out = m.generate_recgen(prompt, max_new_tokens=2 * total_c, temperature=1e-6, top_k=1)
    assert out.shape == (2, 4 * total_c)
    # and it must agree with HierGen on the first meta-context
    out_hier = m.generate(prompt, max_new_tokens=total_c, temperature=1e-6, top_k=1)
    assert torch.equal(out_hier[:, :3 * total_c], out[:, :3 * total_c])


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
