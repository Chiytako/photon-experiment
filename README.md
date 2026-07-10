# PHOTON on GB10 — faithful reimplementation, training, and inference (v2)

Re-implementation of Ichikawa et al. (Fujitsu)'s **PHOTON** ("Parallel
Hierarchical Operation for TOp-down Networks"),
[arXiv:2512.20687](https://arxiv.org/abs/2512.20687), built from the paper's
equations since no official code has been released. Trained and benchmarked
end-to-end on a single GB10 (DGX Spark class, aarch64, 121GB unified memory)
against a matched vanilla LLaMA-style Transformer baseline.

> **v2 notice.** The first public version of this repo deviated from the
> paper's architecture in fundamental ways (see
> [What v1 got wrong](#what-v1-got-wrong) below). v2 is a rewrite against the
> paper's equations (Sec. 2 and Appendix A), with the deviations fixed and
> all experiments re-run. v1 numbers are retained at the bottom for an honest
> before/after comparison — they describe a *different, easier* architecture
> and overstate PHOTON-as-published quality.

## What this is

- `model/photon.py` — PHOTON: hierarchical encoder (chunker + causal
  Transformer per level) and **recursive** hierarchical decoder (converter +
  bounded local Transformer per level). `generate()` implements **HierGen**
  (paper Def. A.2) and `generate_recgen()` implements **RecGen** (Def. A.3),
  both for any number of levels, both decoding one *meta-context*
  (C₁·…·C_L tokens) per top-level step through the same cascade as training.
- `model/baseline.py` — vanilla decoder-only Transformer (same building
  blocks: RMSNorm, RoPE, SwiGLU, SDPA attention) as the efficiency baseline.
- `data.py`, `train.py`, `evaluate.py`, `generate.py`, `benchmark.py` — data
  prep, training, perplexity evaluation, a text-generation CLI, and an
  inference benchmark following the paper's TPM protocol.
- `tests/test_models.py` — causality checks through the **real
  `forward()`**, HierGen ⇄ teacher-forcing self-consistency, RecGen
  first-meta-context equivalence, multi-level configs.
- `scripts/run_v2.sh` — the full v2 experiment pipeline (as run for the
  results below).

Scale note: the paper trains 600M–1.2B-parameter models on 134B tokens of
The Pile on DGX H200s. That's infeasible on a single GB10 in reasonable
time, so this experiment uses **~44–48M-parameter models on 200M tokens**
(plus a ~187M/204M pair on 1.5B tokens) of FineWeb-Edu — enough to probe the
architecture and its qualitative claims, not the paper's absolute numbers.

## The architecture, faithfully

Notation: L levels above tokens (default L=2), chunk sizes C_l, prefix
lengths R_l, X⁰ = token embeddings.

**Bottom-up encoder** (per level l): `A^l = Chunker_l(X^{l-1})` concatenates
C_l consecutive vectors and projects; `X^l = Encoder_l(A^l)` is a causal
Transformer over the M_l-length chunk stream.

**Top-down decoder — the part v1 got wrong.** The cascade is *recursive*
(paper eq.: `X̂⁰ = D¹∘…∘D^L(X^L)`): only the **top** level reads encoder
states; every level below is conditioned on the level above's
**reconstructions**:

- `U^l_{g-1} = Converter_l(X̂^l_{g-1})` — chunk g is conditioned on the
  *previous* level-l latent (shift ⇒ causality), expanded to R_l prefix
  vectors by a `ConvTranspose1d(kernel=stride=R_l)`.
- `X̂^{l-1}_{g,j} = Decoder_l([U^l_{g-1}; X̂^{l-1}_{g,<j}])` — within a chunk
  the decoder recurses on **its own outputs** (the paper's mask
  `M_{R_l,j} ∈ (R_l+j-1)×(R_l+j-1)`), NOT on teacher-forced true states.
  Bounded window ≤ R_l+C_l−1 regardless of T.

Two consequences worth internalizing (both verified by unit tests):

1. **Tokens within one meta-context are conditionally independent** given the
   top-level history. Sampled tokens do not feed back into the latent
   trajectory; position j has *zero* gradient from every token of its own
   meta-context — including j−1. This is what buys chunk-parallel decoding,
   and it is also why PHOTON's perplexity is *architecturally* worse than a
   vanilla Transformer's (the paper's own Table 1 shows the same: 600M
   WikiText PPL 29.9 vs vanilla 22.4).
2. Training, HierGen, and RecGen all run the **same** recursive cascade
   (`_decode_chunks` / `_decode_meta_context` — one implementation). RecGen
   falls out naturally: advance the top stream from the chunker applied to
   the cascade's own X̂^{L-1} instead of re-encoding sampled tokens
   (recursive consistency, paper Sec. 2.3.3).

**Generation.** One meta-context (C₁·…·C_L tokens) per top-level step:
decode all its latents top-down, sample all its tokens (independently —
they're conditionally independent), then advance the streams:
- *HierGen* (Def. A.2): re-encode the sampled tokens bottom-up; every
  encoder KV stream grows (Σ_l O(T/C≤l) memory).
- *RecGen* (Def. A.3): summarize the reconstructions X̂^{L-1} with the top
  chunker; **only the top-level KV grows** (O(T/C≤L)).

**Loss** = token cross-entropy + α · recursive reconstruction loss (paper
Eq. 1: position-averaged cosine distance between X̂^l and X^l, summed over
levels). Reported perplexities always use the **token component only**.

### Remaining documented deviations from the paper

- Data: FineWeb-Edu instead of The Pile; LLaMA tokenizer (32K vocab, same as
  the paper); seq_len 1024 instead of 2048; ~3 orders of magnitude fewer
  tokens and ~13x fewer parameters.
- R_l is not specified in the paper; we use R_l = 4 (= C_l).
- The reconstruction-loss targets are detached (stop-gradient placement is
  unspecified in the paper).
- The learned start latents X̂₀ are initialized N(0, 0.02) — zeros create
  exactly-zero activation paths whose RMSNorm backward explodes to inf
  through the recursion (found the hard way; see git history).
- No zero-shot evals (HellaSwag/SciQ/ARC-e); quality here = perplexity only.
- Presets mirror the paper's *shape* (same hidden width at every level, equal
  layer counts in all four stacks — paper Appendix C.2) scaled down to GB10.

## What v1 got wrong

For the record (and the article's before/after): the original implementation

1. **Was not recursive.** Every level's converter read the *true encoder
   states* (`enc_states[i+1]`) instead of the reconstruction chain — teacher
   forcing baked into the architecture at every level.
2. **Teacher-forced within chunks too**, feeding the decoder the true
   shifted states in training and *sampled-token embeddings* at inference —
   a token-autoregressive local decoder, which the paper's design explicitly
   is not (its within-chunk recursion is over the decoder's own latents; an
   invented "dup trick" also added a position the paper doesn't have).
3. **HierGen re-encoded every C₁ tokens** and conditioned each level-1 chunk
   on the freshest true X¹ — much richer conditioning than the paper's
   meta-context granularity. Together, 1–3 made v1 a *different, easier*
   model: its quality numbers overstate PHOTON-as-published (v1 conditioned
   on ≤4-token-old context; faithful PHOTON conditions on ≥16-token-old
   context at chunk starts).
4. **RecGen was a hand-rolled L=2-only approximation** that didn't match the
   training dataflow (with the recursive architecture it's a three-line
   variation of HierGen, any L).
5. **The benchmark measured the wrong memory.** The paper's TPM divides
   throughput by **per-sample KV-cache memory** (Block Transformer
   protocol); v1 divided by `torch.cuda.max_memory_allocated()` (weights +
   activations included), so v1 "memory ratio" numbers are not the paper's
   metric.
6. Reported perplexity for α>0 runs was contaminated by the reconstruction
   loss term (exp of the *combined* loss), plus assorted infra bugs
   (`--seed` didn't seed NumPy's batch sampling; entry-point scripts missing
   required args; causality test validated a hand-copied duplicate of the
   forward pass instead of the real one).

## Models trained (v2)

| | Baseline | PHOTON v2 |
|---|---|---|
| small: total / non-emb | 43.66M / 27.27M | 47.85M / 31.47M |
| small shape | 512 × 8 layers | d₀=D₁=D₂=512; 4 stacks × 2 layers |
| medium: total / non-emb | 186.93M / 154.17M | 203.72M / 170.95M |
| medium shape | 1024 × 12 layers | d₀=D₁=D₂=1024; 4 stacks × 3 layers |

(PHOTON slightly larger at equal width/depth — chunkers + converters — the
same pattern as the paper's 646M "600M" PHOTON vs 611M vanilla.)

C=(4,4), R=(4,4), L=2 throughout. Both trained on the same tokens:
small on 200M (data/), medium on 1.5B (data2b/), LLaMA tokenizer, seq_len
1024, batch 16, AdamW (lr 3e-4, cosine, 300-step warmup), bf16 autocast,
`torch.compile`. Baselines are unchanged from v1 and were **not** retrained.

## Results (v2)

All numbers from `scripts/run_v2.sh` (photon2 → photon2_a03 → photon2_med →
benchmarks → diagnostics), artifacts in `checkpoints/`.

### Perplexity (token loss only)

| | FineWeb-Edu val | WikiText-103 test |
|---|---|---|
| Baseline small (44M) | 31.7 | 70.2 |
| PHOTON v2 small (48M, α=0) | 1152.5 | 1831.3 |
| PHOTON v2 small (48M, α=0.3) | 1294.1 | 2011.8 |
| Baseline medium (187M) | 17.4 (data2b val) | 29.2 |
| PHOTON v2 medium (204M, α=0) | 1104.3 (data2b val) | 1753.9 |

The faithful architecture pays a **very large quality cost at this scale**:
~26x worse WikiText PPL than the matched baseline (the paper, at 600M params
/ 134B tokens, reports only 1.34x — 29.9 vs 22.4). Notably, scaling 4.3x
params + 7.5x tokens (small → medium) barely moved PHOTON's PPL (1831 →
1754) while the baseline improved 2.4x (70 → 29). At GB10-scale budgets the
conditional-independence bottleneck dominates; whether the paper's regime
genuinely closes the gap is not verifiable here. α=0.3 again slightly hurts
token PPL (its value is RecGen, below).

### Inference (paper TPM protocol: throughput / per-sample KV-cache GiB)

Small pair, batch 4:

| Regime | Baseline TPM | PHOTON TPM (HierGen) | TPM gain | KV memory |
|---|---|---|---|---|
| Prefill-heavy 2048/128 | 3.9 | 387.1 | **99.6x** | 1/12.8 |
| Decode-heavy 128/2048 | 12.6 | 1002.7 | **79.6x** | 1/12.8 |
| Decode-heavy, **RecGen** | 12.4 | **5417.1** | **437x** | **1/64** |

Medium pair: prefill 0.83 → 76.9 (**92.6x**), decode 2.06 → 194.2
(**94.2x**). Raw throughput (not per-memory): PHOTON small decodes at 10.4K
tok/s vs baseline 1.7K (6.2x); RecGen reaches 11.2K.

The efficiency side of the paper **reproduces dramatically** — far more so
than v1 measured (v1's token-by-token HierGen threw away the chunk-parallel
decoding, and its peak-allocated-memory metric buried the KV advantage).
RecGen's 437x decode TPM gain is within sight of the paper's headline
"up to 10³x" at 3 orders of magnitude smaller scale.

### RecGen diagnostics (α=0.3 checkpoint, `scripts/recgen_diag.py`)

Level-1 reconstruction fidelity: **cosine 0.951**. Forced-decoding
continuation PPL: HierGen 1167.9 → RecGen 1307.4 — a **+12% penalty
(+0.11 nats)**, versus v1's +1.6–3.3 nats (5–28x PPL blowup, "unusable").
With the faithful recursive architecture, training-time and RecGen dataflow
coincide, so skipping re-encoding is nearly free — exactly the paper's
recursive-consistency argument (Sec. 2.3.3) working as designed.

### v1 → v2 quality comparison (the cost of implementing the paper correctly)

| | v1 (unfaithful) WikiText | v2 (faithful) WikiText |
|---|---|---|
| PHOTON small α=0 | 261.8 | 1831.3 (**7.0x worse**) |
| Baseline small | 70.2 | 70.2 (unchanged) |

v1's teacher-forced conditioning (every level reading true encoder states,
token-AR local decoder, per-C₁ re-encoding) was worth a ~7x PPL advantage —
quality that belonged to the deviations, not to PHOTON. Conversely v1
under-measured the efficiency gains by an order of magnitude. Both headline
conclusions of the original article flip in opposite directions.

## Reproducing (v2)

```bash
source .venv/bin/activate

# data (only once; streams FineWeb-Edu, ~1 min for the small set)
python data.py --train_tokens 220000000 --val_tokens 2000000

# the whole v2 pipeline (small α=0, small α=0.3, medium, benchmarks, diag):
tmux new -d -s photon2 'bash scripts/run_v2.sh 2>&1 | tee -a v2_run.log'

# or individual pieces:
python train.py --arch photon --run_name my_photon --compile --total_tokens 200000000
python evaluate.py --ckpt checkpoints/my_photon.pt        # val_bin auto-resolved
python benchmark.py --baseline_ckpt checkpoints/baseline_final.pt \
  --photon_ckpt checkpoints/my_photon.pt
python generate.py --ckpt checkpoints/my_photon.pt --prompt "..." [--mode recgen]

# unit tests (causality, HierGen/RecGen consistency; CPU ok)
python tests/test_models.py
```

---

# Appendix: v1 (unfaithful implementation) results

Kept verbatim for the before/after comparison; **these describe a different
architecture** (teacher-forced at every level, token-AR local decoder,
per-chunk re-encoding) and the benchmark memory metric is not the paper's.

### v1 perplexity (200M tokens, small)

| | FineWeb-Edu val | WikiText-103 test |
|---|---|---|
| Baseline | 31.7 | 70.2 |
| PHOTON v1 | 63.6 | 261.8 |

### v1 ablations

| run | change | FineWeb val ppl | WikiText ppl |
|---|---|---|---|
| photon_final | α=0, C=(4,4), R=(4,4) | 63.6 | 261.8 |
| ph_a01 | α=0.1 | 66.4* | 291.5* |
| ph_a02 | α=0.2 | 69.5* | 291.4* |
| ph_a03 | α=0.3 | 72.5* | 329.7* |
| ph_c22 | C=(2,2) | 56.6 | 174.0 |
| ph_r88 | R=(8,8) | 63.4 | 269.3 |

*α>0 rows are additionally contaminated by the reconstruction term (v1 bug
6): they exponentiate the combined loss, so the true token perplexities were
somewhat lower.

### v1 scale-up (1.5B tokens, medium)

| scale | FineWeb val (baseline / PHOTON) | ratio | WikiText (baseline / PHOTON) | ratio |
|---|---|---|---|---|
| small 44M/36M, 200M tok | 34.1 / 67.4 | 1.98x | 70.2 / 261.8 | 3.73x |
| medium 187M/158M, 1.5B tok | 17.4 / 29.8 | 1.71x | 29.2 / 81.8 | 2.80x |

### v1 inference benchmark (peak-allocated-memory metric — NOT the paper's)

| Regime | Baseline tok/s | PHOTON v1 tok/s | Speedup |
|---|---|---|---|
| Prefill-heavy 2048/128 (small) | 454 | 1234 | 2.72x |
| Decode-heavy 128/2048 (small) | 1437 | 2337 | 1.63x |
| Prefill-heavy (medium) | 275 | 1031 | 3.75x |
| Decode-heavy (medium) | 655 | 1800 | 2.75x |

### v1 GB10 throughput notes (still valid — hardware, not architecture)

- Batch size scaling 16→128 gave nothing (~50K tok/s flat, memory linear to
  77GB): GB10 is near-saturated at batch 16 for these model sizes.
- `torch.compile` gave a real ~2x on both architectures (numerically
  verified identical logits); all training uses `--compile`.
