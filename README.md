# PHOTON on GB10 — from-scratch reimplementation, training, and inference

Re-implementation of Fujitsu et al.'s **PHOTON** ("Parallel Hierarchical
Operation for TOp-down Networks"), [arXiv:2512.20687](https://arxiv.org/abs/2512.20687),
built from the paper's description since no official code has been released.
Trained and benchmarked end-to-end on a single GB10 (DGX Spark, aarch64,
121GB unified memory, CUDA 13.0) against a matched vanilla LLaMA-style
Transformer baseline.

## What this is

- `model/photon.py` — PHOTON: hierarchical encoder (chunker + causal
  Transformer per level) and hierarchical decoder (converter + bounded local
  Transformer per level), trained fully in parallel via teacher forcing.
  `PhotonLM.generate()` implements **HierGen**: sequential chunk-by-chunk
  autoregressive decoding with KV caching at the encoder levels and small
  fixed-size local decoder windows.
- `model/baseline.py` — vanilla decoder-only Transformer (same building
  blocks: RMSNorm, RoPE, SwiGLU, SDPA attention) as the efficiency baseline.
- `data.py`, `train.py`, `evaluate.py`, `generate.py`, `benchmark.py` — data
  prep, training, perplexity evaluation, a text-generation CLI, and an
  inference throughput/memory benchmark.
- `tests/test_models.py` — causality checks (gradient-based proof that a
  position's logits depend only on strictly-prior tokens) and a HierGen vs.
  parallel-teacher-forced self-consistency check, since there's no reference
  implementation to validate against.

Scale note: the paper trains 600M–1.2B-parameter models on 134B tokens of
The Pile. That's infeasible on a single GB10 in a reasonable time, so this
experiment uses **~30–45M-parameter models trained on 200M tokens of
FineWeb-Edu** — enough to validate the architecture and its qualitative
efficiency claims, not to reproduce the paper's absolute numbers.

## Architecture as implemented

Two levels above the token sequence (L=2), chunk sizes C₁=C₂=4, prefix
lengths R₁=R₂=4 (the paper doesn't spell out R_l; this is a documented
assumption). Design decisions made to fill gaps in the paper's description,
verified for internal consistency (see below):

- **Chunker**: concatenate C_l vectors, linear-project to D_l.
- **Converter**: `ConvTranspose1d(kernel=stride=R_l)` — expands one coarse
  latent into R_l fine-grained conditioning vectors, one latent → one output
  block, no cross-chunk leakage.
- **Decoder**: local causal Transformer over `[prefix (R_l); own-level chunk,
  shifted by one position] ` (window R_l+C_l, independent of global sequence
  length T). The shift is what makes position j's output depend only on
  strictly-prior information — verified by `test_photon_causality`.
- **Converters always read the encoder's true state** (X¹, X²), never
  another decoder's reconstruction — training and HierGen inference use the
  identical dataflow, so there's no train/inference mismatch to worry about.
  This means only the **top-level encoder does full (compressed-length)
  attention**; everything else is a bounded local window.
- Recursive/reconstruction loss (α) implemented but **set to 0**, matching
  the paper's main configuration.

**Correctness validation** (`tests/test_models.py`, all passing):
1. Gradient-based causality proof for both models (position j's logits have
   zero gradient w.r.t. token j itself or any later token).
2. Overfit-to-near-zero-loss on a tiny fixed batch (optimization sanity).
3. HierGen generation vs. parallel teacher-forced forward pass produce
   **identical argmax predictions** across several (level count, chunk size,
   prefix length) configurations — the strongest available check given no
   reference implementation exists.

## Models trained

| | Baseline (vanilla Transformer) | PHOTON |
|---|---|---|
| Total params | 43.66M | 36.31M |
| Non-embedding params | 27.27M | 28.12M |
| d_model | 512 | d₀=256, D₁=384, D₂=512 |
| Layers | 8 | 3 enc + 3 dec per level × 2 levels |

Both trained on the **same 200M tokens** of FineWeb-Edu (`sample-10BT`,
LLaMA tokenizer, vocab 32,000), seq_len 1024, batch 16, AdamW (lr 3e-4,
cosine schedule, 300-step warmup), bf16 autocast, `torch.compile`.

## Results

### Perplexity

| | FineWeb-Edu val | WikiText-103 test |
|---|---|---|
| Baseline | 31.7 | 70.2 |
| PHOTON | 63.6 | 261.8 |

At this parameter/token budget, PHOTON trains to meaningfully worse
perplexity than the dense baseline — roughly 2x worse in-distribution, ~3.7x
worse out-of-distribution (WikiText). This is a genuine, verified result
(not a bug — see correctness validation above): compressing context through
a hierarchy with bounded local decoders appears to need more
tokens/tuning/scale to match a dense model's quality, consistent with
efficient-attention architectures elsewhere in the literature. The paper's
own results are at 3–6x the parameters and >600x the tokens used here.

### Inference throughput & memory (`benchmark.py`, batch=4)

| Regime | Baseline tok/s | PHOTON tok/s | Speedup | Baseline mem | PHOTON mem | Mem ratio |
|---|---|---|---|---|---|---|
| Prefill-heavy (2048 in / 128 out) | 454 | 1234 | **2.72x** | 0.51 GiB | 0.27 GiB | **1.89x** |
| Decode-heavy (128 in / 2048 out) | 1437 | 2337 | **1.63x** | 0.43 GiB | 0.26 GiB | **1.62x** |

Directionally confirms the paper's central claim — PHOTON is faster and
lighter at inference, especially on long-context prefill, because only the
top-level encoder does full-length attention while everything else is a
bounded local window with a much smaller KV footprint.

### Generation samples (temperature 0.8, top_k 50)

Prompt: *"The history of artificial intelligence began in the"*

**Baseline:**
> The history of artificial intelligence began in the 1930s and the beginning of the 1930s. The 1940s, when the world's first artificial intelligence began to spread more and more widely the population, came into play. The 1960s, called the world's first artificial intelligence program, led to a huge increase in the number of artificial intelligence programs...

**PHOTON (HierGen):**
> The history of artificial intelligence began in the 1850s Feb. in 1850, and in 1849 the new military force that was followed by a new military force. After its new war in the 1920s, the war effort was created in the 1970s...

Both produce fluent (if factually nonsensical, as expected at this scale)
English via genuine autoregressive sampling. PHOTON's output is visibly
less coherent, consistent with its higher perplexity.

## Getting the most out of GB10

Investigated where training throughput actually comes from on this
hardware, since GPU occupancy alone (96% SM busy) doesn't mean the
hardware's potential is being used:

- **Batch size scaling gave nothing.** Scanned batch 16→128 for the
  baseline model: throughput was flat at ~50K tok/s regardless (49.7K →
  51.2K → 51.0K → 50.7K, within noise), while peak memory grew linearly
  (10GB → 77GB) toward OOM at 256. At this model size, GB10 is already
  near-saturated at batch 16 — larger batches just burn memory for no
  speed gain.
- **`torch.compile` gave a real ~2x.** Baseline: 47.8K → 95.4K tok/s.
  PHOTON: 68.2K → 129.5K tok/s. Verified numerically identical logits
  between eager and compiled execution before trusting it for training.
  `train.py --compile` uses this; checkpointing always saves the
  *uncompiled* module's `state_dict`, so compiled and eager checkpoints
  load identically in `evaluate.py`/`generate.py`.

Applying `--compile`, both models were retrained on the full 200M-token
budget in **~63 minutes combined** (36.5 min baseline + 26.6 min PHOTON),
versus ~118 minutes without it — validated to reproduce the same perplexity
(within noise) as the original eager-mode run.

---

# Continuation experiments: ablations, RecGen, scale-up

Second round of experiments on the same GB10, addressing three questions:
does the paper's reconstruction loss / different chunking close the quality
gap (ablations)? does the paper's second inference mode work (RecGen)? and
does the quality gap shrink with scale (~4x params, 7.5x tokens)?

## Ablations (small PHOTON, 200M tokens each, ~27 min/run)

| run | change vs default | FineWeb val ppl | WikiText ppl |
|---|---|---|---|
| photon_final | α=0, C=(4,4), R=(4,4) | 63.6 | 261.8 |
| ph_a01 | α=0.1 | 66.4 | 291.5 |
| ph_a02 | α=0.2 | 69.5 | 291.4 |
| ph_a03 | α=0.3 | 72.5 | 329.7 |
| **ph_c22** | **C=(2,2)** | **56.6** | **174.0** |
| ph_r88 | R=(8,8) | 63.4 | 269.3 |

- **α (reconstruction loss) monotonically hurts token perplexity** — consistent
  with the paper using α=0 for its main results. Its value is enabling RecGen
  (below), not quality.
- **Gentler compression C=(2,2) meaningfully improves quality** (WikiText
  262→174), trading away compression ratio (4x vs 16x). The quality gap is
  directly related to how hard the hierarchy squeezes context.
- Converter prefix length R has no measurable effect.

## RecGen (second inference mode)

`PhotonLM.generate_recgen()` implements the paper's RecGen: after prefill the
level-0 encoder is never called again — the X¹ latent stream is continued by
the level-1 *decoder's* recursive reconstructions, and only the top-level
encoder KV grows with T (O(T/C₁C₂) vs HierGen's additional O(T/C₁)). Where
the paper is vague we made the design mirror the training dataflow exactly
(documented in `model/photon.py`); a unit test proves the first generated
chunk matches HierGen bit-for-bit, and `scripts/recgen_diag.py` measures the
approximation cost on real text without sampling noise (forced decoding).

| checkpoint | X̂¹ vs X¹ cosine | forced-continuation ppl HierGen → RecGen |
|---|---|---|
| α=0 | 0.64 | 60 → 1679 (unusable) |
| α=0.1 | 0.94 | 61 → 315 |
| α=0.2 | 0.94 | 60 → 309 |
| α=0.3 | 0.94 | 60 → 304 |

- **RecGen requires α>0**, exactly as the paper implies: the reconstruction
  loss lifts X̂¹ fidelity from 0.64 to ~0.94 cosine and halves the RecGen
  penalty (+3.3 → +1.6 nats). α beyond 0.1 adds nothing.
- **At this scale RecGen remains far behind HierGen** (~5x worse ppl even
  with α): the latent trajectory runs open-loop within each meta-context
  (sampled tokens don't feed back until the next X² refresh), and a 36M model
  can't predict latents accurately enough. The paper's usable-RecGen claims
  presumably need their 600M+/134B-token regime.
- Speed: RecGen ≈ HierGen at ≤8K contexts on GB10 (the skipped re-encoding
  is cheap here); its concrete win is **~30% lower decode memory**
  (0.26 vs 0.37 GiB at 8K decode).

## Scale-up: does the quality gap close with scale?

Medium pair — baseline 187M / PHOTON 158M (`--preset medium`), both trained
on the same fresh 1.5B FineWeb-Edu tokens (`data2b/`, held-out val is
disjoint from the training stream), α=0, identical settings. Wall-clock on
GB10 with `--compile`: baseline 12.7h @ 32.7K tok/s, PHOTON 7.5h @ 55.5K
tok/s (PHOTON trains 1.7x faster at this scale).

**Quality gap (PHOTON ppl / baseline ppl) — the headline result:**

| scale | FineWeb val (baseline / PHOTON) | ratio | WikiText (baseline / PHOTON) | ratio |
|---|---|---|---|---|
| small: 44M/36M, 200M tok | 34.1 / 67.4 | 1.98x | 70.2 / 261.8 | 3.73x |
| medium: 187M/158M, 1.5B tok | 17.4 / 29.8 | **1.71x** | 29.2 / 81.8 | **2.80x** |

(both scales evaluated on the same data2b val set; the small models' numbers
here differ slightly from the first-round table, which used the old val set)

**The gap shrinks with scale** — in-distribution 1.98→1.71, out-of-
distribution 3.73→2.80 — supporting the paper's premise that hierarchical
compression becomes less costly as capacity grows, though at 158M/1.5B the
gap is still substantial.

**Inference efficiency grows with scale AND length** (batch 4, medium pair):

| regime | baseline tok/s | PHOTON tok/s | speedup | mem ratio |
|---|---|---|---|---|
| prefill-heavy 2048/128 | 275 | 1031 | **3.75x** (was 2.7x at small) | 1.74x |
| decode-heavy 128/2048 | 655 | 1800 | **2.75x** (was 1.6x) | 1.65x |
| decode-heavy 128/8192 | 273 | 1733 | **6.36x** | 2.37x |

The bigger the model and the longer the sequence, the more PHOTON's bounded
local windows pay off — the trend that, extrapolated, underlies the paper's
headline throughput-per-memory numbers.

Generation samples (temp 0.8): both medium models produce fluent multi-
sentence English; photon_med is grammatical but drifts topically sooner than
baseline_med, consistent with its higher perplexity.

## Reproducing (continuation experiments)

```bash
source .venv/bin/activate

# ablations (5 runs + eval each)
./scripts/run_ablations.sh

# RecGen quality diagnostics / benchmark / demo
python scripts/recgen_diag.py --ckpt checkpoints/ph_a01.pt
python benchmark.py --photon_ckpt checkpoints/ph_a01.pt --photon_mode recgen
python generate.py --ckpt checkpoints/ph_a01.pt --mode recgen --prompt "..."

# scale-up (2.2B-token dataset, then ~20h of training)
python data.py --out_dir data2b --train_tokens 2200000000 --val_tokens 4000000
TOTAL_TOKENS=1500000000 PHOTON_ALPHA=0.0 ./scripts/run_scaleup.sh
python evaluate.py --ckpt checkpoints/photon_med.pt --val_bin data2b/val.bin
```

## Reproducing (first round)

```bash
source .venv/bin/activate

# 1. data (streams FineWeb-Edu, tokenizes with the LLaMA tokenizer, ~1 min)
python data.py --train_tokens 220000000 --val_tokens 2000000

# 2. train (use --compile; ~2x faster, numerically verified equivalent)
python train.py --arch baseline --compile --total_tokens 200000000
python train.py --arch photon   --compile --total_tokens 200000000

# 3. evaluate perplexity
python evaluate.py --ckpt checkpoints/baseline_final.pt
python evaluate.py --ckpt checkpoints/photon_final.pt

# 4. inference throughput/memory benchmark
python benchmark.py

# 5. generate text
python generate.py --ckpt checkpoints/photon_final.pt \
  --prompt "Your prompt here" --max_new_tokens 200

# unit tests (causality + HierGen self-consistency)
python tests/test_models.py
```
