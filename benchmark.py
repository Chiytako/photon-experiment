"""Inference throughput / memory benchmark for baseline vs PHOTON, following
the paper's protocol (Sec. 3, after Ho et al. 2024 / Block Transformer):
TPM = throughput / per-sample KV-cache memory (K tokens/s/GiB), measured under
prefill-heavy (2048 in / 128 out) and decode-heavy (128 in / 2048 out) regimes.

KV-cache memory is accounted analytically from the architecture (exact for
these models, bf16):
  baseline       : n_layers * T * dim          (* 2 for K+V, * B, * 2 bytes)
  PHOTON HierGen : sum_l enc_layers_l * M_l * D_l   with M_l = T / C_{<=l}
  PHOTON RecGen  : top level term only
PHOTON's local decoders use bounded windows (<= R+C-1) recomputed per chunk
and hold no persistent KV. `torch.cuda.max_memory_allocated()` (weights +
activations included) is reported as a secondary diagnostic only -- it is NOT
the paper's TPM denominator."""
import argparse
import json
import os
import time

import torch

from evaluate import load_model, CKPT_DIR

BF16_BYTES = 2


def kv_cache_gib(arch, cfg, batch_size, total_len, mode="hiergen"):
    """Per-run KV-cache size in GiB at the end of generation (analytic)."""
    if arch == "baseline":
        entries = cfg.n_layers * batch_size * total_len * cfg.dim
    else:
        dims = [cfg.d0] + [lv.dim for lv in cfg.levels]
        m = total_len
        entries = 0
        for i, lv in enumerate(cfg.levels):
            m //= lv.chunk_size  # encoder i runs at M_{i+1} = T / C_{<=i+1}
            is_top = i == cfg.num_levels - 1
            if mode == "hiergen" or is_top:
                entries += lv.enc_layers * batch_size * m * dims[i + 1]
    return entries * 2 * BF16_BYTES / (1024 ** 3)  # *2 for K and V


@torch.no_grad()
def run_once(model, arch, cfg, prompt_len, gen_len, batch_size, vocab_size,
             total_c=1, device="cuda", photon_mode="hiergen"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pl = prompt_len if arch == "baseline" else (prompt_len // total_c) * total_c
    idx = torch.randint(0, vocab_size, (batch_size, pl), device=device)

    gen_fn = model.generate
    if arch == "photon" and photon_mode == "recgen":
        gen_fn = model.generate_recgen
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = gen_fn(idx, max_new_tokens=gen_len, temperature=1.0, top_k=50)
    torch.cuda.synchronize()
    dt = time.time() - t0

    n_generated = out.shape[1] - pl
    total_len = out.shape[1]
    mode = photon_mode if arch == "photon" else "hiergen"
    kv_gib = kv_cache_gib(arch, cfg, batch_size, total_len, mode)
    peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    throughput = (batch_size * n_generated) / dt
    return {
        "prompt_len": pl, "gen_len": n_generated, "batch_size": batch_size,
        "wall_time_s": dt, "throughput_tok_per_s": throughput,
        "kv_cache_gib": kv_gib,
        "tpm_k_tok_per_s_per_gib": (throughput / 1000) / kv_gib if kv_gib > 0 else None,
        "peak_mem_gib_secondary": peak_mem_gib,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", required=True,
                     help="e.g. checkpoints/baseline_final.pt")
    ap.add_argument("--photon_ckpt", required=True,
                     help="e.g. checkpoints/photon2.pt")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--prefill_prompt", type=int, default=2048)
    ap.add_argument("--prefill_gen", type=int, default=128)
    ap.add_argument("--decode_prompt", type=int, default=128)
    ap.add_argument("--decode_gen", type=int, default=2048)
    ap.add_argument("--photon_mode", choices=["hiergen", "recgen"], default="hiergen",
                     help="Which PHOTON decoding mode to benchmark. RecGen skips "
                          "bottom-up re-encoding and needs an alpha>0-trained "
                          "checkpoint to be meaningful quality-wise.")
    ap.add_argument("--out_suffix", default=None,
                     help="Suffix for the results JSON filename; defaults to the "
                          "photon mode (empty for hiergen).")
    args = ap.parse_args()

    device = "cuda"
    results = {}
    for arch, ckpt_path in [("baseline", args.baseline_ckpt), ("photon", args.photon_ckpt)]:
        print(f"\n=== Benchmarking {arch} ===")
        model, loaded_arch, cfg, seq_len = load_model(ckpt_path, device)
        assert loaded_arch == arch
        vocab_size = cfg.vocab_size
        total_c = cfg.total_downsample if arch == "photon" else 1

        print(f"[{arch}] prefill-heavy ({args.prefill_prompt} in / {args.prefill_gen} out)")
        prefill_res = run_once(model, arch, cfg, args.prefill_prompt, args.prefill_gen,
                                args.batch_size, vocab_size, total_c, device,
                                photon_mode=args.photon_mode)
        print(json.dumps(prefill_res, indent=2))

        print(f"[{arch}] decode-heavy ({args.decode_prompt} in / {args.decode_gen} out)")
        decode_res = run_once(model, arch, cfg, args.decode_prompt, args.decode_gen,
                               args.batch_size, vocab_size, total_c, device,
                               photon_mode=args.photon_mode)
        print(json.dumps(decode_res, indent=2))

        results[arch] = {"prefill_heavy": prefill_res, "decode_heavy": decode_res}
        del model
        torch.cuda.empty_cache()

    suffix = args.out_suffix
    if suffix is None:
        suffix = "" if args.photon_mode == "hiergen" else f"_{args.photon_mode}"
    out_path = os.path.join(CKPT_DIR, f"benchmark_results_v2{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved benchmark results to {out_path}")

    if "baseline" in results and "photon" in results:
        for regime in ["prefill_heavy", "decode_heavy"]:
            b = results["baseline"][regime]
            p = results["photon"][regime]
            speedup = p["throughput_tok_per_s"] / b["throughput_tok_per_s"]
            kv_ratio = b["kv_cache_gib"] / p["kv_cache_gib"] if p["kv_cache_gib"] > 0 else float("inf")
            tpm_ratio = (p["tpm_k_tok_per_s_per_gib"] / b["tpm_k_tok_per_s_per_gib"]
                         if b["tpm_k_tok_per_s_per_gib"] else float("inf"))
            print(f"\n[{regime}] PHOTON throughput: {speedup:.2f}x, "
                  f"KV memory reduction: {kv_ratio:.1f}x, TPM gain: {tpm_ratio:.1f}x")


if __name__ == "__main__":
    main()
