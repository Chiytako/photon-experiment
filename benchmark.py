"""Inference throughput / peak-memory benchmark for baseline vs PHOTON,
mirroring the paper's own regimes: prefill-heavy (long prompt, short
generation) and decode-heavy (short prompt, long generation)."""
import argparse
import json
import os
import time

import torch

from evaluate import load_model, CKPT_DIR


@torch.no_grad()
def run_once(model, arch, prompt_len, gen_len, batch_size, vocab_size, total_c=1,
             device="cuda", photon_mode="hiergen"):
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
    peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    throughput = (batch_size * n_generated) / dt
    return {
        "prompt_len": pl, "gen_len": n_generated, "batch_size": batch_size,
        "wall_time_s": dt, "throughput_tok_per_s": throughput,
        "peak_mem_gib": peak_mem_gib,
        "tpm_k_tok_per_s_per_gib": (throughput / 1000) / peak_mem_gib if peak_mem_gib > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", default=os.path.join(CKPT_DIR, "baseline_final.pt"))
    ap.add_argument("--photon_ckpt", default=os.path.join(CKPT_DIR, "photon_final.pt"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--prefill_prompt", type=int, default=2048)
    ap.add_argument("--prefill_gen", type=int, default=128)
    ap.add_argument("--decode_prompt", type=int, default=128)
    ap.add_argument("--decode_gen", type=int, default=2048)
    ap.add_argument("--photon_mode", choices=["hiergen", "recgen"], default="hiergen",
                     help="Which PHOTON decoding mode to benchmark. RecGen skips "
                          "bottom-up re-encoding and needs an alpha>0-trained "
                          "checkpoint to be meaningful quality-wise.")
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
        prefill_res = run_once(model, arch, args.prefill_prompt, args.prefill_gen,
                                args.batch_size, vocab_size, total_c, device,
                                photon_mode=args.photon_mode)
        print(json.dumps(prefill_res, indent=2))

        print(f"[{arch}] decode-heavy ({args.decode_prompt} in / {args.decode_gen} out)")
        decode_res = run_once(model, arch, args.decode_prompt, args.decode_gen,
                               args.batch_size, vocab_size, total_c, device,
                               photon_mode=args.photon_mode)
        print(json.dumps(decode_res, indent=2))

        results[arch] = {"prefill_heavy": prefill_res, "decode_heavy": decode_res}
        del model
        torch.cuda.empty_cache()

    suffix = "" if args.photon_mode == "hiergen" else f"_{args.photon_mode}"
    out_path = os.path.join(CKPT_DIR, f"benchmark_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved benchmark results to {out_path}")

    if "baseline" in results and "photon" in results:
        for regime in ["prefill_heavy", "decode_heavy"]:
            b = results["baseline"][regime]
            p = results["photon"][regime]
            speedup = p["throughput_tok_per_s"] / b["throughput_tok_per_s"]
            mem_ratio = b["peak_mem_gib"] / p["peak_mem_gib"] if p["peak_mem_gib"] > 0 else float("inf")
            print(f"\n[{regime}] PHOTON speedup: {speedup:.2f}x, "
                  f"baseline/PHOTON memory ratio: {mem_ratio:.2f}x")


if __name__ == "__main__":
    main()
