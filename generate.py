"""CLI demo: load a trained checkpoint (baseline or PHOTON/HierGen) and
generate text from a prompt."""
import argparse
import os

import torch
from transformers import AutoTokenizer

from evaluate import load_model, TOKENIZER_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--mode", choices=["hiergen", "recgen"], default="hiergen",
                     help="PHOTON decoding mode (ignored for baseline).")
    args = ap.parse_args()

    device = "cuda"
    model, arch, cfg, seq_len = load_model(args.ckpt, device)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
    if arch == "photon":
        # left-pad with EOS up to a full multiple of total_downsample, so the
        # model's internal alignment never has to drop prompt tokens
        total_c = cfg.total_downsample
        target_len = max(total_c, -(-len(ids) // total_c) * total_c)
        ids = [tok.eos_token_id] * (target_len - len(ids)) + ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    gen_fn = model.generate
    mode_str = ""
    if arch == "photon" and args.mode == "recgen":
        gen_fn = model.generate_recgen
        mode_str = " (RecGen)"
    print(f"=== {arch}{mode_str} generation ===")
    print(f"prompt: {args.prompt!r}")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = gen_fn(idx, max_new_tokens=args.max_new_tokens,
                     temperature=args.temperature, top_k=args.top_k)
    text = tok.decode(out[0].tolist(), skip_special_tokens=True)
    print("--- output ---")
    print(text)


if __name__ == "__main__":
    main()
