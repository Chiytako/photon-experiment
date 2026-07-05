"""Perplexity evaluation for trained checkpoints: on our own held-out
FineWeb-Edu val split, and on WikiText-103 (test split) for an
external/standard reference point."""
import argparse
import math
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model.baseline import BaselineLM, BaselineConfig
from model.photon import PhotonLM, PhotonConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"


def load_model(ckpt_path, device="cuda", return_meta=False):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    cfg = ckpt["cfg"]
    if arch == "baseline":
        model = BaselineLM(cfg)
    else:
        model = PhotonLM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    if return_meta:
        meta = {"run_name": ckpt.get("run_name", arch), "seq_len": ckpt["seq_len"]}
        return model, arch, cfg, ckpt["seq_len"], meta
    return model, arch, cfg, ckpt["seq_len"]


@torch.no_grad()
def perplexity_on_tokens(model, arch, tokens: np.ndarray, seq_len: int, device, batch_size=8, stride=None):
    """Non-overlapping (or strided) windows, loss averaged over all predicted
    tokens (token-count-weighted, standard PPL convention)."""
    if stride is None:
        stride = seq_len
    total_loss = 0.0
    total_count = 0
    n = len(tokens)
    starts = list(range(0, n - seq_len - 1, stride))
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i:i + batch_size]
        if arch == "baseline":
            chunk = np.stack([tokens[s:s + seq_len + 1] for s in batch_starts]).astype(np.int64)
            chunk = torch.from_numpy(chunk).to(device)
            x, y = chunk[:, :-1], chunk[:, 1:]
        else:
            chunk = np.stack([tokens[s:s + seq_len] for s in batch_starts]).astype(np.int64)
            chunk = torch.from_numpy(chunk).to(device)
            x, y = chunk, chunk
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        n_tok = x.numel()
        total_loss += loss.item() * n_tok
        total_count += n_tok
    avg_loss = total_loss / total_count
    return avg_loss, math.exp(min(avg_loss, 20))


def get_wikitext_tokens(tokenizer, max_tokens=2_000_000):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return np.array(ids[:max_tokens], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_bin", default=os.path.join(DATA_DIR, "val.bin"),
                     help="Held-out FineWeb bin to evaluate on. Models trained on "
                          "the 2.2B-token set must use data2b/val.bin -- the "
                          "original data/val.bin overlaps that training stream.")
    args = ap.parse_args()

    device = "cuda"
    model, arch, cfg, train_seq_len, meta = load_model(args.ckpt, device, return_meta=True)
    run_name = meta["run_name"]
    n_params = model.num_params()
    print(f"loaded {run_name} ({arch}) checkpoint ({n_params/1e6:.2f}M params)")

    seq_len = args.eval_seq_len
    if arch == "photon":
        total_c = cfg.total_downsample
        seq_len = (seq_len // total_c) * total_c

    val_tokens = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    val_loss, val_ppl = perplexity_on_tokens(model, arch, val_tokens, seq_len, device, args.batch_size)
    print(f"[{arch}] FineWeb-Edu val: loss={val_loss:.4f} ppl={val_ppl:.2f}")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    wt_tokens = get_wikitext_tokens(tok)
    wt_loss, wt_ppl = perplexity_on_tokens(model, arch, wt_tokens, seq_len, device, args.batch_size)
    print(f"[{arch}] WikiText-103 test: loss={wt_loss:.4f} ppl={wt_ppl:.2f}")

    result = {
        "arch": arch, "run_name": run_name, "n_params": n_params, "eval_seq_len": seq_len,
        "fineweb_val_loss": val_loss, "fineweb_val_ppl": val_ppl,
        "wikitext_loss": wt_loss, "wikitext_ppl": wt_ppl,
    }
    import json
    out_path = os.path.join(CKPT_DIR, f"{run_name}_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved eval results to {out_path}")


if __name__ == "__main__":
    main()
