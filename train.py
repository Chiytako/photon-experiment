"""Training script for BaselineLM / PhotonLM on tokenized FineWeb-Edu data
(produced by data.py). bf16 autocast, AdamW, cosine LR with warmup, gradient
clipping, periodic validation and checkpointing."""
import argparse
import math
import os
import time
import json

import numpy as np
import torch

from model.baseline import BaselineLM, BaselineConfig
from model.photon import PhotonLM, PhotonConfig, LevelConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def get_batch(arr, batch_size, seq_len, device, arch):
    """Returns (input, target).

    For `baseline` (a plain causal transformer, where position j's causal
    self-attention includes token j itself), inputs and targets must be
    offset by one position -- otherwise the model trivially "predicts" token
    j from token j via self-attention, collapsing loss to ~0 with no genuine
    learning. For `photon`, the shift is already baked into the decoder's
    own construction (verified by the causality unit tests: logits at
    position j provably do not depend on token j or later), so target=input
    is correct there and shifting again would be wrong.
    """
    if arch == "baseline":
        max_start = len(arr) - seq_len - 1
        ix = np.random.randint(0, max_start, size=batch_size)
        chunk = np.stack([arr[i:i + seq_len + 1].astype(np.int64) for i in ix])
        chunk = torch.from_numpy(chunk).to(device, non_blocking=True)
        return chunk[:, :-1], chunk[:, 1:]
    else:
        max_start = len(arr) - seq_len - 1
        ix = np.random.randint(0, max_start, size=batch_size)
        x = np.stack([arr[i:i + seq_len].astype(np.int64) for i in ix])
        x = torch.from_numpy(x).to(device, non_blocking=True)
        return x, x


def cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return max_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * min_lr_ratio + (max_lr - max_lr * min_lr_ratio) * coeff


# Preset shapes. "small" is the original ~36-44M pair from the first round of
# experiments; "medium" targets a ~150-190M matched pair for the scale-up run.
BASELINE_PRESETS = {
    "small": dict(dim=512, n_layers=8, n_heads=8, mlp_hidden=1536),
    "medium": dict(dim=1024, n_layers=12, n_heads=16, mlp_hidden=2816),
}
PHOTON_PRESETS = {
    # (d0, per-level [dim, enc_layers, heads, mlp] bottom→top)
    "small": dict(d0=256, level_dims=[384, 512], layers=3, heads=[6, 8],
                   mlps=[1024, 1536]),
    "medium": dict(d0=512, level_dims=[768, 1024], layers=4, heads=[12, 16],
                    mlps=[2048, 2816]),
}


def build_model(arch: str, seq_len: int, preset: str = "small",
                chunk_sizes=(4, 4), prefix_lens=(4, 4), alpha: float = 0.0):
    if arch == "baseline":
        p = BASELINE_PRESETS[preset]
        cfg = BaselineConfig(vocab_size=32000, max_seq_len=seq_len, **p)
        return BaselineLM(cfg), cfg
    elif arch == "photon":
        p = PHOTON_PRESETS[preset]
        assert len(chunk_sizes) == len(prefix_lens) == len(p["level_dims"])
        levels = [
            LevelConfig(chunk_size=chunk_sizes[i], dim=p["level_dims"][i],
                        prefix_len=prefix_lens[i],
                        enc_layers=p["layers"], enc_heads=p["heads"][i],
                        enc_mlp_hidden=p["mlps"][i],
                        dec_layers=p["layers"], dec_heads=p["heads"][i],
                        dec_mlp_hidden=p["mlps"][i])
            for i in range(len(p["level_dims"]))
        ]
        cfg = PhotonConfig(vocab_size=32000, d0=p["d0"], levels=levels,
                            max_seq_len=seq_len, recon_loss_weight=alpha)
        return PhotonLM(cfg), cfg
    else:
        raise ValueError(arch)


@torch.no_grad()
def evaluate(model, val_arr, batch_size, seq_len, device, arch, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val_arr, batch_size, seq_len, device, arch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["baseline", "photon"], required=True)
    ap.add_argument("--run_name", required=True,
                     help="Names the checkpoint ({run_name}.pt) and log "
                          "({run_name}_log.jsonl) under checkpoints/. Required so "
                          "every run writes to its own files; refuses to overwrite "
                          "an existing checkpoint unless --overwrite is given.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ckpt_dir", default=CKPT_DIR,
                     help="Where to write checkpoints/logs. Point smoke tests at "
                          "/tmp/photon_smoke so they can never touch real runs.")
    ap.add_argument("--data_dir", default=DATA_DIR,
                     help="Directory containing train.bin/val.bin. The scale-up "
                          "dataset lives in its own directory (data2b/) so it can "
                          "be generated while other runs read the original bins.")
    ap.add_argument("--preset", choices=["small", "medium"], default="small")
    ap.add_argument("--alpha", type=float, default=0.0,
                     help="Weight of PHOTON's recursive reconstruction loss.")
    ap.add_argument("--chunk_sizes", default="4,4",
                     help="PHOTON per-level chunk sizes C_l, comma-separated bottom-up.")
    ap.add_argument("--prefix_lens", default="4,4",
                     help="PHOTON per-level converter prefix lengths R_l.")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--total_tokens", type=int, default=200_000_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--compile", action="store_true",
                     help="Wrap the training forward pass in torch.compile. Measured "
                          "~2x step-time speedup for both archs on GB10, with logits "
                          "numerically matching eager mode. Checkpointing always saves "
                          "the original (uncompiled) module's state_dict, so compiled "
                          "and eager checkpoints are interchangeable.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    train_arr = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_arr = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    tokens_per_step = args.batch_size * args.seq_len
    max_steps = args.total_tokens // tokens_per_step
    print(f"tokens_per_step={tokens_per_step}, max_steps={max_steps}, "
          f"total_tokens~={max_steps * tokens_per_step}")

    chunk_sizes = tuple(int(c) for c in args.chunk_sizes.split(","))
    prefix_lens = tuple(int(r) for r in args.prefix_lens.split(","))
    model, cfg = build_model(args.arch, args.seq_len, preset=args.preset,
                              chunk_sizes=chunk_sizes, prefix_lens=prefix_lens,
                              alpha=args.alpha)
    model = model.to(device)
    train_model = torch.compile(model) if args.compile else model
    print(f"[{args.run_name}/{args.arch}] total params: {model.num_params()/1e6:.2f}M, "
          f"non-embedding: {model.num_params(True)/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.run_name}.pt")
    if os.path.exists(ckpt_path) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing checkpoint {ckpt_path} "
                         f"(pass --overwrite to allow)")
    log_path = os.path.join(args.ckpt_dir, f"{args.run_name}_log.jsonl")
    log_f = open(log_path, "w")

    t_start = time.time()
    running_loss = 0.0
    running_count = 0
    for step in range(max_steps):
        lr = cosine_lr(step, args.warmup_steps, max_steps, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_arr, args.batch_size, args.seq_len, device, args.arch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = train_model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        running_loss += loss.item()
        running_count += 1

        if step % args.log_every == 0 or step == max_steps - 1:
            elapsed = time.time() - t_start
            toks_per_sec = (step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            avg_loss = running_loss / max(1, running_count)
            print(f"[{args.run_name}] step {step}/{max_steps} loss={avg_loss:.4f} lr={lr:.2e} "
                  f"tok/s={toks_per_sec:.0f} elapsed={elapsed:.0f}s")
            log_f.write(json.dumps({"step": step, "train_loss": avg_loss, "lr": lr,
                                     "tok_per_s": toks_per_sec, "elapsed": elapsed}) + "\n")
            log_f.flush()
            running_loss = 0.0
            running_count = 0

        if (step > 0 and step % args.eval_every == 0) or step == max_steps - 1:
            val_loss = evaluate(train_model, val_arr, args.batch_size, args.seq_len, device, args.arch)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"[{args.run_name}] step {step} VAL loss={val_loss:.4f} ppl={val_ppl:.2f}")
            log_f.write(json.dumps({"step": step, "val_loss": val_loss, "val_ppl": val_ppl}) + "\n")
            log_f.flush()

    torch.save({"model_state": model.state_dict(), "cfg": cfg, "arch": args.arch,
                "seq_len": args.seq_len, "run_name": args.run_name,
                "train_args": vars(args)}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")
    log_f.close()


if __name__ == "__main__":
    main()
