"""Stream a slice of FineWeb-Edu, tokenize with the LLaMA tokenizer (vocab
32,000, matching the PHOTON paper), and write contiguous uint16 token arrays
(train.bin / val.bin) via numpy memmap, ready for fixed-length LM training."""
import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"


def window_size(seq_len: int, arch: str) -> int:
    """Raw-token window length needed to build one (input, target) pair.
    baseline needs seq_len+1 tokens (targets are inputs shifted by one);
    photon needs exactly seq_len (the causal shift is baked into the model's
    converter conditioning, so target == input)."""
    return seq_len + 1 if arch == "baseline" else seq_len


def split_input_target(chunk, arch: str):
    """chunk: (B, window_size) int64 token tensor. Returns (input, target).

    For `baseline` (a plain causal transformer, where position j's causal
    self-attention includes token j itself), inputs and targets must be
    offset by one position -- otherwise the model trivially "predicts" token
    j from token j via self-attention, collapsing loss to ~0 with no genuine
    learning. For `photon`, position j's reconstruction provably depends only
    on strictly-prior context (see tests/test_models.py), so target=input is
    correct there and shifting again would be wrong.

    This is the single source of truth for the shift rule, shared by
    train.py and evaluate.py."""
    if arch == "baseline":
        return chunk[:, :-1], chunk[:, 1:]
    return chunk, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--train_tokens", type=int, default=220_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--batch_texts", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    eos_id = tok.eos_token_id
    print(f"tokenizer vocab_size={tok.vocab_size}, eos_id={eos_id}")

    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)

    total_target = args.train_tokens + args.val_tokens
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    train_arr = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(args.train_tokens,))
    val_arr = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(args.val_tokens,))

    train_ptr = 0
    val_ptr = 0
    text_batch = []
    pbar = tqdm(total=total_target, unit="tok")

    def flush(texts):
        nonlocal train_ptr, val_ptr
        if not texts:
            return True
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            ids = ids + [eos_id]
            n = len(ids)
            if train_ptr < args.train_tokens:
                take = min(n, args.train_tokens - train_ptr)
                train_arr[train_ptr:train_ptr + take] = ids[:take]
                train_ptr += take
                pbar.update(take)
                ids = ids[take:]
            if ids and val_ptr < args.val_tokens:
                take = min(len(ids), args.val_tokens - val_ptr)
                val_arr[val_ptr:val_ptr + take] = ids[:take]
                val_ptr += take
                pbar.update(take)
            if train_ptr >= args.train_tokens and val_ptr >= args.val_tokens:
                return False
        return True

    keep_going = True
    for ex in ds:
        text_batch.append(ex["text"])
        if len(text_batch) >= args.batch_texts:
            keep_going = flush(text_batch)
            text_batch = []
            if not keep_going:
                break
    if keep_going and text_batch:
        flush(text_batch)

    pbar.close()
    train_arr.flush()
    val_arr.flush()
    print(f"train tokens written: {train_ptr} / {args.train_tokens}")
    print(f"val tokens written:   {val_ptr} / {args.val_tokens}")
    if train_ptr < args.train_tokens or val_ptr < args.val_tokens:
        print("WARNING: stream exhausted before reaching target token counts; "
              "trimming bin files to actual counts.")
        # rewrite trimmed files
        if train_ptr < args.train_tokens:
            trimmed = np.array(train_arr[:train_ptr])
            del train_arr
            np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(train_ptr,))[:] = trimmed
        if val_ptr < args.val_tokens:
            trimmed = np.array(val_arr[:val_ptr])
            del val_arr
            np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(val_ptr,))[:] = trimmed


if __name__ == "__main__":
    main()
    print("done.", flush=True)
    os._exit(0)  # HF datasets streaming leaves background threads that can
                 # crash the interpreter during normal shutdown; data is
                 # already flushed to disk by this point.
