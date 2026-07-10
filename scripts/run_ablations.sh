#!/bin/bash
# Ablation sweep: 5 small-PHOTON training runs (200M tokens each) + evaluation
# after each. Reference point is the alpha=0, C=(4,4), R=(4,4) small run
# (produced separately, e.g. checkpoints/photon2.pt from scripts/run_v2.sh).
set -e
cd ~/photon-experiment
source .venv/bin/activate

COMMON="--arch photon --compile --total_tokens 200000000 --seq_len 1024 --batch_size 16 --warmup_steps 300 --log_every 100 --eval_every 1000"

run () {
  name=$1; shift
  echo "=== ABLATION RUN: $name ($*) ==="
  python train.py --run_name "$name" $COMMON "$@"
  python evaluate.py --ckpt "checkpoints/$name.pt"
}

run ph_a01 --alpha 0.1
run ph_a02 --alpha 0.2
run ph_a03 --alpha 0.3
run ph_c22 --chunk_sizes 2,2 --prefix_lens 2,2
run ph_r88 --prefix_lens 8,8

echo "=== ABLATION SWEEP DONE ==="
