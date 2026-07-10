#!/bin/bash
set -e
cd ~/photon-experiment
source .venv/bin/activate
echo "=== Waiting for baseline recompile-train (already launched separately) ==="
while pgrep -f "train.py --arch baseline --compile" > /dev/null; do sleep 5; done
echo "=== baseline done, training photon with --compile ==="
python train.py --arch photon --run_name photon --compile --total_tokens 200000000 --seq_len 1024 --batch_size 16 \
  --warmup_steps 300 --log_every 50 --eval_every 500
echo "=== ALL RECOMPILE TRAINING DONE ==="
