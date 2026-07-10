#!/bin/bash
set -e
cd ~/photon-experiment
source .venv/bin/activate
echo "=== Training baseline ==="
python train.py --arch baseline --run_name baseline --total_tokens 200000000 --seq_len 1024 --batch_size 16 \
  --warmup_steps 300 --log_every 50 --eval_every 500
echo "=== Training photon ==="
python train.py --arch photon --run_name photon --total_tokens 200000000 --seq_len 1024 --batch_size 16 \
  --warmup_steps 300 --log_every 50 --eval_every 500
echo "=== ALL TRAINING DONE ==="
