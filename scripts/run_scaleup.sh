#!/bin/bash
# Scale-up runs: medium preset (~187M baseline / ~158M PHOTON) on the 2.2B-token
# FineWeb-Edu set (data2b/). TOTAL_TOKENS is set from the calibration measurement
# so that both runs together fit the agreed wall-clock budget.
set -e
cd ~/photon-experiment
source .venv/bin/activate

TOTAL_TOKENS=${TOTAL_TOKENS:-1500000000}
PHOTON_ALPHA=${PHOTON_ALPHA:-0.0}   # set from the ablation sweep's best alpha

COMMON="--compile --preset medium --data_dir data2b --total_tokens $TOTAL_TOKENS \
  --seq_len 1024 --batch_size 16 --warmup_steps 300 --log_every 100 --eval_every 2000"

echo "=== SCALE-UP: baseline_med (total_tokens=$TOTAL_TOKENS) ==="
python train.py --arch baseline --run_name baseline_med $COMMON

echo "=== SCALE-UP: photon_med (alpha=$PHOTON_ALPHA) ==="
python train.py --arch photon --run_name photon_med --alpha "$PHOTON_ALPHA" $COMMON

echo "=== SCALE-UP DONE ==="
