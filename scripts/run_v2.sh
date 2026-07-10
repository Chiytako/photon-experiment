#!/bin/bash
# V2 experiment pipeline: retrain PHOTON with the paper-faithful architecture
# (recursive top-down cascade, meta-context generation) and re-evaluate /
# re-benchmark everything with the corrected protocols. Baselines are NOT
# retrained (their architecture is unchanged); existing baseline_final.pt /
# baseline_med.pt are reused.
#
# Run inside tmux on the GB10:
#   tmux new -d -s photon2 'bash scripts/run_v2.sh 2>&1 | tee -a v2_run.log'
set -e
cd ~/photon-experiment
source .venv/bin/activate

COMMON_SMALL="--arch photon --compile --total_tokens 200000000 --seq_len 1024 \
  --batch_size 16 --warmup_steps 300 --log_every 100 --eval_every 1000"

stamp() { date "+%F %T"; }

echo "[$(stamp)] === V2 RUN 1/3: photon2 (small, alpha=0) ==="
python train.py --run_name photon2 $COMMON_SMALL
python evaluate.py --ckpt checkpoints/photon2.pt

echo "[$(stamp)] === V2 RUN 2/3: photon2_a03 (small, alpha=0.3, paper B.1 best) ==="
python train.py --run_name photon2_a03 --alpha 0.3 $COMMON_SMALL
python evaluate.py --ckpt checkpoints/photon2_a03.pt

echo "[$(stamp)] === V2 RUN 3/3: photon2_med (medium, data2b, alpha=0) ==="
python train.py --arch photon --run_name photon2_med --compile --preset medium \
  --data_dir data2b --total_tokens 1500000000 --seq_len 1024 --batch_size 16 \
  --warmup_steps 300 --log_every 100 --eval_every 2000
python evaluate.py --ckpt checkpoints/photon2_med.pt

echo "[$(stamp)] === BENCHMARKS (paper TPM protocol) ==="
python benchmark.py --baseline_ckpt checkpoints/baseline_final.pt \
  --photon_ckpt checkpoints/photon2.pt --photon_mode hiergen --out_suffix ""
python benchmark.py --baseline_ckpt checkpoints/baseline_final.pt \
  --photon_ckpt checkpoints/photon2_a03.pt --photon_mode recgen --out_suffix "_recgen"
python benchmark.py --baseline_ckpt checkpoints/baseline_med.pt \
  --photon_ckpt checkpoints/photon2_med.pt --photon_mode hiergen --out_suffix "_med"

echo "[$(stamp)] === RECGEN DIAGNOSTICS (alpha=0.3 checkpoint) ==="
python scripts/recgen_diag.py --ckpt checkpoints/photon2_a03.pt

touch checkpoints/RUN_V2_DONE
echo "[$(stamp)] === V2 PIPELINE DONE ==="
