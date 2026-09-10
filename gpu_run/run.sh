#!/usr/bin/env bash
# Turnkey RunPod runner for the SegFormer-B0 + B1 @768 real training run.
# Assumes: this gpu_run/ folder is present AND dataset_bundle/ is present alongside it.
# Container should already have torch 2.5.1 + CUDA 12.1 (see RUNPOD_EXECUTION_PLAN.md).
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/4] install deps"
pip install --no-cache-dir -r requirements-gpu.txt

echo "==> [2/4] preflight (GPU + dataset integrity)"
python preflight.py dataset_bundle

echo "==> [3/4] train B0 + B1 @768 (18 epochs each, bs=4, AdamW 6e-5, weighted CE [1,3], seed 0)"
python train_gpu.py --data dataset_bundle --out run_outputs --models b0,b1 \
  --res 768 --epochs 18 --bs 4 --lr 6e-5 --device auto 2>&1 | tee run_outputs_train.log

echo "==> [4/4] collect + checksum artifacts"
bash collect_and_checksum.sh

echo "ALL DONE. Review run_outputs/ then download BEFORE terminating the Pod."
