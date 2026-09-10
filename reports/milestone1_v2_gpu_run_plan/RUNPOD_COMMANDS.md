# RunPod — exact command sheet (holdout FIXED at 13, never 19)

Legend: [LOCAL] = your machine (has gpu_run/ + dataset_bundle/)  ·  [POD] = RunPod terminal.

## 0. One-time: install + auth runpodctl (both machines that need it)
```bash
# [LOCAL] and [POD]
wget -qO- cli.runpod.net | sudo bash
runpodctl config --apiKey <YOUR_RUNPOD_API_KEY>      # from RunPod console > Settings > API Keys
```

## 1. Create / configure the Pod (RTX 4090, Secure Cloud)
```bash
# [LOCAL] — verify exact flag names first: runpodctl create pod --help
runpodctl create pod \
  --name roofline-4090 \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --gpuCount 1 \
  --imageName "runpod/pytorch:2.5.1-py3.11-cuda12.1.0-devel-ubuntu22.04" \
  --containerDiskSize 30 \
  --volumeSize 20 \
  --volumePath /workspace \
  --ports "22/tcp" \
  --cloudType SECURE
# Then get the pod id + SSH:
runpodctl get pods
```
(Reliable alternative: RunPod **web console** → Deploy → Secure Cloud → RTX 4090 → template
"PyTorch 2.5.1 / CUDA 12.1" → Container Disk 30GB, Volume 20GB @/workspace → Deploy → open Web Terminal.)

## 2. Transfer gpu_run/ (scripts only, excluding the big bundle)
```bash
# [LOCAL]
tar --exclude='dataset_bundle' -czf gpu_run_code.tar.gz -C /app gpu_run
runpodctl send gpu_run_code.tar.gz          # prints a one-time CODE
```
```bash
# [POD]  (cd /workspace first if using the volume)
cd /workspace
runpodctl receive <CODE_FROM_LOCAL>
tar -xzf gpu_run_code.tar.gz                 # -> ./gpu_run/
```

## 3. Transfer dataset_bundle/ (573 MB)
```bash
# [LOCAL]
tar -czf dataset_bundle.tar.gz -C /app/gpu_run dataset_bundle
runpodctl send dataset_bundle.tar.gz         # prints a one-time CODE
```
```bash
# [POD]
cd /workspace
runpodctl receive <CODE_FROM_LOCAL>
tar -xzf dataset_bundle.tar.gz -C gpu_run/   # -> ./gpu_run/dataset_bundle/
```

## 4. Preflight (install deps + integrity/GPU checks)
```bash
# [POD]
cd /workspace/gpu_run
pip install --no-cache-dir -r requirements-gpu.txt
python preflight.py dataset_bundle           # must print PREFLIGHT: PASS  (holdout==13, no leak, checksums OK)
```

## 5. Training (both models, run inside tmux so an SSH drop can't kill it)
```bash
# [POD]
cd /workspace/gpu_run
tmux new -s train
python train_gpu.py --data dataset_bundle --out run_outputs \
  --models b0,b1 --res 768 --epochs 18 --bs 4 --lr 6e-5 --device auto 2>&1 | tee run_outputs_train.log
# detach: Ctrl-b then d   ·   reattach later: tmux attach -t train
```

## 6. Collect + checksum artifacts
```bash
# [POD]
cd /workspace/gpu_run
bash collect_and_checksum.sh                 # must print ARTIFACT CHECK: PASS ; builds run_outputs.tar.gz (+ .sha256)
runpodctl send run_outputs.tar.gz            # prints a CODE
runpodctl send run_outputs.tar.gz.sha256     # prints a CODE
```
```bash
# [LOCAL]
runpodctl receive <CODE_FOR_TARBALL>
runpodctl receive <CODE_FOR_SHA256>
```

## 7. Verify SHA locally (do this BEFORE terminating)
```bash
# [LOCAL] — both files in the same directory
sha256sum -c run_outputs.tar.gz.sha256       # -> run_outputs.tar.gz: OK
# then verify every inner artifact:
tar -xzf run_outputs.tar.gz
cd run_outputs && sha256sum -c SHA256SUMS.txt   # -> all files: OK
```

## 8. Terminate the Pod (ONLY after step 7 shows OK)
```bash
# [LOCAL]
runpodctl get pods                           # note the POD_ID
runpodctl remove pod <POD_ID>                # terminates (stops billing)
```
(Or web console → your pod → Terminate.)

---
Holdout is FIXED at 13 (baked into dataset_bundle/split.json, verified by preflight). Never switched to 19.
SSH/scp alternative to runpodctl: `scp -P <port> -r gpu_run root@<pod-ip>:/workspace/` etc.
