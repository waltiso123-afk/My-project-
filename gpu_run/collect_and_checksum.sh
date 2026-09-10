#!/usr/bin/env bash
# Verify all required artifacts exist, then package + checksum them for safe download.
set -euo pipefail
cd "$(dirname "$0")"
OUT=run_outputs
fail=0
need() { if [ ! -e "$1" ]; then echo "MISSING: $1"; fail=1; else echo "OK: $1"; fi; }

echo "=== verify checkpoints ==="
for t in b0 b1; do need "$OUT/models/${t}_best.pt"; need "$OUT/models/${t}_last.pt"; done
echo "=== verify ONNX exports ==="
for t in b0 b1; do need "$OUT/onnx/segformer_${t}_768_binary.onnx"; done
echo "=== verify holdout logits/probability maps ==="
for t in b0 b1; do
  n=$(ls "$OUT/holdout_outputs/$t"/*_logits.npy 2>/dev/null | wc -l || echo 0)
  echo "$t: $n holdout logit files"; [ "$n" -ge 13 ] || { echo "WARN <13 logit files for $t"; fail=1; }
done
echo "=== verify reports/logs ==="
need "$OUT/run_summary.json"; need "$OUT/metrics_b0.json"; need "$OUT/metrics_b1.json"
need "run_outputs_train.log"

if [ "$fail" -ne 0 ]; then echo "ARTIFACT CHECK: FAIL — do NOT terminate the Pod"; exit 1; fi

echo "=== checksums ==="
find "$OUT" -type f \( -name '*.pt' -o -name '*.onnx' -o -name '*.json' -o -name '*.npy' \) -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS.txt"
echo "wrote $OUT/SHA256SUMS.txt"

echo "=== package ==="
tar -czf run_outputs.tar.gz "$OUT" run_outputs_train.log
sha256sum run_outputs.tar.gz | tee run_outputs.tar.gz.sha256
echo "ARTIFACT CHECK: PASS — download run_outputs.tar.gz, verify sha256, THEN terminate the Pod."
