"""Preflight checks for the RunPod GPU run. Verifies GPU/CUDA/VRAM/disk/packages and the
frozen dataset bundle integrity (counts, fixed 13-image holdout, no group leakage, checksums).
Run BEFORE training. Exits non-zero on any hard failure."""
import os, sys, json, shutil, hashlib
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
FROZEN13 = ["0016","0023","0031","0032","0037","0044","0051","0054","0058","0064","0085","0120","0138"]

def main(data="dataset_bundle"):
    ok = True
    print("=== packages ===")
    import torch, transformers
    print("python", sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__)
    try:
        import onnx, onnxruntime
        print("onnx", onnx.__version__, "onnxruntime", onnxruntime.__version__)
    except Exception as e:
        print("WARN onnx/onnxruntime:", e)

    print("=== GPU / CUDA ===")
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        total = torch.cuda.get_device_properties(0).total_memory/1e9
        print(f"VRAM total: {total:.1f} GB")
        print("torch CUDA build:", torch.version.cuda)
        x = torch.randn(1024,1024, device="cuda"); (x@x).sum().item()
        print("cuda matmul OK")
        if total < 20: print("WARN: <20GB VRAM")
    else:
        print("WARN: no CUDA — will run on CPU (slow).")

    print("=== disk ===")
    tot, used, free = shutil.disk_usage(".")
    print(f"disk free: {free/1e9:.1f} GB (total {tot/1e9:.1f} GB)")
    if free < 5e9: print("WARN: <5GB free"); ok = False

    print("=== dataset bundle integrity ===")
    dd = Path(data)
    split = json.loads((dd/"split.json").read_text())
    manifest = json.loads((dd/"manifest.json").read_text())
    hold, pool = split["holdout"], split["pool"]
    print("holdout:", len(hold), "pool:", len(pool), "excluded_group_overlap:", len(split.get("excluded_group_overlap",[])))
    if sorted(hold) != sorted(FROZEN13):
        print("FAIL: holdout != fixed 13"); ok = False
    else:
        print("holdout == fixed 13: OK")
    # group leakage
    hg = split["house_groups"]; hold_g = set(hg[i] for i in hold)
    leak = [i for i in pool if hg.get(i) in hold_g]
    print("train↔holdout group leakage:", leak or "NONE")
    if leak: ok = False
    # checksums
    bad = []
    for rel, h in manifest["files"].items():
        if sha(dd/rel) != h: bad.append(rel)
    print("checksum mismatches:", bad or "NONE (all verified)")
    if bad: ok = False

    print("=== code version ===")
    os.system("git rev-parse HEAD 2>/dev/null || echo 'no git'")
    print("\nPREFLIGHT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset_bundle")
