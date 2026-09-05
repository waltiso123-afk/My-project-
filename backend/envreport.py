"""Honest, real hardware/software environment detection. No fabrication."""
import os
import shutil
import platform
import subprocess
from datetime import datetime, timezone


def _torch_info():
    try:
        import torch
        info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_compiled_version": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [],
        }
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
                })
        return info
    except Exception as e:
        return {"installed": False, "version": None, "cuda_available": False,
                "device_count": 0, "devices": [], "error": str(e)}


def _nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            gpus = []
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                gpus.append({"name": parts[0], "vram_mb": parts[1], "driver": parts[2]})
            return {"present": True, "gpus": gpus}
        return {"present": False, "gpus": []}
    except Exception:
        return {"present": False, "gpus": []}


def _mem():
    try:
        with open("/proc/meminfo") as f:
            lines = {l.split(":")[0]: l.split(":")[1].strip() for l in f}
        total_kb = int(lines["MemTotal"].split()[0])
        avail_kb = int(lines["MemAvailable"].split()[0])
        return {"total_gb": round(total_kb / 1048576, 2),
                "available_gb": round(avail_kb / 1048576, 2)}
    except Exception:
        return {"total_gb": None, "available_gb": None}


def build_report() -> dict:
    torch_info = _torch_info()
    smi = _nvidia_smi()
    du = shutil.disk_usage("/app")
    gpu_present = bool(torch_info.get("cuda_available")) or smi.get("present", False)

    if gpu_present:
        verdict = "GPU_AVAILABLE"
        message = ("GPU CUDA detected. SAM2 + SegFormer heavy inference will run on GPU. "
                   "No automatic CPU fallback for heavy inference.")
        heavy_inference_backend = "cuda"
    else:
        verdict = "NO_GPU_CPU_ONLY"
        message = ("No CUDA GPU detected in this build environment. Pipeline is CUDA-ready and will "
                   "use the GPU automatically when available. CPU is used only for light ops/tests. "
                   "Heavy SAM2 batch inference over the 217 images is NOT run on CPU. GPU is required "
                   "and clearly flagged for heavy inference.")
        heavy_inference_backend = "cpu_light_ops_only"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "message": message,
        "heavy_inference_backend": heavy_inference_backend,
        "gpu_present": gpu_present,
        "torch": torch_info,
        "nvidia_smi": smi,
        "cpu": {
            "processor": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "memory": _mem(),
        "disk_app": {
            "total_gb": round(du.total / (1024 ** 3), 2),
            "used_gb": round(du.used / (1024 ** 3), 2),
            "free_gb": round(du.free / (1024 ** 3), 2),
        },
    }
