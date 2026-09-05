#!/usr/bin/env python3
"""Print the REAL hardware/software environment report (GPU/CUDA/CPU/RAM/disk)."""
import json
import sys
sys.path.insert(0, "/app/backend")
import envreport  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(envreport.build_report(), indent=2))
