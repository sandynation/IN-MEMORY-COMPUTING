# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
#!/usr/bin/env python3
"""
Generate the correct 16×16 GEMM design that serves as the seed for broken variants.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec

def get_correct_workload():
    return {
        "name": "gemm_16x16_correct",
        "m": 16,
        "n": 16,
        "k": 16,
        "dtype": "f16",
        "sparsity": 0.0,
        "target": "tensor_core",
        "tensor_core_tiles": 2,
        "cgra_tiles": 8,
        "memory_bandwidth_gbps": 800.0,
        "clock_frequency_ghz": 2.0,
    }

def generate_correct_design():
    """Generate Verilog for the correct design and return the files dict."""
    arch = ArchitectureSpec()
    workload = get_correct_workload()
    files = generate_verilog(workload, arch)
    return files

if __name__ == "__main__":
    files = generate_correct_design()
    print(f"Generated {len(files)} Verilog files.")