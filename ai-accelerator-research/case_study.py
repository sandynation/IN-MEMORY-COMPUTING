#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from backend.accel_tool.arch import ArchitectureSpec
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.verilog.synthesis_validator import SynthesisValidator
from backend.accel_tool.verilog.area_timing_estimator import AreaTimingEstimator

# 1. Workload
workload = {
    "name": "gemm_128",
    "m": 128,
    "n": 128,
    "k": 128,
    "dtype": "f16",
    "sparsity": 0.2,
    "target": "tensor_core"
}
print("Workload:", workload)

# 2. Generate Verilog
arch = ArchitectureSpec()
files = generate_verilog(workload, arch)
print("Generated Verilog files:", list(files.keys()))

# 3. Synthesis validation
validator = SynthesisValidator()
report = validator.validate_synthesis(files)
print("Synthesis passed:", report["passed"])
print("Errors:", report["errors"])
print("Warnings (first 3):", report["warnings"][:3])

# 4. Area/timing estimation
estimator = AreaTimingEstimator()
area_report = estimator.estimate_post_synthesis(files, target_mhz=500.0)
print("Area (µm²):", area_report["area_um2"])
print("Max freq (MHz):", area_report["max_freq_mhz"])
print("Cell count:", area_report["cell_count"])

# Save results for the paper
with open("case_study_results.json", "w") as f:
    json.dump({
        "workload": workload,
        "synthesis_passed": report["passed"],
        "errors": report["errors"],
        "warnings_count": len(report["warnings"]),
        "area_um2": area_report["area_um2"],
        "max_freq_mhz": area_report["max_freq_mhz"],
        "cell_count": area_report["cell_count"]
    }, f, indent=2)
print("\nResults saved to case_study_results.json")