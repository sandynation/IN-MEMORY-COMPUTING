#!/usr/bin/env python3
"""
Synthesis pass rate over 100 random GEMM workloads.
Generates Verilog, runs Yosys validation, reports pass rate and warnings.
"""
import random
import json
from pathlib import Path
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec
from backend.accel_tool.verilog.synthesis_validator import SynthesisValidator

def random_workload():
    return {
        "name": "rand_gemm",
        "m": random.choice([64, 128, 256, 512]),
        "n": random.choice([64, 128, 256, 512]),
        "k": random.choice([64, 128, 256, 512]),
        "dtype": random.choice(["f16", "f32"]),
        "sparsity": random.uniform(0, 0.5),
        "target": "tensor_core"
    }

def main():
    n = 100
    passed = 0
    total_warnings = 0
    arch = ArchitectureSpec()
    validator = SynthesisValidator()
    results = []

    print(f"Generating {n} random workloads and running Yosys synthesis...")
    for i in range(n):
        workload = random_workload()
        try:
            files = generate_verilog(workload, arch)
            report = validator.validate_synthesis(files)
            if report["passed"]:
                passed += 1
            total_warnings += len(report["warnings"])
            results.append({
                "workload": workload,
                "passed": report["passed"],
                "errors": report["errors"],
                "warnings": report["warnings"]
            })
        except Exception as e:
            print(f"  Run {i+1} failed: {e}")
            results.append({"workload": workload, "passed": False, "error": str(e)})
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{n} configurations")

    pass_rate = passed / n * 100
    avg_warnings = total_warnings / n
    print(f"\nSynthesis pass rate: {pass_rate:.1f}% ({passed}/{n})")
    print(f"Average warnings per design: {avg_warnings:.1f}")
    with open("synthesis_pass_rate.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to synthesis_pass_rate.json")

if __name__ == "__main__":
    main()