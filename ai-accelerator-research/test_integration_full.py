#!/usr/bin/env python3
"""
End‑to‑end integration test: causal discovery → Verilog generation → Yosys synthesis.
Uses a tiny synthetic dataset and a small GEMM configuration.
"""
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.novelty_features.ncpg_full import NCPG
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec
from backend.accel_tool.verilog.synthesis_validator import SynthesisValidator

def main():
    print("=== Full Integration Test ===")

    # 1. Causal discovery
    print("Step 1: Training NCPG on tiny dataset...")
    torch.manual_seed(42)
    X = torch.randn(100, 3)
    y = X[:,0] + X[:,1] + 0.1 * torch.randn(100)
    y = y.unsqueeze(1)
    model = NCPG(num_nodes=3, output_dim=1, use_mlp=False, device="cpu")
    model.fit(X, y, num_epochs=50, verbose=False)
    print("  Training completed.")

    # 2. Verilog generation
    print("Step 2: Generating Verilog for small GEMM...")
    arch = ArchitectureSpec()
    workload = {"name": "test_gemm", "m": 16, "n": 16, "k": 16,
                "dtype": "f16", "sparsity": 0.0, "target": "tensor_core"}
    files = generate_verilog(workload, arch)
    print(f"  Generated {len(files)} files: {list(files.keys())}")

    # 3. Synthesis validation
    print("Step 3: Running Yosys synthesis validation...")
    validator = SynthesisValidator()
    report = validator.validate_synthesis(files)
    if report["passed"]:
        print("  OK: Synthesis passed (no errors).")
    else:
        print("  FAIL: Synthesis failed:", report["errors"])
        sys.exit(1)

    print("\nIntegration test PASSED – full pipeline works.")

if __name__ == "__main__":
    main()
