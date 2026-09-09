#!/usr/bin/env python3
# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# SparseX – Neural Causal Silicon Compiler
"""
Run all benchmarks and generate a report for the paper.
"""
import subprocess
import sys
import json

def run(cmd, description):
    print(f"\n=== {description} ===")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode

def main():
    print("SparseX – Full Evaluation Suite")
    print("Copyright (c) 2026 Sandipan Pal. All rights reserved.\n")

    # 1. Causal discovery on synthetic data
    run("python benchmark_causal_fixed.py", "Causal Discovery (Synthetic)")

    # 2. AutoMPG real-world dataset
    run("python benchmark_autompg.py", "AutoMPG Causal Discovery")

    # 3. Physics-hard baseline comparison (GES, PC, NCPG)
    run("python compare_physics_hard.py", "Physics‑Hard Baseline Comparison")

    # 4. Ablation: geometric loss
    run("python ablation_causal_vs_noncausal.py", "Ablation: Geometric Loss")

    # 5. Verilog generation & synthesis pass rate
    run("python synthesis_pass_rate.py", "Synthesis Pass Rate (100 random GEMMs)")

    # 6. Functional correctness (16×16 GEMM)
    run("python test_functional_correctness.py", "Functional Correctness (16×16 GEMM)")

    # 7. Transfer learning speedup
    run("python backend/novelty_features/transfer_efficiency.py", "Transfer Learning Speedup")

    # 8. Scalability
    run("python benchmark_causal_fixed.py --scalability-only", "Scalability (3–20 inputs)")

    print("\nAll benchmarks completed. Please check the outputs for final numbers.")
    print("For the paper, copy the relevant numbers into the tables.")

if __name__ == "__main__":
    main()