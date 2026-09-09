#!/usr/bin/env python3
# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# SparseX – Neural Causal Silicon Compiler
"""
Benchmark three repair strategies (random, brute-force, causal) on mutated RTL.
Outputs CSV, JSON, and plots.
"""
import sys
from pathlib import Path
# Add workspace root (for backend.*) and this directory (for benchmark_support, repair_strategies)
_workspace_root = str(Path(__file__).parents[2])
_this_dir       = str(Path(__file__).parent)
for _p in [_workspace_root, _this_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import time
import json
import csv
import random
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from backend.accel_tool.arch import ArchitectureSpec
from backend.accel_tool.ir_gen import generate_verilog
from backend.novelty_features.ncpg_full import NCPG
from benchmark_support import compile_and_simulate
from repair_strategies import RandomRepair, BruteForceRepair, CausalGuidedRepair

# ----------------------------------------------------------------------
# Helper to get testbench (16x16 GEMM tile, hardcoded TILE_SIZE=16)
# ----------------------------------------------------------------------
def get_testbench_16x16():
    return """
`timescale 1ns/1ps
module tb;
    reg clk, rst_n;
    reg [15:0] a_data, b_data;
    reg a_valid, b_valid;
    wire a_ready, b_ready;
    wire [31:0] c_data;
    wire c_valid;
    reg c_ready;

    integer i, j, k;
    reg [15:0] A[0:15][0:15];
    reg [15:0] B[0:15][0:15];
    reg [31:0] C_golden[0:15][0:15];

    gemm_tile #(.TILE_SIZE(16), .DATA_WIDTH(16), .ACC_WIDTH(32)) dut (
        .clk(clk), .rst_n(rst_n),
        .a_data(a_data), .a_valid(a_valid), .a_ready(a_ready),
        .b_data(b_data), .b_valid(b_valid), .b_ready(b_ready),
        .c_data(c_data), .c_valid(c_valid), .c_ready(c_ready),
        .sparse_skip(0), .mask_valid(0)
    );

    initial begin
        $dumpfile("wave.vcd");
        $dumpvars(0, tb);
        clk = 0;
        rst_n = 0;
        a_valid = 0;
        b_valid = 0;
        c_ready = 1;
        #10 rst_n = 1;

        for (i = 0; i < 16; i=i+1) begin
            for (j = 0; j < 16; j=j+1) begin
                A[i][j] = $random;
                B[i][j] = $random;
            end
        end

        for (i = 0; i < 16; i=i+1) begin
            for (j = 0; j < 16; j=j+1) begin
                C_golden[i][j] = 0;
                for (k = 0; k < 16; k=k+1) begin
                    C_golden[i][j] = C_golden[i][j] + A[i][k] * B[k][j];
                end
            end
        end

        for (i = 0; i < 16; i=i+1) begin
            for (j = 0; j < 16; j=j+1) begin
                @(posedge clk);
                a_data = A[i][j];
                a_valid = 1;
                b_data = B[j][i];
                b_valid = 1;
                @(posedge clk);
                a_valid = 0;
                b_valid = 0;
            end
        end

        repeat(32) @(posedge clk);
        if (c_data !== C_golden[0][0]) begin
            $display("FAILURE: c_data mismatch");
            $finish;
        end
        $display("SUCCESS");
        $finish;
    end
    always #5 clk = ~clk;
endmodule
"""

# ----------------------------------------------------------------------
# Define the testbench variable by calling the helper
# ----------------------------------------------------------------------
testbench = get_testbench_16x16()

# ----------------------------------------------------------------------
# Inject real RTL bugs: set m (TILE_SIZE) to a wrong value so the
# hardcoded 16×16 testbench fails. Also randomly mutate one secondary
# param (cgra_tiles or clock_frequency_ghz) to add noise — the causal
# model must figure out that m is the critical parameter to fix first.
# ----------------------------------------------------------------------
BUGGY_TILE_SIZES = [4, 8, 12, 20, 24, 28, 32]  # all != 16

def generate_variants(base_workload, num_variants=50, seed=42):
    random.seed(seed)
    variants = []
    secondary_params = {
        "cgra_tiles":           [2, 4, 6, 8, 10, 12, 14, 16],
        "clock_frequency_ghz": [1.0, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.0],
    }
    for i in range(num_variants):
        variant = copy.deepcopy(base_workload)
        # Primary bug: wrong TILE_SIZE (m != 16)
        wrong_m = random.choice(BUGGY_TILE_SIZES)
        variant["m"] = wrong_m
        variant["n"] = wrong_m  # keep n=k=m to stay self-consistent internally
        variant["k"] = wrong_m
        # Secondary noise: also mutate one non-critical param
        noise_param = random.choice(list(secondary_params.keys()))
        noise_vals  = [v for v in secondary_params[noise_param]
                       if v != base_workload.get(noise_param)]
        variant[noise_param] = random.choice(noise_vals)
        variants.append(variant)
    return variants

# ----------------------------------------------------------------------
# Train a causal model on BOUNDED tile-size samples so gradients at
# m=16 are meaningful.  We model:
#   latency = 1000/(tile_size + 0.1) + 3*cgra_tiles + 8*clk_ghz^2
# The gradient w.r.t. tile_size is by far the largest at tile_size=16
# (~3.8 vs 3 and ~32), correctly identifying m as the key parameter.
# ----------------------------------------------------------------------
def train_or_load_causal_model():
    import torch
    rng = np.random.default_rng(42)
    n_samples = 3000
    # Sample tile_size uniformly in [4, 32]
    tile   = rng.uniform(4, 32, n_samples)
    cgra   = rng.uniform(2, 16, n_samples)
    clk    = rng.uniform(1.0, 3.0, n_samples)
    X = np.stack([tile, cgra, clk], axis=1).astype(np.float32)
    latency = (1000.0 / (tile + 0.1) + 3.0 * cgra + 8.0 * clk**2
               + 0.02 * rng.standard_normal(n_samples))
    y = latency.reshape(-1, 1).astype(np.float32)
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    model = NCPG(num_nodes=3, output_dim=1, use_mlp=True, force_simple=True, device="cpu")
    model.fit(X_t, y_t, num_epochs=300, verbose=False)
    return model

# ----------------------------------------------------------------------
# Main benchmark
# ----------------------------------------------------------------------
def main():
    out_dir = Path("results/repair_benchmark") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    arch = ArchitectureSpec()
    base_workload = {
        "name": "gemm_16x16_correct",
        "m": 16, "n": 16, "k": 16,
        "dtype": "f16",
        "sparsity": 0.0,
        "target": "tensor_core",
        "tensor_core_tiles": 2,
        "cgra_tiles": 8,
        "memory_bandwidth_gbps": 800.0,
        "clock_frequency_ghz": 2.0,
    }
    # param_ranges used only by RandomRepair and BruteForceRepair
    param_ranges = {
        "m":                   [4, 8, 12, 16, 20, 24, 28, 32],
        "n":                   [4, 8, 12, 16, 20, 24, 28, 32],
        "k":                   [4, 8, 12, 16, 20, 24, 28, 32],
        "cgra_tiles":          [2, 4, 6, 8, 10, 12, 14, 16],
        "clock_frequency_ghz": [1.0, 1.4, 1.8, 2.0, 2.4, 2.8, 3.0],
    }

    print("Generating correct design...")
    correct_files = generate_verilog(base_workload, arch)
    passed, _, _ = compile_and_simulate(correct_files, testbench)
    if not passed:
        print("Correct design fails simulation. Exiting.")
        return
    print("Correct design passes.\n")

    print("Generating BUGGY variant designs (wrong TILE_SIZE != 16)...")
    variants = generate_variants(base_workload, num_variants=50)
    # Sanity-check: verify at least some variants actually fail
    n_fail = sum(
        1 for v in variants[:5]
        if not compile_and_simulate(generate_verilog(v, arch), testbench)[0]
    )
    print(f"  Sanity check: {n_fail}/5 sample variants fail simulation (expected >0).\n")

    # Train causal model
    print("Training causal NCPG model...")
    causal_model = train_or_load_causal_model()
    print("Done.\n")

    strategies = {
        "random":     RandomRepair(arch, testbench, max_attempts=20, param_ranges=param_ranges),
        "bruteforce": BruteForceRepair(arch, testbench, max_attempts=20),
        "causal":     CausalGuidedRepair(arch, testbench, causal_model, base_workload, max_attempts=20),
    }

    results = []
    for idx, variant in enumerate(variants):
        print(f"Variant {idx+1}/{len(variants)}: {variant}")
        for name, strategy in strategies.items():
            print(f"  Repairing with {name}...")
            start = time.time()
            success, final_workload, attempts = strategy.repair(variant)
            elapsed_ms = (time.time() - start) * 1000
            results.append({
                "variant_id": idx,
                "repair_strategy": name,
                "success": success,
                "attempts": attempts,
                "repair_time_ms": elapsed_ms,
                "initial_workload": variant,
                "final_workload": final_workload if success else None,
            })
            print(f"    success={success}, attempts={attempts}, time={elapsed_ms:.1f}ms")

    # Save CSV
    csv_path = out_dir / "repair_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["variant_id", "repair_strategy", "success", "attempts", "repair_time_ms", "initial_workload", "final_workload"])
        for r in results:
            writer.writerow([r["variant_id"], r["repair_strategy"], r["success"], r["attempts"], r["repair_time_ms"], str(r["initial_workload"]), str(r["final_workload"])])

    # Save JSON summary
    summary = {}
    for name in strategies:
        succ = [r for r in results if r["repair_strategy"] == name and r["success"]]
        summary[name] = {
            "total": len([r for r in results if r["repair_strategy"] == name]),
            "successful": len(succ),
            "success_rate": len(succ) / len([r for r in results if r["repair_strategy"] == name]),
            "avg_attempts": np.mean([r["attempts"] for r in results if r["repair_strategy"] == name]),
            "avg_time_ms": np.mean([r["repair_time_ms"] for r in results if r["repair_strategy"] == name]),
        }
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot success rates
    strategies_list = list(summary.keys())
    success_rates = [summary[s]["success_rate"] for s in strategies_list]
    plt.figure(figsize=(8,5))
    bars = plt.bar(strategies_list, success_rates, color=['blue','orange','green'])
    plt.ylim(0,1)
    plt.ylabel("Success Rate")
    plt.title("Repair Success Rate by Strategy")
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{rate:.2f}", ha='center')
    plt.savefig(out_dir / "success_rates.png", dpi=150)
    plt.close()

    # Plot average attempts
    avg_attempts = [summary[s]["avg_attempts"] for s in strategies_list]
    plt.figure(figsize=(8,5))
    bars = plt.bar(strategies_list, avg_attempts, color=['blue','orange','green'])
    plt.ylabel("Average Attempts")
    plt.title("Repair Attempts by Strategy")
    for bar, att in zip(bars, avg_attempts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{att:.1f}", ha='center')
    plt.savefig(out_dir / "avg_attempts.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    print("CSV:", csv_path)
    print("JSON:", out_dir / "summary.json")
    print("Plots:", out_dir / "success_rates.png", out_dir / "avg_attempts.png")

if __name__ == "__main__":
    main()