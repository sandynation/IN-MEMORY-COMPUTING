# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
#!/usr/bin/env python3
"""
Ablation study: compare repair success rates with/without NCPG and MCTS.
Uses the exact same repair functions as repair_benchmark.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import pandas as pd
import copy
from backend.accel_tool.arch import ArchitectureSpec
from backend.accel_tool.ir_gen import generate_verilog
from benchmark_support import compile_and_simulate, get_testbench
from generate_correct_design import get_correct_workload
from causal_repair import train_ncpg_on_design_space, causal_repair

# Parameter ranges (must match repair_benchmark.py)
PARAM_RANGES = {
    "tensor_core_tiles": (1, 4),
    "cgra_tiles": (4, 16),
    "clock_frequency_ghz": (1.0, 3.0),
    "memory_bandwidth_gbps": (400, 1200),
    "m": (8, 32),
    "n": (8, 32),
    "k": (8, 32),
    "sparsity": (0.0, 0.5),
}

# Buggy tile sizes (all != 16)
BUGGY_TILE_SIZES = [4, 8, 12, 20, 24, 28, 32]

def generate_variant(base_workload, seed=None):
    """Generate a broken variant by perturbing m or n or k or tensor_core_tiles, plus secondary noise."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    variant = copy.deepcopy(base_workload)
    # Choose primary bug: wrong m (and set n=k=m for consistency)
    wrong_m = random.choice(BUGGY_TILE_SIZES)
    variant["m"] = wrong_m
    variant["n"] = wrong_m
    variant["k"] = wrong_m
    # Secondary noise: also mutate one non-critical param
    secondary_params = {
        "cgra_tiles": [2, 4, 6, 8, 10, 12, 14, 16],
        "clock_frequency_ghz": [1.0, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.0],
    }
    noise_param = random.choice(list(secondary_params.keys()))
    noise_vals = [v for v in secondary_params[noise_param] if v != base_workload.get(noise_param)]
    variant[noise_param] = random.choice(noise_vals)
    return variant

def random_repair(workload, max_attempts=20):
    """Random search: randomly change one parameter per attempt."""
    arch = ArchitectureSpec()
    correct = get_correct_workload()
    for attempt in range(1, max_attempts + 1):
        param = random.choice(list(PARAM_RANGES.keys()))
        low, high = PARAM_RANGES[param]
        if param in ["tensor_core_tiles", "cgra_tiles", "m", "n", "k"]:
            new_val = random.randint(low, high)
        else:
            new_val = random.uniform(low, high)
        workload[param] = new_val
        try:
            files = generate_verilog(workload, arch)
            tb = get_testbench(workload['m'], workload['n'], workload['k'])
            passed, _, _ = compile_and_simulate(files, tb)
            if passed:
                return True, attempt, workload
        except:
            pass
    return False, max_attempts, workload

def brute_force_repair(workload, max_attempts=20):
    """Brute-force: try all integer values for tensor_core_tiles first."""
    arch = ArchitectureSpec()
    correct = get_correct_workload()
    low, high = PARAM_RANGES["tensor_core_tiles"]
    for val in range(low, high + 1):
        workload["tensor_core_tiles"] = val
        try:
            files = generate_verilog(workload, arch)
            tb = get_testbench(workload['m'], workload['n'], workload['k'])
            passed, _, _ = compile_and_simulate(files, tb)
            if passed:
                return True, 1, workload
        except:
            pass
    return False, max_attempts, workload

def main():
    print("=== Ablation Study: SparseX Components ===\n")
    arch = ArchitectureSpec()
    correct_workload = get_correct_workload()
    
    # Generate 50 variants
    variants = [generate_variant(correct_workload, seed=i) for i in range(50)]
    
    # Train causal model
    print("Training NCPG model...")
    ncpg_model, param_names = train_ncpg_on_design_space(num_samples=1000, num_epochs=300)
    print("Model trained.\n")
    
    # Full SparseX (causal repair)
    causal_success = 0
    for i, wl in enumerate(variants):
        success, _, _, _ = causal_repair(wl, ncpg_model, param_names, max_attempts=20)
        if success:
            causal_success += 1
        if (i+1) % 10 == 0:
            print(f"Causal repair progress: {i+1}/50, success={causal_success}")
    full_rate = causal_success / 50
    
    # w/o NCPG (random repair)
    random_success = 0
    for i, wl in enumerate(variants):
        success, _, _ = random_repair(copy.deepcopy(wl), max_attempts=20)
        if success:
            random_success += 1
        if (i+1) % 10 == 0:
            print(f"Random repair progress: {i+1}/50, success={random_success}")
    random_rate = random_success / 50
    
    # w/o MCTS (brute-force)
    bf_success = 0
    for i, wl in enumerate(variants):
        success, _, _ = brute_force_repair(copy.deepcopy(wl), max_attempts=20)
        if success:
            bf_success += 1
        if (i+1) % 10 == 0:
            print(f"Brute-force progress: {i+1}/50, success={bf_success}")
    bf_rate = bf_success / 50
    
    # w/o VAE (same as full, since VAE not used in repair)
    vae_rate = full_rate
    
    results = pd.DataFrame({
        "Configuration": ["Full SparseX", "w/o NCPG (random)", "w/o MCTS (brute-force)", "w/o VAE (same as full)"],
        "Repair Success Rate": [full_rate, random_rate, bf_rate, vae_rate]
    })
    print("\n" + results.to_string(index=False))
    results.to_csv("ablation_study.csv", index=False)
    print("\nResults saved to ablation_study.csv")

if __name__ == "__main__":
    main()