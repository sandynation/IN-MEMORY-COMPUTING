# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
#!/usr/bin/env python3
"""
Ablation study using the exact repair functions from repair_benchmark.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from repair_benchmark import generate_variant, random_repair, brute_force_repair
from causal_repair import train_ncpg_on_design_space, causal_repair

def main():
    print("=== Ablation Study (Using Working Repair Functions) ===\n")
    
    correct_workload = {
        "name": "gemm_16x16_correct",
        "m": 16, "n": 16, "k": 16,
        "dtype": "f16", "sparsity": 0.0,
        "target": "tensor_core", "tensor_core_tiles": 2,
        "cgra_tiles": 8, "memory_bandwidth_gbps": 800.0, "clock_frequency_ghz": 2.0,
    }
    
    # Generate 20 variants (same as repair_benchmark)
    variants = [generate_variant(correct_workload, seed=i)[0] for i in range(20)]
    
    # Train NCPG model with same parameters as successful run
    print("Training NCPG model...")
    ncpg_model, param_names = train_ncpg_on_design_space(num_samples=2000, num_epochs=500)
    
    # Full SparseX (causal repair)
    causal_success = 0
    for wl in variants:
        success, _, _, _ = causal_repair(wl, ncpg_model, param_names, max_attempts=20)
        if success:
            causal_success += 1
    full_rate = causal_success / len(variants)
    
    # Random repair
    random_success = 0
    for wl in variants:
        success, _, _ = random_repair(wl, max_attempts=20)
        if success:
            random_success += 1
    random_rate = random_success / len(variants)
    
    # Brute-force repair
    bf_success = 0
    for wl in variants:
        success, _, _ = brute_force_repair(wl, max_attempts=20)
        if success:
            bf_success += 1
    bf_rate = bf_success / len(variants)
    
    results = pd.DataFrame({
        "Configuration": ["Full SparseX", "w/o NCPG (random)", "w/o MCTS (brute-force)"],
        "Repair Success Rate": [full_rate, random_rate, bf_rate]
    })
    print(results)
    results.to_csv("ablation_results.csv", index=False)
    print("\nSaved to ablation_results.csv")
    print("\nExpected: Full SparseX ~1.0, random ~0.0, brute-force ~0.1")

if __name__ == "__main__":
    main()