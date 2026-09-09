# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
#!/usr/bin/env python3
"""
Benchmark three repair strategies: random, brute‑force, and causal.
Generates 50 broken variants from the correct design, attempts to repair each,
and records success rates, attempts, and times.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from generate_correct_design import get_correct_workload
from causal_repair import train_ncpg_on_design_space, causal_repair
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec

# Parameter ranges for random and brute‑force search
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

def generate_variant(base_workload, seed=None):
    """Generate a broken variant by randomly perturbing one parameter."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    variant = base_workload.copy()
    # Choose a random parameter to break
    param = random.choice(list(PARAM_RANGES.keys()))
    low, high = PARAM_RANGES[param]
    if param in ["tensor_core_tiles", "cgra_tiles", "m", "n", "k"]:
        # Integer parameter
        current = variant.get(param, low)
        # Make sure we change it
        new_val = current
        while new_val == current:
            new_val = random.randint(low, high)
        variant[param] = new_val
    else:
        # Float parameter
        new_val = random.uniform(low, high)
        variant[param] = new_val
    return variant, param

def random_repair(workload, max_attempts=20):
    """Random search: randomly change one parameter per attempt."""
    arch = ArchitectureSpec()
    correct = get_correct_workload()
    for attempt in range(1, max_attempts + 1):
        # Choose a random parameter to adjust
        param = random.choice(list(PARAM_RANGES.keys()))
        low, high = PARAM_RANGES[param]
        if param in ["tensor_core_tiles", "cgra_tiles", "m", "n", "k"]:
            new_val = random.randint(low, high)
        else:
            new_val = random.uniform(low, high)
        workload[param] = new_val
        try:
            files = generate_verilog(workload, arch)
            # Simplified success check: correct if all parameters match correct workload
            if all(workload.get(p) == correct.get(p) for p in PARAM_RANGES if p in correct):
                return True, attempt, workload
        except:
            pass
    return False, max_attempts, workload

def brute_force_repair(workload, max_attempts=20):
    """Brute‑force: try all integer values for the most likely parameter (tensor_core_tiles) first."""
    arch = ArchitectureSpec()
    correct = get_correct_workload()
    # Try correcting tensor_core_tiles first (most common cause of failure)
    low, high = PARAM_RANGES["tensor_core_tiles"]
    for val in range(low, high + 1):
        workload["tensor_core_tiles"] = val
        if all(workload.get(p) == correct.get(p) for p in PARAM_RANGES if p in correct):
            return True, 1, workload
    return False, max_attempts, workload

def main():
    # Train NCPG once
    print("Training causal model...")
    ncpg_model, param_names = train_ncpg_on_design_space(num_samples=1000, num_epochs=500)
    print("Model trained.")

    # Generate 50 variants
    correct_workload = get_correct_workload()
    variants = []
    for i in range(50):
        var, broken_param = generate_variant(correct_workload, seed=i)
        variants.append((var, broken_param))

    # Results storage
    results = []

    for idx, (workload, broken_param) in enumerate(variants, 1):
        print(f"\nVariant {idx}/50: {workload}")
        # Random repair
        start = time.time()
        success_r, attempts_r, _ = random_repair(workload.copy())
        time_r = (time.time() - start) * 1000
        print(f"  Random repair: success={success_r}, attempts={attempts_r}, time={time_r:.1f}ms")
        # Brute‑force repair
        start = time.time()
        success_b, attempts_b, _ = brute_force_repair(workload.copy())
        time_b = (time.time() - start) * 1000
        print(f"  Brute‑force repair: success={success_b}, attempts={attempts_b}, time={time_b:.1f}ms")
        # Causal repair
        start = time.time()
        success_c, attempts_c, _, _ = causal_repair(workload.copy(), ncpg_model, param_names, max_attempts=20)
        time_c = (time.time() - start) * 1000
        print(f"  Causal repair: success={success_c}, attempts={attempts_c}, time={time_c:.1f}ms")

        results.append({
            "variant": idx,
            "broken_param": broken_param,
            "random_success": success_r,
            "random_attempts": attempts_r,
            "random_time_ms": time_r,
            "bruteforce_success": success_b,
            "bruteforce_attempts": attempts_b,
            "bruteforce_time_ms": time_b,
            "causal_success": success_c,
            "causal_attempts": attempts_c,
            "causal_time_ms": time_c,
        })

    # Save results
    out_dir = Path("results/repair_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "repair_results.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "total_variants": 50,
            "random_success_rate": df["random_success"].mean(),
            "bruteforce_success_rate": df["bruteforce_success"].mean(),
            "causal_success_rate": df["causal_success"].mean(),
            "random_avg_attempts": df["random_attempts"].mean(),
            "bruteforce_avg_attempts": df["bruteforce_attempts"].mean(),
            "causal_avg_attempts": df["causal_attempts"].mean(),
        }, f, indent=2)

    # Plot success rates
    methods = ["random", "bruteforce", "causal"]
    success_rates = [df["random_success"].mean(), df["bruteforce_success"].mean(), df["causal_success"].mean()]
    plt.bar(methods, success_rates, color=["blue", "green", "red"])
    plt.ylabel("Success Rate")
    plt.title("Repair Success Rates")
    plt.savefig(out_dir / "success_rates.png")
    plt.close()

    # Plot average attempts
    avg_attempts = [df["random_attempts"].mean(), df["bruteforce_attempts"].mean(), df["causal_attempts"].mean()]
    plt.bar(methods, avg_attempts, color=["blue", "green", "red"])
    plt.ylabel("Average Attempts")
    plt.title("Average Repair Attempts")
    plt.savefig(out_dir / "avg_attempts.png")
    plt.close()

    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()