# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# SparseX – Neural Causal Silicon Compiler
"""
Repair strategies for broken RTL:
- Random: mutate random parameter by random step.
- Brute‑force: enumerate parameter values in order.
- Causal‑guided: use NCPG sensitivity to choose parameter and step.
"""
import sys
import random
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

# Ensure workspace root (d:\CodexTest) and this directory are on sys.path
_workspace_root = str(Path(__file__).parents[2])
_this_dir = str(Path(__file__).parent)
for _p in [_workspace_root, _this_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec
from benchmark_support import compile_and_simulate  # shared helper

class BaseRepairStrategy:
    def __init__(self, arch: ArchitectureSpec, testbench_code: str, max_attempts: int = 20):
        self.arch = arch
        self.testbench = testbench_code
        self.max_attempts = max_attempts

    def repair(self, workload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], int]:
        """
        Attempt to repair the given workload.
        Returns (success, final_workload, attempts_used).
        """
        raise NotImplementedError

    def _simulate(self, workload):
        files = generate_verilog(workload, self.arch)
        passed, log, metric = compile_and_simulate(files, self.testbench)
        return passed, metric

class RandomRepair(BaseRepairStrategy):
    def __init__(self, arch, testbench, max_attempts=20,
                 param_ranges: Dict[str, List[Any]] = None):
        super().__init__(arch, testbench, max_attempts)
        self.param_ranges = param_ranges or {
            "tensor_core_tiles": [1,2,3,4],
            "cgra_tiles": [2,4,6,8,10,12,14,16],
            "memory_bandwidth_gbps": [200,400,600,800,1000,1200,1400,1600,1800,2000],
            "clock_frequency_ghz": [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0],
            "sparsity": [0.0,0.1,0.2,0.3,0.4,0.5],
            "m": [4,8,12,16,20,24,28,32],
            "n": [4,8,12,16,20,24,28,32],
            "k": [4,8,12,16,20,24,28,32],
        }

    def repair(self, workload):
        current = copy.deepcopy(workload)
        for attempt in range(1, self.max_attempts+1):
            # Choose a random parameter
            param = random.choice(list(self.param_ranges.keys()))
            # Choose a random value from its range (not equal to current if possible)
            current_val = current.get(param, None)
            possible = [v for v in self.param_ranges[param] if v != current_val]
            if not possible:
                possible = self.param_ranges[param]
            new_val = random.choice(possible)
            current[param] = new_val
            passed, _ = self._simulate(current)
            if passed:
                return True, current, attempt
        return False, current, self.max_attempts

class BruteForceRepair(BaseRepairStrategy):
    def __init__(self, arch, testbench, max_attempts=20,
                 param_order: List[str] = None,
                 param_values: Dict[str, List[Any]] = None):
        super().__init__(arch, testbench, max_attempts)
        self.param_order = param_order or ["tensor_core_tiles", "cgra_tiles", "memory_bandwidth_gbps", "clock_frequency_ghz", "sparsity", "m", "n", "k"]
        self.param_values = param_values or {
            "tensor_core_tiles": [1,2,3,4],
            "cgra_tiles": [2,4,6,8,10,12,14,16],
            "memory_bandwidth_gbps": [200,400,600,800,1000,1200,1400,1600,1800,2000],
            "clock_frequency_ghz": [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0],
            "sparsity": [0.0,0.1,0.2,0.3,0.4,0.5],
            "m": [4,8,12,16,20,24,28,32],
            "n": [4,8,12,16,20,24,28,32],
            "k": [4,8,12,16,20,24,28,32],
        }

    def repair(self, workload):
        current = copy.deepcopy(workload)
        attempt = 0
        for param in self.param_order:
            for val in self.param_values[param]:
                attempt += 1
                if attempt > self.max_attempts:
                    return False, current, self.max_attempts
                current[param] = val
                passed, _ = self._simulate(current)
                if passed:
                    return True, current, attempt
        return False, current, attempt

class CausalGuidedRepair(BaseRepairStrategy):
    def __init__(self, arch, testbench, causal_model, nominal_workload: Dict[str, Any],
                 param_step: Dict[str, float] = None,
                 param_bounds: Dict[str, Tuple[float,float]] = None,
                 max_attempts=20):
        super().__init__(arch, testbench, max_attempts)
        self.causal_model = causal_model
        self.nominal_workload = nominal_workload
        # Step sizes — keyed by workload dict field names
        self.param_step = param_step or {
            "m": 4,
            "n": 4,
            "k": 4,
            "tensor_core_tiles": 1,
            "cgra_tiles": 2,
            "memory_bandwidth_gbps": 100,
            "clock_frequency_ghz": 0.2,
            "sparsity": 0.1,
        }
        self.param_bounds = param_bounds or {
            "m": (4, 32),
            "n": (4, 32),
            "k": (4, 32),
            "tensor_core_tiles": (1, 4),
            "cgra_tiles": (2, 16),
            "memory_bandwidth_gbps": (200, 2000),
            "clock_frequency_ghz": (1.0, 3.0),
            "sparsity": (0.0, 0.5),
        }

    def _causal_sensitivity(self, workload):
        """Compute |d_output/d_param| for each workload parameter using the NCPG model.

        The NCPG is trained on 3 inputs: [tensor_dim, cgra_rows, voltage_proxy].
        We map these to workload dict keys: m → tensor_dim, cgra_tiles → cgra_rows,
        clock_frequency_ghz → voltage_proxy.  The returned dict uses the WORKLOAD
        key names so repair() can look them up directly.
        """
        tensor_dim    = float(workload.get("m", 16))
        cgra_rows     = float(workload.get("cgra_tiles", 8))
        voltage_proxy = float(workload.get("clock_frequency_ghz", 2.0))
        x = torch.tensor([[tensor_dim, cgra_rows, voltage_proxy]],
                         dtype=torch.float32, requires_grad=True)
        y = self.causal_model.model(x)
        y.sum().backward()
        grads = x.grad.detach().numpy().flatten()
        # Keys match workload dict field names
        return {
            "m":                    abs(float(grads[0])),
            "cgra_tiles":           abs(float(grads[1])),
            "clock_frequency_ghz": abs(float(grads[2])),
        }

    def repair(self, workload):
        current = copy.deepcopy(workload)
        for attempt in range(1, self.max_attempts+1):
            # Get sensitivity of current workload
            sens = self._causal_sensitivity(current)
            # Choose parameter with largest absolute gradient
            param_to_adjust = max(sens.keys(), key=lambda p: abs(sens[p]))
            step = self.param_step.get(param_to_adjust, 1)
            # Determine direction: if sensitivity is negative, decrease parameter? 
            # Actually, we want to move in direction that improves performance.
            # For simplicity, we'll adjust by +step if sensitivity positive? 
            # But here we repair failures, so we may need to explore both directions.
            # Let's first try the direction that reduces the gradient magnitude.
            # For this prototype, we'll try both +step and -step.
            original = current.get(param_to_adjust, None)
            if original is None:
                continue
            best = None
            best_passed = False
            for delta in [step, -step]:
                new_val = original + delta
                # Clamp to bounds
                lo, hi = self.param_bounds.get(param_to_adjust, (0,1e6))
                new_val = max(lo, min(hi, new_val))
                if new_val == original:
                    continue
                current[param_to_adjust] = new_val
                passed, _ = self._simulate(current)
                if passed:
                    return True, current, attempt
                if not best_passed:
                    best = new_val
                    best_passed = passed
            # If neither direction worked, revert and continue
            if best is not None:
                current[param_to_adjust] = best
            # else: no change, continue
        return False, current, self.max_attempts