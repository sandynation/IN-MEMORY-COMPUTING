# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
#!/usr/bin/env python3
"""
Cold-start design using NCPG gradient descent.
Trains NCPG to predict a cost function that is minimized at m=n=k=16,
then starts from random parameters and takes gradient steps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import tempfile
import re
import subprocess
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec
from benchmark_support import get_testbench
from backend.novelty_features.ncpg_full import NCPG

def sanitize_verilog(text: str) -> str:
    text = re.sub(r'[\u2010-\u2015]', '-', text)
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201c\u201d]', '"', text)
    return text.encode('ascii', errors='replace').decode('ascii')

def simulate_workload(workload):
    arch = ArchitectureSpec()
    with tempfile.TemporaryDirectory() as tmpdir:
        files_dict = generate_verilog(workload, arch)
        file_paths = []
        for fname, content in files_dict.items():
            file_path = Path(tmpdir) / fname
            safe_content = sanitize_verilog(content)
            file_path.write_text(safe_content, encoding='utf-8')
            file_paths.append(str(file_path))
        tb = get_testbench(workload['m'], workload['n'], workload['k'])
        tb_path = Path(tmpdir) / "testbench.v"
        tb_path.write_text(sanitize_verilog(tb), encoding='utf-8')
        all_files = file_paths + [str(tb_path)]
        try:
            subprocess.run(['iverilog', '-o', str(Path(tmpdir)/'sim_out')] + all_files,
                           check=True, capture_output=True, text=True, cwd=tmpdir)
            result = subprocess.run(['vvp', str(Path(tmpdir)/'sim_out')],
                                    capture_output=True, text=True, timeout=10, cwd=tmpdir)
            return 'SUCCESS' in result.stdout
        except:
            return False

def train_cost_model():
    """Train NCPG to predict cost = (m-16)^2 + (n-16)^2 + (k-16)^2."""
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 1000
    m = np.random.randint(8, 33, n_samples)
    n = np.random.randint(8, 33, n_samples)
    k = np.random.randint(8, 33, n_samples)
    # Also include other parameters but set them to default (they don't affect cost)
    cgra = np.full(n_samples, 8)
    clk = np.full(n_samples, 2.0)
    X = np.column_stack([m, n, k, cgra, clk]).astype(np.float32)
    y = ((m-16)**2 + (n-16)**2 + (k-16)**2).reshape(-1, 1).astype(np.float32)
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    model = NCPG(num_nodes=5, output_dim=1, use_mlp=True, force_simple=True, device="cpu")
    model.fit(X_t, y_t, num_epochs=300, verbose=False)
    return model

def workload_to_vector(wl):
    """Convert workload dict to normalized vector for NCPG."""
    # Parameters: m, n, k, cgra_tiles, clock_frequency_ghz
    param_names = ['m', 'n', 'k', 'cgra_tiles', 'clock_frequency_ghz']
    ranges = [(8,32), (8,32), (8,32), (4,16), (1.0,3.0)]
    vec = []
    for name, (low, high) in zip(param_names, ranges):
        val = wl.get(name, 16)
        norm = (val - low) / (high - low) if high > low else 0.5
        vec.append(norm)
    return np.array(vec, dtype=np.float32)

def vector_to_workload(vec, wl):
    """Convert normalized vector back to workload dict."""
    param_names = ['m', 'n', 'k', 'cgra_tiles', 'clock_frequency_ghz']
    ranges = [(8,32), (8,32), (8,32), (4,16), (1.0,3.0)]
    for i, (name, (low, high)) in enumerate(zip(param_names, ranges)):
        new_val = low + vec[i] * (high - low)
        if name in ['m', 'n', 'k', 'cgra_tiles']:
            wl[name] = int(round(new_val))
        else:
            wl[name] = float(new_val)
    return wl

def cold_start_with_ncpg():
    print("Training NCPG cost model...")
    model = train_cost_model()
    print("Model trained.\n")
    
    # Start from random workload
    wl = {
        "name": "cold_start",
        "m": np.random.randint(8, 33),
        "n": np.random.randint(8, 33),
        "k": np.random.randint(8, 33),
        "cgra_tiles": 8,
        "clock_frequency_ghz": 2.0,
        "dtype": "f16",
        "sparsity": 0.0,
        "target": "tensor_core",
        "tensor_core_tiles": 2,
        "memory_bandwidth_gbps": 800.0,
    }
    print(f"Initial random workload: m={wl['m']} n={wl['n']} k={wl['k']}")
    
    vec = workload_to_vector(wl)
    history = []
    
    for step in range(30):
        vec_t = torch.tensor(vec, requires_grad=True).unsqueeze(0)
        pred = model.model(vec_t)
        grad = torch.autograd.grad(pred, vec_t, create_graph=False)[0].squeeze().detach().numpy()
        vec = vec - 0.2 * grad  # step size
        vec = np.clip(vec, 0.0, 1.0)
        wl = vector_to_workload(vec, wl)
        cost = (wl['m']-16)**2 + (wl['n']-16)**2 + (wl['k']-16)**2
        history.append(cost)
        if step % 5 == 0:
            print(f"Step {step}: m={wl['m']} n={wl['n']} k={wl['k']}, cost={cost:.1f}")
        
        # Check simulation
        if wl['m'] == 16 and wl['n'] == 16 and wl['k'] == 16:
            passed = simulate_workload(wl)
            if passed:
                print(f"\nSUCCESS! Found correct design at step {step}")
                return True, wl, history
    
    # Final simulation attempt
    passed = simulate_workload(wl)
    if passed:
        print("\nSUCCESS! Final design passes simulation.")
        return True, wl, history
    else:
        print("\nFAILED: Could not reach correct design.")
        return False, wl, history

def main():
    success, final_wl, hist = cold_start_with_ncpg()
    if success:
        print(f"\n*** COLD START SUCCESS ***")
        print(f"Final workload: m={final_wl['m']}, n={final_wl['n']}, k={final_wl['k']}")
    else:
        print("\nCold start failed.")
    
    plt.plot(hist)
    plt.xlabel("Gradient step")
    plt.ylabel("Cost ((m-16)^2+(n-16)^2+(k-16)^2)")
    plt.title("Cold-start convergence using NCPG gradients")
    plt.savefig("cold_start_convergence_ncpg.png")
    plt.close()
    print("Convergence plot saved to cold_start_convergence_ncpg.png")

if __name__ == "__main__":
    main()