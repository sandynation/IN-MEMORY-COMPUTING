#!/usr/bin/env python3
"""
Cold-start design from scratch using NCPG gradients.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import torch
import tempfile
import subprocess
import re
from causal_repair import train_ncpg_on_design_space, PARAM_RANGES
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec
from benchmark_support import get_testbench

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

def cold_start_ncpg(ncpg_model, param_names, max_iterations=30, step_size=0.3):
    def workload_to_vector(wl):
        vec = []
        for name in param_names:
            val = wl.get(name, 0.0)
            low, high = PARAM_RANGES[name]
            norm = (val - low) / (high - low) if high > low else 0.5
            vec.append(norm)
        return np.array(vec, dtype=np.float32)
    
    def vector_to_workload(vec, wl):
        for i, name in enumerate(param_names):
            low, high = PARAM_RANGES[name]
            new_val = low + vec[i] * (high - low)
            if name in ["tensor_core_tiles", "cgra_tiles", "m", "n", "k"]:
                wl[name] = int(round(new_val))
            else:
                wl[name] = float(new_val)
        return wl
    
    # Random initial workload
    wl = {name: random.randint(*PARAM_RANGES[name]) if name in ['m','n','k','tensor_core_tiles','cgra_tiles'] 
          else random.uniform(*PARAM_RANGES[name]) for name in param_names}
    wl.update({"name": "cold_start", "dtype": "f16", "target": "tensor_core"})
    vec = workload_to_vector(wl)
    print(f"Initial random: m={wl['m']}, n={wl['n']}, k={wl['k']}")
    
    for it in range(max_iterations):
        vec_t = torch.tensor(vec, requires_grad=True).unsqueeze(0)
        pred = ncpg_model.model(vec_t)
        grad = torch.autograd.grad(pred, vec_t, create_graph=False)[0].squeeze().detach().numpy()
        vec = vec - step_size * grad
        vec = np.clip(vec, 0.0, 1.0)
        wl = vector_to_workload(vec, wl.copy())
        if it % 5 == 0:
            print(f"Iter {it}: m={wl['m']}, n={wl['n']}, k={wl['k']}")
        if wl['m'] == 16 and wl['n'] == 16 and wl['k'] == 16:
            passed = simulate_workload(wl)
            if passed:
                print(f"SUCCESS after {it+1} iterations!")
                return True, wl, it+1
            else:
                print("Reached correct dimensions but simulation failed (RTL issue).")
                return False, wl, it+1
    print("Failed to converge to correct dimensions.")
    return False, wl, max_iterations

def main():
    print("Training NCPG model...")
    ncpg_model, param_names = train_ncpg_on_design_space(num_samples=2000, num_epochs=500)
    print("Model trained.\n")
    success, final_wl, iters = cold_start_ncpg(ncpg_model, param_names)
    if success:
        print(f"\n*** Cold-start success after {iters} iterations ***")
    else:
        print("\n*** Cold-start failed ***")

if __name__ == "__main__":
    main()