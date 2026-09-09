#!/usr/bin/env python3
"""
RTLLM subset benchmark: run 50 random GEMM configurations, simulate, report pass rate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import subprocess
import tempfile
import csv
import re

from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec
from benchmark_support import compile_and_simulate, get_testbench

def sanitize_verilog(text: str) -> str:
    text = re.sub(r'[\u2010-\u2015]', '-', text)
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201c\u201d]', '"', text)
    return text.encode('ascii', errors='replace').decode('ascii')

def random_workload(seed):
    random.seed(seed)
    m = random.randint(8, 32)
    n = random.randint(8, 32)
    k = random.randint(8, 32)
    return {
        "name": f"random_{m}x{n}x{k}",
        "m": m, "n": n, "k": k,
        "dtype": "f16",
        "sparsity": 0.0,
        "target": "tensor_core",
        "tensor_core_tiles": 2,
        "cgra_tiles": 8,
        "memory_bandwidth_gbps": 800.0,
        "clock_frequency_ghz": 2.0,
    }

def simulate_workload(workload, arch):
    with tempfile.TemporaryDirectory() as tmpdir:
        files = generate_verilog(workload, arch)
        file_paths = []
        for fname, content in files.items():
            p = Path(tmpdir) / fname
            p.write_text(sanitize_verilog(content), encoding='utf-8')
            file_paths.append(str(p))
        tb = get_testbench(workload['m'], workload['n'], workload['k'])
        tb_path = Path(tmpdir) / "testbench.v"
        tb_path.write_text(sanitize_verilog(tb), encoding='utf-8')
        all_files = file_paths + [str(tb_path)]
        try:
            subprocess.run(['iverilog', '-o', str(Path(tmpdir)/'sim')] + all_files,
                           check=True, capture_output=True, cwd=tmpdir)
            result = subprocess.run(['vvp', str(Path(tmpdir)/'sim')],
                                    capture_output=True, text=True, timeout=10, cwd=tmpdir)
            return 'SUCCESS' in result.stdout
        except:
            return False

def main():
    arch = ArchitectureSpec()
    num_tasks = 50
    results = []
    for i in range(num_tasks):
        wl = random_workload(i)
        print(f"Task {i+1}/{num_tasks}: {wl['m']}x{wl['n']}x{wl['k']}")
        passed = simulate_workload(wl, arch)
        results.append({"task_id": i, "m": wl['m'], "n": wl['n'], "k": wl['k'], "passed": passed})
        print(f"  -> {'PASS' if passed else 'FAIL'}")
    out_file = "rtllm_results.csv"
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "m", "n", "k", "passed"])
        writer.writeheader()
        writer.writerows(results)
    pass_count = sum(1 for r in results if r["passed"])
    print(f"\nFunctional pass rate: {pass_count/num_tasks*100:.1f}% ({pass_count}/{num_tasks})")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()