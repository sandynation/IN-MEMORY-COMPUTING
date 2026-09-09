#!/usr/bin/env python3
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

def sanitize(s):
    s = re.sub(r'[\u2010-\u2015]','-',s)
    s = re.sub(r'[\u2018\u2019]',"'",s)
    s = re.sub(r'[\u201c\u201d]','"',s)
    return s.encode('ascii',errors='replace').decode('ascii')

def simulate(wl):
    arch = ArchitectureSpec()
    with tempfile.TemporaryDirectory() as d:
        files = generate_verilog(wl, arch)
        paths = []
        for name, cont in files.items():
            p = Path(d)/name
            p.write_text(sanitize(cont), encoding='utf-8')
            paths.append(str(p))
        tb = get_testbench(wl['m'], wl['n'], wl['k'])
        tb_path = Path(d)/'tb.v'
        tb_path.write_text(sanitize(tb), encoding='utf-8')
        all_files = paths + [str(tb_path)]
        try:
            subprocess.run(['iverilog','-o',str(Path(d)/'sim')]+all_files, check=True, capture_output=True, cwd=d)
            r = subprocess.run(['vvp',str(Path(d)/'sim')], capture_output=True, text=True, timeout=10, cwd=d)
            return 'SUCCESS' in r.stdout
        except:
            return False

def cold_start():
    ncpg, pnames = train_ncpg_on_design_space(num_samples=2000, num_epochs=500)
    # random init
    wl = {n: random.randint(*PARAM_RANGES[n]) if n in ['m','n','k','tensor_core_tiles','cgra_tiles']
          else random.uniform(*PARAM_RANGES[n]) for n in pnames}
    wl.update({'name':'cold','dtype':'f16','target':'tensor_core'})
    print(f"Start: m={wl['m']}, n={wl['n']}, k={wl['k']}")
    def to_vec(w):
        return np.array([(w.get(n,0)-PARAM_RANGES[n][0])/(PARAM_RANGES[n][1]-PARAM_RANGES[n][0]) for n in pnames], dtype=np.float32)
    def to_wl(vec, w):
        for i,n in enumerate(pnames):
            lo,hi = PARAM_RANGES[n]
            val = lo + vec[i]*(hi-lo)
            if n in ['m','n','k','tensor_core_tiles','cgra_tiles']:
                w[n] = int(round(val))
            else:
                w[n] = float(val)
        return w
    vec = to_vec(wl)
    step = 0.6
    for it in range(40):
        vt = torch.tensor(vec, requires_grad=True).unsqueeze(0)
        pred = ncpg.model(vt)
        grad = torch.autograd.grad(pred, vt, create_graph=False)[0].squeeze().detach().numpy()
        vec = vec - step*grad
        vec = np.clip(vec,0,1)
        wl = to_wl(vec, wl.copy())
        if it%5==0:
            print(f"Iter {it}: m={wl['m']}, n={wl['n']}, k={wl['k']}")
        if wl['m']==16 and wl['n']==16 and wl['k']==16:
            if simulate(wl):
                print(f"SUCCESS after {it+1} iterations!")
                return True
            else:
                print("Reached 16 but simulation failed.")
                return False
    print("Failed to converge.")
    return False

if __name__=='__main__':
    print("\n=== Cold-start with NCPG ===\n")
    cold_start()