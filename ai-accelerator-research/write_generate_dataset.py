# write_generate_dataset.py
import pathlib

content = '''#!/usr/bin/env python3
# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from backend.accel_tool.arch import ArchitectureSpec
from backend.accel_tool.power_grid import PowerGridSimulator
from backend.accel_tool.verification import TapeoutChecker
from backend.novelty_features.physics.foundry_nodes import get_foundry_node, list_available_nodes

class DesignGenerator:
    def __init__(self):
        self.power_sim = PowerGridSimulator()
        self.checker = TapeoutChecker(rules={"min_width_nm":16,"min_spacing_nm":18,"max_density":0.85,"max_ucie_lanes":64})
        self.valid_nodes = ["TSMC_N3E", "TSMC_N2", "Intel_18A"]
        available = list_available_nodes()
        if not any(n in available for n in self.valid_nodes):
            self.valid_nodes = available

    def random_arch(self):
        arch = ArchitectureSpec()
        arch.num_chiplets = random.choice([1,2,4,8])
        arch.process_node = random.choice(self.valid_nodes)
        arch.compute.tensor_dim = random.choice([32,64,128,256])
        arch.compute.cgra_rows = random.choice([4,8,16])
        arch.compute.cgra_cols = arch.compute.cgra_rows
        arch.memory.bandwidth_gbps = random.uniform(800,4000)
        arch.link.gtps = random.uniform(16,64)
        arch.perf.tensor_core_tflops = 100.0 * (arch.compute.tensor_dim/64)**1.5
        arch.perf.tensor_core_energy_pj = 2.0 + 0.02*arch.compute.tensor_dim
        arch.perf.cgra_tops = 50.0 * (arch.compute.cgra_rows*arch.compute.cgra_cols)/64
        arch.perf.cgra_energy_pj = 1.5 + 0.01*arch.compute.cgra_rows
        return arch

    def is_physically_valid(self, arch):
        total_w = (arch.perf.tensor_core_tflops*1e12*arch.perf.tensor_core_energy_pj*1e-12 +
                   arch.perf.cgra_tops*1e12*arch.perf.cgra_energy_pj*1e-12)
        if total_w > 300: return False
        powers = [total_w/arch.num_chiplets]*min(4,arch.num_chiplets) + [0]*(4-min(4,arch.num_chiplets))
        report = self.power_sim.simulate_cluster_droop(powers)
        return all(r["status"]=="PASS" for r in report)

    def to_vector(self, arch):
        vec = np.array([arch.num_chiplets, arch.compute.tensor_dim, arch.compute.cgra_rows, arch.compute.cgra_cols,
                        arch.memory.bandwidth_gbps, arch.link.gtps, arch.perf.tensor_core_tflops,
                        arch.perf.tensor_core_energy_pj, arch.perf.cgra_tops, arch.perf.cgra_energy_pj], dtype=np.float32)
        norms = np.array([8,256,16,16,4000,64,2000,8,1000,8])
        return vec/norms

    def generate(self, n=100000, output_path="designs/latent_data.npy"):
        vectors = []
        pbar = tqdm(total=n)
        while len(vectors) < n:
            arch = self.random_arch()
            if self.is_physically_valid(arch):
                vectors.append(self.to_vector(arch))
                pbar.update(1)
        np.save(output_path, np.stack(vectors))
        print(f"Saved {n} vectors to {output_path}")

if __name__ == "__main__":
    gen = DesignGenerator()
    gen.generate()
'''

path = pathlib.Path('backend/novelty_features/latent/generate_dataset.py')
path.write_text(content, encoding='utf-8')
print(f"File written to {path}")