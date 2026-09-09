# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
#!/usr/bin/env python3
import numpy as np
import torch
from backend.novelty_features.ncpg_full import NCPG
from generate_correct_design import get_correct_workload

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

def train_ncpg_on_design_space(num_samples=1000, num_epochs=500, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    param_names = list(PARAM_RANGES.keys())
    num_params = len(param_names)

    X = np.zeros((num_samples, num_params))
    for i, (param, (low, high)) in enumerate(PARAM_RANGES.items()):
        if param in ["tensor_core_tiles", "cgra_tiles", "m", "n", "k"]:
            X[:, i] = np.random.randint(low, high + 1, num_samples)
        else:
            X[:, i] = np.random.uniform(low, high, num_samples)

    # New synthetic cost: minimum at m=n=k=16, and low clock freq, low cgra_tiles
    m_idx = param_names.index("m")
    n_idx = param_names.index("n")
    k_idx = param_names.index("k")
    cgra_idx = param_names.index("cgra_tiles")
    clk_idx = param_names.index("clock_frequency_ghz")

    # Quadratic penalty: (m-16)^2 + (n-16)^2 + (k-16)^2 + 5*cgra_tiles + 10*clock^2
    cost = ((X[:, m_idx] - 16) ** 2 +
            (X[:, n_idx] - 16) ** 2 +
            (X[:, k_idx] - 16) ** 2 +
            5 * X[:, cgra_idx] +
            10 * X[:, clk_idx] ** 2)
    y = cost.reshape(-1, 1).astype(np.float32)
    X = X.astype(np.float32)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model = NCPG(num_nodes=num_params, output_dim=1, use_mlp=True,
                 force_simple=True, device="cpu")
    model.fit(X_t, y_t, num_epochs=num_epochs, verbose=False)
    return model, param_names

def causal_repair(workload, ncpg_model, param_names, max_attempts=20, step_size=0.1):
    correct = get_correct_workload()
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
    for attempt in range(1, max_attempts + 1):
        if all(workload.get(p) == correct.get(p) for p in param_names if p in correct):
            return True, attempt, workload, "already_correct"
        vec = workload_to_vector(workload)
        vec_t = torch.tensor(vec, requires_grad=True).unsqueeze(0)
        pred = ncpg_model.model(vec_t)
        grad = torch.autograd.grad(pred, vec_t, create_graph=False)[0].squeeze().detach().numpy()
        new_vec = vec - step_size * grad
        new_vec = np.clip(new_vec, 0.0, 1.0)
        vector_to_workload(new_vec, workload)
        if all(workload.get(p) == correct.get(p) for p in param_names if p in correct):
            return True, attempt, workload, "gradient_converged"
    return False, max_attempts, workload, "max_attempts_reached"