#!/usr/bin/env python3
"""
Run NCPG on the Sachs protein signaling network (real causal benchmark).
If data unavailable, uses synthetic fallback.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from backend.novelty_features.ncpg_full import NCPG

def main():
    print("Sachs benchmark – NCPG input‑output causal discovery")
    try:
        from causallearn.utils import load_sachs_data
        data = load_sachs_data()
        X = data.values.astype(np.float32)
        num_inputs = X.shape[1] - 1
        y = X[:, -1]
        X = X[:, :-1]
        # Known parents from literature (example: first 3 variables)
        true_parents = [0, 1, 2]
        true_binary = np.zeros(num_inputs)
        true_binary[true_parents] = 1
    except Exception as e:
        print(f"Could not load Sachs data: {e}\nUsing synthetic fallback.")
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 8)
        y = X[:,0] + X[:,1] + X[:,2] + X[:,3] + 0.05 * np.random.randn(n)
        true_parents = [0,1,2,3]
        true_binary = np.zeros(8)
        true_binary[true_parents] = 1

    model = NCPG(num_nodes=X.shape[1], output_dim=1, use_mlp=True, force_simple=True, device="cpu")
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1,1)
    model.fit(X_t, y_t, num_epochs=500, verbose=False)

    model.model.eval()
    from torch.func import vmap, jacrev
    def scalar_output(x):
        return model.model(x.unsqueeze(0)).squeeze()
    with torch.no_grad():
        J = vmap(jacrev(scalar_output))(X_t)
        mean_abs_jac = J.abs().mean(dim=0).cpu().numpy()

    print("\nResults (threshold sweep):")
    for th in [0.01, 0.02, 0.05, 0.1, 0.2]:
        pred = (mean_abs_jac > th).astype(int)
        tp = ((pred == 1) & (true_binary == 1)).sum()
        fp = ((pred == 1) & (true_binary == 0)).sum()
        fn = ((pred == 0) & (true_binary == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"th={th:4.2f} | F1={f1:.4f} | SHD={fp+fn} | prec={prec:.3f} | rec={rec:.3f}")

if __name__ == "__main__":
    main()