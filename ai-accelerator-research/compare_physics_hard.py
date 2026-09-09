#!/usr/bin/env python3
"""
Compare NCPG, GES, and PC on the physics‑hard dataset (8 inputs, non‑linear target).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import time
from backend.novelty_features.ncpg_full import NCPG
from backend.novelty_features.benchmarking.physics_hard_dataset import generate_physics_hard_dataset
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges

def run_ncpg(X, y):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1,1)
    model = NCPG(num_nodes=X.shape[1], output_dim=1, use_mlp=True, force_simple=True, device="cpu")
    t0 = time.time()
    model.fit(X_t, y_t, num_epochs=500, verbose=False)
    t = time.time() - t0
    model.model.eval()
    from torch.func import vmap, jacrev
    def scalar_output(x):
        return model.model(x.unsqueeze(0)).squeeze()
    with torch.no_grad():
        J = vmap(jacrev(scalar_output))(X_t)
        mean_abs_jac = J.abs().mean(dim=0).cpu().numpy()
    # True parents in physics‑hard dataset: first 4 inputs
    true_binary = np.zeros(X.shape[1])
    true_binary[:4] = 1
    pred_binary = (mean_abs_jac > 0.05).astype(int)
    tp = ((pred_binary == 1) & (true_binary == 1)).sum()
    fp = ((pred_binary == 1) & (true_binary == 0)).sum()
    fn = ((pred_binary == 0) & (true_binary == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"F1": f1, "SHD": fp+fn, "time": t}

def run_ges_pc(data, true_binary):
    # data: X + y as last column
    last = data.shape[1] - 1
    # GES
    t0 = time.time()
    Record = ges(data, score_func='local_score_BIC')
    adj_ges = np.abs(Record['G'].graph).clip(0,1)
    t_ges = time.time() - t0
    # PC
    t0 = time.time()
    cg = pc(data, alpha=0.05, indep_test='fisherz')
    adj_pc = np.abs(cg.G.graph).clip(0,1)
    t_pc = time.time() - t0
    def metrics(adj):
        pred = adj[:last, last]
        pred_bin = (pred > 0.5).astype(int)
        tp = ((pred_bin == 1) & (true_binary == 1)).sum()
        fp = ((pred_bin == 1) & (true_binary == 0)).sum()
        fn = ((pred_bin == 0) & (true_binary == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        return {"F1": f1, "SHD": fp+fn}
    return metrics(adj_ges), metrics(adj_pc), t_ges, t_pc

def main():
    data, true_dag = generate_physics_hard_dataset(n_samples=1000, noise_std=0.05)
    X = data.drop("y", axis=1).values
    y = data["y"].values
    true_binary = np.zeros(X.shape[1])
    true_binary[:4] = 1   # first 4 inputs are true parents

    print("NCPG on physics‑hard:")
    res_ncpg = run_ncpg(X, y)
    print(f"  F1={res_ncpg['F1']:.4f}, SHD={res_ncpg['SHD']}, time={res_ncpg['time']:.2f}s")

    data_full = np.column_stack([X, y])
    res_ges, res_pc, t_ges, t_pc = run_ges_pc(data_full, true_binary)
    print(f"GES:  F1={res_ges['F1']:.4f}, SHD={res_ges['SHD']}, time={t_ges:.2f}s")
    print(f"PC:   F1={res_pc['F1']:.4f}, SHD={res_pc['SHD']}, time={t_pc:.2f}s")

if __name__ == "__main__":
    main()