#!/usr/bin/env python3
"""
Compare NCPG with GES and PC on input‑output task.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
if not hasattr(np, "mat"):
    np.mat = np.asmatrix  # compatibility for causal-learn on NumPy 2.x
_np_where = np.where

def _compat_where(*args, **kwargs):
    if len(args) == 1:
        cond = np.asarray(args[0])
        if cond.ndim == 0:
            return (np.array([0], dtype=int),) if bool(cond) else (np.array([], dtype=int),)
    return _np_where(*args, **kwargs)

np.where = _compat_where
import torch
import time
from backend.novelty_features.ncpg_full import NCPG

def generate_linear_io_data(n=1000, d=8, noise=0.1):
    np.random.seed(42)
    X = np.random.randn(n, d)
    y = X[:,0] + X[:,1] + X[:,2] + X[:,3] + noise * np.random.randn(n)
    true_parents = [0,1,2,3]
    return X, y, true_parents

def run_ncpg(X, y, true_binary):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1,1)
    model = NCPG(num_nodes=X.shape[1], output_dim=1, use_mlp=True, force_simple=True, device="cpu")
    t0 = time.time()
    model.fit(X_t, y_t, num_epochs=300, verbose=False)
    t_train = time.time() - t0
    model.model.eval()
    from torch.func import vmap, jacrev
    def scalar_output(x):
        return model.model(x.unsqueeze(0)).squeeze()
    with torch.no_grad():
        J = vmap(jacrev(scalar_output))(X_t)
        mean_abs_jac = J.abs().mean(dim=0).cpu().numpy()
    pred = (mean_abs_jac > 0.05).astype(int)
    tp = ((pred == 1) & (true_binary == 1)).sum()
    fp = ((pred == 1) & (true_binary == 0)).sum()
    fn = ((pred == 0) & (true_binary == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"f1": f1, "shd": fp+fn, "time": t_train, "prec": prec, "rec": rec}

def run_ges_pc(X, y, true_binary):
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.ConstraintBased.PC import pc
    data = np.column_stack([X, y])
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
        return {"f1": f1, "shd": fp+fn, "prec": prec, "rec": rec}
    return metrics(adj_ges), metrics(adj_pc), t_ges, t_pc

def main():
    X, y, true_parents = generate_linear_io_data(n=2000)
    true_binary = np.zeros(X.shape[1])
    true_binary[true_parents] = 1

    print("NCPG:")
    res = run_ncpg(X, y, true_binary)
    print(f"  F1={res['f1']:.4f}, SHD={res['shd']}, Time={res['time']:.2f}s")

    res_ges, res_pc, t_ges, t_pc = run_ges_pc(X, y, true_binary)
    print(f"\nGES:  F1={res_ges['f1']:.4f}, SHD={res_ges['shd']}, Time={t_ges:.2f}s")
    print(f"PC:   F1={res_pc['f1']:.4f}, SHD={res_pc['shd']}, Time={t_pc:.2f}s")

if __name__ == "__main__":
    main()
