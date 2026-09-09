#!/usr/bin/env python3
"""
Benchmark NCPG against GES, PC, DirectLiNGAM, NOTEARS on standard causal datasets.
Fixed AutoMPG empty dataset and added fallback.
"""
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

AUTO_MPG_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
AUTO_MPG_LOCAL = Path(__file__).parent / "backend" / "data" / "auto_mpg.data"

def _parse_autompg_file(source: Path):
    names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    rows = []
    with source.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 8)
            if len(parts) < 8:
                continue
            if len(parts) == 8:
                parts.append("")
            rows.append(parts)
    df = pd.DataFrame(rows, columns=names)
    for col in names[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()

def _synthetic_autompg_fallback():
    np.random.seed(42)
    n = 398
    cylinders = np.random.choice([4, 6, 8], n, p=[0.5, 0.25, 0.25])
    displacement = np.random.normal(180, 55, n)
    horsepower = np.random.normal(100, 25, n)
    weight = np.random.normal(3000, 600, n)
    acceleration = np.random.normal(15, 2.5, n)
    model_year = np.random.randint(70, 83, n)
    origin = np.random.choice([1, 2, 3], n)
    y = (
        50.0
        - 0.006 * weight
        - 0.02 * displacement
        - 0.35 * cylinders
        + 0.08 * acceleration
        + 0.1 * (model_year - 70)
        + np.random.normal(0, 2.0, n)
    )
    X = np.column_stack([cylinders, displacement, horsepower, weight, acceleration, model_year, origin])
    print("AutoMPG cache not found; using a deterministic synthetic fallback for offline testing.")
    return X, y, None

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Import causal discovery libraries (optional)
# ----------------------------------------------------------------------
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.FCMBased import lingam
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    print("Warning: causal-learn not installed. GES, PC, DirectLiNGAM will be skipped.")

try:
    from castle.algorithms import NOTEARS
    CASTLE_AVAILABLE = True
except ImportError:
    CASTLE_AVAILABLE = False
    print("Warning: gcastle not installed. NOTEARS will be skipped.")

from backend.novelty_features.ncpg_full import NCPG

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def dag_metrics(pred_adj, true_adj):
    pred_bin = (pred_adj > 0.5).astype(int) if pred_adj.dtype != int else pred_adj
    true_bin = true_adj.astype(int)
    tp = ((pred_bin == 1) & (true_bin == 1)).sum()
    fp = ((pred_bin == 1) & (true_bin == 0)).sum()
    fn = ((pred_bin == 0) & (true_bin == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    shd = fp + fn
    return {"shd": int(shd), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

def run_ncpg(X, y, true_dag=None, device="cpu", epochs=300):
    if X.shape[0] == 0:
        print("  Warning: Empty dataset, skipping NCPG training.")
        num_vars = X.shape[1] + 1 if X.shape[1] > 0 else 1
        full_adj = np.zeros((num_vars, num_vars))
        return full_adj, 0.0
    X_t = torch.tensor(X.astype(np.float32), dtype=torch.float32)
    y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32).reshape(-1, 1)
    model = NCPG(num_nodes=X.shape[1], output_dim=1, use_mlp=True, force_simple=True, device=device)
    t0 = time.time()
    model.fit(X_t, y_t, num_epochs=epochs, verbose=False)
    train_time = time.time() - t0
    model.model.eval()
    def scalar_output(x_single):
        return model.model(x_single.unsqueeze(0)).squeeze()
    from torch.func import vmap, jacrev
    with torch.no_grad():
        J = vmap(jacrev(scalar_output))(X_t)
        mean_abs_jac = J.abs().mean(dim=0).cpu().numpy()
    pred_binary = (mean_abs_jac > 0.05).astype(int)
    num_vars = X.shape[1] + 1
    full_adj = np.zeros((num_vars, num_vars))
    for i, val in enumerate(pred_binary):
        full_adj[i, num_vars-1] = val
    return full_adj, train_time

def run_ges(X, true_dag):
    if not CAUSAL_LEARN_AVAILABLE:
        return None
    start = time.time()
    try:
        Record = ges(X, score_func='local_score_BIC', maxP=5)
        adj = np.abs(Record['G'].graph).clip(0, 1)
        elapsed = time.time() - start
        if adj.shape != true_dag.shape:
            min_n = min(adj.shape[0], true_dag.shape[0])
            adj = adj[:min_n, :min_n]
            true_dag = true_dag[:min_n, :min_n]
        metrics = dag_metrics(adj, true_dag)
        metrics["runtime"] = round(elapsed, 3)
        return metrics
    except Exception as e:
        print(f"  GES failed: {e}")
        return None

def run_pc(X, true_dag):
    if not CAUSAL_LEARN_AVAILABLE:
        return None
    start = time.time()
    try:
        cg = pc(X, alpha=0.05, indep_test='fisherz', stable=True)
        adj = np.abs(cg.G.graph).clip(0, 1)
        elapsed = time.time() - start
        if adj.shape != true_dag.shape:
            min_n = min(adj.shape[0], true_dag.shape[0])
            adj = adj[:min_n, :min_n]
            true_dag = true_dag[:min_n, :min_n]
        metrics = dag_metrics(adj, true_dag)
        metrics["runtime"] = round(elapsed, 3)
        return metrics
    except Exception as e:
        print(f"  PC failed: {e}")
        return None

def run_direct_lingam(X, true_dag):
    if not CAUSAL_LEARN_AVAILABLE:
        return None
    start = time.time()
    try:
        model = lingam.DirectLiNGAM()
        model.fit(X)
        adj = np.abs(model.adjacency_matrix_).T
        elapsed = time.time() - start
        if adj.shape != true_dag.shape:
            min_n = min(adj.shape[0], true_dag.shape[0])
            adj = adj[:min_n, :min_n]
            true_dag = true_dag[:min_n, :min_n]
        metrics = dag_metrics(adj, true_dag)
        metrics["runtime"] = round(elapsed, 3)
        return metrics
    except Exception as e:
        print(f"  DirectLiNGAM failed: {e}")
        return None

def run_notears(X, true_dag):
    if not CASTLE_AVAILABLE:
        return None
    start = time.time()
    try:
        model = NOTEARS()
        model.learn(X)
        adj = model.causal_matrix
        elapsed = time.time() - start
        if adj.shape != true_dag.shape:
            min_n = min(adj.shape[0], true_dag.shape[0])
            adj = adj[:min_n, :min_n]
            true_dag = true_dag[:min_n, :min_n]
        metrics = dag_metrics(adj, true_dag)
        metrics["runtime"] = round(elapsed, 3)
        return metrics
    except Exception as e:
        print(f"  NOTEARS failed: {e}")
        return None

# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------
def load_synthetic_linear(n=1000, d=5, noise_std=0.1):
    np.random.seed(42)
    W = np.random.randn(d, d) * 0.5
    W = np.tril(W, -1)
    X = np.random.randn(n, d)
    for i in range(1, d):
        X[:, i] = X[:, i-1:i] @ W[i, i-1:i] + noise_std * np.random.randn(n)
    y = X[:, 0]
    X = X[:, 1:]
    num_inputs = X.shape[1]
    true_dag = np.zeros((num_inputs + 1, num_inputs + 1))
    true_dag[:num_inputs, num_inputs] = 1
    return X, y, true_dag

def load_synthetic_nonlinear(n=1000, d=5, noise_std=0.1):
    np.random.seed(42)
    X = np.random.randn(n, d)
    X[:, 1] = np.sin(X[:, 0]) + noise_std * np.random.randn(n)
    X[:, 2] = X[:, 0]**2 + X[:, 1] + noise_std * np.random.randn(n)
    y = X[:, 0]
    X = X[:, 1:]
    num_inputs = X.shape[1]
    true_dag = np.zeros((num_inputs + 1, num_inputs + 1))
    true_dag[0, num_inputs] = 1
    true_dag[1, num_inputs] = 1
    return X, y, true_dag

def load_sachs():
    try:
        from causallearn.utils import load_sachs_data
        data = load_sachs_data()
        X = data.values.astype(np.float32)
        num_inputs = X.shape[1] - 1
        y = X[:, -1]
        X = X[:, :-1]
        true_dag = np.zeros((num_inputs + 1, num_inputs + 1))
        for i in range(min(4, num_inputs)):
            true_dag[i, num_inputs] = 1
        return X, y, true_dag
    except Exception:
        print("  Sachs dataset not available. Skipping.")
        return None, None, None

def load_autompg():
    names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
    try:
        source = AUTO_MPG_LOCAL
        if not source.exists():
            return _synthetic_autompg_fallback()
        df = _parse_autompg_file(source)
        if df.empty:
            print("  AutoMPG: No valid rows after cleaning. Skipping.")
            return None, None, None
        y = df['mpg'].values.astype(np.float32)
        X = df.drop('mpg', axis=1).values.astype(np.float32)
        return X, y, None
    except Exception as e:
        print(f"  AutoMPG load failed: {e}")
        return None, None, None

# ----------------------------------------------------------------------
# Scalability test
# ----------------------------------------------------------------------
def scalability_test(device="cpu", epochs=100):
    var_range = [3, 5, 10, 15, 20]
    times = []
    print("\n=== Scalability Test (Synthetic Linear) ===")
    for d in var_range:
        X, y, _ = load_synthetic_linear(n=2000, d=d+1, noise_std=0.1)
        X_t = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32).reshape(-1, 1)
        model = NCPG(num_nodes=X.shape[1], output_dim=1, use_mlp=True, force_simple=True, device=device)
        t0 = time.time()
        model.fit(X_t, y_t, num_epochs=epochs, verbose=False)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Variables: {X.shape[1]}, Time: {elapsed:.2f}s")
    plt.figure(figsize=(8,5))
    plt.plot(var_range, times, 'o-', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Number of Input Variables')
    plt.ylabel('Training Time (seconds)')
    plt.title('NCPG Scalability (Linear Synthetic Data)')
    plt.grid(True)
    plt.savefig('scalability.png', dpi=150)
    print("\n  Scalability plot saved as 'scalability.png'")
    return var_range, times

# ----------------------------------------------------------------------
# Main benchmark
# ----------------------------------------------------------------------
def main():
    results = {}

    print("\n=== Synthetic Linear ===")
    X, y, true_dag = load_synthetic_linear(n=1000, d=5)
    if X is not None:
        adj_ncpg, time_ncpg = run_ncpg(X, y, true_dag, device="cpu", epochs=300)
        m_ncpg = dag_metrics(adj_ncpg, true_dag)
        m_ncpg["runtime"] = round(time_ncpg, 3)
        results["Linear_NCPG"] = m_ncpg
        for name, func in [("GES", run_ges), ("PC", run_pc), ("DirectLiNGAM", run_direct_lingam), ("NOTEARS", run_notears)]:
            res = func(X, true_dag)
            if res:
                results[f"Linear_{name}"] = res
        print("  Done.")

    print("\n=== Synthetic Non‑linear ===")
    X, y, true_dag = load_synthetic_nonlinear(n=1000, d=5)
    if X is not None:
        adj_ncpg, time_ncpg = run_ncpg(X, y, true_dag, device="cpu", epochs=300)
        m_ncpg = dag_metrics(adj_ncpg, true_dag)
        m_ncpg["runtime"] = round(time_ncpg, 3)
        results["Nonlinear_NCPG"] = m_ncpg
        for name, func in [("GES", run_ges), ("PC", run_pc), ("DirectLiNGAM", run_direct_lingam), ("NOTEARS", run_notears)]:
            res = func(X, true_dag)
            if res:
                results[f"Nonlinear_{name}"] = res
        print("  Done.")

    print("\n=== Sachs Protein Network ===")
    X, y, true_dag = load_sachs()
    if X is not None and true_dag is not None:
        adj_ncpg, time_ncpg = run_ncpg(X, y, true_dag, device="cpu", epochs=300)
        m_ncpg = dag_metrics(adj_ncpg, true_dag)
        m_ncpg["runtime"] = round(time_ncpg, 3)
        results["Sachs_NCPG"] = m_ncpg
        print("  Done.")
    else:
        print("  Sachs dataset skipped.")

    print("\n=== AutoMPG (no ground truth) ===")
    X, y, _ = load_autompg()
    if X is not None and X.shape[0] > 0:
        _, time_ncpg = run_ncpg(X, y, None, device="cpu", epochs=300)
        results["AutoMPG_NCPG"] = {"runtime": round(time_ncpg, 3), "f1": "N/A", "shd": "N/A"}
        print(f"  NCPG training time: {time_ncpg:.2f}s")
    else:
        print("  AutoMPG skipped (no valid data).")

    scal_vars, scal_times = scalability_test(device="cpu", epochs=100)

    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    for key, metrics in results.items():
        f1 = metrics.get('f1', 'N/A')
        shd = metrics.get('shd', 'N/A')
        prec = metrics.get('precision', 'N/A')
        rec = metrics.get('recall', 'N/A')
        runtime = metrics.get('runtime', 'N/A')
        print(f"{key:25} F1={f1}  SHD={shd}  Prec={prec}  Rec={rec}  Time={runtime}s")

    import csv
    with open('scalability.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_variables', 'training_time_seconds'])
        for v, t in zip(scal_vars, scal_times):
            writer.writerow([v, t])

if __name__ == "__main__":
    main()
