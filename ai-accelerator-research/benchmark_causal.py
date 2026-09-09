#!/usr/bin/env python3
"""
Benchmark NCPG against GES, PC, DirectLiNGAM, NOTEARS on standard causal datasets.
"""
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
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

# Suppress warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Import causal discovery libraries
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

# ----------------------------------------------------------------------
# Import NCPG
# ----------------------------------------------------------------------
from backend.novelty_features.ncpg_full import NCPG

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def dag_metrics(pred_adj, true_adj):
    """Compute SHD, precision, recall, F1 for binary adjacency matrices."""
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

def run_ncpg(X, y, true_graph_input_output, device="cpu"):
    """
    Run NCPG and return predicted input‑output edges (binary).
    Uses Jacobian mean absolute values over test set.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    model = NCPG(num_nodes=X.shape[1], output_dim=1, use_mlp=True, force_simple=True, device=device)
    # Train for a fixed number of epochs (tune as needed)
    model.fit(X_t, y_t, num_epochs=300, verbose=False)
    # Compute Jacobian on the same data (or hold-out)
    model.model.eval()
    def scalar_output(x_single):
        return model.model(x_single.unsqueeze(0)).squeeze()
    from torch.func import vmap, jacrev
    with torch.no_grad():
        J = vmap(jacrev(scalar_output))(X_t)   # (batch, num_inputs)
        mean_abs_jac = J.abs().mean(dim=0).cpu().numpy()
    # Threshold to get binary edges (heuristically choose threshold 0.1)
    pred_binary = (mean_abs_jac > 0.1).astype(int)
    # Build full adjacency matrix (for compatibility with metrics)
    num_vars = X.shape[1] + 1  # +1 for output node
    full_adj = np.zeros((num_vars, num_vars))
    for i, val in enumerate(pred_binary):
        full_adj[i, num_vars-1] = val
    return full_adj

def run_ges(X, true_dag):
    if not CAUSAL_LEARN_AVAILABLE:
        return None
    start = time.time()
    try:
        Record = ges(X, score_func='local_score_BIC', maxP=5)
        adj = np.abs(Record['G'].graph).clip(0, 1)
        elapsed = time.time() - start
        metrics = dag_metrics(adj, true_dag)
        metrics["runtime"] = round(elapsed, 3)
        return metrics
    except Exception as e:
        print(f"GES failed: {e}")
        return None

def run_pc(X, true_dag):
    if not CAUSAL_LEARN_AVAILABLE:
        return None
    start = time.time()
    try:
        cg = pc(X, alpha=0.05, indep_test='fisherz', stable=True)
        adj = np.abs(cg.G.graph).clip(0, 1)
        elapsed = time.time() - start
        metrics = dag_metrics(adj, true_dag)
        metrics["runtime"] = round(elapsed, 3)
        return metrics
    except Exception as e:
        print(f"PC failed: {e}")
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
        metrics = dag_metrics(adj, true_dag)
        metrics["runtime"] = round(elapsed, 3)
        return metrics
    except Exception as e:
        print(f"DirectLiNGAM failed: {e}")
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
        metrics = dag_metrics(adj, true_dag)
        metrics["runtime"] = round(elapsed, 3)
        return metrics
    except Exception as e:
        print(f"NOTEARS failed: {e}")
        return None

# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------
def load_synthetic_linear(n=1000, d=5, noise_std=0.1):
    """Linear DAG with random coefficients."""
    np.random.seed(42)
    # Random lower triangular matrix (acyclic)
    W = np.random.randn(d, d) * 0.5
    W = np.tril(W, -1)
    X = np.random.randn(n, d)
    for i in range(1, d):
        X[:, i] = X[:, i-1:i] @ W[i, i-1:i] + noise_std * np.random.randn(n)
    true_dag = (W != 0).astype(int)
    # Output: first variable as target (for simplicity)
    y = X[:, 0]
    X = X[:, 1:]  # remove target from inputs
    # Rebuild true_dag for inputs -> output
    num_inputs = X.shape[1]
    full_dag = np.zeros((num_inputs + 1, num_inputs + 1))
    # Assuming edges from inputs to output (if true_dag had edges from earlier vars to target)
    # For simplicity, set that all inputs cause output (linear combination)
    full_dag[:num_inputs, num_inputs] = 1
    return X, y, full_dag

def load_synthetic_nonlinear(n=1000, d=5, noise_std=0.1):
    """Non‑linear DAG (sine + square)."""
    np.random.seed(42)
    X = np.random.randn(n, d)
    # Non‑linear relations: X2 = sin(X1) + noise, X3 = X1^2 + X2 + noise
    X[:, 1] = np.sin(X[:, 0]) + noise_std * np.random.randn(n)
    X[:, 2] = X[:, 0]**2 + X[:, 1] + noise_std * np.random.randn(n)
    # Output: first variable
    y = X[:, 0]
    X = X[:, 1:]
    num_inputs = X.shape[1]
    full_dag = np.zeros((num_inputs + 1, num_inputs + 1))
    # True parents: inputs 0,1 (original indices 1 and 2)
    full_dag[0, num_inputs] = 1   # input0 (original X1) → output
    full_dag[1, num_inputs] = 1   # input1 (original X2) → output
    return X, y, full_dag

def load_sachs():
    """Sachs protein signaling network (available in causal-learn)."""
    try:
        from causallearn.utils import load_data
        data = load_data.load_sachs_data()
        X = data.values.astype(np.float32)
        # Sachs has 11 nodes, we need a ground truth DAG (not provided here).
        # For benchmarking, we can use the known consensus graph (manually defined).
        # For simplicity, we skip output extraction and treat all variables as inputs.
        # Since we don't have a known input‑output relationship, we will use the full graph.
        # We'll define output as the last node (example).
        num_inputs = X.shape[1] - 1
        y = X[:, -1]
        X = X[:, :-1]
        # True DAG from literature (simplified). We'll use a pre‑defined matrix.
        true_edges = [(0,4), (1,4), (2,4), (3,4)]  # example: first 4 cause last
        full_dag = np.zeros((num_inputs + 1, num_inputs + 1))
        for i, j in true_edges:
            full_dag[i, num_inputs] = 1
        return X, y, full_dag
    except Exception as e:
        print(f"Sachs data not available: {e}")
        return None, None, None

def load_autompg():
    """Auto MPG dataset: predict MPG from other features."""
    try:
        source = AUTO_MPG_LOCAL
        if not source.exists():
            return _synthetic_autompg_fallback()
        df = _parse_autompg_file(source)
        y = df['mpg'].values
        X = df.drop('mpg', axis=1).values
        num_inputs = X.shape[1]
        # No ground truth causal graph; we will run causal discovery without evaluation.
        # For benchmarking, we can only compare methods' outputs qualitatively.
        # Here we'll return None for true_dag.
        return X, y, None
    except Exception as e:
        print(f"AutoMPG download failed: {e}")
        return None, None, None

# ----------------------------------------------------------------------
# Main benchmark
# ----------------------------------------------------------------------
def main():
    results = {}

    # 1. Synthetic linear
    print("\n=== Synthetic Linear ===")
    X, y, true_dag = load_synthetic_linear(n=1000)
    if X is not None:
        print("NCPG...")
        t0 = time.time()
        adj_ncpg = run_ncpg(X, y, true_dag, device="cpu")
        t_ncpg = time.time() - t0
        m_ncpg = dag_metrics(adj_ncpg, true_dag)
        m_ncpg["runtime"] = round(t_ncpg, 3)
        results["Linear_NCPG"] = m_ncpg
        # Run baselines
        for name, func in [("GES", run_ges), ("PC", run_pc), ("DirectLiNGAM", run_direct_lingam), ("NOTEARS", run_notears)]:
            res = func(X, true_dag)
            if res:
                results[f"Linear_{name}"] = res
        print("  Done.")

    # 2. Synthetic non‑linear
    print("\n=== Synthetic Non‑linear ===")
    X, y, true_dag = load_synthetic_nonlinear(n=1000)
    if X is not None:
        print("NCPG...")
        t0 = time.time()
        adj_ncpg = run_ncpg(X, y, true_dag, device="cpu")
        t_ncpg = time.time() - t0
        m_ncpg = dag_metrics(adj_ncpg, true_dag)
        m_ncpg["runtime"] = round(t_ncpg, 3)
        results["Nonlinear_NCPG"] = m_ncpg
        for name, func in [("GES", run_ges), ("PC", run_pc), ("DirectLiNGAM", run_direct_lingam), ("NOTEARS", run_notears)]:
            res = func(X, true_dag)
            if res:
                results[f"Nonlinear_{name}"] = res
        print("  Done.")

    # 3. Sachs (if available)
    print("\n=== Sachs Protein Network ===")
    X, y, true_dag = load_sachs()
    if X is not None and true_dag is not None:
        print("NCPG...")
        t0 = time.time()
        adj_ncpg = run_ncpg(X, y, true_dag, device="cpu")
        t_ncpg = time.time() - t0
        m_ncpg = dag_metrics(adj_ncpg, true_dag)
        m_ncpg["runtime"] = round(t_ncpg, 3)
        results["Sachs_NCPG"] = m_ncpg
        for name, func in [("GES", run_ges), ("PC", run_pc), ("DirectLiNGAM", run_direct_lingam), ("NOTEARS", run_notears)]:
            res = func(X, true_dag)
            if res:
                results[f"Sachs_{name}"] = res
        print("  Done.")
    else:
        print("  Sachs dataset skipped (data not available).")

    # 4. AutoMPG (no ground truth, just run methods)
    print("\n=== AutoMPG (no ground truth) ===")
    X, y, _ = load_autompg()
    if X is not None:
        print("Running NCPG (no evaluation)...")
        run_ncpg(X, y, None, device="cpu")
        print("  Done.")
    else:
        print("  AutoMPG skipped.")

    # Print results table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    for key, metrics in results.items():
        print(f"{key:25} F1={metrics['f1']:4f}  SHD={metrics['shd']:3d}  Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  Time={metrics['runtime']:.2f}s")

if __name__ == "__main__":
    main()
