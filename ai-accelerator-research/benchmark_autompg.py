#!/usr/bin/env python3
"""
Run NCPG on the AutoMPG dataset (real regression).
Predict MPG from vehicle attributes. Compare with known causal structure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from backend.novelty_features.ncpg_full import NCPG

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
    df = df.dropna()
    return df

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
    mpg = (
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
    return X, mpg

def load_autompg():
    # The dataset has 9 columns: mpg, cylinders, displacement, horsepower, weight, acceleration, model_year, origin, car_name
    # We will ignore the last column (car_name)
    source = AUTO_MPG_LOCAL
    if not source.exists():
        return _synthetic_autompg_fallback()

    df = _parse_autompg_file(source)
    # Extract target and features (exclude car_name)
    y = df['mpg'].values
    X = df.drop(['mpg', 'car_name'], axis=1).values
    return X, y

def main():
    X, y = load_autompg()
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Known causal parents from literature: weight, displacement, cylinders
    # Indices: after dropping car_name and mpg, columns are: cylinders(0), displacement(1), horsepower(2), weight(3), acceleration(4), model_year(5), origin(6)
    true_parents = [3, 1, 0]  # weight, displacement, cylinders
    true_binary = np.zeros(X.shape[1])
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

    print("\nAutoMPG results (known causal parents: weight, displacement, cylinders):")
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
