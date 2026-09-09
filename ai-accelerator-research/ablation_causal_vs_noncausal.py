#!/usr/bin/env python3
# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# SparseX – Neural Causal Silicon Compiler
"""
Ablation study: NCPG (full) vs. NCPG without geometric loss.
Compares performance prediction accuracy on the physics‑hard dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error
from backend.novelty_features.ncpg_full import NCPG
from backend.novelty_features.benchmarking.physics_hard_dataset import generate_physics_hard_dataset
from sklearn.model_selection import train_test_split

def main():
    # Load dataset
    data, _ = generate_physics_hard_dataset(n_samples=1000, noise_std=0.05)
    X = data.drop("y", axis=1).values.astype(np.float32)
    y = data["y"].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).reshape(-1,1)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).reshape(-1,1)

    # Full NCPG (with geometric loss)
    model_full = NCPG(num_nodes=X.shape[1], output_dim=1, use_mlp=True, force_simple=True,
                      alpha_geo=0.1, alpha_top=0.05, lambda_dag=0.01, lambda_sparse=0.001,
                      device="cpu")
    model_full.fit(X_train_t, y_train_t, num_epochs=500, verbose=False)
    pred_full = model_full.predict(X_test_t).numpy()
    r2_full = r2_score(y_test, pred_full)
    mse_full = mean_squared_error(y_test, pred_full)

    # NCPG without geometric loss (alpha_geo=0)
    model_no_geo = NCPG(num_nodes=X.shape[1], output_dim=1, use_mlp=True, force_simple=True,
                        alpha_geo=0.0, alpha_top=0.05, lambda_dag=0.01, lambda_sparse=0.001,
                        device="cpu")
    model_no_geo.fit(X_train_t, y_train_t, num_epochs=500, verbose=False)
    pred_no_geo = model_no_geo.predict(X_test_t).numpy()
    r2_no_geo = r2_score(y_test, pred_no_geo)
    mse_no_geo = mean_squared_error(y_test, pred_no_geo)

    print("Physics‑hard performance prediction ablation:")
    print(f"  Full NCPG:        R² = {r2_full:.4f}, MSE = {mse_full:.4f}")
    print(f"  NCPG (no L_geo):  R² = {r2_no_geo:.4f}, MSE = {mse_no_geo:.4f}")

if __name__ == "__main__":
    main()