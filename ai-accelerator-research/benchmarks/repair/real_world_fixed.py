#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from backend.novelty_features.ncpg_full import NCPG

def load_automp_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    local_path = Path("auto-mpg.data")
    if not local_path.exists():
        import urllib.request
        print("Downloading AutoMPG...")
        urllib.request.urlretrieve(url, local_path)
    col_names = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
    df = pd.read_csv(local_path, sep=r'\s+', names=col_names, na_values='?')
    df.dropna(inplace=True)
    X = df[['cylinders','displacement','horsepower','weight','acceleration','model_year','origin']].values.astype(np.float32)
    y = df['mpg'].values.astype(np.float32)
    true_parents = ['weight','displacement','cylinders']
    features = ['cylinders','displacement','horsepower','weight','acceleration','model_year','origin']
    return X, y, true_parents, features

def generate_hardware_dataset(n_samples=500):
    np.random.seed(42)
    m = np.random.randint(8,33,n_samples)
    n = np.random.randint(8,33,n_samples)
    k = np.random.randint(8,33,n_samples)
    cgra = np.random.randint(4,17,n_samples)
    clk = np.random.uniform(1,3,n_samples)
    latency = 1000/(m+0.1) + 3*cgra + 8*clk**2 + 0.02*np.random.randn(n_samples)
    X = np.column_stack([m,n,k,cgra,clk])
    y = latency.astype(np.float32)
    true_parents = ['m','cgra','clk']
    features = ['m','n','k','cgra','clk']
    return X, y, true_parents, features

def evaluate_linear(X,y,true_parents,features):
    lr = LinearRegression().fit(X,y)
    imp = np.abs(lr.coef_)
    top3 = np.argsort(imp)[::-1][:3]
    discovered = [features[i] for i in top3]
    tp = sum(1 for f in discovered if f in true_parents)
    fp = sum(1 for f in discovered if f not in true_parents)
    fn = sum(1 for f in true_parents if f not in discovered)
    prec = tp/(tp+fp+1e-8)
    rec = tp/(tp+fn+1e-8)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    return {'precision':prec,'recall':rec,'f1':f1,'discovered':discovered}

def evaluate_ncpg(X,y,true_parents,features):
    Xt = torch.tensor(X)
    yt = torch.tensor(y.reshape(-1,1))
    model = NCPG(num_nodes=X.shape[1], output_dim=1, use_mlp=True, force_simple=True, device='cpu')
    model.fit(Xt, yt, num_epochs=300, verbose=False)
    importances = []
    model.model.eval()
    for i in range(X.shape[1]):
        xs = torch.tensor(X[:100], requires_grad=True)
        yp = model.model(xs)
        grad = torch.autograd.grad(yp.sum(), xs, create_graph=False)[0]
        imp = grad[:,i].abs().mean().item()
        importances.append(imp)
    top3 = np.argsort(importances)[::-1][:3]
    discovered = [features[i] for i in top3]
    tp = sum(1 for f in discovered if f in true_parents)
    fp = sum(1 for f in discovered if f not in true_parents)
    fn = sum(1 for f in true_parents if f not in discovered)
    prec = tp/(tp+fp+1e-8)
    rec = tp/(tp+fn+1e-8)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    return {'precision':prec,'recall':rec,'f1':f1,'discovered':discovered}

def main():
    print("\n=== Real-World Causal Discovery ===\n")
    Xa, ya, ta, fa = load_automp_dataset()
    res_a = evaluate_linear(Xa, ya, ta, fa)
    print("AutoMPG (linear proxy):")
    print(f"  True: {ta}\n  Discovered: {res_a['discovered']}\n  F1 = {res_a['f1']:.2f}\n")
    Xh, yh, th, fh = generate_hardware_dataset()
    res_h = evaluate_ncpg(Xh, yh, th, fh)
    print("Synthetic Hardware (NCPG):")
    print(f"  True: {th}\n  Discovered: {res_h['discovered']}\n  F1 = {res_h['f1']:.2f}\n")
    pd.DataFrame([res_a, res_h], index=['AutoMPG','Hardware']).to_csv('causal_discovery_results.csv')
    print("Saved to causal_discovery_results.csv")

if __name__ == '__main__':
    main()