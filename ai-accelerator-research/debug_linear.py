import torch
import numpy as np
from backend.novelty_features.experiments.ncpg_full import NCPG

# Generate data
np.random.seed(42)
n = 500
m = np.random.uniform(1,10,n)
n_ = np.random.uniform(1,10,n)
k = np.random.uniform(1,10,n)
y = m + n_ + np.random.normal(0,0.05,n)

X = torch.tensor(np.column_stack([m, n_, k]), dtype=torch.float32)
Y = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

model = NCPG(input_dim=3, output_dim=1, device="cpu")
# Train with only data loss (no causal losses)
model.train(X, Y, lr=0.01, alpha_geo=0, alpha_top=0, lambda_dag=0, num_epochs=200, verbose=True)
print("Final weights:", model.scm.W.detach().numpy())
print("Adjacency (threshold=0.01):", (np.abs(model.scm.W.detach().numpy()) > 0.01).flatten())