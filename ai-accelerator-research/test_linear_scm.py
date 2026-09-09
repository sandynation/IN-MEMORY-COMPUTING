import torch
import torch.nn as nn
import numpy as np

# Generate known-causal dataset
np.random.seed(42)
n_samples = 500
m = np.random.uniform(1, 10, n_samples)
n = np.random.uniform(1, 10, n_samples)
k = np.random.uniform(1, 10, n_samples)
y = m + n + np.random.normal(0, 0.05, n_samples)

X = torch.tensor(np.column_stack([m, n, k]), dtype=torch.float32)
y_true = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

# Linear SCM (single layer)
model = nn.Linear(3, 1, bias=False)  # y = X @ W
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, loss={loss.item():.6f}")

print("Learned weights:", model.weight.detach().numpy())
print("Expected weights: [1, 1, 0]")