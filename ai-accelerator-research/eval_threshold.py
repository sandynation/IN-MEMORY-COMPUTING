import torch
import numpy as np
from backend.novelty_features.benchmarking.ncpg_full import NCPG
from backend.novelty_features.benchmarking.ncpg_optuna_tuning import load_dataset
from torch.func import vmap, jacrev

# Load data
X_train, y_train, X_val, y_val, X_test, y_test, true_dag = load_dataset(2000, 0.05)

num_inputs = X_train.shape[1]
print(f"Number of input features: {num_inputs}")

# Debug true_dag
print(f"Type of true_dag: {type(true_dag)}")
if hasattr(true_dag, 'shape'):
    print(f"Shape of true_dag: {true_dag.shape}")
else:
    print(f"true_dag (first few elements): {true_dag[:5] if len(true_dag) > 5 else true_dag}")

# ---------- Extract true parents ----------
# If true_dag is a square matrix (num_inputs+1) x (num_inputs+1), output is last column
if hasattr(true_dag, 'shape') and len(true_dag.shape) == 2 and true_dag.shape[0] == true_dag.shape[1]:
    output_idx = true_dag.shape[0] - 1
    true_parents = [i for i in range(num_inputs) if true_dag[i, output_idx] > 0.5]
    print(f"True parents from DAG matrix: {true_parents}")
else:
    # Fallback: assume first 4 features are parents (heuristic for physics-hard dataset)
    true_parents = list(range(min(4, num_inputs)))
    print(f"Using heuristic true parents: {true_parents}")

true_binary = np.zeros(num_inputs)
true_binary[true_parents] = 1

# ---------- Train model ----------
model = NCPG(
    num_nodes=num_inputs,
    output_dim=1,
    hidden_dim=256,
    use_mlp=True,
    alpha_geo=0.01,
    alpha_top=0.0,
    lambda_dag=0.0,
    lambda_sparse=0.0,
    lr=0.01,
    device='cpu'
)

print("\nTraining...")
model.fit(X_train, y_train, num_epochs=2000, batch_size=256, verbose=True)

# ---------- Compute Jacobian ----------
model.model.eval()
X_test_t = X_test.to('cpu').float()

def scalar_output(x_single):
    return model.model(x_single.unsqueeze(0)).squeeze()

with torch.no_grad():
    J = vmap(jacrev(scalar_output))(X_test_t)          # (batch, num_inputs)
    mean_abs_jac = J.abs().mean(dim=0).cpu().numpy()   # (num_inputs,)

print(f"\nMean absolute Jacobian per input: {mean_abs_jac}")

# ---------- Evaluate at thresholds ----------
print("\nInput-output causal discovery performance:")
for th in [0.01, 0.02, 0.05, 0.1, 0.2]:
    pred_binary = (mean_abs_jac > th).astype(int)
    tp = ((pred_binary == 1) & (true_binary == 1)).sum()
    fp = ((pred_binary == 1) & (true_binary == 0)).sum()
    fn = ((pred_binary == 0) & (true_binary == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    shd = fp + fn
    print(f"  threshold={th:4.2f}: F1={f1:.4f}, SHD={shd}, precision={prec:.3f}, recall={rec:.3f}")

# Show strongest predicted parents
print("\nPredicted causal strengths (top 3 inputs):")
sorted_idx = np.argsort(mean_abs_jac)[::-1]
for i in sorted_idx[:3]:
    print(f"  Input {i}: Jacobian = {mean_abs_jac[i]:.4f} (true parent: {i in true_parents})")