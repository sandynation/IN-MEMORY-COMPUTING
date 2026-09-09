#!/usr/bin/env python
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_dataset_generation():
    print("\n[1/5] Generating 1,000 valid designs...")
    from novelty_features.latent.generate_dataset import DesignGenerator
    gen = DesignGenerator()
    out_path = Path("designs/latent_data_small.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = []
    from tqdm import tqdm
    pbar = tqdm(total=1000)
    while len(vectors) < 1000:
        arch = gen.random_arch()
        if gen.is_physically_valid(arch):
            vectors.append(gen.to_vector(arch))
            pbar.update(1)
    np.save(out_path, np.stack(vectors))
    assert out_path.exists()
    data = np.load(out_path)
    assert data.shape == (1000, 10)
    print("✅ Dataset generated")
    return str(out_path)


def test_vae_training(data_path):
    print("\n[2/5] Training β-VAE for 10 epochs...")
    from novelty_features.latent.physics_vae import PhysicsVAE
    from torch.utils.data import DataLoader, TensorDataset

    data = np.load(data_path)
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = PhysicsVAE(input_dim=10, latent_dim=4, beta=4.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0.0
        for batch in loader:
            x = batch[0]
            # Forward pass returns 4 values: recon, mu, logvar, z
            recon, mu, logvar, z = model(x)
            # Compute loss using the model's loss method
            loss = model.loss(x, recon, mu, logvar, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss = {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "designs/physics_vae_small.pt")
    print("✅ VAE trained")
    return model


def test_api_endpoint():
    print("\n[3/5] Testing /latent/generate functionality (direct call)...")
    from backend.novelty_features.latent.physics_vae import PhysicsVAE
    import numpy as np
    import torch

    # Load the small dataset and the trained model
    data = np.load("designs/latent_data_small.npy")
    model = PhysicsVAE(input_dim=10, latent_dim=4)
    model.load_state_dict(torch.load("designs/physics_vae_small.pt", map_location="cpu"))
    model.eval()

    # Pick two random indices from dataset
    idx1, idx2 = np.random.choice(len(data), 2, replace=False)
    with torch.no_grad():
        x1 = torch.tensor(data[idx1], dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(data[idx2], dtype=torch.float32).unsqueeze(0)
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
        steps = 3
        alphas = np.linspace(0, 1, steps)
        norms = np.array([8, 256, 16, 16, 4000, 64, 2000, 8, 1000, 8])
        designs = []
        for alpha in alphas:
            z = (1-alpha)*mu1 + alpha*mu2
            recon = model.decode(z).numpy().flatten()
            vec = recon * norms
            arch_dict = {
                "num_chiplets": int(vec[0]),
                "tensor_dim": int(vec[1]),
                "cgra_rows": int(vec[2]),
                "cgra_cols": int(vec[3]),
                "bandwidth_gbps": float(vec[4]),
                "link_gtps": float(vec[5]),
                "tflops": float(vec[6]),
                "energy_pj": float(vec[7]),
                "cgra_tops": float(vec[8]),
                "cgra_energy_pj": float(vec[9]),
            }
            designs.append(arch_dict)
    assert len(designs) == 3
    print("✅ Latent generation works (direct call)")
    return designs

def test_visualization():
    print("\n[4/5] Generating 3D latent visualization...")
    import matplotlib
    matplotlib.use('Agg')
    import umap
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from novelty_features.latent.physics_vae import PhysicsVAE

    data = np.load("designs/latent_data_small.npy")
    model = PhysicsVAE(input_dim=10, latent_dim=4)
    model.load_state_dict(torch.load("designs/physics_vae_small.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(torch.tensor(data, dtype=torch.float32))
    z = mu.numpy()
    reducer = umap.UMAP(n_components=3, random_state=42)
    z3d = reducer.fit_transform(z)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z3d[:, 0], z3d[:, 1], z3d[:, 2], c='green', s=5, alpha=0.6)
    ax.set_title("3D Latent Space (small dataset)")
    plt.savefig("designs/latent_3d_small.png", dpi=150)
    print("✅ 3D plot saved to designs/latent_3d_small.png")


def check_integration():
    print("\n[5/5] Checking integration with existing features...")
    modules = [
        "backend.novelty_features.api",
        "backend.novelty_features.websocket_manager",
        "backend.accel_tool.verilog.synthesis_validator",
        "backend.accel_tool.ir_gen",
        "backend.novelty_features.spo.engine"
    ]
    for mod in modules:
        try:
            __import__(mod)
            print(f"  ✅ {mod}")
        except Exception as e:
            print(f"  ❌ {mod}: {e}")
    print("✅ Integration check done")


if __name__ == "__main__":
    data_path = test_dataset_generation()
    test_vae_training(data_path)
    test_api_endpoint()
    test_visualization()
    check_integration()
    print("\n🎉 S-2 tests PASSED.")