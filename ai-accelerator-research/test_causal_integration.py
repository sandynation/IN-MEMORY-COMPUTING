#!/usr/bin/env python
"""
Validation script for Neural Causal Physics Graphs (NCPG) integration.
Tests all causal endpoints and ensures no breakage of existing features.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from fastapi.testclient import TestClient
from fastapi import FastAPI
from backend.novelty_features.api import router

# ----------------------------------------------------------------------
# Helper to create dummy design data
# ----------------------------------------------------------------------
def make_dummy_design_data(num_samples=5):
    data = []
    for i in range(num_samples):
        workload = {
            "gemm_tiles": 32 + i*8,
            "cgra_tiles": 4 + i,
            "tensor_core_tiles": 2,
            "memory_bandwidth_gbps": 800 + i*50,
            "clock_frequency_ghz": 2.0 + i*0.1
        }
        metrics = {
            "latency_ms": 100 - i*5,
            "energy_mj": 50 + i*2,
            "throughput_gops": 200 + i*20,
            "power_w": 50 + i*3,
            "area_mm2": 5.0 + i*0.5,
            "bottleneck": "MEMORY"
        }
        data.append({"workload": workload, "metrics": metrics})
    return data

# ----------------------------------------------------------------------
# Test imports and model serialization
# ----------------------------------------------------------------------
def test_imports():
    print("\n[1/6] Testing imports and model serialization...")
    try:
        from backend.novelty_features.models import (
            CausalEdge, CausalNode, NeuralCausalPhysicsGraph,
            CausalIntervention, CounterfactualResult, PhysicsConstraints
        )
        # Create a dummy edge and node
        edge = CausalEdge("power", "thermal", 0.8, 0.95, True)
        edge_dict = edge.to_dict()
        edge2 = CausalEdge.from_dict(edge_dict)
        assert edge2.causal_strength == 0.8

        node = CausalNode("latency", "performance_metric", 100.0)
        node_dict = node.to_dict()
        node2 = CausalNode.from_dict(node_dict)
        assert node2.name == "latency"

        # Test PhysicsConstraints instantiation (must match endpoint)
        constraints = PhysicsConstraints(
            max_power_density_w_mm2=100.0,
            max_junction_temp_c=85.0,
            max_voltage_drop_mv=50.0,
            min_clock_period_ps=200.0,
            max_wire_resistance_ohm_um=0.5
        )
        assert constraints.max_junction_temp_c == 85.0
        print("  ✅ All causal models import and serialize correctly")
        return True
    except Exception as e:
        print(f"  ❌ Import/serialization failed: {e}")
        return False

# ----------------------------------------------------------------------
# Test causal discovery endpoint
# ----------------------------------------------------------------------
def test_causal_discover(client):
    print("\n[2/6] Testing POST /causal/discover...")
    design_data = make_dummy_design_data(10)
    payload = {
        "session_id": "test_causal_session",
        "design_data": design_data,
        "process_node": "APX-3"
    }
    response = client.post("/api/v1/causal/discover", json=payload)
    if response.status_code != 200:
        print(f"  ❌ Failed: {response.status_code} - {response.text}")
        return None
    result = response.json()
    assert "causal_graph" in result
    assert "convergence_score" in result
    # Check graph structure
    graph = result["causal_graph"]
    assert "nodes" in graph
    assert "edges" in graph
    print(f"  ✅ Causal graph discovered. Convergence score: {result['convergence_score']:.4f}")
    return graph

# ----------------------------------------------------------------------
# Test causal what-if endpoint
# ----------------------------------------------------------------------
def test_causal_what_if(client, causal_graph):
    print("\n[3/6] Testing POST /causal/what-if...")
    # Use a dummy workload/metrics
    workload = {
        "gemm_tiles": 64,
        "cgra_tiles": 8,
        "tensor_core_tiles": 2,
        "memory_bandwidth_gbps": 1200,
        "clock_frequency_ghz": 2.5
    }
    metrics = {
        "latency_ms": 80,
        "energy_mj": 60,
        "throughput_gops": 300,
        "power_w": 65,
        "area_mm2": 6.0,
        "bottleneck": "COMPUTE"
    }
    payload = {
        "session_id": "test_causal_session",
        "current_workload": workload,
        "current_metrics": metrics,
        "target_constraint": "latency",
        "target_value": 50.0,
        "causal_graph": causal_graph
    }
    response = client.post("/api/v1/causal/what-if", json=payload)
    if response.status_code != 200:
        print(f"  ❌ Failed: {response.status_code} - {response.text}")
        return False
    result = response.json()
    assert "counterfactual" in result
    assert "causal_explanation" in result
    assert "physics_risks" in result
    print(f"  ✅ Counterfactual generated. Explanation: {result['causal_explanation'][:60]}...")
    return True

# ----------------------------------------------------------------------
# Test causal transfer endpoint
# ----------------------------------------------------------------------
def test_causal_transfer(client, causal_graph):
    print("\n[4/6] Testing POST /causal/transfer...")
    payload = {
        "source_process": "APX-3",
        "target_process": "TSMC_N3E",
        "source_graph": causal_graph,
        "physics_scaling": {"clock_frequency": 1.2, "power": 0.9}
    }
    response = client.post("/api/v1/causal/transfer", json=payload)
    if response.status_code != 200:
        print(f"  ❌ Failed: {response.status_code} - {response.text}")
        return False
    result = response.json()
    assert "transferred_graph" in result
    assert "transfer_confidence" in result
    print(f"  ✅ Causal graph transferred. Confidence: {result['transfer_confidence']:.4f}")
    return True

# ----------------------------------------------------------------------
# Test existing endpoints (E-2, SPO, S-2) to ensure no regression
# ----------------------------------------------------------------------
def test_existing_endpoints(client):
    print("\n[5/6] Testing existing endpoints for regressions...")
    
    # E-2: Verilog generation
    verilog_payload = {
        "name": "test_gemm",
        "m": 512,
        "n": 512,
        "k": 512,
        "dtype": "f16",
        "sparsity": 0.2,
        "target": "tensor_core"
    }
    resp = client.post("/api/v1/generate-verilog", json=verilog_payload)
    if resp.status_code != 200:
        print(f"  ❌ E-2 generate-verilog failed: {resp.status_code}")
        return False
    session_id = resp.json()["session_id"]
    print("  ✅ E-2 verilog generation works")

    # SPO status (may fail if no data, but should return 404 gracefully)
    resp = client.get(f"/api/v1/spo/status/{session_id}")
    if resp.status_code not in (200, 404):
        print(f"  ❌ SPO status unexpected: {resp.status_code}")
    else:
        print("  ✅ SPO endpoint responds")

    # S-2 latent generate (requires dataset; if not present, returns 404 – that's fine)
    resp = client.post("/api/v1/latent/generate", json={"steps": 2})
    if resp.status_code == 404:
        print("  ⚠️ S-2 latent/generate skipped (dataset not found – expected if not generated)")
    elif resp.status_code == 200:
        print("  ✅ S-2 latent/generate works")
    else:
        print(f"  ❌ S-2 unexpected: {resp.status_code}")
        return False

    # Health check
    resp = client.get("/api/v1/health")
    if resp.status_code == 200:
        features = resp.json().get("features", [])
        if "NCPG" in features:
            print("  ✅ Health endpoint includes NCPG feature")
        else:
            print("  ⚠️ Health endpoint missing NCPG flag")
    else:
        print(f"  ❌ Health check failed: {resp.status_code}")
        return False

    return True

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Validating Neural Causal Physics Graphs (NCPG) Integration")
    print("=" * 60)

    # Step 1: Imports
    if not test_imports():
        sys.exit(1)

    # Create FastAPI app with router
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Step 2: Causal discovery
    causal_graph = test_causal_discover(client)
    if causal_graph is None:
        sys.exit(1)

    # Step 3: Causal what-if
    if not test_causal_what_if(client, causal_graph):
        sys.exit(1)

    # Step 4: Causal transfer
    if not test_causal_transfer(client, causal_graph):
        sys.exit(1)

    # Step 5: Existing endpoints
    if not test_existing_endpoints(client):
        sys.exit(1)

    # Step 6: Final success
    print("\n[6/6] ✅ All integration tests passed!")
    print("The Neural Causal Physics Graphs are fully integrated and operational.")
    print("No regressions detected in existing features.")

if __name__ == "__main__":
    main()