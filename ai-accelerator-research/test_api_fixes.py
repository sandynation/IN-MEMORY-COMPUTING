#!/usr/bin/env python3
"""
Test all API endpoints for 500 errors using FastAPI TestClient.
No need to run uvicorn manually.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from backend.novelty_features.api import app

client = TestClient(app)

# ----------------------------------------------------------------------
# Helper to print test results
# ----------------------------------------------------------------------
def test_endpoint(name, method, url, json_data=None):
    print(f"\nTesting: {name} ({method} {url})")
    if method == "GET":
        response = client.get(url)
    elif method == "POST":
        response = client.post(url, json=json_data)
    else:
        return
    status = response.status_code
    if status == 500:
        print(f"  FAIL: 500 Internal Server Error")
        print(f"     Response: {response.text[:200]}")
    elif status >= 400:
        print(f"  WARN: {status} Client Error (input issue - not a backend bug)")
    else:
        print(f"  OK: {status}")
    return response

# ----------------------------------------------------------------------
# Test all endpoints that previously had 500 errors
# ----------------------------------------------------------------------
print("=" * 60)
print("Testing SparseX API Endpoints (no live server needed)")
print("=" * 60)

# 1. Health (should work)
test_endpoint("Health", "GET", "/api/v1/health")

# 2. Foundry nodes (should work)
test_endpoint("Foundry nodes", "GET", "/api/v1/foundry-nodes")

# 3. Generate Verilog (should work)
verilog_payload = {
    "name": "test_gemm",
    "m": 128, "n": 128, "k": 128,
    "dtype": "f16", "sparsity": 0.2,
    "target": "tensor_core"
}
resp = test_endpoint("Generate Verilog", "POST", "/api/v1/generate-verilog", verilog_payload)
session_id = resp.json().get("session_id") if resp and resp.status_code == 200 else None

# 4. Validate Verilog (if session exists)
if session_id:
    test_endpoint("Validate Verilog", "POST", f"/api/v1/validate-verilog/{session_id}")
    test_endpoint("Synthesis Report", "GET", f"/api/v1/synthesis-report/{session_id}")

# 5. Physics guidance (previously 500)
test_endpoint("Physics guidance generate", "POST", "/api/v1/physics-guidance/generate", {
    "session_id": "test123",
    "current_workload": {"gemm_tiles": 32, "cgra_tiles": 4, "tensor_core_tiles": 2,
                         "memory_bandwidth_gbps": 800, "clock_frequency_ghz": 2.0},
    "current_metrics": {"latency_ms": 100, "energy_mj": 50, "throughput_gops": 200,
                        "power_w": 50, "area_mm2": 5, "bottleneck": "MEMORY"},
    "process_node": "TSMC_N3E",
    "num_candidates": 10
})

# 6. Physics guidance validate (previously 500)
test_endpoint("Physics guidance validate", "POST", "/api/v1/physics-guidance/validate", {
    "workload": {"gemm_tiles": 32, "cgra_tiles": 4, "tensor_core_tiles": 2,
                 "memory_bandwidth_gbps": 800, "clock_frequency_ghz": 2.0},
    "process_node": "TSMC_N3E"
})

# 7. Strategy recommendation (previously 500)
test_endpoint("Strategy recommendation", "POST", "/api/v1/analytics/strategy-recommendation", {
    "workload": {"gemm_tiles": 32, "cgra_tiles": 4, "tensor_core_tiles": 2,
                 "memory_bandwidth_gbps": 800, "clock_frequency_ghz": 2.0},
    "session_id": "test123"
})

# 8. NLP parse constraints (previously 500)
test_endpoint("NLP parse constraints", "POST", "/api/v1/nlp/parse-constraints", {
    "query": "reduce power by 20%",
    "current_workload": {"gemm_tiles": 32, "cgra_tiles": 4, "tensor_core_tiles": 2,
                         "memory_bandwidth_gbps": 800, "clock_frequency_ghz": 2.0}
})

# 9. NLP what-if (previously 500)
test_endpoint("NLP what-if", "POST", "/api/v1/nlp/what-if", {
    "query": "increase clock frequency to 2.5 GHz",
    "current_workload": {"gemm_tiles": 32, "cgra_tiles": 4, "tensor_core_tiles": 2,
                         "memory_bandwidth_gbps": 800, "clock_frequency_ghz": 2.0},
    "iterations": 3
})

# 10. Thermal simulation (previously 500)
test_endpoint("Thermal simulation", "POST", "/api/v1/thermal-simulation", {
    "power_map": [[50, 80], [60, 90]],
    "ambient_temp": 25.0
})

# 11. Causal train (should work)
test_endpoint("Causal train", "POST", "/api/v1/causal/train", {"num_epochs": 5})

# 12. Causal discover (needs proper input – returns 400 but not 500)
test_endpoint("Causal discover (with correct payload)", "POST", "/api/v1/causal/discover", {
    "session_id": "test",
    "design_data": [{"workload": {}, "metrics": {}}],  # minimal valid
    "process_node": "APX-3"
})

# 13. Causal transfer (needs proper input – returns 400 but not 500)
test_endpoint("Causal transfer (with correct payload)", "POST", "/api/v1/causal/transfer", {
    "source_process": "APX-3",
    "target_process": "TSMC_N3E",
    "source_graph": {"nodes": [], "edges": []},
    "physics_scaling": {}
})

print("\n" + "=" * 60)
print("Test complete. No 500 errors should appear (only 400 for missing keys).")
print("=" * 60)
