#!/usr/bin/env python3
"""Quick E-2 Verilog Export Verification"""

import sys
sys.path.insert(0, 'backend')

from backend.novelty_features.api import router
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_e2_endpoints():
    print("🧪 Testing E-2 Verilog Export...")
    
    # 1. Generate Verilog
    print("\n1️⃣ Testing /generate-verilog")
    resp = client.post("/api/v1/generate-verilog", json={
        "name": "test_gemm",
        "m": 512, "n": 512, "k": 512,
        "dtype": "f16", "sparsity": 0.0,
        "target": "tensor_core"
    })
    print(f"   Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"   Error: {resp.text}")
        return False
    
    data = resp.json()
    session_id = data.get("session_id")
    files = data.get("files", {})
    print(f"   Session ID: {session_id}")
    print(f"   Generated files: {list(files.keys())}")
    
    # Check if files have content
    for fname, content in files.items():
        if len(content) < 100:
            print(f"   ⚠️  {fname} seems small: {len(content)} chars")
        else:
            print(f"   ✅ {fname}: {len(content)} chars")
    
    # 2. Validate Verilog with Yosys
    print("\n2️⃣ Testing /validate-verilog (with Yosys)")
    resp = client.post(f"/api/v1/validate-verilog/{session_id}")
    print(f"   Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"   Error: {resp.text}")
        return False
    
    validation = resp.json()
    print(f"   Validation complete: {validation.get('message')}")
    
    # 3. Get synthesis report
    print("\n3️⃣ Testing /synthesis-report")
    resp = client.get(f"/api/v1/synthesis-report/{session_id}")
    print(f"   Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"   Error: {resp.text}")
        return False
    
    report = resp.json()
    print(f"   Synthesis passed: {report.get('passed')}")
    print(f"   Errors: {len(report.get('errors', []))}")
    print(f"   Warnings: {len(report.get('warnings', []))}")
    if report.get('area_um2'):
        print(f"   Area: {report['area_um2']:.2f} μm²")
    if report.get('max_freq_mhz'):
        print(f"   Max freq: {report['max_freq_mhz']:.1f} MHz")
    
    print("✅ E-2 Verilog Export working!")
    return True

if __name__ == "__main__":
    success = test_e2_endpoints()
    sys.exit(0 if success else 1)
