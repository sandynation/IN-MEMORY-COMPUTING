#!/usr/bin/env python
"""Test the /api/v1/demo/run endpoint"""

import requests
import json
import sys

try:
    print("Testing /api/v1/demo/run endpoint...")
    response = requests.get('http://localhost:8000/api/v1/demo/run', timeout=120)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Endpoint returns 200")
        print(f"  demo_version: {data.get('demo_version')}")
        print(f"  step1_nominal_config: {data.get('step1_nominal_config')}")
        print(f"  step1_nominal_metrics: {data.get('step1_nominal_metrics')}")
        print(f"  step2_physics_delta_report: {data.get('step2_physics_delta_report', {}).get('migration_feasibility')}")
        print(f"  step2_derived_config: {data.get('step2_derived_config')}")
        print(f"  step3_apx2_respin_risk_score: {data.get('step3_apx2_respin_risk_score')}")
        print(f"  step4_apx3_respin_risk_score: {data.get('step4_apx3_respin_risk_score')}")
        print(f"  insight: {data.get('insight')}")
        print(f"  audit_trail_ids: {data.get('audit_trail_ids')}")
        
        # Verify all required fields exist
        required_fields = [
            'demo_version', 'step1_nominal_config', 'step1_nominal_metrics',
            'step2_physics_delta_report', 'step2_derived_config',
            'step3_apx2_respin_risk_score', 'step4_apx3_respin_risk_score',
            'insight', 'audit_trail_ids'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            print(f"✗ Missing fields: {missing}")
            sys.exit(1)
        else:
            print(f"✓ All required fields present")
            
        # Check risk scores
        apx2_score = data.get('step3_apx2_respin_risk_score')
        apx3_score = data.get('step4_apx3_respin_risk_score')
        print(f"  APX-2 score: {apx2_score} (expected ~1.0 or close)")
        print(f"  APX-3 score: {apx3_score} (expected ~0.965)")
        
    else:
        print(f"✗ Endpoint returned {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
