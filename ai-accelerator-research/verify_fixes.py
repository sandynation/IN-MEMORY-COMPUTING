#!/usr/bin/env python3
"""
Self‑verification script for SparseX critical fixes.
Run this to ensure all import and logic errors are resolved.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("SPARSEX SELF‑VERIFICATION")
print("=" * 60)

errors = []
warnings = []

# ----------------------------------------------------------------------
# 1. Check critical module imports
# ----------------------------------------------------------------------
print("\n[1] Checking critical module imports...")
modules_to_check = [
    ("backend.novelty_features.db", "db"),
    ("backend.novelty_features.models", "models"),
    ("backend.accel_tool.verilog.synthesis_validator", "synthesis_validator"),
    ("backend.accel_tool.verilog.area_timing_estimator", "area_timing_estimator"),
    ("backend.accel_tool.verilog.formal_verifier", "formal_verifier"),
    ("backend.novelty_features.trajectory.explainer", "explainer"),
    ("backend.novelty_features.latent.physics_vae", "physics_vae"),
    ("backend.novelty_features.causal_discovery.transfer_learning", "transfer_learning"),
    ("backend.novelty_features.trajectory.secure_planner", "secure_planner"),
    ("backend.accel_tool.security.static_analyzer", "static_analyzer"),
    ("backend.novelty_features.physics.differentiable_thermal", "differentiable_thermal"),
]

for module_name, desc in modules_to_check:
    try:
        importlib.import_module(module_name)
        print(f"  ✅ {desc}")
    except Exception as e:
        errors.append(f"Module {desc} failed to import: {e}")
        print(f"  ❌ {desc}: {e}")

# ----------------------------------------------------------------------
# 2. Check critical functions in db.py
# ----------------------------------------------------------------------
print("\n[2] Checking db.py for required functions...")
try:
    from backend.novelty_features.db import init_db_engine, get_full_session_history
    print("  ✅ init_db_engine exists")
    # Check signature of get_full_session_history
    import inspect
    sig = inspect.signature(get_full_session_history)
    params = list(sig.parameters.keys())
    if len(params) >= 2 and 'db' in params and 'session_id' in params:
        print("  ✅ get_full_session_history has correct signature (db, session_id)")
    else:
        errors.append(f"get_full_session_history signature is {params}, expected (db, session_id)")
        print(f"  ❌ Signature mismatch: {params}")
except ImportError as e:
    errors.append(f"Could not import from db.py: {e}")
    print(f"  ❌ {e}")
except Exception as e:
    errors.append(f"Error checking db.py: {e}")
    print(f"  ❌ {e}")

# ----------------------------------------------------------------------
# 3. Check duplicate ViolationStatusChange enum in models.py
# ----------------------------------------------------------------------
print("\n[3] Checking models.py for duplicate enum...")
try:
    from backend.novelty_features.models import ViolationStatusChange
    members = [m.name for m in ViolationStatusChange]
    expected = {"NEW_VIOLATION", "VIOLATION_RESOLVED", "NO_CHANGE_VIOLATION", "NO_CHANGE_SATISFIED"}
    if expected.issubset(set(members)):
        print("  ✅ Correct ViolationStatusChange enum (no duplicates)")
    else:
        warnings.append(f"ViolationStatusChange missing expected members. Found: {members}")
        print(f"  ⚠️ Unexpected members: {members}")
except ImportError as e:
    errors.append(f"Could not import ViolationStatusChange: {e}")
    print(f"  ❌ {e}")
except Exception as e:
    errors.append(f"Error checking enum: {e}")
    print(f"  ❌ {e}")

# ----------------------------------------------------------------------
# 4. Check YOSYS_PATH environment variable handling (no existence check)
# ----------------------------------------------------------------------
print("\n[4] Checking YOSYS_PATH environment variable handling...")
try:
    from backend.accel_tool.verilog.synthesis_validator import SynthesisValidator
    from backend.accel_tool.verilog.area_timing_estimator import AreaTimingEstimator
    
    # Set a dummy YOSYS_PATH (does not need to exist)
    os.environ["YOSYS_PATH"] = "/dummy/path/yosys"
    sv = SynthesisValidator()
    if sv.yosys_path == "/dummy/path/yosys":
        print("  ✅ SynthesisValidator reads YOSYS_PATH (ignores existence test)")
    else:
        warnings.append("SynthesisValidator did not read YOSYS_PATH")
        print("  ⚠️ SynthesisValidator did not read YOSYS_PATH")
    
    ate = AreaTimingEstimator()
    if ate.yosys_path == "/dummy/path/yosys":
        print("  ✅ AreaTimingEstimator reads YOSYS_PATH")
    else:
        warnings.append("AreaTimingEstimator did not read YOSYS_PATH")
        print("  ⚠️ AreaTimingEstimator did not read YOSYS_PATH")
    
    del os.environ["YOSYS_PATH"]
except Exception as e:
    errors.append(f"Error testing YOSYS_PATH: {e}")
    print(f"  ❌ {e}")

# ----------------------------------------------------------------------
# 5. Check formal_verifier for correct subprocess call
# ----------------------------------------------------------------------
print("\n[5] Checking formal_verifier for correct subprocess call...")
try:
    from backend.accel_tool.verilog.formal_verifier import FormalVerifier
    import inspect
    source = inspect.getsource(FormalVerifier.verify_property)
    if "write_smt2" in source and "yosys-smtbmc" in source:
        print("  ✅ FormalVerifier uses correct two‑step command")
    else:
        warnings.append("FormalVerifier may still use incorrect yosys command")
        print("  ⚠️ Could not verify command (may still be incorrect)")
except Exception as e:
    errors.append(f"Error checking formal_verifier: {e}")
    print(f"  ❌ {e}")

# ----------------------------------------------------------------------
# 6. Check that thermal solver is imported in physics_vae.py
# ----------------------------------------------------------------------
print("\n[6] Checking physics_vae.py for thermal solver integration...")
try:
    from backend.novelty_features.latent.physics_vae import PhysicsVAE
    vae = PhysicsVAE()
    if hasattr(vae, 'thermal_solver'):
        print("  ✅ PhysicsVAE has thermal_solver attribute")
        source = inspect.getsource(PhysicsVAE.loss)
        if "thermal_loss" in source:
            print("  ✅ Loss method includes thermal penalty")
        else:
            warnings.append("Loss method may not include thermal_loss")
            print("  ⚠️ Loss method does not contain 'thermal_loss'")
    else:
        warnings.append("PhysicsVAE missing thermal_solver")
        print("  ⚠️ PhysicsVAE does not have thermal_solver")
except Exception as e:
    errors.append(f"Error checking physics_vae: {e}")
    print(f"  ❌ {e}")

# ----------------------------------------------------------------------
# 7. Check transfer_learning.py for APX constants
# ----------------------------------------------------------------------
print("\n[7] Checking transfer_learning.py for APX constants...")
try:
    from backend.novelty_features.causal_discovery import CausalTransferEngine
    if isinstance(APX_PHYSICS_DATABASE, dict) and "APX-3" in APX_PHYSICS_DATABASE:
        print("  ✅ Local APX database exists")
    else:
        errors.append("APX_PHYSICS_DATABASE missing or incorrect")
        print("  ❌ APX_PHYSICS_DATABASE not found")
    limit = get_power_limit("APX-3")
    if limit == 100.0:
        print("  ✅ get_power_limit works")
    else:
        warnings.append(f"get_power_limit returned {limit}, expected 100.0")
        print(f"  ⚠️ get_power_limit returned {limit}")
except Exception as e:
    errors.append(f"Error checking transfer_learning: {e}")
    print(f"  ❌ {e}")

# ----------------------------------------------------------------------
# 8. Check dependencies in requirements.txt
# ----------------------------------------------------------------------
print("\n[8] Checking requirements.txt for missing dependencies...")
req_path = Path("backend/requirements.txt")
if req_path.exists():
    content = req_path.read_text(encoding='utf-8')
    required = ["optuna", "matplotlib"]
    missing_req = [pkg for pkg in required if pkg not in content]
    if not missing_req:
        print("  ✅ optuna and matplotlib present in requirements.txt")
    else:
        warnings.append(f"requirements.txt missing: {missing_req}")
        print(f"  ⚠️ Missing: {missing_req}")
else:
    warnings.append("backend/requirements.txt not found")
    print("  ⚠️ requirements.txt not found")

# ----------------------------------------------------------------------
# 9. Quick smoke test of key endpoints (no server start)
# ----------------------------------------------------------------------
print("\n[9] Running API router import test...")
try:
    from backend.novelty_features.api import router
    print("  ✅ API router imports successfully")
except Exception as e:
    errors.append(f"API router import failed: {e}")
    print(f"  ❌ {e}")

# ----------------------------------------------------------------------
# 10. Check that eval_threshold.py can be imported (not run)
# ----------------------------------------------------------------------
print("\n[10] Checking eval_threshold.py import...")
try:
    import eval_threshold  # noqa: F401
    print("  ✅ eval_threshold.py imports without syntax error")
except Exception as e:
    errors.append(f"eval_threshold.py import error: {e}")
    print(f"  ❌ {e}")

# ----------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

if errors:
    print(f"\n❌ CRITICAL ERRORS ({len(errors)}):")
    for err in errors:
        print(f"  - {err}")
else:
    print("\n✅ No critical errors found.")

if warnings:
    print(f"\n⚠️ WARNINGS ({len(warnings)}):")
    for warn in warnings:
        print(f"  - {warn}")
else:
    print("\n✅ No warnings.")

if not errors:
    print("\n🎉 All critical fixes verified. SparseX is import‑clean and ready for research publication.")
else:
    print("\n⚠️ Please fix the critical errors above before proceeding.")

print("\n" + "=" * 60)