#!/usr/bin/env python3
"""Test all mission modules for import and basic functionality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("1. Testing formal_verifier...")
from backend.accel_tool.verilog.formal_verifier import FormalVerifier
fv = FormalVerifier()
print("   OK")

print("2. Testing equivalence_checker...")
from backend.accel_tool.verilog.equivalence_checker import EquivalenceChecker
ec = EquivalenceChecker()
print("   OK")

print("3. Testing differentiable_thermal...")
from backend.novelty_features.physics.differentiable_thermal import DifferentiableThermalSolver
dts = DifferentiableThermalSolver()
print("   OK")

print("4. Testing explainer...")
from backend.novelty_features.trajectory.explainer import TrajectoryExplainer
te = TrajectoryExplainer()
print("   OK")

print("5. Testing static_analyzer...")
from backend.accel_tool.security.static_analyzer import StaticSecurityAnalyzer
ssa = StaticSecurityAnalyzer()
print("   OK")

print("6. Testing secure_planner...")
from backend.novelty_features.trajectory.secure_planner import SecureTrajectoryPlanner
stp = SecureTrajectoryPlanner({}, "test", 0.2)
print("   OK")

print("\n✅ All modules imported without errors.")