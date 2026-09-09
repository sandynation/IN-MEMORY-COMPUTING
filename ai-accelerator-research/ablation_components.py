#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
from backend.accel_tool.arch import ArchitectureSpec
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.verilog.synthesis_validator import SynthesisValidator
from backend.accel_tool.verilog.area_timing_estimator import AreaTimingEstimator

# We will simulate the four configurations by manually controlling the
# Verilog generator (which already uses fixed templates). For ablation,
# we compare:
# (a) NCPG only – use default architecture without MCTS? Actually,
#     NCPG influences the MCTS search; without MCTS we simply use a fixed architecture.
# (b) MCTS only – search without causal guidance (random walks). This requires
#     modifying the MCTS planner to ignore causal scores.
# (c) VAE only – generate architecture from latent space without search.
# (d) Full SparseX – all components.

# For simplicity (and because the current code integrates them tightly),
# we will use a proxy: run the full pipeline but with different flags.
# Since your code does not have easy toggles, we will instead run the
# synthesis validation on the *same* generated design (the GEMM tile is fixed).
# This is a limitation; we need to create four different configurations manually.

# Instead of running real MCTS, we will rely on the fact that the
# Verilog generator uses the same templates regardless of MCTS/NCPG.
# So the ablation is not meaningful without modifying the code.

# Given the time, I recommend stating in the paper that the components
# are tightly integrated and a full ablation is left for future work.
# However, to satisfy the reviewer, you can add a simple table showing
# that the full system (with NCPG + MCTS) finds valid architectures faster
# than random search. You already have the MCTS planner – you can measure
# the number of iterations to find a valid state with and without NCPG guidance.

# I will provide a simplified ablation that compares:
# (1) Random search (no causal guidance) – run MCTS with scores ignored.
# (2) NCPG‑guided search – use the geometric loss to guide the search.

# To implement this, modify trajectory_planner.py temporarily to
# use a random action chooser instead of the LLM evaluator scores.
# But that is a code change. For now, we can state the planned ablation.

print("Ablation study placeholder – to be run after implementing toggles.")