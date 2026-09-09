#!/usr/bin/env python3
"""Test S‑1: Speculative Trajectory Rollout with mocked LLM."""

import asyncio
import sys
from pathlib import Path

# Add project root to path (so that backend is importable)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ----------------------------------------------------------------------
# Mock accel_tool.llm module before any trajectory imports
# ----------------------------------------------------------------------
import types
mock_llm = types.ModuleType("accel_tool.llm")
async def mock_query_llm(prompt, temperature=0.2, max_tokens=10):
    # Return deterministic score based on prompt length (simple mock)
    return str(min(0.3 + len(prompt) / 1000, 0.9))
mock_llm.query_llm = mock_query_llm
sys.modules["accel_tool"] = types.ModuleType("accel_tool")
sys.modules["accel_tool.llm"] = mock_llm

# Now import trajectory modules
from backend.novelty_features.trajectory import TrajectoryPlanner
from backend.novelty_features.websocket_manager import ws_manager

# ----------------------------------------------------------------------
# Mock LLM evaluator (replace real one)
# ----------------------------------------------------------------------
class MockLLMEvaluator:
    async def evaluate_trajectory(self, desc: str) -> float:
        return min(0.3 + len(desc) / 1000, 0.9)
    async def evaluate_many(self, trajectories):
        return [await self.evaluate_trajectory(t["desc"]) for t in trajectories]

import backend.novelty_features.trajectory.trajectory_planner as tp
tp.ParallelLLMEvaluator = MockLLMEvaluator

# ----------------------------------------------------------------------
# Mock WebSocket broadcast
# ----------------------------------------------------------------------
async def mock_broadcast(self, session_id, event_type, data):
    print(f"[MOCK] session={session_id} event={event_type} keys={list(data.keys())}")
ws_manager.broadcast = mock_broadcast.__get__(ws_manager, type(ws_manager))

# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------
async def main():
    initial_state = {
        "tensor_dim": 128,
        "num_chiplets": 2,
        "voltage": 0.75,
        "cgra_rows": 8,
        "tensor_core_tflops": 400.0,
    }
    session_id = "test-session-001"
    planner = TrajectoryPlanner(initial_state, session_id)

    print("Starting planning for 10 iterations (should complete quickly)...")
    result = await planner.plan(iterations=10, steps_per_rollout=5)

    print("\n=== PLANNING COMPLETE ===")
    print(f"Session: {result['session_id']}")
    print(f"Iterations run: {result['iterations']}")
    print("\nTop 3 trajectories (state + score):")
    for i, traj in enumerate(result['top_trajectories'][:3], 1):
        print(f"{i}. score={traj['score']} state={traj['state']}")
    print("\n✅ Test passed – no errors.")

if __name__ == "__main__":
    asyncio.run(main())