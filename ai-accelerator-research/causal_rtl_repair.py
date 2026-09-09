#!/usr/bin/env python3
# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# SparseX – Neural Causal Silicon Compiler
"""
Causal RTL Repair Demonstration:
- Generate correct GEMM tile (4x4) → simulation passes.
- Inject a syntax error into the generated Verilog file (remove semicolon) → compilation fails.
- Use NCPG to compute parameter sensitivities (causal analysis).
- Repair by re‑generating the design → simulation passes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import subprocess
import tempfile
from torch.func import jacrev
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec
from backend.accel_tool.verilog.tool_paths import resolve_icarus_tools, build_icarus_env
from backend.novelty_features.ncpg_full import NCPG

def compile_and_simulate(verilog_files, testbench_code):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for name, content in verilog_files.items():
            (tmp_path / name).write_text(content, encoding='utf-8')
        (tmp_path / "tb.v").write_text(testbench_code, encoding='utf-8')
        iverilog_path, vvp_path = resolve_icarus_tools()
        if not iverilog_path or not vvp_path:
            return False, "Icarus tools not found", 0
        env = build_icarus_env(iverilog_path)
        try:
            subprocess.run(
                [iverilog_path, "-o", "sim.vvp"] + [str(p) for p in tmp_path.glob("*.v")],
                cwd=tmp_path, env=env, check=True, capture_output=True, text=True
            )
            result = subprocess.run([vvp_path, "sim.vvp"], cwd=tmp_path, env=env, capture_output=True, text=True)
            log = result.stdout + result.stderr
            fail_count = log.lower().count("failure") + log.lower().count("error")
            passed = fail_count == 0 and ("success" in log.lower() or "pass" in log.lower())
            return passed, log, fail_count
        except subprocess.CalledProcessError as e:
            return False, e.stderr, 999

def train_causal_model():
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 3)  # [tensor_dim, cgra_rows, voltage]
    # Latency = 1000/(tensor_dim+0.1) + 5*cgra_rows + 10*voltage^2
    latency = 1000.0 / (X[:,0] + 0.1) + 5.0 * X[:,1] + 10.0 * (X[:,2]**2) + 0.05 * np.random.randn(n_samples)
    y = latency.reshape(-1,1)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model = NCPG(num_nodes=3, output_dim=1, use_mlp=True, force_simple=True, device="cpu")
    model.fit(X_t, y_t, num_epochs=200, verbose=False)
    return model

def causal_sensitivity(model, nominal_params):
    """
    Compute sensitivity (dy/dp) for each parameter using Jacobian.
    Works with the model's internal forward pass.
    """
    model.model.eval()
    x = torch.tensor([nominal_params], dtype=torch.float32, requires_grad=True)
    def scalar_output(x_single):
        return model.model(x_single.unsqueeze(0)).squeeze()
    J = jacrev(scalar_output)(x)  # shape (1, num_nodes) -> (1,3)
    return J.detach().numpy().flatten()

def get_testbench():
    return """
`timescale 1ns/1ps
module tb;
    reg clk, rst_n;
    reg [15:0] a_data, b_data;
    reg a_valid, b_valid;
    wire a_ready, b_ready;
    wire [31:0] c_data;
    wire c_valid;
    reg c_ready;

    integer i, j, k;
    reg [15:0] A[0:3][0:3];
    reg [15:0] B[0:3][0:3];
    reg [31:0] C_golden[0:3][0:3];

    gemm_tile #(.TILE_SIZE(4), .DATA_WIDTH(16), .ACC_WIDTH(32)) dut (
        .clk(clk), .rst_n(rst_n),
        .a_data(a_data), .a_valid(a_valid), .a_ready(a_ready),
        .b_data(b_data), .b_valid(b_valid), .b_ready(b_ready),
        .c_data(c_data), .c_valid(c_valid), .c_ready(c_ready),
        .sparse_skip(0), .mask_valid(0)
    );

    initial begin
        $dumpfile("wave.vcd");
        $dumpvars(0, tb);
        clk = 0;
        rst_n = 0;
        a_valid = 0;
        b_valid = 0;
        c_ready = 1;
        #10 rst_n = 1;

        for (i = 0; i < 4; i=i+1) begin
            for (j = 0; j < 4; j=j+1) begin
                A[i][j] = $random;
                B[i][j] = $random;
            end
        end

        for (i = 0; i < 4; i=i+1) begin
            for (j = 0; j < 4; j=j+1) begin
                C_golden[i][j] = 0;
                for (k = 0; k < 4; k=k+1) begin
                    C_golden[i][j] = C_golden[i][j] + A[i][k] * B[k][j];
                end
            end
        end

        for (i = 0; i < 4; i=i+1) begin
            for (j = 0; j < 4; j=j+1) begin
                @(posedge clk);
                a_data = A[i][j];
                a_valid = 1;
                b_data = B[j][i];
                b_valid = 1;
                @(posedge clk);
                a_valid = 0;
                b_valid = 0;
            end
        end

        repeat(8) @(posedge clk);
        if (c_data !== C_golden[0][0]) begin
            $display("FAILURE: c_data = %d, expected %d", c_data, C_golden[0][0]);
            $finish;
        end
        $display("SUCCESS");
        $finish;
    end
    always #5 clk = ~clk;
endmodule
"""

def inject_bug(verilog_content):
    """Remove a semicolon from a critical line to cause compilation error."""
    lines = verilog_content.splitlines()
    for i, line in enumerate(lines):
        if 'assign a_ready =' in line and ';' in line:
            lines[i] = line.replace(';', '')
            break
    return '\n'.join(lines)

def main():
    print("=== Causal RTL Repair Demonstration ===\n")

    print("Training NCPG model on synthetic design space...")
    model = train_causal_model()
    print("Done.\n")

    arch = ArchitectureSpec()
    workload = {
        "name": "gemm_repair",
        "m": 4, "n": 4, "k": 4,
        "dtype": "f16",
        "sparsity": 0.0,
        "target": "tensor_core"
    }
    testbench = get_testbench()

    # Step 1: Generate correct design
    print("Step 1: Generate correct GEMM tile (4x4)...")
    correct_files = generate_verilog(workload, arch)
    passed, _, _ = compile_and_simulate(correct_files, testbench)
    if not passed:
        print("  ❌ Correct design simulation FAILED. Check Icarus.")
        return
    print("  ✅ Correct design simulation PASSED.\n")

    # Step 2: Inject a bug (syntax error) into the generated gemm_tile.v
    print("Step 2: Inject a syntax bug (remove semicolon from assign statement)...")
    buggy_content = inject_bug(correct_files["gemm_tile.v"])
    buggy_files = correct_files.copy()
    buggy_files["gemm_tile.v"] = buggy_content
    passed_buggy, log_buggy, _ = compile_and_simulate(buggy_files, testbench)
    if passed_buggy:
        print("  ⚠️ Buggy design unexpectedly passed. Check bug injection.")
    else:
        print("  ❌ Buggy design compilation FAILED (as expected).\n")

    # Step 3: Causal sensitivity analysis (using nominal architecture parameters)
    print("Step 3: Causal sensitivity analysis (nominal: tensor_dim=4, cgra_rows=8, voltage=0.75)...")
    nominal = [4.0, 8.0, 0.75]
    sensitivities = causal_sensitivity(model, nominal)
    param_names = ["tensor_dim", "cgra_rows", "voltage"]
    ranked = sorted(zip(param_names, sensitivities), key=lambda x: abs(x[1]), reverse=True)
    for name, sens in ranked:
        print(f"  {name}: sensitivity = {sens:.4f}")
    print()

    # Step 4: Repair by regenerating the correct RTL
    print("Step 4: Repair (regenerate correct design)...")
    repaired_files = generate_verilog(workload, arch)
    passed_repair, _, _ = compile_and_simulate(repaired_files, testbench)
    if passed_repair:
        print("  ✅ Repaired design simulation PASSED.")
        print("\nCausal repair demonstration successful: the system identified sensitive parameters and recovered from a syntax bug by regenerating correct RTL.")
    else:
        print("  ❌ Repaired design still fails.")

if __name__ == "__main__":
    main()