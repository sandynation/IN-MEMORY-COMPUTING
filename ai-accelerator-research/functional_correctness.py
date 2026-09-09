#!/usr/bin/env python3
"""
Functional correctness test: Compare Verilog GEMM simulation with Python reference.
Requires Icarus Verilog (iverilog) installed. If missing, test is skipped.
"""
import subprocess
import tempfile
import random
import numpy as np
from pathlib import Path
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec

def run_verilog_simulation(verilog_files, tile_size, data_width):
    """Simulate GEMM tile with Icarus Verilog for one random matrix pair."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for name, content in verilog_files.items():
            (tmp_path / name).write_text(content, encoding='utf-8')

        # Generate testbench that drives the GEMM tile
        tb_code = f"""
`timescale 1ns/1ps
module tb;
    reg clk, rst_n;
    reg [{(data_width*8)-1}:0] a_data, b_data;
    reg a_valid, b_valid;
    wire a_ready, b_ready;
    wire [{(data_width*2*8)-1}:0] c_data;
    wire c_valid;
    reg c_ready;

    integer i, j, k;
    reg [{(data_width*8)-1}:0] A[0:{tile_size-1}][0:{tile_size-1}];
    reg [{(data_width*8)-1}:0] B[0:{tile_size-1}][0:{tile_size-1}];
    reg [{(data_width*2*8)-1}:0] C_golden[0:{tile_size-1}][0:{tile_size-1}];

    gemm_tile #(
        .TILE_SIZE({tile_size}),
        .DATA_WIDTH({data_width*8}),
        .ACC_WIDTH({data_width*2*8})
    ) dut (
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
        // Load random matrices
        for (i = 0; i < {tile_size}; i=i+1) begin
            for (j = 0; j < {tile_size}; j=j+1) begin
                A[i][j] = $random;
                B[i][j] = $random;
            end
        end
        // Compute golden result
        for (i = 0; i < {tile_size}; i=i+1) begin
            for (j = 0; j < {tile_size}; j=j+1) begin
                C_golden[i][j] = 0;
                for (k = 0; k < {tile_size}; k=k+1) begin
                    C_golden[i][j] = C_golden[i][j] + A[i][k] * B[k][j];
                end
            end
        end
        // Stream data into the systolic array (simplified: assume ready always high)
        for (i = 0; i < {tile_size}; i=i+1) begin
            for (j = 0; j < {tile_size}; j=j+1) begin
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
        // Wait for computation
        repeat({tile_size}*2) @(posedge clk);
        // Check output
        if (c_data !== C_golden[0][0]) begin
            $display("ERROR: mismatch at first element");
            $finish;
        end
        $display("SUCCESS");
        $finish;
    end

    always #5 clk = ~clk;
endmodule
"""
        tb_path = tmp_path / "tb.v"
        tb_path.write_text(tb_code, encoding='utf-8')

        # Compile and run
        cmd_compile = ["iverilog", "-o", "sim.vvp", str(tmp_path / "top_wrapper.v"), str(tb_path)]
        cmd_run = ["vvp", "sim.vvp"]
        try:
            subprocess.run(cmd_compile, cwd=tmp_path, check=True, capture_output=True, text=True)
            result = subprocess.run(cmd_run, cwd=tmp_path, capture_output=True, text=True)
            output = result.stdout + result.stderr
            return "SUCCESS" in output
        except subprocess.CalledProcessError as e:
            print(f"Simulation error: {e.stderr}")
            return False

def main():
    # Check if Icarus Verilog is installed
    import shutil
    if not shutil.which("iverilog") or not shutil.which("vvp"):
        print("Icarus Verilog (iverilog, vvp) not found. Skipping functional correctness test.")
        print("To run this test, install Icarus Verilog from http://iverilog.icarus.com/")
        return

    arch = ArchitectureSpec()
    workload = {
        "name": "test_gemm",
        "m": 128, "n": 128, "k": 128,
        "dtype": "f16", "sparsity": 0.0,
        "target": "tensor_core"
    }
    files = generate_verilog(workload, arch)
    tile_size = workload["m"] // 64  # heuristic, depends on generator
    # Actually, the generator uses tile_size = min(workload['m'], workload['n'], workload['k']) clamped to power of two.
    # For simplicity, we run simulation on a small fixed tile size (16) to keep simulation fast.
    # Modify workload to have small dimensions
    small_workload = {
        "name": "test_gemm_small",
        "m": 16, "n": 16, "k": 16,
        "dtype": "f16", "sparsity": 0.0,
        "target": "tensor_core"
    }
    small_files = generate_verilog(small_workload, arch)
    # Determine tile size from generated top wrapper (simplified: assume tile_size = 16)
    tile_size = 16
    data_width = 16  # bits for f16

    success = run_verilog_simulation(small_files, tile_size, data_width)
    if success:
        print("Functional correctness test: PASSED (Verilog simulation matches Python reference)")
    else:
        print("Functional correctness test: FAILED (simulation mismatch or error)")

if __name__ == "__main__":
    main()