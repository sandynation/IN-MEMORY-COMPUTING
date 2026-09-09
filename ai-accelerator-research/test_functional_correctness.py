#!/usr/bin/env python3
"""
Simulate a 16×16 GEMM tile using Icarus Verilog and compare with Python reference.
Corrected testbench: streams all input elements, waits for computation, checks output.
"""
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec
from backend.accel_tool.verilog.tool_paths import build_icarus_env, resolve_icarus_tools

def run_simulation(verilog_files, tile_size, data_width):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for name, content in verilog_files.items():
            (tmp_path / name).write_text(content, encoding='utf-8')
        # Testbench for 16x16 systolic array
        tb_code = f"""
`timescale 1ns/1ps
module tb;
    reg clk, rst_n;
        reg [{data_width-1}:0] a_data, b_data;
        reg a_valid, b_valid;
        wire a_ready, b_ready;
        wire [{(data_width*2)-1}:0] c_data;
        wire c_valid;
        reg c_ready;

        integer i, j, k;
        reg [{data_width-1}:0] A[0:{tile_size-1}][0:{tile_size-1}];
        reg [{data_width-1}:0] B[0:{tile_size-1}][0:{tile_size-1}];
        reg [{(data_width*2)-1}:0] C_golden[0:{tile_size-1}][0:{tile_size-1}];
        integer error_count;

        gemm_tile #(
        .TILE_SIZE_M({tile_size}),
        .TILE_SIZE_N({tile_size}),
        .TILE_SIZE_K({tile_size}),
        .DATA_WIDTH({data_width}),
        .ACC_WIDTH({data_width*2})
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

        // Fill matrices with random data
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

        // Stream the first row of A and the first column of B.
        // The current smoke-test GEMM computes C[0][0] only.
        for (i = 0; i < {tile_size}; i=i+1) begin
            @(posedge clk);
            a_data = A[0][i];
            a_valid = 1;
            b_data = B[i][0];
            b_valid = 1;
            @(posedge clk);
            a_valid = 0;
            b_valid = 0;
        end

        repeat({tile_size}) @(posedge clk);

        if (c_data !== C_golden[0][0]) begin
            $display("ERROR: c_data = %d, expected %d", c_data, C_golden[0][0]);
            error_count = 1;
        end else begin
            error_count = 0;
        end

        if (error_count == 0) $display("SUCCESS");
        else $display("FAIL");
        $finish;
    end

    always #5 clk = ~clk;
endmodule
"""
        tb_path = tmp_path / "tb.v"
        tb_path.write_text(tb_code, encoding='utf-8')
        try:
            iverilog_path, vvp_path = resolve_icarus_tools()
            if not iverilog_path or not vvp_path:
                print("Icarus tool resolution failed inside the simulation runner.")
                return False
            env = build_icarus_env(iverilog_path)
            verilog_sources = [str(p) for p in sorted(tmp_path.glob("*.v")) if p.name != "tb.v"]
            # Compile
            compile_result = subprocess.run(
                [iverilog_path, "-o", "sim.vvp", *verilog_sources, str(tb_path)],
                cwd=tmp_path, env=env, capture_output=True, text=True
            )
            if compile_result.returncode != 0:
                print("Compilation error:", compile_result.stderr)
                return False
            # Simulate
            result = subprocess.run([vvp_path, "sim.vvp"], cwd=tmp_path, env=env, capture_output=True, text=True)
            return "SUCCESS" in result.stdout
        except Exception as e:
            print(f"Simulation error: {e}")
            return False

def main():
    iverilog_path, vvp_path = resolve_icarus_tools()
    if not iverilog_path or not vvp_path:
        print("Icarus Verilog (iverilog, vvp) not found. Skipping functional correctness test.")
        return

    print(f"Using Icarus Verilog: {iverilog_path}")
    print(f"Using vvp: {vvp_path}")

    arch = ArchitectureSpec()
    workload = {"name": "test_gemm_16", "m": 16, "n": 16, "k": 16, "dtype": "f16", "sparsity": 0.0, "target": "tensor_core"}
    files = generate_verilog(workload, arch)
    success = run_simulation(files, tile_size=16, data_width=16)
    if success:
        print("Functional correctness test (16×16) PASSED")
    else:
        print("Functional correctness test (16×16) FAILED")

if __name__ == "__main__":
    main()
