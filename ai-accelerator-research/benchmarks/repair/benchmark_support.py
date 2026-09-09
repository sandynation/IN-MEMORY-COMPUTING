# Copyright (c) 2026 Sandipan Pal. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
import subprocess
import tempfile
import os

def get_testbench(m, n, k):
    return f"""
`timescale 1ns/1ps
module tb;
    parameter TILE_SIZE_M = {m};
    parameter TILE_SIZE_N = {n};
    parameter TILE_SIZE_K = {k};
    reg clk, rst_n;
    reg [15:0] a_data, b_data;
    reg a_valid, b_valid;
    wire a_ready, b_ready;
    wire [31:0] c_data;
    wire c_valid;
    reg c_ready;
    integer i, j, kk;
    reg [15:0] A[0:{m-1}][0:{k-1}];
    reg [15:0] B[0:{k-1}][0:{n-1}];
    reg [31:0] C_golden[0:{m-1}][0:{n-1}];

    gemm_tile #(
        .TILE_SIZE_M(TILE_SIZE_M),
        .TILE_SIZE_N(TILE_SIZE_N),
        .TILE_SIZE_K(TILE_SIZE_K),
        .DATA_WIDTH(16),
        .ACC_WIDTH(32)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .a_data(a_data), .a_valid(a_valid), .a_ready(a_ready),
        .b_data(b_data), .b_valid(b_valid), .b_ready(b_ready),
        .c_data(c_data), .c_valid(c_valid), .c_ready(c_ready),
        .sparse_skip(1'b0), .mask_valid(1'b0)
    );

    initial begin
        $dumpfile("wave.vcd");
        $dumpvars(0, tb);
        clk = 0; rst_n = 0; a_valid = 0; b_valid = 0; c_ready = 1;
        #10 rst_n = 1;
        // Fill random matrices
        for (i = 0; i < TILE_SIZE_M; i=i+1)
            for (j = 0; j < TILE_SIZE_K; j=j+1)
                A[i][j] = $random;
        for (i = 0; i < TILE_SIZE_K; i=i+1)
            for (j = 0; j < TILE_SIZE_N; j=j+1)
                B[i][j] = $random;
        // Compute golden C[0][0]
        C_golden[0][0] = 0;
        for (kk = 0; kk < TILE_SIZE_K; kk=kk+1)
            C_golden[0][0] = C_golden[0][0] + A[0][kk] * B[kk][0];
        // Stream A row 0 (only first row needed)
        for (j = 0; j < TILE_SIZE_K; j=j+1) begin
            @(posedge clk);
            a_data = A[0][j]; a_valid = 1;
            @(posedge clk);
            a_valid = 0;
        end
        // Stream B column 0 (only first column needed)
        for (i = 0; i < TILE_SIZE_K; i=i+1) begin
            @(posedge clk);
            b_data = B[i][0]; b_valid = 1;
            @(posedge clk);
            b_valid = 0;
        end
        // Wait for computation and output
        repeat(1000) @(posedge clk);
        if (c_data !== C_golden[0][0]) begin
            $display("FAILURE: c_data mismatch");
            $finish;
        end
        $display("SUCCESS");
        $finish;
    end
    always #5 clk = ~clk;
endmodule
"""

def compile_and_simulate(verilog_files, testbench_str):
    """Compile and simulate Verilog files with the given testbench."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
        f.write(testbench_str)
        tb_file = f.name
    all_files = verilog_files + [tb_file]
    try:
        subprocess.run(['iverilog', '-o', 'sim_out'] + all_files, check=True, capture_output=True, text=True)
        result = subprocess.run(['vvp', 'sim_out'], capture_output=True, text=True, timeout=10)
        passed = 'SUCCESS' in result.stdout
        return passed, result.stdout, result.stderr
    except Exception as e:
        return False, '', str(e)
    finally:
        os.unlink(tb_file)
        if os.path.exists('sim_out'):
            os.unlink('sim_out')