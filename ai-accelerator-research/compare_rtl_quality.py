#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from backend.accel_tool.verilog.synthesis_validator import SynthesisValidator
from backend.accel_tool.verilog.area_timing_estimator import AreaTimingEstimator
from backend.accel_tool.ir_gen import generate_verilog
from backend.accel_tool.arch import ArchitectureSpec

# ----------------------------------------------------------------------
# Reference hand‑written GEMM tile (simple multiplier‑accumulator)
# ----------------------------------------------------------------------
REF_GEMM_CODE = """\
module gemm_tile_ref #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH = 32
) (
    input clk,
    input rst_n,
    input [DATA_WIDTH-1:0] a_data,
    input a_valid,
    output a_ready,
    input [DATA_WIDTH-1:0] b_data,
    input b_valid,
    output b_ready,
    output [ACC_WIDTH-1:0] c_data,
    output reg c_valid,
    input c_ready
);
    reg [ACC_WIDTH-1:0] acc;
    assign a_ready = 1;
    assign b_ready = 1;
    assign c_data = acc;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 0;
            c_valid <= 0;
        end else if (a_valid && b_valid) begin
            acc <= a_data * b_data;
            c_valid <= 1;
        end else if (c_ready) begin
            c_valid <= 0;
        end
    end
endmodule
"""

# Create a minimal top wrapper that instantiates the reference GEMM tile
REF_TOP_CODE = """\
module top_wrapper (
    input clk,
    input rst_n,
    // AXI4‑Lite stubs (unused)
    input [31:0] s_axil_awaddr,
    input s_axil_awvalid,
    output s_axil_awready,
    input [31:0] s_axil_wdata,
    input [3:0] s_axil_wstrb,
    input s_axil_wvalid,
    output s_axil_wready,
    output [1:0] s_axil_bresp,
    output s_axil_bvalid,
    input s_axil_bready,
    input [31:0] s_axil_araddr,
    input s_axil_arvalid,
    output s_axil_arready,
    output [31:0] s_axil_rdata,
    output [1:0] s_axil_rresp,
    output s_axil_rvalid,
    input s_axil_rready,
    output interrupt
);
    // Instantiate the reference GEMM tile (inputs tied to constants for synthesis)
    gemm_tile_ref #(
        .DATA_WIDTH(16),
        .ACC_WIDTH(32)
    ) gemm (
        .clk(clk),
        .rst_n(rst_n),
        .a_data(16'h1234),
        .a_valid(1),
        .b_data(16'h5678),
        .b_valid(1),
        .c_ready(1)
    );
    assign interrupt = 0;
    assign s_axil_awready = 0;
    assign s_axil_wready = 0;
    assign s_axil_bresp = 0;
    assign s_axil_bvalid = 0;
    assign s_axil_arready = 0;
    assign s_axil_rdata = 0;
    assign s_axil_rresp = 0;
    assign s_axil_rvalid = 0;
endmodule
"""

def evaluate_design(files_dict: dict) -> dict:
    validator = SynthesisValidator()
    report = validator.validate_synthesis(files_dict)
    estimator = AreaTimingEstimator()
    area_report = estimator.estimate_post_synthesis(files_dict, target_mhz=500.0)
    return {
        "passed": report["passed"],
        "area_um2": area_report["area_um2"],
        "max_freq_mhz": area_report["max_freq_mhz"],
        "cell_count": area_report["cell_count"],
        "errors": report["errors"],
        "warnings": report["warnings"][:3]  # show first 3 warnings
    }

# ----------------------------------------------------------------------
# 1. SparseX‑generated design (full top_wrapper + all modules)
# ----------------------------------------------------------------------
print("Generating SparseX design...")
workload = {"name": "gemm_16", "m": 16, "n": 16, "k": 16, "dtype": "f16", "sparsity": 0.0, "target": "tensor_core"}
arch = ArchitectureSpec()
sparsex_files = generate_verilog(workload, arch)
print("SparseX files:", list(sparsex_files.keys()))
sparsex_metrics = evaluate_design(sparsex_files)
print("SparseX‑generated GEMM:")
print(f"  Synthesizable: {sparsex_metrics['passed']}, area={sparsex_metrics['area_um2']}, fmax={sparsex_metrics['max_freq_mhz']}")
print(f"  Errors: {sparsex_metrics['errors']}")
print(f"  Warnings (first 3): {sparsex_metrics['warnings']}")

# ----------------------------------------------------------------------
# 2. Hand‑written reference design (wrapped in top_wrapper)
# ----------------------------------------------------------------------
ref_files = {
    "gemm_tile_ref.v": REF_GEMM_CODE,
    "top_wrapper.v": REF_TOP_CODE
}
ref_metrics = evaluate_design(ref_files)
print("\nHand‑written reference GEMM:")
print(f"  Synthesizable: {ref_metrics['passed']}, area={ref_metrics['area_um2']}, fmax={ref_metrics['max_freq_mhz']}")
print(f"  Errors: {ref_metrics['errors']}")
print(f"  Warnings (first 3): {ref_metrics['warnings']}")

# ----------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------
with open("rtl_quality_comparison.json", "w") as f:
    json.dump({
        "sparsex": sparsex_metrics,
        "reference": ref_metrics
    }, f, indent=2)
print("\nResults saved to rtl_quality_comparison.json")