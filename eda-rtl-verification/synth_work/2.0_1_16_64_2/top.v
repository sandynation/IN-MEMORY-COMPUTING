// Top wrapper for sparsex_design
// Tile size: 4, Data width: 16, Chiplets: 1

module top_wrapper (
    input  wire clk,
    input  wire rst_n,
    // AXI4-Lite control
    input  wire [31:0] s_axil_awaddr,
    input  wire        s_axil_awvalid,
    output wire        s_axil_awready,
    input  wire [31:0] s_axil_wdata,
    input  wire [3:0]  s_axil_wstrb,
    input  wire        s_axil_wvalid,
    output wire        s_axil_wready,
    output wire [1:0]  s_axil_bresp,
    output wire        s_axil_bvalid,
    input  wire        s_axil_bready,
    input  wire [31:0] s_axil_araddr,
    input  wire        s_axil_arvalid,
    output wire        s_axil_arready,
    output wire [31:0] s_axil_rdata,
    output wire [1:0]  s_axil_rresp,
    output wire        s_axil_rvalid,
    input  wire        s_axil_rready,
    // AXI4 memory (HBM)
    // ... (AXI4 full signals as in memory_controller.v)
    // Interrupt
    output wire        interrupt
);

    // Control/status registers
    reg [31:0] control;
    reg [31:0] status;
    reg start_gemm;
    wire gemm_done;

    // Instantiate GEMM tile
    wire [DATA_WIDTH-1:0] a_data, b_data;
    wire a_valid, b_valid, a_ready, b_ready;
    wire [ACC_WIDTH-1:0] c_data;
    wire c_valid, c_ready;

    gemm_tile #(
        .TILE_SIZE(4),
        .DATA_WIDTH(16),
        .ACC_WIDTH(32)
    ) gemm_core (
        .clk(clk),
        .rst_n(rst_n),
        .a_data(a_data),
        .a_valid(a_valid),
        .a_ready(a_ready),
        .b_data(b_data),
        .b_valid(b_valid),
        .b_ready(b_ready),
        .c_data(c_data),
        .c_valid(c_valid),
        .c_ready(c_ready),
        .sparse_skip(1'b0),
        .mask_valid(1'b0)
    );

    // Control FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            control <= 0;
            status <= 0;
            start_gemm <= 0;
        end else begin
            // AXI4-Lite write handling (stub)
            // ...
            start_gemm <= control[0];
            if (gemm_done) status[0] <= 1;
        end
    end

    assign interrupt = status[0];
    assign gemm_done = (c_valid && c_ready); // simplified
endmodule