module reference_gemm #(parameter WIDTH = 16, SIZE = 8) (
    input clk, rst_n,
    input [WIDTH-1:0] a_data, b_data,
    input a_valid, b_valid,
    output reg [2*WIDTH-1:0] c_data,
    output reg c_valid
);
    // Simple non‑pipelined multiplier‑accumulator (for demonstration)
    reg [2*WIDTH-1:0] acc;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 0;
            c_valid <= 0;
        end else if (a_valid && b_valid) begin
            acc <= a_data * b_data;
            c_valid <= 1;
            c_data <= acc;
        end else begin
            c_valid <= 0;
        end
    end
endmodule