// Public interface contract for Chip A.
// The mechanism-bearing implementation is maintained separately.
module virtual_core_chip_a #(
    parameter N = 8,
    parameter ACCUM_W = 16,
    parameter ADC_W = 6
) (
    input wire clk,
    input wire rst_n,
    input wire [N*ADC_W-1:0] i_col,
    input wire [ADC_W-1:0] i_ref,
    input wire signed [ACCUM_W-1:0] v_th,
    input wire [2:0] leak_sh,
    input wire corr_en,
    input wire cfg_en,
    input wire cfg_si,
    output wire cfg_so,
    output wire [N-1:0] spike
);
endmodule