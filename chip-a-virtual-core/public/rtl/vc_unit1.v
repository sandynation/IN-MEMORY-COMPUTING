// vc_unit1 — LIF accumulator (Chip A, proposal §2.3)
//
//   Vmem[t+1] = λ · Vmem[t] + Icol_corrected[t]
//   λ · Vmem  via arithmetic barrel-shift (power-of-two decay, no multiplier)
//   Icol_corrected = Icol − DAC          (digital model of the analog injection)
//
// A 1-cycle spike pulse is issued when Vmem >= Vth; Vmem resets to 0 with C.
module vc_unit1 #(
    parameter ACCUM_W = 16,
    parameter ADC_W   = 6
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire [ADC_W-1:0]          i_col,
    input  wire [ADC_W-1:0]          i_dac,
    input  wire signed [ACCUM_W-1:0] v_th,
    input  wire [2:0]                leak_sh,
    output wire                      fire,
    output reg                       spike_out,
    output reg  signed [ACCUM_W-1:0] v_mem
);
    wire signed [ACCUM_W-1:0] leaked = v_mem >>> leak_sh;
    wire signed [ACCUM_W-1:0] i_corr =
        $signed({1'b0, i_col}) - $signed({1'b0, i_dac});
    wire signed [ACCUM_W-1:0] v_next = leaked + i_corr;
    assign fire = (v_next >= v_th);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem     <= {ACCUM_W{1'b0}};
            spike_out <= 1'b0;
        end else if (fire) begin
            v_mem     <= {ACCUM_W{1'b0}};
            spike_out <= 1'b1;
        end else begin
            v_mem     <= v_next;
            spike_out <= 1'b0;
        end
    end
endmodule
