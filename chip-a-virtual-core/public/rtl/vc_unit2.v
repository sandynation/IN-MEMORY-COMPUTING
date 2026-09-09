// vc_unit2 — per-timestep sneak-path correction (Chip A, proposal §2.3)
//
//   C[t+1] = C[t] + Iref_ADC[t]          (Iref_ADC is 6-bit)
//   DAC    = sat6(C[t+1])                if corr_en, else 0
//
// C resets to 0 with the neuron spike (shared reset, same temporal window).
// corr_en implements §3.2 “correction on vs off” without changing C history.
module vc_unit2 #(
    parameter ACCUM_W = 16,
    parameter ADC_W   = 6
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire [ADC_W-1:0]          i_ref,
    input  wire                      fire,
    input  wire                      corr_en,
    output wire [ADC_W-1:0]          dac,
    output reg  signed [ACCUM_W-1:0] c_acc
);
    wire signed [ACCUM_W-1:0] c_next = c_acc + $signed({1'b0, i_ref});
    wire signed [ACCUM_W-1:0] dac_max = $signed({1'b0, {ADC_W{1'b1}}});

    wire [ADC_W-1:0] dac_raw =
        c_next[ACCUM_W-1] ? {ADC_W{1'b0}} :
        (c_next > dac_max) ? {ADC_W{1'b1}} :
        c_next[ADC_W-1:0];

    assign dac = corr_en ? dac_raw : {ADC_W{1'b0}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)    c_acc <= {ACCUM_W{1'b0}};
        else if (fire) c_acc <= {ACCUM_W{1'b0}};
        else           c_acc <= c_next;
    end
endmodule
