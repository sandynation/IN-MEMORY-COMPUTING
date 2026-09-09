// virtual_core_chip_a — Chip A digital Virtual Core (proposal §2.3)
//
// N parallel neuron columns. Not a CPU: no fetch, no GPR, no programmable
// datapath. Each column is Unit 1 (16-bit LIF) + Unit 2 (16-bit correction).
// One shared 6-bit reference-column ADC (Iref) feeds every Unit 2; each
// column keeps its own C because neurons spike / reset on different cycles.
//
// Same-cycle timing (proposal §2.3):
//   (1) Iref loaded into C
//   (2) 6-bit DAC updated from C
//   (3) Icol − DAC integrated into Vmem
//   (4) threshold compare
//   (5) spike + shared reset of Vmem and C
//
// SKY130 HD  |  chipIgnite shuttle  |  10 MHz nominal
module virtual_core_chip_a #(
    parameter N       = 8,
    parameter ACCUM_W = 16,
    parameter ADC_W   = 6
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire [N*ADC_W-1:0]        i_col,     // per-column 6-bit ADC
    input  wire [ADC_W-1:0]          i_ref,     // shared reference-column ADC
    input  wire signed [ACCUM_W-1:0] v_th,
    input  wire [2:0]                leak_sh,   // λ = 2^{-leak_sh}
    input  wire                      corr_en,   // §3.2 correction on/off
    input  wire                      cfg_en,    // advance membrane dump index
    input  wire                      cfg_si,    // reserved (dump is output-only)
    output wire                      cfg_so,
    output wire [N-1:0]              spike
);
    wire [ADC_W-1:0]          dac   [0:N-1];
    wire                      fire  [0:N-1];
    wire signed [ACCUM_W-1:0] v_mem [0:N-1];
    wire signed [ACCUM_W-1:0] c_acc [0:N-1];

    genvar k;
    generate
        for (k = 0; k < N; k = k + 1) begin : col
            vc_unit2 #(.ACCUM_W(ACCUM_W), .ADC_W(ADC_W)) u2 (
                .clk(clk), .rst_n(rst_n),
                .i_ref(i_ref),
                .fire(fire[k]),
                .corr_en(corr_en),
                .dac(dac[k]),
                .c_acc(c_acc[k])
            );
            vc_unit1 #(.ACCUM_W(ACCUM_W), .ADC_W(ADC_W)) u1 (
                .clk(clk), .rst_n(rst_n),
                .i_col(i_col[k*ADC_W +: ADC_W]),
                .i_dac(dac[k]),
                .v_th(v_th),
                .leak_sh(leak_sh),
                .fire(fire[k]),
                .spike_out(spike[k]),
                .v_mem(v_mem[k])
            );
        end
    endgenerate

    // Muxed Vmem/C dump for §3.2 membrane-potential trajectories.
    // cfg_si is reserved (output-only dump) so a later scan-in pad stays compatible.
    localparam SCAN_W = N * ACCUM_W * 2;
    localparam IDX_W  = $clog2(SCAN_W);
    wire [SCAN_W-1:0] scan_pack;
    generate
        for (k = 0; k < N; k = k + 1) begin : pack
            assign scan_pack[(2*k)*ACCUM_W +: ACCUM_W]   = v_mem[k];
            assign scan_pack[(2*k+1)*ACCUM_W +: ACCUM_W] = c_acc[k];
        end
    endgenerate

    reg [IDX_W-1:0] scan_idx;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)      scan_idx <= {IDX_W{1'b0}};
        else if (cfg_en) scan_idx <= scan_idx + 1'b1;
    end
    assign cfg_so = scan_pack[scan_idx];
endmodule
