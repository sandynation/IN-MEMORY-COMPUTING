// tb_chip_a — golden-model checks for proposal §2.3
// Run with: iverilog -g2012 -o tb tb_chip_a.v vc_unit1.v vc_unit2.v virtual_core_chip_a.v && vvp tb
`timescale 1ns/1ps
module tb_chip_a;
    localparam N = 8;
    localparam ACCUM_W = 16;
    localparam ADC_W = 6;

    reg clk, rst_n, corr_en, cfg_en, cfg_si;
    reg [N*ADC_W-1:0] i_col;
    reg [ADC_W-1:0] i_ref;
    reg signed [ACCUM_W-1:0] v_th;
    reg [2:0] leak_sh;
    wire cfg_so;
    wire [N-1:0] spike;

    virtual_core_chip_a #(.N(N)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_col(i_col), .i_ref(i_ref),
        .v_th(v_th), .leak_sh(leak_sh),
        .corr_en(corr_en),
        .cfg_en(cfg_en), .cfg_si(cfg_si), .cfg_so(cfg_so),
        .spike(spike)
    );

    integer t, fails;
    initial clk = 0;
    always #5 clk = ~clk;

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    initial begin
        fails = 0;
        rst_n = 0; corr_en = 1; cfg_en = 0; cfg_si = 0;
        i_col = {N{6'd0}}; i_ref = 6'd0; v_th = 16'sd40; leak_sh = 3'd0;
        repeat (4) tick;
        rst_n = 1;
        tick;

        // Integrate 10 per cycle, no leak, no sneak: spike on cycle 4 (10,20,30,40).
        i_col = {N{6'd10}}; i_ref = 6'd0; leak_sh = 3'd0; v_th = 16'sd40;
        tick; if (spike !== 8'h00) begin $display("FAIL early spike"); fails = fails+1; end
        tick;
        tick;
        tick; if (spike !== 8'hFF) begin $display("FAIL expected all spike, got %b", spike); fails = fails+1; end
        tick; if (spike !== 8'h00) begin $display("FAIL no reset pulse"); fails = fails+1; end

        // Correction: Icol=20, Iref=5. Same-cycle DAC = C+Iref so t1: Icorr=15.
        rst_n = 0; tick; rst_n = 1; tick;
        i_col = {N{6'd20}}; i_ref = 6'd5; corr_en = 1; v_th = 16'sd1000; leak_sh = 3'd0;
        tick; // v=15, c=5, dac=5
        tick; // v=25, c=10
        tick; // v=30, c=15
        if (spike !== 8'h00) begin $display("FAIL corr path spiked"); fails = fails+1; end

        // corr_en=0 must not subtract (faster climb).
        rst_n = 0; tick; rst_n = 1; tick;
        corr_en = 0; i_col = {N{6'd20}}; i_ref = 6'd5; v_th = 16'sd40; leak_sh = 3'd0;
        tick; tick;
        if (spike !== 8'hFF) begin $display("FAIL corr_en=0 should spike by t=2 (20+20)"); fails = fails+1; end

        if (fails == 0) $display("PASS Chip A §2.3 checks");
        else $display("FAIL %0d checks", fails);
        $finish;
    end
endmodule
