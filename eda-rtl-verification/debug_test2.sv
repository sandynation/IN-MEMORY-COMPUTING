module tb;
  logic clk;
  logic a;
  logic tb_mismatch;
  initial begin
    $dumpfile("wave.vcd");
    $dumpvars(1, clk, tb_mismatch, a);
  end
endmodule
