module tb;
  logic clk;
  logic a;
  initial begin
    $dumpfile("wave.vcd");
    $dumpvars(1, clk, a);
  end
endmodule
