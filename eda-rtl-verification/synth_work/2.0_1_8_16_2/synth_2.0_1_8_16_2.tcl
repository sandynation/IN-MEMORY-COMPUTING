
read_verilog synth_work/2.0_1_8_16_2/top.v
synth_design -top gemm_tile_0 -part xc7a200tfbg484-1
create_clock -period 2.0 [get_ports clk]
report_timing_summary -file timing_2.0_1_8_16_2.rpt
report_utilization  -file util_2.0_1_8_16_2.rpt
